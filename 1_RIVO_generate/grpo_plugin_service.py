#!/usr/bin/env python3
"""Inference-only GRPO sidecar with a shared-Host ASAG attention probe.

The existing vLLM Host remains responsible for ordinary research.  This
sidecar owns a second, frozen Host copy for corrected REROUTE decoding:

    next_logits = host_logits + alpha * correction_head(plugin_hidden)

The same frozen Host object serves ASAG attention checkpoints. ASAG retains
Stop/Verify priority, while the 0.6B controller may promote Continue to Reroute.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import threading
import types
from pathlib import Path
from typing import Any, Callable

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from asag_attention import LocalAttentionProbe


DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parent.parent / "0_GRPO_plug_end_to_end"
SOURCE_ROOT = Path(
    os.environ.get("PLUGIN_GRPO_SOURCE_ROOT", str(DEFAULT_SOURCE_ROOT))
).expanduser().resolve()
if not SOURCE_ROOT.is_dir():
    raise FileNotFoundError(f"PLUGIN_GRPO_SOURCE_ROOT not found: {SOURCE_ROOT}")
# Load only the inference modules, without importing the verl package
# initializer and its training-only dependencies (for example tensordict).
PLUGIN_PACKAGE = "trained_grpo_plugin"
plugin_package = types.ModuleType(PLUGIN_PACKAGE)
plugin_package.__path__ = [str(SOURCE_ROOT / "verl/experimental/plugin_grpo")]
sys.modules[PLUGIN_PACKAGE] = plugin_package

POLICY_PATH = SOURCE_ROOT / "verl/experimental/plugin_grpo/policy.py"
policy_spec = importlib.util.spec_from_file_location(
    f"{PLUGIN_PACKAGE}.policy", POLICY_PATH
)
if policy_spec is None or policy_spec.loader is None:
    raise ImportError(f"Could not load GRPO policy module: {POLICY_PATH}")
policy_module = importlib.util.module_from_spec(policy_spec)
sys.modules[policy_spec.name] = policy_module
policy_spec.loader.exec_module(policy_module)

HINT_POLICY_PATH = SOURCE_ROOT / "verl/experimental/plugin_grpo/hint_policy.py"
hint_policy_spec = importlib.util.spec_from_file_location(
    f"{PLUGIN_PACKAGE}.hint_policy", HINT_POLICY_PATH
)
if hint_policy_spec is None or hint_policy_spec.loader is None:
    raise ImportError(f"Could not load GRPO hint-policy module: {HINT_POLICY_PATH}")
hint_policy_module = importlib.util.module_from_spec(hint_policy_spec)
sys.modules[hint_policy_spec.name] = hint_policy_module
hint_policy_spec.loader.exec_module(hint_policy_module)

CONTROLLER_LABELS = policy_module.CONTROLLER_LABELS
FrozenHostPluginPolicy = policy_module.FrozenHostPluginPolicy
RetrievalPluginPolicy = policy_module.RetrievalPluginPolicy
HintPolicyRuntime = hint_policy_module.HintPolicyRuntime
build_semantic_vocab_mask = hint_policy_module.build_semantic_vocab_mask


def controller_system_prompt() -> str:
    labels = ", ".join(CONTROLLER_LABELS)
    return (
        "You are a three-class retrieval-control classification model. Given an original "
        "question and the retrieval history available at the current turn, output exactly "
        f"one label from: {labels}. Output the label only. Do not explain, reason, or add punctuation."
    )


def controller_user_prompt(question: str, history: str) -> str:
    return (
        "<question>\n"
        f"{question.strip()}\n"
        "</question>\n\n"
        "<retrieval_history>\n"
        f"{history.strip()}\n"
        "</retrieval_history>\n\n"
        f"Allowed labels: {', '.join(CONTROLLER_LABELS)}\n"
        "Output exactly one label."
    )


def reroute_system_prompt() -> str:
    return (
        "Generate exactly three short search-direction fields in this order: "
        "ENTITY ; RELATION ; DISAMBIGUATOR. The response is already prefixed "
        "with 'Search direction:'. Output only the three field values after that "
        "prefix, separated by exactly two semicolons. Do not output labels, a "
        "sentence, preface, quotes, JSON, XML, tool-call wrapper, explanation, or "
        "the full question."
    )


def reroute_user_prompt(question: str, history: str) -> str:
    return (
        "<question>\n"
        f"{question.strip()}\n"
        "</question>\n\n"
        "<research_history>\n"
        f"{history.strip()}\n"
        "</research_history>\n\n"
        "Choose an unresolved entity, its missing relation, and a useful "
        "disambiguator. Use a substantially different direction from unsuccessful "
        "searches. Return only these semicolon-separated values: "
        "ENTITY ; RELATION ; DISAMBIGUATOR."
    )


ROUTE_HINT_PREFILL = "Search direction: "


class ControlRequest(BaseModel):
    question: str
    history: str
    round_num: int
    last_reroute_round: int | None = None


class RerouteRequest(BaseModel):
    question: str
    history: str
    # Optional only for compatibility with the old route-hint client. The
    # step-70 runtime constructs its own training-identical Host prefixes.
    host_prefix_token_ids: list[int] | None = None


class AnalyzeRequest(BaseModel):
    session_id: str | None = None
    history_token_ids: list[int]
    probe_token_ids: list[int]
    previous_span: dict | None = None
    current_span: dict


def _chat_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        return list(tokenizer.apply_chat_template(messages, **kwargs))
    except TypeError:
        kwargs.pop("enable_thinking")
        return list(tokenizer.apply_chat_template(messages, **kwargs))


def _trimmed_history_ids(
    tokenizer: Any,
    question: str,
    history: str,
    *,
    system_builder: Callable[[], str],
    user_builder: Callable[[str, str], str],
    max_length: int,
) -> list[int]:
    """Match the history truncation used by the GRPO rollout service."""

    def encode(selected_history: str) -> list[int]:
        return _chat_ids(
            tokenizer,
            [
                {"role": "system", "content": system_builder()},
                {
                    "role": "user",
                    "content": user_builder(question, selected_history),
                },
            ],
        )

    ids = encode(history)
    if len(ids) <= max_length:
        return ids
    marker = "[Earlier history omitted]\n"
    low, high = 0, len(history)
    best: list[int] | None = None
    while low <= high:
        middle = (low + high) // 2
        suffix = history[-middle:] if middle else ""
        candidate = encode(marker + suffix)
        if len(candidate) <= max_length:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        raise ValueError("Question and fixed prompt exceed plugin context length")
    return best


def _clean_route_hint(text: str, question: str) -> str:
    """Validate and normalize generated three-slot route-hint content."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?[^>]+>", " ", text)
    hint = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text).strip()
    if not hint:
        raise ValueError("Route hint must be a single non-empty line")
    hint = re.sub(r"^\s*search\s+direction\s*:\s*", "", hint, flags=re.I)
    hint = re.sub(r"[\r\n]+", " ; ", hint)
    hint = re.sub(r"[\"'{}[\]()]", " ", hint)
    hint = re.sub(
        r"\b(entity|relation|disambiguator)\b\s*:\s*",
        "",
        hint,
        flags=re.I,
    )
    hint = " ".join(hint.split())
    if hint.lower().startswith(("the user", "we need", "the question")):
        raise ValueError("Route hint contains a forbidden preface or label")
    parts = [part.strip() for part in hint.split(";")]
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(
            "Route hint must contain exactly three non-empty semicolon-separated fields"
        )
    return " ; ".join(" ".join(part.split()) for part in parts)


_NON_SEMANTIC_HINT_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at",
        "for", "from", "by", "with", "as", "is", "are", "was", "were", "be",
        "been", "being", "do", "does", "did", "have", "has", "had", "this",
        "that", "these", "those", "it", "its", "he", "she", "they", "them",
        "we", "our", "you", "your", "user", "asks", "ask", "question", "need",
        "find", "search", "direction", "query", "relevant", "information",
        "about", "current", "research", "should", "could", "would", "may",
        "might", "please", "output", "entity", "relation", "disambiguator",
    }
)


def _semantic_vocab_mask(tokenizer: Any, vocab_size: int) -> torch.Tensor:
    """Select content-bearing word-like tokens eligible for hint correction."""
    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    allowed: list[bool] = []
    for token_id in range(vocab_size):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        surface = token_text.strip()
        word_like = bool(surface) and all(
            character.isalnum() or character in "-_/" for character in surface
        )
        allowed.append(
            token_id not in special_ids
            and word_like
            and any(character.isalnum() for character in surface)
            and surface.casefold() not in _NON_SEMANTIC_HINT_WORDS
        )
    return torch.tensor(allowed, dtype=torch.bool)


class GRPOPluginInferenceEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        adapter_path = checkpoint / "lora_adapter"
        heads_path = checkpoint / "plugin_heads.pt"
        metadata_path = checkpoint / "metadata.json"
        for path in (adapter_path, heads_path, metadata_path):
            if not path.exists():
                raise FileNotFoundError(path)

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if tuple(metadata.get("labels", ())) != CONTROLLER_LABELS:
            raise ValueError(
                f"Checkpoint labels {metadata.get('labels')} do not match "
                f"{CONTROLLER_LABELS}"
            )
        host_model = args.host_model or metadata.get("host_model")
        if not host_model:
            raise ValueError("Host model is missing from arguments and checkpoint metadata")

        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
            if self.device.type == "cuda"
            else torch.float32
        )
        heads = torch.load(heads_path, map_location="cpu", weights_only=True)
        correction_rank = int(heads["correction_rank"])

        self.host_tokenizer = AutoTokenizer.from_pretrained(
            host_model,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        self.host = AutoModelForCausalLM.from_pretrained(
            host_model,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        self.host.to(self.device).eval()
        host_vocab_size = int(self.host.config.vocab_size)
        if int(heads["host_vocab_size"]) != host_vocab_size:
            raise ValueError(
                f"Checkpoint Host vocab={heads['host_vocab_size']} but loaded Host "
                f"vocab={host_vocab_size}"
            )

        plugin, self.plugin_tokenizer = RetrievalPluginPolicy.from_adapter(
            adapter_path,
            host_vocab_size=host_vocab_size,
            correction_rank=correction_rank,
            torch_dtype=dtype,
            train_full_backbone=False,
            local_files_only=args.local_files_only,
        )
        plugin.controller_head.load_state_dict(heads["controller_head"])
        plugin.correction_head.load_state_dict(heads["correction_head"])
        plugin.to(self.device).eval()

        alpha = float(heads["alpha"] if args.alpha is None else args.alpha)
        # FrozenHostPluginPolicy rejects negative alpha because its training
        # interface only supports forward corrections. This inference service
        # also runs signed-beta ablations, where a negative value is a
        # deliberate reverse-direction control. Initialize the shared policy
        # with a valid value, then install the requested signed coefficient;
        # both inference paths compute host_logits + alpha * delta.
        self.policy = FrozenHostPluginPolicy(
            self.host,
            plugin,
            alpha=max(alpha, 0.0),
            valid_vocab_size=len(self.host_tokenizer),
        )
        self.policy.alpha = alpha
        self.policy.to(self.device).eval()

        host_vocab = self.host_tokenizer.get_vocab()
        plugin_vocab = self.plugin_tokenizer.get_vocab()
        if host_vocab != plugin_vocab:
            mismatches = sum(
                plugin_vocab.get(token) != token_id
                for token, token_id in host_vocab.items()
            )
            raise ValueError(
                f"Host/plugin token-id maps differ ({mismatches} mismatches)"
            )

        self.checkpoint = checkpoint
        self.checkpoint_step = int(metadata.get("step", -1))
        self.policy_mode = str(metadata.get("policy_mode", ""))
        if self.policy_mode != "three_field_hint":
            raise ValueError(
                f"Expected a three_field_hint checkpoint, got {self.policy_mode!r}"
            )
        self.alpha = alpha
        self.controller_temperature = args.controller_temperature
        self.reroute_temperature = args.reroute_temperature
        self.reroute_top_p = args.reroute_top_p
        self.reroute_max_new_tokens = args.reroute_max_new_tokens
        self.delta_top_k = args.hint_correction_topk
        self.semantic_vocab_mask = build_semantic_vocab_mask(
            self.host_tokenizer, host_vocab_size
        ).to(self.device)
        self.hint_runtime = HintPolicyRuntime(
            self.policy,
            self.host_tokenizer,
            self.plugin_tokenizer,
            semantic_vocab_mask=self.semantic_vocab_mask,
            max_context_length=args.max_plugin_length,
        )
        self.hint_slot_token_budgets = args.hint_slot_token_budgets
        self.hint_temperature = args.hint_temperature
        self.hint_top_p = args.hint_top_p
        self.hint_query_max_words = args.hint_query_max_words
        self.reroute_seed = args.reroute_seed
        self.reroute_format_attempts = args.reroute_format_attempts
        if self.reroute_format_attempts < 1:
            raise ValueError("reroute_format_attempts must be positive")
        self.max_plugin_length = args.max_plugin_length
        self.min_stop_rounds = args.min_stop_rounds
        self.min_reroute_rounds = args.min_reroute_rounds
        self.reroute_cooldown_rounds = args.reroute_cooldown_rounds
        self.lock = threading.RLock()
        expected_asag_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.asag_attention_dtype]
        if self.host.dtype != expected_asag_dtype:
            raise RuntimeError(
                "Shared Host dtype does not match ASAG_ATTENTION_DTYPE: "
                f"host={self.host.dtype}, requested={expected_asag_dtype}"
            )
        self.asag_probe = LocalAttentionProbe.from_existing_model(
            self.host,
            device=str(self.device),
            lock=self.lock,
        )
        smoke_metrics = self.asag_probe.smoke_test()
        print(
            "Shared-Host ASAG smoke test passed: "
            f"layers={smoke_metrics['layers_used']} "
            f"window={smoke_metrics['decoding_window_tokens']} "
            f"sessions={self.asag_probe.max_cached_sessions} "
            f"max_kv_per_session={self.asag_probe.estimated_kv_gib_per_session():.2f}GiB"
        )

    @torch.inference_mode()
    def control(self, request: ControlRequest) -> dict[str, Any]:
        input_ids = _trimmed_history_ids(
            self.plugin_tokenizer,
            request.question,
            request.history,
            system_builder=controller_system_prompt,
            user_builder=controller_user_prompt,
            max_length=self.max_plugin_length,
        )
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        logits = self.policy.plugin.controller_logits(ids).float()[0]
        logits /= self.controller_temperature
        cooldown_ok = (
            request.last_reroute_round is None
            or request.round_num - request.last_reroute_round
            > self.reroute_cooldown_rounds
        )
        allowed = [
            True,
            # Keep legacy reroute-only behavior. The trained STOP logit remains
            # in the checkpoint but is never an executable evaluation action.
            False,
            request.round_num >= self.min_reroute_rounds and cooldown_ok,
        ]
        allowed_tensor = torch.tensor(allowed, dtype=torch.bool, device=self.device)
        masked_logits = logits.masked_fill(~allowed_tensor, -torch.inf)
        probabilities = F.softmax(masked_logits, dim=-1)
        action = int(masked_logits.argmax().item())
        return {
            "action": CONTROLLER_LABELS[action],
            "action_id": action,
            "probabilities": dict(
                zip(CONTROLLER_LABELS, probabilities.cpu().tolist(), strict=True)
            ),
            "allowed": dict(zip(CONTROLLER_LABELS, allowed, strict=True)),
        }

    @torch.inference_mode()
    def _generate_reroute_with_diagnostics(
        self,
        host_prefix_ids: torch.Tensor,
        plugin_prefix_ids: torch.Tensor,
        *,
        eos_token_ids: list[int],
        sampling_seed: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample corrected logits and retain token-level evidence of their effect."""
        if host_prefix_ids.ndim != 2 or plugin_prefix_ids.ndim != 2:
            raise ValueError("Host and plugin prefixes must be rank-2 tensors")
        if host_prefix_ids.shape[0] != 1 or plugin_prefix_ids.shape[0] != 1:
            raise ValueError("The reroute endpoint accepts exactly one request")
        if not eos_token_ids:
            raise ValueError("At least one EOS token id is required")

        host_past = None
        plugin_past = None
        host_step_ids = host_prefix_ids
        plugin_step_ids = plugin_prefix_ids
        eos = torch.tensor(eos_token_ids, device=self.device)
        sampled_tokens: list[torch.Tensor] = []
        per_token: list[dict[str, Any]] = []
        valid_vocab_size = int(self.policy.valid_vocab_size)
        generator = None
        effective_seed = self.reroute_seed if sampling_seed is None else sampling_seed
        if effective_seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(effective_seed)

        for step in range(self.reroute_max_new_tokens):
            host_output = self.host(
                input_ids=host_step_ids,
                past_key_values=host_past,
                use_cache=True,
                return_dict=True,
            )
            plugin_output = self.policy.plugin.backbone(
                input_ids=plugin_step_ids,
                past_key_values=plugin_past,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
            host_past = host_output.past_key_values
            plugin_past = plugin_output.past_key_values

            host_logits = host_output.logits[:, -1].detach().float()
            plugin_hidden = policy_module._last_hidden(plugin_output)[:, -1]
            raw_delta = self.policy.plugin.correction_head(plugin_hidden).float()
            semantic_mask = self.semantic_vocab_mask[:valid_vocab_size]
            semantic_token_count = int(semantic_mask.sum().item())
            if semantic_token_count < 1:
                raise ValueError("No semantic hint tokens are available for correction")
            sparse_top_k = min(self.delta_top_k, semantic_token_count)
            sparse_scores = raw_delta[:, :valid_vocab_size].masked_fill(
                ~semantic_mask.unsqueeze(0), -torch.inf
            )
            sparse_indices = torch.topk(
                sparse_scores, sparse_top_k, dim=-1
            ).indices
            sparse_delta = torch.zeros_like(raw_delta)
            sparse_delta[:, :valid_vocab_size].scatter_(
                1,
                sparse_indices,
                raw_delta[:, :valid_vocab_size].gather(1, sparse_indices),
            )
            applied_delta = self.alpha * sparse_delta
            corrected_logits = host_logits + applied_delta
            if valid_vocab_size < corrected_logits.shape[-1]:
                corrected_logits[:, valid_vocab_size:] = -torch.inf

            host_valid = host_logits[:, :valid_vocab_size]
            delta_valid = raw_delta[:, :valid_vocab_size]
            applied_valid = applied_delta[:, :valid_vocab_size]
            corrected_valid = corrected_logits[:, :valid_vocab_size]
            host_l2 = torch.linalg.vector_norm(host_valid, dim=-1)
            delta_l2 = torch.linalg.vector_norm(delta_valid, dim=-1)
            sparse_delta_l2 = torch.linalg.vector_norm(
                sparse_delta[:, :valid_vocab_size], dim=-1
            )
            applied_l2 = torch.linalg.vector_norm(applied_valid, dim=-1)
            ratio = applied_l2 / host_l2.clamp_min(1e-12)

            host_log_probs = F.log_softmax(
                host_valid / self.reroute_temperature, dim=-1
            )
            corrected_log_probs = F.log_softmax(
                corrected_valid / self.reroute_temperature, dim=-1
            )
            host_probs = host_log_probs.exp()
            kl_host_to_corrected = (
                host_probs * (host_log_probs - corrected_log_probs)
            ).sum(dim=-1).clamp_min(0.0)

            top_k = min(10, valid_vocab_size)
            host_top_ids = torch.topk(host_valid, top_k, dim=-1).indices[0]
            corrected_top_ids = torch.topk(
                corrected_valid, top_k, dim=-1
            ).indices[0]
            host_top_set = set(host_top_ids.cpu().tolist())
            corrected_top_set = set(corrected_top_ids.cpu().tolist())
            overlap_count = len(host_top_set & corrected_top_set)
            host_top1_id = int(host_top_ids[0].item())
            corrected_top1_id = int(corrected_top_ids[0].item())

            sampling_logits = corrected_logits / self.reroute_temperature
            sampling_logits = policy_module._top_p_filter(
                sampling_logits, self.reroute_top_p
            )
            sampling_log_probs = F.log_softmax(sampling_logits, dim=-1)
            next_token = torch.multinomial(
                sampling_log_probs.exp(), 1, generator=generator
            ).squeeze(-1)
            sampled_token_id = int(next_token[0].item())

            per_token.append(
                {
                    "step": step,
                    "sampled_token_id": sampled_token_id,
                    "sampled_token": self.host_tokenizer.decode(
                        [sampled_token_id], skip_special_tokens=False
                    ),
                    "host_logits_l2": float(host_l2[0].item()),
                    "delta_logits_l2": float(delta_l2[0].item()),
                    "sparse_delta_logits_l2": float(sparse_delta_l2[0].item()),
                    "delta_top_k": sparse_top_k,
                    "semantic_token_count": semantic_token_count,
                    "applied_delta_logits_l2": float(applied_l2[0].item()),
                    "applied_delta_to_host_ratio": float(ratio[0].item()),
                    "kl_host_to_corrected": float(
                        kl_host_to_corrected[0].item()
                    ),
                    "host_top1_id": host_top1_id,
                    "host_top1_token": self.host_tokenizer.decode(
                        [host_top1_id], skip_special_tokens=False
                    ),
                    "corrected_top1_id": corrected_top1_id,
                    "corrected_top1_token": self.host_tokenizer.decode(
                        [corrected_top1_id], skip_special_tokens=False
                    ),
                    "top1_changed": host_top1_id != corrected_top1_id,
                    "top10_overlap_count": overlap_count,
                    "top10_overlap_ratio": overlap_count / top_k,
                    "sampled_host_probability": float(
                        host_log_probs[0, sampled_token_id].exp().item()
                    ),
                    "sampled_corrected_probability": float(
                        corrected_log_probs[0, sampled_token_id].exp().item()
                    ),
                    "sampled_top_p_probability": float(
                        sampling_log_probs[0, sampled_token_id].exp().item()
                    ),
                }
            )
            sampled_tokens.append(next_token)
            host_step_ids = next_token.unsqueeze(-1)
            plugin_step_ids = next_token.unsqueeze(-1)
            if bool((next_token.unsqueeze(-1) == eos.unsqueeze(0)).any()):
                break

        ratios = [item["applied_delta_to_host_ratio"] for item in per_token]
        divergences = [item["kl_host_to_corrected"] for item in per_token]
        overlaps = [item["top10_overlap_ratio"] for item in per_token]
        top1_changes = sum(item["top1_changed"] for item in per_token)
        diagnostics = {
            "schema_version": 1,
            "alpha": self.alpha,
            "temperature": self.reroute_temperature,
            "top_p": self.reroute_top_p,
            "seed": effective_seed,
            "intervention_scope": "semantic_search_direction_hint_only",
            "delta_top_k": self.delta_top_k,
            "semantic_token_count": int(
                self.semantic_vocab_mask[:valid_vocab_size].sum().item()
            ),
            "num_tokens": len(per_token),
            "summary": {
                "top1_changed_tokens": top1_changes,
                "top1_change_rate": top1_changes / len(per_token),
                "mean_applied_delta_to_host_ratio": sum(ratios) / len(ratios),
                "max_applied_delta_to_host_ratio": max(ratios),
                "mean_kl_host_to_corrected": sum(divergences)
                / len(divergences),
                "max_kl_host_to_corrected": max(divergences),
                "mean_top10_overlap_ratio": sum(overlaps) / len(overlaps),
            },
            "per_token": per_token,
        }
        return torch.stack(sampled_tokens, dim=1), diagnostics

    @torch.inference_mode()
    def reroute(self, request: RerouteRequest) -> dict[str, Any]:
        """Generate the exact three-field corrected-logits hint used in training."""
        _ = request.host_prefix_token_ids
        format_errors: list[str] = []
        for format_attempt in range(self.reroute_format_attempts):
            attempt_seed = (
                self.reroute_seed + format_attempt
                if self.reroute_seed is not None
                else None
            )
            try:
                devices = (
                    [self.device.index if self.device.index is not None else 0]
                    if self.device.type == "cuda"
                    else []
                )
                with torch.random.fork_rng(devices=devices):
                    if attempt_seed is not None:
                        torch.manual_seed(attempt_seed)
                        if self.device.type == "cuda":
                            torch.cuda.manual_seed(attempt_seed)
                    sample = self.hint_runtime.generate_hint(
                        request.question,
                        request.history,
                        slot_token_budgets=self.hint_slot_token_budgets,
                        temperature=self.hint_temperature,
                        top_p=self.hint_top_p,
                        correction_topk=self.delta_top_k,
                    )
                query = " ".join(
                    re.sub(r"[;\r\n]+", " ", " ".join(sample.fields)).split()
                )
                query = " ".join(query.split()[: self.hint_query_max_words])
                if not query:
                    raise ValueError("Hint fields produced an empty query")
            except ValueError as exc:
                format_errors.append(
                    f"attempt={format_attempt + 1} seed={attempt_seed}: {exc}"
                )
                continue

            token_ids = [
                token_id
                for segment in sample.segments
                for token_id in segment.target_ids
            ]
            logits_diagnostics = {
                "schema_version": 2,
                "alpha": self.alpha,
                "temperature": self.hint_temperature,
                "top_p": self.hint_top_p,
                "seed": attempt_seed,
                "intervention_scope": "three_field_hint",
                "correction_top_k": self.delta_top_k,
                "slot_token_budgets": list(self.hint_slot_token_budgets),
                "num_tokens": sample.semantic_tokens,
                "format_attempt": format_attempt + 1,
                "format_errors": format_errors,
                "corrected_logits_applied": self.alpha != 0.0,
            }
            return {
                "route_hint": sample.text,
                "hint": sample.text,
                "fields": list(sample.fields),
                "query": query,
                "token_ids": token_ids,
                "alpha": self.alpha,
                "reroute_seed": attempt_seed,
                "logits_diagnostics": logits_diagnostics,
            }
        raise ValueError(
            "Three-field hint generation failed after "
            f"{self.reroute_format_attempts} attempts: " + " | ".join(format_errors)
        )


def build_app(engine: GRPOPluginInferenceEngine) -> FastAPI:
    app = FastAPI(title="Shared Host + ASAG attention + trained GRPO plugin")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "checkpoint": str(engine.checkpoint),
            "checkpoint_step": engine.checkpoint_step,
            "policy_mode": getattr(engine, "policy_mode", None),
            "alpha": engine.alpha,
            "reroute_seed": getattr(engine, "reroute_seed", None),
            "intervention_scope": "three_field_hint",
            "delta_top_k": getattr(engine, "delta_top_k", None),
            "hint_slot_token_budgets": list(
                getattr(engine, "hint_slot_token_budgets", ())
            ),
            "hint_query_max_words": getattr(engine, "hint_query_max_words", None),
            "semantic_token_count": (
                int(engine.semantic_vocab_mask.sum().item())
                if getattr(engine, "semantic_vocab_mask", None) is not None
                else None
            ),
            "shared_host_asag": True,
            "asag_max_cached_sessions": engine.asag_probe.max_cached_sessions,
            "asag_max_kv_gib_per_session": (
                engine.asag_probe.estimated_kv_gib_per_session()
            ),
        }

    @app.post("/analyze")
    def analyze(request: AnalyzeRequest) -> dict[str, Any]:
        try:
            # The probe uses engine.lock internally. Reroute decoding uses the
            # same RLock, so temporary eager-attention wrappers and GRPO Host
            # forwards can never overlap on the shared model object.
            return engine.asag_probe.analyze(
                request.history_token_ids,
                request.probe_token_ids,
                request.previous_span,
                request.current_span,
                request.session_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/sessions/{session_id:path}")
    def release_session(session_id: str) -> dict[str, bool]:
        return {"released": engine.asag_probe.release_session(session_id)}

    @app.post("/control")
    def control(request: ControlRequest) -> dict[str, Any]:
        try:
            with engine.lock:
                return engine.control(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/reroute")
    def reroute(request: RerouteRequest) -> dict[str, Any]:
        try:
            with engine.lock:
                return engine.reroute(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host-model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8032)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--controller-temperature", type=float, default=1.0)
    parser.add_argument("--reroute-temperature", type=float, default=1.0)
    parser.add_argument("--reroute-top-p", type=float, default=0.95)
    parser.add_argument("--reroute-max-new-tokens", type=int, default=16)
    parser.add_argument("--delta-top-k", type=int, default=128)
    parser.add_argument("--reroute-seed", type=int)
    parser.add_argument("--reroute-format-attempts", type=int, default=3)
    parser.add_argument("--hint-slot-token-budgets", default="16,12,12")
    parser.add_argument("--hint-temperature", type=float, default=1.0)
    parser.add_argument("--hint-top-p", type=float, default=0.95)
    parser.add_argument("--hint-correction-topk", type=int, default=128)
    parser.add_argument("--hint-query-max-words", type=int, default=24)
    parser.add_argument("--max-plugin-length", type=int, default=4096)
    parser.add_argument(
        "--asag-attention-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Required dtype of the Host shared with the ASAG attention probe.",
    )
    parser.add_argument("--min-stop-rounds", type=int, default=15)
    parser.add_argument("--min-reroute-rounds", type=int, default=8)
    parser.add_argument("--reroute-cooldown-rounds", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    try:
        args.hint_slot_token_budgets = tuple(
            int(item.strip()) for item in args.hint_slot_token_budgets.split(",")
        )
    except ValueError:
        parser.error("--hint-slot-token-budgets must be comma-separated integers")
    if (
        len(args.hint_slot_token_budgets) != 3
        or min(args.hint_slot_token_budgets) <= 0
        or sum(args.hint_slot_token_budgets) > 64
    ):
        parser.error(
            "--hint-slot-token-budgets must be three positive integers totaling <= 64"
        )
    if args.controller_temperature <= 0 or args.reroute_temperature <= 0:
        parser.error("temperatures must be positive")
    if not 0 < args.reroute_top_p <= 1:
        parser.error("--reroute-top-p must be in (0, 1]")
    if (
        args.reroute_max_new_tokens < 1
        or args.delta_top_k < 1
        or args.hint_correction_topk < 1
        or args.hint_query_max_words < 1
        or args.max_plugin_length < 1
    ):
        parser.error("token limits and correction top-k must be positive")
    if args.hint_temperature <= 0 or not 0 < args.hint_top_p <= 1:
        parser.error(
            "hint temperature must be positive and hint top-p must be in (0, 1]"
        )
    return args


def main() -> None:
    import uvicorn

    args = parse_args()
    engine = GRPOPluginInferenceEngine(args)
    uvicorn.run(build_app(engine), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
