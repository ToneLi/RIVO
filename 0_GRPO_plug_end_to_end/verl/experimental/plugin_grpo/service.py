"""HTTP service owning the frozen Host and trainable 0.6B plugin policy."""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .grpo import compute_group_advantages, plugin_grpo_loss
from .hint_quality import score_hint_fields
from .hint_policy import HintPolicyRuntime, HintSegment, build_semantic_vocab_mask
from .policy import CONTROLLER_LABELS, FrozenHostPluginPolicy, RetrievalPluginPolicy
from .prompts import (
    controller_system_prompt,
    controller_user_prompt,
    reroute_system_prompt,
    reroute_user_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class ControllerStep:
    input_ids: list[int]
    action: int
    allowed: list[bool]
    old_log_prob: float


@dataclass
class RerouteStep:
    host_prefix_ids: list[int]
    plugin_prefix_ids: list[int]
    target_ids: list[int]
    old_log_probs: list[float]
    temperature: float
    top_p: float


@dataclass
class PluginTrace:
    group_id: str
    training: bool
    controller_steps: list[ControllerStep] = field(default_factory=list)
    reroute_steps: list[RerouteStep] = field(default_factory=list)
    hint_steps: list[HintSegment] = field(default_factory=list)
    hint_quality_rewards: list[float] = field(default_factory=list)
    last_reroute_round: int = -(10**9)

    @property
    def action_count(self) -> int:
        return (
            len(self.controller_steps)
            + sum(len(step.target_ids) for step in self.reroute_steps)
            + sum(len(step.target_ids) for step in self.hint_steps)
        )


class BeginRequest(BaseModel):
    trace_id: str
    group_id: str
    training: bool = True


class ControlRequest(BaseModel):
    trace_id: str
    group_id: str
    question: str
    history: str
    round_num: int
    training: bool = True


class RerouteRequest(BaseModel):
    trace_id: str
    group_id: str
    question: str
    history: str
    training: bool = True


class HintRequest(BaseModel):
    trace_id: str
    group_id: str
    question: str
    history: str
    training: bool = True


class RewardItem(BaseModel):
    trace_id: str
    group_id: str
    reward: float


class UpdateRequest(BaseModel):
    items: list[RewardItem]


class SaveRequest(BaseModel):
    step: int


def _chat_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
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
    tokenizer,
    question: str,
    history: str,
    *,
    system_builder,
    user_builder,
    max_length: int,
) -> list[int]:
    def encode(selected_history: str) -> list[int]:
        return _chat_ids(
            tokenizer,
            [
                {"role": "system", "content": system_builder()},
                {"role": "user", "content": user_builder(question, selected_history)},
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


def build_hint_query(fields: tuple[str, str, str], max_words: int = 24) -> str:
    raw = re.sub(r"[;\r\n]+", " ", " ".join(fields))
    query = " ".join(raw.split())
    if not query:
        raise ValueError("Hint fields produced an empty query")
    return " ".join(query.split()[:max_words])


def _parse_hint_budgets(value: str) -> tuple[int, int, int]:
    budgets = tuple(int(item.strip()) for item in value.split(","))
    if len(budgets) != 3 or min(budgets) <= 0 or sum(budgets) > 64:
        raise argparse.ArgumentTypeError("hint slot budgets must be three positive integers with total <= 64")
    return budgets


def _clean_query(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?[^>]+>", " ", text)
    text = text.strip().strip('"').strip("'").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Corrected policy generated an empty reroute query")
    query = lines[0]
    query = re.sub(r"^(?:query|search query)\s*:\s*", "", query, flags=re.IGNORECASE)
    if not query:
        raise ValueError("Corrected policy generated an empty reroute query")
    return query[:1000]


class PluginGRPOEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        hint_checkpoint = (
            Path(args.hint_checkpoint).expanduser().resolve() if getattr(args, "hint_checkpoint", None) else None
        )
        resume_checkpoint = Path(args.resume).expanduser().resolve() if getattr(args, "resume", None) else None
        if hint_checkpoint is not None and resume_checkpoint is not None:
            raise ValueError("--hint-checkpoint and --resume are mutually exclusive")
        initial_checkpoint = hint_checkpoint or resume_checkpoint
        adapter_path = (
            initial_checkpoint / "lora_adapter"
            if initial_checkpoint is not None
            else Path(args.adapter_path).expanduser().resolve()
        )
        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Plugin LoRA adapter is missing: {adapter_path}")

        self.host_tokenizer = AutoTokenizer.from_pretrained(
            args.host_model, use_fast=True, trust_remote_code=True, local_files_only=args.local_files_only
        )
        self.host = AutoModelForCausalLM.from_pretrained(
            args.host_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        self.host.to(self.device)
        self.host.eval()
        host_vocab_size = int(self.host.config.vocab_size)

        plugin, self.plugin_tokenizer = RetrievalPluginPolicy.from_adapter(
            adapter_path,
            host_vocab_size=host_vocab_size,
            correction_rank=args.correction_rank,
            torch_dtype=dtype,
            train_full_backbone=args.train_full_backbone,
            local_files_only=args.local_files_only,
        )
        plugin.to(self.device)
        self.policy = FrozenHostPluginPolicy(
            self.host,
            plugin,
            alpha=args.alpha,
            valid_vocab_size=len(self.host_tokenizer),
        )
        self.policy.to(self.device)
        self.policy.train()

        host_vocab = self.host_tokenizer.get_vocab()
        plugin_vocab = self.plugin_tokenizer.get_vocab()
        if host_vocab != plugin_vocab:
            mismatches = sum(plugin_vocab.get(token) != token_id for token, token_id in host_vocab.items())
            raise ValueError(f"Host/plugin token-id maps differ ({mismatches} mismatches)")

        self.hint_policy_enabled = bool(getattr(args, "hint_policy", False) or hint_checkpoint)
        if hint_checkpoint is not None:
            heads = torch.load(hint_checkpoint / "plugin_heads.pt", map_location=self.device, weights_only=True)
            checkpoint_rank = int(heads.get("correction_rank", args.correction_rank))
            if checkpoint_rank != args.correction_rank:
                raise ValueError(
                    f"Hint checkpoint correction_rank={checkpoint_rank}, but --correction-rank={args.correction_rank}"
                )
            self.policy.plugin.correction_head.load_state_dict(heads["correction_head"])
            self.policy.alpha = float(heads.get("alpha", self.policy.alpha))

        self.hint_runtime: HintPolicyRuntime | None = None
        if self.hint_policy_enabled:
            self.policy.plugin.controller_head.requires_grad_(False)
            semantic_mask = build_semantic_vocab_mask(
                self.plugin_tokenizer,
                host_vocab_size,
                # v2 admitted whitespace-only tokens. Use a new cache name so
                # resumed runs cannot silently reload the invalid old mask.
                self.output_dir / "semantic_vocab_mask_v4_strict_english.pt",
            )
            self.hint_runtime = HintPolicyRuntime(
                self.policy,
                self.host_tokenizer,
                self.plugin_tokenizer,
                semantic_vocab_mask=semantic_mask,
                max_context_length=args.max_plugin_length,
            )

        if self.hint_policy_enabled:
            correction_parameters = list(self.policy.plugin.correction_head.parameters())
            correction_ids = {id(parameter) for parameter in correction_parameters}
            lora_parameters = [
                parameter
                for parameter in self.policy.plugin.backbone.parameters()
                if parameter.requires_grad and id(parameter) not in correction_ids
            ]
            if not lora_parameters:
                raise RuntimeError("Hint policy has no trainable LoRA parameters")
            parameter_groups = [
                {"params": correction_parameters, "lr": args.correction_learning_rate},
                {"params": lora_parameters, "lr": args.learning_rate},
            ]
        else:
            parameters = list(self.policy.plugin.trainable_parameters())
            if not parameters:
                raise RuntimeError("Plugin has no trainable parameters")
            parameter_groups = [{"params": parameters, "lr": args.learning_rate}]
        self.trainable_parameters = [parameter for group in parameter_groups for parameter in group["params"]]
        self.optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
        self.max_grad_norm = args.max_grad_norm
        self.clip_ratio = args.clip_ratio
        self.controller_temperature = args.controller_temperature
        self.reroute_temperature = args.reroute_temperature
        self.reroute_top_p = args.reroute_top_p
        self.reroute_max_new_tokens = args.reroute_max_new_tokens
        self.hint_slot_token_budgets = args.hint_slot_token_budgets
        self.hint_temperature = args.hint_temperature
        self.hint_top_p = args.hint_top_p
        self.hint_correction_topk = args.hint_correction_topk
        self.hint_query_max_words = args.hint_query_max_words
        self.hint_quality_reward_weight = args.hint_quality_reward_weight
        self.hint_invalid_penalty = args.hint_invalid_penalty
        self.hint_quality_max_field_words = args.hint_quality_max_field_words
        self.reset_optimizer_on_resume = bool(args.reset_optimizer)
        self.reset_correction_head_on_resume = bool(args.reset_correction_head)
        self.max_plugin_length = args.max_plugin_length
        self.min_stop_rounds = args.min_stop_rounds
        self.min_reroute_rounds = args.min_reroute_rounds
        self.reroute_cooldown_rounds = args.reroute_cooldown_rounds
        self.traces: dict[str, PluginTrace] = {}
        self.update_step = 0
        self.lock = threading.RLock()

        if resume_checkpoint is not None:
            self.load_checkpoint(resume_checkpoint, load_optimizer=not self.reset_optimizer_on_resume)
        if self.reset_correction_head_on_resume:
            # Keep the resumed LoRA/controller and trainer step, but discard a
            # correction direction already shown to produce meta-language.
            self.policy.plugin.correction_head.down.reset_parameters()
            torch.nn.init.zeros_(self.policy.plugin.correction_head.up.weight)
            # Stale moments must never be applied to freshly reset weights.
            self.optimizer.state.clear()

    def begin(self, trace_id: str, group_id: str, training: bool) -> None:
        existing = self.traces.get(trace_id)
        if existing is not None and existing.group_id != group_id:
            raise ValueError(f"trace_id {trace_id!r} was reused across groups")
        self.traces.setdefault(trace_id, PluginTrace(group_id=group_id, training=training))

    def _trace(self, trace_id: str, group_id: str, training: bool) -> PluginTrace:
        self.begin(trace_id, group_id, training)
        return self.traces[trace_id]

    def control(self, request: ControlRequest) -> dict[str, Any]:
        trace = self._trace(request.trace_id, request.group_id, request.training)
        input_ids = _trimmed_history_ids(
            self.plugin_tokenizer,
            request.question,
            request.history,
            system_builder=controller_system_prompt,
            user_builder=controller_user_prompt,
            max_length=self.max_plugin_length,
        )
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        self.policy.plugin.eval()
        with torch.no_grad():
            logits = self.policy.plugin.controller_logits(ids).float()[0] / self.controller_temperature

        allowed = [
            True,
            request.round_num >= self.min_stop_rounds,
            request.round_num >= self.min_reroute_rounds
            and request.round_num - trace.last_reroute_round > self.reroute_cooldown_rounds,
        ]
        allowed_tensor = torch.tensor(allowed, dtype=torch.bool, device=self.device)
        masked_logits = logits.masked_fill(~allowed_tensor, -torch.inf)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        if request.training:
            action = int(torch.multinomial(log_probs.exp(), 1).item())
        else:
            action = int(masked_logits.argmax().item())
        if CONTROLLER_LABELS[action] == "REROUTE":
            trace.last_reroute_round = request.round_num
        if request.training:
            trace.controller_steps.append(
                ControllerStep(
                    input_ids=input_ids,
                    action=action,
                    allowed=allowed,
                    old_log_prob=float(log_probs[action].item()),
                )
            )
        probabilities = log_probs.exp().cpu().tolist()
        return {
            "action": CONTROLLER_LABELS[action],
            "action_id": action,
            "probabilities": dict(zip(CONTROLLER_LABELS, probabilities, strict=True)),
            "allowed": dict(zip(CONTROLLER_LABELS, allowed, strict=True)),
        }

    def reroute(self, request: RerouteRequest) -> dict[str, Any]:
        trace = self._trace(request.trace_id, request.group_id, request.training)
        plugin_prefix = _trimmed_history_ids(
            self.plugin_tokenizer,
            request.question,
            request.history,
            system_builder=reroute_system_prompt,
            user_builder=reroute_user_prompt,
            max_length=self.max_plugin_length,
        )
        # The vocabularies have an identical id mapping, but each tokenizer keeps
        # its own chat template and EOS semantics.
        host_prefix = _trimmed_history_ids(
            self.host_tokenizer,
            request.question,
            request.history,
            system_builder=reroute_system_prompt,
            user_builder=reroute_user_prompt,
            max_length=min(self.max_plugin_length, int(self.host.config.max_position_embeddings)),
        )
        host_ids = torch.tensor([host_prefix], dtype=torch.long, device=self.device)
        plugin_ids = torch.tensor([plugin_prefix], dtype=torch.long, device=self.device)
        eos_ids = sorted(
            {
                token_id
                for token_id in (
                    self.host_tokenizer.eos_token_id,
                    self.host_tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                    self.host_tokenizer.convert_tokens_to_ids("<|im_end|>"),
                )
                if isinstance(token_id, int) and token_id >= 0
            }
        )
        self.policy.eval()
        generated = self.policy.generate_reroute(
            host_ids,
            plugin_ids,
            max_new_tokens=self.reroute_max_new_tokens,
            eos_token_ids=eos_ids,
            temperature=self.reroute_temperature,
            top_p=self.reroute_top_p,
        )
        target_ids = generated.token_ids[0].cpu().tolist()
        old_log_probs = generated.log_probs[0].float().cpu().tolist()
        raw = self.host_tokenizer.decode(target_ids, skip_special_tokens=True)
        query = _clean_query(raw)
        if request.training:
            trace.reroute_steps.append(
                RerouteStep(
                    host_prefix_ids=host_prefix,
                    plugin_prefix_ids=plugin_prefix,
                    target_ids=target_ids,
                    old_log_probs=old_log_probs,
                    temperature=self.reroute_temperature,
                    top_p=self.reroute_top_p,
                )
            )
        return {"query": query, "token_ids": target_ids, "old_log_probs": old_log_probs}

    def hint(self, request: HintRequest) -> dict[str, Any]:
        if self.hint_runtime is None:
            raise RuntimeError("Three-field hint policy is not enabled")
        trace = self._trace(request.trace_id, request.group_id, request.training)
        self.policy.eval()
        sample = self.hint_runtime.generate_hint(
            request.question,
            request.history,
            slot_token_budgets=self.hint_slot_token_budgets,
            temperature=self.hint_temperature,
            top_p=self.hint_top_p,
            correction_topk=self.hint_correction_topk,
        )
        query = build_hint_query(sample.fields, max_words=self.hint_query_max_words)
        quality = score_hint_fields(
            sample.fields,
            request.question,
            request.history,
            max_field_words=self.hint_quality_max_field_words,
        )
        if request.training:
            trace.hint_steps.extend(sample.segments)
            trace.hint_quality_rewards.append(float(quality["reward"]))
        return {
            "query": query,
            "fields": list(sample.fields),
            "hint": sample.text,
            "token_count": sample.semantic_tokens,
            "quality": quality,
        }

    def _controller_new_log_prob(self, step: ControllerStep) -> torch.Tensor:
        ids = torch.tensor([step.input_ids], dtype=torch.long, device=self.device)
        logits = self.policy.plugin.controller_logits(ids).float()
        allowed = torch.tensor([step.allowed], dtype=torch.bool, device=self.device)
        logits = logits.masked_fill(~allowed, -torch.inf) / self.controller_temperature
        action = torch.tensor([[step.action]], dtype=torch.long, device=self.device)
        return F.log_softmax(logits, dim=-1).gather(-1, action).reshape(-1)

    def _reroute_new_log_probs(self, step: RerouteStep) -> torch.Tensor:
        host_prefix = torch.tensor([step.host_prefix_ids], dtype=torch.long, device=self.device)
        plugin_prefix = torch.tensor([step.plugin_prefix_ids], dtype=torch.long, device=self.device)
        target = torch.tensor([step.target_ids], dtype=torch.long, device=self.device)
        return self.policy.score_reroute_tokens(
            host_prefix,
            plugin_prefix,
            target,
            temperature=step.temperature,
            top_p=step.top_p,
        ).reshape(-1)

    def _hint_new_log_probs(self, step: HintSegment) -> torch.Tensor:
        if self.hint_runtime is None:
            raise RuntimeError("Cannot score hint tokens when hint policy is disabled")
        return self.hint_runtime.score_segment(step).reshape(-1)

    def _backward_action_group(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: list[float],
        advantage: torch.Tensor,
        total_actions: int,
    ) -> tuple[float, float, float, int]:
        """Backpropagate one controller decision or one reroute token group.

        Keeping every forward graph for an entire trajectory can exceed an
        H200 even though the 0.6B plugin itself is small.  A trace-level mean
        weighted by ``trace.action_count / total_actions`` is algebraically
        identical to immediately backpropagating each group weighted by its
        own action count over ``total_actions``.
        """
        new = new_log_probs.reshape(1, -1)
        count = int(new.shape[1])
        if count == 0:
            return 0.0, 0.0, 0.0, 0
        if len(old_log_probs) != count:
            raise ValueError(f"Expected {count} old log-probs, got {len(old_log_probs)}")
        old = torch.tensor([old_log_probs], dtype=torch.float32, device=self.device)
        mask = torch.ones_like(new, dtype=torch.bool)
        loss, metrics = plugin_grpo_loss(
            new,
            old,
            advantage.reshape(1).to(self.device),
            mask,
            clip_ratio=self.clip_ratio,
        )
        weight = count / total_actions
        (loss * weight).backward()
        return (
            float(loss.detach().item()) * weight,
            float(metrics["plugin/approx_kl"].item()) * count,
            float(metrics["plugin/clip_fraction"].item()) * count,
            count,
        )

    def update(self, items: list[RewardItem]) -> dict[str, Any]:
        if not items:
            raise ValueError("An update requires at least one rewarded trajectory")
        traces: list[PluginTrace] = []
        for item in items:
            trace = self.traces.get(item.trace_id)
            if trace is None:
                raise KeyError(f"Unknown plugin trace: {item.trace_id}")
            if trace.group_id != item.group_id:
                raise ValueError(f"Group mismatch for trace {item.trace_id}")
            traces.append(trace)

        outcome_rewards = torch.tensor([item.reward for item in items], dtype=torch.float32)
        hint_quality_rewards = torch.tensor(
            [
                sum(trace.hint_quality_rewards) / len(trace.hint_quality_rewards)
                if trace.hint_quality_rewards
                else 0.0
                for trace in traces
            ],
            dtype=torch.float32,
        )
        has_hint = torch.tensor(
            [bool(trace.hint_quality_rewards) for trace in traces],
            dtype=torch.bool,
        )
        quality_weight = float(getattr(self, "hint_quality_reward_weight", 0.5))
        invalid_penalty = float(getattr(self, "hint_invalid_penalty", 0.0))
        rewards = outcome_rewards + quality_weight * hint_quality_rewards
        invalid_hint = has_hint & hint_quality_rewards.ne(1.0)
        # Keep dense component ranking among invalid samples, but prevent an
        # invalid HINT from winning merely because the final answer was lucky.
        rewards = rewards - invalid_penalty * invalid_hint.float()

        group_lookup: dict[str, int] = {}
        group_ids = []
        for item in items:
            group_lookup.setdefault(item.group_id, len(group_lookup))
            group_ids.append(group_lookup[item.group_id])
        advantages = compute_group_advantages(rewards, torch.tensor(group_ids))

        total_actions = sum(trace.action_count for trace in traces if trace.training)
        if total_actions == 0:
            for item in items:
                self.traces.pop(item.trace_id, None)
            return {"updated": False, "reason": "no_plugin_actions", "trajectories": len(items)}

        self.policy.eval()  # keep SFT LoRA dropout off while retaining gradients
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        approx_kl_sum = 0.0
        clip_count = 0.0
        for trace, advantage in zip(traces, advantages, strict=True):
            if not trace.training or trace.action_count == 0:
                continue
            for step in trace.controller_steps:
                loss_part, kl_part, clip_part, _ = self._backward_action_group(
                    self._controller_new_log_prob(step),
                    [step.old_log_prob],
                    advantage,
                    total_actions,
                )
                loss_sum += loss_part
                approx_kl_sum += kl_part
                clip_count += clip_part
            for step in trace.reroute_steps:
                loss_part, kl_part, clip_part, _ = self._backward_action_group(
                    self._reroute_new_log_probs(step),
                    step.old_log_probs,
                    advantage,
                    total_actions,
                )
                loss_sum += loss_part
                approx_kl_sum += kl_part
                clip_count += clip_part
            for step in trace.hint_steps:
                loss_part, kl_part, clip_part, _ = self._backward_action_group(
                    self._hint_new_log_probs(step),
                    step.old_log_probs,
                    advantage,
                    total_actions,
                )
                loss_sum += loss_part
                approx_kl_sum += kl_part
                clip_count += clip_part

        trainable_parameters = getattr(
            self,
            "trainable_parameters",
            list(self.policy.plugin.trainable_parameters()),
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, self.max_grad_norm)
        self.optimizer.step()
        self.update_step += 1
        for item in items:
            self.traces.pop(item.trace_id, None)
        return {
            "updated": True,
            "update_step": self.update_step,
            "loss": loss_sum,
            "approx_kl": approx_kl_sum / total_actions,
            "clip_fraction": clip_count / total_actions,
            "grad_norm": float(grad_norm),
            "actions": total_actions,
            "trajectories": len(items),
            "reward_mean": float(rewards.mean()),
            "outcome_reward_mean": float(outcome_rewards.mean()),
            "hint_quality_reward_mean": float(hint_quality_rewards.mean()),
            "hint_quality_valid_rate": float((hint_quality_rewards == 1.0).float().mean()),
            "hint_quality_reward_weight": quality_weight,
            "hint_invalid_rate": float(invalid_hint.float().mean()),
            "hint_invalid_penalty": invalid_penalty,
        }

    def save_checkpoint(self, step: int) -> Path:
        destination = self.output_dir / f"global_step_{step}" / "plugin"
        destination.mkdir(parents=True, exist_ok=True)
        adapter_dir = destination / "lora_adapter"
        self.policy.plugin.backbone.save_pretrained(adapter_dir)
        self.plugin_tokenizer.save_pretrained(adapter_dir)
        torch.save(
            {
                "controller_head": self.policy.plugin.controller_head.state_dict(),
                "correction_head": self.policy.plugin.correction_head.state_dict(),
                "alpha": self.policy.alpha,
                "correction_rank": self.policy.plugin.correction_rank,
                "host_vocab_size": self.policy.plugin.host_vocab_size,
                "update_step": self.update_step,
                "policy_mode": "three_field_hint" if self.hint_policy_enabled else "controller_reroute",
                "hint_slot_token_budgets": self.hint_slot_token_budgets,
                "hint_correction_topk": self.hint_correction_topk,
                "hint_query_max_words": self.hint_query_max_words,
                "hint_quality_reward_weight": self.hint_quality_reward_weight,
                "hint_invalid_penalty": self.hint_invalid_penalty,
                "hint_quality_max_field_words": self.hint_quality_max_field_words,
            },
            destination / "plugin_heads.pt",
        )
        torch.save(self.optimizer.state_dict(), destination / "optimizer.pt")
        (destination / "metadata.json").write_text(
            json.dumps(
                {
                    "host_frozen": True,
                    "host_model": self.host.config._name_or_path,
                    "labels": list(CONTROLLER_LABELS),
                    "step": step,
                    "update_step": self.update_step,
                    "policy_mode": "three_field_hint" if self.hint_policy_enabled else "controller_reroute",
                    "hint_slot_token_budgets": self.hint_slot_token_budgets,
                    "hint_correction_topk": self.hint_correction_topk,
                    "hint_query_max_words": self.hint_query_max_words,
                    "hint_quality_reward_weight": self.hint_quality_reward_weight,
                    "hint_invalid_penalty": self.hint_invalid_penalty,
                    "hint_quality_max_field_words": self.hint_quality_max_field_words,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination

    def load_checkpoint(self, checkpoint: Path, *, load_optimizer: bool = True) -> None:
        heads = torch.load(checkpoint / "plugin_heads.pt", map_location=self.device, weights_only=True)
        self.policy.plugin.controller_head.load_state_dict(heads["controller_head"])
        self.policy.plugin.correction_head.load_state_dict(heads["correction_head"])
        self.policy.alpha = float(heads.get("alpha", self.policy.alpha))
        optimizer_path = checkpoint / "optimizer.pt"
        if load_optimizer and optimizer_path.is_file():
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device, weights_only=True))
        self.update_step = int(heads.get("update_step", 0))


def build_app(engine: PluginGRPOEngine) -> FastAPI:
    app = FastAPI(title="Frozen Host + trainable retrieval plugin GRPO")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "update_step": engine.update_step,
            "traces": len(engine.traces),
            "hint_quality_reward_weight": getattr(engine, "hint_quality_reward_weight", 0.0),
            "hint_invalid_penalty": getattr(engine, "hint_invalid_penalty", 0.0),
            "correction_head_reset": getattr(
                engine, "reset_correction_head_on_resume", False
            ),
        }

    @app.post("/begin")
    def begin(request: BeginRequest) -> dict[str, bool]:
        with engine.lock:
            engine.begin(request.trace_id, request.group_id, request.training)
        return {"ok": True}

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

    @app.post("/hint")
    def hint(request: HintRequest) -> dict[str, Any]:
        try:
            with engine.lock:
                return engine.hint(request)
        except Exception as exc:
            logger.exception("Plugin /hint failed for trace_id=%s", request.trace_id)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/update")
    def update(request: UpdateRequest) -> dict[str, Any]:
        try:
            with engine.lock:
                return engine.update(request.items)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/save")
    def save(request: SaveRequest) -> dict[str, str]:
        with engine.lock:
            path = engine.save_checkpoint(request.step)
        return {"path": str(path)}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-model", default="simplex-ai-inc/LiteResearcher-4B")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--correction-rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--correction-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--controller-temperature", type=float, default=1.0)
    parser.add_argument("--reroute-temperature", type=float, default=1.0)
    parser.add_argument("--reroute-top-p", type=float, default=0.95)
    parser.add_argument("--reroute-max-new-tokens", type=int, default=64)
    parser.add_argument("--hint-policy", action="store_true")
    parser.add_argument("--hint-checkpoint")
    parser.add_argument("--hint-slot-token-budgets", type=_parse_hint_budgets, default=(16, 12, 12))
    parser.add_argument("--hint-temperature", type=float, default=1.0)
    parser.add_argument("--hint-top-p", type=float, default=0.95)
    parser.add_argument("--hint-correction-topk", type=int, default=128)
    parser.add_argument("--hint-query-max-words", type=int, default=24)
    parser.add_argument("--hint-quality-reward-weight", type=float, default=0.5)
    parser.add_argument("--hint-invalid-penalty", type=float, default=2.0)
    parser.add_argument("--hint-quality-max-field-words", type=int, default=8)
    parser.add_argument("--max-plugin-length", type=int, default=8192)
    parser.add_argument("--min-stop-rounds", type=int, default=15)
    parser.add_argument("--min-reroute-rounds", type=int, default=8)
    parser.add_argument("--reroute-cooldown-rounds", type=int, default=4)
    parser.add_argument("--train-full-backbone", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Load resumed plugin weights and step, but start with a fresh optimizer state.",
    )
    parser.add_argument(
        "--reset-correction-head",
        action="store_true",
        help="Discard a resumed correction head while retaining LoRA/controller weights.",
    )
    args = parser.parse_args()
    if args.resume and args.hint_checkpoint:
        parser.error("--hint-checkpoint and --resume are mutually exclusive")
    if args.hint_correction_topk < 0:
        parser.error("--hint-correction-topk must be non-negative")
    if args.hint_query_max_words <= 0:
        parser.error("--hint-query-max-words must be positive")
    if args.hint_quality_reward_weight < 0:
        parser.error("--hint-quality-reward-weight must be non-negative")
    if args.hint_invalid_penalty < 0:
        parser.error("--hint-invalid-penalty must be non-negative")
    if args.hint_quality_max_field_words <= 0:
        parser.error("--hint-quality-max-field-words must be positive")
    return args


def main() -> None:
    import uvicorn

    args = parse_args()
    engine = PluginGRPOEngine(args)
    uvicorn.run(build_app(engine), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
