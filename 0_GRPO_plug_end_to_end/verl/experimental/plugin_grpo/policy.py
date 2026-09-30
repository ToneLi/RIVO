"""Frozen-Host policy with a trainable controller and token-logit correction.

The Host and plugin tokenizers must use the same token-to-id mapping.  The Host
produces the base next-token logits under ``torch.no_grad``; the plugin produces
``delta_logits`` through a low-rank head.  Only LoRA/plugin parameters and the
two heads are exposed to the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

CONTROLLER_LABELS = ("CONTINUE", "STOP", "REROUTE")


def _last_hidden(model_output) -> Tensor:
    hidden_states = getattr(model_output, "hidden_states", None)
    if not hidden_states:
        raise RuntimeError("Plugin backbone must return hidden_states")
    return hidden_states[-1]


def _validate_2d_ids(name: str, value: Tensor) -> None:
    if value.ndim != 2 or value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be a rank-2 integer tensor, got {value.shape}/{value.dtype}")


class LowRankCorrectionHead(nn.Module):
    """Map plugin hidden states to Host-vocabulary logit corrections."""

    def __init__(self, hidden_size: int, rank: int, vocab_size: int) -> None:
        super().__init__()
        if min(hidden_size, rank, vocab_size) <= 0:
            raise ValueError("hidden_size, rank, and vocab_size must be positive")
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, vocab_size, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=hidden_size**-0.5)
        # Start from exactly the frozen Host policy.  As with LoRA, a random
        # first projection and zero second projection still permits gradients.
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.up(self.down(hidden_states.to(self.down.weight.dtype)))


class RetrievalPluginPolicy(nn.Module):
    """Qwen3-0.6B LoRA backbone with Controller and Correction heads."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_size: int,
        host_vocab_size: int,
        correction_rank: int = 64,
        label_token_ids: Sequence[int] | None = None,
        train_full_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.controller_head = nn.Linear(hidden_size, len(CONTROLLER_LABELS), bias=False)
        self.correction_head = LowRankCorrectionHead(hidden_size, correction_rank, host_vocab_size)
        self.host_vocab_size = int(host_vocab_size)
        self.correction_rank = int(correction_rank)

        if not train_full_backbone:
            # PeftModel marks only adapter weights trainable when loaded with
            # is_trainable=True.  For a plain model, freeze the whole backbone.
            has_trainable_adapter = any(
                parameter.requires_grad and "lora_" in name for name, parameter in self.backbone.named_parameters()
            )
            if not has_trainable_adapter:
                self.backbone.requires_grad_(False)

        if label_token_ids is not None:
            self.initialize_controller_from_lm_head(label_token_ids)

    @classmethod
    def from_adapter(
        cls,
        adapter_path: str | Path,
        *,
        host_vocab_size: int,
        correction_rank: int = 64,
        torch_dtype: torch.dtype | None = None,
        device_map=None,
        train_full_backbone: bool = False,
        local_files_only: bool = False,
    ) -> tuple[RetrievalPluginPolicy, object]:
        """Load the SFT LoRA and create both plugin heads.

        The returned tokenizer is the adapter tokenizer used by the original
        three-class service.
        """
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter = Path(adapter_path).expanduser().resolve()
        if not adapter.is_dir():
            raise FileNotFoundError(adapter)
        tokenizer = AutoTokenizer.from_pretrained(
            adapter, use_fast=True, trust_remote_code=True, local_files_only=local_files_only
        )
        peft_config = PeftConfig.from_pretrained(adapter, local_files_only=local_files_only)
        base = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        backbone = PeftModel.from_pretrained(base, adapter, is_trainable=True)
        hidden_size = int(backbone.config.hidden_size)
        label_token_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in CONTROLLER_LABELS]
        if len(set(label_token_ids)) != len(label_token_ids):
            raise ValueError(f"Controller labels do not have distinct first tokens: {label_token_ids}")
        policy = cls(
            backbone,
            hidden_size=hidden_size,
            host_vocab_size=host_vocab_size,
            correction_rank=correction_rank,
            label_token_ids=label_token_ids,
            train_full_backbone=train_full_backbone,
        )
        return policy, tokenizer

    def initialize_controller_from_lm_head(self, label_token_ids: Sequence[int]) -> None:
        if len(label_token_ids) != len(CONTROLLER_LABELS):
            raise ValueError(f"Expected {len(CONTROLLER_LABELS)} label token ids")
        output_embeddings = self.backbone.get_output_embeddings()
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise TypeError("Plugin backbone has no usable output embedding matrix")
        weight = output_embeddings.weight
        if max(label_token_ids) >= weight.shape[0]:
            raise ValueError("A controller label token is outside the plugin vocabulary")
        with torch.no_grad():
            self.controller_head.weight.copy_(weight[list(label_token_ids)].detach())

    def _backbone_hidden(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return _last_hidden(output)

    def controller_logits(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        _validate_2d_ids("input_ids", input_ids)
        hidden = self._backbone_hidden(input_ids, attention_mask)
        if attention_mask is None:
            last_hidden = hidden[:, -1]
        else:
            positions = attention_mask.long().sum(dim=-1).sub(1).clamp_min(0)
            last_hidden = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        return self.controller_head(last_hidden.to(self.controller_head.weight.dtype))

    def controller_log_probs(self, input_ids: Tensor, actions: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        logits = self.controller_logits(input_ids, attention_mask)
        return F.log_softmax(logits.float(), dim=-1).gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)

    def delta_logits(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        _validate_2d_ids("input_ids", input_ids)
        return self.correction_head(self._backbone_hidden(input_ids, attention_mask))

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)


@dataclass
class RerouteGeneration:
    token_ids: Tensor
    log_probs: Tensor
    finished: Tensor


class FrozenHostPluginPolicy(nn.Module):
    """Compose frozen Host logits with trainable plugin corrections."""

    def __init__(
        self,
        host: nn.Module,
        plugin: RetrievalPluginPolicy,
        *,
        alpha: float = 1.0,
        valid_vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        # if alpha < 0:
        #     raise ValueError("alpha must be non-negative")
        self.host = host
        self.plugin = plugin
        self.alpha = float(alpha)
        self.valid_vocab_size = int(valid_vocab_size or plugin.host_vocab_size)
        self.host.requires_grad_(False)
        self.host.eval()

        host_vocab_size = int(self.host.config.vocab_size)
        if host_vocab_size != plugin.host_vocab_size:
            raise ValueError(f"Host vocab={host_vocab_size} does not match correction vocab={plugin.host_vocab_size}")
        if not 0 < self.valid_vocab_size <= host_vocab_size:
            raise ValueError("valid_vocab_size must be within the Host vocabulary")

    def train(self, mode: bool = True):
        super().train(mode)
        # A parent .train() call must never enable Host dropout or cache changes.
        self.host.eval()
        return self

    def corrected_logits(self, host_logits: Tensor, plugin_hidden: Tensor) -> Tensor:
        if host_logits.shape[:-1] != plugin_hidden.shape[:-1]:
            raise ValueError(f"Host/plugin prefix shapes differ: {host_logits.shape} vs {plugin_hidden.shape}")
        if host_logits.shape[-1] != self.plugin.host_vocab_size:
            raise ValueError("Host logits have the wrong vocabulary dimension")
        delta = self.plugin.correction_head(plugin_hidden).float()
        logits = host_logits.detach().float() + self.alpha * delta
        if self.valid_vocab_size < logits.shape[-1]:
            logits[..., self.valid_vocab_size :] = -torch.inf
        return logits

    def score_reroute_tokens(
        self,
        host_prefix_ids: Tensor,
        plugin_prefix_ids: Tensor,
        target_ids: Tensor,
        *,
        host_attention_mask: Tensor | None = None,
        plugin_attention_mask: Tensor | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> Tensor:
        """Recompute on-policy log-probs for recorded reroute query tokens."""
        for name, value in (
            ("host_prefix_ids", host_prefix_ids),
            ("plugin_prefix_ids", plugin_prefix_ids),
            ("target_ids", target_ids),
        ):
            _validate_2d_ids(name, value)
        if not (host_prefix_ids.shape[0] == plugin_prefix_ids.shape[0] == target_ids.shape[0]):
            raise ValueError("Host, plugin, and target batch sizes must match")
        if target_ids.shape[1] == 0:
            return torch.empty(target_ids.shape, dtype=torch.float32, device=target_ids.device)

        host_ids = torch.cat((host_prefix_ids, target_ids), dim=1)
        plugin_ids = torch.cat((plugin_prefix_ids, target_ids), dim=1)
        with torch.no_grad():
            host_output = self.host(
                input_ids=host_ids,
                attention_mask=None,
                use_cache=False,
                return_dict=True,
            )
            host_logits = host_output.logits[:, -target_ids.shape[1] - 1 : -1]
        plugin_hidden = self.plugin._backbone_hidden(plugin_ids, None)
        plugin_hidden = plugin_hidden[:, -target_ids.shape[1] - 1 : -1]
        if temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("Invalid scoring parameters")
        logits = self.corrected_logits(host_logits, plugin_hidden) / temperature
        logits = _top_p_filter(logits, top_p)
        return F.log_softmax(logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def generate_reroute(
        self,
        host_prefix_ids: Tensor,
        plugin_prefix_ids: Tensor,
        *,
        max_new_tokens: int,
        eos_token_ids: Sequence[int],
        temperature: float = 1.0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> RerouteGeneration:
        """Sample a reroute query one corrected Host token at a time."""
        _validate_2d_ids("host_prefix_ids", host_prefix_ids)
        _validate_2d_ids("plugin_prefix_ids", plugin_prefix_ids)
        if host_prefix_ids.shape[0] != plugin_prefix_ids.shape[0]:
            raise ValueError("Host and plugin batch sizes must match")
        if max_new_tokens <= 0 or temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("Invalid generation parameters")

        batch_size = host_prefix_ids.shape[0]
        host_ids = host_prefix_ids
        plugin_ids = plugin_prefix_ids
        finished = torch.zeros(batch_size, dtype=torch.bool, device=host_ids.device)
        sampled_tokens: list[Tensor] = []
        sampled_log_probs: list[Tensor] = []
        eos = torch.tensor(list(eos_token_ids), device=host_ids.device)
        host_past = None
        plugin_past = None
        host_step_ids = host_ids
        plugin_step_ids = plugin_ids

        for _ in range(max_new_tokens):
            host_output = self.host(
                input_ids=host_step_ids,
                past_key_values=host_past,
                use_cache=True,
                return_dict=True,
            )
            plugin_output = self.plugin.backbone(
                input_ids=plugin_step_ids,
                past_key_values=plugin_past,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
            host_past = host_output.past_key_values
            plugin_past = plugin_output.past_key_values
            host_logits = host_output.logits[:, -1]
            plugin_hidden = _last_hidden(plugin_output)[:, -1]
            logits = self.corrected_logits(host_logits, plugin_hidden) / temperature
            logits = _top_p_filter(logits, top_p)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            next_token = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            selected_log_prob = log_probs.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)

            # Finished rows are retained for a rectangular result but do not
            # contribute additional policy-gradient terms.
            next_token = torch.where(finished, eos[0], next_token)
            selected_log_prob = torch.where(finished, torch.zeros_like(selected_log_prob), selected_log_prob)
            sampled_tokens.append(next_token)
            sampled_log_probs.append(selected_log_prob)
            finished |= (next_token.unsqueeze(-1) == eos.unsqueeze(0)).any(dim=-1)
            host_step_ids = next_token.unsqueeze(-1)
            plugin_step_ids = next_token.unsqueeze(-1)
            if bool(finished.all()):
                break

        return RerouteGeneration(
            token_ids=torch.stack(sampled_tokens, dim=1),
            log_probs=torch.stack(sampled_log_probs, dim=1),
            finished=finished,
        )


def _top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
    return torch.empty_like(logits).scatter(-1, sorted_indices, sorted_logits)
