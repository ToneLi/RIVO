"""KV-cache attention probe over the agent's real tokenized trajectory."""

from __future__ import annotations

import math
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int

    @classmethod
    def from_dict(cls, value: Optional[dict]) -> Optional["TokenSpan"]:
        if value is None:
            return None
        span = cls(start=int(value["start"]), end=int(value["end"]))
        if span.start < 0 or span.end <= span.start:
            raise ValueError(f"invalid token span: {value}")
        return span


@dataclass
class AttentionSession:
    prefix_token_ids: list[int]
    past_key_values: Any


class LocalAttentionProbe:
    """Measure last-layer attention without reconstructing a synthetic prompt.

    The vLLM tokenizer supplies the exact history/probe token IDs.  The long
    prefix is prefetched once into a KV cache; only the fixed recent-history
    window plus answer-probe suffix requests attention tensors.
    """

    def __init__(self, model_name_or_path: str, device: str, dtype: str) -> None:
        self.device = device
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        model_kwargs = {
            "trust_remote_code": True,
            "dtype": torch_dtype,
            # Prefill starts with memory-efficient SDPA; analyze() explicitly
            # switches only the short monitoring pass to eager attention.
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }
        if device == "auto":
            model_kwargs["device_map"] = "balanced"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **model_kwargs
            ).eval()
        else:
            target_device = torch.device(device)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **model_kwargs
            ).to(target_device).eval()
        self._initialize_runtime()

    @classmethod
    def from_existing_model(
        cls,
        model: Any,
        *,
        device: str,
        lock: Optional[Any] = None,
    ) -> "LocalAttentionProbe":
        """Build an identical ASAG probe around an already-loaded Host model.

        The GRPO sidecar already owns the frozen LiteResearcher Host needed for
        corrected-logits decoding. Reusing that exact object avoids loading a
        second 4B copy while preserving the same attention implementation,
        token IDs, KV-cache behavior, and ASAG equations.
        """
        probe = cls.__new__(cls)
        probe.device = device
        probe.model = model.eval()
        probe._initialize_runtime(lock=lock)
        return probe

    def _initialize_runtime(self, lock: Optional[Any] = None) -> None:
        self.input_device = self.model.get_input_embeddings().weight.device
        self.lock = lock if lock is not None else threading.Lock()
        self.attention_window_tokens = int(
            os.getenv("ASAG_DECODING_WINDOW_TOKENS", "32")
        )
        # Backward-compatible diagnostic/config attribute.
        self.monitor_history_tokens = self.attention_window_tokens
        self.max_probe_tokens = int(os.getenv("ASAG_MAX_PROBE_TOKENS", "64"))
        self.max_history_tokens = int(os.getenv("ASAG_MAX_HISTORY_TOKENS", "24000"))
        self.max_cached_sessions = int(os.getenv("ASAG_MAX_CACHED_SESSIONS", "6"))
        self.kv_offload = os.getenv("ASAG_KV_OFFLOAD", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        self.selective_attention = os.getenv(
            "ASAG_SELECTIVE_ATTENTION", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.monitored_layer_count = int(
            os.getenv("ASAG_MONITORED_LAYERS", "4")
        )
        if self.monitored_layer_count < 1:
            raise ValueError("ASAG_MONITORED_LAYERS must be positive")
        self.sessions: OrderedDict[str, AttentionSession] = OrderedDict()

    def estimated_kv_gib_per_session(self) -> float:
        """Worst-case full-history KV footprint for one resident session."""
        config = self.model.config
        layers = int(config.num_hidden_layers)
        kv_heads = int(config.num_key_value_heads)
        head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        element_size = torch.empty((), dtype=self.model.dtype).element_size()
        byte_count = (
            self.max_history_tokens
            * layers
            * kv_heads
            * head_dim
            * 2  # key and value
            * element_size
        )
        return byte_count / (1024 ** 3)

    def smoke_test(self) -> dict[str, Any]:
        """Exercise both the cached SDPA prefill and eager monitoring pass."""
        token_id = self.model.config.eos_token_id
        if isinstance(token_id, list):
            token_id = token_id[0] if token_id else None
        if token_id is None:
            token_id = self.model.config.bos_token_id
        if token_id is None:
            token_id = 0
        history_length = max(self.attention_window_tokens + 2, 4)
        return self.analyze(
            history_token_ids=[int(token_id)] * history_length,
            probe_token_ids=[int(token_id)],
            previous_span=None,
            current_span={"start": history_length - 2, "end": history_length},
        )

    @staticmethod
    def _span_attention(attention: torch.Tensor, span: TokenSpan) -> float:
        return float(attention[..., span.start : span.end].mean().item())

    @staticmethod
    def _validate_span(span: Optional[TokenSpan], history_length: int) -> None:
        if span is not None and span.end > history_length:
            raise ValueError(
                f"token span {span} exceeds history length {history_length}"
            )

    @staticmethod
    def _common_prefix_length(left: list[int], right: list[int]) -> int:
        limit = min(len(left), len(right))
        index = 0
        while index < limit and left[index] == right[index]:
            index += 1
        return index

    def _new_dynamic_cache(self) -> DynamicCache:
        return DynamicCache(
            config=self.model.config,
            offloading=self.kv_offload,
            offload_only_non_sliding=True,
        )

    def _forward_prefix(self, token_ids: list[int], past_key_values: Any) -> Any:
        if not token_ids:
            return past_key_values
        prefix_tensor = torch.tensor(
            [token_ids], dtype=torch.long, device=self.input_device
        )
        kwargs = {
            "input_ids": prefix_tensor,
            "use_cache": True,
            "output_attentions": False,
            "return_dict": True,
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        elif getattr(self, "kv_offload", False):
            kwargs["past_key_values"] = self._new_dynamic_cache()
        outputs = self.model(**kwargs)
        return outputs.past_key_values

    def _decoder_layers(self) -> Optional[list[Any]]:
        """Return decoder layers for architectures with a standard HF layout."""
        monitored_layer_count = getattr(self, "monitored_layer_count", 4)
        candidates = [
            getattr(getattr(self.model, "model", None), "layers", None),
            getattr(self.model, "layers", None),
            getattr(getattr(self.model, "transformer", None), "h", None),
        ]
        for layers in candidates:
            if layers is not None and len(layers) >= monitored_layer_count:
                return list(layers)
        return None

    @contextmanager
    def _capture_monitored_attentions(self):
        """Keep early layers on SDPA and capture eager weights from only the tail.

        Qwen attention selects its backend inside each attention module from the
        shared config.  Wrapping only the final attention modules lets their
        individual calls use eager attention while every preceding layer keeps
        using SDPA.  The model-level forward may discard attention weights; the
        wrappers retain the four tensors needed by ASAG.
        """
        selective_attention = getattr(self, "selective_attention", True)
        monitored_layer_count = getattr(self, "monitored_layer_count", 4)
        layers = self._decoder_layers() if selective_attention else None
        if layers is None:
            yield None
            return

        captured: list[torch.Tensor] = []
        patched: list[tuple[Any, bool, Any]] = []
        for layer in layers[-monitored_layer_count:]:
            attention = getattr(layer, "self_attn", None)
            if attention is None or not hasattr(attention, "config"):
                for module, had_instance_forward, previous_forward in reversed(patched):
                    if had_instance_forward:
                        module.forward = previous_forward
                    else:
                        delattr(module, "forward")
                yield None
                return

            had_instance_forward = "forward" in vars(attention)
            previous_forward = vars(attention).get("forward")
            original_forward = attention.forward

            def monitored_forward(module, *args, _original=original_forward, **kwargs):
                config = module.config
                previous_implementation = config._attn_implementation
                config._attn_implementation = "eager"
                try:
                    result = _original(*args, **kwargs)
                finally:
                    config._attn_implementation = previous_implementation
                if not isinstance(result, tuple) or len(result) < 2 or result[1] is None:
                    raise RuntimeError(
                        "monitored attention layer did not return attention weights"
                    )
                captured.append(result[1])
                return result

            attention.forward = MethodType(monitored_forward, attention)
            patched.append((attention, had_instance_forward, previous_forward))

        try:
            yield captured
        finally:
            for module, had_instance_forward, previous_forward in reversed(patched):
                if had_instance_forward:
                    module.forward = previous_forward
                else:
                    delattr(module, "forward")

    def _prepare_session_prefix(
        self,
        session_id: str,
        target_prefix_ids: list[int],
    ) -> tuple[Any, int, int, bool]:
        state = self.sessions.pop(session_id, None)
        reused_tokens = 0
        cache_reset = state is None
        if state is None:
            past_key_values = None
            cached_ids: list[int] = []
        else:
            past_key_values = state.past_key_values
            cached_ids = state.prefix_token_ids
            reused_tokens = self._common_prefix_length(cached_ids, target_prefix_ids)
            if reused_tokens < len(cached_ids):
                if hasattr(past_key_values, "crop"):
                    past_key_values.crop(reused_tokens)
                    cached_ids = cached_ids[:reused_tokens]
                else:
                    past_key_values = None
                    cached_ids = []
                    reused_tokens = 0
                    cache_reset = True

        delta_ids = target_prefix_ids[len(cached_ids) :]
        past_key_values = self._forward_prefix(delta_ids, past_key_values)
        self.sessions[session_id] = AttentionSession(
            prefix_token_ids=list(target_prefix_ids),
            past_key_values=past_key_values,
        )
        while len(self.sessions) > self.max_cached_sessions:
            self.sessions.popitem(last=False)
        return past_key_values, reused_tokens, len(delta_ids), cache_reset

    def release_session(self, session_id: str) -> bool:
        with self.lock:
            return self.sessions.pop(str(session_id), None) is not None

    @torch.inference_mode()
    def analyze(
        self,
        history_token_ids: list[int],
        probe_token_ids: list[int],
        previous_span: Optional[dict],
        current_span: dict,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if not history_token_ids:
            raise ValueError("history_token_ids must not be empty")
        if not probe_token_ids:
            raise ValueError("probe_token_ids must not be empty")
        if len(history_token_ids) > self.max_history_tokens:
            raise ValueError("history token input exceeds ASAG_MAX_HISTORY_TOKENS")
        if len(probe_token_ids) > self.max_probe_tokens:
            raise ValueError("probe token input exceeds ASAG_MAX_PROBE_TOKENS")

        previous = TokenSpan.from_dict(previous_span)
        current = TokenSpan.from_dict(current_span)
        assert current is not None
        self._validate_span(previous, len(history_token_ids))
        self._validate_span(current, len(history_token_ids))

        request_started = time.perf_counter()
        with self.lock:
            lock_acquired = time.perf_counter()
            attention_window_tokens = getattr(
                self, "attention_window_tokens", self.monitor_history_tokens
            )
            monitor_count = min(
                max(attention_window_tokens - len(probe_token_ids), 0),
                len(history_token_ids),
            )
            if monitor_count:
                prefix_ids = history_token_ids[:-monitor_count]
                monitored_ids = history_token_ids[-monitor_count:] + probe_token_ids
            else:
                prefix_ids = history_token_ids
                monitored_ids = list(probe_token_ids)

            reused_prefix_tokens = 0
            cache_reset = True
            if session_id:
                # Tests constructing the probe via __new__ retain stateless
                # behavior unless session storage is explicitly configured.
                if not hasattr(self, "sessions"):
                    self.sessions = OrderedDict()
                    self.max_cached_sessions = 1
                    self.kv_offload = False
                past_key_values, reused_prefix_tokens, prefill_tokens, cache_reset = (
                    self._prepare_session_prefix(str(session_id), prefix_ids)
                )
            else:
                past_key_values = self._forward_prefix(prefix_ids, None)
                prefill_tokens = len(prefix_ids)

            monitor_tensor = torch.tensor(
                [monitored_ids], dtype=torch.long, device=self.input_device
            )
            attentions = None
            try:
                with self._capture_monitored_attentions() as captured:
                    if captured is None:
                        # Compatibility fallback for non-Qwen/custom models.
                        self.model.set_attn_implementation("eager")
                        outputs = self.model(
                            input_ids=monitor_tensor,
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_attentions=True,
                            return_dict=True,
                        )
                        attentions = outputs.attentions
                    else:
                        # The first N-4 layers retain the configured SDPA
                        # backend. Only wrapped tail layers materialize weights.
                        self.model(
                            input_ids=monitor_tensor,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )
                        attentions = tuple(captured)
            finally:
                self.model.set_attn_implementation("sdpa")
                # DynamicCache is updated in place by the monitoring branch.
                # Remove the replay/probe tokens so the persistent session
                # remains exactly at prefix_ids.
                if session_id and past_key_values is not None and hasattr(past_key_values, "crop"):
                    past_key_values.crop(len(prefix_ids))
            if not attentions:
                raise RuntimeError(
                    "model returned no monitored attention weights"
                )

            monitored_layer_count = getattr(self, "monitored_layer_count", 4)
            layers = attentions[-monitored_layer_count:]
            entropy_total = 0.0
            previous_total = 0.0
            current_total = 0.0
            for layer_attention in layers:
                # All rows belong to the recent-history + answer-probe window;
                # keys retain their original full-history coordinates via KV.
                attention = layer_attention[0].float()
                probabilities = attention.clamp_min(1e-12)
                row_entropy = -(probabilities * probabilities.log()).sum(dim=-1)
                normalizer = max(math.log(attention.shape[-1]), 1.0)
                entropy_total += float((row_entropy / normalizer).sum().item())
                if previous is not None:
                    previous_total += self._span_attention(attention, previous)
                current_total += self._span_attention(attention, current)

            layer_count = len(layers)
            result = {
                "entropy": entropy_total,
                "previous_attention": (
                    previous_total / layer_count if previous is not None else None
                ),
                "current_attention": current_total / layer_count,
                "layers_used": layer_count,
                "heads": int(layers[-1].shape[1]),
                "sequence_tokens": len(history_token_ids) + len(probe_token_ids),
                "prefill_tokens": prefill_tokens,
                "reused_prefix_tokens": reused_prefix_tokens,
                "cache_reset": cache_reset,
                "probe_tokens": len(probe_token_ids),
                "decoding_window_tokens": len(monitored_ids),
                "service_lock_wait_s": lock_acquired - request_started,
            }
            result["service_compute_s"] = time.perf_counter() - lock_acquired
            return result
