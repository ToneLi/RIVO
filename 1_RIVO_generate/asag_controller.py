"""Retrieval-boundary adaptation of ASAG for LiteResearcher.

The original paper probes at reasoning action-transition points such as
"Wait".  Deep research already has a natural, deterministic transition point:
the completion of a browser retrieval.  This module keeps the paper's
confidence/entropy definitions while replacing only that checkpoint boundary.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Optional

import httpx


DEFAULT_REROUTE_PROMPT = (
    "The current research direction is not making sufficient progress. "
    "Reconsider the problem from a substantially different angle and generate "
    "a new search query."
)
DEFAULT_VERIFICATION_PROMPT = (
    "The current provisional answer is: {answer}. Treat it as an unverified "
    "hypothesis. Check it "
    "against the retrieved primary evidence, look for contradictions, and use "
    "browser.open/browser.find (or a new search when necessary) to resolve any "
    "missing fact before answering."
)
ASAG_ANSWER_PROBE_PROMPT = (
    "\n\n Final Answer\n\n \\boxed"
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ResearchSegment:
    turn: int
    reasoning: str
    query: str
    evidence: str
    tool_name: str


@dataclass
class ASAGConfig:
    enabled: bool = True
    confidence_threshold: float = 0.95
    entropy_delta_threshold: float = -0.10
    attention_margin: float = 0.0
    max_reroutes: int = 1
    max_verifications: int = 1
    fail_open: bool = False
    attention_timeout_s: float = 300.0
    reroute_prompt: str = DEFAULT_REROUTE_PROMPT
    verification_prompt: str = DEFAULT_VERIFICATION_PROMPT

    @classmethod
    def from_env(cls) -> "ASAGConfig":
        return cls(
            enabled=_env_bool("ASAG_ENABLED", True),
            confidence_threshold=float(os.getenv("ASAG_CONFIDENCE_THRESHOLD", "0.95")),
            entropy_delta_threshold=float(os.getenv("ASAG_ENTROPY_DELTA_THRESHOLD", "-0.10")),
            attention_margin=float(os.getenv("ASAG_ATTENTION_MARGIN", "0.0")),
            max_reroutes=int(os.getenv("ASAG_MAX_REROUTES", "1")),
            max_verifications=int(os.getenv("ASAG_MAX_VERIFICATIONS", "1")),
            fail_open=_env_bool("ASAG_FAIL_OPEN", False),
            attention_timeout_s=float(os.getenv("ASAG_ATTENTION_TIMEOUT_S", "300")),
            reroute_prompt=os.getenv("ASAG_REROUTE_PROMPT", DEFAULT_REROUTE_PROMPT),
            verification_prompt=os.getenv(
                "ASAG_VERIFICATION_PROMPT", DEFAULT_VERIFICATION_PROMPT
            ),
        )


def normalize_provisional_answer(answer: str) -> str:
    """Normalize superficial formatting while preserving answer semantics."""
    value = unicodedata.normalize("NFKC", answer or "").strip().lower()
    value = re.sub(r"</?answer>|</?suggested_answer>", "", value)
    value = re.sub(r"\\(?:boxed|text)\s*", "", value)
    # Spaces, punctuation, and TeX delimiters do not constitute an answer change.
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE)


def token_confidence(token_logprobs: list[Optional[float]]) -> float:
    """Equation (5): arithmetic mean of selected-token probabilities."""
    probabilities = [math.exp(value) for value in token_logprobs if value is not None]
    if not probabilities:
        raise ValueError("answer probe did not return selected-token logprobs")
    return sum(probabilities) / len(probabilities)


def boxed_answer_prefix(text: str) -> tuple[str, bool]:
    """Keep only the first complete braced/boxed answer from a probe."""
    value = text.strip()
    if not value:
        return "", False
    if value.startswith("{"):
        brace_start = 0
    elif value.startswith("\\boxed"):
        brace_start = value.find("{")
        if brace_start < 0:
            return value, False
    else:
        return value.splitlines()[0].strip(), False

    depth = 0
    for index in range(brace_start, len(value)):
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
            if depth == 0:
                return value[: index + 1], True
    return value, False


def boxed_answer_logprobs(
    tokens: list[str],
    token_logprobs: list[Optional[float]],
) -> list[Optional[float]]:
    """Select probabilities only through the closing answer brace."""
    accumulated = ""
    for index, token in enumerate(tokens):
        accumulated += token
        _, complete = boxed_answer_prefix(accumulated)
        if complete:
            return token_logprobs[: index + 1]
    return token_logprobs


class RetrievalBoundaryASAG:
    """Stateful Continue/Reroute/Stop policy for one research question."""

    def __init__(
        self,
        question: str,
        attention_url: Optional[str],
        config: Optional[ASAGConfig] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.question = question
        self.attention_url = attention_url.rstrip("/") if attention_url else None
        self.session_id = str(session_id) if session_id is not None else None
        self.config = config or ASAGConfig.from_env()
        self.previous_segment: Optional[ResearchSegment] = None
        self.initial_entropy: Optional[float] = None
        self.checkpoint_count = 0
        self.reroute_count = 0
        self.verification_count = 0
        self.previous_normalized_answer = ""
        self.previous_message_span: Optional[dict[str, int]] = None

        if self.config.enabled and not self.attention_url:
            raise ValueError("ASAG is enabled but --asag_attention_url was not provided")
        self.client = httpx.AsyncClient(timeout=self.config.attention_timeout_s)

    async def close(self) -> None:
        if self.config.enabled and self.attention_url and self.session_id:
            try:
                await self.client.delete(
                    f"{self.attention_url}/sessions/{self.session_id}"
                )
            except Exception:
                # Session cleanup is best effort; the attention service also
                # bounds its cache with an LRU policy.
                pass
        await self.client.aclose()

    async def evaluate(
        self,
        segment: ResearchSegment,
        provisional_answer: str,
        confidence: float,
        attention_context: dict[str, Any],
        message_span: Optional[dict[str, int]] = None,
    ) -> dict[str, Any]:
        self.checkpoint_count += 1
        record: dict[str, Any] = {
            "checkpoint": self.checkpoint_count,
            "segment": asdict(segment),
            "confidence": confidence,
            "provisional_answer": provisional_answer,
            "decision": "continue",
            "reason": "default_continue",
        }

        normalized_answer = normalize_provisional_answer(provisional_answer)
        answer_stable = bool(
            normalized_answer
            and self.previous_normalized_answer
            and normalized_answer == self.previous_normalized_answer
        )
        record.update(
            {
                "normalized_answer": normalized_answer,
                "answer_stable": answer_stable,
                "evidence_eligible_for_stop": segment.tool_name in {"open", "find"},
            }
        )

        if not self.config.enabled:
            record["reason"] = "asag_disabled"
            self.previous_segment = segment
            self.previous_normalized_answer = normalized_answer
            self.previous_message_span = message_span
            return record

        # Match Algorithm 1 in the paper: at the first ATP, confidence alone
        # is sufficient for early exit.  No entropy forward is needed because
        # there will be no later checkpoint that depends on H1.
        high_confidence = confidence > self.config.confidence_threshold
        if self.checkpoint_count == 1 and high_confidence:
            record.update(
                {
                    "decision": "stop",
                    "reason": "paper_first_checkpoint_high_confidence",
                    "attention_probe_skipped": True,
                }
            )
            self.previous_segment = segment
            self.previous_normalized_answer = normalized_answer
            self.previous_message_span = message_span
            return record

        try:
            attention_context = dict(attention_context)
            if self.session_id is not None:
                attention_context["session_id"] = self.session_id
            response = await self.client.post(
                f"{self.attention_url}/analyze",
                json=attention_context,
            )
            response.raise_for_status()
            metrics = response.json()
        except Exception as exc:
            if not self.config.fail_open:
                raise RuntimeError(f"ASAG attention probe failed: {exc}") from exc
            record.update({"reason": "attention_probe_failed", "error": repr(exc)})
            self.previous_segment = segment
            self.previous_normalized_answer = normalized_answer
            self.previous_message_span = message_span
            return record

        entropy = float(metrics["entropy"])
        if self.initial_entropy is None:
            self.initial_entropy = entropy
            entropy_delta = None
        else:
            entropy_delta = (entropy - self.initial_entropy) / max(abs(self.initial_entropy), 1e-12)

        previous_attention = metrics.get("previous_attention")
        current_attention = float(metrics["current_attention"])
        record.update(
            {
                "entropy": entropy,
                "initial_entropy": self.initial_entropy,
                "entropy_delta": entropy_delta,
                "previous_attention": previous_attention,
                "current_attention": current_attention,
                "layers_used": metrics.get("layers_used"),
                "heads": metrics.get("heads"),
                "attention_sequence_tokens": metrics.get("sequence_tokens"),
                "attention_decoding_window_tokens": metrics.get("decoding_window_tokens"),
                "attention_prefill_tokens": metrics.get("prefill_tokens"),
                "attention_reused_prefix_tokens": metrics.get("reused_prefix_tokens"),
                "attention_cache_reset": metrics.get("cache_reset"),
                "attention_service_lock_wait_s": metrics.get("service_lock_wait_s"),
                "attention_service_compute_s": metrics.get("service_compute_s"),
            }
        )

        converged = (
            entropy_delta is not None
            and entropy_delta < self.config.entropy_delta_threshold
        )
        # Match Algorithm 1 after the first ATP: high confidence and entropy
        # convergence are sufficient for early exit.
        if (
            high_confidence
            and converged
        ):
            record.update(
                {
                    "decision": "stop",
                    "reason": "paper_high_confidence_with_entropy_convergence",
                }
            )
        elif (
            not high_confidence
            and not converged
            and previous_attention is not None
            and float(previous_attention) > current_attention + self.config.attention_margin
        ):
            if self.reroute_count < self.config.max_reroutes:
                self.reroute_count += 1
                record.update(
                    {
                        "decision": "reroute",
                        "reason": "low_confidence_not_converged_and_stuck_on_old_evidence",
                        "reroute_count": self.reroute_count,
                        "reroute_prompt": self.config.reroute_prompt,
                    }
                )
            else:
                record.update(
                    {
                        "decision": "force_answer",
                        "reason": "reroute_limit_reached",
                        "reroute_count": self.reroute_count,
                    }
                )
        elif not high_confidence and converged:
            if self.verification_count < self.config.max_verifications:
                self.verification_count += 1
                record.update(
                    {
                        "decision": "verify",
                        "reason": "attention_converged_but_answer_confidence_low",
                        "verification_count": self.verification_count,
                        "verification_prompt": self.config.verification_prompt.replace(
                            "{answer}", provisional_answer
                        ),
                    }
                )
            else:
                record["reason"] = "verification_limit_reached"
        elif high_confidence and not converged:
            record["reason"] = "confidence_high_but_attention_not_converged"
        else:
            record["reason"] = "evidence_still_advancing_or_confidence_low"

        self.previous_segment = segment
        self.previous_normalized_answer = normalized_answer
        self.previous_message_span = message_span
        return record
