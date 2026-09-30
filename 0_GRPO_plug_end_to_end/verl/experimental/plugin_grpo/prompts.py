"""Prompt and history formatting shared by plugin rollout and training."""

from __future__ import annotations

import json
from typing import Any

from .policy import CONTROLLER_LABELS


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
        "Generate one concise web-search query that redirects a stalled research trajectory. "
        "Output the query only: no explanation, quotes, JSON, XML, or tool-call wrapper."
    )


def reroute_user_prompt(question: str, history: str) -> str:
    return (
        "<question>\n"
        f"{question.strip()}\n"
        "</question>\n\n"
        "<research_history>\n"
        f"{history.strip()}\n"
        "</research_history>\n\n"
        "Search from a substantially different direction and avoid previously unsuccessful queries."
    )


def _clip_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n...[middle omitted]...\n"
    if limit <= len(marker):
        return text[:limit]
    remaining = limit - len(marker)
    head = remaining // 2
    return text[:head] + marker + text[-(remaining - head) :]


def _compact_message(message: dict[str, Any], per_message_limit: int) -> str | None:
    role = message.get("role")
    if role in ("system", "user"):
        return None
    compact: dict[str, Any] = {"role": role}
    content = message.get("content")
    if content not in (None, ""):
        compact["content"] = _clip_middle(content, per_message_limit) if isinstance(content, str) else content
    if role == "assistant" and message.get("tool_calls"):
        calls = []
        for call in message["tool_calls"]:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            name = function.get("name")
            calls.append(
                {
                    "name": name.rsplit(".", 1)[-1].lower() if isinstance(name, str) else name,
                    "arguments": function.get("arguments"),
                }
            )
        if calls:
            compact["tool_calls"] = calls
    return json.dumps(compact, ensure_ascii=False) if len(compact) > 1 else None


def render_history(
    messages: list[dict[str, Any]], history_char_limit: int = 16000, per_message_char_limit: int = 6000
) -> str:
    """Match the SFT classifier's compact retrieval-history representation."""
    pieces: list[str] = []
    used = 0
    omitted = False
    for message in reversed(messages):
        piece = _compact_message(message, per_message_char_limit)
        if not piece:
            continue
        remaining = history_char_limit - used
        if remaining <= 0:
            omitted = True
            break
        if len(piece) > remaining:
            piece = _clip_middle(piece, remaining)
            omitted = True
        pieces.append(piece)
        used += len(piece)
    pieces.reverse()
    if omitted:
        pieces.insert(0, "[Earlier history omitted]")
    return "\n".join(pieces)
