"""Async client and training-compatible history renderer for GRPO plugin inference."""

from __future__ import annotations

import json
from typing import Any

import httpx


def _clip_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n...[middle omitted]...\n"
    if limit <= len(marker):
        return text[:limit]
    remaining = limit - len(marker)
    head = remaining // 2
    return text[:head] + marker + text[-(remaining - head) :]


def _compact_message(
    message: dict[str, Any], per_message_limit: int
) -> str | None:
    role = message.get("role")
    if role in ("system", "user"):
        return None
    compact: dict[str, Any] = {"role": role}
    content = message.get("content")
    if content not in (None, ""):
        compact["content"] = (
            _clip_middle(content, per_message_limit)
            if isinstance(content, str)
            else content
        )
    if role == "assistant" and message.get("tool_calls"):
        calls = []
        for call in message["tool_calls"]:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            name = function.get("name")
            calls.append(
                {
                    "name": name.rsplit(".", 1)[-1].lower()
                    if isinstance(name, str)
                    else name,
                    "arguments": function.get("arguments"),
                }
            )
        if calls:
            compact["tool_calls"] = calls
    return json.dumps(compact, ensure_ascii=False) if len(compact) > 1 else None


def render_history(
    messages: list[dict[str, Any]],
    history_char_limit: int = 16000,
    per_message_char_limit: int = 6000,
) -> str:
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


class GRPOPluginClient:
    def __init__(self, base_url: str, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self.client.post(f"{self.base_url}{path}", json=payload)
        if response.is_error:
            detail = response.text.strip() or "<empty response body>"
            raise httpx.HTTPStatusError(
                f"GRPO plugin {path} returned HTTP {response.status_code}: {detail}",
                request=response.request,
                response=response,
            )
        result = response.json()
        if not isinstance(result, dict):
            raise TypeError(f"GRPO plugin returned non-object JSON from {path}")
        return result

    async def control(
        self,
        question: str,
        messages: list[dict[str, Any]],
        *,
        round_num: int,
        last_reroute_round: int | None,
    ) -> dict[str, Any]:
        return await self._post(
            "/control",
            {
                "question": question,
                "history": render_history(messages),
                "round_num": round_num,
                "last_reroute_round": last_reroute_round,
            },
        )

    async def reroute(
        self,
        question: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Request the training-identical corrected-logits three-field hint."""
        return await self._post(
            "/reroute",
            {
                "question": question,
                "history": render_history(messages),
            },
        )

    async def close(self) -> None:
        await self.client.aclose()
