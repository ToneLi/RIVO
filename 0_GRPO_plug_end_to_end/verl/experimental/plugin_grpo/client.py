"""Async client used by DeepResearch agent-loop workers."""

from __future__ import annotations

from typing import Any

import httpx


class PluginPolicyClient:
    def __init__(self, base_url: str, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self.client.post(f"{self.base_url}{path}", json=payload)
        except httpx.HTTPError as exc:
            # httpx exceptions carry request/response objects and cannot be
            # reconstructed by Ray's exception unpickler. Cross the actor
            # boundary with a plain, serializable exception instead.
            raise RuntimeError(f"Plugin service request failed on {path}: {exc}") from None
        if response.is_error:
            try:
                error_payload = response.json()
            except ValueError:
                detail = response.text.strip() or response.reason_phrase
            else:
                detail = (
                    error_payload.get("detail", error_payload) if isinstance(error_payload, dict) else error_payload
                )
            raise RuntimeError(f"Plugin service HTTP {response.status_code} on {path}: {detail}")
        result = response.json()
        if not isinstance(result, dict):
            raise TypeError(f"Plugin service returned non-object JSON from {path}")
        return result

    async def begin(self, trace_id: str, group_id: str, training: bool) -> None:
        await self._post(
            "/begin",
            {"trace_id": trace_id, "group_id": group_id, "training": training},
        )

    async def control(
        self,
        *,
        trace_id: str,
        group_id: str,
        question: str,
        history: str,
        round_num: int,
        training: bool,
    ) -> dict[str, Any]:
        return await self._post(
            "/control",
            {
                "trace_id": trace_id,
                "group_id": group_id,
                "question": question,
                "history": history,
                "round_num": round_num,
                "training": training,
            },
        )

    async def reroute(
        self,
        *,
        trace_id: str,
        group_id: str,
        question: str,
        history: str,
        training: bool,
    ) -> dict[str, Any]:
        return await self._post(
            "/reroute",
            {
                "trace_id": trace_id,
                "group_id": group_id,
                "question": question,
                "history": history,
                "training": training,
            },
        )

    async def hint(
        self,
        *,
        trace_id: str,
        group_id: str,
        question: str,
        history: str,
        training: bool,
    ) -> dict[str, Any]:
        return await self._post(
            "/hint",
            {
                "trace_id": trace_id,
                "group_id": group_id,
                "question": question,
                "history": history,
                "training": training,
            },
        )

    async def close(self) -> None:
        await self.client.aclose()
