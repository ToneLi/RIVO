#!/usr/bin/env python
"""Serve last-four-layer attention metrics for retrieval-boundary ASAG."""

import argparse
import asyncio

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from asag_attention import LocalAttentionProbe


class AnalyzeRequest(BaseModel):
    session_id: str | None = None
    history_token_ids: list[int]
    probe_token_ids: list[int]
    previous_span: dict | None = None
    current_span: dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device such as cuda:0/cpu, or auto to shard across visible GPUs.",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()

    probe = LocalAttentionProbe(args.model, device=args.device, dtype=args.dtype)
    smoke_metrics = probe.smoke_test()
    print(
        "ASAG attention smoke test passed: "
        f"layers={smoke_metrics['layers_used']} "
        f"window={smoke_metrics['decoding_window_tokens']} "
        f"selective={probe.selective_attention} "
        f"kv_offload={probe.kv_offload} "
        f"sessions={probe.max_cached_sessions} "
        f"max_kv_per_session={probe.estimated_kv_gib_per_session():.2f}GiB"
    )
    app = FastAPI(title="LiteResearcher retrieval-boundary ASAG attention probe")

    @app.get("/")
    async def health() -> dict:
        return {
            "status": "ok",
            "device": str(probe.device),
            "selective_attention": probe.selective_attention,
            "monitored_layers": probe.monitored_layer_count,
            "kv_offload": probe.kv_offload,
            "max_cached_sessions": probe.max_cached_sessions,
            "max_kv_gib_per_session": probe.estimated_kv_gib_per_session(),
        }

    @app.post("/analyze")
    async def analyze(request: AnalyzeRequest) -> dict:
        # Model forward is blocking and memory-heavy; the probe itself also
        # serializes forwards. Moving it off the event loop keeps health checks
        # responsive while concurrent checkpoint requests queue safely.
        return await asyncio.to_thread(
            probe.analyze,
            request.history_token_ids,
            request.probe_token_ids,
            request.previous_span,
            request.current_span,
            request.session_id,
        )

    @app.delete("/sessions/{session_id:path}")
    async def release_session(session_id: str) -> dict:
        released = await asyncio.to_thread(probe.release_session, session_id)
        return {"released": released}

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
