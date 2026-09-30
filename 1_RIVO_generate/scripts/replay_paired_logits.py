#!/usr/bin/env python3
"""Replay the same saved reroute prefix with alpha treatment and control."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from grpo_plugin_service import GRPOPluginInferenceEngine, RerouteRequest  # noqa: E402
from utils.grpo_plugin_client import render_history  # noqa: E402


def load_records(source_dir: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for shard in sorted(source_dir.glob("node_*_shard_*.jsonl")):
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    records[str(row["qid"])] = row
    return records


def intervention_prefix(row: dict) -> tuple[list[dict], dict]:
    successful = [
        trace
        for trace in row.get("grpo_reroute_trace", [])
        if trace.get("logits_applied") and trace.get("query")
    ]
    if len(successful) != 1:
        raise ValueError(
            f"QID {row['qid']} needs exactly one successful reroute; got {len(successful)}"
        )
    trace = successful[0]
    query = trace["query"]
    indices = []
    for index, message in enumerate(row.get("messages", [])):
        if not message.get("retrieval_control"):
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments") or {}
            if (
                function.get("name", "").endswith("search")
                and arguments.get("query") == query
            ):
                indices.append(index)
    if len(indices) != 1:
        raise ValueError(
            f"QID {row['qid']} needs one intervention message; got {len(indices)}"
        )
    return row["messages"][: indices[0]], trace


def set_alpha(engine: GRPOPluginInferenceEngine, alpha: float) -> None:
    engine.alpha = alpha
    engine.policy.alpha = alpha


def replay(
    engine: GRPOPluginInferenceEngine, request: RerouteRequest, alpha: float
) -> dict:
    set_alpha(engine, alpha)
    return engine.reroute(request)


def build_engine(args: argparse.Namespace) -> GRPOPluginInferenceEngine:
    return GRPOPluginInferenceEngine(
        Namespace(
            checkpoint=str(args.checkpoint),
            host_model=args.host_model,
            device=args.device,
            alpha=args.treatment_alpha,
            controller_temperature=1.0,
            reroute_temperature=1.0,
            reroute_top_p=0.95,
            reroute_max_new_tokens=16,
            delta_top_k=128,
            reroute_seed=args.seed,
            reroute_format_attempts=3,
            hint_slot_token_budgets=(16, 12, 12),
            hint_temperature=1.0,
            hint_top_p=0.95,
            hint_correction_topk=128,
            hint_query_max_words=24,
            max_plugin_length=4096,
            asag_attention_dtype="bfloat16",
            min_stop_rounds=15,
            min_reroute_rounds=8,
            reroute_cooldown_rounds=4,
            local_files_only=args.local_files_only,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host-model", default="simplex-ai-inc/LiteResearcher-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--qids", default="")
    parser.add_argument("--treatment-alpha", type=float, default=20.0)
    parser.add_argument("--control-alpha", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    records = load_records(args.source_dir)
    qids = [item.strip() for item in args.qids.split(",") if item.strip()]
    if not qids:
        qids = sorted(records, key=int)

    engine = build_engine(args)
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, qid in enumerate(qids, start=1):
        row = records[qid]
        prefix, source_trace = intervention_prefix(row)
        request = RerouteRequest(
            question=row["question"],
            history=render_history(prefix),
        )
        treatment = replay(engine, request, args.treatment_alpha)
        control = replay(engine, request, args.control_alpha)
        treatment_repeat = replay(engine, request, args.treatment_alpha)
        result = {
            "qid": row["qid"],
            "source_query": source_trace["query"],
            "source_alpha": source_trace.get("alpha"),
            "source_seed": source_trace.get("hint_seed"),
            "prefix_messages": len(prefix),
            "treatment_alpha": args.treatment_alpha,
            "treatment_query": treatment["query"],
            "treatment_hint": treatment["hint"],
            "control_alpha": args.control_alpha,
            "control_query": control["query"],
            "control_hint": control["hint"],
            "treatment_reproduces_source": treatment["query"] == source_trace["query"],
            "treatment_is_deterministic": treatment["query"]
            == treatment_repeat["query"],
            "query_changed_by_alpha": treatment["query"] != control["query"],
        }
        results.append(result)
        print(
            f"[{index}/{len(qids)}] qid={qid} "
            f"source_match={result['treatment_reproduces_source']} "
            f"deterministic={result['treatment_is_deterministic']} "
            f"alpha_changed_query={result['query_changed_by_alpha']}",
            flush=True,
        )

    with args.output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = {
        "pairs": len(results),
        "treatment_reproduces_source": sum(
            result["treatment_reproduces_source"] for result in results
        ),
        "treatment_is_deterministic": sum(
            result["treatment_is_deterministic"] for result in results
        ),
        "query_changed_by_alpha": sum(
            result["query_changed_by_alpha"] for result in results
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
