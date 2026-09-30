#!/usr/bin/env python
"""Summarize two full-agent ASAG runs over the same QID set."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import Counter
from pathlib import Path


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def rounded(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def final_answer(record: dict) -> str:
    for message in reversed(record.get("messages", [])):
        if message.get("role") == "assistant":
            return " ".join(str(message.get("content") or "").split()).lower()
    return ""


def load_run(path: Path) -> tuple[dict[str, dict], dict]:
    records: dict[str, dict] = {}
    for shard in glob.glob(str(path / "node_*_shard_*.jsonl")):
        with open(shard, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                records[str(row["qid"])] = row

    rows = list(records.values())
    traces = [trace for row in rows for trace in row.get("asag_trace", [])]
    high_confidence = [t for t in traces if float(t.get("confidence", 0.0)) > 0.95]
    converged = [
        t
        for t in traces
        if isinstance(t.get("entropy_delta"), (int, float))
        and float(t["entropy_delta"]) < -0.10
    ]
    stop_eligible = [t for t in high_confidence if t in converged]
    first_checkpoint_high_confidence = sum(
        bool(row.get("asag_trace"))
        and float(row["asag_trace"][0].get("confidence", 0.0)) > 0.95
        for row in rows
    )
    latencies = [float(row["latency_s"]) for row in rows if row.get("latency_s")]
    tool_rounds = [
        sum(message.get("role") == "tool" for message in row.get("messages", []))
        for row in rows
    ]
    wall_path = path / "wall_seconds.txt"
    wall_seconds = float(wall_path.read_text().strip()) if wall_path.exists() else None

    def trace_values(key: str) -> list[float]:
        return [float(t[key]) for t in traces if isinstance(t.get(key), (int, float))]

    summary = {
        "questions": len(rows),
        "success": sum(row.get("status") == "success" for row in rows),
        "wall_seconds": rounded(wall_seconds),
        "questions_per_hour": rounded(
            len(rows) * 3600.0 / wall_seconds if wall_seconds else None
        ),
        "mean_question_latency_s": rounded(mean(latencies)),
        "median_question_latency_s": rounded(
            statistics.median(latencies) if latencies else None
        ),
        "mean_tool_rounds": rounded(mean(tool_rounds)),
        "asag_checkpoints": len(traces),
        "decisions": dict(Counter(t.get("decision") for t in traces)),
        "decision_reasons": dict(Counter(t.get("reason") for t in traces)),
        "first_checkpoint_high_confidence": first_checkpoint_high_confidence,
        "later_high_confidence_checkpoints": len(high_confidence),
        "entropy_converged_checkpoints": len(converged),
        "later_stop_eligible_checkpoints": len(stop_eligible),
        "mean_attention_total_s": rounded(mean(trace_values("attention_probe_s"))),
        "mean_attention_compute_s": rounded(
            mean(trace_values("attention_service_compute_s"))
        ),
        "mean_attention_lock_wait_s": rounded(
            mean(trace_values("attention_service_lock_wait_s"))
        ),
        "mean_prefill_tokens": rounded(mean(trace_values("attention_prefill_tokens"))),
        "mean_reused_prefix_tokens": rounded(
            mean(trace_values("attention_reused_prefix_tokens"))
        ),
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_rows, baseline = load_run(args.baseline)
    optimized_rows, optimized = load_run(args.optimized)
    shared = sorted(set(baseline_rows) & set(optimized_rows))
    exact_matches = sum(
        final_answer(baseline_rows[qid]) == final_answer(optimized_rows[qid])
        for qid in shared
    )
    report = {
        "baseline": baseline,
        "optimized": optimized,
        "comparison": {
            "shared_qids": len(shared),
            "exact_final_answer_matches": exact_matches,
            "exact_final_answer_rate": rounded(exact_matches / len(shared) if shared else None),
            "wall_speedup": rounded(
                baseline["wall_seconds"] / optimized["wall_seconds"]
                if baseline["wall_seconds"] and optimized["wall_seconds"]
                else None
            ),
            "attention_speedup": rounded(
                baseline["mean_attention_total_s"] / optimized["mean_attention_total_s"]
                if baseline["mean_attention_total_s"]
                and optimized["mean_attention_total_s"]
                else None
            ),
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
