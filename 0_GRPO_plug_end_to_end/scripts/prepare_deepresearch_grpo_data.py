#!/usr/bin/env python3
"""Convert the local deep-research JSONL splits to verl RL parquet files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "deepresearch_data"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "deepresearch"
DEFAULT_PROMPT = ROOT / "prompt" / "deepresearch_system_prompt.txt"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def _target_list(answer) -> list[str]:
    if isinstance(answer, list):
        values = answer
    else:
        values = [answer]
    return [str(value).strip() for value in values if str(value).strip()]


def convert_split(input_path: Path, output_path: Path, system_prompt: str) -> int:
    converted = []
    seen_qids = set()
    for index, sample in enumerate(_read_jsonl(input_path)):
        qid = str(sample["qid"])
        if qid in seen_qids:
            raise ValueError(f"Duplicate qid={qid!r} in {input_path}")
        seen_qids.add(qid)

        question = str(sample["question"]).strip()
        targets = _target_list(sample["answer"])
        if not question or not targets:
            raise ValueError(f"qid={qid!r} has an empty question or answer")

        converted.append(
            {
                "task": "deepresearch",
                "id": qid,
                "question": question,
                "golden_answers": targets,
                "data_source": "deepresearch",
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": targets},
                },
                "agent_name": "deepresearch_agent",
                "ability": "deep_research",
                "extra_info": {
                    "id": qid,
                    "qid": qid,
                    "index": index,
                    "question": question,
                    "answer": targets,
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(converted).to_parquet(output_path, index=False)
    return len(converted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    args = parser.parse_args()

    system_prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ValueError(f"Prompt file is empty: {args.prompt_file}")
    # Match OpenResearcher/deploy_agent.py, which appends the current date to
    # DEVELOPER_CONTENT when each evaluation trajectory is created.
    system_prompt += f"\n\nToday's date: {datetime.now().strftime('%Y-%m-%d')}"

    split_map = {
        "train_dev": "train_dev.jsonl",
        "test": "test.jsonl",
    }
    for split, filename in split_map.items():
        source = args.data_dir / filename
        destination = args.output_dir / f"{split}.parquet"
        count = convert_split(source, destination, system_prompt)
        print(f"{split}: {count} rows -> {destination}")


if __name__ == "__main__":
    main()
