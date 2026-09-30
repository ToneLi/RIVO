#!/usr/bin/env python
"""Select a deterministic BrowseComp+ QID subset for ASAG A/B runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_utils import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")

    rows = load_dataset("browsecomp-plus", data_path=args.data_path)
    qids = [str(row["qid"]) for row in rows[: args.count]]
    if len(qids) != args.count:
        raise ValueError(f"requested {args.count} QIDs but dataset has {len(rows)} rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{qid}\n" for qid in qids), encoding="utf-8")
    print(f"Wrote {len(qids)} deterministic QIDs to {args.output}")


if __name__ == "__main__":
    main()
