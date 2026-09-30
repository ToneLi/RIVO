#!/usr/bin/env python3
"""Compute per-prompt trajectory diversity stats for rollout JSONL files.

Metrics for each prompt group (G trajectories):
- query string similarity (pairwise SequenceMatcher ratio)
- query embedding similarity (pairwise cosine similarity)
- reward variance

Usage examples:
  python scripts/trajectory_group_stats.py \
    --input-glob "rollout_data_batch_dev_equ_train_fast/*.jsonl"

  python scripts/trajectory_group_stats.py \
    --input-glob "rollout_data_batch_dev_equ_train_fast/21.jsonl" \
    --embedding-backend sentence-transformers \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --output-csv /tmp/group_stats.csv
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


QUERY_RE = re.compile(r"<query>\s*(.*?)\s*</query>", flags=re.IGNORECASE | re.DOTALL)
SUMMARY_RE = re.compile(r"<summary>\s*(.*?)\s*</summary>", flags=re.IGNORECASE | re.DOTALL)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def extract_original_query(input_text: str) -> str:
    m = QUERY_RE.search(input_text or "")
    if not m:
        return ""
    return _clean_text(m.group(1))


def extract_last_summary(output_text: str) -> str:
    matches = SUMMARY_RE.findall(output_text or "")
    if not matches:
        return ""
    return _clean_text(matches[-1])


def select_query_text(input_text: str, output_text: str, query_source: str) -> str:
    original_query = extract_original_query(input_text)
    if query_source == "original":
        return original_query

    summary = extract_last_summary(output_text)
    if query_source == "summary":
        return summary

    # summary_or_original
    return summary if summary else original_query


def pairwise_string_similarity(texts: list[str]) -> tuple[float, float]:
    n = len(texts)
    if n < 2:
        return np.nan, np.nan

    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(SequenceMatcher(None, texts[i], texts[j]).ratio())
    arr = np.array(sims, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


@dataclass
class Embedder:
    backend: str
    model_name: str

    def __post_init__(self):
        self._model = None
        if self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        elif self.backend == "tfidf":
            self._model = None
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.backend == "sentence-transformers":
            emb = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            return emb.astype(np.float32)

        # tfidf fallback: lightweight semantic-ish baseline when embedding model unavailable
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        mat = vec.fit_transform(texts)
        dense = mat.toarray().astype(np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12
        return dense / norms


def pairwise_embedding_similarity(texts: list[str], embedder: Embedder) -> tuple[float, float]:
    n = len(texts)
    if n < 2:
        return np.nan, np.nan

    emb = embedder.embed(texts)
    sim = emb @ emb.T

    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(float(sim[i, j]))
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def group_key_from_input(input_text: str) -> str:
    # Stable compact key for a prompt. We also keep short preview in output.
    return hashlib.md5((input_text or "").encode("utf-8")).hexdigest()


def analyze_file(path: str, embedder: Embedder, query_source: str) -> pd.DataFrame:
    groups: dict[str, list[dict]] = {}

    for rec in iter_jsonl(path):
        input_text = rec.get("input", "")
        output_text = rec.get("output", "")
        score = float(rec.get("score", np.nan))

        qtxt = select_query_text(input_text, output_text, query_source=query_source)
        gkey = group_key_from_input(input_text)
        groups.setdefault(gkey, []).append(
            {
                "score": score,
                "query_text": qtxt,
                "input_preview": _clean_text(input_text)[:160],
            }
        )

    rows = []
    for gkey, items in groups.items():
        scores = np.array([x["score"] for x in items], dtype=float)
        qtexts = [x["query_text"] for x in items]

        str_mean, str_std = pairwise_string_similarity(qtexts)
        emb_mean, emb_std = pairwise_embedding_similarity(qtexts, embedder)

        nonempty = sum(1 for t in qtexts if t)
        uniq = len(set(qtexts))
        rows.append(
            {
                "file": path,
                "prompt_group": gkey,
                "group_size": len(items),
                "nonempty_query_count": nonempty,
                "unique_query_count": uniq,
                "unique_query_ratio": uniq / max(len(items), 1),
                "query_str_sim_mean": str_mean,
                "query_str_sim_std": str_std,
                "query_emb_sim_mean": emb_mean,
                "query_emb_sim_std": emb_std,
                "reward_mean": float(np.nanmean(scores)),
                "reward_var": float(np.nanvar(scores)),
                "reward_std": float(np.nanstd(scores)),
                "input_preview": items[0]["input_preview"],
            }
        )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "prompt_groups": len(df),
            "group_size_mean": float(df["group_size"].mean()),
            "query_str_sim_mean": float(df["query_str_sim_mean"].mean()),
            "query_emb_sim_mean": float(df["query_emb_sim_mean"].mean()),
            "reward_var_mean": float(df["reward_var"].mean()),
            "reward_std_mean": float(df["reward_std"].mean()),
            "unique_query_ratio_mean": float(df["unique_query_ratio"].mean()),
        }
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-glob", required=True, help="Glob for rollout jsonl files")
    p.add_argument(
        "--query-source",
        default="summary_or_original",
        choices=["summary", "original", "summary_or_original"],
        help="Which query text to compare among trajectories",
    )
    p.add_argument(
        "--embedding-backend",
        default="tfidf",
        choices=["tfidf", "sentence-transformers"],
        help="Embedding backend for query embedding similarity",
    )
    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Used when --embedding-backend sentence-transformers",
    )
    p.add_argument("--output-csv", default="", help="Optional output csv path")
    p.add_argument("--topk", type=int, default=10, help="Print top-k most similar groups")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.input_glob}")

    embedder = Embedder(backend=args.embedding_backend, model_name=args.embedding_model)

    all_df = []
    for path in paths:
        df = analyze_file(path, embedder=embedder, query_source=args.query_source)
        if df.empty:
            continue
        all_df.append(df)

    if not all_df:
        raise SystemExit("No valid groups found in input files")

    out = pd.concat(all_df, ignore_index=True)

    print("=== Overall Summary ===")
    print(summarize(out).to_string())

    print("\n=== Most Similar Query Groups (by embedding) ===")
    top = out.sort_values("query_emb_sim_mean", ascending=False).head(args.topk)
    cols = [
        "file",
        "group_size",
        "query_emb_sim_mean",
        "query_str_sim_mean",
        "reward_var",
        "unique_query_ratio",
    ]
    print(top[cols].to_string(index=False))

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
