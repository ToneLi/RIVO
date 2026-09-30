"""LLM-as-judge outcome reward for the deep-research GRPO baseline.

The grading prompt, parser, and judge behavior are loaded directly from
OpenResearcher/eval.py so training and offline evaluation use the same judge.
"""

from __future__ import annotations

import importlib.util
import os
import re
import threading
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_EVAL_PATH = None
_LOCK = threading.Lock()
_EVAL_MODULE: ModuleType | None = None
_JUDGE: Any | None = None


def _load_eval_module() -> ModuleType:
    global _EVAL_MODULE
    if _EVAL_MODULE is not None:
        return _EVAL_MODULE

    configured_path = os.environ.get("DEEPRESEARCH_EVAL_PATH")
    if not configured_path:
        raise FileNotFoundError("Set DEEPRESEARCH_EVAL_PATH to the external OpenResearcher evaluator.")
    eval_path = Path(configured_path)
    if not eval_path.is_file():
        raise FileNotFoundError(f"OpenResearcher evaluator not found: {eval_path}")
    spec = importlib.util.spec_from_file_location("openresearcher_eval_for_grpo", eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load OpenResearcher evaluator: {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _EVAL_MODULE = module
    return module


def _get_judge():
    global _JUDGE
    if _JUDGE is not None:
        return _JUDGE

    with _LOCK:
        if _JUDGE is not None:
            return _JUDGE
        module = _load_eval_module()
        judge = module.LLMJudge(
            llm=os.environ.get("LLM_JUDGE_MODEL", "gpt-4.1-2025-04-14"),
            qps=float(os.environ.get("LLM_JUDGE_QPS", "50")),
            max_retries=int(os.environ.get("LLM_JUDGE_MAX_RETRIES", "5")),
        )

        # eval.py defaults to the public OpenAI endpoint. This optional override
        # also permits an OpenAI-compatible local judge without changing eval.py.
        base_url = os.environ.get("LLM_JUDGE_BASE_URL")
        if base_url:
            api_key = os.environ.get("LLM_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
            judge.client = module.openai.Client(api_key=api_key, base_url=base_url.rstrip("/"))

        _JUDGE = judge
        return judge


def _final_response(solution: str) -> str:
    """Match eval.py's final-message input while avoiding full tool transcripts."""
    matches = re.findall(r"<answer>.*?</answer>", solution, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    # Missing answer tags should normally be judged incorrect, but retain a
    # bounded fallback so eval.py can apply its own final-answer extraction.
    return solution[-8000:].strip()


def _correct_answer(ground_truth: Any) -> Any:
    if isinstance(ground_truth, dict):
        targets = ground_truth.get("target", [])
    else:
        targets = ground_truth
    if hasattr(targets, "tolist"):
        targets = targets.tolist()
    if isinstance(targets, (list, tuple)):
        return targets[0] if len(targets) == 1 else list(targets)
    return targets


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    data_source: str | None = None,
    **kwargs,
) -> dict:
    """Return a binary, trajectory-level correctness reward from eval.py."""
    del data_source, kwargs
    info = extra_info or {}
    result = _get_judge()._judge(
        {
            "qid": str(info.get("qid", info.get("id", ""))),
            "question": str(info.get("question", "")),
            "answer": _correct_answer(ground_truth),
            "messages": [{"role": "assistant", "content": _final_response(str(solution_str))}],
        }
    )
    correct = result.get("correct") is True and not result.get("parse_error", False)
    output = {
        "score": float(correct),
        "acc": float(correct),
        "judge_parse_error": bool(result.get("parse_error", False)),
    }
    if result.get("extracted_final_answer") is not None:
        output["judge_extracted_answer"] = result["extracted_final_answer"]
    if result.get("error"):
        output["judge_error"] = result["error"]
    return output
