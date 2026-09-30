"""Deterministic semantic reward for generated three-field retrieval hints."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any


_TOKEN_RE = re.compile(r"[^\W_]+(?:[._'/-][^\W_]+)*", re.UNICODE)
_EAST_ASIAN_RE = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_FIRST_PERSON_RE = re.compile(
    r"\b(?:i|i['’]m|i['’]ve|i['’]ll|me|my|mine|we|we['’]re|us|our|ours)\b",
    re.IGNORECASE,
)
_CLAUSE_RE = re.compile(
    r"^(?:so|then|there|here|this|that|it|he|she|they|trying|seeing|finding|"
    r"searching|answering|extracting)\b",
    re.IGNORECASE,
)
_META_MARKERS = (
    "we need", "i need", "need to answer", "to answer the question",
    "the user", "user asks", "user question", "the question", "question is",
    "let us analyze", "let's analyze", "first we", "so i", "i'm trying",
    "i am trying", "trying to", "i see", "i found", "let me", "it looks like",
    "looks like", "it seems", "there is no", "there's no", "no content",
    "record the user", "record the internal", "tool call", "tool usage",
    "search tool", "我们需要", "我需要", "需要回答", "必须回答", "用户提问",
    "用户问题", "问题是", "当前问题", "这个问题", "让我们", "所以我们",
    "看起来", "似乎", "工具调用", "工具使用", "搜索工具",
)
_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
        "from", "in", "is", "it", "of", "on", "or", "that", "the", "this",
        "to", "was", "were", "with",
    }
)
_GENERIC_ENTITY_WORDS = _STOP_WORDS | {
    "answer", "article", "content", "data", "date", "document", "entity",
    "extracting", "fact", "figure", "find", "how", "information", "number",
    "page", "question", "record", "report", "reported", "result", "search",
    "source", "topic", "value", "volume", "what", "when", "where", "which",
    "who", "why", "do", "does", "did",
}
_COMPONENT_WEIGHTS = {
    "format": 0.10,
    "english": 0.10,
    "no_self_talk": 0.20,
    "entity_anchor": 0.25,
    "relation": 0.15,
    "disambiguator": 0.10,
    "distinct": 0.10,
}


def _tokens(value: str) -> list[str]:
    return _TOKEN_RE.findall(value)


def _content_tokens(value: str) -> set[str]:
    return {token.casefold() for token in _tokens(value) if token.casefold() not in _STOP_WORDS}


def _has_meta_language(value: str) -> bool:
    normalized = " ".join(value.casefold().split())
    return _FIRST_PERSON_RE.search(value) is not None or any(marker in normalized for marker in _META_MARKERS)


def _duplicated(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    overlap = len(left & right) / len(left | right)
    return left == right or (min(len(left), len(right)) >= 2 and overlap >= 0.75)


def score_hint_fields(
    fields: Sequence[str],
    question: str,
    history: str,
    *,
    max_field_words: int = 8,
) -> dict[str, Any]:
    """Return a dense quality reward in [-1, 1] without rejecting rollout samples."""
    normalized = [" ".join(str(field).strip().split()) for field in fields]
    format_ok = (
        len(normalized) == 3
        and all(normalized)
        and all(len(_tokens(field)) <= max_field_words for field in normalized)
        and all(not any(char in field for char in ";\r\n{}[]<>") for field in normalized)
    )
    if len(normalized) != 3:
        components = {name: False for name in _COMPONENT_WEIGHTS}
    else:
        token_sets = [_content_tokens(field) for field in normalized]
        context_tokens = {
            token.casefold()
            for token in _tokens(question + "\n" + history)
            if token.casefold() not in _GENERIC_ENTITY_WORDS
        }
        entity_tokens = token_sets[0] - _GENERIC_ENTITY_WORDS
        components = {
            "format": format_ok,
            "english": not any(_EAST_ASIAN_RE.search(field) for field in normalized),
            "no_self_talk": not any(_has_meta_language(field) for field in normalized),
            "entity_anchor": bool(entity_tokens & context_tokens),
            "relation": bool(token_sets[1] - token_sets[0])
            and _CLAUSE_RE.search(normalized[1]) is None
            and re.search(r"[.!?]$", normalized[1]) is None,
            "disambiguator": bool(token_sets[2] - token_sets[0] - token_sets[1]),
            "distinct": not any(
                _duplicated(token_sets[left], token_sets[right])
                for left in range(3)
                for right in range(left + 1, 3)
            ),
        }

    component_score = sum(
        _COMPONENT_WEIGHTS[name] for name, passed in components.items() if passed
    )
    valid = all(components.values())
    # Every invalid triple receives a negative reward, while the component score
    # still ranks bad samples so GRPO has a useful within-group learning signal.
    reward = 1.0 if valid else -1.0 + 0.5 * component_score
    return {
        "reward": float(reward),
        "valid": valid,
        "component_score": float(component_score),
        "components": components,
        "violations": [name for name, passed in components.items() if not passed],
    }
