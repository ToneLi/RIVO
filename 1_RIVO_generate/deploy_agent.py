# -*- coding: utf-8 -*-
import duckdb
import argparse
import base64
import hashlib
import os, json, time, argparse, math, traceback, glob, fcntl
from typing import List, Dict, Any
import multiprocessing as mp
from pathlib import Path
import asyncio
import datetime
import runpy
import uuid
import tqdm
from openai_harmony import (
    Message, Conversation, SystemContent, Role, ReasoningEffort,
    StreamableParser, load_harmony_encoding, HarmonyEncodingName
)
from browser import BrowserTool, LocalServiceBrowserBackend, SerperServiceBrowserBackend
from data_utils import load_dataset, list_available_datasets, DEVELOPER_CONTENT, TOOL_CONTENT
import dotenv
import json5
import re
from asag_controller import (
    ASAG_ANSWER_PROBE_PROMPT,
    ResearchSegment,
    RetrievalBoundaryASAG,
    boxed_answer_logprobs,
    boxed_answer_prefix,
    token_confidence,
)
from utils.grpo_plugin_client import GRPOPluginClient

dotenv.load_dotenv()

os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

# Pre-import transformers in main process to avoid multiprocessing issues
try:
    import transformers
    print(f"Pre-loaded transformers version: {transformers.__version__}")
except ImportError:
    print("Warning: transformers not available")

def _extract_text_from_harmony(messages: List[Message]) -> str:
    """Extract text content from Harmony messages.

    Handles GPT OSS format where content is a list:
    [{"type": "text", "text": "..."}]
    """
    text_parts = []
    for msg in messages:
        if hasattr(msg, 'content') and isinstance(msg.content, list):
            for item in msg.content:
                if hasattr(item, 'text'):
                    text_parts.append(item.text)
                elif isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])

    return '\n'.join(text_parts) if text_parts else ""


TRUNCATION_NOTICE = "\n\n[... middle of tool output truncated to keep the request within the model context window ...]\n\n"
HISTORY_TRUNCATION_NOTICE = (
    "[... older assistant/tool interaction omitted to keep the request within "
    "the model context window ...]"
)

GRPO_ROUTE_HINT_INSTRUCTION = (
    "The current research trajectory is stalled. Generate exactly three short "
    "search-direction fields in this order: ENTITY ; RELATION ; DISAMBIGUATOR. "
    "The assistant response is already prefixed with 'Search direction:'. "
    "Output only the three field values after that prefix, separated by exactly "
    "two semicolons. Do not output labels, a sentence, a preface, quotes, JSON, "
    "XML, a tool call, an explanation, or a repetition of the full question."
)
GRPO_ROUTE_HINT_PREFILL = "Search direction: "
GRPO_HOST_QUERY_INSTRUCTION = (
    "The current research trajectory is stalled. A small retrieval controller "
    "suggested this direction:\nSearch direction: {route_hint}\n"
    "Use the direction only as guidance. Generate one concise English "
    "browser.search query that targets the unresolved evidence gap and avoids "
    "failed queries. Use only keywords and named entities. Do not include "
    "Chinese, labels, explanations, full sentences, or tool-error text. Keep "
    "the query under 16 words. Issue the browser.search call without answering "
    "or explaining."
)
GRPO_QUERY_SENTINEL = "__GRPO_NATIVE_QUERY_START_7F3A9C__"
_CJK_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\u3040-\u30ff\uac00-\ud7af]"
)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HOST_QUERY_FORBIDDEN_RE = re.compile(
    r"\b("
    r"we need|the user|current question|the question|question is|"
    r"record(?:ed)? (?:the )?(?:user|error)|tool (?:call|error|failed)|"
    r"error message|looks like|i need|i should|let'?s|first(?:ly)?|namely"
    r")\b",
    re.IGNORECASE,
)


def _validated_route_hint(value: str, question: str) -> str:
    """Validate and normalize an ENTITY ; RELATION ; DISAMBIGUATOR hint."""
    if not isinstance(value, str):
        raise ValueError("GRPO plugin returned a non-string route hint")
    hint = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL | re.IGNORECASE)
    hint = re.sub(r"</?[^>]+>", " ", hint)
    hint = _CONTROL_CHAR_RE.sub(" ", hint).strip()
    if not hint:
        raise ValueError("GRPO plugin returned an empty route hint")
    hint = re.sub(r"^\s*search\s+direction\s*:\s*", "", hint, flags=re.I)
    hint = re.sub(r"[\r\n]+", " ; ", hint)
    hint = re.sub(r"[\"'{}[\]()]", " ", hint)
    hint = re.sub(
        r"\b(entity|relation|disambiguator)\b\s*:\s*",
        "",
        hint,
        flags=re.I,
    )
    hint = " ".join(hint.split())
    if hint.lower().startswith(("the user", "we need", "the question")):
        raise ValueError("Route hint contains a forbidden preface or label")
    parts = [part.strip() for part in hint.split(";")]
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(
            "Route hint must contain exactly three non-empty semicolon-separated fields"
        )
    return " ; ".join(" ".join(part.split()) for part in parts)


def _build_prompt_and_tokens(messages: List[dict], tools: List[dict], tokenizer: Any) -> tuple[str, List[int]]:
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return prompt, tokens


def _truncate_text_tokens(tokenizer: Any, text: str, max_tokens: int) -> str:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text

    if max_tokens <= 1:
        return TRUNCATION_NOTICE.strip()

    head_tokens = max_tokens // 2
    tail_tokens = max_tokens - head_tokens
    head = tokenizer.decode(tokens[:head_tokens], skip_special_tokens=False).rstrip()
    tail = tokenizer.decode(tokens[-tail_tokens:], skip_special_tokens=False).lstrip()
    return head + TRUNCATION_NOTICE + tail


def _fit_messages_to_input_budget(
    messages: List[dict],
    tools: List[dict],
    tokenizer: Any,
    max_input_tokens: int,
    return_metadata: bool = False,
) -> Any:
    # Keep tool responses stable from their first appearance.  The previous
    # overflow-driven truncation repeatedly shortened old responses as history
    # grew, invalidating both vLLM prefix cache entries and the stateful ASAG KV
    # cache at almost every checkpoint.
    fitted_messages = [dict(message) for message in messages]
    fitted_original_indices: List[int | None] = list(range(len(messages)))
    min_tool_tokens = int(os.environ.get("OPENRESEARCHER_MIN_TOOL_TOKENS", "256"))
    configured_tool_tokens = int(
        os.environ.get("OPENRESEARCHER_TOOL_RESPONSE_TOKENS", "2048")
    )
    fixed_tool_tokens = min(
        configured_tool_tokens,
        max(min_tool_tokens, max_input_tokens // 2),
    )
    for message in fitted_messages:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if len(tokenizer.encode(content, add_special_tokens=False)) > fixed_tool_tokens:
            message["content"] = _truncate_text_tokens(
                tokenizer, content, fixed_tool_tokens
            )

    prompt, tokens = _build_prompt_and_tokens(fitted_messages, tools, tokenizer)
    if len(tokens) <= max_input_tokens:
        if return_metadata:
            return prompt, tokens, {
                "messages": fitted_messages,
                "original_indices": fitted_original_indices,
            }
        return prompt, tokens

    # Rebuild the rolling window in coarse epochs.  At the first overflow we
    # remove enough old turns to leave headroom; the selected suffix then stays
    # byte-for-byte stable until that headroom is consumed.  This is much more
    # prefix-cache-friendly than adjusting old messages on every request.
    min_recent_messages = int(os.environ.get("OPENRESEARCHER_MIN_RECENT_MESSAGES", "8"))
    protected_prefix = 2
    rebuild_target = int(
        os.environ.get(
            "OPENRESEARCHER_CONTEXT_REBUILD_TOKENS",
            str(max(min_tool_tokens * 4, int(max_input_tokens * 0.70))),
        )
    )
    rebuild_target = min(rebuild_target, max_input_tokens - 1)
    headroom = max(max_input_tokens - rebuild_target, 1)
    overflow = len(tokens) - max_input_tokens
    rebuild_epoch = max(1, math.ceil(overflow / headroom))
    desired_length = max(rebuild_target, len(tokens) - rebuild_epoch * headroom)
    original_fitted_messages = fitted_messages
    original_fitted_indices = fitted_original_indices
    last_suffix_start = len(fitted_messages) - min_recent_messages
    suffix_start = protected_prefix
    while suffix_start <= last_suffix_start:
        while (
            suffix_start <= last_suffix_start
            and original_fitted_messages[suffix_start].get("role") == "tool"
        ):
            suffix_start += 1
        candidate_messages = (
            original_fitted_messages[:protected_prefix]
            + [{"role": "assistant", "content": HISTORY_TRUNCATION_NOTICE}]
            + original_fitted_messages[suffix_start:]
        )
        candidate_indices = (
            original_fitted_indices[:protected_prefix]
            + [None]
            + original_fitted_indices[suffix_start:]
        )
        candidate_prompt, candidate_tokens = _build_prompt_and_tokens(
            candidate_messages, tools, tokenizer
        )
        fitted_messages = candidate_messages
        fitted_original_indices = candidate_indices
        prompt, tokens = candidate_prompt, candidate_tokens
        if len(tokens) <= desired_length:
            break
        suffix_start += 1

    if len(tokens) > max_input_tokens:
        raise ValueError(
            f"Prompt still has {len(tokens)} input tokens after tool-response/history truncation; "
            f"budget is {max_input_tokens}. Try lowering OPENRESEARCHER_MAX_INPUT_TOKENS "
            "or OPENRESEARCHER_MIN_RECENT_MESSAGES."
        )

    if return_metadata:
        return prompt, tokens, {
            "messages": fitted_messages,
            "original_indices": fitted_original_indices,
        }
    return prompt, tokens


def _serialized_message_token_span(
    metadata: dict,
    original_span: dict[str, int] | None,
    tools: List[dict],
    tokenizer: Any,
    history_token_count: int,
) -> dict[str, int] | None:
    """Map an original message interval into the fitted prompt token space."""
    if original_span is None:
        return None
    start_original = int(original_span["start"])
    end_original = int(original_span["end"])
    positions = [
        position
        for position, original_index in enumerate(metadata["original_indices"])
        if original_index is not None
        and start_original <= original_index < end_original
    ]
    if not positions:
        return None
    start_message = min(positions)
    end_message = max(positions) + 1

    def prefix_length(message_count: int) -> int:
        if message_count <= 0:
            return 0
        text = tokenizer.apply_chat_template(
            metadata["messages"][:message_count],
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        return len(tokenizer.encode(text, add_special_tokens=False))

    start = min(prefix_length(start_message), history_token_count)
    end = min(prefix_length(end_message), history_token_count)
    return {"start": start, "end": end} if end > start else None

class BrowserPool:
    def __init__(self, search_url, browser_backend='local'):
        self.search_url = search_url
        self.browser_backend = browser_backend
        self.sessions: Dict[Any, BrowserTool] = {}

    def init_session(self, qid: Any) -> dict:
        if self.browser_backend == 'serper':
            backend = SerperServiceBrowserBackend()
        else:
            backend = LocalServiceBrowserBackend(base_url=self.search_url)
        tool = BrowserTool(backend=backend)
        self.sessions[qid] = tool
        return tool.tool_config

    async def call_tool(self, qid: Any, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Call browser tool and return text result."""
        tool = self.sessions[qid]

        # Map tool names to browser recipients
        recipient_map = {
            'search': 'browser.search',
            'find': 'browser.find',
            'open': 'browser.open'
        }

        recipient = recipient_map.get(tool_name.lower())
        if not recipient:
            return f"Unknown browser tool: {tool_name}"

        # Create Harmony message with JSON args in content (matching GPT OSS format)
        import json
        from openai_harmony import TextContent
        args_json = json.dumps(tool_args, ensure_ascii=False)

        tool_msg = Message.from_role_and_content(Role.ASSISTANT, TextContent(text=args_json))
        tool_msg.recipient = recipient

        # Execute tool and collect results
        results = []
        async for msg in tool.process(tool_msg):
            results.append(msg)

        # Extract and return text
        return _extract_text_from_harmony(results)

    def cleanup(self, qid: Any):
        if qid in self.sessions:
            del self.sessions[qid]


def _parse_call_tool_xml(content: str, round_num: int):
    """Parse Dr.tulu-style <call_tool name="...">...</call_tool> output."""
    match = re.search(
        r'<call_tool\s+name=["\']([^"\']+)["\']([^>]*)>(.*?)</call_tool>',
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return content, None, None

    raw_name = match.group(1).strip()
    attributes_text = match.group(2)
    body = match.group(3).strip()
    aliases = {
        "google_search": "search",
        "browse_webpage": "open",
        "snippet_search": "search",
    }
    tool_name = aliases.get(raw_name.lower(), raw_name)
    if tool_name.startswith("browser."):
        tool_name = tool_name.split(".", 1)[1]
    tool_name = tool_name.lower()
    if tool_name not in {"search", "open", "find"}:
        return content, None, None

    tool_args = {}
    for attribute in re.finditer(
        r'([\w_]+)\s*=\s*(["\'])(.*?)\2',
        attributes_text,
        re.DOTALL,
    ):
        key = attribute.group(1)
        value = attribute.group(3).strip()
        if key in {"topn", "cursor", "id", "loc", "num_lines"} and value.lstrip("-").isdigit():
            value = int(value)
        elif key == "view_source" and value.lower() in {"true", "false"}:
            value = value.lower() == "true"
        tool_args[key] = value

    if tool_name == "search" and body:
        tool_args.setdefault("query", body)
    elif tool_name == "find" and body:
        tool_args.setdefault("pattern", body)
    elif tool_name == "open" and body and "id" not in tool_args:
        tool_args["id"] = int(body) if body.lstrip("-").isdigit() else body

    required_key = {"search": "query", "find": "pattern"}.get(tool_name)
    if required_key and not tool_args.get(required_key):
        return content, None, None

    parsed_tool_calls = [{
        "id": f"{round_num}",
        "type": "function",
        "function": {
            "name": f"browser.{tool_name}",
            "arguments": tool_args,
        },
    }]
    cleaned_content = content.replace(match.group(0), "").strip()
    return cleaned_content, match.group(0), parsed_tool_calls


def _is_terminal_assistant_response(content: str, parsed_tool_calls: Any) -> bool:
    """A tool-calling turn ends when it emits content instead of an action."""
    return not parsed_tool_calls and bool((content or "").strip())


async def _generate_with_retry(
    generator: Any,
    tokens: List[int],
    stop_strings: List[str],
    max_retries: int = 20,
    verbose: bool = False,
    seed: int | None = None,
) -> str:
    """
    HuggingFace-based generation (interface matches _generate_with_retry)
    Args:
        generator: Generator (vLLMAsyncGenerator or OpenAIAsyncGenerator) with tokenizer
        tokens: Pre-tokenized input
        stop_strings: Stop strings (e.g., ["\n<tool_response>", "<tool_response>"])
        max_retries: Max retry attempts
    Returns:
        Generated text string
    """
    assert max_retries > 0
    last_exception = None

    # Retry only the generation part
    for attempt in range(1, max_retries + 1):
        stream = generator.generate(tokens, stop_strings=stop_strings, seed=seed)
        try:
            # Generate and collect tokens with client-side stop checking
            generated_tokens = []
            accumulated_text = ""

            async for token_id in stream:
                generated_tokens.append(token_id)

                # Periodically check for stop strings (every 10 tokens)
                if len(generated_tokens) % 10 == 0:
                    accumulated_text = generator.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    # Check if we hit any stop string
                    for stop_str in stop_strings:
                        if stop_str in accumulated_text:
                            if verbose:
                                print(f"[DEBUG] Client-side stop detected: found '{stop_str}' in generated text")
                            break
                    else:
                        continue
                    break

            # Final decode
            generated_text = generator.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Remove any stop strings from the end
            for stop_str in stop_strings:
                if stop_str in generated_text:
                    pos = generated_text.find(stop_str)
                    generated_text = generated_text[:pos]

            if verbose:
                print(f"[DEBUG] Generated {len(generated_tokens)} tokens, text length: {len(generated_text)}")
            return generated_text

        except Exception as e:
            last_exception = e
            if verbose:
                print(f"\n--- Generation failed on attempt {attempt}/{max_retries} ---")
                import traceback as _tb
                print(_tb.format_exc())

        finally:
            try:
                await stream.aclose()
            except Exception:
                pass

    if last_exception:
        raise last_exception
    raise RuntimeError("Generation failed after retries without a captured exception.")


async def _run_asag_answer_probe(
    messages: List[dict],
    tools: List[dict],
    generator: Any,
    previous_message_span: dict[str, int] | None,
    current_message_span: dict[str, int],
) -> tuple[str, float, int, dict]:
    """Probe the current answer and compute ASAG Equation (5) confidence."""
    probe_suffix_tokens = generator.tokenizer.encode(
        ASAG_ANSWER_PROBE_PROMPT,
        add_special_tokens=False,
    )
    max_input_tokens = int(os.environ.get("OPENRESEARCHER_MAX_INPUT_TOKENS", "23000"))
    _, history_tokens, fit_metadata = _fit_messages_to_input_budget(
        messages,
        tools,
        generator.tokenizer,
        max_input_tokens=max_input_tokens - len(probe_suffix_tokens),
        return_metadata=True,
    )
    previous_token_span = _serialized_message_token_span(
        fit_metadata,
        previous_message_span,
        tools,
        generator.tokenizer,
        len(history_tokens),
    )
    current_token_span = _serialized_message_token_span(
        fit_metadata,
        current_message_span,
        tools,
        generator.tokenizer,
        len(history_tokens),
    )
    if current_token_span is None:
        raise RuntimeError("current retrieval message was lost while fitting ASAG input")
    # ASAG probes x + I as a raw continuation. Adding a new chat turn would
    # make Qwen's template insert another <think> block and contaminate C with
    # reasoning-token probabilities rather than answer-token probabilities.
    prompt_tokens = history_tokens + probe_suffix_tokens
    completion = await generator.complete_with_logprobs(
        prompt_tokens,
        max_tokens=int(os.environ.get("ASAG_ANSWER_PROBE_MAX_TOKENS", "32")),
        temperature=0.0,
        stop_strings=["<tool_call>", "</think>"],
    )
    answer_suffix, _ = boxed_answer_prefix(completion["text"])
    if not answer_suffix:
        raise RuntimeError("ASAG answer probe returned an empty provisional answer")
    if answer_suffix.startswith("{"):
        provisional_answer = "\\boxed" + answer_suffix
    elif answer_suffix.startswith("\\boxed"):
        provisional_answer = answer_suffix
    else:
        provisional_answer = answer_suffix
    answer_logprobs = boxed_answer_logprobs(
        completion.get("tokens", []),
        completion["token_logprobs"],
    )
    confidence = token_confidence(answer_logprobs)
    attention_context = {
        "history_token_ids": history_tokens,
        "probe_token_ids": probe_suffix_tokens,
        "previous_span": previous_token_span,
        "current_span": current_token_span,
    }
    return provisional_answer, confidence, len(answer_logprobs), attention_context


def _research_query_text(tool_name: str, tool_args: Dict[str, Any]) -> str:
    for key in ("query", "pattern", "url", "id"):
        if key in tool_args:
            return str(tool_args[key])
    return json.dumps(tool_args, ensure_ascii=False, sort_keys=True)


def _asag_final_message(provisional_answer: str) -> dict:
    answer = provisional_answer.strip()
    if "<answer>" not in answer.lower():
        answer = f"<answer>{answer}</answer>"
    return {"role": "assistant", "content": answer, "reasoning_content": None}


ASAG_FORCE_ANSWER_PROMPT = (
    "The research budget is exhausted. Do not call any tool. Synthesize the "
    "best-supported final answer from the evidence already in this conversation. "
    "Return only <answer>...</answer>."
)


async def _append_forced_final_answer(
    messages: List[dict],
    generator: Any,
    fallback_answer: str,
    verbose: bool,
) -> None:
    """Run one final evidence synthesis with tool definitions removed."""
    messages.append({"role": "user", "content": ASAG_FORCE_ANSWER_PROMPT})
    max_input_tokens = int(os.environ.get("OPENRESEARCHER_MAX_INPUT_TOKENS", "23000"))
    _, tokens = _fit_messages_to_input_budget(
        messages,
        [],
        generator.tokenizer,
        max_input_tokens=max_input_tokens,
    )
    continuation_seed = os.environ.get("CONTINUATION_SEED")
    forced_seed = int(continuation_seed) + 900000 if continuation_seed else None
    generated = await _generate_with_retry(
        generator, tokens, [], verbose=verbose, seed=forced_seed
    )
    generated = re.sub(r"<think>.*?</think>", "", generated, flags=re.DOTALL).strip()
    generated = generated.split("<tool_call>", 1)[0].strip()
    if not generated:
        generated = fallback_answer
    messages.append(_asag_final_message(generated))


def _browser_tool_name(function_name: str) -> str:
    """Return the unqualified browser action name used by the executor."""
    name = (function_name or "").strip().lower()
    if name.startswith("browser."):
        name = name.split(".", 1)[1]
    return name


def _retrieval_result_is_error(result: str) -> bool:
    value = (result or "").strip().lower()
    return (
        not value
        or value.startswith("error executing ")
        or value.startswith("error during ")
        or value.startswith("unknown browser tool:")
        or value.startswith("tool ") and value.endswith(" not available")
    )


def _extend_round_limit_after_reroute(
    *,
    base_max_rounds: int,
    current_round_limit: int,
    reroute_round: int,
    min_post_reroute_rounds: int,
    max_extra_rounds: int,
) -> int:
    """Reserve completion rounds after a successful corrected-logits search."""
    desired_limit = reroute_round + min_post_reroute_rounds
    hard_limit = base_max_rounds + max_extra_rounds
    return min(hard_limit, max(current_round_limit, desired_limit))


def _should_flush_pending_checkpoint(pending_tool: str, next_tool: str | None) -> bool:
    """End one retrieval chain before an unrelated or new search action.

    Search results are provisional: an immediately following open/find upgrades
    the same chain.  Open results are likewise held for a following open/find.
    A new search starts a new chain, so the best result from the previous chain
    must be evaluated first.
    """
    pending = _browser_tool_name(pending_tool)
    following = _browser_tool_name(next_tool or "")
    if pending == "search":
        return following not in {"open", "find"}
    if pending == "open":
        return following not in {"open", "find"}
    return True


def _native_tool_call_prefix(
    tools: List[dict], tokenizer: Any, *, tool_name: str, argument_name: str
) -> str:
    """Derive the model-native tool-call prefix from its chat template."""
    probe_messages = [
        {"role": "system", "content": "System probe."},
        {"role": "user", "content": "User probe."},
    ]
    generation_prompt = tokenizer.apply_chat_template(
        probe_messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    tool_call_prompt = tokenizer.apply_chat_template(
        probe_messages
        + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "grpo-native-prefix-probe",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": {argument_name: GRPO_QUERY_SENTINEL},
                        },
                    }
                ],
            }
        ],
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not tool_call_prompt.startswith(generation_prompt):
        raise ValueError(
            "Tokenizer chat template cannot derive a native assistant tool-call prefix"
        )
    suffix = tool_call_prompt[len(generation_prompt) :]
    if suffix.count(GRPO_QUERY_SENTINEL) != 1:
        raise ValueError(
            "Tokenizer chat template did not preserve the native query sentinel exactly once"
        )
    return suffix.split(GRPO_QUERY_SENTINEL, 1)[0]


def _build_grpo_route_hint_prefix(
    messages: List[dict],
    tokenizer: Any,
    *,
    max_input_tokens: int,
) -> List[int]:
    """Build Host tokens ending after the forced Search direction prefill."""
    prefill_ids = tokenizer.encode(
        GRPO_ROUTE_HINT_PREFILL, add_special_tokens=False
    )
    if len(prefill_ids) >= max_input_tokens:
        raise ValueError("Route-hint prefill exceeds the Host input budget")

    hint_messages = [dict(message) for message in messages]
    hint_messages.append(
        {"role": "user", "content": GRPO_ROUTE_HINT_INSTRUCTION}
    )
    prompt, _ = _fit_messages_to_input_budget(
        hint_messages,
        [],
        tokenizer,
        max_input_tokens=max_input_tokens - len(prefill_ids),
    )
    host_prefix = tokenizer.encode(
        prompt + GRPO_ROUTE_HINT_PREFILL, add_special_tokens=False
    )
    if not host_prefix or len(host_prefix) > max_input_tokens:
        raise ValueError("Could not build the route-hint Host prefix")
    return host_prefix


def _build_grpo_native_search_prefix(
    messages: List[dict],
    tools: List[dict],
    tokenizer: Any,
    *,
    max_input_tokens: int,
    route_hint: str,
) -> List[int]:
    """Build Host tokens ending inside browser.search's JSON query string."""
    tool_prefix = _native_tool_call_prefix(
        tools,
        tokenizer,
        tool_name="browser.search",
        argument_name="query",
    )
    prefix_reserve = len(
        tokenizer.encode(tool_prefix, add_special_tokens=False)
    ) + 8
    if prefix_reserve >= max_input_tokens:
        raise ValueError("Native browser.search prefix exceeds the Host input budget")

    reroute_messages = [dict(message) for message in messages]
    reroute_messages.append(
        {
            "role": "user",
            "content": GRPO_HOST_QUERY_INSTRUCTION.format(route_hint=route_hint),
        }
    )
    prompt, _ = _fit_messages_to_input_budget(
        reroute_messages,
        tools,
        tokenizer,
        max_input_tokens=max_input_tokens - prefix_reserve,
    )
    host_prefix = tokenizer.encode(prompt + tool_prefix, add_special_tokens=False)
    if len(host_prefix) > max_input_tokens:
        excess = len(host_prefix) - max_input_tokens
        prompt, _ = _fit_messages_to_input_budget(
            reroute_messages,
            tools,
            tokenizer,
            max_input_tokens=max_input_tokens - prefix_reserve - excess - 8,
        )
        host_prefix = tokenizer.encode(prompt + tool_prefix, add_special_tokens=False)
    if not host_prefix or len(host_prefix) > max_input_tokens:
        raise ValueError(
            "Could not fit the native browser.search Host prefix in the input budget"
        )
    return host_prefix


def _native_query_closing_quote(text: str) -> int | None:
    escaped = False
    for index, character in enumerate(text):
        if character == "\\" and not escaped:
            escaped = True
            continue
        if character == "\"" and not escaped:
            return index
        escaped = False
    return None


def _clean_host_query_completion(text: str) -> str:
    closing_quote = _native_query_closing_quote(text)
    fragment = text if closing_quote is None else text[:closing_quote]
    if closing_quote is not None:
        try:
            fragment = json.loads(f"\"{fragment}\"")
        except json.JSONDecodeError:
            pass
    fragment = re.sub(r"</?[^>]+>", " ", fragment).strip()
    lines = [line.strip() for line in fragment.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Host generated an empty reroute query")
    query = re.sub(
        r"^(?:query|search query)\s*:\s*",
        "",
        lines[0],
        flags=re.IGNORECASE,
    ).strip().strip("\"").strip("\x27").strip()
    if not query:
        raise ValueError("Host generated an empty reroute query")
    return query[:1000]


def _validated_host_rewritten_query(value: str, *, max_words: int = 16) -> str:
    query = re.sub(r"\s+", " ", value).strip().strip("\"").strip("\x27").strip()
    if not query:
        raise ValueError("Host generated an empty reroute query")
    if _CONTROL_CHAR_RE.search(query):
        raise ValueError("Host reroute query contains control characters")
    if _CJK_RE.search(query):
        raise ValueError("Host reroute query must not contain CJK text")
    if _HOST_QUERY_FORBIDDEN_RE.search(query):
        raise ValueError("Host reroute query contains reasoning or error text")
    if any(character in query for character in "\r\n{}<>"):
        raise ValueError("Host reroute query contains forbidden formatting")
    words = query.split()
    if max_words > 0 and len(words) > max_words:
        query = " ".join(words[:max_words])
    if not re.search(r"[A-Za-z0-9]", query):
        raise ValueError("Host reroute query must contain searchable text")
    return query


async def _generate_host_reroute_query(
    generator: Any,
    host_prefix_token_ids: List[int],
    *,
    temperature: float,
    max_new_tokens: int,
    seed: int | None,
) -> tuple[str, List[int]]:
    """Generate the final query with the main Host and no plugin correction."""
    generated_token_ids: List[int] = []
    stream = generator.generate(
        host_prefix_token_ids,
        temperature=temperature,
        max_tokens=max_new_tokens,
        seed=seed,
    )
    async for token_id in stream:
        generated_token_ids.append(int(token_id))
        partial = generator.tokenizer.decode(
            generated_token_ids, skip_special_tokens=False
        )
        if _native_query_closing_quote(partial) is not None:
            break
    raw = generator.tokenizer.decode(
        generated_token_ids, skip_special_tokens=True
    )
    return _clean_host_query_completion(raw), generated_token_ids


async def _apply_grpo_corrected_reroute(
    *,
    question: str,
    qid: Any,
    round_num: int,
    checkpoint: dict,
    messages: List[dict],
    tools: List[dict],
    generator: Any,
    browser_pool: BrowserPool,
    plugin_client: GRPOPluginClient | None,
    grpo_reroute_trace: List[dict] | None,
    verbose: bool,
) -> dict | None:
    """Generate a corrected-logits hint, then let the Host rewrite it as a query."""
    reroute_trigger = checkpoint.get("reroute_trigger", "asag_reroute")
    retrieval_control = (
        "joint_controller_grpo_logits_reroute"
        if reroute_trigger == "joint_controller_reroute"
        else "asag_grpo_logits_reroute"
    )
    trace = {
        "asag_checkpoint": checkpoint.get("checkpoint"),
        "round": round_num,
        "trigger": reroute_trigger,
        "policy": "grpo_e2e_three_field_hint_host_rewrite",
        "logits_applied": False,
        "search_success": False,
        "fallback_to_asag_prompt": False,
        "error": None,
    }

    try:
        if plugin_client is None:
            raise RuntimeError("GRPO plugin URL is not configured")
        reroute_result = await plugin_client.reroute(question, messages)
        route_hint = _validated_route_hint(
            reroute_result.get("hint", reroute_result.get("route_hint")),
            question,
        )
        plugin_query = str(reroute_result.get("query", "")).strip()
        max_input_tokens = int(
            os.environ.get("OPENRESEARCHER_MAX_INPUT_TOKENS", "23000")
        )
        host_query_temperature = float(
            os.environ.get("GRPO_HOST_QUERY_TEMPERATURE", "0.7")
        )
        host_query_max_new_tokens = int(
            os.environ.get("GRPO_HOST_QUERY_MAX_NEW_TOKENS", "32")
        )
        host_query_max_words = int(os.environ.get("GRPO_HOST_QUERY_MAX_WORDS", "16"))
        host_query_seed = reroute_result.get("reroute_seed")
        if host_query_seed is None and os.environ.get("PLUGIN_REROUTE_SEED"):
            host_query_seed = int(os.environ["PLUGIN_REROUTE_SEED"])
        host_prefix_token_ids = _build_grpo_native_search_prefix(
            messages,
            tools,
            generator.tokenizer,
            max_input_tokens=max_input_tokens,
            route_hint=route_hint,
        )
        host_query, host_query_token_ids = await _generate_host_reroute_query(
            generator,
            host_prefix_token_ids,
            temperature=host_query_temperature,
            max_new_tokens=host_query_max_new_tokens,
            seed=host_query_seed,
        )
        forced_query = _validated_host_rewritten_query(
            host_query,
            max_words=host_query_max_words,
        )

        tool_id = (
            f"asag-grpo-reroute-{checkpoint.get('checkpoint', 0)}-"
            f"round-{round_num}"
        )
        assistant_message_index = len(messages)
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": None,
                "tool_calls": [
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": "browser.search",
                            "arguments": {"query": forced_query},
                        },
                    }
                ],
                "retrieval_control": retrieval_control,
            }
        )
        trace.update(
            {
                "logits_applied": True,
                "query": forced_query,
                "query_source": "host_rewrite_from_corrected_hint",
                "plugin_query": plugin_query,
                "route_hint": route_hint,
                "hint": route_hint,
                "hint_fields": reroute_result.get("fields", []),
                "token_count": len(reroute_result.get("token_ids", [])),
                "hint_token_count": len(reroute_result.get("token_ids", [])),
                "host_query_token_count": len(host_query_token_ids),
                "host_query_temperature": host_query_temperature,
                "host_query_max_new_tokens": host_query_max_new_tokens,
                "host_query_max_words": host_query_max_words,
                "host_query_seed": host_query_seed,
                "alpha": reroute_result.get("alpha"),
                "hint_seed": reroute_result.get("reroute_seed"),
                "logits_diagnostics": reroute_result.get("logits_diagnostics"),
            }
        )

        retrieval_started = time.perf_counter()
        max_search_attempts = max(
            1, int(os.environ.get("GRPO_REROUTE_SEARCH_ATTEMPTS", "2"))
        )
        result = ""
        search_errors: List[str] = []
        for search_attempt in range(1, max_search_attempts + 1):
            try:
                result = await browser_pool.call_tool(
                    qid, "search", {"query": forced_query}
                )
                if not result:
                    result = "Error executing browser.search: empty result"
            except Exception as exc:
                result = (
                    "Error executing browser.search: "
                    f"{type(exc).__name__}: {exc}"
                )
            if not _retrieval_result_is_error(result):
                break
            search_errors.append(result)
        retrieval_s = time.perf_counter() - retrieval_started
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result,
                "retrieval_control": retrieval_control,
            }
        )
        trace["retrieval_s"] = retrieval_s
        trace["search_attempts"] = search_attempt
        if search_errors:
            trace["search_errors"] = search_errors
        trace["search_success"] = not _retrieval_result_is_error(result)
        if not trace["search_success"]:
            raise RuntimeError(result)

        pending_checkpoint = {
            "segment": ResearchSegment(
                turn=round_num,
                reasoning=(
                    "ASAG selected reroute; step-70 corrected logits generated a "
                    "three-field hint, then the Host rewrote it as an English query."
                ),
                query=forced_query,
                evidence=result,
                tool_name="search",
            ),
            "retrieval_s": retrieval_s,
            "message_span": {
                "start": assistant_message_index,
                "end": len(messages),
            },
        }
        if verbose:
            print(
                "[ASAG_GRPO] Applied step-70 corrected-logits hint with Host "
                f"query rewrite at checkpoint {checkpoint.get('checkpoint')}: "
                f"{forced_query}"
            )
    except Exception as exc:
        trace["fallback_to_asag_prompt"] = True
        trace["error"] = f"{type(exc).__name__}: {exc}"
        messages.append({"role": "user", "content": checkpoint["reroute_prompt"]})
        pending_checkpoint = None
        if verbose:
            print(
                "[ASAG_GRPO] Corrected hint reroute failed; using ASAG prompt fallback: "
                f"{exc}"
            )

    checkpoint["grpo_reroute"] = trace
    if grpo_reroute_trace is not None:
        grpo_reroute_trace.append(trace)
    return pending_checkpoint


async def _maybe_promote_joint_controller_reroute(
    *,
    asag: RetrievalBoundaryASAG,
    checkpoint: dict,
    question: str,
    messages: List[dict],
    plugin_client: GRPOPluginClient | None,
    round_num: int,
    last_reroute_round: int | None,
    controller_reroute_count: int,
    enabled: bool,
    reroute_threshold: float,
    max_reroutes: int,
    verbose: bool,
) -> bool:
    """Promote ASAG Continue to Reroute when the trained 0.6B controller agrees.

    ASAG owns Stop/Verify/Force-answer. The learned controller is consulted
    only for ASAG Continue decisions, and high-confidence ASAG checkpoints are
    protected from controller intervention.
    """
    if checkpoint.get("decision") != "continue":
        return False

    trace: dict[str, Any] = {
        "enabled": enabled,
        "consulted": False,
        "triggered": False,
        "threshold": reroute_threshold,
        "reroute_count_before": controller_reroute_count,
        "reason": None,
        "error": None,
    }
    checkpoint["joint_controller"] = trace

    if not enabled:
        trace["reason"] = "joint_controller_disabled"
        return False
    if plugin_client is None:
        trace["reason"] = "plugin_client_unavailable"
        return False
    if asag.reroute_count >= asag.config.max_reroutes:
        trace["reason"] = "shared_asag_controller_reroute_limit_reached"
        return False
    if controller_reroute_count >= max_reroutes:
        trace["reason"] = "joint_controller_reroute_limit_reached"
        return False

    confidence = float(checkpoint.get("confidence", 0.0))
    if confidence > asag.config.confidence_threshold:
        trace["reason"] = "asag_high_confidence_guard"
        return False

    started_at = time.perf_counter()
    try:
        result = await plugin_client.control(
            question,
            messages,
            round_num=round_num,
            last_reroute_round=last_reroute_round,
        )
        trace["consulted"] = True
        trace["latency_s"] = time.perf_counter() - started_at
        trace["action"] = result.get("action")
        trace["action_id"] = result.get("action_id")
        trace["probabilities"] = result.get("probabilities")
        trace["allowed"] = result.get("allowed")

        probabilities = result.get("probabilities")
        if not isinstance(probabilities, dict):
            raise TypeError("0.6B controller returned invalid probabilities")
        reroute_probability = float(probabilities.get("REROUTE", 0.0))
        trace["reroute_probability"] = reroute_probability
        reroute_allowed = bool((result.get("allowed") or {}).get("REROUTE", True))
        if not reroute_allowed:
            trace["reason"] = "controller_reroute_masked_by_round_or_cooldown"
            return False
        if reroute_probability < reroute_threshold:
            trace["reason"] = "controller_probability_below_threshold"
            return False

        trace.update(
            {
                "triggered": True,
                "reason": "low_confidence_controller_probability_above_threshold",
                "reroute_count_after": controller_reroute_count + 1,
                "shared_reroute_count_before": asag.reroute_count,
            }
        )
        asag.reroute_count += 1
        checkpoint.update(
            {
                "asag_decision": "continue",
                "asag_reason": checkpoint.get("reason"),
                "decision": "reroute",
                "reason": "joint_controller_reroute",
                "reroute_trigger": "joint_controller_reroute",
                "reroute_count": asag.reroute_count,
                "reroute_prompt": asag.config.reroute_prompt,
            }
        )
        trace["shared_reroute_count_after"] = asag.reroute_count
        if verbose:
            print(
                "[ASAG_06B_JOINT] Promoted CONTINUE to REROUTE "
                f"at checkpoint {checkpoint.get('checkpoint')}: "
                f"P(REROUTE)={reroute_probability:.4f} "
                f">= {reroute_threshold:.4f}"
            )
        return True
    except Exception as exc:
        trace["latency_s"] = time.perf_counter() - started_at
        trace["reason"] = "controller_request_failed"
        trace["error"] = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"[ASAG_06B_JOINT] Controller failed; keeping ASAG Continue: {exc}")
        return False


async def _evaluate_asag_checkpoint(
    asag: RetrievalBoundaryASAG,
    pending: dict,
    messages: List[dict],
    tools: List[dict],
    generator: Any,
    asag_trace: List[dict] | None,
    verbose: bool,
) -> tuple[dict, str]:
    """Probe and evaluate the highest-priority evidence in one retrieval chain."""
    answer_probe_started = time.perf_counter()
    provisional_answer, confidence, probe_tokens, attention_context = await _run_asag_answer_probe(
        messages,
        tools,
        generator,
        asag.previous_message_span,
        pending["message_span"],
    )
    answer_probe_s = time.perf_counter() - answer_probe_started
    attention_probe_started = time.perf_counter()
    checkpoint = await asag.evaluate(
        pending["segment"],
        provisional_answer,
        confidence,
        attention_context,
        pending["message_span"],
    )
    attention_probe_s = time.perf_counter() - attention_probe_started
    checkpoint["answer_probe_tokens"] = probe_tokens
    checkpoint["retrieval_s"] = pending["retrieval_s"]
    checkpoint["answer_probe_s"] = answer_probe_s
    checkpoint["attention_probe_s"] = attention_probe_s
    if asag_trace is not None:
        asag_trace.append(checkpoint)
    if verbose:
        print(
            "[ASAG] "
            f"checkpoint={checkpoint['checkpoint']} "
            f"tool={pending['segment'].tool_name} "
            f"C={checkpoint['confidence']:.4f} "
            f"H={checkpoint.get('entropy')} "
            f"dH={checkpoint.get('entropy_delta')} "
            f"retrieval={pending['retrieval_s']:.2f}s "
            f"answer_probe={answer_probe_s:.2f}s "
            f"attention_probe={attention_probe_s:.2f}s "
            f"decision={checkpoint['decision']}"
        )
    return checkpoint, provisional_answer


async def run_one(
    question: str,
    qid: Any,
    generator: Any,
    browser_pool: BrowserPool,
    max_rounds: int = 200,
    verbose: bool = False,
    system_prompt_content: str | None = None,
    asag_attention_url: str | None = None,
    asag_trace: List[dict] | None = None,
    grpo_plugin_url: str | None = None,
    grpo_reroute_trace: List[dict] | None = None,
    resume_reroute: dict | None = None,
) -> List[dict]:
    """
    Helper function for native tool calling using tokenizer's chat template
    Uses tokenizer.apply_chat_template with tools parameter instead of OpenAI API
    """
    # Initialize browser session and get tool config
    tool_config = browser_pool.init_session(qid)

    # Initialize tokenizer
    if hasattr(generator, '_init_tokenizer'):
        await generator._init_tokenizer()

    # Initialize messages (Standard approach)
    prompt_content = system_prompt_content if system_prompt_content is not None else DEVELOPER_CONTENT
    system_prompt = prompt_content + f"\n\nToday's date: {datetime.datetime.now().strftime('%Y-%m-%d')}"
    if resume_reroute is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    else:
        messages = json.loads(json.dumps(resume_reroute["messages"]))

    # Parse TOOL_CONTENT from JSON string to list
    tools = json.loads(TOOL_CONTENT)

    # ASAG state is per question: H1 and the previous research segment must
    # never leak across concurrent dataset items.
    asag = RetrievalBoundaryASAG(
        question,
        asag_attention_url,
        session_id=(
            f"{os.getpid()}:"
            f"{hashlib.sha1(str(qid).encode('utf-8')).hexdigest()}"
        ),
    )
    plugin_client = GRPOPluginClient(grpo_plugin_url) if grpo_plugin_url else None
    joint_controller_enabled = os.getenv(
        "JOINT_CONTROLLER_ENABLED", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    joint_controller_threshold = float(
        os.getenv("JOINT_CONTROLLER_REROUTE_THRESHOLD", "0.70")
    )
    joint_controller_max_reroutes = int(
        os.getenv("JOINT_CONTROLLER_MAX_REROUTES", "1")
    )
    if not 0.0 <= joint_controller_threshold <= 1.0:
        raise ValueError(
            "JOINT_CONTROLLER_REROUTE_THRESHOLD must be between 0 and 1"
        )
    if joint_controller_max_reroutes < 0:
        raise ValueError("JOINT_CONTROLLER_MAX_REROUTES must be non-negative")
    min_post_reroute_rounds = int(
        os.getenv("GRPO_MIN_POST_REROUTE_ROUNDS", "20")
    )
    max_extra_reroute_rounds = int(
        os.getenv("GRPO_MAX_EXTRA_REROUTE_ROUNDS", "20")
    )
    if min_post_reroute_rounds < 0:
        raise ValueError("GRPO_MIN_POST_REROUTE_ROUNDS must be non-negative")
    if max_extra_reroute_rounds < 0:
        raise ValueError("GRPO_MAX_EXTRA_REROUTE_ROUNDS must be non-negative")
    controller_reroute_count = 0
    last_reroute_round: int | None = None

    if resume_reroute is not None:
        prior_trace = resume_reroute.get("asag_trace") or []
        if not prior_trace:
            raise ValueError("resume_reroute requires an ASAG trace")
        last_checkpoint = prior_trace[-1]
        asag.checkpoint_count = int(last_checkpoint["checkpoint"])
        asag.initial_entropy = last_checkpoint.get("initial_entropy")
        asag.reroute_count = max((int(x.get("reroute_count", 0)) for x in prior_trace), default=0)
        asag.verification_count = max((int(x.get("verification_count", 0)) for x in prior_trace), default=0)
        asag.previous_normalized_answer = str(last_checkpoint.get("normalized_answer", ""))
        asag.previous_segment = ResearchSegment(**last_checkpoint["segment"])
        asag.previous_message_span = dict(resume_reroute["previous_message_span"])
        controller_reroute_count = sum(bool((x.get("joint_controller") or {}).get("triggered")) for x in prior_trace)
        last_reroute_round = int(resume_reroute["round_num"])

    # Define stop strings
    stop_strings = ["\n<tool_response>", "<tool_response>"]

    round_num = 0 if resume_reroute is None else int(resume_reroute["round_num"])
    round_limit = max_rounds if resume_reroute is None else int(resume_reroute.get("round_limit", max_rounds))
    # A search may be followed by open and then find.  Hold the best evidence
    # until the chain ends so ASAG observes find > open > search, rather than
    # stopping immediately on snippets.
    pending_checkpoint: dict | None = None

    try:
        if resume_reroute is not None:
            checkpoint = json.loads(json.dumps(resume_reroute["checkpoint"]))
            pending_checkpoint = await _apply_grpo_corrected_reroute(
                question=question, qid=qid, round_num=round_num,
                checkpoint=checkpoint, messages=messages, tools=tools,
                generator=generator, browser_pool=browser_pool,
                plugin_client=plugin_client,
                grpo_reroute_trace=grpo_reroute_trace, verbose=verbose,
            )
            branch_trace = checkpoint.get("grpo_reroute", {})
            if not (branch_trace.get("logits_applied") and branch_trace.get("search_success")):
                raise RuntimeError("continuation failed to apply a successful reroute")
            branch_trace["round_limit_before"] = round_limit
            branch_trace["round_limit_after"] = round_limit

        while round_num < round_limit:
            round_num += 1

            if verbose:
                print(f"\n{'='*60}")
            if verbose:
                print(f"Round {round_num}")
            if verbose:
                print(f"{'='*60}")

            max_input_tokens = int(os.environ.get("OPENRESEARCHER_MAX_INPUT_TOKENS", "23000"))
            prompt, tokens = _fit_messages_to_input_budget(
                messages,
                tools,
                generator.tokenizer,
                max_input_tokens=max_input_tokens,
            )

            # Generate using /completions endpoint
            continuation_seed = os.environ.get("CONTINUATION_SEED")
            generation_seed = (
                int(continuation_seed) + round_num if continuation_seed else None
            )
            content = await _generate_with_retry(
                generator, tokens, stop_strings, verbose=verbose,
                seed=generation_seed,
            )
            non_thinking_content = content

            if verbose:
                print(f'[NATIVE_TOOLS] Round {round_num}: {content[:500] if len(content) > 500 else content}')

            # Remove tool_response marker if present
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]

            # Step 1: Extract and remove <think> tags from content
            reasoning_content = None
            if '<think>' in content and '</think>' in content:
                # Match <think>...</think>
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                if think_match:
                    reasoning_content = think_match.group(1).strip()
                    # Remove the entire <think>...</think> block
                    content = content.replace(think_match.group(0), "").strip() # TODO: Note
            elif '</think>' in content:
                # No opening tag, match from start to </think> (inclusive)
                think_match = re.search(r'^(.*?)</think>', content, re.DOTALL)
                if think_match:
                    reasoning_content = think_match.group(1).strip()
                    # group(0) already includes </think> tag, so just replace it
                    content = content.replace(think_match.group(0), "").strip() # TODO: Note 

            # Step 2: Extract and remove <tool_call> tags from content
            parsed_tool_calls = None
            tool_call_text = None

            if '<tool_call>' in content and '</tool_call>' in content:
                # Match <tool_call>...</tool_call>
                tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
                if tool_call_match:
                    tool_call_text = tool_call_match.group(1).strip()
                    # Remove the entire <tool_call>...</tool_call> block
                    content = content.replace(tool_call_match.group(0), "").strip()
            elif '</tool_call>' in content:
                # No opening tag, match from start to </tool_call> (inclusive)
                tool_call_match = re.search(r'^(.*?)</tool_call>', content, re.DOTALL)
                if tool_call_match:
                    tool_call_text = tool_call_match.group(1).strip()
                    # group(0) already includes </tool_call> tag
                    content = content.replace(tool_call_match.group(0), "").strip()

            if tool_call_text:
                # Try to parse as JSON first
                try:
                    parsed_tool_call = json5.loads(tool_call_text)
                    # Convert to tool_calls format for consistency
                    parsed_tool_calls = [{
                        "id": f"{round_num}",
                        "type": "function",
                        "function": {
                            "name": parsed_tool_call.get("name", ""),
                            "arguments": parsed_tool_call.get("arguments", {})
                        }
                    }]
                    if verbose:
                        print(f"[NATIVE_TOOLS] Parsed tool call (JSON): {parsed_tool_call}")
                except Exception as e:
                    # Fallback: Try to parse XML format
                    # <function=browser.search>
                    # <parameter=query>value</parameter>
                    # </function>
                    if verbose:
                        print(f"[NATIVE_TOOLS] JSON parsing failed, trying XML format: {e}")
                    # Match function name (allow dots and other characters)
                    func_match = re.search(r'<function=([\w.]+)>', tool_call_text)
                    if func_match:
                        tool_name = func_match.group(1)
                        tool_args = {}
                        # Match parameters with values that may span multiple lines
                        params = re.finditer(r'<parameter=([\w]+)>\s*(.*?)\s*</parameter>', tool_call_text, re.DOTALL)
                        for p in params:
                            param_name = p.group(1)
                            param_value = p.group(2).strip()
                            # Remove quotes if present
                            if param_value.startswith('"') and param_value.endswith('"'):
                                param_value = param_value[1:-1]
                            # Try to parse as int if it looks like a number
                            try:
                                if param_value.isdigit():
                                    param_value = int(param_value)
                            except:
                                pass
                            tool_args[param_name] = param_value

                        # Convert to tool_calls format
                        parsed_tool_calls = [{
                            "id": f"{round_num}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_args
                            }
                        }]
                        if verbose:
                            print(f"[NATIVE_TOOLS] Parsed tool call (XML): name={tool_name}, args={tool_args}")
                    else:
                        if verbose:
                            print(f"[NATIVE_TOOLS] Failed to parse tool call in both JSON and XML formats")
                        if verbose:
                            print(f"[NATIVE_TOOLS] Tool call text: {tool_call_text}")

            # Dr.tulu-style direct XML tool call, for example:
            # <call_tool name="search" topn="10">query</call_tool>
            if not parsed_tool_calls:
                content, direct_tool_text, direct_tool_calls = _parse_call_tool_xml(
                    content, round_num
                )
                if direct_tool_calls:
                    tool_call_text = direct_tool_text
                    parsed_tool_calls = direct_tool_calls
                    if verbose:
                        print(
                            f"[NATIVE_TOOLS] Parsed <call_tool>: "
                            f"{parsed_tool_calls[0]['function']}"
                        )

            if verbose:
                print(f"[NATIVE_TOOLS] Assistant response (cleaned):\n{content}")
            if reasoning_content:
                if verbose:
                    print(f"[NATIVE_TOOLS] Reasoning content:\n{reasoning_content}")

            # A new search (or a non-retrieval tool) closes the previous
            # search/open chain. Evaluate it before adding the new assistant
            # action so the answer probe sees exactly the evidence available at
            # that boundary. A natural final answer needs no ASAG intervention.
            next_tool_name = None
            if parsed_tool_calls:
                next_tool_name = _browser_tool_name(
                    parsed_tool_calls[0].get("function", {}).get("name", "")
                )
            if (
                asag.config.enabled
                and pending_checkpoint is not None
                and parsed_tool_calls
                and _should_flush_pending_checkpoint(
                    pending_checkpoint["segment"].tool_name,
                    next_tool_name,
                )
            ):
                checkpoint, provisional_answer = await _evaluate_asag_checkpoint(
                    asag,
                    pending_checkpoint,
                    messages,
                    tools,
                    generator,
                    asag_trace,
                    verbose,
                )
                controller_triggered = await _maybe_promote_joint_controller_reroute(
                    asag=asag,
                    checkpoint=checkpoint,
                    question=question,
                    messages=messages,
                    plugin_client=plugin_client,
                    round_num=round_num,
                    last_reroute_round=last_reroute_round,
                    controller_reroute_count=controller_reroute_count,
                    enabled=joint_controller_enabled,
                    reroute_threshold=joint_controller_threshold,
                    max_reroutes=joint_controller_max_reroutes,
                    verbose=verbose,
                )
                if controller_triggered:
                    controller_reroute_count += 1
                    last_reroute_round = round_num
                pending_checkpoint = None
                if checkpoint["decision"] == "stop":
                    messages.append(_asag_final_message(provisional_answer))
                    break
                if checkpoint["decision"] == "reroute":
                    last_reroute_round = round_num
                    pending_checkpoint = await _apply_grpo_corrected_reroute(
                        question=question,
                        qid=qid,
                        round_num=round_num,
                        checkpoint=checkpoint,
                        messages=messages,
                        tools=tools,
                        generator=generator,
                        browser_pool=browser_pool,
                        plugin_client=plugin_client,
                        grpo_reroute_trace=grpo_reroute_trace,
                        verbose=verbose,
                    )
                    reroute_trace = checkpoint.get("grpo_reroute", {})
                    if (
                        reroute_trace.get("logits_applied")
                        and reroute_trace.get("search_success")
                    ):
                        previous_round_limit = round_limit
                        round_limit = _extend_round_limit_after_reroute(
                            base_max_rounds=max_rounds,
                            current_round_limit=round_limit,
                            reroute_round=round_num,
                            min_post_reroute_rounds=min_post_reroute_rounds,
                            max_extra_rounds=max_extra_reroute_rounds,
                        )
                        reroute_trace["round_limit_before"] = previous_round_limit
                        reroute_trace["round_limit_after"] = round_limit
                    continue
                if checkpoint["decision"] == "verify":
                    messages.append(
                        {"role": "user", "content": checkpoint["verification_prompt"]}
                    )
                    continue
                if checkpoint["decision"] == "force_answer":
                    await _append_forced_final_answer(
                        messages, generator, provisional_answer, verbose
                    )
                    break

            if tool_call_text is None:
                non_thinking_content = non_thinking_content.split('</think>', 1)[1].strip() if '</think>' in non_thinking_content else non_thinking_content.strip()
            assistant_message_index = len(messages)
            messages.append({
                "role": "assistant",
                "content": non_thinking_content if tool_call_text is None else "",
                "reasoning_content": reasoning_content,
                "tool_calls": parsed_tool_calls
            })

            # Check if there are tool calls
            if parsed_tool_calls:
                if verbose:
                    print(f"[NATIVE_TOOLS] Tool calls: {len(parsed_tool_calls)}")

                asag_stopped = False

                # Execute each tool call
                for tool_index, tool_call in enumerate(parsed_tool_calls):
                    tool_id = tool_call["id"]
                    function_name = tool_call["function"]["name"]  # e.g., "browser.search"
                    function_args_raw = tool_call["function"]["arguments"]
                    retrieval_completed = False

                    try:
                        # Parse arguments (handle both dict and string formats)
                        if isinstance(function_args_raw, dict):
                            function_args = function_args_raw
                        else:
                            function_args = json.loads(function_args_raw)
                        if verbose:
                            print(f"\n[NATIVE_TOOLS] === Tool Call ===")
                        if verbose:
                            print(f"[NATIVE_TOOLS] Tool ID: {tool_id}")
                        if verbose:
                            print(f"[NATIVE_TOOLS] Function: {function_name}")
                        if verbose:
                            print(f"[NATIVE_TOOLS] Arguments (full):\n{json.dumps(function_args, indent=2, ensure_ascii=False)}")

                        # Extract actual function name from browser.xxx format
                        actual_function_name = _browser_tool_name(function_name)

                        # Execute browser tool
                        retrieval_started = time.perf_counter()
                        if actual_function_name.lower() in ['search', 'find', 'open']:
                            result = await browser_pool.call_tool(qid, actual_function_name, function_args)
                            if not result:
                                result = f"Error executing {function_name}: empty result"
                        else:
                            result = f"Tool {function_name} not available"
                        retrieval_s = time.perf_counter() - retrieval_started

                        # Add tool response to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result
                        })
                        retrieval_completed = (
                            actual_function_name.lower() in {"search", "find", "open"}
                            and not _retrieval_result_is_error(result)
                        )

                        # Build a retrieval-chain candidate. Search is held for
                        # a possible open/find, open supersedes search, and find
                        # is the highest-priority boundary and is evaluated now.
                        if asag.config.enabled and retrieval_completed:
                            segment = ResearchSegment(
                                turn=round_num,
                                reasoning=reasoning_content or content or "",
                                query=_research_query_text(actual_function_name, function_args),
                                evidence=result,
                                tool_name=actual_function_name.lower(),
                            )
                            candidate = {
                                "segment": segment,
                                "retrieval_s": retrieval_s,
                                "message_span": {
                                    "start": assistant_message_index,
                                    "end": len(messages),
                                },
                            }
                            if actual_function_name == "find":
                                pending_checkpoint = None
                                checkpoint, provisional_answer = await _evaluate_asag_checkpoint(
                                    asag,
                                    candidate,
                                    messages,
                                    tools,
                                    generator,
                                    asag_trace,
                                    verbose,
                                )
                                controller_triggered = await _maybe_promote_joint_controller_reroute(
                                    asag=asag,
                                    checkpoint=checkpoint,
                                    question=question,
                                    messages=messages,
                                    plugin_client=plugin_client,
                                    round_num=round_num,
                                    last_reroute_round=last_reroute_round,
                                    controller_reroute_count=controller_reroute_count,
                                    enabled=joint_controller_enabled,
                                    reroute_threshold=joint_controller_threshold,
                                    max_reroutes=joint_controller_max_reroutes,
                                    verbose=verbose,
                                )
                                if controller_triggered:
                                    controller_reroute_count += 1
                                    last_reroute_round = round_num

                                if checkpoint["decision"] == "stop":
                                    for pending_call in parsed_tool_calls[tool_index + 1:]:
                                        messages.append(
                                            {
                                                "role": "tool",
                                                "tool_call_id": pending_call["id"],
                                                "content": "Skipped because ASAG selected Stop.",
                                            }
                                        )
                                    messages.append(_asag_final_message(provisional_answer))
                                    asag_stopped = True
                                    break
                                if checkpoint["decision"] == "reroute":
                                    last_reroute_round = round_num
                                    for pending_call in parsed_tool_calls[tool_index + 1:]:
                                        messages.append(
                                            {
                                                "role": "tool",
                                                "tool_call_id": pending_call["id"],
                                                "content": "Skipped because ASAG selected Reroute.",
                                            }
                                        )
                                    pending_checkpoint = await _apply_grpo_corrected_reroute(
                                        question=question,
                                        qid=qid,
                                        round_num=round_num,
                                        checkpoint=checkpoint,
                                        messages=messages,
                                        tools=tools,
                                        generator=generator,
                                        browser_pool=browser_pool,
                                        plugin_client=plugin_client,
                                        grpo_reroute_trace=grpo_reroute_trace,
                                        verbose=verbose,
                                    )
                                    reroute_trace = checkpoint.get("grpo_reroute", {})
                                    if (
                                        reroute_trace.get("logits_applied")
                                        and reroute_trace.get("search_success")
                                    ):
                                        previous_round_limit = round_limit
                                        round_limit = _extend_round_limit_after_reroute(
                                            base_max_rounds=max_rounds,
                                            current_round_limit=round_limit,
                                            reroute_round=round_num,
                                            min_post_reroute_rounds=min_post_reroute_rounds,
                                            max_extra_rounds=max_extra_reroute_rounds,
                                        )
                                        reroute_trace["round_limit_before"] = previous_round_limit
                                        reroute_trace["round_limit_after"] = round_limit
                                    break
                                if checkpoint["decision"] == "verify":
                                    for pending_call in parsed_tool_calls[tool_index + 1:]:
                                        messages.append(
                                            {
                                                "role": "tool",
                                                "tool_call_id": pending_call["id"],
                                                "content": "Skipped because ASAG requested verification.",
                                            }
                                        )
                                    messages.append(
                                        {
                                            "role": "user",
                                            "content": checkpoint["verification_prompt"],
                                        }
                                    )
                                    break
                                if checkpoint["decision"] == "force_answer":
                                    for pending_call in parsed_tool_calls[tool_index + 1:]:
                                        messages.append(
                                            {
                                                "role": "tool",
                                                "tool_call_id": pending_call["id"],
                                                "content": "Skipped because the ASAG reroute budget was exhausted.",
                                            }
                                        )
                                    await _append_forced_final_answer(
                                        messages, generator, provisional_answer, verbose
                                    )
                                    asag_stopped = True
                                    break
                            else:
                                pending_checkpoint = candidate

                        if verbose:
                            print(f"[NATIVE_TOOLS] Tool Result (full):\n{result}")
                        if verbose:
                            print(f"[NATIVE_TOOLS] === End Tool Call ===\n")

                    except Exception as e:
                        # Once retrieval succeeded, failures belong to the ASAG
                        # checkpoint and must not be mislabeled as browser tool
                        # errors or silently bypass the strict attention policy.
                        if retrieval_completed:
                            raise
                        error_msg = f"Error executing {function_name}: {str(e)}"
                        if verbose:
                            print(f"[NATIVE_TOOLS] Error: {error_msg}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": error_msg
                        })

                if asag_stopped:
                    break

                # Continue to next round (including the reroute prompt path).
                continue

            # Check for answer termination
            content_lower = content.lower()
            if '<answer>' in content_lower and '</answer>' in content_lower:
                if verbose:
                    print(f"\n✅ Found <answer> tag - conversation completed")
                break

            if '<suggested_answer>' in content_lower and '</suggested_answer>' in content_lower:
                if verbose:
                    print(f"\n✅ Found <suggested_answer> tag - conversation completed")
                break

            if "exact answer:" in content_lower and "confidence:" in content_lower:
                if verbose:
                    print(f"\n✅ Found 'Exact Answer:' and 'Confidence:' - conversation completed")
                break

            if "final answer:" in content_lower or "answer:" in content_lower:
                if verbose:
                    print(f"\n✅ Found 'Final Answer:' or 'Answer:' - conversation completed")
                break

            # In a tool-calling loop, reaching this point already means no
            # parseable tool call was emitted. A non-empty assistant response
            # is therefore the terminal answer even when the model omits the
            # requested <answer> or "Final Answer:" wrapper.
            if _is_terminal_assistant_response(content, parsed_tool_calls):
                if verbose:
                    print("\n✅ Found non-empty assistant response without a tool call - conversation completed")
                break

        # If the round budget ends directly after search/open, there is no next
        # action to close the chain. Record one final checkpoint using the best
        # available evidence. There is no remaining round for Continue,
        # Verify, or Reroute, so synthesize an answer with tools disabled.
        if (
            asag.config.enabled
            and pending_checkpoint is not None
            and messages
            and messages[-1].get("role") == "tool"
        ):
            checkpoint, provisional_answer = await _evaluate_asag_checkpoint(
                asag,
                pending_checkpoint,
                messages,
                tools,
                generator,
                asag_trace,
                verbose,
            )
            if checkpoint["decision"] == "stop":
                messages.append(_asag_final_message(provisional_answer))
            else:
                await _append_forced_final_answer(
                    messages, generator, provisional_answer, verbose
                )

        # A failed retrieval can leave the final message as a tool error with
        # no pending ASAG checkpoint. Never expose that tool payload as the
        # model's answer; synthesize one final answer with tools disabled.
        if messages and messages[-1].get("role") == "tool":
            await _append_forced_final_answer(messages, generator, "", verbose)

        return messages

    finally:
        if plugin_client is not None:
            await plugin_client.close()
        await asag.close()
        browser_pool.cleanup(qid)



def _load_continuation_sources(path: str) -> dict[str, dict]:
    records = {}
    source = Path(path)
    paths = sorted(source.glob("node_*_shard_*.jsonl")) if source.is_dir() else [source]
    for shard in paths:
        with shard.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    records[str(row["qid"])] = row
    return records


def _build_reroute_resume(row: dict) -> dict:
    successful = [x for x in row.get("grpo_reroute_trace", []) if x.get("logits_applied") and x.get("search_success") and x.get("query")]
    if not successful:
        raise ValueError(f"QID {row['qid']} has no successful alpha=20 reroute")
    source = successful[0]
    intervention_indices = []
    for index, message in enumerate(row.get("messages", [])):
        if not message.get("retrieval_control"):
            continue
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name", "").endswith("search") and (fn.get("arguments") or {}).get("query") == source["query"]:
                intervention_indices.append(index)
    if len(intervention_indices) != 1:
        raise ValueError(f"QID {row['qid']} intervention matches={len(intervention_indices)}")
    cut = intervention_indices[0]
    prefix = json.loads(json.dumps(row["messages"][:cut]))
    checkpoint_id = int(source["asag_checkpoint"])
    prior = [json.loads(json.dumps(x)) for x in row.get("asag_trace", []) if int(x.get("checkpoint", 0)) <= checkpoint_id]
    if not prior or int(prior[-1]["checkpoint"]) != checkpoint_id:
        raise ValueError(f"QID {row['qid']} missing checkpoint {checkpoint_id}")
    tool_index = next((i for i in range(len(prefix)-1, -1, -1) if prefix[i].get("role") == "tool"), None)
    if tool_index is None:
        raise ValueError(f"QID {row['qid']} has no retrieval before reroute")
    assistant_index = next((i for i in range(tool_index-1, -1, -1) if prefix[i].get("role") == "assistant" and prefix[i].get("tool_calls")), None)
    if assistant_index is None:
        raise ValueError(f"QID {row['qid']} has no retrieval assistant span")
    checkpoint = json.loads(json.dumps(prior[-1]))
    checkpoint.pop("grpo_reroute", None)
    return {
        "messages": prefix,
        "asag_trace": prior,
        "checkpoint": checkpoint,
        "previous_message_span": {"start": assistant_index, "end": tool_index + 1},
        "round_num": int(source["round"]),
        "round_limit": int(source.get("round_limit_after", 80)),
        "source_query": source["query"],
        "source_alpha": source.get("alpha"),
    }

def worker_entry(
    worker_idx,
    num_workers,
    args,
    gpu_ids,
):
    # Set visible GPUs for this worker (empty list for API mode)
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gid) for gid in gpu_ids)
    else:
        # API mode - no GPUs needed
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    node_rank = int(os.getenv("RANK", 0))
    node_size = int(os.getenv("WORLD_SIZE", 1))

    async def _run():
        try:
            # Initialize generator based on mode
            if args.vllm_server_url:
                # Get the server URL for this worker
                if hasattr(args, 'vllm_server_urls') and len(args.vllm_server_urls) > 1:
                    server_url = args.vllm_server_urls[worker_idx % len(args.vllm_server_urls)]
                else:
                    server_url = args.vllm_server_url

                # Use OpenAI API with optional native tools support
                from utils.openai_generator import OpenAIAsyncGenerator
                generator = OpenAIAsyncGenerator(
                    base_url=server_url,
                    model_name=args.model_name_or_path,
                    tokenizer_name=args.tokenizer_name_or_path,
                    use_native_tools=True
                )

                print(f"[Worker {worker_idx}] Using OpenAI API (native function calling) at {server_url}")
            else:
                # Use local vLLM engine (slow startup)
                from utils.vllm_generator import vLLMAsyncGenerator
                generator = vLLMAsyncGenerator(
                    args.model_name_or_path,
                    tensor_parallel_size=args.tensor_parallel_size
                )
                print(f"[Worker {worker_idx}] Using local vLLM engine")


            search_urls = getattr(args, "search_urls", [args.search_url])
            search_url = search_urls[0 if len(search_urls) == 1 else worker_idx]
            plugin_urls = getattr(args, "grpo_plugin_urls", [])
            plugin_url = (
                plugin_urls[0 if len(plugin_urls) == 1 else worker_idx]
                if plugin_urls
                else None
            )
            asag_urls = getattr(args, "asag_attention_urls", [])
            asag_url = (
                asag_urls[0 if len(asag_urls) == 1 else worker_idx]
                if asag_urls
                else None
            )
            browser_pool = BrowserPool(search_url, browser_backend=args.browser_backend)
            print(f"[Worker {worker_idx}] Using search service at {search_url}")
            if asag_url:
                print(
                    f"[Worker {worker_idx}] Using ASAG attention endpoint at "
                    f"{asag_url}"
                )
            if plugin_url:
                print(
                    f"[Worker {worker_idx}] ASAG-triggered GRPO logits reroute at "
                    f"{plugin_url}"
                )
            sem = asyncio.Semaphore(args.max_concurrency_per_worker)

            shard_path = os.path.join(args.output_dir, f"node_{node_rank}_shard_{worker_idx}.jsonl")
            os.makedirs(args.output_dir, exist_ok=True)

            processed_qids = set()
            if args.fresh_run:
                print(
                    f"[Worker {worker_idx}] Fresh run: no existing QIDs will be skipped."
                )
            else:
                # Load completed tasks from ALL shard files, including shards
                # written by another worker before a resumed launch.
                print(
                    f"[Worker {worker_idx}] Scanning all shard files for completed tasks..."
                )
                for shard_file in glob.glob(
                    os.path.join(args.output_dir, "node_*_shard_*.jsonl")
                ):
                    try:
                        with open(shard_file, "r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    record = json.loads(line)
                                    if (
                                        not args.resume_success_only
                                        or record.get("status") == "success"
                                    ):
                                        processed_qids.add(str(record["qid"]))
                                except Exception:
                                    continue
                    except Exception as e:
                        print(
                            f"[Worker {worker_idx}] Warning: Could not read "
                            f"{shard_file}: {e}"
                        )
                print(
                    f"[Worker {worker_idx}] Found {len(processed_qids)} "
                    "completed tasks across all shards."
                )

            continuation_sources = None
            if args.continuation_source:
                continuation_sources = _load_continuation_sources(args.continuation_source)
                print(f"[Worker {worker_idx}] Loaded {len(continuation_sources)} continuation sources")

            # Load dataset using unified loader
            # If data_path provided, pass it (for backward compatibility with browsecomp-plus)
            if args.data_path:
                # Legacy mode: explicit data_path for browsecomp-plus
                data = load_dataset(args.dataset_name, data_path=args.data_path)
            else:
                # New unified mode: load from HuggingFace
                data = load_dataset(args.dataset_name)

            if args.qid_file:
                with open(args.qid_file, "r", encoding="utf-8") as f:
                    qid_filter = {line.strip() for line in f if line.strip()}
                before_filter = len(data)
                data = [x for x in data if str(x.get("qid")) in qid_filter]
                missing = len(qid_filter) - len({str(x.get("qid")) for x in data})
                print(
                    f"[Worker {worker_idx}] QID filter: loaded {len(qid_filter)} qids from {args.qid_file}; "
                    f"dataset {before_filter} -> {len(data)}; missing={missing}"
                )
                if not data:
                    raise ValueError(f"QID filter {args.qid_file} matched 0 tasks.")

            total_workers = node_size * num_workers
            global_worker_idx = num_workers * node_rank + worker_idx

            # Dynamic load balancing: redistribute unprocessed tasks among all workers
            # This ensures all workers stay busy even if previous runs were interrupted
            all_unprocessed_tasks = [x for x in data if str(x['qid']) not in processed_qids]
            tasks_to_process = all_unprocessed_tasks[global_worker_idx::total_workers]

            print(f"[Worker {worker_idx}] Total tasks: {len(data)}, "
                  f"Unprocessed: {len(all_unprocessed_tasks)}, "
                  f"Assigned to this worker: {len(tasks_to_process)}")
            
            if not tasks_to_process:
                print(f"[Worker {worker_idx}] Nothing to do.")
                return

            async def process_item(item_data: dict) -> dict:
                async with sem:
                    qid = item_data['qid']
                    question = item_data['question']
                    MAX_RETRY = 5
                    attempt = 0
                    error_msg = None
                    t0 = time.time()
                    while attempt < MAX_RETRY:
                        attempt += 1
                        try:
                            asag_trace = []
                            grpo_reroute_trace = []
                            # run_one now automatically handles both native and custom tool formats.
                            messages = await run_one(
                                question=question,
                                qid=qid,
                                generator=generator,
                                browser_pool=browser_pool,
                                max_rounds=args.max_rounds,
                                verbose=args.verbose,
                                system_prompt_content=args.system_prompt_content,
                                asag_attention_url=asag_url,
                                asag_trace=asag_trace,
                                grpo_plugin_url=plugin_url,
                                grpo_reroute_trace=grpo_reroute_trace,
                                resume_reroute=(
                                    _build_reroute_resume(continuation_sources[str(qid)])
                                    if continuation_sources is not None else None
                                ),
                            )
                            dt = time.time() - t0
                            rec = item_data.copy()
                            rec.update({
                                "messages": messages,
                                "asag_trace": asag_trace,
                                "grpo_reroute_trace": grpo_reroute_trace,
                                "latency_s": dt,
                                "error": None,
                                "attempts": attempt,
                                "status": "success",
                            })
                            return rec
                        except Exception as e:
                            error_msg = traceback.format_exc()
                            print(f"[Worker {worker_idx}] qid {qid} attempt {attempt}/{MAX_RETRY} failed: {e}")
                    rec = item_data.copy()
                    rec.update({"messages": [], "latency_s": 0.0, "error": error_msg, "attempts": attempt, "status":"fail"})
                    return rec

            tasks = [asyncio.create_task(process_item(task)) for task in tasks_to_process]

            completed_count = len(processed_qids)
            total_count = len(data)
            progress = tqdm.tqdm(
                total=total_count,
                initial=completed_count,
                desc=f"Worker {worker_idx} total",
                unit="q",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, left={postfix}]",
            )

            with open(shard_path, "a", encoding="utf-8") as writer:
                try:
                    for fut in asyncio.as_completed(tasks):
                        rec = await fut
                        writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        writer.flush()
                        completed_count += 1
                        left_count = max(total_count - completed_count, 0)
                        progress.set_postfix_str(str(left_count), refresh=False)
                        progress.update(1)
                finally:
                    progress.close()
        finally:
            print(f"[Worker {worker_idx}] Done.")

    asyncio.run(_run())
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument(
        "--system_prompt_file",
        default=None,
        help="Python file exporting SYSTEM_PROMPT; explicitly propagated to spawn workers.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        default=None,
        help="Tokenizer name/path when the API served model name is an alias (for example a LoRA name)",
    )
    parser.add_argument("--search_url", required=True)
    parser.add_argument(
        "--asag_attention_url",
        default=None,
        help=(
            "URL(s) of the last-four-layer ASAG attention endpoint. Supply one "
            "shared URL or one comma-separated URL per worker."
        ),
    )
    parser.add_argument(
        "--grpo_plugin_url",
        default=None,
        help=(
            "URL of the GRPO corrected-logits sidecar. When set, ASAG REROUTE "
            "decisions use the plugin query generator and fall back to the ASAG prompt."
        ),
    )
    parser.add_argument("--dataset_name", type=str, default="browsecomp-plus",
                        help=f"Dataset name (default: browsecomp-plus). Available: {', '.join(list_available_datasets())}")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to local data files (only required for browsecomp-plus dataset)")
    parser.add_argument("--qid_file", type=str, default=None,
                        help="Optional text file with one qid per line. If set, only these qids are evaluated.")
    parser.add_argument("--continuation_source", type=str, default=None,
                        help="Alpha=20 result directory/JSONL used to resume at the first successful reroute prefix.")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--resume_success_only",
        action="store_true",
        help="When resuming, skip only QIDs with an existing status=success record; retry existing failures.",
    )
    run_mode.add_argument(
        "--fresh_run",
        action="store_true",
        help=(
            "Process every selected QID. Refuses to start if output shard files "
            "already exist, preventing duplicate rows."
        ),
    )
    parser.add_argument("--browser_backend", type=str, default="local", choices=["local", "serper"],
                        help="Browser backend: 'local' (default) or 'serper'")
    parser.add_argument("--max_concurrency_per_worker", type=int, default=6)
    parser.add_argument("--max_rounds", type=int, default=200,
                        help="Maximum browser-agent rounds per question (default: 200).")
    parser.add_argument("--reasoning_effort", default='high')
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for local vLLM (default: 1, ignored if using --vllm_server_url)")
    parser.add_argument("--vllm_server_url", type=str, default=None,
                        help="URL(s) of vLLM OpenAI-compatible server. "
                             "Single URL: http://localhost:8001/v1 "
                             "Multiple URLs (comma-separated): http://localhost:8001/v1,http://localhost:8002/v1 "
                             "If provided, will use API instead of local vLLM engine (recommended for faster startup)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-round model/tool debug logs. Default is quiet progress-only mode.")

    args = parser.parse_args()
    print(args)

    if args.fresh_run:
        existing_shards = glob.glob(
            os.path.join(args.output_dir, "node_*_shard_*.jsonl")
        )
        if existing_shards:
            raise ValueError(
                "--fresh_run requires an output directory with no existing shard "
                f"files; found {len(existing_shards)} in {args.output_dir}."
            )

    if args.system_prompt_file:
        prompt_namespace = runpy.run_path(args.system_prompt_file)
        if "SYSTEM_PROMPT" not in prompt_namespace:
            raise ValueError(
                f"Prompt file does not export SYSTEM_PROMPT: {args.system_prompt_file}"
            )
        args.system_prompt_content = prompt_namespace["SYSTEM_PROMPT"]
        if not isinstance(args.system_prompt_content, str) or not args.system_prompt_content.strip():
            raise ValueError(f"SYSTEM_PROMPT must be a non-empty string: {args.system_prompt_file}")
        print(
            f"Loaded SYSTEM_PROMPT from {args.system_prompt_file} "
            f"({len(args.system_prompt_content)} characters); it will be passed to every worker."
        )
    else:
        args.system_prompt_content = DEVELOPER_CONTENT
        print("Using default data_utils.DEVELOPER_CONTENT for every worker.")

    # Auto-detect number of available CUDA devices
    import torch

    if args.vllm_server_url:
        # Parse server URLs (support comma-separated list)
        server_urls = [url.strip() for url in args.vllm_server_url.split(',')]
        args.vllm_server_urls = server_urls  # Store as list

        # Using external vLLM server - create one worker per server URL
        NUM_WORKERS = len(server_urls)
        available_gpu_ids = []

        print(f"Using {NUM_WORKERS} external vLLM server(s):")
        for i, url in enumerate(server_urls):
            print(f"  - Server {i+1}: {url}")
        print(f"Launching {NUM_WORKERS} worker(s) (CPU-based, no local model loading)")
    else:
        # Using local vLLM engine - need GPU allocation
        # Get the list of available GPU IDs from CUDA_VISIBLE_DEVICES
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices is not None:
            # User specified GPU IDs
            available_gpu_ids = [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()]
            print(f"Using user-specified GPUs: {available_gpu_ids}")
        else:
            # Auto-detect all available GPUs
            num_gpus = torch.cuda.device_count()
            available_gpu_ids = list(range(num_gpus))
            print(f"Auto-detected {num_gpus} CUDA device(s)")

        if len(available_gpu_ids) == 0:
            raise RuntimeError("No CUDA devices found. Cannot proceed without GPUs.")

        # Calculate number of workers based on tensor_parallel_size
        tp_size = args.tensor_parallel_size
        if len(available_gpu_ids) % tp_size != 0:
            raise ValueError(
                f"Number of GPUs ({len(available_gpu_ids)}) must be divisible by "
                f"tensor_parallel_size ({tp_size})"
            )

        NUM_WORKERS = len(available_gpu_ids) // tp_size
        print(f"Launching {NUM_WORKERS} worker(s) with tensor_parallel_size={tp_size}")

    search_urls = [
        url.strip() for url in args.search_url.split(",") if url.strip()
    ]
    if len(search_urls) not in {1, NUM_WORKERS}:
        raise ValueError(
            "search_url must contain either one shared URL or one URL per worker; "
            f"got {len(search_urls)} URL(s) for {NUM_WORKERS} worker(s)."
        )
    args.search_urls = search_urls
    print(
        "Using "
        + ("one shared search service" if len(search_urls) == 1 else f"{len(search_urls)} worker-specific search services")
        + ":"
    )
    for url in search_urls:
        print(f"  - {url}")

    plugin_urls = [
        url.strip()
        for url in (args.grpo_plugin_url or "").split(",")
        if url.strip()
    ]
    if plugin_urls and len(plugin_urls) not in {1, NUM_WORKERS}:
        raise ValueError(
            "grpo_plugin_url must contain either one shared URL or one URL per worker; "
            f"got {len(plugin_urls)} URL(s) for {NUM_WORKERS} worker(s)."
        )
    args.grpo_plugin_urls = plugin_urls
    if plugin_urls:
        print(
            "Using "
            + ("one shared GRPO sidecar" if len(plugin_urls) == 1 else f"{len(plugin_urls)} worker-specific GRPO sidecars")
            + ":"
        )
        for url in plugin_urls:
            print(f"  - {url}")

    asag_urls = [
        url.strip()
        for url in (args.asag_attention_url or "").split(",")
        if url.strip()
    ]
    if asag_urls and len(asag_urls) not in {1, NUM_WORKERS}:
        raise ValueError(
            "asag_attention_url must contain either one shared URL or one URL "
            f"per worker; got {len(asag_urls)} URL(s) for {NUM_WORKERS} worker(s)."
        )
    args.asag_attention_urls = asag_urls
    if asag_urls:
        print(
            "Using "
            + (
                "one shared ASAG attention endpoint"
                if len(asag_urls) == 1
                else f"{len(asag_urls)} worker-specific ASAG attention endpoints"
            )
            + ":"
        )
        for url in asag_urls:
            print(f"  - {url}")

    if asag_urls and os.getenv("ASAG_ENABLED", "1").strip().lower() \
            not in {"0", "false", "no", "off"}:
        max_cached_sessions = int(
            os.getenv("ASAG_MAX_CACHED_SESSIONS", str(args.max_concurrency_per_worker))
        )
        sessions_per_endpoint = (
            NUM_WORKERS * args.max_concurrency_per_worker
            if len(asag_urls) == 1
            else args.max_concurrency_per_worker
        )
        if max_cached_sessions < sessions_per_endpoint:
            raise ValueError(
                "ASAG_MAX_CACHED_SESSIONS must cover every concurrently active "
                "question routed to each attention endpoint so its per-question "
                f"KV cache cannot be evicted: sessions={max_cached_sessions}, "
                f"required_per_endpoint={sessions_per_endpoint}."
            )
        print(
            f"ASAG session capacity validated: {max_cached_sessions} cached "
            f"sessions per endpoint for {sessions_per_endpoint} active questions."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Prevent accidentally running two generators against the same output_dir.
    # Multiple workers inside this process are fine; a second process would corrupt
    # the shard by duplicating qids because completed_qids is only scanned at startup.
    lock_path = os.path.join(args.output_dir, ".deploy_agent.lock")
    lock_file = open(lock_path, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.write(f"pid={os.getpid()} time={time.time()}\n")
        lock_file.flush()
    except BlockingIOError:
        raise RuntimeError(
            f"Another deploy_agent.py process is already using output_dir={args.output_dir}. "
            f"Stop it first, or choose a different output_dir. Lock: {lock_path}"
        )

    completed_qids = set()
    node_rank = int(os.getenv("RANK", 0))

    if args.fresh_run:
        print("Fresh run validated: parent process will not scan or skip old QIDs.")
    else:
        print("Scanning for completed tasks across all shards...")
        for shard_file in glob.glob(
            os.path.join(args.output_dir, "node_*_shard_*.jsonl")
        ):
            try:
                with open(shard_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if (
                                not args.resume_success_only
                                or record.get("status") == "success"
                            ):
                                completed_qids.add(str(record["qid"]))
                        except Exception:
                            continue
            except Exception as e:
                print(f"Warning: Could not read {shard_file}: {e}")

    if completed_qids:
        print(f"Found {len(completed_qids)} completed tasks from existing shards.")
        # Write to global completed file
        global_completed_path = os.path.join(args.output_dir, "completed_qids.txt")
        with open(global_completed_path, "w", encoding="utf-8") as f:
            for qid in sorted(completed_qids):
                f.write(f"{qid}\n")
        print(f"Wrote completed qids to {global_completed_path}")
    else:
        print("No completed tasks found. Starting fresh.")

    procs: List[mp.Process] = []
    for i in range(NUM_WORKERS):
        if args.vllm_server_url:
            # No GPU assignment needed for API mode
            worker_gpu_ids = []
            if hasattr(args, 'vllm_server_urls') and len(args.vllm_server_urls) > 1:
                server_url = args.vllm_server_urls[i % len(args.vllm_server_urls)]
                print(f"Worker {i} → Server: {server_url}")
            else:
                print(f"Worker {i} → Server: {args.vllm_server_url}")
        else:
            # Assign GPU IDs for this worker based on tensor parallelism
            tp_size = args.tensor_parallel_size
            worker_gpu_ids = available_gpu_ids[i * tp_size:(i + 1) * tp_size]
            print(f"Worker {i} assigned GPUs: {worker_gpu_ids}")

        p = mp.Process(
            target=worker_entry,
            args=(i, NUM_WORKERS, args, worker_gpu_ids)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=None)
        if p.exitcode != 0:
            print(f"Worker process {p.pid} exited with code {p.exitcode}")

    print("All workers finished. Script done.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
