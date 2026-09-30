# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ast
import json
import logging
import os
import warnings
from abc import ABC, abstractmethod

import regex
from pydantic import BaseModel

from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# A malformed call must remain a tool turn. Dropping it makes the agent loop
# interpret the response as a final answer and terminates the trajectory.
INVALID_TOOL_CALL_NAME = "__invalid_tool_call__"


def _close_unterminated_json(text: str) -> str | None:
    """Append only unambiguous missing JSON container delimiters."""
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    closing = {"{": "}", "[": "]"}

    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in closing:
            stack.append(char)
        elif char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return None
            stack.pop()

    # An unfinished quoted value is ambiguous and should be retried by the
    # model rather than guessed by the parser.
    if in_string:
        return None
    return text + "".join(closing[char] for char in reversed(stack))


def _repair_invalid_json_escapes(text: str) -> str:
    """Repair common model-produced escapes that strict JSON rejects."""
    text = text.replace("\\'", "'")
    # Preserve valid JSON escapes; make e.g. \d a literal backslash plus d.
    return regex.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _tool_call_candidates(raw: str) -> list[str]:
    """Generate conservative JSON candidates, ordered from strict to repaired."""
    candidates: list[str] = []

    def add(candidate: str | None) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    base = raw.strip()
    add(base)

    # Models occasionally emit `});` around an otherwise complete object.
    no_semicolon = base[:-1].rstrip() if base.endswith(";") else base
    add(no_semicolon)
    if no_semicolon.endswith(")") and no_semicolon.startswith("{"):
        add(no_semicolon[:-1].rstrip())

    for candidate in list(candidates):
        add(_repair_invalid_json_escapes(candidate))
    for candidate in list(candidates):
        add(_close_unterminated_json(candidate))
    return candidates


def parse_json_like_tool_call(raw: str) -> tuple[dict, bool]:
    """Parse a tool call, repairing only unambiguous model formatting errors."""
    if not raw.strip():
        raise ValueError("tool call body is empty")

    errors: list[str] = []
    for candidate_index, candidate in enumerate(_tool_call_candidates(raw)):
        parsers = (
            lambda value: json.loads(value),
            lambda value: json.loads(value, strict=False),
            ast.literal_eval,
        )
        for parser_index, parser in enumerate(parsers):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    parsed = parser(candidate)
            except (TypeError, ValueError, SyntaxError, json.JSONDecodeError) as exc:
                errors.append(str(exc))
                continue

            if not isinstance(parsed, dict):
                errors.append("tool call must be a JSON object")
                continue
            name = parsed.get("name")
            arguments = parsed.get("arguments")
            if not isinstance(name, str) or not name.strip():
                errors.append("tool call 'name' must be a non-empty string")
                continue
            if not isinstance(arguments, dict):
                errors.append("tool call 'arguments' must be a JSON object")
                continue
            repaired = candidate_index > 0 or parser_index > 0
            return {"name": name.strip(), "arguments": arguments}, repaired

    detail = errors[-1] if errors else "unknown parse error"
    raise ValueError(f"invalid tool call JSON: {detail}")


class FunctionCall(BaseModel):
    arguments: str
    """
    The arguments to call the function with, as generated by the model in JSON
    format. Note that the model does not always generate valid JSON, and may
    hallucinate parameters not defined by your function schema. Validate the
    arguments in your code before calling your function.
    """

    name: str
    """The name of the function to call."""
    was_repaired: bool = False


class ToolParser(ABC):
    _registry: dict[str, type["ToolParser"]] = {}

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        """Extract tool calls from the responses.

        Args:
            responses_ids (List[int]): The ids of the responses.

        Returns:
            Tuple[str, List[FunctionCall]]: Content and extracted tool calls.
        """
        raise NotImplementedError

    @classmethod
    def get_tool_parser(cls, name: str, tokenizer):
        if name not in cls._registry:
            raise ValueError(f"Unknown tool parser: {name}")
        return cls._registry[name](tokenizer)

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: type[ToolParser]) -> type[ToolParser]:
            cls._registry[name] = subclass
            return subclass

        return decorator


@ToolParser.register("hermes")
class HermesToolParser(ToolParser):
    """Adapted from https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py"""

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = regex.compile(r"<tool_call>(.*?)</tool_call>", regex.DOTALL)

    @rollout_trace_op
    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)
        if self.tool_call_start_token not in text:
            return text, []

        matches = self.tool_call_regex.findall(text)
        # Preserve an unterminated trailing call as an invalid tool turn. If it
        # were ignored, DeepResearchAgentLoop would terminate the trajectory.
        last_start = text.rfind(self.tool_call_start_token)
        last_end = text.rfind(self.tool_call_end_token)
        has_unterminated_call = last_start > last_end
        if has_unterminated_call:
            matches.append(text[last_start + len(self.tool_call_start_token) :])

        function_calls = []
        for match in matches:
            try:
                function_call, repaired = parse_json_like_tool_call(match)
                name, arguments = function_call["name"], function_call["arguments"]
                function_calls.append(
                    FunctionCall(
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                        was_repaired=repaired,
                    )
                )
                if repaired:
                    logger.info("Repaired malformed tool call JSON for tool %r", name)
            except Exception as e:
                # Keep the state machine alive and let the model correct its
                # own format after receiving an explicit tool observation.
                function_calls.append(
                    FunctionCall(
                        name=INVALID_TOOL_CALL_NAME,
                        arguments=json.dumps({"error": str(e)}, ensure_ascii=False),
                    )
                )

        # Remaining text excludes tool-call tokens.
        content = self.tool_call_regex.sub("", text)
        if has_unterminated_call:
            content = content[: content.rfind(self.tool_call_start_token)]

        return content, function_calls


@ToolParser.register("gpt-oss")
class GptOssToolParser(ToolParser):
    """
    Tool parser for gpt-oss model.
    Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/gpt_oss_detector.py

    Args:
        tokenizer: The tokenizer to use.
    """

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        # check https://cookbook.openai.com/articles/openai-harmony for more details.
        self.cot_pattern = regex.compile(
            r"<\|start\|>assistant<\|channel\|>analysis<\|message\|>.*?<\|end\|>", regex.DOTALL
        )
        # <|start|>assistant may be pre-appended in prompts, so we need to remove it.
        self.partial_cot_pattern = regex.compile(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", regex.DOTALL)
        self.tool_call_pattern = regex.compile(
            r"<\|start\|>assistant<\|channel\|>[^<]* to=functions\.([^<]+) "
            r"<\|constrain\|>json<\|message\|>(.*?)<\|call\|>",
            regex.DOTALL,
        )

    @rollout_trace_op
    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        # We need to keep special tokens for gpt-oss model for better tool call extraction.
        text = await loop.run_in_executor(None, lambda: self.tokenizer.decode(responses_ids, skip_special_tokens=False))
        # Need to remove padding tokens for better tool call extraction.
        text = text.replace(self.tokenizer.pad_token, "")
        # Need to reomve COT since COT may contain tool call tokens.But they are not valid tool calls.
        text = regex.sub(self.cot_pattern, "", text)
        text = regex.sub(self.partial_cot_pattern, "", text)

        # check if there are tool calls in the text by re.findall
        matches = regex.findall(self.tool_call_pattern, text)
        if not matches:
            return text, []

        function_calls = []
        for match in matches:
            try:
                name, arguments = match[0], match[1]
                # don't check if arguments is valid JSON and leave it to client
                function_calls.append(FunctionCall(name=name, arguments=arguments))
            except Exception as e:
                logger.error(f"Failed to decode tool call: {e}")

        # remaing text exclude tool call tokens
        content = regex.sub(self.tool_call_pattern, "", text)

        return content, function_calls
