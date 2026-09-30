# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
import asyncio
import json

import pytest

from verl.experimental.agent_loop.tool_agent_loop import AgentData, ToolAgentLoop, _resolve_tool_name
from verl.experimental.agent_loop.tool_parser import INVALID_TOOL_CALL_NAME, FunctionCall, HermesToolParser


class _TextTokenizer:
    def decode(self, value, *args, **kwargs):
        return value


@pytest.mark.parametrize(
    ("body", "expected_name", "expected_query"),
    [
        ('{"name":"search","arguments":{"query":"valid"}}', "search", "valid"),
        ('{"name":"search","arguments":{"query":"Gaza citrus production"});', "search", "Gaza citrus production"),
        ('{"name":"open","arguments":{"cursor":0,"id":2}', "open", None),
        ('{"name":"search","arguments":{"query":"founder\\\'s name"}}', "search", "founder's name"),
        ('{"name":"search","arguments":{"query":"year \\d+"}}', "search", r"year \d+"),
        ('{"name":"search","arguments":{"query":"line one\nline two"}}', "search", "line one\nline two"),
    ],
)
def test_hermes_parser_repairs_unambiguous_json(body, expected_name, expected_query):
    parser = HermesToolParser(_TextTokenizer())
    _, calls = asyncio.run(parser.extract_tool_calls(f"reasoning<tool_call>{body}</tool_call>"))

    assert len(calls) == 1
    assert calls[0].name == expected_name
    arguments = json.loads(calls[0].arguments)
    if expected_query is not None:
        assert arguments["query"] == expected_query


@pytest.mark.parametrize(
    "text",
    [
        "reasoning<tool_call></tool_call>",
        '<tool_call>{"name":"search","arguments":{"query":"unfinished',
        '<tool_call>{"name":"search","arguments":{"query":"x"} garbage',
    ],
)
def test_hermes_parser_preserves_invalid_call_for_retry(text):
    parser = HermesToolParser(_TextTokenizer())
    _, calls = asyncio.run(parser.extract_tool_calls(text))

    assert len(calls) == 1
    assert calls[0].name == INVALID_TOOL_CALL_NAME


def _agent_data():
    return AgentData(
        messages=[],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="test",
        tools_kwargs={},
    )


def test_invalid_and_unknown_tools_return_retry_observations():
    loop = object.__new__(ToolAgentLoop)
    loop.tools = {"browser.search": object(), "browser.open": object(), "browser.find": object()}
    data = _agent_data()

    async def invoke():
        invalid = await loop._call_tool(
            FunctionCall(name=INVALID_TOOL_CALL_NAME, arguments='{"error":"bad JSON"}'), {}, data
        )
        visit = await loop._call_tool(FunctionCall(name="visit", arguments='{"id":1}'), {}, data)
        return invalid, visit

    (invalid_response, _, invalid_metadata), (visit_response, _, visit_metadata) = asyncio.run(invoke())

    assert "Retry" in invalid_response.text
    assert invalid_metadata["status"] == "invalid_tool_call"
    assert "browser.open (not visit)" in visit_response.text
    assert visit_metadata == {"status": "unknown_tool", "requested_tool": "visit"}
    assert data.metrics == {"invalid_tool_calls": 1, "unknown_tool_calls": 1}
    assert _resolve_tool_name("visit", loop.tools) == "visit"
