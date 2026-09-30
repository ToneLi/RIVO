import copy
import os
import unittest
from unittest.mock import patch

from deploy_agent import (
    GRPO_ROUTE_HINT_INSTRUCTION,
    GRPO_ROUTE_HINT_PREFILL,
    _apply_grpo_corrected_reroute,
    _validated_route_hint,
)
from grpo_plugin_service import (
    ROUTE_HINT_PREFILL,
    _clean_route_hint,
    reroute_system_prompt,
    reroute_user_prompt,
)


VALID_HINT = "WLUJ ; licensed municipality ; incorporation date"


class FakeTokenizer:
    def __init__(self):
        self.encoded_texts = []

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=False,
        add_generation_prompt=True,
    ):
        if messages and messages[-1].get("tool_calls"):
            query = messages[-1]["tool_calls"][0]["function"]["arguments"]["query"]
            return (
                'prompt<tool_call>{"name":"browser.search",'
                f'"arguments":{{"query":"{query}"}}}}</tool_call>'
            )
        return "prompt"

    def encode(self, text, add_special_tokens=False):
        self.encoded_texts.append(text)
        return [1, 2, 3]

    def decode(self, tokens, skip_special_tokens=False):
        if tokens == [9]:
            return 'host generated query"'
        return "decoded"


class FakeGenerator:
    def __init__(self):
        self.calls = []
        self.tokenizer = FakeTokenizer()

    async def generate(self, tokens, **kwargs):
        self.calls.append((list(tokens), copy.deepcopy(kwargs)))
        yield 9


class FakeBrowserPool:
    def __init__(self, result="search evidence"):
        self.result = result
        self.calls = []

    async def call_tool(self, qid, name, arguments):
        self.calls.append((qid, name, copy.deepcopy(arguments)))
        return self.result


class FakePluginClient:
    def __init__(self, result=None, error=None):
        self.result = result or {
            "route_hint": VALID_HINT,
            "hint": VALID_HINT,
            "fields": ["WLUJ", "licensed municipality", "incorporation date"],
            "query": "WLUJ licensed municipality incorporation date",
            "token_ids": [4, 5],
            "alpha": 20.0,
            "reroute_seed": 17,
            "logits_diagnostics": {
                "summary": {"top1_change_rate": 0.5}
            },
        }
        self.error = error
        self.calls = []

    async def reroute(self, question, messages):
        self.calls.append((question, copy.deepcopy(messages)))
        if self.error is not None:
            raise self.error
        return self.result


class RouteHintFormatTests(unittest.TestCase):
    def test_prompts_require_exact_three_slot_output(self):
        prompts = (
            GRPO_ROUTE_HINT_INSTRUCTION,
            reroute_system_prompt(),
            reroute_user_prompt("question", "history"),
        )
        for prompt in prompts:
            self.assertIn("ENTITY ; RELATION ; DISAMBIGUATOR", prompt)
            self.assertIn("semicolon", prompt)
        self.assertEqual(GRPO_ROUTE_HINT_PREFILL, "Search direction: ")
        self.assertEqual(ROUTE_HINT_PREFILL, GRPO_ROUTE_HINT_PREFILL)

    def test_valid_hint_is_normalized(self):
        expected = "WLUJ ; licensed municipality ; incorporation date"
        valid_variants = (
            " WLUJ; licensed municipality; incorporation date ",
            'Search direction: "WLUJ" ; "licensed municipality" ; "incorporation date"',
            "ENTITY: WLUJ ; RELATION: licensed municipality ; DISAMBIGUATOR: incorporation date",
            "ENTITY: WLUJ\nRELATION: licensed municipality\nDISAMBIGUATOR: incorporation date",
            "[WLUJ] ; {licensed municipality} ; (incorporation date)",
        )
        for hint in valid_variants:
            with self.subTest(hint=hint):
                self.assertEqual(
                    _validated_route_hint(
                        hint,
                        "When was the town licensed to WLUJ incorporated?",
                    ),
                    expected,
                )
                self.assertEqual(
                    _clean_route_hint(
                        hint,
                        "When was the town licensed to WLUJ incorporated?",
                    ),
                    expected,
                )

    def test_invalid_hints_are_rejected(self):
        invalid = (
            'The user asks: "When was WLUJ licensed?"',
            "WLUJ incorporation date",
            "WLUJ ; licensed municipality",
            "WLUJ ; licensed municipality ;",
        )
        for hint in invalid:
            with self.subTest(hint=hint):
                with self.assertRaises(ValueError):
                    _validated_route_hint(hint, "question")
                with self.assertRaises(ValueError):
                    _clean_route_hint(hint, "question")


class ASAGGRPORerouteTests(unittest.IsolatedAsyncioTestCase):
    def base_arguments(self):
        return {
            "question": "question",
            "qid": "qid",
            "round_num": 12,
            "checkpoint": {"checkpoint": 3, "reroute_prompt": "ASAG fallback"},
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
            "tools": [{"type": "function", "function": {"name": "browser.search"}}],
            "generator": FakeGenerator(),
            "grpo_reroute_trace": [],
            "verbose": False,
        }

    async def test_corrected_hint_is_rewritten_by_host_then_searched(self):
        arguments = self.base_arguments()
        browser = FakeBrowserPool()
        plugin = FakePluginClient()
        arguments.update(browser_pool=browser, plugin_client=plugin)

        pending = await _apply_grpo_corrected_reroute(**arguments)

        self.assertIsNotNone(pending)
        self.assertEqual(pending["segment"].query, "host generated query")
        self.assertEqual(
            browser.calls,
            [("qid", "search", {"query": "host generated query"})],
        )
        self.assertEqual(arguments["messages"][-2]["role"], "assistant")
        self.assertEqual(
            arguments["messages"][-2]["retrieval_control"],
            "asag_grpo_logits_reroute",
        )
        self.assertEqual(arguments["messages"][-1]["role"], "tool")
        self.assertEqual(len(arguments["generator"].calls), 1)

        trace = arguments["grpo_reroute_trace"][0]
        self.assertTrue(trace["logits_applied"])
        self.assertTrue(trace["search_success"])
        self.assertFalse(trace["fallback_to_asag_prompt"])
        self.assertEqual(trace["policy"], "grpo_e2e_three_field_hint_host_rewrite")
        self.assertEqual(trace["query_source"], "host_rewrite_from_corrected_hint")
        self.assertEqual(trace["query"], "host generated query")
        self.assertEqual(trace["plugin_query"], "WLUJ licensed municipality incorporation date")
        self.assertEqual(trace["alpha"], 20.0)
        self.assertEqual(trace["route_hint"], VALID_HINT)
        self.assertEqual(
            trace["hint_fields"],
            ["WLUJ", "licensed municipality", "incorporation date"],
        )
        self.assertEqual(trace["hint_seed"], 17)
        self.assertEqual(
            trace["logits_diagnostics"]["summary"]["top1_change_rate"],
            0.5,
        )
        self.assertIs(arguments["checkpoint"]["grpo_reroute"], trace)

    async def test_invalid_hint_falls_back_without_host_query_or_search(self):
        arguments = self.base_arguments()
        browser = FakeBrowserPool()
        plugin = FakePluginClient(
            result={
                "route_hint": 'The user asks: "question"',
                "token_ids": [4, 5],
            }
        )
        arguments.update(browser_pool=browser, plugin_client=plugin)

        pending = await _apply_grpo_corrected_reroute(**arguments)

        self.assertIsNone(pending)
        self.assertEqual(arguments["generator"].calls, [])
        self.assertEqual(browser.calls, [])
        self.assertEqual(
            arguments["messages"][-1],
            {"role": "user", "content": "ASAG fallback"},
        )
        trace = arguments["grpo_reroute_trace"][0]
        self.assertFalse(trace["logits_applied"])
        self.assertTrue(trace["fallback_to_asag_prompt"])
        self.assertIn("forbidden", trace["error"].lower())

    async def test_plugin_failure_falls_back_to_original_asag_prompt(self):
        arguments = self.base_arguments()
        arguments.update(
            browser_pool=FakeBrowserPool(),
            plugin_client=FakePluginClient(error=RuntimeError("sidecar unavailable")),
        )

        pending = await _apply_grpo_corrected_reroute(**arguments)

        self.assertIsNone(pending)
        self.assertEqual(
            arguments["messages"][-1],
            {"role": "user", "content": "ASAG fallback"},
        )
        trace = arguments["grpo_reroute_trace"][0]
        self.assertFalse(trace["logits_applied"])
        self.assertTrue(trace["fallback_to_asag_prompt"])
        self.assertIn("sidecar unavailable", trace["error"])

    async def test_search_failure_is_recorded_before_prompt_fallback(self):
        arguments = self.base_arguments()
        arguments.update(
            browser_pool=FakeBrowserPool("Error executing browser.search: failed"),
            plugin_client=FakePluginClient(),
        )

        pending = await _apply_grpo_corrected_reroute(**arguments)

        self.assertIsNone(pending)
        self.assertEqual(arguments["messages"][-2]["role"], "tool")
        self.assertEqual(arguments["messages"][-1]["content"], "ASAG fallback")
        trace = arguments["grpo_reroute_trace"][0]
        self.assertTrue(trace["logits_applied"])
        self.assertFalse(trace["search_success"])
        self.assertTrue(trace["fallback_to_asag_prompt"])

    async def test_search_service_error_retries_then_falls_back(self):
        arguments = self.base_arguments()
        browser = FakeBrowserPool(
            "Error during search for query: All strings must be XML compatible"
        )
        arguments.update(browser_pool=browser, plugin_client=FakePluginClient())

        with patch.dict(os.environ, {"GRPO_REROUTE_SEARCH_ATTEMPTS": "2"}):
            pending = await _apply_grpo_corrected_reroute(**arguments)

        self.assertIsNone(pending)
        self.assertEqual(len(browser.calls), 2)
        trace = arguments["grpo_reroute_trace"][0]
        self.assertEqual(trace["search_attempts"], 2)
        self.assertEqual(len(trace["search_errors"]), 2)
        self.assertFalse(trace["search_success"])
        self.assertTrue(trace["fallback_to_asag_prompt"])


if __name__ == "__main__":
    unittest.main()
