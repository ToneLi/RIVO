import unittest

from deploy_agent import (
    _browser_tool_name,
    _extend_round_limit_after_reroute,
    _is_terminal_assistant_response,
    _retrieval_result_is_error,
    _should_flush_pending_checkpoint,
)


class AgentTerminationTests(unittest.TestCase):
    def test_tool_call_continues_research(self):
        calls = [{"function": {"name": "browser.search"}}]
        self.assertFalse(_is_terminal_assistant_response("possible answer", calls))

    def test_nonempty_content_without_tool_call_is_final(self):
        self.assertTrue(_is_terminal_assistant_response("Electra", None))

    def test_empty_content_without_tool_call_is_not_final(self):
        self.assertFalse(_is_terminal_assistant_response("   ", None))

    def test_browser_tool_name_removes_runtime_prefix(self):
        self.assertEqual(_browser_tool_name("browser.open"), "open")
        self.assertEqual(_browser_tool_name("find"), "find")

    def test_search_waits_for_open_or_find(self):
        self.assertFalse(_should_flush_pending_checkpoint("search", "open"))
        self.assertFalse(_should_flush_pending_checkpoint("search", "find"))
        self.assertTrue(_should_flush_pending_checkpoint("search", "search"))

    def test_open_waits_for_find_and_flushes_before_new_search(self):
        self.assertFalse(_should_flush_pending_checkpoint("open", "find"))
        self.assertFalse(_should_flush_pending_checkpoint("open", "open"))
        self.assertTrue(_should_flush_pending_checkpoint("open", "search"))

    def test_non_retrieval_action_closes_pending_chain(self):
        self.assertTrue(_should_flush_pending_checkpoint("search", "unknown"))
    def test_search_service_error_is_not_treated_as_evidence(self):
        self.assertTrue(
            _retrieval_result_is_error(
                "Error during search for query: All strings must be XML compatible"
            )
        )

    def test_successful_search_result_is_not_an_error(self):
        self.assertFalse(_retrieval_result_is_error("[1] useful search evidence"))

    def test_late_reroute_receives_completion_budget_with_cap(self):
        self.assertEqual(
            _extend_round_limit_after_reroute(
                base_max_rounds=80,
                current_round_limit=80,
                reroute_round=72,
                min_post_reroute_rounds=20,
                max_extra_rounds=20,
            ),
            92,
        )
        self.assertEqual(
            _extend_round_limit_after_reroute(
                base_max_rounds=80,
                current_round_limit=80,
                reroute_round=95,
                min_post_reroute_rounds=20,
                max_extra_rounds=20,
            ),
            100,
        )



if __name__ == "__main__":
    unittest.main()
