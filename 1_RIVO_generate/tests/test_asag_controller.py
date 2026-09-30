import math
import unittest

from asag_controller import (
    ASAGConfig,
    ResearchSegment,
    RetrievalBoundaryASAG,
    boxed_answer_logprobs,
    boxed_answer_prefix,
    normalize_provisional_answer,
    token_confidence,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Client:
    def __init__(self, payloads):
        self.payloads = iter(payloads)

    async def post(self, *args, **kwargs):
        return _Response(next(self.payloads))

    async def aclose(self):
        return None


def _segment(turn, tool_name="open"):
    return ResearchSegment(
        turn=turn,
        reasoning=f"reasoning {turn}",
        query=f"query {turn}",
        evidence=f"evidence {turn}",
        tool_name=tool_name,
    )


def _context():
    return {
        "history_token_ids": [1, 2, 3],
        "probe_token_ids": [4],
        "previous_span": None,
        "current_span": {"start": 1, "end": 3},
    }


def _message_span(turn):
    return {"start": turn * 2, "end": turn * 2 + 2}


class ASAGControllerTests(unittest.IsolatedAsyncioTestCase):
    def test_token_confidence_is_arithmetic_probability_mean(self):
        value = token_confidence([math.log(0.9), math.log(0.5), None])
        self.assertAlmostEqual(value, 0.7)

    def test_probe_keeps_only_complete_boxed_answer(self):
        text, complete = boxed_answer_prefix("{\\text{New York}} More reasoning")
        self.assertTrue(complete)
        self.assertEqual(text, "{\\text{New York}}")
        selected = boxed_answer_logprobs(
            ["{", "30", "}", " But", " more"],
            [-0.1, -0.2, -0.3, -1.0, -2.0],
        )
        self.assertEqual(selected, [-0.1, -0.2, -0.3])

    def test_answer_normalization_ignores_wrappers_and_spacing(self):
        self.assertEqual(
            normalize_provisional_answer("<answer>\\boxed{ New York }</answer>"),
            normalize_provisional_answer("new-york"),
        )

    async def test_first_checkpoint_uses_paper_confidence_only_exit(self):
        controller = RetrievalBoundaryASAG(
            "question",
            "http://attention",
            ASAGConfig(confidence_threshold=0.95, entropy_delta_threshold=-0.1),
        )
        controller.client = _Client([])
        first = await controller.evaluate(
            _segment(1), "answer", 0.99, _context(), _message_span(1)
        )
        self.assertEqual(first["decision"], "stop")
        self.assertTrue(first["attention_probe_skipped"])

    async def test_later_stop_requires_confidence_and_entropy_convergence(self):
        controller = RetrievalBoundaryASAG(
            "question",
            "http://attention",
            ASAGConfig(confidence_threshold=0.95, entropy_delta_threshold=-0.1),
        )
        controller.client = _Client(
            [
                {"entropy": 100.0, "previous_attention": None, "current_attention": 0.2},
                {"entropy": 80.0, "previous_attention": 0.1, "current_attention": 0.2},
            ]
        )
        first = await controller.evaluate(
            _segment(1), "answer", 0.50, _context(), _message_span(1)
        )
        second = await controller.evaluate(
            _segment(2), "answer", 0.99, _context(), _message_span(2)
        )
        self.assertEqual(first["decision"], "continue")
        self.assertEqual(second["decision"], "stop")
        self.assertAlmostEqual(second["entropy_delta"], -0.2)

    async def test_low_confidence_and_previous_segment_attention_reroutes(self):
        controller = RetrievalBoundaryASAG(
            "question",
            "http://attention",
            ASAGConfig(confidence_threshold=0.95, max_reroutes=1),
        )
        controller.client = _Client(
            [
                {"entropy": 100.0, "previous_attention": None, "current_attention": 0.2},
                {"entropy": 101.0, "previous_attention": 0.3, "current_attention": 0.1},
            ]
        )
        await controller.evaluate(
            _segment(1), "answer", 0.5, _context(), _message_span(1)
        )
        result = await controller.evaluate(
            _segment(2), "answer", 0.5, _context(), _message_span(2)
        )
        self.assertEqual(result["decision"], "reroute")
        self.assertIn("different angle", result["reroute_prompt"])

    async def test_converged_low_confidence_answer_requests_verification(self):
        controller = RetrievalBoundaryASAG(
            "question", "http://attention", ASAGConfig(max_verifications=1)
        )
        controller.client = _Client(
            [
                {"entropy": 100.0, "previous_attention": None, "current_attention": 0.2},
                {"entropy": 80.0, "previous_attention": 0.1, "current_attention": 0.2},
            ]
        )
        await controller.evaluate(
            _segment(1), "answer", 0.5, _context(), _message_span(1)
        )
        result = await controller.evaluate(
            _segment(2), "answer", 0.5, _context(), _message_span(2)
        )
        self.assertEqual(result["decision"], "verify")
        self.assertIn("answer", result["verification_prompt"])

    async def test_repeated_stuck_state_forces_answer_after_reroute_limit(self):
        controller = RetrievalBoundaryASAG(
            "question", "http://attention", ASAGConfig(max_reroutes=1)
        )
        controller.client = _Client(
            [
                {"entropy": 100.0, "previous_attention": None, "current_attention": 0.2},
                {"entropy": 101.0, "previous_attention": 0.3, "current_attention": 0.1},
                {"entropy": 102.0, "previous_attention": 0.3, "current_attention": 0.1},
            ]
        )
        await controller.evaluate(
            _segment(1), "answer", 0.5, _context(), _message_span(1)
        )
        reroute = await controller.evaluate(
            _segment(2), "answer", 0.5, _context(), _message_span(2)
        )
        forced = await controller.evaluate(
            _segment(3), "answer", 0.5, _context(), _message_span(3)
        )
        self.assertEqual(reroute["decision"], "reroute")
        self.assertEqual(forced["decision"], "force_answer")

    async def test_paper_first_checkpoint_rule_is_tool_agnostic(self):
        controller = RetrievalBoundaryASAG("question", "http://attention")
        controller.client = _Client([])
        result = await controller.evaluate(
            _segment(1, "search"), "answer", 0.99, _context(), _message_span(1)
        )
        self.assertEqual(result["decision"], "stop")
        self.assertFalse(result["evidence_eligible_for_stop"])


if __name__ == "__main__":
    unittest.main()
