import copy
import types
import unittest

from deploy_agent import _maybe_promote_joint_controller_reroute


class FakeASAG:
    config = types.SimpleNamespace(
        confidence_threshold=0.95,
        max_reroutes=1,
        reroute_prompt="joint fallback",
    )

    def __init__(self):
        self.reroute_count = 0


class FakePluginClient:
    def __init__(self, reroute_probability=0.8, allowed=True, error=None):
        self.reroute_probability = reroute_probability
        self.allowed = allowed
        self.error = error
        self.calls = []

    async def control(
        self,
        question,
        messages,
        *,
        round_num,
        last_reroute_round,
    ):
        self.calls.append(
            (
                question,
                copy.deepcopy(messages),
                round_num,
                last_reroute_round,
            )
        )
        if self.error is not None:
            raise self.error
        return {
            "action": (
                "REROUTE" if self.reroute_probability >= 0.5 else "CONTINUE"
            ),
            "action_id": 2,
            "probabilities": {
                "CONTINUE": 1.0 - self.reroute_probability,
                "STOP": 0.0,
                "REROUTE": self.reroute_probability,
            },
            "allowed": {
                "CONTINUE": True,
                "STOP": False,
                "REROUTE": self.allowed,
            },
        }


class JointControllerTests(unittest.IsolatedAsyncioTestCase):
    def base_arguments(self, checkpoint, plugin):
        return {
            "asag": FakeASAG(),
            "checkpoint": checkpoint,
            "question": "question",
            "messages": [{"role": "user", "content": "history"}],
            "plugin_client": plugin,
            "round_num": 12,
            "last_reroute_round": None,
            "controller_reroute_count": 0,
            "enabled": True,
            "reroute_threshold": 0.70,
            "max_reroutes": 1,
            "verbose": False,
        }

    async def test_low_confidence_controller_promotes_continue_to_reroute(self):
        checkpoint = {
            "checkpoint": 3,
            "decision": "continue",
            "reason": "evidence_still_advancing_or_confidence_low",
            "confidence": 0.6,
        }
        plugin = FakePluginClient(reroute_probability=0.8)

        triggered = await _maybe_promote_joint_controller_reroute(
            **self.base_arguments(checkpoint, plugin)
        )

        self.assertTrue(triggered)
        self.assertEqual(checkpoint["decision"], "reroute")
        self.assertEqual(checkpoint["asag_decision"], "continue")
        self.assertEqual(checkpoint["reroute_trigger"], "joint_controller_reroute")
        self.assertEqual(checkpoint["reroute_prompt"], "joint fallback")
        self.assertEqual(checkpoint["reroute_count"], 1)
        self.assertTrue(checkpoint["joint_controller"]["triggered"])
        self.assertEqual(len(plugin.calls), 1)

    async def test_probability_below_threshold_keeps_continue(self):
        checkpoint = {
            "checkpoint": 3,
            "decision": "continue",
            "reason": "evidence_still_advancing_or_confidence_low",
            "confidence": 0.6,
        }
        plugin = FakePluginClient(reroute_probability=0.69)

        triggered = await _maybe_promote_joint_controller_reroute(
            **self.base_arguments(checkpoint, plugin)
        )

        self.assertFalse(triggered)
        self.assertEqual(checkpoint["decision"], "continue")
        self.assertEqual(
            checkpoint["joint_controller"]["reason"],
            "controller_probability_below_threshold",
        )

    async def test_high_confidence_asag_guard_skips_controller(self):
        checkpoint = {
            "checkpoint": 3,
            "decision": "continue",
            "reason": "confidence_high_but_attention_not_converged",
            "confidence": 0.96,
        }
        plugin = FakePluginClient(reroute_probability=0.99)

        triggered = await _maybe_promote_joint_controller_reroute(
            **self.base_arguments(checkpoint, plugin)
        )

        self.assertFalse(triggered)
        self.assertEqual(checkpoint["decision"], "continue")
        self.assertEqual(
            checkpoint["joint_controller"]["reason"],
            "asag_high_confidence_guard",
        )
        self.assertEqual(plugin.calls, [])

    async def test_asag_and_controller_share_the_reroute_budget(self):
        checkpoint = {
            "checkpoint": 3,
            "decision": "continue",
            "reason": "evidence_still_advancing_or_confidence_low",
            "confidence": 0.6,
        }
        plugin = FakePluginClient(reroute_probability=0.99)
        arguments = self.base_arguments(checkpoint, plugin)
        arguments["asag"].reroute_count = 1

        triggered = await _maybe_promote_joint_controller_reroute(**arguments)

        self.assertFalse(triggered)
        self.assertEqual(checkpoint["decision"], "continue")
        self.assertEqual(
            checkpoint["joint_controller"]["reason"],
            "shared_asag_controller_reroute_limit_reached",
        )
        self.assertEqual(plugin.calls, [])

    async def test_asag_stop_has_priority(self):
        checkpoint = {
            "checkpoint": 3,
            "decision": "stop",
            "reason": "paper_high_confidence_with_entropy_convergence",
            "confidence": 0.99,
        }
        plugin = FakePluginClient(reroute_probability=0.99)

        triggered = await _maybe_promote_joint_controller_reroute(
            **self.base_arguments(checkpoint, plugin)
        )

        self.assertFalse(triggered)
        self.assertEqual(checkpoint["decision"], "stop")
        self.assertNotIn("joint_controller", checkpoint)
        self.assertEqual(plugin.calls, [])

    async def test_controller_failure_fails_closed_to_asag_continue(self):
        checkpoint = {
            "checkpoint": 3,
            "decision": "continue",
            "reason": "evidence_still_advancing_or_confidence_low",
            "confidence": 0.6,
        }
        plugin = FakePluginClient(error=RuntimeError("unavailable"))

        triggered = await _maybe_promote_joint_controller_reroute(
            **self.base_arguments(checkpoint, plugin)
        )

        self.assertFalse(triggered)
        self.assertEqual(checkpoint["decision"], "continue")
        self.assertIn("unavailable", checkpoint["joint_controller"]["error"])


if __name__ == "__main__":
    unittest.main()
