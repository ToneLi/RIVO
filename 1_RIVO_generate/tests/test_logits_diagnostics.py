import types
import unittest

import torch

from grpo_plugin_service import (
    GRPOPluginInferenceEngine,
    RerouteRequest,
    _semantic_vocab_mask,
)


class _FakeHost:
    def __call__(
        self,
        *,
        input_ids,
        past_key_values,
        use_cache,
        return_dict,
    ):
        del past_key_values, use_cache, return_dict
        batch, sequence = input_ids.shape
        logits = torch.tensor(
            [2.0, 1.0, 0.0, -1.0], dtype=torch.float32
        ).expand(batch, sequence, -1)
        return types.SimpleNamespace(logits=logits, past_key_values=None)


class _FakeBackbone:
    def __call__(
        self,
        *,
        input_ids,
        past_key_values,
        output_hidden_states,
        use_cache,
        return_dict,
    ):
        del past_key_values, output_hidden_states, use_cache, return_dict
        batch, sequence = input_ids.shape
        hidden = torch.ones(batch, sequence, 2)
        return types.SimpleNamespace(
            hidden_states=(hidden,), past_key_values=None
        )


class _FakeCorrectionHead:
    def __call__(self, hidden):
        shape = (*hidden.shape[:-1], 4)
        delta = torch.zeros(shape, dtype=torch.float32)
        delta[..., 1] = 2.0
        return delta


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return f"<token-{token_ids[0]}>"


def _engine(alpha):
    engine = object.__new__(GRPOPluginInferenceEngine)
    engine.device = torch.device("cpu")
    engine.host = _FakeHost()
    engine.host_tokenizer = _FakeTokenizer()
    engine.alpha = alpha
    engine.reroute_temperature = 1.0
    engine.reroute_top_p = 1.0
    engine.reroute_max_new_tokens = 1
    engine.delta_top_k = 4
    engine.semantic_vocab_mask = torch.ones(4, dtype=torch.bool)
    engine.reroute_seed = 17
    engine.policy = types.SimpleNamespace(
        valid_vocab_size=4,
        plugin=types.SimpleNamespace(
            backbone=_FakeBackbone(),
            correction_head=_FakeCorrectionHead(),
        ),
    )
    return engine


class LogitsDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_detect_a_top1_change(self):
        generated, diagnostics = _engine(1.0)._generate_reroute_with_diagnostics(
            torch.tensor([[1]], dtype=torch.long),
            torch.tensor([[1]], dtype=torch.long),
            eos_token_ids=[3],
        )

        self.assertEqual(generated.shape, (1, 1))
        token = diagnostics["per_token"][0]
        self.assertEqual(token["host_top1_id"], 0)
        self.assertEqual(token["corrected_top1_id"], 1)
        self.assertTrue(token["top1_changed"])
        self.assertGreater(token["delta_logits_l2"], 0.0)
        self.assertGreater(token["applied_delta_to_host_ratio"], 0.0)
        self.assertGreater(token["kl_host_to_corrected"], 0.0)
        self.assertEqual(diagnostics["summary"]["top1_change_rate"], 1.0)
        self.assertEqual(diagnostics["seed"], 17)

    def test_fixed_seed_replays_the_same_sampling_draw(self):
        first, _ = _engine(1.0)._generate_reroute_with_diagnostics(
            torch.tensor([[1]], dtype=torch.long),
            torch.tensor([[1]], dtype=torch.long),
            eos_token_ids=[3],
        )
        second, _ = _engine(1.0)._generate_reroute_with_diagnostics(
            torch.tensor([[1]], dtype=torch.long),
            torch.tensor([[1]], dtype=torch.long),
            eos_token_ids=[3],
        )

        self.assertTrue(torch.equal(first, second))

    def test_sampling_seed_override_is_reported(self):
        _, diagnostics = _engine(1.0)._generate_reroute_with_diagnostics(
            torch.tensor([[1]], dtype=torch.long),
            torch.tensor([[1]], dtype=torch.long),
            eos_token_ids=[3],
            sampling_seed=18,
        )

        self.assertEqual(diagnostics["seed"], 18)

    def test_alpha_zero_is_an_exact_host_control(self):
        _, diagnostics = _engine(0.0)._generate_reroute_with_diagnostics(
            torch.tensor([[1]], dtype=torch.long),
            torch.tensor([[1]], dtype=torch.long),
            eos_token_ids=[3],
        )

        token = diagnostics["per_token"][0]
        self.assertFalse(token["top1_changed"])
        self.assertEqual(token["applied_delta_logits_l2"], 0.0)
        self.assertEqual(token["applied_delta_to_host_ratio"], 0.0)
        self.assertAlmostEqual(token["kl_host_to_corrected"], 0.0, places=7)

    def test_excluded_nonsemantic_token_receives_no_delta(self):
        engine = _engine(1.0)
        engine.delta_top_k = 1
        engine.semantic_vocab_mask = torch.tensor(
            [True, False, True, True], dtype=torch.bool
        )

        _, diagnostics = engine._generate_reroute_with_diagnostics(
            torch.tensor([[1]], dtype=torch.long),
            torch.tensor([[1]], dtype=torch.long),
            eos_token_ids=[3],
        )

        token = diagnostics["per_token"][0]
        self.assertFalse(token["top1_changed"])
        self.assertEqual(token["applied_delta_logits_l2"], 0.0)
        self.assertEqual(token["delta_top_k"], 1)



class _FakeHintRuntime:
    def __init__(self):
        self.calls = []

    def generate_hint(self, question, history, **kwargs):
        self.calls.append((question, history, kwargs))
        segments = [
            types.SimpleNamespace(target_ids=[1, 2]),
            types.SimpleNamespace(target_ids=[3]),
            types.SimpleNamespace(target_ids=[4, 5]),
        ]
        return types.SimpleNamespace(
            fields=("WLUJ", "licensed municipality", "incorporation date"),
            text="WLUJ ; licensed municipality ; incorporation date",
            segments=segments,
            semantic_tokens=5,
        )


class Step70HintRuntimeTests(unittest.TestCase):
    def test_reroute_uses_training_identical_hint_fields_as_query(self):
        engine = object.__new__(GRPOPluginInferenceEngine)
        engine.device = torch.device("cpu")
        engine.reroute_seed = 17
        engine.reroute_format_attempts = 1
        engine.hint_runtime = _FakeHintRuntime()
        engine.hint_slot_token_budgets = (16, 12, 12)
        engine.hint_temperature = 1.0
        engine.hint_top_p = 0.95
        engine.hint_query_max_words = 24
        engine.delta_top_k = 128
        engine.alpha = 20.0

        result = engine.reroute(
            RerouteRequest(question="question", history="failed history")
        )

        self.assertEqual(
            result["query"], "WLUJ licensed municipality incorporation date"
        )
        self.assertEqual(
            result["fields"],
            ["WLUJ", "licensed municipality", "incorporation date"],
        )
        self.assertEqual(result["token_ids"], [1, 2, 3, 4, 5])
        self.assertTrue(
            result["logits_diagnostics"]["corrected_logits_applied"]
        )
        _, _, kwargs = engine.hint_runtime.calls[0]
        self.assertEqual(kwargs["slot_token_budgets"], (16, 12, 12))
        self.assertEqual(kwargs["correction_topk"], 128)



class _MaskTokenizer:
    all_special_ids = [0]

    _tokens = {
        0: "<special>",
        1: " the",
        2: " WLUJ",
        3: ";",
        4: " licensed",
        5: " asks",
        6: " relation",
        7: " 2025",
        8: "South-West",
        9: "{",
    }

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return self._tokens[token_ids[0]]


class SemanticMaskTests(unittest.TestCase):
    def test_mask_keeps_content_words_and_numbers_only(self):
        mask = _semantic_vocab_mask(_MaskTokenizer(), 10)
        kept = {index for index, value in enumerate(mask.tolist()) if value}
        self.assertEqual(kept, {2, 4, 7, 8})


if __name__ == "__main__":
    unittest.main()
