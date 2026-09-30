import threading
import types
import unittest
from collections import OrderedDict

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from asag_attention import LocalAttentionProbe


class _FakeModel:
    def __init__(self):
        self.calls = []
        self.attention_implementations = []

    def set_attn_implementation(self, implementation):
        self.attention_implementations.append(implementation)

    def __call__(self, input_ids, **kwargs):
        self.calls.append((input_ids.tolist(), kwargs))
        if len(self.calls) == 1:
            return types.SimpleNamespace(past_key_values="prefilled", attentions=None)
        query_length = input_ids.shape[1]
        key_length = 8
        attention = torch.full((1, 2, query_length, key_length), 1.0 / key_length)
        return types.SimpleNamespace(attentions=(attention,) * 4)


class ASAGAttentionTests(unittest.TestCase):
    def test_existing_host_reuse_matches_identical_isolated_model(self):
        torch.manual_seed(7)
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            use_cache=True,
        )
        shared_model = Qwen3ForCausalLM(config).eval()
        isolated_model = Qwen3ForCausalLM(config).eval()
        isolated_model.load_state_dict(shared_model.state_dict())
        shared_lock = threading.RLock()

        shared_probe = LocalAttentionProbe.from_existing_model(
            shared_model,
            device="cpu",
            lock=shared_lock,
        )
        isolated_probe = LocalAttentionProbe.from_existing_model(
            isolated_model,
            device="cpu",
        )

        self.assertIs(shared_probe.model, shared_model)
        self.assertIs(shared_probe.lock, shared_lock)
        arguments = {
            "history_token_ids": list(range(20)),
            "probe_token_ids": [20, 21],
            "previous_span": {"start": 4, "end": 8},
            "current_span": {"start": 16, "end": 20},
        }
        shared_metrics = shared_probe.analyze(**arguments)
        isolated_metrics = isolated_probe.analyze(**arguments)

        for metric in ("entropy", "previous_attention", "current_attention"):
            self.assertAlmostEqual(shared_metrics[metric], isolated_metrics[metric], places=6)
        for metric in ("layers_used", "heads", "sequence_tokens", "prefill_tokens"):
            self.assertEqual(shared_metrics[metric], isolated_metrics[metric])

    def test_long_history_is_prefilled_and_only_window_returns_attention(self):
        probe = LocalAttentionProbe.__new__(LocalAttentionProbe)
        probe.model = _FakeModel()
        probe.input_device = torch.device("cpu")
        probe.lock = threading.Lock()
        probe.monitor_history_tokens = 2
        probe.attention_window_tokens = 3
        probe.max_probe_tokens = 4
        probe.max_history_tokens = 20

        result = probe.analyze(
            history_token_ids=[0, 1, 2, 3, 4, 5, 6],
            probe_token_ids=[7],
            previous_span={"start": 1, "end": 3},
            current_span={"start": 5, "end": 7},
        )

        self.assertEqual(probe.model.calls[0][0], [[0, 1, 2, 3, 4]])
        self.assertEqual(probe.model.calls[1][0], [[5, 6, 7]])
        self.assertEqual(probe.model.calls[1][1]["past_key_values"], "prefilled")
        self.assertEqual(probe.model.attention_implementations, ["eager", "sdpa"])
        self.assertEqual(result["prefill_tokens"], 5)
        self.assertEqual(result["decoding_window_tokens"], 3)
        self.assertAlmostEqual(result["previous_attention"], 1.0 / 8)
        self.assertAlmostEqual(result["current_attention"], 1.0 / 8)

    def test_session_cache_prefills_only_new_prefix_tokens(self):
        class FakeCache:
            def __init__(self, length):
                self.length = length

            def crop(self, length):
                self.length = length

        class IncrementalModel:
            def __init__(self):
                self.calls = []

            def set_attn_implementation(self, implementation):
                pass

            def __call__(self, input_ids, **kwargs):
                values = input_ids.tolist()[0]
                self.calls.append((values, kwargs.get("output_attentions", False)))
                cache = kwargs.get("past_key_values")
                if cache is None:
                    cache = FakeCache(0)
                cache.length += len(values)
                if kwargs.get("output_attentions"):
                    attention = torch.full(
                        (1, 2, len(values), cache.length), 1.0 / cache.length
                    )
                    return types.SimpleNamespace(
                        past_key_values=cache, attentions=(attention,) * 4
                    )
                return types.SimpleNamespace(past_key_values=cache, attentions=None)

        probe = LocalAttentionProbe.__new__(LocalAttentionProbe)
        probe.model = IncrementalModel()
        probe.input_device = torch.device("cpu")
        probe.lock = threading.Lock()
        probe.monitor_history_tokens = 3
        probe.attention_window_tokens = 3
        probe.max_probe_tokens = 4
        probe.max_history_tokens = 20
        probe.max_cached_sessions = 2
        probe.kv_offload = False
        probe.sessions = OrderedDict()

        probe.analyze(
            [0, 1, 2, 3, 4, 5, 6], [9], None, {"start": 5, "end": 7}, "q1"
        )
        second = probe.analyze(
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [9],
            {"start": 5, "end": 7},
            {"start": 7, "end": 9},
            "q1",
        )

        self.assertEqual(probe.model.calls[0], ([0, 1, 2, 3, 4], False))
        self.assertEqual(probe.model.calls[2], ([5, 6], False))
        self.assertEqual(second["prefill_tokens"], 2)
        self.assertEqual(second["reused_prefix_tokens"], 5)
        self.assertFalse(second["cache_reset"])

        stateless = probe.analyze(
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [9],
            {"start": 5, "end": 7},
            {"start": 7, "end": 9},
        )
        self.assertAlmostEqual(second["entropy"], stateless["entropy"])
        self.assertAlmostEqual(
            second["previous_attention"], stateless["previous_attention"]
        )
        self.assertAlmostEqual(
            second["current_attention"], stateless["current_attention"]
        )

    def test_only_last_four_layers_materialize_attention(self):
        class Config:
            _attn_implementation = "sdpa"

        class FakeAttention:
            def __init__(self, config, calls):
                self.config = config
                self.calls = calls

            def forward(self, hidden_states, **kwargs):
                implementation = self.config._attn_implementation
                self.calls.append(implementation)
                query_length = hidden_states.shape[1]
                key_length = 8
                weights = None
                if implementation == "eager":
                    weights = torch.full(
                        (1, 2, query_length, key_length), 1.0 / key_length
                    )
                return hidden_states, weights

            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

        class FakeLayer:
            def __init__(self, config, calls):
                self.self_attn = FakeAttention(config, calls)

        class SelectiveModel:
            def __init__(self):
                self.config = Config()
                self.calls = []
                self.layers = [FakeLayer(self.config, self.calls) for _ in range(6)]

            def set_attn_implementation(self, implementation):
                self.config._attn_implementation = implementation

            def __call__(self, input_ids, **kwargs):
                hidden_states = input_ids.float().unsqueeze(-1)
                for layer in self.layers:
                    hidden_states, _ = layer.self_attn(
                        hidden_states, **kwargs
                    )
                return types.SimpleNamespace(
                    past_key_values=kwargs.get("past_key_values"), attentions=None
                )

        probe = LocalAttentionProbe.__new__(LocalAttentionProbe)
        probe.model = SelectiveModel()
        probe.input_device = torch.device("cpu")
        probe.lock = threading.Lock()
        probe.monitor_history_tokens = 2
        probe.attention_window_tokens = 3
        probe.max_probe_tokens = 4
        probe.max_history_tokens = 20
        probe.monitored_layer_count = 4
        probe.selective_attention = True

        result = probe.analyze(
            history_token_ids=[0, 1, 2, 3, 4, 5, 6],
            probe_token_ids=[7],
            previous_span={"start": 1, "end": 3},
            current_span={"start": 5, "end": 7},
        )

        # Prefix prefill is six SDPA calls; monitor is two SDPA + four eager.
        self.assertEqual(
            probe.model.calls[-6:],
            ["sdpa", "sdpa", "eager", "eager", "eager", "eager"],
        )
        self.assertEqual(result["layers_used"], 4)

    def test_real_qwen3_incremental_matches_full_recompute(self):
        torch.manual_seed(0)
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            use_cache=True,
        )
        probe = LocalAttentionProbe.__new__(LocalAttentionProbe)
        probe.model = Qwen3ForCausalLM(config).eval()
        probe.input_device = torch.device("cpu")
        probe.lock = threading.Lock()
        probe.attention_window_tokens = 8
        probe.monitor_history_tokens = 8
        probe.max_probe_tokens = 8
        probe.max_history_tokens = 128
        probe.max_cached_sessions = 2
        probe.kv_offload = False
        probe.sessions = OrderedDict()
        probe.monitored_layer_count = 4
        probe.selective_attention = True

        probe.analyze(
            list(range(20)), [20, 21], None, {"start": 15, "end": 20}, "q1"
        )
        incremental = probe.analyze(
            list(range(24)),
            [24, 25],
            {"start": 15, "end": 20},
            {"start": 20, "end": 24},
            "q1",
        )
        recomputed = probe.analyze(
            list(range(24)),
            [24, 25],
            {"start": 15, "end": 20},
            {"start": 20, "end": 24},
        )

        self.assertEqual(incremental["prefill_tokens"], 4)
        self.assertEqual(incremental["reused_prefix_tokens"], 14)
        self.assertFalse(incremental["cache_reset"])
        self.assertAlmostEqual(incremental["entropy"], recomputed["entropy"], places=4)
        self.assertAlmostEqual(
            incremental["previous_attention"],
            recomputed["previous_attention"],
            places=7,
        )
        self.assertAlmostEqual(
            incremental["current_attention"],
            recomputed["current_attention"],
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
