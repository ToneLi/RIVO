from types import SimpleNamespace

import torch
from torch import nn

from verl.experimental.plugin_grpo.hint_policy import HintSegment, build_semantic_vocab_mask
from verl.experimental.plugin_grpo.policy import FrozenHostPluginPolicy, RetrievalPluginPolicy
from verl.experimental.plugin_grpo.prompts import render_history
from verl.experimental.plugin_grpo.service import (
    ControllerStep,
    PluginGRPOEngine,
    PluginTrace,
    RewardItem,
    build_hint_query,
)


class TinyLM(nn.Module):
    def __init__(self, vocab_size=11, hidden_size=7):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size, hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, output_hidden_states=False, **kwargs):
        del kwargs
        hidden = self.embed(input_ids)
        return SimpleNamespace(
            logits=self.lm_head(hidden),
            hidden_states=(hidden,) if output_hidden_states else None,
        )


def test_history_renderer_keeps_research_turns_only():
    history = render_history(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "searching"},
            {"role": "tool", "content": "evidence"},
        ]
    )
    assert "system" not in history
    assert "question" not in history
    assert "searching" in history
    assert "evidence" in history


def test_service_update_changes_plugin_but_not_host():
    torch.manual_seed(19)
    host = TinyLM()
    plugin = RetrievalPluginPolicy(
        TinyLM(),
        hidden_size=7,
        host_vocab_size=11,
        correction_rank=3,
        label_token_ids=[1, 2, 3],
        train_full_backbone=True,
    )
    engine = PluginGRPOEngine.__new__(PluginGRPOEngine)
    engine.device = torch.device("cpu")
    engine.policy = FrozenHostPluginPolicy(host, plugin)
    engine.optimizer = torch.optim.AdamW(plugin.trainable_parameters(), lr=1e-2)
    engine.max_grad_norm = 1.0
    engine.clip_ratio = 0.2
    engine.controller_temperature = 1.0
    engine.update_step = 0

    ids_a = [1, 2, 3]
    ids_b = [3, 4, 5]
    allowed = [True, True, True]

    def make_step(ids, action):
        provisional = ControllerStep(ids, action, allowed, 0.0)
        old = float(engine._controller_new_log_prob(provisional).detach())
        return ControllerStep(ids, action, allowed, old)

    engine.traces = {
        "a": PluginTrace("question", True, controller_steps=[make_step(ids_a, 0)]),
        "b": PluginTrace("question", True, controller_steps=[make_step(ids_b, 1)]),
    }
    host_before = [parameter.detach().clone() for parameter in host.parameters()]
    controller_before = plugin.controller_head.weight.detach().clone()

    metrics = engine.update(
        [
            RewardItem(trace_id="a", group_id="question", reward=0.0),
            RewardItem(trace_id="b", group_id="question", reward=1.0),
        ]
    )

    assert metrics["updated"] is True
    assert metrics["actions"] == 2
    assert all(torch.equal(before, after) for before, after in zip(host_before, host.parameters(), strict=True))
    assert not torch.equal(controller_before, plugin.controller_head.weight)
    assert engine.traces == {}


def test_direct_hint_query_joins_fields_and_limits_words():
    fields = ("Jackson Browne Stay", "co lead vocalist", "Running on Empty 1977")
    assert build_hint_query(fields, max_words=6) == "Jackson Browne Stay co lead vocalist"


def test_semantic_vocab_mask_rejects_meta_and_east_asian_tokens(tmp_path):
    class TinyTokenizer:
        all_special_ids = []

        def __len__(self):
            return 6

        def decode(self, token_ids, **kwargs):
            del kwargs
            return [" we", " question", " 工具", " entity", " Grazia", " CEO"][token_ids[0]]

    mask = build_semantic_vocab_mask(
        TinyTokenizer(),
        6,
        tmp_path / "strict-mask.pt",
    )

    assert mask.tolist() == [False, False, False, True, True, True]


def test_semantic_vocab_mask_rejects_whitespace_only_tokens(tmp_path):
    class TinyTokenizer:
        all_special_ids = []

        def __len__(self):
            return 5

        def decode(self, token_ids, **kwargs):
            del kwargs
            return [" ", "   ", " word", "x", "!"][token_ids[0]]

    mask = build_semantic_vocab_mask(TinyTokenizer(), 5, tmp_path / "mask.pt")

    assert mask.tolist() == [False, False, True, True, False]


def test_outcome_reward_updates_hint_logits_but_not_host():
    torch.manual_seed(23)
    host = TinyLM()
    plugin = RetrievalPluginPolicy(
        TinyLM(),
        hidden_size=7,
        host_vocab_size=11,
        correction_rank=3,
        train_full_backbone=True,
    )
    policy = FrozenHostPluginPolicy(host, plugin)

    class TinyHintRuntime:
        def score_segment(self, step):
            host_prefix = torch.tensor([step.host_prefix_ids], dtype=torch.long)
            plugin_prefix = torch.tensor([step.plugin_prefix_ids], dtype=torch.long)
            target = torch.tensor([step.target_ids], dtype=torch.long)
            return policy.score_reroute_tokens(
                host_prefix,
                plugin_prefix,
                target,
                temperature=step.temperature,
                top_p=step.top_p,
            )[0]

    runtime = TinyHintRuntime()
    engine = PluginGRPOEngine.__new__(PluginGRPOEngine)
    engine.device = torch.device("cpu")
    engine.policy = policy
    engine.hint_runtime = runtime
    engine.trainable_parameters = list(plugin.trainable_parameters())
    engine.optimizer = torch.optim.AdamW(engine.trainable_parameters, lr=1e-2)
    engine.max_grad_norm = 1.0
    engine.clip_ratio = 0.2
    engine.update_step = 0
    engine.hint_quality_reward_weight = 0.5
    engine.hint_invalid_penalty = 2.0

    def make_segment(prefix):
        provisional = HintSegment(prefix, prefix, [6, 7], [], 1.0, 1.0, 0)
        old = runtime.score_segment(provisional).detach().tolist()
        return HintSegment(prefix, prefix, [6, 7], old, 1.0, 1.0, 0)

    engine.traces = {
        "a": PluginTrace(
            "question",
            True,
            hint_steps=[make_segment([1, 2, 3])],
            hint_quality_rewards=[-1.0],
        ),
        "b": PluginTrace(
            "question",
            True,
            hint_steps=[make_segment([3, 4, 5])],
            hint_quality_rewards=[1.0],
        ),
    }
    host_before = [parameter.detach().clone() for parameter in host.parameters()]
    correction_before = plugin.correction_head.up.weight.detach().clone()

    metrics = engine.update(
        [
            RewardItem(trace_id="a", group_id="question", reward=0.0),
            RewardItem(trace_id="b", group_id="question", reward=1.0),
        ]
    )

    assert metrics["updated"] is True
    assert metrics["actions"] == 4
    assert metrics["outcome_reward_mean"] == 0.5
    assert metrics["hint_quality_reward_mean"] == 0.0
    assert metrics["hint_quality_valid_rate"] == 0.5
    assert metrics["hint_quality_reward_weight"] == 0.5
    assert metrics["hint_invalid_rate"] == 0.5
    assert metrics["hint_invalid_penalty"] == 2.0
    assert metrics["reward_mean"] == -0.5
    assert all(torch.equal(before, after) for before, after in zip(host_before, host.parameters(), strict=True))
    assert not torch.equal(correction_before, plugin.correction_head.up.weight)
    assert engine.traces == {}
