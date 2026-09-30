from types import SimpleNamespace

import torch
from torch import nn

from verl.experimental.plugin_grpo.policy import (
    FrozenHostPluginPolicy,
    LowRankCorrectionHead,
    RetrievalPluginPolicy,
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


def test_correction_head_starts_at_zero_and_receives_gradient():
    head = LowRankCorrectionHead(7, 3, 11)
    hidden = torch.randn(2, 7)
    assert torch.equal(head(hidden), torch.zeros(2, 11))
    head(hidden).sum().backward()
    assert head.up.weight.grad is not None
    assert torch.count_nonzero(head.up.weight.grad) > 0


def test_host_is_frozen_and_only_plugin_changes_logits():
    torch.manual_seed(4)
    host = TinyLM()
    backbone = TinyLM()
    plugin = RetrievalPluginPolicy(
        backbone,
        hidden_size=7,
        host_vocab_size=11,
        correction_rank=3,
        label_token_ids=[1, 2, 3],
    )
    policy = FrozenHostPluginPolicy(host, plugin, alpha=0.5)
    ids = torch.tensor([[1, 4, 5]])
    with torch.no_grad():
        base_logits = host(ids).logits[:, -1].float()
        hidden = backbone(ids, output_hidden_states=True).hidden_states[-1][:, -1]
    assert torch.equal(policy.corrected_logits(base_logits, hidden), base_logits)
    assert all(not parameter.requires_grad for parameter in host.parameters())
    assert all(parameter.requires_grad for parameter in plugin.correction_head.parameters())


def test_reroute_score_backpropagates_only_to_plugin():
    torch.manual_seed(8)
    host = TinyLM()
    backbone = TinyLM()
    plugin = RetrievalPluginPolicy(
        backbone,
        hidden_size=7,
        host_vocab_size=11,
        correction_rank=3,
        label_token_ids=[1, 2, 3],
        train_full_backbone=True,
    )
    policy = FrozenHostPluginPolicy(host, plugin)
    host_prefix = torch.tensor([[1, 2]])
    plugin_prefix = torch.tensor([[3, 4, 5]])
    target = torch.tensor([[6, 7]])
    loss = -policy.score_reroute_tokens(host_prefix, plugin_prefix, target).mean()
    loss.backward()
    assert all(parameter.grad is None for parameter in host.parameters())
    assert plugin.correction_head.up.weight.grad is not None
