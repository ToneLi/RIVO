import torch

from verl.experimental.plugin_grpo.grpo import compute_group_advantages, plugin_grpo_loss


def test_group_advantages_are_centered_per_question():
    rewards = torch.tensor([0.0, 1.0, 1.0, 1.0, 3.0, 2.0])
    groups = torch.tensor([10, 10, 10, 20, 20, 20])
    advantages = compute_group_advantages(rewards, groups)
    assert torch.allclose(advantages[:3].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(advantages[3:].mean(), torch.tensor(0.0), atol=1e-6)


def test_grpo_ignores_frozen_host_positions():
    old = torch.zeros(2, 4)
    new = torch.zeros(2, 4, requires_grad=True)
    advantages = torch.tensor([1.0, -1.0])
    # First two columns stand for ordinary frozen-Host tokens.
    mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0]], dtype=torch.bool)
    loss, metrics = plugin_grpo_loss(new, old, advantages, mask)
    loss.backward()
    assert torch.equal(new.grad[:, :2], torch.zeros_like(new.grad[:, :2]))
    assert metrics["plugin/action_count"].item() == 3
