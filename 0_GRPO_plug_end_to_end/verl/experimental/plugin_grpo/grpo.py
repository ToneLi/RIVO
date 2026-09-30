"""GRPO utilities for controller and corrected-query actions."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_group_advantages(
    rewards: Tensor, group_ids: Tensor, *, normalize_by_std: bool = True, eps: float = 1e-6
) -> Tensor:
    """Compute leave-in group-normalized outcome advantages."""
    if rewards.ndim != 1 or group_ids.ndim != 1 or rewards.shape != group_ids.shape:
        raise ValueError("rewards and group_ids must be same-shaped rank-1 tensors")
    advantages = torch.empty_like(rewards, dtype=torch.float32)
    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        values = rewards[mask].float()
        centered = values - values.mean()
        if normalize_by_std and values.numel() > 1:
            centered = centered / (values.std(unbiased=False) + eps)
        advantages[mask] = centered
    return advantages


def plugin_grpo_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    action_mask: Tensor,
    *,
    clip_ratio: float = 0.2,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Token/action-level clipped GRPO objective.

    Controller decisions and reroute-query tokens are concatenated along the
    second dimension. ``action_mask`` excludes frozen Host tokens and padding.
    """
    if new_log_probs.shape != old_log_probs.shape or new_log_probs.shape != action_mask.shape:
        raise ValueError("new/old log-probs and action_mask must have identical shapes")
    if advantages.ndim != 1 or advantages.shape[0] != new_log_probs.shape[0]:
        raise ValueError("advantages must contain one scalar per trajectory")
    if clip_ratio < 0:
        raise ValueError("clip_ratio must be non-negative")
    mask = action_mask.to(dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError("At least one plugin action is required")

    log_ratio = new_log_probs.float() - old_log_probs.float()
    ratio = log_ratio.exp()
    expanded_advantage = advantages.float().unsqueeze(-1)
    unclipped = ratio * expanded_advantage
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * expanded_advantage
    per_action_loss = -torch.minimum(unclipped, clipped)
    loss = per_action_loss.masked_select(mask).mean()
    with torch.no_grad():
        metrics = {
            "plugin/approx_kl": ((ratio - 1.0) - log_ratio).masked_select(mask).mean(),
            "plugin/clip_fraction": ((ratio - 1.0).abs() > clip_ratio).masked_select(mask).float().mean(),
            "plugin/action_count": mask.sum(),
        }
    return loss, metrics
