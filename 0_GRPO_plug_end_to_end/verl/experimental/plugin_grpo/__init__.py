"""Trainable retrieval-control plugin for a frozen research host."""

from .grpo import compute_group_advantages, plugin_grpo_loss
from .policy import (
    CONTROLLER_LABELS,
    FrozenHostPluginPolicy,
    LowRankCorrectionHead,
    RerouteGeneration,
    RetrievalPluginPolicy,
)

__all__ = [
    "CONTROLLER_LABELS",
    "FrozenHostPluginPolicy",
    "LowRankCorrectionHead",
    "RetrievalPluginPolicy",
    "RerouteGeneration",
    "compute_group_advantages",
    "plugin_grpo_loss",
]
