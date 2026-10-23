"""Shared value objects for PhysEvolve online self-play training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from physevolve.training.reward_batch import RewardBatch


@dataclass(frozen=True, slots=True)
class OnlineSelfPlayStep:
    """One PhysEvolve self-play step consumed by the trainer."""

    step: int
    reward_batch: RewardBatch
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


__all__ = ["OnlineSelfPlayStep"]
