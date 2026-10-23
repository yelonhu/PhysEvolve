"""Reward tensor helpers for PhysEvolve-native training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class RewardRecord:
    """One scalar reward and its response-placement metadata."""

    problem_type: str
    reward: float
    valid_response_length: int
    response_length: int
    scene_id: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RewardBatch:
    """Token-level reward tensor plus aggregate diagnostics."""

    tensor: torch.Tensor
    records: tuple[RewardRecord, ...]
    metrics: dict[str, Any]


def final_token_index(valid_response_length: int, response_length: int) -> int:
    """Return the clamped zero-based index for final-token reward placement."""

    if response_length <= 0:
        raise ValueError("response_length must be positive")
    valid = max(1, min(int(valid_response_length), int(response_length)))
    return valid - 1


def build_reward_batch(
    records: Sequence[RewardRecord],
    *,
    response_length: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> RewardBatch:
    """Build a finite reward tensor with one scalar at each final valid token."""

    rows = tuple(records)
    if response_length is None:
        response_length = max((row.response_length for row in rows), default=1)
    if response_length <= 0:
        raise ValueError("response_length must be positive")

    tensor = torch.zeros((len(rows), int(response_length)), dtype=dtype)
    for i, row in enumerate(rows):
        reward = float(row.reward)
        if not np.isfinite(reward):
            raise ValueError(f"non-finite reward at row {i}: {reward}")
        idx = final_token_index(row.valid_response_length, response_length)
        tensor[i, idx] = reward

    metrics = reward_batch_metrics(rows, tensor)
    return RewardBatch(tensor=tensor, records=rows, metrics=metrics)


def reward_batch_metrics(records: Iterable[RewardRecord], tensor: torch.Tensor) -> dict[str, Any]:
    """Return sentinel metrics for training legality checks."""

    rows = tuple(records)
    problem_types_seen = sorted({row.problem_type for row in rows})
    rewards = [float(row.reward) for row in rows]
    return {
        "n_reward_records": len(rows),
        "problem_types_seen": problem_types_seen,
        "reward_tensor_shape": list(tensor.shape),
        "reward_tensor_finite": bool(torch.isfinite(tensor).all().item()),
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_min": float(np.min(rewards)) if rewards else 0.0,
        "reward_max": float(np.max(rewards)) if rewards else 0.0,
    }


__all__ = [
    "RewardBatch",
    "RewardRecord",
    "build_reward_batch",
    "final_token_index",
    "reward_batch_metrics",
]
