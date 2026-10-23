"""Tests for PhysEvolve reward-batch helpers."""

from __future__ import annotations

import pytest
import torch

from physevolve.training.reward_batch import RewardRecord, build_reward_batch, final_token_index


def test_final_token_index_clamps_to_valid_response() -> None:
    assert final_token_index(5, 32) == 4
    assert final_token_index(0, 32) == 0
    assert final_token_index(100, 32) == 31


def test_build_reward_batch_places_only_final_token() -> None:
    batch = build_reward_batch(
        [
            RewardRecord("pred_forward", 1.0, 5, 8, scene_id="a"),
            RewardRecord("gen_physics", -1.0, 8, 8, scene_id="b"),
        ],
        response_length=8,
    )

    assert batch.tensor.shape == (2, 8)
    assert batch.tensor[0, 4].item() == pytest.approx(1.0)
    assert batch.tensor[1, 7].item() == pytest.approx(-1.0)
    assert torch.isfinite(batch.tensor).all()
    assert batch.metrics["reward_tensor_finite"] is True
    assert batch.metrics["problem_types_seen"] == ["gen_physics", "pred_forward"]


def test_build_reward_batch_rejects_non_finite_reward() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        build_reward_batch([RewardRecord("pred_forward", float("nan"), 1, 8)])
