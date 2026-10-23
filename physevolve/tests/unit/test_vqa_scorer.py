"""Tests for :func:`physevolve.verification.vqa_scorer.score_vqa`."""

from __future__ import annotations

import pytest

from physevolve.verification.vqa_scorer import score_vqa


def test_score_vqa_exact_match_returns_1() -> None:
    parsed = {"letter": "C", "predicted_idx": 2}
    ground_truth = {"correct_idx": 2, "correct_letter": "C"}
    assert score_vqa(parsed, ground_truth) == 1.0


@pytest.mark.parametrize("predicted,correct", [(0, 1), (3, 2), (1, 3), (2, 0)])
def test_score_vqa_mismatch_returns_0(predicted: int, correct: int) -> None:
    parsed = {"letter": "X", "predicted_idx": predicted}
    ground_truth = {"correct_idx": correct}
    assert score_vqa(parsed, ground_truth) == 0.0


def test_score_vqa_none_parsed_returns_0() -> None:
    """A parser-failure (None) MUST short-circuit to reward 0."""

    assert score_vqa(None, {"correct_idx": 0}) == 0.0


def test_score_vqa_missing_predicted_idx_returns_0() -> None:
    assert score_vqa({"letter": "A"}, {"correct_idx": 0}) == 0.0


def test_score_vqa_non_int_predicted_idx_returns_0() -> None:
    assert score_vqa({"predicted_idx": "A"}, {"correct_idx": 0}) == 0.0
    assert score_vqa({"predicted_idx": None}, {"correct_idx": 1}) == 0.0


def test_score_vqa_rejects_out_of_range_ground_truth() -> None:
    with pytest.raises(ValueError, match=r"correct_idx must be in"):
        score_vqa({"predicted_idx": 0}, {"correct_idx": 5})


def test_score_vqa_missing_ground_truth_raises_key_error() -> None:
    with pytest.raises(KeyError):
        score_vqa({"predicted_idx": 0}, {})


def test_score_vqa_full_loop_through_router() -> None:
    """Integration: router parser → scorer roundtrip on a realistic blob.

    Mirrors what happens in the live Stage-2 pipeline.
    """

    from physevolve.tasks.router import TaskMode, parse_solver_output

    raw = "<think>I think it is C</think>\n<answer>C</answer>"
    parsed = parse_solver_output(raw, TaskMode.VQA)
    assert parsed is not None
    assert score_vqa(parsed, {"correct_idx": 2}) == 1.0
    assert score_vqa(parsed, {"correct_idx": 0}) == 0.0
