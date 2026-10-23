"""Unit tests for :mod:`physevolve.verification.dense_scorer` and metrics."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

import physevolve.engine  # noqa: F401 — EGL env.
from physevolve.executor import run_executor
from physevolve.verification.dense_scorer import (
    score_forward,
    score_inverse,
    score_parameter,
)
from physevolve.verification.metrics import (
    LEVEL_TOLERANCE,
    POS_WEIGHT,
    STATUS_WEIGHT,
    level_score,
    manhattan,
    mean,
    pos_score,
    status_score,
)


# ─── Pure metric functions ────────────────────────────────────────────


class TestMetrics:
    def test_manhattan_basic(self) -> None:
        assert manhattan([0, 0, 0], [0, 0, 0]) == 0
        assert manhattan([1, 2, 3], [4, 6, 0]) == 3 + 4 + 3

    def test_manhattan_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            manhattan([1, 2], [1, 2, 3])

    def test_pos_score_exact(self) -> None:
        assert pos_score([5, 5, 5], [5, 5, 5]) == pytest.approx(1.0)

    def test_pos_score_softness(self) -> None:
        # Distance 4 → exp(-1) ≈ 0.368
        assert pos_score([0, 0, 0], [4, 0, 0]) == pytest.approx(math.exp(-1.0))

    def test_status_score(self) -> None:
        assert status_score("stable", "stable") == 1.0
        assert status_score("Stable", "stable") == 1.0  # case insensitive
        assert status_score("stable", "moving") == 0.0

    def test_level_score(self) -> None:
        assert level_score(3, 3) == 1.0
        assert level_score(1, 5) == 0.0  # distance 4 = full tolerance
        assert level_score(2, 4) == 0.5
        assert level_score(1, 1) == 1.0

    def test_mean_empty_is_zero(self) -> None:
        assert mean([]) == 0.0
        assert mean([1.0, 0.0, 0.5]) == 0.5

    def test_weights_locked(self) -> None:
        assert POS_WEIGHT == pytest.approx(0.7)
        assert STATUS_WEIGHT == pytest.approx(0.3)
        assert LEVEL_TOLERANCE == 4


# ─── Forward scorer ──────────────────────────────────────────────────


class TestScoreForward:
    gt = {
        "a": {"grid": [5, 5, 5], "status": "stable"},
        "b": {"grid": [10, 0, 3], "status": "moving"},
    }

    def test_perfect(self) -> None:
        assert score_forward(self.gt, self.gt) == pytest.approx(1.0)

    def test_empty_prediction(self) -> None:
        assert score_forward({}, self.gt) == 0.0

    def test_empty_gt(self) -> None:
        assert score_forward(self.gt, {}) == 0.0

    def test_missing_body_scores_zero(self) -> None:
        pred = {"a": self.gt["a"]}
        # body `a` scores 1.0, body `b` missing → 0.0. Mean = 0.5.
        assert score_forward(pred, self.gt) == pytest.approx(0.5)

    def test_off_by_one_manhattan(self) -> None:
        pred = {"a": {"grid": [5, 5, 6], "status": "stable"}}
        gt = {"a": {"grid": [5, 5, 5], "status": "stable"}}
        # pos_score = exp(-1/4), status = 1.0 → 0.7 * exp(-0.25) + 0.3.
        expected = 0.7 * math.exp(-0.25) + 0.3
        assert score_forward(pred, gt) == pytest.approx(expected)

    def test_malformed_entry_scores_zero(self) -> None:
        pred = {"a": "not a dict"}
        gt = {"a": {"grid": [5, 5, 5], "status": "stable"}}
        assert score_forward(pred, gt) == 0.0


# ─── Parameter scorer ────────────────────────────────────────────────


class TestScoreParameter:
    gt = {
        "a": {"mass_level": 3, "friction_level": 2},
        "b": {"mass_level": 5, "friction_level": 1},
    }

    def test_perfect(self) -> None:
        assert score_parameter(self.gt, self.gt) == pytest.approx(1.0)

    def test_empty_prediction(self) -> None:
        assert score_parameter({}, self.gt) == 0.0

    def test_off_by_four_is_zero(self) -> None:
        pred = {"a": {"mass_level": 1, "friction_level": 5}}
        gt = {"a": {"mass_level": 5, "friction_level": 1}}
        # Both dimensions off by 4 → both 0.0 → mean 0.0.
        assert score_parameter(pred, gt) == 0.0

    def test_off_by_two_is_half(self) -> None:
        pred = {"a": {"mass_level": 1, "friction_level": 2}}
        gt = {"a": {"mass_level": 3, "friction_level": 2}}
        # mass_level off by 2 → 0.5; friction_level exact → 1.0. Mean = 0.75.
        assert score_parameter(pred, gt) == pytest.approx(0.75)


# ─── Inverse scorer ──────────────────────────────────────────────────


_INVERSE_SRC = '''
def build_scene(api):
    api.add_sphere('ball', pos=(0.0, 0.0, 1.5), radius=0.1, mass=1.0)
    api.set_velocity('ball', linear=(3.0, 0.0, 0.0))
    api.set_duration(0.5)
'''


@pytest.fixture(scope="module")
def inverse_sim_state():
    r = run_executor(_INVERSE_SRC, stage=1, seed=0, sim_timeout_s=10.0)
    assert r.ok, r.error_message
    return r.sim_state


class TestScoreInverse:
    def test_perfect_direct_high_replay(self, inverse_sim_state) -> None:
        # Ground truth: ball has vx=3 m/s, so vx_bin is the index of 3.0
        # in VELOCITY_BINS (= 5). vy = vz = 0.0 → bin 3.
        gt = {"ball": {"initial_vel_grid": [5, 3, 3]}}
        pred = gt
        s = score_inverse(pred, gt, inverse_sim_state, replay_timeout_s=15.0)
        # Direct is 1.0; replay simulates with vx=3 m/s which matches
        # the original sim → replay score ~1.0; total ~1.0.
        assert s > 0.9

    def test_empty_prediction(self, inverse_sim_state) -> None:
        gt = {"ball": {"initial_vel_grid": [5, 3, 3]}}
        s = score_inverse({}, gt, inverse_sim_state, replay_timeout_s=5.0)
        # Direct = 0; replay uses original velocities (conservative
        # default) → replay ≈ 1.0 → total ≈ 0.5.
        assert s == pytest.approx(0.5, abs=0.05)

    def test_replay_failure_zeroes_replay_component(
        self, inverse_sim_state
    ) -> None:
        """Monkeypatch simulate to raise; replay score must be 0."""

        gt = {"ball": {"initial_vel_grid": [5, 3, 3]}}
        pred = gt
        with patch(
            "physevolve.engine.simulator.simulate",
            side_effect=RuntimeError("boom"),
        ):
            s = score_inverse(pred, gt, inverse_sim_state, replay_timeout_s=5.0)
        # direct_score = 1.0; replay_score = 0 → total = 0.5.
        assert s == pytest.approx(0.5)
