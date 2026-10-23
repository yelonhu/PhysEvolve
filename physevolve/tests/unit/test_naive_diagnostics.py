"""Unit tests for :mod:`physevolve.verification.naive_diagnostics`.

Covers the three exact-match scorers, the three collapse classifiers, the
dispatcher, and the empty / malformed-prediction guards. The Forward
velocity-extrapolation cases use a lightweight duck-typed ``sim_state``
(the classifier only reads ``trajectories[name].positions``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from physevolve.engine.schema import world_to_grid
from physevolve.verification.naive_diagnostics import (
    COLLAPSE_MID_RUNG,
    COLLAPSE_VELOCITY_EXTRAPOLATION,
    COLLAPSE_ZERO_VELOCITY,
    _extrapolated_final_grid,
    classify_collapse_forward,
    classify_collapse_inverse,
    classify_collapse_parameter,
    exact_match_forward,
    exact_match_inverse,
    exact_match_parameter,
    naive_diagnostics,
)


# ─── Helpers ───────────────────────────────────────────────────────────


def _fake_sim_state(positions: np.ndarray) -> SimpleNamespace:
    """Build a duck-typed sim_state with one ``ball`` trajectory.

    ``classify_collapse_forward`` only reads ``trajectories[name].positions``,
    so a :class:`SimpleNamespace` is sufficient and avoids constructing a
    full :class:`SimState` (which needs a complete ``SceneDef``).
    """

    return SimpleNamespace(trajectories={"ball": SimpleNamespace(positions=positions)})


# ─── Exact match ──────────────────────────────────────────────────────


class TestExactMatchForward:
    gt = {
        "a": {"grid": [5, 5, 5], "status": "stable"},
        "b": {"grid": [10, 0, 3], "status": "moving"},
    }

    def test_perfect(self) -> None:
        assert exact_match_forward(self.gt, self.gt) is True

    def test_status_case_insensitive(self) -> None:
        pred = {
            "a": {"grid": [5, 5, 5], "status": "STABLE"},
            "b": {"grid": [10, 0, 3], "status": "Moving"},
        }
        assert exact_match_forward(pred, self.gt) is True

    def test_grid_off_by_one_is_not_exact(self) -> None:
        pred = {
            "a": {"grid": [5, 5, 6], "status": "stable"},
            "b": {"grid": [10, 0, 3], "status": "moving"},
        }
        assert exact_match_forward(pred, self.gt) is False

    def test_status_wrong_is_not_exact(self) -> None:
        pred = {
            "a": {"grid": [5, 5, 5], "status": "moving"},
            "b": {"grid": [10, 0, 3], "status": "moving"},
        }
        assert exact_match_forward(pred, self.gt) is False

    def test_missing_body_is_not_exact(self) -> None:
        pred = {"a": self.gt["a"]}
        assert exact_match_forward(pred, self.gt) is False

    def test_extra_body_ignored(self) -> None:
        pred = dict(self.gt)
        pred["c"] = {"grid": [0, 0, 0], "status": "ejected"}
        assert exact_match_forward(pred, self.gt) is True

    def test_empty_gt_is_false(self) -> None:
        assert exact_match_forward(self.gt, {}) is False

    def test_non_dict_prediction_is_false(self) -> None:
        assert exact_match_forward("not a dict", self.gt) is False  # type: ignore[arg-type]


class TestExactMatchInverse:
    gt = {"a": {"initial_vel_grid": [5, 3, 3]}, "b": {"initial_vel_grid": [0, 0, 6]}}

    def test_perfect(self) -> None:
        assert exact_match_inverse(self.gt, self.gt) is True

    def test_off_by_one_bin_is_not_exact(self) -> None:
        pred = {"a": {"initial_vel_grid": [4, 3, 3]}, "b": {"initial_vel_grid": [0, 0, 6]}}
        assert exact_match_inverse(pred, self.gt) is False

    def test_missing_body_is_not_exact(self) -> None:
        pred = {"a": self.gt["a"]}
        assert exact_match_inverse(pred, self.gt) is False


class TestExactMatchParameter:
    gt = {"a": {"mass_level": 3, "friction_level": 2}, "b": {"mass_level": 5, "friction_level": 1}}

    def test_perfect(self) -> None:
        assert exact_match_parameter(self.gt, self.gt) is True

    def test_mass_off_is_not_exact(self) -> None:
        pred = {"a": {"mass_level": 4, "friction_level": 2}, "b": {"mass_level": 5, "friction_level": 1}}
        assert exact_match_parameter(pred, self.gt) is False

    def test_friction_off_is_not_exact(self) -> None:
        pred = {"a": {"mass_level": 3, "friction_level": 3}, "b": {"mass_level": 5, "friction_level": 1}}
        assert exact_match_parameter(pred, self.gt) is False


# ─── Forward extrapolation ─────────────────────────────────────────────


class TestExtrapolatedFinalGrid:
    def test_linear_motion_extrapolation_matches_final(self) -> None:
        # Pure linear motion: p0=[0,0,0], p_half=pos[2]=[0.4,0,0], ratio=1.5
        # -> extrap = [0.6,0,0] -> grid (11,8,0). Final pos[3]=[0.6,0,0] too.
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0]]
        )
        sim_state = _fake_sim_state(positions)
        assert _extrapolated_final_grid(sim_state, "ball") == world_to_grid((0.6, 0.0, 0.0))

    def test_short_trajectory_returns_none(self) -> None:
        sim_state = _fake_sim_state(np.array([[0.0, 0.0, 0.0]]))
        assert _extrapolated_final_grid(sim_state, "ball") is None

    def test_missing_body_returns_none(self) -> None:
        sim_state = _fake_sim_state(np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
        assert _extrapolated_final_grid(sim_state, "missing") is None


class TestClassifyCollapseForward:
    def test_velocity_extrapolation_collapse(self) -> None:
        # First half moves +x; second half deflects back to a different grid.
        # extrap = [0.6,0,0] -> grid (11,8,0); GT final = [0.1,0,0] -> (8,8,0).
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.1, 0.0, 0.0]]
        )
        sim_state = _fake_sim_state(positions)
        gt = {"ball": {"grid": [8, 8, 0], "status": "moving"}}
        pred = {"ball": {"grid": [11, 8, 0], "status": "stable"}}
        assert classify_collapse_forward(pred, gt, sim_state) == COLLAPSE_VELOCITY_EXTRAPOLATION
        assert exact_match_forward(pred, gt) is False

    def test_exact_match_has_no_collapse(self) -> None:
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0]]
        )
        sim_state = _fake_sim_state(positions)
        gt = {"ball": {"grid": world_to_grid((0.6, 0.0, 0.0)), "status": "stable"}}
        # Prediction equals GT -> exact, no collapse.
        assert classify_collapse_forward(gt, gt, sim_state) is None
        assert exact_match_forward(gt, gt) is True

    def test_wrong_but_not_extrapolation_is_none(self) -> None:
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.1, 0.0, 0.0]]
        )
        sim_state = _fake_sim_state(positions)
        gt = {"ball": {"grid": [8, 8, 0], "status": "moving"}}
        # A grid that is neither GT nor the extrapolation (11,8,0).
        pred = {"ball": {"grid": [2, 2, 2], "status": "fallen"}}
        assert classify_collapse_forward(pred, gt, sim_state) is None

    def test_missing_body_is_none(self) -> None:
        sim_state = _fake_sim_state(
            np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.1, 0.0, 0.0]])
        )
        gt = {"ball": {"grid": [8, 8, 0], "status": "moving"}}
        assert classify_collapse_forward({}, gt, sim_state) is None

    def test_none_sim_state_is_none(self) -> None:
        gt = {"ball": {"grid": [8, 8, 0], "status": "moving"}}
        pred = {"ball": {"grid": [11, 8, 0], "status": "stable"}}
        assert classify_collapse_forward(pred, gt, None) is None


# ─── Inverse collapse ──────────────────────────────────────────────────


class TestClassifyCollapseInverse:
    def test_zero_velocity_collapse(self) -> None:
        # GT bins differ from [3,3,3]; every wrong bin is the zero-velocity bin 3.
        gt = {"a": {"initial_vel_grid": [5, 3, 4]}, "b": {"initial_vel_grid": [0, 1, 2]}}
        pred = {"a": {"initial_vel_grid": [3, 3, 3]}, "b": {"initial_vel_grid": [3, 3, 3]}}
        assert classify_collapse_inverse(pred, gt) == COLLAPSE_ZERO_VELOCITY
        assert exact_match_inverse(pred, gt) is False

    def test_partial_correctness_still_collapse_on_wrong(self) -> None:
        # Body a is fully correct; body b is wrong and all-wrong bins are 3.
        gt = {"a": {"initial_vel_grid": [3, 3, 3]}, "b": {"initial_vel_grid": [5, 5, 5]}}
        pred = {"a": {"initial_vel_grid": [3, 3, 3]}, "b": {"initial_vel_grid": [3, 3, 3]}}
        assert classify_collapse_inverse(pred, gt) == COLLAPSE_ZERO_VELOCITY

    def test_exact_match_has_no_collapse(self) -> None:
        gt = {"a": {"initial_vel_grid": [3, 3, 3]}}
        assert classify_collapse_inverse(gt, gt) is None

    def test_wrong_but_not_zero_velocity_is_none(self) -> None:
        gt = {"a": {"initial_vel_grid": [5, 5, 5]}}
        pred = {"a": {"initial_vel_grid": [1, 2, 3]}}  # not all bin 3
        assert classify_collapse_inverse(pred, gt) is None

    def test_missing_body_is_none(self) -> None:
        gt = {"a": {"initial_vel_grid": [5, 5, 5]}}
        assert classify_collapse_inverse({}, gt) is None


# ─── Parameter collapse ────────────────────────────────────────────────


class TestClassifyCollapseParameter:
    def test_mid_rung_collapse(self) -> None:
        gt = {"a": {"mass_level": 1, "friction_level": 5}, "b": {"mass_level": 5, "friction_level": 1}}
        pred = {"a": {"mass_level": 3, "friction_level": 3}, "b": {"mass_level": 3, "friction_level": 3}}
        assert classify_collapse_parameter(pred, gt) == COLLAPSE_MID_RUNG
        assert exact_match_parameter(pred, gt) is False

    def test_exact_match_has_no_collapse(self) -> None:
        gt = {"a": {"mass_level": 3, "friction_level": 3}}
        assert classify_collapse_parameter(gt, gt) is None

    def test_wrong_but_not_mid_rung_is_none(self) -> None:
        gt = {"a": {"mass_level": 1, "friction_level": 1}}
        pred = {"a": {"mass_level": 2, "friction_level": 4}}  # not level 3
        assert classify_collapse_parameter(pred, gt) is None

    def test_partial_correctness_still_collapse_on_wrong(self) -> None:
        # mass exact, friction wrong=3; the wrong param collapses to mid-rung.
        gt = {"a": {"mass_level": 3, "friction_level": 1}}
        pred = {"a": {"mass_level": 3, "friction_level": 3}}
        assert classify_collapse_parameter(pred, gt) == COLLAPSE_MID_RUNG

    def test_missing_body_is_none(self) -> None:
        gt = {"a": {"mass_level": 1, "friction_level": 1}}
        assert classify_collapse_parameter({}, gt) is None


# ─── Dispatcher ────────────────────────────────────────────────────────


class TestNaiveDiagnosticsDispatcher:
    def test_none_prediction_is_not_exact_no_collapse(self) -> None:
        gt = {"ball": {"grid": [8, 8, 0], "status": "moving"}}
        assert naive_diagnostics("forward", None, gt, None) == (False, None)

    def test_forward_dispatch(self) -> None:
        gt = {"ball": {"grid": [8, 8, 0], "status": "moving"}}
        assert naive_diagnostics("forward", gt, gt, None) == (True, None)

    def test_inverse_dispatch(self) -> None:
        gt = {"ball": {"initial_vel_grid": [3, 3, 3]}}
        assert naive_diagnostics("inverse", gt, gt, None) == (True, None)

    def test_parameter_dispatch(self) -> None:
        gt = {"ball": {"mass_level": 3, "friction_level": 3}}
        assert naive_diagnostics("parameter", gt, gt, None) == (True, None)

    def test_vqa_returns_false_none(self) -> None:
        # The discrete bottleneck has no soft / collapse track.
        gt = {"correct_idx": 2}
        assert naive_diagnostics("vqa", {"letter": "A", "predicted_idx": 0}, gt, None) == (False, None)

    def test_unknown_mode_returns_false_none(self) -> None:
        gt = {"ball": {"grid": [8, 8, 0], "status": "moving"}}
        assert naive_diagnostics("bogus", gt, gt, None) == (False, None)

    def test_taskmode_str_enum_accepted(self) -> None:
        # TaskMode is a str enum, so its values compare equal to the string.
        from physevolve.tasks.router import TaskMode

        gt = {"ball": {"mass_level": 3, "friction_level": 3}}
        assert naive_diagnostics(TaskMode.PARAMETER, gt, gt, None) == (True, None)
