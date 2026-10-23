"""Exact-match and collapse diagnostics for the Naive baseline.

The dense rewards assign partial credit on discrete outputs:

* Forward: ``0.7 * exp(-manhattan / 4) + 0.3 * status_match`` on a 16-cubed grid.
* Inverse: ``0.5 * direct + 0.5 * replay`` on seven velocity bins.
* Parameter: ``max(0, 1 - abs(pred - gt) / 4)`` on five parameter levels.

``exact_match_*`` checks all requested bodies. ``classify_collapse_*`` identifies
incorrect predictions consistent with velocity extrapolation, a parameter
midpoint, or zero initial velocity. These functions read numeric outputs and
trajectories without invoking a model.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from physevolve.engine.discretization import VELOCITY_BINS
from physevolve.engine.schema import SimState, world_to_grid

# ─── Collapse labels (Naive diagnostics) ──────────────────────────

COLLAPSE_VELOCITY_EXTRAPOLATION: str = "velocity_extrapolation"
"""Forward: the predicted final grid equals a linear extrapolation of the
observed (first-half) velocity, ignoring collisions / walls / gravity —
i.e. "extrapolation along the current velocity"."""

COLLAPSE_MID_RUNG: str = "mid_rung"
"""Parameter: every wrong (body, param) prediction is level 3, the middle
rung of the 1..5 mass / friction ladder."""

COLLAPSE_ZERO_VELOCITY: str = "zero_velocity"
"""Inverse: every wrong velocity-bin prediction is bin 3 (= 0.0 m/s), the
"stationary initial velocity" summary statistic."""

# Constants of the locked ladders.
_PARAMETER_MID_RUNG: int = 3  # middle of MASS_LEVELS / FRICTION_LEVELS (1..5)
_ZERO_VELOCITY_BIN: int = VELOCITY_BINS.index(0.0)  # bin index of 0.0 m/s


# ─── Exact match ───────────────────────────────────────────────────────


def _is_int(value: Any) -> bool:
    """True for a real int (excludes bool, which is an int subclass)."""

    return isinstance(value, int) and not isinstance(value, bool)


def exact_match_forward(prediction: dict, ground_truth: dict) -> bool:
    """True iff every GT body is present with exact grid AND exact status.

    Extra bodies in ``prediction`` are ignored (same convention as
    :func:`score_forward`). Empty GT -> ``False`` (matches the scorer's
    "empty -> 0.0" rule: no correct answer was given).
    """

    if not ground_truth or not isinstance(prediction, dict):
        return False
    for name, gt_entry in ground_truth.items():
        pred_entry = prediction.get(name)
        if not isinstance(pred_entry, dict):
            return False
        pred_grid = pred_entry.get("grid")
        pred_status = pred_entry.get("status")
        if not isinstance(pred_grid, (list, tuple)) or len(pred_grid) != 3:
            return False
        if not all(_is_int(g) for g in pred_grid):
            return False
        if list(pred_grid) != list(gt_entry["grid"]):
            return False
        if not isinstance(pred_status, str):
            return False
        if str(pred_status).strip().lower() != str(gt_entry["status"]).strip().lower():
            return False
    return True


def exact_match_inverse(prediction: dict, ground_truth: dict) -> bool:
    """True iff every GT body is present with the exact ``initial_vel_grid``."""

    if not ground_truth or not isinstance(prediction, dict):
        return False
    for name, gt_entry in ground_truth.items():
        pred_entry = prediction.get(name)
        if not isinstance(pred_entry, dict):
            return False
        pred_vg = pred_entry.get("initial_vel_grid")
        if not isinstance(pred_vg, (list, tuple)) or len(pred_vg) != 3:
            return False
        if not all(_is_int(v) for v in pred_vg):
            return False
        if list(pred_vg) != list(gt_entry["initial_vel_grid"]):
            return False
    return True


def exact_match_parameter(prediction: dict, ground_truth: dict) -> bool:
    """True iff every GT body has exact ``mass_level`` AND ``friction_level``."""

    if not ground_truth or not isinstance(prediction, dict):
        return False
    for name, gt_entry in ground_truth.items():
        pred_entry = prediction.get(name)
        if not isinstance(pred_entry, dict):
            return False
        pred_ml = pred_entry.get("mass_level")
        pred_fl = pred_entry.get("friction_level")
        if not _is_int(pred_ml) or not _is_int(pred_fl):
            return False
        if int(pred_ml) != int(gt_entry["mass_level"]):
            return False
        if int(pred_fl) != int(gt_entry["friction_level"]):
            return False
    return True


# ─── Collapse classification ────────────────────────────────────────────


def _extrapolated_final_grid(
    sim_state: SimState, body_name: str
) -> Optional[tuple[int, int, int]]:
    """Grid cell of a linear extrapolation of the first-half velocity.

    Estimates the body's velocity from the trajectory segment the Forward
    Solver actually sees (the first half of the clip) and propagates it
    ballistically to the final frame index, ignoring every collision / wall
    / gravity event in the second half. Time is linear in the trajectory row
    index, so the extrapolation is a pure index-ratio projection and needs
    no ``duration`` / ``timestep`` reads.

    Returns ``None`` when the trajectory is too short or missing.
    """

    traj = sim_state.trajectories.get(body_name)
    if traj is None:
        return None
    pos = traj.positions
    if not isinstance(pos, np.ndarray) or pos.ndim != 2 or pos.shape[0] < 2:
        return None
    n_rows = int(pos.shape[0])
    half = max(1, n_rows // 2)
    p0 = pos[0]
    p_half = pos[half]
    # Linear projection to the final row: p0 + (p_half - p0) * (n-1)/half.
    ratio = (n_rows - 1) / float(half)
    extrap = p0 + (p_half - p0) * ratio
    return world_to_grid(
        (float(extrap[0]), float(extrap[1]), float(extrap[2]))
    )


def classify_collapse_forward(
    prediction: dict,
    ground_truth: dict,
    sim_state: SimState,
) -> Optional[str]:
    """``"velocity_extrapolation"`` when every predicted grid equals the
    first-half linear extrapolation AND the prediction is not exact.

    A body missing from the prediction, or with a non-list grid, is treated
    as "not a clean extrapolation" and forces ``None`` (a missing body is a
    parse failure, not a summary statistic). Returns ``None`` when
    ``sim_state`` is ``None`` or no body can be checked.
    """

    if (
        not ground_truth
        or not isinstance(prediction, dict)
        or sim_state is None
        or exact_match_forward(prediction, ground_truth)
    ):
        return None
    checked = 0
    extrap_matches = 0
    for name in ground_truth:
        pred_entry = prediction.get(name)
        if not isinstance(pred_entry, dict):
            return None
        pred_grid = pred_entry.get("grid")
        if not isinstance(pred_grid, (list, tuple)) or len(pred_grid) != 3:
            return None
        if not all(_is_int(g) for g in pred_grid):
            return None
        extrap = _extrapolated_final_grid(sim_state, name)
        if extrap is None:
            continue
        checked += 1
        if [int(pred_grid[i]) for i in range(3)] == list(extrap):
            extrap_matches += 1
    if checked > 0 and extrap_matches == checked:
        return COLLAPSE_VELOCITY_EXTRAPOLATION
    return None


def classify_collapse_inverse(
    prediction: dict,
    ground_truth: dict,
    sim_state: Any = None,
) -> Optional[str]:
    """``"zero_velocity"`` when every wrong vel-bin prediction is bin 3
    (0.0 m/s) AND the prediction is not exact.

    ``sim_state`` is accepted for dispatcher symmetry but unused: the
    collapse is read off the predicted bins alone.
    """

    _ = sim_state
    if not ground_truth or not isinstance(prediction, dict):
        return None
    if exact_match_inverse(prediction, ground_truth):
        return None
    wrong = 0
    zero_collapsed = 0
    for name, gt_entry in ground_truth.items():
        pred_entry = prediction.get(name)
        if not isinstance(pred_entry, dict):
            return None
        pred_vg = pred_entry.get("initial_vel_grid")
        if not isinstance(pred_vg, (list, tuple)) or len(pred_vg) != 3:
            return None
        if not all(_is_int(v) for v in pred_vg):
            return None
        if list(pred_vg) == list(gt_entry["initial_vel_grid"]):
            continue  # this body is correct; collapse is about wrong ones
        wrong += 1
        if all(int(v) == _ZERO_VELOCITY_BIN for v in pred_vg):
            zero_collapsed += 1
    if wrong > 0 and zero_collapsed == wrong:
        return COLLAPSE_ZERO_VELOCITY
    return None


def classify_collapse_parameter(
    prediction: dict,
    ground_truth: dict,
    sim_state: Any = None,
) -> Optional[str]:
    """``"mid_rung"`` when every wrong (body, param) prediction is level 3
    AND the prediction is not exact."""

    _ = sim_state
    if not ground_truth or not isinstance(prediction, dict):
        return None
    if exact_match_parameter(prediction, ground_truth):
        return None
    wrong = 0
    mid_collapsed = 0
    for name, gt_entry in ground_truth.items():
        pred_entry = prediction.get(name)
        if not isinstance(pred_entry, dict):
            return None
        pred_ml = pred_entry.get("mass_level")
        pred_fl = pred_entry.get("friction_level")
        if not _is_int(pred_ml) or not _is_int(pred_fl):
            return None
        for pred_val, gt_val in (
            (int(pred_ml), int(gt_entry["mass_level"])),
            (int(pred_fl), int(gt_entry["friction_level"])),
        ):
            if pred_val == gt_val:
                continue
            wrong += 1
            if pred_val == _PARAMETER_MID_RUNG:
                mid_collapsed += 1
    if wrong > 0 and mid_collapsed == wrong:
        return COLLAPSE_MID_RUNG
    return None


# ─── Dispatcher ────────────────────────────────────────────────────────


def naive_diagnostics(
    mode: str,
    prediction: Optional[dict],
    ground_truth: dict,
    sim_state: Optional[SimState] = None,
) -> tuple[bool, Optional[str]]:
    """Return ``(exact_match, collapse_type)`` for one naive prediction.

    Args:
        mode: Task-mode *name* (``"forward"`` / ``"inverse"`` / \
            ``"parameter"``). ``TaskMode`` values are accepted because
            ``TaskMode`` is a ``str`` enum. ``"vqa"`` returns \
            ``(False, None)`` — the discrete bottleneck has no soft track.
        prediction: Parsed, schema-valid Solver output (``None`` means \
            parse failure -> not exact, no collapse).
        ground_truth: Ground-truth dict of the matching shape.
        sim_state: Source :class:`SimState`; required only for the Forward \
            velocity-extrapolation check.

    Returns:
        ``(exact_match, collapse_type)`` where ``collapse_type`` is one of
        the ``COLLAPSE_*`` labels or ``None``.
    """

    if prediction is None:
        return (False, None)
    mode_name = str(mode.value) if hasattr(mode, "value") else str(mode)
    if mode_name == "forward":
        return (
            exact_match_forward(prediction, ground_truth),
            classify_collapse_forward(prediction, ground_truth, sim_state),
        )
    if mode_name == "inverse":
        return (
            exact_match_inverse(prediction, ground_truth),
            classify_collapse_inverse(prediction, ground_truth, sim_state),
        )
    if mode_name == "parameter":
        return (
            exact_match_parameter(prediction, ground_truth),
            classify_collapse_parameter(prediction, ground_truth, sim_state),
        )
    # vqa / unknown: the discrete bottleneck has no soft / collapse track.
    return (False, None)


__all__ = [
    "COLLAPSE_MID_RUNG",
    "COLLAPSE_VELOCITY_EXTRAPOLATION",
    "COLLAPSE_ZERO_VELOCITY",
    "classify_collapse_forward",
    "classify_collapse_inverse",
    "classify_collapse_parameter",
    "exact_match_forward",
    "exact_match_inverse",
    "exact_match_parameter",
    "naive_diagnostics",
]
