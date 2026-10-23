"""Stage-1 dense reward scorers.

All three scorers are pure numeric functions over dicts whose schema is
already enforced by :func:`tasks.router.parse_solver_output`. If the
parser returned ``None`` (bad JSON, schema violation), the caller must
short-circuit to reward 0 *without* invoking these functions.

Aggregation convention: every scorer returns the mean over the ground-
truth bodies. A body that appears in GT but is missing in the
prediction scores 0; extra bodies in the prediction are ignored. An
empty GT returns 0.0.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from physevolve.engine.discretization import VELOCITY_BINS, bin_to_velocity
from physevolve.engine.schema import (
    BodyDef,
    EnvConfig,
    SceneDef,
    SimState,
    world_to_grid,
)
from physevolve.verification.metrics import (
    body_score_forward,
    level_score,
    mean,
    pos_score,
)


# ─── Forward ──────────────────────────────────────────────────────────


def score_forward(prediction: dict, ground_truth: dict) -> float:
    """Compute ``0.7 * pos_score + 0.3 * status_score``."""

    if not ground_truth:
        return 0.0
    body_scores: list[float] = []
    for name, gt_entry in ground_truth.items():
        pred_entry = prediction.get(name) if isinstance(prediction, dict) else None
        if not isinstance(pred_entry, dict):
            body_scores.append(0.0)
            continue
        try:
            score = body_score_forward(
                pred_grid=pred_entry["grid"],
                pred_status=str(pred_entry["status"]),
                gt_grid=gt_entry["grid"],
                gt_status=str(gt_entry["status"]),
            )
        except (KeyError, TypeError, ValueError):
            score = 0.0
        body_scores.append(score)
    return mean(body_scores)


# ─── Inverse ──────────────────────────────────────────────────────────


def _direct_inverse(prediction: dict, ground_truth: dict) -> float:
    if not ground_truth:
        return 0.0
    scores: list[float] = []
    for name, gt_entry in ground_truth.items():
        pred_entry = prediction.get(name) if isinstance(prediction, dict) else None
        if not isinstance(pred_entry, dict):
            scores.append(0.0)
            continue
        try:
            s = pos_score(
                pred_entry["initial_vel_grid"], gt_entry["initial_vel_grid"]
            )
        except (KeyError, TypeError, ValueError):
            s = 0.0
        scores.append(s)
    return mean(scores)


def _scene_def_with_replayed_velocities(
    scene_def: SceneDef, prediction: dict
) -> SceneDef:
    """Clone ``scene_def`` with ``initial_lin_vel`` replaced from ``prediction``.

    Bodies absent from ``prediction`` (or with malformed entries) keep
    their original velocity, which is a conservative choice: the replay
    score for those bodies will naturally be high (they move as in the
    ground-truth sim) and will only harm the aggregate if the *other*
    bodies' predictions are bad.
    """

    new_bodies: list[BodyDef] = []
    for b in scene_def.bodies:
        entry = prediction.get(b.name) if isinstance(prediction, dict) else None
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("initial_vel_grid"), list)
            and len(entry["initial_vel_grid"]) == 3
        ):
            try:
                new_v = tuple(
                    bin_to_velocity(int(i)) for i in entry["initial_vel_grid"]
                )
            except (ValueError, TypeError):
                new_v = b.initial_lin_vel
        else:
            new_v = b.initial_lin_vel
        new_bodies.append(
            BodyDef(
                name=b.name,
                geom_type=b.geom_type,
                pos=b.pos,
                size=b.size,
                mass=b.mass,
                rgba=b.rgba,
                friction_slide=b.friction_slide,
                restitution=b.restitution,
                joint_type=b.joint_type,
                fixed=b.fixed,
                initial_lin_vel=tuple(float(x) for x in new_v),  # type: ignore[arg-type]
                initial_ang_vel=b.initial_ang_vel,
                initial_quat=b.initial_quat,
            )
        )
    return SceneDef(
        bodies=tuple(new_bodies),
        joints=scene_def.joints,
        cameras=scene_def.cameras,
        env=EnvConfig(
            gravity=scene_def.env.gravity,
            timestep=scene_def.env.timestep,
            duration=scene_def.env.duration,
            n_frames=scene_def.env.n_frames,
        ),
        workspace_bounds=scene_def.workspace_bounds,
    )


def _replay_inverse(
    sim_state: SimState,
    prediction: dict,
    ground_truth: dict,
    replay_timeout_s: float,
) -> float:
    """Simulate with predicted initial velocities; compare final grids.

    Any failure during replay (compile error, NaN, timeout, render
    error) short-circuits to 0.0 — the Solver shouldn't be rewarded
    for predictions that lead to non-physical scenes.
    """

    _ = ground_truth  # reference ground truth comes from sim_state

    try:
        replay_scene = _scene_def_with_replayed_velocities(
            sim_state.scene_def, prediction
        )
        # Late import: avoids circular (simulator → engine init) and
        # keeps this module import-cheap for consumers that only want
        # the direct scorer.
        from physevolve.engine.simulator import simulate

        replay_state = simulate(
            replay_scene,
            seed=0,
            sim_timeout_s=replay_timeout_s,
            render_height=64,
            render_width=64,
        )
    except Exception:  # noqa: BLE001 — any replay failure → 0 reward.
        return 0.0

    # Compare replay final grid to ground-truth (= original sim) final grid.
    scores: list[float] = []
    for name, traj in sim_state.trajectories.items():
        if name not in replay_state.trajectories:
            scores.append(0.0)
            continue
        gt_grid = world_to_grid(
            (
                float(traj.positions[-1][0]),
                float(traj.positions[-1][1]),
                float(traj.positions[-1][2]),
            )
        )
        rep_pT = replay_state.trajectories[name].positions[-1]
        rep_grid = world_to_grid(
            (float(rep_pT[0]), float(rep_pT[1]), float(rep_pT[2]))
        )
        scores.append(pos_score(rep_grid, gt_grid))
    return mean(scores)


def score_inverse(
    prediction: dict,
    ground_truth: dict,
    sim_state: SimState,
    replay_timeout_s: float = 15.0,
) -> float:
    """Dual score: ``0.5 * direct + 0.5 * replay``.

    Args:
        prediction: Parsed Solver output
            (``{body: {"initial_vel_grid": [vx, vy, vz]}}``).
        ground_truth: Ground-truth mapping of the same shape.
        sim_state: Source-of-truth :class:`SimState` used to both
            provide the reference final-state grid and supply the
            ``scene_def`` that replay will clone.
        replay_timeout_s: Wall-clock budget for the replay sim.

    Returns:
        Score in ``[0, 1]``.
    """

    direct = _direct_inverse(prediction, ground_truth)
    replay = _replay_inverse(sim_state, prediction, ground_truth, replay_timeout_s)
    return 0.5 * direct + 0.5 * replay


# ─── Parameter ───────────────────────────────────────────────────────


def score_parameter(prediction: dict, ground_truth: dict) -> float:
    """Mean of ``level_score`` over every (body, param) pair (§5.1)."""

    if not ground_truth:
        return 0.0
    pair_scores: list[float] = []
    for name, gt_entry in ground_truth.items():
        pred_entry = prediction.get(name) if isinstance(prediction, dict) else None
        if not isinstance(pred_entry, dict):
            pair_scores.extend([0.0, 0.0])
            continue
        for key in ("mass_level", "friction_level"):
            try:
                s = level_score(int(pred_entry[key]), int(gt_entry[key]))
            except (KeyError, TypeError, ValueError):
                s = 0.0
            pair_scores.append(s)
    return mean(pair_scores)


# Convenience: re-export for callers that want the canonical axis
# without reaching into `tasks.inverse`.
INVERSE_VELOCITY_BINS: tuple[float, ...] = VELOCITY_BINS


__all__ = [
    "score_forward",
    "score_inverse",
    "score_parameter",
    "INVERSE_VELOCITY_BINS",
]


# Suppress "imported but unused" warnings for np — currently used only
# transitively through metrics.pos_score, but keeping the import
# avoids a surprise if a future scorer needs it here.
_ = np
_ = Optional
