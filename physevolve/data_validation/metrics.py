"""Deterministic quality metrics derived from executor ``SimState``."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from physevolve.engine.schema import (
    FRICTION_LEVELS,
    MASS_LEVELS,
    RESTITUTION_LEVELS,
    BodyStatus,
    SimState,
)


def compute_quality_metrics(sim_state: SimState) -> dict[str, Any]:
    """Compute JSON-friendly physical and visual statistics for a scene."""

    scene = sim_state.scene_def
    bodies = tuple(scene.bodies)
    dynamic_bodies = tuple(b for b in bodies if not b.fixed)
    fixed_bodies = tuple(b for b in bodies if b.fixed)

    geom_counts = Counter(b.geom_type.value for b in bodies)
    joint_counts = Counter(j.joint_type.value for j in scene.joints)
    collision_pairs = {
        tuple(sorted((event.body_a, event.body_b))) for event in sim_state.collisions
    }

    displacement_maxima: list[float] = []
    displacement_means: list[float] = []
    speed_values: list[np.ndarray] = []
    final_statuses: list[str] = []
    for traj in sim_state.trajectories.values():
        positions = np.asarray(traj.positions, dtype=np.float64)
        if positions.size > 0:
            deltas = positions - positions[0]
            disp = np.linalg.norm(deltas, axis=1)
            displacement_maxima.append(float(disp.max()))
            displacement_means.append(float(disp.mean()))
        velocities = np.asarray(traj.lin_velocities, dtype=np.float64)
        if velocities.size > 0:
            speed_values.append(np.linalg.norm(velocities, axis=1))
        final_statuses.append(traj.final_status.value)

    if displacement_maxima:
        max_displacement = float(max(displacement_maxima))
        mean_displacement = float(np.mean(displacement_means))
        median_displacement = float(np.median(displacement_maxima))
    else:
        max_displacement = 0.0
        mean_displacement = 0.0
        median_displacement = 0.0

    if speed_values:
        speeds = np.concatenate(speed_values)
        max_speed = float(speeds.max())
        mean_speed = float(speeds.mean())
    else:
        max_speed = 0.0
        mean_speed = 0.0

    video = sim_state.video_frames
    video_complete = (
        isinstance(video, np.ndarray)
        and video.ndim == 5
        and video.shape[1] == scene.env.n_frames
        and video.shape[-1] == 3
        and video.dtype == np.uint8
    )

    n_dynamic = len(dynamic_bodies)
    n_ejected = sum(status == BodyStatus.EJECTED.value for status in final_statuses)
    out_of_frame_ratio = float(n_ejected / n_dynamic) if n_dynamic else 0.0

    mass_levels = _levels_present((b.mass for b in dynamic_bodies), MASS_LEVELS)
    friction_levels = _levels_present(
        (b.friction_slide for b in dynamic_bodies), FRICTION_LEVELS
    )
    restitution_levels = _levels_present(
        (b.restitution for b in dynamic_bodies), RESTITUTION_LEVELS
    )

    return {
        "body_count": len(bodies),
        "dynamic_body_count": n_dynamic,
        "fixed_body_count": len(fixed_bodies),
        "geom_type_counts": dict(sorted(geom_counts.items())),
        "joint_type_counts": dict(sorted(joint_counts.items())),
        "has_collision": len(sim_state.collisions) > 0,
        "collision_count": len(sim_state.collisions),
        "unique_collision_pairs": len(collision_pairs),
        "max_displacement": max_displacement,
        "mean_displacement": mean_displacement,
        "median_displacement": median_displacement,
        "max_speed": max_speed,
        "mean_speed": mean_speed,
        "has_spring": len(scene.springs) > 0,
        "has_hinge": any(j.joint_type.value == "hinge" for j in scene.joints),
        "has_slide": any(j.joint_type.value == "slide" for j in scene.joints),
        "mass_levels_present": mass_levels,
        "friction_levels_present": friction_levels,
        "restitution_levels_present": restitution_levels,
        "n_cameras": len(sim_state.camera_names),
        "n_frames": int(video.shape[1]) if isinstance(video, np.ndarray) and video.ndim >= 2 else 0,
        "video_complete": video_complete,
        "out_of_frame_ratio": out_of_frame_ratio,
        "color_diversity_score": _color_diversity(dynamic_bodies),
        "final_status_counts": dict(sorted(Counter(final_statuses).items())),
    }


def _levels_present(values, levels: tuple[float, ...]) -> list[int]:
    """Return sorted 1-indexed nearest locked levels present in values."""

    present: set[int] = set()
    for value in values:
        diffs = [abs(float(value) - level) for level in levels]
        present.add(int(np.argmin(diffs)) + 1)
    return sorted(present)


def _color_diversity(dynamic_bodies) -> float:
    """Heuristic diversity score based on rounded RGBA colors."""

    bodies = tuple(dynamic_bodies)
    if not bodies:
        return 0.0
    colors = {
        tuple(round(float(channel), 2) for channel in body.rgba)
        for body in bodies
    }
    if len(bodies) == 1:
        return 0.5
    return float(min(1.0, len(colors) / len(bodies)))


__all__ = ["compute_quality_metrics"]
