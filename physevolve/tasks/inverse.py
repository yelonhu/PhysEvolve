"""Inverse-mode observation builder.

Solver sees:
    * second N/2 video frames
    * full params
    * full final state (grid pos + status per body)

Solver predicts per-axis initial-velocity bin index in [0, 6] over the
locked 7-bin axis ``[-6, -3, -1, 0, 1, 3, 6] m/s``.

Ground-truth schema:
    ``{body_name: {"initial_vel_grid": [vx_bin, vy_bin, vz_bin]}}``

The inverse scorer's replay branch dequantizes the
predicted bins using :data:`VELOCITY_BINS` and re-simulates — so this
module is also the single source of truth for the axis.
"""

from __future__ import annotations

import numpy as np

# Fixed velocity axis lives at the engine layer so both
# ``tasks.inverse`` (this module) and ``verification.dense_scorer`` can
# consume it without a cross-layer import. Re-export here to preserve
# the ``from physevolve.tasks.inverse import VELOCITY_BINS``
# surface used by callers.
from physevolve.engine.discretization import (
    VELOCITY_BINS,
    bin_to_velocity,
    velocity_to_bin,
)
from physevolve.engine.schema import SimState
from physevolve.prompts.solver import (
    INVERSE_TEMPLATE,
    _dyn_bodies,
    _format_final_block,
    _format_params_block,
)


def build_observation(
    sim_state: SimState,
) -> tuple[np.ndarray, str, dict]:
    """Assemble the Inverse observation triple."""

    n_frames = sim_state.video_frames.shape[1]
    half = n_frames // 2
    video_slice = sim_state.video_frames[:, half:]

    text_prompt = INVERSE_TEMPLATE.format(
        params_block=_format_params_block(sim_state.scene_def),
        final_block=_format_final_block(
            sim_state.scene_def, sim_state.trajectories
        ),
    )

    ground_truth: dict = {}
    for b in _dyn_bodies(sim_state.scene_def):
        vx, vy, vz = b.initial_lin_vel
        ground_truth[b.name] = {
            "initial_vel_grid": [
                velocity_to_bin(vx),
                velocity_to_bin(vy),
                velocity_to_bin(vz),
            ]
        }

    return video_slice, text_prompt, ground_truth


__all__ = [
    "VELOCITY_BINS",
    "velocity_to_bin",
    "bin_to_velocity",
    "build_observation",
]
