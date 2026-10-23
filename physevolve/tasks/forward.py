"""Forward-mode observation builder.

Solver sees:
    * first N/2 video frames
    * full params (mass, friction per body)
    * full initial conditions (pos, velocity per body)

Solver predicts final ``(grid, status)`` per body.

Ground-truth schema:
    ``{body_name: {"grid": [gx, gy, gz], "status": "stable"|...}}``
"""

from __future__ import annotations

import numpy as np

from physevolve.engine.schema import SimState, world_to_grid
from physevolve.prompts.solver import (
    FORWARD_TEMPLATE,
    _dyn_bodies,
    _format_initial_block,
    _format_params_block,
)


def build_observation(
    sim_state: SimState,
) -> tuple[np.ndarray, str, dict]:
    """Assemble the Forward observation triple.

    Args:
        sim_state: Full engine output. Scene_def, trajectories, and
            video_frames are consumed.

    Returns:
        ``(video_slice, text_prompt, ground_truth)`` where ``video_slice``
        is ``(n_cam, n_frames // 2, H, W, 3)`` uint8, ``text_prompt``
        is the fully-formatted Solver prompt, and ``ground_truth`` is
        the dict the scorer consumes.
    """

    n_frames = sim_state.video_frames.shape[1]
    half = n_frames // 2
    video_slice = sim_state.video_frames[:, :half]

    text_prompt = FORWARD_TEMPLATE.format(
        params_block=_format_params_block(sim_state.scene_def),
        initial_block=_format_initial_block(sim_state.scene_def),
    )

    ground_truth: dict = {}
    for b in _dyn_bodies(sim_state.scene_def):
        traj = sim_state.trajectories[b.name]
        pT = traj.positions[-1]
        grid = world_to_grid((float(pT[0]), float(pT[1]), float(pT[2])))
        ground_truth[b.name] = {
            "grid": list(grid),
            "status": traj.final_status.value,
        }

    return video_slice, text_prompt, ground_truth


__all__ = ["build_observation"]
