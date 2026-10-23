"""Parameter-mode observation builder.

Solver sees:
    * full video (all N frames)
    * full initial state (pos, initial velocity)
    * full final state (grid pos + status)

Solver predicts ``(mass_level, friction_level)`` in [1, 5] per body.

Because PhysDSL snaps every mass/friction value to
:data:`MASS_LEVELS` / :data:`FRICTION_LEVELS` at ``freeze()`` time, the
ground truth is deterministic and lossless.

Ground-truth schema:
    ``{body_name: {"mass_level": int[1..5], "friction_level": int[1..5]}}``
"""

from __future__ import annotations

import numpy as np

from physevolve.engine.schema import FRICTION_LEVELS, MASS_LEVELS, SimState
from physevolve.prompts.solver import (
    PARAMETER_TEMPLATE,
    _dyn_bodies,
    _format_final_block,
    _format_initial_block,
    _friction_level,
    _mass_level,
)


def build_observation(
    sim_state: SimState,
) -> tuple[np.ndarray, str, dict]:
    """Assemble the Parameter observation triple."""

    video_slice = sim_state.video_frames  # full (n_cam, n_frames, H, W, 3)

    text_prompt = PARAMETER_TEMPLATE.format(
        initial_block=_format_initial_block(sim_state.scene_def),
        final_block=_format_final_block(
            sim_state.scene_def, sim_state.trajectories
        ),
        mass_levels=list(MASS_LEVELS),
        friction_levels=list(FRICTION_LEVELS),
    )

    ground_truth: dict = {}
    for b in _dyn_bodies(sim_state.scene_def):
        ground_truth[b.name] = {
            "mass_level": _mass_level(b.mass),
            "friction_level": _friction_level(b.friction_slide),
        }

    return video_slice, text_prompt, ground_truth


__all__ = ["build_observation"]
