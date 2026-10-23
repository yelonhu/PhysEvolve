"""Tests for Stage-2 VQA router pieces (docs/architecture.md).

Covers:

* :class:`physevolve.tasks.router.TaskMode` includes ``VQA``.
* :func:`physevolve.tasks.router.parse_solver_output` extracts a single
  A/B/C/D letter for ``TaskMode.VQA`` and rejects multi-letter / JSON /
  empty outputs.
* :func:`physevolve.tasks.vqa.build_observation` renders the locked
  VQA template, returns a ``ground_truth`` dict with the expected
  fields, and validates payload shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from physevolve.engine.schema import (
    BodyDef,
    BodyStatus,
    BodyTrajectory,
    EnvConfig,
    GeomType,
    JointType,
    SceneDef,
    SimState,
)
from physevolve.tasks.router import TaskMode, parse_solver_output
from physevolve.tasks.vqa import build_observation as build_vqa_observation


# ─── Fake SimState helper ────────────────────────────────────────────


def _fake_sim_state(n_frames: int = 32) -> SimState:
    """Minimal SimState with one body and one camera, enough for the VQA
    observation builder.

    The builder forwards ``video_frames`` verbatim and reads no
    trajectory fields, so we can fabricate the smallest valid object.
    """

    body = BodyDef(
        name="ball",
        geom_type=GeomType.SPHERE,
        pos=(0.0, 0.0, 1.0),
        size=(0.1,),
        mass=1.0,
        rgba=(1.0, 0.0, 0.0, 1.0),
        friction_slide=0.5,
        restitution=0.0,
        joint_type=JointType.FREE,
        fixed=False,
        initial_lin_vel=(0.0, 0.0, 0.0),
        initial_ang_vel=(0.0, 0.0, 0.0),
        initial_quat=(1.0, 0.0, 0.0, 0.0),
    )
    scene_def = SceneDef(
        bodies=(body,),
        joints=(),
        cameras=(),
        env=EnvConfig(
            gravity=(0.0, 0.0, -9.81),
            timestep=0.01,
            duration=1.0,
            n_frames=n_frames,
        ),
        workspace_bounds=(1.5, 1.5, 2.5),
    )
    positions = np.tile(np.array([0.0, 0.0, 1.0]), (n_frames, 1))
    lin_velocities = np.zeros((n_frames, 3))
    ang_velocities = np.zeros((n_frames, 3))
    orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, 1))
    traj = BodyTrajectory(
        name="ball",
        positions=positions,
        lin_velocities=lin_velocities,
        ang_velocities=ang_velocities,
        orientations=orientations,
        final_status=BodyStatus.STABLE,
    )
    video = np.zeros((1, n_frames, 16, 16, 3), dtype=np.uint8)
    return SimState(
        scene_id="abc123",
        scene_def=scene_def,
        trajectories={"ball": traj},
        collisions=(),
        video_frames=video,
        camera_names=("default_cam",),
        sim_walltime_s=0.0,
        initial_positions={"ball": np.array([0.0, 0.0, 1.0])},
        initial_lin_velocities={"ball": np.array([0.0, 0.0, 0.0])},
    )


# ─── TaskMode.VQA presence ───────────────────────────────────────────


def test_task_mode_vqa_exists() -> None:
    assert TaskMode.VQA.value == "vqa"
    # Make sure adding VQA did not change Stage-1 modes.
    assert {m.value for m in TaskMode} == {"forward", "inverse", "parameter", "vqa"}


# ─── parse_solver_output for VQA ─────────────────────────────────────


@pytest.mark.parametrize(
    "raw,letter,idx",
    [
        ("<answer>A</answer>", "A", 0),
        ("<answer>b</answer>", "B", 1),
        ("blah<answer>\nC\n</answer>tail", "C", 2),
        ("<answer>D.</answer>", "D", 3),
        ("<answer> a </answer>", "A", 0),
    ],
)
def test_parse_vqa_accepts_letter(raw: str, letter: str, idx: int) -> None:
    parsed = parse_solver_output(raw, TaskMode.VQA)
    assert parsed == {"letter": letter, "predicted_idx": idx}


@pytest.mark.parametrize(
    "raw",
    [
        "no answer block at all",
        "<answer></answer>",
        "<answer>AB</answer>",
        "<answer>1</answer>",
        '<answer>{"letter": "A"}</answer>',
        "<answer>E</answer>",
        "<answer>A B</answer>",
    ],
)
def test_parse_vqa_rejects_malformed(raw: str) -> None:
    assert parse_solver_output(raw, TaskMode.VQA) is None


# ─── build_vqa_observation contract ──────────────────────────────────


def test_build_observation_renders_template_and_ground_truth() -> None:
    sim_state = _fake_sim_state()
    payload = (
        "Which ball will reach the ground first?",
        ["red sphere", "blue cube", "neither — they tie", "they will not fall"],
        2,
    )
    video, text, gt = build_vqa_observation(sim_state, payload)

    assert video is sim_state.video_frames
    assert "Which ball will reach the ground first?" in text
    assert "A. red sphere" in text
    assert "B. blue cube" in text
    assert "C. neither — they tie" in text
    assert "D. they will not fall" in text
    assert "Answer with the option's letter from the given choices directly" in text
    assert "3-5 concise sentences" not in text
    assert "<answer>" in text and "</answer>" in text

    assert gt == {
        "correct_idx": 2,
        "correct_letter": "C",
        "options": ["red sphere", "blue cube", "neither — they tie", "they will not fall"],
        "question": "Which ball will reach the ground first?",
    }


@pytest.mark.parametrize(
    "payload",
    [
        ("", ["a", "b", "c", "d"], 0),
        ("q?", ["a", "b", "c"], 0),
        ("q?", ["a", "b", "c", ""], 0),
        ("q?", ["a", "b", "c", "d"], -1),
        ("q?", ["a", "b", "c", "d"], 4),
        ("q?", ["a", "b", "c", "d"], True),
    ],
)
def test_build_observation_rejects_invalid_payload(payload: tuple[Any, Any, Any]) -> None:
    sim_state = _fake_sim_state()
    with pytest.raises(ValueError):
        build_vqa_observation(sim_state, payload)  # type: ignore[arg-type]


def test_assemble_observation_requires_payload_for_vqa() -> None:
    from physevolve.tasks.router import assemble_observation

    sim_state = _fake_sim_state()
    with pytest.raises(ValueError, match="vqa_payload"):
        assemble_observation(sim_state, TaskMode.VQA)
