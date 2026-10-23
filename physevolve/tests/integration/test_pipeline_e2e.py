"""End-to-end smoke test: 3 hand-written Proposer programs.

These are not GT-matched golden scenes (covered by the golden-scene suite). The goal is to
verify the executor pipeline on realistic DSL programs that exercise
multiple body types, hinge joints, and collisions.
"""

from __future__ import annotations

import pytest

import physevolve.engine  # noqa: F401 — EGL env setup.
from physevolve.engine.failure import FailureReason
from physevolve.executor import run_executor


_SIM_TIMEOUT_S = 30.0


_FALLING_BALL = '''
def build_scene(api):
    api.add_sphere('ball', pos=(0.0, 0.0, 2.0), radius=0.1, mass=1.0)
    api.set_duration(1.0)
'''


_TWO_BOX_COLLISION = '''
def build_scene(api):
    api.add_box('target', pos=(0.5, 0.0, 0.2), size=(0.15, 0.15, 0.15), mass=1.0)
    api.add_box('projectile', pos=(-0.5, 0.0, 0.2), size=(0.1, 0.1, 0.1), mass=1.0)
    api.set_velocity('projectile', linear=(3.0, 0.0, 0.0))
    api.set_duration(1.5)
'''


# Hinge pendulum checks that joint children are nested under their parent.
_HINGE_PENDULUM = '''
def build_scene(api):
    api.add_box('anchor', pos=(0.0, 0.0, 2.0),
                size=(0.05, 0.05, 0.05), mass=1.0, fixed=True)
    api.add_sphere('bob', pos=(0.5, 0.0, 2.0), radius=0.05, mass=0.5)
    api.add_hinge('pivot', parent='anchor', child='bob',
                  pos=(0.0, 0.0, 2.0), axis=(0.0, 1.0, 0.0))
    api.set_duration(1.5)
'''


@pytest.mark.parametrize(
    "name,src",
    [
        ("falling_ball", _FALLING_BALL),
        ("two_box_collision", _TWO_BOX_COLLISION),
        ("hinge_pendulum", _HINGE_PENDULUM),
    ],
)
def test_e2e_stage1_ok(name: str, src: str) -> None:
    """Each scene must run OK, produce a sim_state, finish under budget."""

    r = run_executor(src, stage=1, seed=0, sim_timeout_s=_SIM_TIMEOUT_S)
    assert r.status == FailureReason.OK, f"{name}: {r.error_message}"
    assert r.sim_state is not None
    assert r.vqa_payload is None
    assert r.scene_id != ""
    assert r.walltime_s < _SIM_TIMEOUT_S
    # At least one body's trajectory recorded.
    assert len(r.sim_state.trajectories) >= 1
    # Video has the expected (n_cam, n_frames, H, W, 3) layout.
    frames = r.sim_state.video_frames
    assert frames.ndim == 5
    assert frames.shape[-1] == 3


def test_e2e_two_box_collides() -> None:
    """The projectile scene should record at least one collision event."""

    r = run_executor(_TWO_BOX_COLLISION, stage=1, seed=0, sim_timeout_s=_SIM_TIMEOUT_S)
    assert r.status == FailureReason.OK
    assert r.sim_state is not None
    # Projectile moving at 3 m/s into a target 1 m away at ground level
    # reliably produces a collision event within 1.5 s.
    assert len(r.sim_state.collisions) >= 1


def test_e2e_stage2_with_manifest() -> None:
    """Stage-2 Proposer writes build_scene + SCENE_MANIFEST on a falling ball.

    The unified executor path simulates once and derives the ball's final-state
    question via the ``collision_arena`` scene template's collision_final_state
    family (the ball settles to STABLE after dropping for 1 s).
    """

    src = '''
def build_scene(api):
    api.add_sphere('ball', pos=(0.0, 0.0, 2.0), radius=0.1, mass=1.0,
                   rgba=(0.9, 0.15, 0.15, 1.0), friction=0.5, restitution=0.0,
                   fixed=False)
    api.set_duration(1.5)

SCENE_MANIFEST = {
    'scene_kind': 'collision_arena',
    'target': 'ball',
    'families': ['collision_final_state'],
}
'''
    r = run_executor(src, stage=2, seed=0, sim_timeout_s=_SIM_TIMEOUT_S)
    assert r.status == FailureReason.OK
    assert r.vqa_payload is not None
    assert r.vqa_payloads is not None and len(r.vqa_payloads) == 1
    question, options, idx = r.vqa_payload
    assert isinstance(question, str) and question
    assert len(options) == 4
    assert 0 <= idx < 4
