"""Tests for :mod:`physevolve.integrations.video_bridge`.

Scope:

* Shape / dtype / metadata-key contract.
* Frame budget (F=[0:16], I=[16:32], P=[0:32]).
* Short-sim robustness (fewer than 32 frames → clipped slice, no raise).
* Input tensor-shape flexibility (4-D and 5-D both accepted).
* Observation-level API mirrors the SimState-level API.
* Error paths: bad dtype, bad ndim, bad camera_idx.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import physevolve.engine  # noqa: F401 — EGL env setup.
from physevolve.engine.schema import (
    BodyDef,
    BodyStatus,
    BodyTrajectory,
    EnvConfig,
    GeomType,
    JointType,
    SceneDef,
    SimState,
    compute_scene_id,
)
from physevolve.integrations.video_bridge import (
    VLLM_VIDEO_METADATA_KEYS,
    VideoBundle,
    sim_state_to_vllm_video,
    video_bundle_from_observation,
)
from physevolve.tasks.router import SolverObservation, TaskMode


# ─── Fixtures ─────────────────────────────────────────────────────────


def _make_video(n_cam: int = 1, n_frames: int = 32, h: int = 24, w: int = 24) -> np.ndarray:
    """Deterministic video tensor where ``frame[t, 0, 0, 0] == t``.

    This lets assertions verify that exactly the right temporal slice
    was picked: the bytes at ``subset[i, 0, 0, 0]`` must equal the
    original frame index.
    """

    vid = np.zeros((n_cam, n_frames, h, w, 3), dtype=np.uint8)
    for cam in range(n_cam):
        for t in range(n_frames):
            vid[cam, t, :, :, :] = t % 256
    return vid


def _make_sim_state(
    video: np.ndarray, *, duration: float = 8.0, n_frames: int = 32
) -> SimState:
    body = BodyDef(
        name="ball",
        geom_type=GeomType.SPHERE,
        pos=(0.0, 0.0, 1.0),
        size=(0.1,),
        mass=1.0,
        rgba=(1.0, 0.0, 0.0, 1.0),
        friction_slide=0.2,
        restitution=0.0,
        joint_type=JointType.FREE,
    )
    env = EnvConfig(
        gravity=(0.0, 0.0, -9.81),
        timestep=0.002,
        duration=duration,
        n_frames=n_frames,
    )
    scene = SceneDef(
        bodies=(body,),
        joints=(),
        cameras=(),
        env=env,
        workspace_bounds=(1.5, 1.5, 3.0),
    )
    positions = np.zeros((4, 3), dtype=np.float64)
    orient = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (4, 1))
    vels = np.zeros((4, 3), dtype=np.float64)
    traj = BodyTrajectory(
        name="ball",
        positions=positions,
        orientations=orient,
        lin_velocities=vels,
        ang_velocities=vels,
        final_status=BodyStatus.STABLE,
    )
    return SimState(
        scene_id=compute_scene_id(scene, 0),
        scene_def=scene,
        trajectories={"ball": traj},
        collisions=(),
        video_frames=video,
        camera_names=("cam0",),
        sim_walltime_s=0.1,
        initial_positions={"ball": np.array([0.0, 0.0, 1.0])},
        initial_lin_velocities={"ball": np.array([0.0, 0.0, 0.0])},
    )


# ─── Metadata contract ────────────────────────────────────────────────


class TestMetadataContract:
    """Each VideoBundle must carry exactly the video metadata keys."""

    def test_forward_metadata_keys(self) -> None:
        sim = _make_sim_state(_make_video())
        bundle = sim_state_to_vllm_video(sim, TaskMode.FORWARD)
        assert set(bundle.metadata.keys()) == set(VLLM_VIDEO_METADATA_KEYS)

    def test_metadata_values(self) -> None:
        sim = _make_sim_state(_make_video(n_frames=32), duration=8.0, n_frames=32)
        bundle = sim_state_to_vllm_video(sim, TaskMode.PARAMETER)
        m = bundle.metadata
        assert m["total_num_frames"] == 32
        assert m["width"] == 24
        assert m["height"] == 24
        assert m["fps"] == pytest.approx(4.0)  # 32 frames / 8 s
        assert m["duration"] == pytest.approx(8.0)
        assert m["do_sample_frames"] is False
        assert m["frames_indices"] == list(range(0, 32))

    def test_bundle_as_tuple_copies_metadata(self) -> None:
        sim = _make_sim_state(_make_video())
        bundle = sim_state_to_vllm_video(sim, TaskMode.FORWARD)
        frames, meta = bundle.as_tuple()
        meta["total_num_frames"] = -1  # mutate the copy
        assert bundle.metadata["total_num_frames"] != -1


# ─── Frame-budget per mode ──────────────────────────────────────


class TestFrameBudget:
    """FORWARD=16, INVERSE=16, PARAMETER=32 on a 32-frame input."""

    @pytest.mark.parametrize(
        "mode,expected_start,expected_end",
        [
            (TaskMode.FORWARD, 0, 16),
            (TaskMode.INVERSE, 16, 32),
            (TaskMode.PARAMETER, 0, 32),
        ],
    )
    def test_frame_budget(self, mode, expected_start, expected_end) -> None:
        sim = _make_sim_state(_make_video(n_frames=32))
        bundle = sim_state_to_vllm_video(sim, mode)
        assert bundle.n_frames == expected_end - expected_start
        # The deterministic video encodes frame idx into byte [t,0,0,0],
        # so verify the first & last bytes match the expected indices.
        assert int(bundle.frames[0, 0, 0, 0]) == expected_start
        assert int(bundle.frames[-1, 0, 0, 0]) == expected_end - 1
        assert bundle.metadata["frames_indices"] == list(
            range(expected_start, expected_end)
        )

    def test_short_sim_is_clipped_not_raised(self) -> None:
        """A 12-frame sim should still work for FORWARD (12 frames) and
        INVERSE (0 frames) without an exception."""

        sim = _make_sim_state(_make_video(n_frames=12), duration=3.0, n_frames=12)
        fwd = sim_state_to_vllm_video(sim, TaskMode.FORWARD)
        inv = sim_state_to_vllm_video(sim, TaskMode.INVERSE)
        par = sim_state_to_vllm_video(sim, TaskMode.PARAMETER)
        assert fwd.n_frames == 12
        assert inv.n_frames == 0
        assert par.n_frames == 12


# ─── Shape / dtype contract ───────────────────────────────────────────


class TestTensorShape:
    def test_output_is_4d_uint8_contiguous(self) -> None:
        sim = _make_sim_state(_make_video())
        bundle = sim_state_to_vllm_video(sim, TaskMode.FORWARD)
        assert bundle.frames.ndim == 4
        assert bundle.frames.dtype == np.uint8
        assert bundle.frames.shape[-1] == 3
        assert bundle.frames.flags["C_CONTIGUOUS"]

    def test_invalid_mode_raises(self) -> None:
        sim = _make_sim_state(_make_video())
        with pytest.raises(TypeError, match="mode must be TaskMode"):
            sim_state_to_vllm_video(sim, "forward")  # type: ignore[arg-type]

    def test_invalid_camera_idx_raises(self) -> None:
        sim = _make_sim_state(_make_video(n_cam=2))
        with pytest.raises(IndexError):
            sim_state_to_vllm_video(sim, TaskMode.FORWARD, camera_idx=5)


# ─── Observation-level API ────────────────────────────────────────────


class TestObservationAPI:
    """``video_bundle_from_observation`` must produce the same contract
    as the SimState-level path."""

    def _make_observation(self, mode: TaskMode) -> SolverObservation:
        sim = _make_sim_state(_make_video())
        # Build a stub observation whose video payload has already been
        # trimmed to the mode's frame budget (mirrors what the real
        # per-mode builders in ``physevolve/tasks/*`` do).
        per_cam = sim.video_frames[0]
        if mode == TaskMode.FORWARD:
            slice_ = per_cam[:16]
        elif mode == TaskMode.INVERSE:
            slice_ = per_cam[16:]
        else:
            slice_ = per_cam
        return SolverObservation(
            scene_id=sim.scene_id,
            mode=mode,
            video_frames=slice_,
            text_prompt="(stub prompt)",
            ground_truth={},
        )

    @pytest.mark.parametrize(
        "mode,expected",
        [
            (TaskMode.FORWARD, 16),
            (TaskMode.INVERSE, 16),
            (TaskMode.PARAMETER, 32),
        ],
    )
    def test_n_frames_matches_mode(self, mode, expected) -> None:
        obs = self._make_observation(mode)
        bundle = video_bundle_from_observation(obs)
        assert bundle.n_frames == expected

    def test_metadata_indices_back_half_for_inverse(self) -> None:
        obs = self._make_observation(TaskMode.INVERSE)
        bundle = video_bundle_from_observation(obs)
        assert bundle.metadata["frames_indices"][0] == 16

    def test_metadata_indices_front_half_for_forward(self) -> None:
        obs = self._make_observation(TaskMode.FORWARD)
        bundle = video_bundle_from_observation(obs)
        assert bundle.metadata["frames_indices"][0] == 0

    def test_accepts_explicit_fps_override(self) -> None:
        obs = self._make_observation(TaskMode.PARAMETER)
        bundle = video_bundle_from_observation(obs, fps=16.0, total_num_frames=32)
        assert bundle.metadata["fps"] == pytest.approx(16.0)
        assert bundle.metadata["duration"] == pytest.approx(2.0)


# ─── Input validation ────────────────────────────────────────────────


class TestInputValidation:
    def test_rejects_float_dtype(self) -> None:
        sim = _make_sim_state(_make_video())
        bad = SimState(
            scene_id=sim.scene_id,
            scene_def=sim.scene_def,
            trajectories=sim.trajectories,
            collisions=sim.collisions,
            video_frames=sim.video_frames.astype(np.float32),
            camera_names=sim.camera_names,
            sim_walltime_s=sim.sim_walltime_s,
            initial_positions=sim.initial_positions,
            initial_lin_velocities=sim.initial_lin_velocities,
        )
        with pytest.raises(ValueError, match="dtype must be uint8"):
            sim_state_to_vllm_video(bad, TaskMode.FORWARD)

    def test_rejects_3d_video(self) -> None:
        obs = SolverObservation(
            scene_id="x",
            mode=TaskMode.FORWARD,
            video_frames=np.zeros((16, 24, 3), dtype=np.uint8),
            text_prompt="p",
            ground_truth={},
        )
        with pytest.raises(ValueError, match="frames must be 4-D"):
            video_bundle_from_observation(obs)
