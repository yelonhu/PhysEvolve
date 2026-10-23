"""Unit tests for :mod:`physevolve.engine.renderer`.

Tests ``engine/renderer.py``: shape / dtype,
determinism, multi-camera axis. All tests run at a tiny config
(64×64, 4 frames) to keep CI wall-clock low.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from physevolve.engine.compiler import compile_to_mjcf
from physevolve.engine.physdsl import PhysDSLApi
from physevolve.engine.renderer import _clear_renderer_cache, render_multiview


@pytest.fixture()
def single_cam_model() -> tuple[mujoco.MjModel, tuple[str, ...]]:
    api = PhysDSLApi()
    api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.1)
    scene = api.freeze()
    model = mujoco.MjModel.from_xml_string(compile_to_mjcf(scene))
    cams = tuple(c.name for c in scene.cameras)
    return model, cams


@pytest.fixture()
def two_cam_model() -> tuple[mujoco.MjModel, tuple[str, ...]]:
    api = PhysDSLApi()
    api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.1)
    api.add_camera("front", pos=(0.0, -1.5, 1.0), target=(0.0, 0.0, 0.5))
    api.add_camera("side", pos=(1.5, 0.0, 1.0), target=(0.0, 0.0, 0.5))
    scene = api.freeze()
    model = mujoco.MjModel.from_xml_string(compile_to_mjcf(scene))
    cams = tuple(c.name for c in scene.cameras)
    return model, cams


def _make_snapshots(
    model: mujoco.MjModel, n_frames: int
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return [
        (data.qpos.copy(), data.qvel.copy(), float(i) * 0.01)
        for i in range(n_frames)
    ]


class TestRenderer:
    def test_shape_and_dtype(
        self, single_cam_model: tuple[mujoco.MjModel, tuple[str, ...]]
    ) -> None:
        _clear_renderer_cache()
        model, cams = single_cam_model
        snaps = _make_snapshots(model, n_frames=4)
        frames = render_multiview(
            model, snaps, cams, height=64, width=64
        )
        assert frames.shape == (len(cams), 4, 64, 64, 3)
        assert frames.dtype == np.uint8

    def test_determinism(
        self, single_cam_model: tuple[mujoco.MjModel, tuple[str, ...]]
    ) -> None:
        """Two renders of the same scene must be bit-near-identical.

        Note: EGL drivers occasionally flip one LSB between back-to-back
        renders due to tiny FP non-associativity inside the raster
        pipeline. We allow ``max|a - b| <= 2`` (well below any perceptual
        threshold) rather than require byte-exact output.
        """

        _clear_renderer_cache()
        model, cams = single_cam_model
        snaps = _make_snapshots(model, n_frames=4)
        a = render_multiview(model, snaps, cams, height=64, width=64)
        b = render_multiview(model, snaps, cams, height=64, width=64)
        assert a.shape == b.shape
        max_abs = int(np.abs(a.astype(int) - b.astype(int)).max())
        assert max_abs <= 2, (
            f"renderer drifted by {max_abs} LSB; expected <= 2"
        )

    def test_multi_camera_axis(
        self, two_cam_model: tuple[mujoco.MjModel, tuple[str, ...]]
    ) -> None:
        _clear_renderer_cache()
        model, cams = two_cam_model
        assert len(cams) == 2
        snaps = _make_snapshots(model, n_frames=3)
        frames = render_multiview(model, snaps, cams, height=64, width=64)
        assert frames.shape[0] == 2
        # Two different viewpoints on the same non-symmetric scene
        # should not produce byte-identical images.
        assert not np.array_equal(frames[0], frames[1])

    def test_zero_frames_rejected(
        self, single_cam_model: tuple[mujoco.MjModel, tuple[str, ...]]
    ) -> None:
        model, cams = single_cam_model
        with pytest.raises(ValueError):
            render_multiview(model, [], cams, height=64, width=64)

    def test_zero_cameras_rejected(
        self, single_cam_model: tuple[mujoco.MjModel, tuple[str, ...]]
    ) -> None:
        model, _ = single_cam_model
        snaps = _make_snapshots(model, n_frames=1)
        with pytest.raises(ValueError):
            render_multiview(model, snaps, (), height=64, width=64)
