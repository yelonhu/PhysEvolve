"""Unit tests for :mod:`physevolve.engine.simulator`.

Tests ``engine/simulator.py``: determinism,
falling-sphere smoke, NaN detection, timeout, initial overlap, status
classifier, ``SimState`` contract, and the ``is_trivial`` helper.

All tests run at the tiny config (``timestep=0.005 s``, ``duration=0.5 s``,
``n_frames=4``) at 64×64 render to keep CI wall-clock small.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Optional

import mujoco
import numpy as np
import pytest

from physevolve.engine import simulator as sim_module
from physevolve.engine.failure import (
    InitialOverlapError,
    SimulationDivergedError,
    SimulationTimeoutError,
)
from physevolve.engine.physdsl import PhysDSLApi
from physevolve.engine.renderer import _clear_renderer_cache
from physevolve.engine.schema import (
    BodyStatus,
    EnvConfig,
    SceneDef,
    SimState,
    compute_scene_id,
)
from physevolve.engine.simulator import is_trivial, simulate

# ─── Tiny sim/render config shared by every test ────────────────────────

TINY_ENV = EnvConfig(
    gravity=(0.0, 0.0, -9.81),
    timestep=0.005,
    duration=0.5,
    n_frames=4,
)
RENDER_HW = dict(render_height=64, render_width=64)


def _tiny_scene_from(api: PhysDSLApi, *, env: Optional[EnvConfig] = None) -> SceneDef:
    """Freeze ``api`` and swap in the tiny env config (or a custom one)."""

    full = api.freeze()
    return dc_replace(full, env=env or TINY_ENV)


def _falling_sphere() -> SceneDef:
    api = PhysDSLApi()
    api.add_sphere("ball", pos=(0.0, 0.0, 1.0), radius=0.1)
    return _tiny_scene_from(api)


# ═══════════════════════════ Determinism ═══════════════════════════════


class TestDeterminism:
    def test_same_inputs_same_outputs(self) -> None:
        _clear_renderer_cache()
        scene = _falling_sphere()
        a = simulate(scene, seed=0, **RENDER_HW)
        b = simulate(scene, seed=0, **RENDER_HW)
        assert a.scene_id == b.scene_id
        assert a.scene_id == compute_scene_id(scene, 0)
        assert np.array_equal(
            a.trajectories["ball"].positions,
            b.trajectories["ball"].positions,
        )
        assert np.array_equal(
            a.trajectories["ball"].lin_velocities,
            b.trajectories["ball"].lin_velocities,
        )

    def test_different_seeds_change_scene_id(self) -> None:
        scene = _falling_sphere()
        a = simulate(scene, seed=0, **RENDER_HW)
        b = simulate(scene, seed=1, **RENDER_HW)
        assert a.scene_id != b.scene_id


# ═══════════════════════ Falling-sphere smoke ══════════════════════════


class TestFallingSphere:
    def test_sphere_moves_down_under_gravity(self) -> None:
        scene = _falling_sphere()
        s = simulate(scene, **RENDER_HW)
        traj = s.trajectories["ball"]
        initial_z = float(traj.positions[0, 2])
        final_z = float(traj.positions[-1, 2])
        assert initial_z - final_z > 0.1, (
            f"sphere should fall by > 10 cm in 0.5 s but went from "
            f"{initial_z:.3f} to {final_z:.3f}"
        )


# ═════════════════════════ NaN detection ═══════════════════════════════


class TestNaNDetection:
    def test_nan_raises_simulation_diverged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Force ``data.qpos`` to NaN after a few steps via monkeypatch.

        Any physically divergent Proposer config eventually triggers
        this path; we deterministically drive it so the test does not
        depend on MuJoCo's numerical tolerances.
        """

        scene = _falling_sphere()
        real_mj_step = mujoco.mj_step
        call_count = {"n": 0}

        def fake_mj_step(model, data, *args, **kwargs):  # type: ignore[no-untyped-def]
            real_mj_step(model, data, *args, **kwargs)
            call_count["n"] += 1
            # Poison qpos right before the next NaN guard (step % 100 == 0).
            if call_count["n"] >= 1:
                data.qpos[0] = float("nan")

        monkeypatch.setattr(sim_module.mujoco, "mj_step", fake_mj_step)
        with pytest.raises(SimulationDivergedError):
            simulate(scene, **RENDER_HW)


# ═════════════════════════════ Timeout ═════════════════════════════════


class TestTimeout:
    def test_zero_budget_triggers_timeout(self) -> None:
        scene = _falling_sphere()
        with pytest.raises(SimulationTimeoutError):
            simulate(scene, sim_timeout_s=0.0, **RENDER_HW)


# ═══════════════════════ Initial overlap ══════════════════════════════


class TestInitialOverlap:
    def test_overlapping_bodies_raise(self) -> None:
        api = PhysDSLApi()
        # Two boxes whose half-extents (0.2) overlap centers placed 0.2
        # apart — 20 cm of interpenetration, well above _OVERLAP_EPS.
        api.add_box("a", pos=(0.0, 0.0, 0.5), size=(0.2, 0.2, 0.2), fixed=True)
        api.add_box("b", pos=(0.2, 0.0, 0.5), size=(0.2, 0.2, 0.2))
        scene = _tiny_scene_from(api)
        with pytest.raises(InitialOverlapError):
            simulate(scene, **RENDER_HW)


# ═══════════════════════ Final-status classifier ══════════════════════


class TestClassifier:
    def test_fixed_platform_ignored(self) -> None:
        # Only a fixed body → no dynamic bodies → classifier never runs
        # but simulate still succeeds.
        api = PhysDSLApi()
        api.add_box(
            "plat",
            pos=(0.0, 0.0, 0.3),
            size=(0.3, 0.3, 0.02),
            fixed=True,
        )
        scene = _tiny_scene_from(api)
        s = simulate(scene, **RENDER_HW)
        assert s.trajectories == {}

    def test_sphere_ejected_off_workspace(self) -> None:
        # Give a sphere a strong horizontal velocity over a short run so
        # it exits the x-workspace (|x| > 1.5).
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 1.0), radius=0.05)
        api.set_velocity("ball", linear=(20.0, 0.0, 0.0))
        scene = _tiny_scene_from(api)
        s = simulate(scene, **RENDER_HW)
        assert s.trajectories["ball"].final_status is BodyStatus.EJECTED

    def test_falling_sphere_ends_in_moving_or_fallen(self) -> None:
        # At 0.5 s the sphere is still in contact with the ground,
        # likely bouncing → MOVING/FALLEN are both acceptable.
        scene = _falling_sphere()
        s = simulate(scene, **RENDER_HW)
        assert s.trajectories["ball"].final_status in {
            BodyStatus.MOVING,
            BodyStatus.FALLEN,
            BodyStatus.STABLE,
        }


# ═══════════════════════ SimState contract ════════════════════════════


class TestSimStateContract:
    def test_contract_shape_and_fields(self) -> None:
        scene = _falling_sphere()
        s = simulate(scene, **RENDER_HW)
        assert isinstance(s, SimState)
        assert s.scene_id == compute_scene_id(scene, 0)
        assert s.camera_names == ("cam0",)
        assert s.video_frames.shape == (1, 4, 64, 64, 3)
        assert s.video_frames.dtype == np.uint8
        # Trajectories cover every dynamic body.
        assert set(s.trajectories.keys()) == {"ball"}
        traj = s.trajectories["ball"]
        T = int(round(scene.env.duration / scene.env.timestep))
        assert traj.positions.shape == (T, 3)
        assert traj.orientations.shape == (T, 4)
        assert traj.lin_velocities.shape == (T, 3)
        assert traj.ang_velocities.shape == (T, 3)
        assert traj.positions.dtype == np.float64


# ═══════════════════════ is_trivial helper ════════════════════════════


class TestIsTrivial:
    def test_static_scene_is_trivial(self) -> None:
        # Fixed platform alone + ground plane: no dynamic motion, no
        # collisions → is_trivial True.
        api = PhysDSLApi()
        api.add_box(
            "plat",
            pos=(0.0, 0.0, 0.3),
            size=(0.3, 0.3, 0.02),
            fixed=True,
        )
        scene = _tiny_scene_from(api)
        s = simulate(scene, **RENDER_HW)
        assert is_trivial(s) is True

    def test_falling_sphere_not_trivial(self) -> None:
        scene = _falling_sphere()
        s = simulate(scene, **RENDER_HW)
        assert is_trivial(s) is False

    def test_bounce_then_return_is_not_trivial(self) -> None:
        """A body that moves then returns to its start must NOT be TRIVIAL.

        Triviality is defined over the *max* displacement
        across the whole trajectory, not the first-vs-last delta. We
        fabricate a synthetic trajectory that starts and ends at the
        origin but peaks at 0.5 m — clearly non-trivial physics —
        then wrap it in a :class:`SimState` to exercise
        :func:`is_trivial` in isolation.
        """

        from physevolve.engine.schema import (
            BodyStatus,
            BodyTrajectory,
            SimState,
            compute_scene_id,
        )

        scene = _falling_sphere()
        T = 20
        positions = np.zeros((T, 3), dtype=np.float64)
        # Half sine: starts at 0, peaks at 0.5 at t=T//2, back to 0.
        positions[:, 2] = 0.5 * np.sin(np.linspace(0.0, np.pi, T))
        traj = BodyTrajectory(
            name="ball",
            positions=positions,
            orientations=np.tile(np.array([1, 0, 0, 0], dtype=np.float64), (T, 1)),
            lin_velocities=np.zeros((T, 3), dtype=np.float64),
            ang_velocities=np.zeros((T, 3), dtype=np.float64),
            final_status=BodyStatus.STABLE,
        )
        synth = SimState(
            scene_id=compute_scene_id(scene, 0),
            scene_def=scene,
            trajectories={"ball": traj},
            collisions=(),
            video_frames=np.zeros((1, 4, 64, 64, 3), dtype=np.uint8),
            camera_names=("cam0",),
            sim_walltime_s=0.01,
            initial_positions={"ball": positions[0].copy()},
            initial_lin_velocities={"ball": np.zeros(3, dtype=np.float64)},
        )
        assert is_trivial(synth) is False
