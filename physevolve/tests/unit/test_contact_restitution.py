"""Regression tests for DSL restitution → MuJoCo contact behavior."""

from __future__ import annotations

from dataclasses import replace as dc_replace

import mujoco
import numpy as np

from physevolve.engine.compiler import compile_to_mjcf
from physevolve.engine.contact_params import (
    DEFAULT_CONTACT_SOLIMP,
    DEFAULT_CONTACT_TIMECONST,
)
from physevolve.engine.physdsl import PhysDSLApi
from physevolve.engine.schema import (
    BodyStatus,
    BodyTrajectory,
    EnvConfig,
    SimState,
    compute_scene_id,
)
from physevolve.executor.state_view import SimStateView


def _geom_line(xml: str, geom_name: str) -> str:
    """Return the single MJCF line containing ``geom_name``."""

    matches = [line.strip() for line in xml.splitlines() if geom_name in line]
    assert len(matches) == 1
    return matches[0]


def _scene_with_env(api: PhysDSLApi, env: EnvConfig):
    """Freeze ``api`` and replace the environment config for fast tests."""

    return dc_replace(api.freeze(), env=env)


def _simulate_raw(scene, *, seed: int = 0) -> SimState:
    """Run a CPU-only MuJoCo simulation without invoking the renderer."""

    model = mujoco.MjModel.from_xml_string(compile_to_mjcf(scene))
    data = mujoco.MjData(model)

    dyn_bodies = [body for body in scene.bodies if not body.fixed]
    body_ids = {
        body.name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.name)
        for body in dyn_bodies
    }

    for body in dyn_bodies:
        bid = body_ids[body.name]
        jnt_adr = int(model.body_jntadr[bid])
        qpos_adr = int(model.jnt_qposadr[jnt_adr])
        dof_adr = int(model.jnt_dofadr[jnt_adr])
        data.qpos[qpos_adr : qpos_adr + 3] = body.pos
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = body.initial_quat
        data.qvel[dof_adr : dof_adr + 3] = body.initial_lin_vel
        data.qvel[dof_adr + 3 : dof_adr + 6] = body.initial_ang_vel

    mujoco.mj_forward(model, data)

    initial_positions = {
        name: np.array(data.xpos[bid], dtype=np.float64)
        for name, bid in body_ids.items()
    }
    initial_lin_velocities = {
        name: np.array(data.cvel[bid][3:6], dtype=np.float64)
        for name, bid in body_ids.items()
    }

    steps = int(round(scene.env.duration / scene.env.timestep))
    positions = {
        name: np.zeros((steps, 3), dtype=np.float64) for name in body_ids
    }
    velocities = {
        name: np.zeros((steps, 3), dtype=np.float64) for name in body_ids
    }
    orientations = {
        name: np.zeros((steps, 4), dtype=np.float64) for name in body_ids
    }
    angular = {
        name: np.zeros((steps, 3), dtype=np.float64) for name in body_ids
    }

    for step in range(steps):
        mujoco.mj_step(model, data)
        for name, bid in body_ids.items():
            positions[name][step] = data.xpos[bid]
            velocities[name][step] = data.cvel[bid][3:6]
            orientations[name][step] = data.xquat[bid]
            angular[name][step] = data.cvel[bid][0:3]

    trajectories = {}
    for name in body_ids:
        speed = float(np.linalg.norm(velocities[name][-1]))
        trajectories[name] = BodyTrajectory(
            name=name,
            positions=positions[name],
            orientations=orientations[name],
            lin_velocities=velocities[name],
            ang_velocities=angular[name],
            final_status=BodyStatus.MOVING if speed > 0.05 else BodyStatus.STABLE,
        )

    return SimState(
        scene_id=compute_scene_id(scene, seed),
        scene_def=scene,
        trajectories=trajectories,
        collisions=(),
        video_frames=np.zeros((0, 0, 0, 0, 3), dtype=np.uint8),
        camera_names=(),
        sim_walltime_s=0.0,
        initial_positions=initial_positions,
        initial_lin_velocities=initial_lin_velocities,
    )


def test_zero_restitution_emits_no_contact_override() -> None:
    api = PhysDSLApi(continuous_levels=True)
    api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05, restitution=0.0)

    line = _geom_line(compile_to_mjcf(api.freeze()), "ball_geom")

    assert "solref=" not in line
    assert "solimp=" not in line


def test_positive_restitution_emits_solref_and_fixed_solimp() -> None:
    api = PhysDSLApi(continuous_levels=True)
    api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05, restitution=0.9)

    line = _geom_line(compile_to_mjcf(api.freeze()), "ball_geom")

    assert f'solref="{DEFAULT_CONTACT_TIMECONST} ' in line
    expected_solimp = (
        f'solimp="{DEFAULT_CONTACT_SOLIMP[0]} '
        f'{DEFAULT_CONTACT_SOLIMP[1]} {DEFAULT_CONTACT_SOLIMP[2]}"'
    )
    assert expected_solimp in line
    dampratio = float(
        line.split(f'solref="{DEFAULT_CONTACT_TIMECONST} ', 1)[1].split('"', 1)[0]
    )
    assert 0.0 < dampratio < 1.0


def test_equal_mass_head_on_transfer_is_elastic_like() -> None:
    api = PhysDSLApi(continuous_levels=True)
    api.set_gravity((0.0, 0.0, 0.0))
    api.add_sphere(
        "striker",
        pos=(-0.2, 0.0, 0.2),
        radius=0.05,
        mass=1.0,
        friction=0.0,
        restitution=0.9,
    )
    api.add_sphere(
        "target",
        pos=(0.0, 0.0, 0.2),
        radius=0.05,
        mass=1.0,
        friction=0.0,
        restitution=0.9,
    )
    api.set_velocity("striker", linear=(1.0, 0.0, 0.0))
    scene = _scene_with_env(
        api,
        EnvConfig(gravity=(0.0, 0.0, 0.0), timestep=0.002, duration=0.4, n_frames=1),
    )

    state = _simulate_raw(scene)
    striker_vx = float(state.trajectories["striker"].lin_velocities[-1, 0])
    target_vx = float(state.trajectories["target"].lin_velocities[-1, 0])

    assert abs(striker_vx) < 0.25
    assert target_vx > 0.5
    assert abs((striker_vx + target_vx) - 1.0) < 0.25


def test_restitution_changes_vertical_rebound_height() -> None:
    api = PhysDSLApi(continuous_levels=True)
    api.add_sphere(
        "high_bounce",
        pos=(-0.2, 0.0, 1.0),
        radius=0.05,
        friction=0.0,
        restitution=0.9,
    )
    api.add_sphere(
        "low_bounce",
        pos=(0.2, 0.0, 1.0),
        radius=0.05,
        friction=0.0,
        restitution=0.0,
    )
    scene = _scene_with_env(
        api,
        EnvConfig(gravity=(0.0, 0.0, -9.81), timestep=0.002, duration=1.5, n_frames=1),
    )

    view = SimStateView(_simulate_raw(scene))

    assert view["high_bounce"].rebound_height - view["low_bounce"].rebound_height >= 0.05
