"""End-to-end physics pipeline: ``SceneDef`` → ``SimState``.

This is the one place in the engine layer that owns a live
:class:`mujoco.MjModel` / :class:`mujoco.MjData` pair. Everything else
(DSL, compiler) is pure Python and does not touch MuJoCo.

Pipeline:

1. Compile ``SceneDef`` → MJCF → ``MjModel``.
2. Allocate fresh ``MjData`` (never shared across threads).
3. Apply initial conditions: write each free-joint body's
   ``initial_quat``, ``pos``, ``initial_lin_vel``, ``initial_ang_vel``
   into the ``qpos``/``qvel`` slots via ``body_jntadr``/``jnt_qposadr``.
4. Initial overlap check: run ``mj_forward`` once and scan
   ``data.contact[:data.ncon]``; penetration > ``eps`` raises
   :class:`InitialOverlapError`.
5. Main loop for ``T = round(duration / timestep)`` steps:
   - wall-time budget check → :class:`SimulationTimeoutError`.
   - ``mj_step``.
   - per-step NaN guard every 100 steps → :class:`SimulationDivergedError`.
   - per-body trajectory logging.
   - collision aggregation over 10-step windows (max contact force kept).
   - frame snapshots at ``linspace(0, T-1, n_frames)``.
6. Build per-body ``BodyTrajectory`` with the final-status classifier.
7. Render via :func:`render_multiview` and return a :class:`SimState`.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import mujoco
import numpy as np

from physevolve.engine.compiler import compile_to_mjcf
from physevolve.engine.failure import (
    InitialOverlapError,
    SimulationDivergedError,
    SimulationTimeoutError,
)
from physevolve.engine.renderer import render_multiview
from physevolve.engine.schema import (
    BodyDef,
    BodyStatus,
    BodyTrajectory,
    CollisionEvent,
    SceneDef,
    SimState,
    compute_scene_id,
)

# ─── Classifier thresholds ────────────────────────────

_WORKSPACE_Z_UPPER: float = 3.0
_WORKSPACE_Z_FLOOR_EJECT: float = -0.5
_FALLEN_Z_MAX: float = 0.05
_NEAR_GROUND_INIT_Z: float = 0.1
_LIN_VEL_STABLE_MAX: float = 0.05
_ANG_VEL_STABLE_MAX: float = 0.1

# Overlap detection: allow this much numerical slop before rejecting.
_OVERLAP_EPS: float = 0.01
# Triviality: max body displacement under which a scene is considered
# "nothing happened".
_TRIVIAL_DISP: float = 0.05
# NaN check cadence: every N steps to keep overhead bounded.
_NAN_CHECK_EVERY: int = 100
# Collision aggregation window.
_COLLISION_WINDOW: int = 10


def simulate(
    scene_def: SceneDef,
    seed: int = 0,
    log_collisions: bool = True,
    log_full_trajectory: bool = True,
    sim_timeout_s: float = 30.0,
    render_height: int = 480,
    render_width: int = 480,
) -> SimState:
    """Run the full simulate+render pipeline on a frozen scene.

    Args:
        scene_def: Output of :meth:`PhysDSLApi.freeze`.
        seed: Simulation seed (mixed into ``scene_id``; MuJoCo stepping
            itself is deterministic per model/state and does not read a
            random stream for the scenes we generate).
        log_collisions: If True, aggregate contact pairs into
            :class:`CollisionEvent` tuples; otherwise returns an empty
            collisions tuple.
        log_full_trajectory: If True, record per-step
            :class:`BodyTrajectory` arrays. If False, only the final
            step is recorded (saves memory for rollouts that only need
            ``final_status``).
        sim_timeout_s: Wall-clock ceiling for the main loop.
        render_height: Pixel height for multi-view output.
        render_width: Pixel width for multi-view output.

    Returns:
        A fully populated :class:`SimState`.

    Raises:
        InitialOverlapError: Bodies interpenetrate at t=0.
        SimulationDivergedError: ``qpos``/``qvel`` went NaN mid-rollout.
        SimulationTimeoutError: Wall-clock budget exceeded.
    """

    wall_start = time.perf_counter()
    env = scene_def.env
    n_steps = int(round(env.duration / env.timestep))
    if n_steps < 1:
        raise ValueError(
            f"duration/timestep = {env.duration}/{env.timestep} yields no steps"
        )

    xml = compile_to_mjcf(scene_def)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    dyn_bodies: List[BodyDef] = [b for b in scene_def.bodies if not b.fixed]
    # Resolve body name → MuJoCo body id once up front.
    body_ids: Dict[str, int] = {}
    for b in scene_def.bodies:
        if b.fixed:
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b.name)
        if bid == -1:
            raise RuntimeError(
                f"body {b.name!r} not found in compiled model"
            )
        body_ids[b.name] = bid

    _apply_initial_conditions(model, data, dyn_bodies, body_ids)
    _check_initial_overlap(model, data)

    # Capture the TRUE t=0 state before the first mj_step. Trajectory
    # row 0 will later hold the post-step state, so we keep these as a
    # separate pre-step snapshot for Stage-2 SimStateView.
    # cvel is spatial (angular, linear) in world frame.
    initial_positions: Dict[str, np.ndarray] = {}
    initial_lin_velocities: Dict[str, np.ndarray] = {}
    for b in dyn_bodies:
        bid = body_ids[b.name]
        initial_positions[b.name] = np.array(data.xpos[bid], dtype=np.float64)
        initial_lin_velocities[b.name] = np.array(
            data.cvel[bid][3:6], dtype=np.float64
        )

    # Pre-allocate trajectory storage. We reserve T rows even when
    # ``log_full_trajectory`` is False so the row index math stays the
    # same; the final-status classifier reads from the last populated row.
    traj_T = n_steps if log_full_trajectory else 1
    positions: Dict[str, np.ndarray] = {}
    orientations: Dict[str, np.ndarray] = {}
    lin_velocities: Dict[str, np.ndarray] = {}
    ang_velocities: Dict[str, np.ndarray] = {}
    for b in dyn_bodies:
        positions[b.name] = np.zeros((traj_T, 3), dtype=np.float64)
        orientations[b.name] = np.zeros((traj_T, 4), dtype=np.float64)
        lin_velocities[b.name] = np.zeros((traj_T, 3), dtype=np.float64)
        ang_velocities[b.name] = np.zeros((traj_T, 3), dtype=np.float64)

    frame_indices = np.linspace(0, n_steps - 1, env.n_frames, dtype=int)
    frame_index_set = set(int(i) for i in frame_indices)
    frame_snapshots: List[Tuple[np.ndarray, np.ndarray, float]] = []

    collisions: List[CollisionEvent] = []
    window_max: Dict[Tuple[str, str], Tuple[int, float]] = {}

    # Geom id → body name lookup for collision attribution.
    geom_body_name: Dict[int, str] = {}
    for g_idx in range(model.ngeom):
        bid = int(model.geom_bodyid[g_idx])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        geom_body_name[g_idx] = name if name is not None else f"body{bid}"

    for step in range(n_steps):
        if time.perf_counter() - wall_start > sim_timeout_s:
            raise SimulationTimeoutError(
                f"simulate() exceeded {sim_timeout_s}s wall budget at step {step}"
            )

        mujoco.mj_step(model, data)

        # Cheap per-step guard on qpos[0] — catches the most common
        # divergence pattern (position blow-up) within one step at
        # negligible cost. The full ``.any()`` scan still runs every
        # _NAN_CHECK_EVERY steps as a belt-and-braces fallback for
        # divergence that lands in qvel slots first.
        if data.qpos.size > 0 and not np.isfinite(data.qpos[0]):
            raise SimulationDivergedError(
                f"non-finite qpos[0] at step {step}"
            )
        if step % _NAN_CHECK_EVERY == 0:
            if bool(np.isnan(data.qpos).any()) or bool(
                np.isnan(data.qvel).any()
            ):
                raise SimulationDivergedError(
                    f"NaN detected in qpos/qvel at step {step}"
                )

        if log_full_trajectory:
            row = step
        else:
            row = 0  # constantly overwritten; final row = last step.
        for b in dyn_bodies:
            bid = body_ids[b.name]
            positions[b.name][row] = data.xpos[bid]
            orientations[b.name][row] = data.xquat[bid]
            # mjData.cvel is spatial velocity (angular, linear) in world
            # frame, expressed at the body's CoM.
            cvel = data.cvel[bid]
            ang_velocities[b.name][row] = cvel[0:3]
            lin_velocities[b.name][row] = cvel[3:6]

        if log_collisions:
            _aggregate_contacts(
                model, data, step, geom_body_name, window_max, collisions
            )

        if int(step) in frame_index_set:
            frame_snapshots.append(
                (data.qpos.copy(), data.qvel.copy(), float(data.time))
            )

    # Flush any residual collision-window state.
    if log_collisions:
        for (a, b), (t_step, force) in window_max.items():
            collisions.append(
                CollisionEvent(
                    t_step=t_step, body_a=a, body_b=b, contact_force_norm=force
                )
            )

    # Ensure we actually captured env.n_frames snapshots (float endpoint
    # quirks of linspace should never miss, but guard anyway).
    while len(frame_snapshots) < env.n_frames:
        frame_snapshots.append(
            (data.qpos.copy(), data.qvel.copy(), float(data.time))
        )
    frame_snapshots = frame_snapshots[: env.n_frames]

    # Final-status classifier.
    trajectories: Dict[str, BodyTrajectory] = {}
    for b in dyn_bodies:
        final_status = _classify_final_status(
            b,
            positions[b.name][-1],
            lin_velocities[b.name][-1],
            ang_velocities[b.name][-1],
        )
        trajectories[b.name] = BodyTrajectory(
            name=b.name,
            positions=positions[b.name],
            orientations=orientations[b.name],
            lin_velocities=lin_velocities[b.name],
            ang_velocities=ang_velocities[b.name],
            final_status=final_status,
        )

    # Render (only if there's at least one camera; SceneDef always has one).
    camera_names = tuple(cam.name for cam in scene_def.cameras)
    video_frames = render_multiview(
        model,
        frame_snapshots,
        camera_names,
        height=render_height,
        width=render_width,
    )

    return SimState(
        scene_id=compute_scene_id(scene_def, seed),
        scene_def=scene_def,
        trajectories=trajectories,
        collisions=tuple(collisions),
        video_frames=video_frames,
        camera_names=camera_names,
        sim_walltime_s=time.perf_counter() - wall_start,
        initial_positions=initial_positions,
        initial_lin_velocities=initial_lin_velocities,
    )


# ─── Triviality helper ────────────────────────────────────────────────


def is_trivial(sim_state: SimState) -> bool:
    """Return True if every body stayed within 5 cm and nothing collided.

    "Stayed within 5 cm" means the **maximum** displacement of every
    logged trajectory row from the starting position is < 0.05 m.
    Comparing only first-vs-last rows would incorrectly classify scenes
    where a body moves significantly and returns (e.g. an elastic bounce
    that settles) as TRIVIAL.

    The executor pipeline uses this to emit
    :attr:`FailureReason.TRIVIAL`.
    """

    if len(sim_state.collisions) > 0:
        return False
    for traj in sim_state.trajectories.values():
        p0 = traj.positions[0]
        # Norm of per-row displacement, then max. Vectorised so cost is
        # O(T) with tiny constants regardless of trajectory length.
        deltas = traj.positions - p0
        max_disp = float(np.linalg.norm(deltas, axis=1).max())
        if max_disp >= _TRIVIAL_DISP:
            return False
    return True


# ═══════════════════════ Private helpers ═══════════════════════════════


def _apply_initial_conditions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    dyn_bodies: List[BodyDef],
    body_ids: Dict[str, int],
) -> None:
    """Write per-body initial ``qpos`` / ``qvel`` for free-joint bodies.

    Non-free joint children (hinge / slide) are left at their natural
    MJCF defaults; rotating a pendulum's bob via ``initial_quat`` would
    fight the hinge constraint and is not supported in v0.1.
    """

    for body in dyn_bodies:
        bid = body_ids[body.name]
        if model.body_jntnum[bid] == 0:
            continue
        jnt_id = int(model.body_jntadr[bid])
        jnt_type = int(model.jnt_type[jnt_id])
        qpos_adr = int(model.jnt_qposadr[jnt_id])
        dof_adr = int(model.jnt_dofadr[jnt_id])

        if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE):
            # qpos layout for free joint: [x, y, z, qw, qx, qy, qz].
            data.qpos[qpos_adr : qpos_adr + 3] = body.pos
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = body.initial_quat
            # qvel layout for free joint: [vx, vy, vz, wx, wy, wz].
            data.qvel[dof_adr : dof_adr + 3] = body.initial_lin_vel
            data.qvel[dof_adr + 3 : dof_adr + 6] = body.initial_ang_vel
        else:
            # For hinge/slide, only velocity seeding makes physical
            # sense — and only the component along the joint axis. We
            # leave the default 0 to keep v0.1 simple; body.pos is
            # already enforced by the MJCF nesting.
            pass

    # Propagate kinematics so xpos/xquat/cvel reflect the new qpos.
    mujoco.mj_forward(model, data)


def _check_initial_overlap(
    model: mujoco.MjModel, data: mujoco.MjData
) -> None:
    """Raise if any contact pair penetrates beyond :data:`_OVERLAP_EPS`.

    Assumes ``mj_forward`` has just been called by the caller so that
    ``data.contact[:data.ncon]`` reflects the initial state.
    """

    mujoco.mj_forward(model, data)
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.dist < -_OVERLAP_EPS:
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            raise InitialOverlapError(
                f"initial overlap of {-contact.dist:.4f} m between geoms "
                f"{g1} ({_geom_debug_name(model, g1)}), "
                f"{g2} ({_geom_debug_name(model, g2)})"
            )


def _geom_debug_name(model: mujoco.MjModel, geom_id: int) -> str:
    """Return ``geom_name/body_name`` for overlap diagnostics."""

    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return (
        f"geom={geom_name or f'geom{geom_id}'}, "
        f"body={body_name or f'body{body_id}'}"
    )


def _aggregate_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: int,
    geom_body_name: Dict[int, str],
    window_max: Dict[Tuple[str, str], Tuple[int, float]],
    collisions: List[CollisionEvent],
) -> None:
    """Fold per-step contacts into a rolling-window max-force aggregator.

    Emits one :class:`CollisionEvent` per body-pair per 10-step window.
    """

    force6 = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        contact = data.contact[i]
        mujoco.mj_contactForce(model, data, i, force6)
        force_norm = float(np.linalg.norm(force6[:3]))
        a_name = geom_body_name.get(int(contact.geom1), "world")
        b_name = geom_body_name.get(int(contact.geom2), "world")
        # Canonicalise pair ordering for stable keys.
        pair = (a_name, b_name) if a_name <= b_name else (b_name, a_name)
        current = window_max.get(pair)
        if current is None or force_norm > current[1]:
            window_max[pair] = (step, force_norm)

    if (step + 1) % _COLLISION_WINDOW == 0 and window_max:
        for (a, b), (t_step, f_norm) in window_max.items():
            collisions.append(
                CollisionEvent(
                    t_step=t_step, body_a=a, body_b=b, contact_force_norm=f_norm
                )
            )
        window_max.clear()


def _classify_final_status(
    body: BodyDef,
    pos: np.ndarray,
    lin_vel: np.ndarray,
    ang_vel: np.ndarray,
) -> BodyStatus:
    """Apply the status classification rules to final-frame state."""

    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    if (
        abs(x) > 1.5
        or abs(y) > 1.5
        or z > _WORKSPACE_Z_UPPER
        or z < _WORKSPACE_Z_FLOOR_EJECT
    ):
        return BodyStatus.EJECTED
    initial_z = float(body.pos[2])
    if z < _FALLEN_Z_MAX and initial_z >= _NEAR_GROUND_INIT_Z:
        return BodyStatus.FALLEN
    if (
        float(np.linalg.norm(lin_vel)) > _LIN_VEL_STABLE_MAX
        or float(np.linalg.norm(ang_vel)) > _ANG_VEL_STABLE_MAX
    ):
        return BodyStatus.MOVING
    return BodyStatus.STABLE


__all__ = ["simulate", "is_trivial"]
