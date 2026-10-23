"""Unit tests for :mod:`physevolve.engine.schema`.

Checks:

  - All dataclass construction paths (mutable + frozen).
  - Frozen round-trip (``SceneDef`` constructed and fields recoverable).
  - ``compute_scene_id`` determinism.
  - ``world_to_grid`` cell-center and boundary clipping.
  - Discrete parameter level arrays.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from physevolve.engine.schema import (
    FRICTION_LEVELS,
    GRID_CELL,
    GRID_RES,
    MASS_LEVELS,
    RESTITUTION_LEVELS,
    WORKSPACE_X_HALF,
    WORKSPACE_Y_HALF,
    WORKSPACE_Z_MAX,
    BodyDef,
    BodyStatus,
    BodyTrajectory,
    CameraDef,
    CollisionEvent,
    EnvConfig,
    GeomType,
    JointDef,
    JointType,
    SceneDef,
    SimState,
    SpringDef,
    compute_scene_id,
    world_to_grid,
)


# ─── Helpers ───────────────────────────────────────────────────────────


def _make_body(name: str = "ball") -> BodyDef:
    return BodyDef(
        name=name,
        geom_type=GeomType.SPHERE,
        pos=(0.0, 0.0, 1.0),
        size=(0.1,),
        mass=1.0,
        rgba=(0.8, 0.2, 0.2, 1.0),
        friction_slide=0.5,
        restitution=0.0,
    )


def _make_scene(seed_name: str = "ball") -> SceneDef:
    body = _make_body(seed_name)
    cam = CameraDef(name="cam0", pos=(1.8, -1.8, 1.8), target=(0.0, 0.0, 0.5))
    return SceneDef(
        bodies=(body,),
        joints=(),
        cameras=(cam,),
        env=EnvConfig(),
        workspace_bounds=(WORKSPACE_X_HALF, WORKSPACE_Y_HALF, WORKSPACE_Z_MAX),
    )


# ─── Enum tests ────────────────────────────────────────────────────────


class TestEnums:
    def test_geom_type_members(self) -> None:
        assert {g.value for g in GeomType} == {
            "box", "sphere", "cylinder", "capsule", "plane",
        }

    def test_joint_type_members(self) -> None:
        assert {j.value for j in JointType} == {
            "free", "hinge", "slide", "ball", "fixed",
        }

    def test_body_status_members(self) -> None:
        assert {s.value for s in BodyStatus} == {
            "stable", "moving", "fallen", "ejected",
        }

    def test_enums_are_str(self) -> None:
        # Str-valued enums so MJCF compiler can pass .value directly.
        assert GeomType.BOX == "box"
        assert JointType.FREE == "free"
        assert BodyStatus.STABLE == "stable"


# ─── Mutable dataclasses ───────────────────────────────────────────────


class TestMutableBuilderDataclasses:
    def test_body_def_defaults_and_mutation(self) -> None:
        body = _make_body()
        assert body.joint_type is JointType.FREE
        assert body.fixed is False
        assert body.initial_lin_vel == (0.0, 0.0, 0.0)
        assert body.initial_quat == (1.0, 0.0, 0.0, 0.0)
        body.mass = 5.0
        assert body.mass == 5.0

    def test_joint_def_minimal(self) -> None:
        j = JointDef(
            name="hinge0",
            joint_type=JointType.HINGE,
            parent="base",
            child="arm",
            pos=(0.0, 0.0, 0.5),
        )
        assert j.axis == (0.0, 0.0, 1.0)
        assert j.limit_range is None
        j.limit_range = (-1.0, 1.0)
        assert j.limit_range == (-1.0, 1.0)

    def test_camera_def_required_fields(self) -> None:
        c = CameraDef(
            name="cam0", pos=(1.0, -1.0, 1.0), target=(0.0, 0.0, 0.0)
        )
        assert c.name == "cam0"
        c.target = (0.1, 0.2, 0.3)
        assert c.target == (0.1, 0.2, 0.3)

    def test_env_config_defaults(self) -> None:
        env = EnvConfig()
        assert env.gravity == (0.0, 0.0, -9.81)
        assert env.timestep == pytest.approx(0.002)
        assert env.duration == pytest.approx(8.0)
        assert env.n_frames == 32


# ─── Frozen dataclasses ────────────────────────────────────────────────


class TestFrozenDataclasses:
    def test_scene_def_round_trip(self) -> None:
        scene = _make_scene()
        assert scene.bodies[0].name == "ball"
        assert scene.env.duration == pytest.approx(8.0)
        assert scene.workspace_bounds == (1.5, 1.5, 3.0)

    def test_scene_def_is_frozen(self) -> None:
        scene = _make_scene()
        with pytest.raises(dataclasses.FrozenInstanceError):
            scene.bodies = ()  # type: ignore[misc]

    def test_body_trajectory_is_frozen(self) -> None:
        traj = BodyTrajectory(
            name="ball",
            positions=np.zeros((4, 3)),
            orientations=np.zeros((4, 4)),
            lin_velocities=np.zeros((4, 3)),
            ang_velocities=np.zeros((4, 3)),
            final_status=BodyStatus.STABLE,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            traj.name = "other"  # type: ignore[misc]
        # But ndarray *contents* are mutable; only the binding is frozen.
        assert traj.positions.shape == (4, 3)

    def test_collision_event_is_frozen(self) -> None:
        ev = CollisionEvent(
            t_step=10, body_a="a", body_b="b", contact_force_norm=1.5
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.t_step = 11  # type: ignore[misc]

    def test_sim_state_round_trip(self) -> None:
        scene = _make_scene()
        sid = compute_scene_id(scene, seed=0)
        sim = SimState(
            scene_id=sid,
            scene_def=scene,
            trajectories={},
            collisions=(),
            video_frames=np.zeros((1, 4, 8, 8, 3), dtype=np.uint8),
            camera_names=("cam0",),
            sim_walltime_s=0.0,
            initial_positions={},
            initial_lin_velocities={},
        )
        assert sim.scene_id == sid
        assert sim.scene_def is scene
        assert sim.video_frames.dtype == np.uint8
        with pytest.raises(dataclasses.FrozenInstanceError):
            sim.sim_walltime_s = 1.0  # type: ignore[misc]


# ─── Scene ID determinism ──────────────────────────────────────────────


class TestSceneId:
    def test_determinism(self) -> None:
        s1 = _make_scene()
        s2 = _make_scene()
        assert compute_scene_id(s1, seed=0) == compute_scene_id(s2, seed=0)

    def test_seed_varies_hash(self) -> None:
        scene = _make_scene()
        assert compute_scene_id(scene, seed=0) != compute_scene_id(scene, seed=1)

    def test_scene_varies_hash(self) -> None:
        a = _make_scene("ball_a")
        b = _make_scene("ball_b")
        assert compute_scene_id(a, seed=0) != compute_scene_id(b, seed=0)

    def test_length_is_sixteen_hex_chars(self) -> None:
        sid = compute_scene_id(_make_scene(), seed=42)
        assert len(sid) == 16
        int(sid, 16)  # must parse as hex; raises if not


# ─── Grid discretization ───────────────────────────────────────────────


class TestWorldToGrid:
    def test_origin_maps_to_xy_center(self) -> None:
        assert world_to_grid((0.0, 0.0, 0.0)) == (8, 8, 0)

    def test_workspace_far_corner_clips_to_max(self) -> None:
        assert world_to_grid(
            (WORKSPACE_X_HALF, WORKSPACE_Y_HALF, WORKSPACE_Z_MAX)
        ) == (GRID_RES - 1, GRID_RES - 1, GRID_RES - 1)

    def test_workspace_near_corner_maps_to_zero(self) -> None:
        assert world_to_grid(
            (-WORKSPACE_X_HALF, -WORKSPACE_Y_HALF, 0.0)
        ) == (0, 0, 0)

    def test_negative_z_clips_to_zero(self) -> None:
        assert world_to_grid((0.0, 0.0, -10.0))[2] == 0

    def test_beyond_workspace_clips_upper(self) -> None:
        gx, gy, gz = world_to_grid((100.0, 100.0, 100.0))
        assert (gx, gy, gz) == (GRID_RES - 1, GRID_RES - 1, GRID_RES - 1)

    def test_cell_width_matches_constants(self) -> None:
        # Stepping by exactly GRID_CELL from the lower corner advances one
        # cell along x.
        base = world_to_grid((-WORKSPACE_X_HALF, 0.0, 0.0))
        shifted = world_to_grid((-WORKSPACE_X_HALF + GRID_CELL, 0.0, 0.0))
        assert shifted[0] == base[0] + 1

    def test_returns_ints(self) -> None:
        gx, gy, gz = world_to_grid((0.3, -0.7, 1.1))
        assert isinstance(gx, int)
        assert isinstance(gy, int)
        assert isinstance(gz, int)


# ─── Discrete parameter levels ─────────────────────────────────────────


class TestLevelArrays:
    def test_mass_levels_shape(self) -> None:
        assert len(MASS_LEVELS) == 5
        assert list(MASS_LEVELS) == sorted(MASS_LEVELS)
        assert all(v > 0 for v in MASS_LEVELS)

    def test_friction_levels_shape(self) -> None:
        assert len(FRICTION_LEVELS) == 5
        assert list(FRICTION_LEVELS) == sorted(FRICTION_LEVELS)
        assert all(v > 0 for v in FRICTION_LEVELS)

    def test_mass_levels_locked_values(self) -> None:
        # Check the default discretization constants.
        assert MASS_LEVELS == (0.1, 0.5, 1.0, 5.0, 10.0)

    def test_friction_levels_locked_values(self) -> None:
        assert FRICTION_LEVELS == (0.05, 0.2, 0.5, 1.0, 2.0)

    def test_restitution_levels_locked_values(self) -> None:
        # Discrete restitution
        # levels that the DSL snaps to (see DSLValidationPolicy in
        # physdsl.py).
        assert RESTITUTION_LEVELS == (0.0, 0.5, 0.9)

    def test_restitution_levels_shape(self) -> None:
        assert len(RESTITUTION_LEVELS) == 3
        assert list(RESTITUTION_LEVELS) == sorted(RESTITUTION_LEVELS)
        assert all(0.0 <= v <= 1.0 for v in RESTITUTION_LEVELS)


# ─── SpringDef ─────────────────────────────────────────────────────────


class TestSpringDef:
    def test_minimal_construction(self) -> None:
        s = SpringDef(
            name="spring_ab",
            body_a="a",
            body_b="b",
            stiffness=100.0,
            damping=2.0,
        )
        assert s.anchor_a == (0.0, 0.0, 0.0)
        assert s.anchor_b == (0.0, 0.0, 0.0)
        assert s.rest_length is None

    def test_is_mutable_like_other_builder_state(self) -> None:
        # SpringDef matches JointDef's dataclass flavor: slots=True,
        # frozen=False (builder state).
        s = SpringDef(
            name="spring_ab",
            body_a="a",
            body_b="b",
            stiffness=50.0,
            damping=1.0,
        )
        s.stiffness = 80.0
        assert s.stiffness == 80.0

    def test_slots_reject_unknown_attr(self) -> None:
        s = SpringDef(
            name="s", body_a="a", body_b="b", stiffness=1.0, damping=0.1
        )
        with pytest.raises(AttributeError):
            s.not_a_field = 123  # type: ignore[attr-defined]

    def test_rest_length_optional(self) -> None:
        s = SpringDef(
            name="s",
            body_a="a",
            body_b="b",
            stiffness=10.0,
            damping=0.5,
            anchor_a=(0.1, 0.0, 0.0),
            anchor_b=(-0.1, 0.0, 0.0),
            rest_length=0.5,
        )
        assert s.rest_length == pytest.approx(0.5)

    def test_scene_def_carries_empty_springs_by_default(self) -> None:
        scene = _make_scene()
        assert scene.springs == ()
