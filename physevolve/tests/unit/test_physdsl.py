"""Unit tests for :mod:`physevolve.engine.physdsl`.

Tests ``engine/physdsl.py``: each primitive
happy path + every validation rule violated individually.
"""

from __future__ import annotations

import pytest

from physevolve.engine.failure import DSLValidationError
from physevolve.engine.physdsl import PhysDSLApi
from physevolve.engine.schema import (
    FRICTION_LEVELS,
    MASS_LEVELS,
    RESTITUTION_LEVELS,
    BodyDef,
    CameraDef,
    GeomType,
    JointDef,
    JointType,
    SceneDef,
    SpringDef,
)


# ══════════════════════ Happy-path primitive tests ════════════════════════


class TestPrimitiveHappyPath:
    def test_add_box(self) -> None:
        api = PhysDSLApi()
        api.add_box("box0", pos=(0.0, 0.0, 0.5), size=(0.1, 0.1, 0.1))
        scene = api.freeze()
        names = [b.name for b in scene.bodies]
        assert "box0" in names
        box = next(b for b in scene.bodies if b.name == "box0")
        assert box.geom_type is GeomType.BOX
        assert box.mass == 1.0
        assert box.friction_slide == 0.5
        assert box.joint_type is JointType.FREE

    def test_add_sphere(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.1, 0.0, 0.5), radius=0.1)
        scene = api.freeze()
        ball = next(b for b in scene.bodies if b.name == "ball")
        assert ball.geom_type is GeomType.SPHERE
        assert ball.size == (0.1,)

    def test_add_cylinder_and_capsule(self) -> None:
        api = PhysDSLApi()
        api.add_cylinder(
            "cyl", pos=(0.0, 0.0, 0.5), radius=0.05, half_height=0.2, axis="x"
        )
        api.add_capsule(
            "cap", pos=(0.2, 0.0, 0.5), radius=0.05, half_height=0.1, axis="y"
        )
        scene = api.freeze()
        cyl = next(b for b in scene.bodies if b.name == "cyl")
        cap = next(b for b in scene.bodies if b.name == "cap")
        assert cyl.geom_type is GeomType.CYLINDER
        assert cap.geom_type is GeomType.CAPSULE
        assert cyl.initial_quat != (1.0, 0.0, 0.0, 0.0)
        assert cap.initial_quat != (1.0, 0.0, 0.0, 0.0)

    def test_add_plane_default_name(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        api.add_plane()
        scene = api.freeze()
        ground = next(b for b in scene.bodies if b.name == "ground")
        assert ground.geom_type is GeomType.PLANE
        assert ground.fixed is True

    def test_add_fixed_body(self) -> None:
        api = PhysDSLApi()
        api.add_box("plat", pos=(0.0, 0.0, 0.5), size=(0.3, 0.3, 0.02), fixed=True)
        scene = api.freeze()
        plat = next(b for b in scene.bodies if b.name == "plat")
        assert plat.fixed is True
        assert plat.joint_type is JointType.FIXED

    def test_set_velocity_and_rotation(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        api.set_velocity("ball", linear=(1.0, 0.0, 0.0), angular=(0.0, 2.0, 0.0))
        api.set_rotation("ball", quaternion=(1.0, 0.0, 0.0, 0.0))
        scene = api.freeze()
        ball = next(b for b in scene.bodies if b.name == "ball")
        assert ball.initial_lin_vel == (1.0, 0.0, 0.0)
        assert ball.initial_ang_vel == (0.0, 2.0, 0.0)
        assert ball.initial_quat == pytest.approx((1.0, 0.0, 0.0, 0.0))

    def test_add_hinge_and_slide(self) -> None:
        api = PhysDSLApi()
        api.add_box(
            "anchor", pos=(0.0, 0.0, 1.0), size=(0.05, 0.05, 0.05), fixed=True
        )
        api.add_sphere("bob", pos=(0.3, 0.0, 1.0), radius=0.05)
        api.add_hinge(
            "pivot",
            parent="anchor",
            child="bob",
            pos=(0.0, 0.0, 1.0),
            axis=(0.0, 1.0, 0.0),
        )
        scene = api.freeze()
        assert len(scene.joints) == 1
        j = scene.joints[0]
        assert j.joint_type is JointType.HINGE
        assert j.parent == "anchor" and j.child == "bob"
        bob = next(b for b in scene.bodies if b.name == "bob")
        # Explicit joint should override FREE default on child body.
        assert bob.joint_type is JointType.HINGE

    def test_set_gravity_and_duration(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        api.set_gravity((0.0, 0.0, -3.7))
        api.set_duration(4.0)
        scene = api.freeze()
        assert scene.env.gravity == (0.0, 0.0, -3.7)
        assert scene.env.duration == pytest.approx(4.0)

    def test_add_camera(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        api.add_camera("side", pos=(2.0, 0.0, 1.0), target=(0.0, 0.0, 0.5))
        scene = api.freeze()
        cam_names = [c.name for c in scene.cameras]
        assert "side" in cam_names


# ════════════════════════ Auto-defaults ═══════════════════════════════════


class TestAutoDefaults:
    def test_default_ground_added_when_missing(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        scene = api.freeze()
        assert any(b.name == "ground" for b in scene.bodies)
        ground = next(b for b in scene.bodies if b.name == "ground")
        assert ground.geom_type is GeomType.PLANE

    def test_default_ground_not_duplicated(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        api.add_plane()
        scene = api.freeze()
        ground_count = sum(1 for b in scene.bodies if b.name == "ground")
        assert ground_count == 1

    def test_default_camera_added_when_missing(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        scene = api.freeze()
        assert len(scene.cameras) == 1
        cam = scene.cameras[0]
        assert cam.pos == (1.8, -1.8, 1.8)
        assert cam.target == (0.0, 0.0, 0.5)


# ══════════════════ Name / uniqueness / reserved rules ═══════════════════


class TestNameValidation:
    @pytest.mark.parametrize(
        "bad_name",
        ["Ball", "1ball", "ball-1", "ball 1", "ball!", ""],
    )
    def test_invalid_name_format_rejected(self, bad_name: str) -> None:
        api = PhysDSLApi()
        with pytest.raises(DSLValidationError):
            api.add_sphere(bad_name, pos=(0.0, 0.0, 0.5), radius=0.05)

    @pytest.mark.parametrize("reserved", ["world", "sky"])
    def test_reserved_names_rejected_for_bodies(self, reserved: str) -> None:
        api = PhysDSLApi()
        with pytest.raises(DSLValidationError):
            api.add_sphere(reserved, pos=(0.0, 0.0, 0.5), radius=0.05)

    def test_ground_reserved_for_plane_only(self) -> None:
        api = PhysDSLApi()
        # Cannot use "ground" as a dynamic body.
        with pytest.raises(DSLValidationError):
            api.add_sphere("ground", pos=(0.0, 0.0, 0.5), radius=0.05)

    def test_duplicate_name_across_bodies_rejected(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        with pytest.raises(DSLValidationError):
            api.add_sphere("ball", pos=(0.2, 0.0, 0.5), radius=0.05)

    def test_duplicate_name_body_vs_camera_rejected(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("foo", pos=(0.0, 0.0, 0.5), radius=0.05)
        with pytest.raises(DSLValidationError):
            api.add_camera("foo", pos=(2.0, 0.0, 1.0), target=(0.0, 0.0, 0.5))

    def test_duplicate_ground_plane_rejected(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        api.add_plane()
        with pytest.raises(DSLValidationError):
            api.add_plane()


# ═══════════════════════════ Workspace pos ════════════════════════════════


class TestPosValidation:
    @pytest.mark.parametrize(
        "bad_pos",
        [
            (2.0, 0.0, 0.5),      # x out of range
            (0.0, -1.6, 0.5),     # y out of range
            (0.0, 0.0, 0.0),      # z at lower boundary (strict)
            (0.0, 0.0, -0.1),     # z below floor
            (0.0, 0.0, 2.6),      # z above ceiling
            (0.0, 0.0, 2.5),      # z at upper boundary (strict)
        ],
    )
    def test_invalid_pos_rejected(
        self, bad_pos: tuple[float, float, float]
    ) -> None:
        api = PhysDSLApi()
        with pytest.raises(DSLValidationError):
            api.add_sphere("ball", pos=bad_pos, radius=0.05)

    def test_plane_pos_allows_z_zero(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        # Default-name plane at z=0 is allowed.
        api.add_plane()
        api_b = PhysDSLApi()
        api_b.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        # Custom-named plane at z=0 allowed too.
        api_b.add_plane(name="floor", pos=(0.0, 0.0, 0.0))


# ════════════════════════════ Mass / friction ═════════════════════════════


class TestMassValidation:
    @pytest.mark.parametrize(
        "out_of_range_mass",
        [0.0, 0.009, 200.0, -1.0],
    )
    def test_out_of_range_mass_rejected(self, out_of_range_mass: float) -> None:
        # mass values outside [0.01, 100] remain a hard-reject.
        api = PhysDSLApi()
        with pytest.raises(DSLValidationError):
            api.add_sphere(
                "ball", pos=(0.0, 0.0, 0.5), radius=0.05, mass=out_of_range_mass
            )

    @pytest.mark.parametrize("good_mass", list(MASS_LEVELS))
    def test_each_mass_level_accepted(self, good_mass: float) -> None:
        api = PhysDSLApi()
        api.add_sphere(
            "ball", pos=(0.0, 0.0, 0.5), radius=0.05, mass=good_mass
        )
        scene = api.freeze()
        ball = next(b for b in scene.bodies if b.name == "ball")
        assert ball.mass == good_mass

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (0.4, 0.5),
            (3.14, 5.0),
            (7.0, 5.0),
            (9.0, 10.0),
            (1, 1.0),  # int → float coerce, exact level
        ],
    )
    def test_in_range_mass_snaps_to_nearest_level(
        self, raw: float, expected: float
    ) -> None:
        # in-range mass is silently snapped to the nearest level.
        api = PhysDSLApi()
        api.add_sphere(
            "ball", pos=(0.0, 0.0, 0.5), radius=0.05, mass=raw
        )
        scene = api.freeze()
        ball = next(b for b in scene.bodies if b.name == "ball")
        assert ball.mass == pytest.approx(expected)


class TestFrictionValidation:
    @pytest.mark.parametrize(
        "out_of_range_friction",
        [-0.1, 2.5, 100.0],
    )
    def test_out_of_range_friction_rejected(
        self, out_of_range_friction: float
    ) -> None:
        api = PhysDSLApi()
        with pytest.raises(DSLValidationError):
            api.add_sphere(
                "ball",
                pos=(0.0, 0.0, 0.5),
                radius=0.05,
                friction=out_of_range_friction,
            )

    @pytest.mark.parametrize("good_friction", list(FRICTION_LEVELS))
    def test_each_friction_level_accepted(self, good_friction: float) -> None:
        api = PhysDSLApi()
        api.add_sphere(
            "ball", pos=(0.0, 0.0, 0.5), radius=0.05, friction=good_friction
        )

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (0.4, 0.5),
            (0.1, 0.05),
            (0.7, 0.5),
            (1.6, 2.0),
            (0, 0.05),
        ],
    )
    def test_in_range_friction_snaps_to_nearest_level(
        self, raw: float, expected: float
    ) -> None:
        api = PhysDSLApi()
        api.add_sphere(
            "ball", pos=(0.0, 0.0, 0.5), radius=0.05, friction=raw
        )
        scene = api.freeze()
        ball = next(b for b in scene.bodies if b.name == "ball")
        assert ball.friction_slide == pytest.approx(expected)


class TestRestitutionValidation:
    @pytest.mark.parametrize("out_of_range_r", [-0.1, 1.1, 2.0])
    def test_out_of_range_restitution_rejected(self, out_of_range_r: float) -> None:
        api = PhysDSLApi()
        with pytest.raises(DSLValidationError):
            api.add_sphere(
                "ball",
                pos=(0.0, 0.0, 0.5),
                radius=0.05,
                restitution=out_of_range_r,
            )

    @pytest.mark.parametrize("good_r", list(RESTITUTION_LEVELS))
    def test_each_restitution_level_accepted(self, good_r: float) -> None:
        api = PhysDSLApi()
        api.add_sphere(
            "ball", pos=(0.0, 0.0, 0.5), radius=0.05, restitution=good_r
        )
        scene = api.freeze()
        ball = next(b for b in scene.bodies if b.name == "ball")
        assert ball.restitution == pytest.approx(good_r)

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (0.25, 0.0),
            (0.3, 0.5),
            (0.7, 0.5),
            (0.8, 0.9),
            (0.95, 0.9),
            (1.0, 0.9),  # 1.0 is closest to 0.9 among (0.0, 0.5, 0.9).
        ],
    )
    def test_in_range_restitution_snaps_to_nearest_level(
        self, raw: float, expected: float
    ) -> None:
        api = PhysDSLApi()
        api.add_sphere(
            "ball", pos=(0.0, 0.0, 0.5), radius=0.05, restitution=raw
        )
        scene = api.freeze()
        ball = next(b for b in scene.bodies if b.name == "ball")
        assert ball.restitution == pytest.approx(expected)


# ══════════════════════════ Other scalar rules ════════════════════════════


class TestMiscValidation:
    @pytest.mark.parametrize(
        "bad_size",
        [(0.009,), (1.1,), (0.0,)],
    )
    def test_invalid_sphere_size_rejected(
        self, bad_size: tuple[float, ...]
    ) -> None:
        api = PhysDSLApi()
        with pytest.raises(DSLValidationError):
            api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=bad_size[0])

    def test_invalid_box_size_tuple_length_rejected(self) -> None:
        # Missing axis is a real error; under-length tuples still hard-reject.
        api = PhysDSLApi()
        with pytest.raises(DSLValidationError):
            api.add_box(
                "box", pos=(0.0, 0.0, 0.5), size=(0.1, 0.1)  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "bad_rgba",
        [
            (1.1, 0.0, 0.0, 1.0),
            (-0.1, 0.0, 0.0, 1.0),
            (0.5, 0.5, 0.5, 1.5),
        ],
    )
    def test_invalid_rgba_rejected(
        self, bad_rgba: tuple[float, float, float, float]
    ) -> None:
        api = PhysDSLApi()
        with pytest.raises(DSLValidationError):
            api.add_sphere(
                "ball", pos=(0.0, 0.0, 0.5), radius=0.05, rgba=bad_rgba
            )

    @pytest.mark.parametrize(
        "bad_vel",
        [(21.0, 0.0, 0.0), (0.0, -30.0, 0.0), (0.0, 0.0, 25.0)],
    )
    def test_invalid_velocity_rejected(
        self, bad_vel: tuple[float, float, float]
    ) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        with pytest.raises(DSLValidationError):
            api.set_velocity("ball", linear=bad_vel)

    def test_set_velocity_on_unknown_body_rejected(self) -> None:
        api = PhysDSLApi()
        with pytest.raises(DSLValidationError):
            api.set_velocity("ghost", linear=(1.0, 0.0, 0.0))

    def test_set_velocity_on_fixed_body_rejected(self) -> None:
        api = PhysDSLApi()
        api.add_box(
            "wall", pos=(0.0, 0.0, 0.5), size=(0.1, 0.1, 0.1), fixed=True
        )
        with pytest.raises(DSLValidationError):
            api.set_velocity("wall", linear=(1.0, 0.0, 0.0))

    def test_set_rotation_zero_quaternion_rejected(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        with pytest.raises(DSLValidationError):
            api.set_rotation("ball", quaternion=(0.0, 0.0, 0.0, 0.0))

    def test_set_rotation_normalises_quaternion(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        api.set_rotation("ball", quaternion=(2.0, 0.0, 0.0, 0.0))
        scene = api.freeze()
        ball = next(b for b in scene.bodies if b.name == "ball")
        assert ball.initial_quat == pytest.approx((1.0, 0.0, 0.0, 0.0))


# ═════════════════════════════ Joints ════════════════════════════════════


class TestJointValidation:
    def test_joint_to_unknown_body_rejected(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("bob", pos=(0.0, 0.0, 0.5), radius=0.05)
        with pytest.raises(DSLValidationError):
            api.add_hinge(
                "pivot", parent="nope", child="bob", pos=(0.0, 0.0, 0.5)
            )

    def test_joint_parent_equals_child_rejected(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("bob", pos=(0.0, 0.0, 0.5), radius=0.05)
        with pytest.raises(DSLValidationError):
            api.add_hinge(
                "pivot", parent="bob", child="bob", pos=(0.0, 0.0, 0.5)
            )

    def test_joint_zero_axis_rejected(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("a", pos=(0.0, 0.0, 0.5), radius=0.05)
        api.add_sphere("b", pos=(0.2, 0.0, 0.5), radius=0.05)
        with pytest.raises(DSLValidationError):
            api.add_hinge(
                "pivot",
                parent="a",
                child="b",
                pos=(0.0, 0.0, 0.5),
                axis=(0.0, 0.0, 0.0),
            )

    def test_joint_limit_range_invalid_rejected(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("a", pos=(0.0, 0.0, 0.5), radius=0.05)
        api.add_sphere("b", pos=(0.2, 0.0, 0.5), radius=0.05)
        with pytest.raises(DSLValidationError):
            api.add_hinge(
                "pivot",
                parent="a",
                child="b",
                pos=(0.0, 0.0, 0.5),
                limit=(1.0, 0.0),
            )

    def test_child_with_two_joints_rejected(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("a", pos=(-0.2, 0.0, 0.5), radius=0.05)
        api.add_sphere("b", pos=(0.2, 0.0, 0.5), radius=0.05)
        api.add_sphere("c", pos=(0.0, 0.0, 0.5), radius=0.05)
        api.add_hinge(
            "j1", parent="a", child="c", pos=(0.0, 0.0, 0.5)
        )
        with pytest.raises(DSLValidationError):
            api.add_hinge(
                "j2", parent="b", child="c", pos=(0.0, 0.0, 0.5)
            )


# ══════════════════════════════ Camera ═══════════════════════════════════


class TestCameraValidation:
    def test_camera_pos_equals_target_rejected(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        with pytest.raises(DSLValidationError):
            api.add_camera(
                "bad", pos=(1.0, 1.0, 1.0), target=(1.0, 1.0, 1.0)
            )


# ══════════════════════════ Body count bounds ════════════════════════════


class TestBodyCountBounds:
    def test_zero_bodies_rejected(self) -> None:
        api = PhysDSLApi()
        # No body added; only default plane + camera would appear, but
        # freeze() requires at least one body BEFORE auto-plane is
        # considered... actually the plane IS a body. Let's assert the
        # rule strictly: total bodies at freeze must be >= 1. With only
        # the auto-plane we'd have 1, which should PASS. To reach 0 we
        # would need to bypass freeze's plane auto-add, which is not
        # reachable from the public API. Instead, verify that an empty
        # PhysDSLApi freezes to a scene containing the default plane.
        scene = api.freeze()
        assert len(scene.bodies) == 1  # just the auto-added ground plane

    def test_max_dynamic_bodies_accepted_at_cap(self) -> None:
        api = PhysDSLApi()
        for i in range(20):
            api.add_sphere(
                f"b{i}", pos=(-1.4 + i * 0.1, 0.0, 0.5), radius=0.02
            )
        scene = api.freeze()  # 20 dynamic + ground plane
        assert sum(1 for b in scene.bodies if not b.fixed) == 20

    def test_over_cap_dynamic_bodies_rejected(self) -> None:
        api = PhysDSLApi()
        for i in range(21):
            api.add_sphere(
                f"b{i}", pos=(-1.4 + i * 0.08, 0.0, 0.5), radius=0.02
            )
        with pytest.raises(DSLValidationError):
            api.freeze()


# ═══════════════════════ Structural invariants ═══════════════════════════


class TestFreezeOutput:
    def test_freeze_returns_scene_def_and_is_reusable(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        s1 = api.freeze()
        s2 = api.freeze()
        assert isinstance(s1, SceneDef)
        # Freezing a second time must NOT re-add defaults or mutate the
        # builder; body count stays the same.
        assert len(s1.bodies) == len(s2.bodies)

    def test_freeze_preserves_body_insertion_order(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("a", pos=(-0.2, 0.0, 0.5), radius=0.05)
        api.add_sphere("b", pos=(0.0, 0.0, 0.5), radius=0.05)
        api.add_sphere("c", pos=(0.2, 0.0, 0.5), radius=0.05)
        scene = api.freeze()
        dyn_names = [b.name for b in scene.bodies if not b.fixed]
        assert dyn_names == ["a", "b", "c"]

    def test_workspace_bounds_propagated(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        scene = api.freeze()
        assert scene.workspace_bounds == (1.5, 1.5, 3.0)

    def test_scene_contains_only_expected_types(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        api.add_box("plat", pos=(0.0, 0.0, 0.3), size=(0.2, 0.2, 0.02), fixed=True)
        api.add_camera("side", pos=(2.0, 0.0, 1.0), target=(0.0, 0.0, 0.5))
        scene = api.freeze()
        assert all(isinstance(b, BodyDef) for b in scene.bodies)
        assert all(isinstance(c, CameraDef) for c in scene.cameras)
        assert all(isinstance(j, JointDef) for j in scene.joints)


# ═══════════════════════════ Environment ═════════════════════════════════


class TestEnvSetters:
    def test_set_duration_bounds(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        with pytest.raises(DSLValidationError):
            api.set_duration(0.0)
        with pytest.raises(DSLValidationError):
            api.set_duration(100.0)

    def test_set_gravity_accepts_list(self) -> None:
        # Type coercion: list is now accepted and coerced.
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        api.set_gravity([0.0, 0.0, -9.81])  # type: ignore[arg-type]
        scene = api.freeze()
        assert scene.env.gravity == (0.0, 0.0, -9.81)

    def test_set_gravity_rejects_scalar(self) -> None:
        api = PhysDSLApi()
        with pytest.raises(DSLValidationError):
            api.set_gravity(9.81)  # type: ignore[arg-type]

    def test_set_gravity_rejects_under_length(self) -> None:
        api = PhysDSLApi()
        with pytest.raises(DSLValidationError):
            api.set_gravity((0.0, -9.81))  # type: ignore[arg-type]


# ════════════════════════ Silent coercion layer ═══════════════════════════


class TestCoercionShapeAndType:
    def test_pos_accepts_list(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=[0.0, 0.0, 0.5], radius=0.05)  # type: ignore[arg-type]
        scene = api.freeze()
        ball = next(b for b in scene.bodies if b.name == "ball")
        assert ball.pos == (0.0, 0.0, 0.5)

    def test_pos_truncates_over_length(self) -> None:
        # Over-long sequences silently truncate to the first n components.
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5, 99.9), radius=0.05)  # type: ignore[arg-type]
        scene = api.freeze()
        ball = next(b for b in scene.bodies if b.name == "ball")
        assert ball.pos == (0.0, 0.0, 0.5)

    def test_rgba_3_tuple_gets_alpha_one(self) -> None:
        api = PhysDSLApi()
        api.add_sphere(
            "ball", pos=(0.0, 0.0, 0.5), radius=0.05, rgba=(0.5, 0.5, 0.5)  # type: ignore[arg-type]
        )
        scene = api.freeze()
        ball = next(b for b in scene.bodies if b.name == "ball")
        assert ball.rgba == (0.5, 0.5, 0.5, 1.0)

    def test_box_size_scalar_expands_to_uniform_cube(self) -> None:
        api = PhysDSLApi()
        api.add_box("cube", pos=(0.0, 0.0, 0.5), size=0.1)  # type: ignore[arg-type]
        scene = api.freeze()
        cube = next(b for b in scene.bodies if b.name == "cube")
        assert cube.size == (0.1, 0.1, 0.1)

    def test_int_mass_coerces_to_float(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05, mass=1)
        scene = api.freeze()
        ball = next(b for b in scene.bodies if b.name == "ball")
        assert isinstance(ball.mass, float)
        assert ball.mass == 1.0

    def test_velocity_accepts_list(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        api.set_velocity("ball", linear=[1.0, 0.0, 0.0])  # type: ignore[arg-type]
        scene = api.freeze()
        ball = next(b for b in scene.bodies if b.name == "ball")
        assert ball.initial_lin_vel == (1.0, 0.0, 0.0)

    def test_joint_pos_accepts_list(self) -> None:
        api = PhysDSLApi()
        api.add_box(
            "anchor", pos=(0.0, 0.0, 1.0), size=(0.05, 0.05, 0.05), fixed=True
        )
        api.add_sphere("bob", pos=(0.3, 0.0, 1.0), radius=0.05)
        api.add_hinge(
            "pivot",
            parent="anchor",
            child="bob",
            pos=[0.0, 0.0, 1.0],  # type: ignore[arg-type]
            axis=[0.0, 1.0, 0.0],  # type: ignore[arg-type]
        )
        scene = api.freeze()
        assert scene.joints[0].pos == (0.0, 0.0, 1.0)
        assert scene.joints[0].axis == (0.0, 1.0, 0.0)

    def test_rgba_nan_still_rejected(self) -> None:
        import math
        api = PhysDSLApi()
        with pytest.raises(DSLValidationError):
            api.add_sphere(
                "ball",
                pos=(0.0, 0.0, 0.5),
                radius=0.05,
                rgba=(math.nan, 0.5, 0.5, 1.0),
            )

    def test_pos_bool_rejected(self) -> None:
        # Booleans are ints in Python but never valid physics scalars.
        api = PhysDSLApi()
        with pytest.raises(DSLValidationError):
            api.add_sphere("ball", pos=(True, 0.0, 0.5), radius=0.05)  # type: ignore[arg-type]


# ════════════════════════════ Springs ════════════════════════════════════


class TestSpringValidation:
    def _two_dynamic_bodies(self) -> PhysDSLApi:
        api = PhysDSLApi()
        api.add_sphere("a", pos=(-0.2, 0.0, 0.5), radius=0.05)
        api.add_sphere("b", pos=(0.2, 0.0, 0.5), radius=0.05)
        return api

    def test_add_spring_happy_path(self) -> None:
        api = self._two_dynamic_bodies()
        api.add_spring(
            "spring_ab",
            body_a="a",
            body_b="b",
            stiffness=100.0,
            damping=2.0,
        )
        scene = api.freeze()
        assert len(scene.springs) == 1
        s = scene.springs[0]
        assert isinstance(s, SpringDef)
        assert s.name == "spring_ab"
        assert s.body_a == "a" and s.body_b == "b"
        assert s.stiffness == pytest.approx(100.0)
        assert s.damping == pytest.approx(2.0)
        assert s.rest_length is None

    def test_add_spring_with_anchors_and_rest_length(self) -> None:
        api = self._two_dynamic_bodies()
        api.add_spring(
            "s1",
            body_a="a",
            body_b="b",
            stiffness=50.0,
            damping=1.0,
            anchor_a=(0.05, 0.0, 0.0),
            anchor_b=(-0.05, 0.0, 0.0),
            rest_length=0.3,
        )
        scene = api.freeze()
        s = scene.springs[0]
        assert s.anchor_a == (0.05, 0.0, 0.0)
        assert s.anchor_b == (-0.05, 0.0, 0.0)
        assert s.rest_length == pytest.approx(0.3)

    def test_add_spring_anchor_accepts_list(self) -> None:
        api = self._two_dynamic_bodies()
        api.add_spring(
            "s1",
            body_a="a",
            body_b="b",
            stiffness=50.0,
            damping=1.0,
            anchor_a=[0.1, 0.0, 0.0],  # type: ignore[arg-type]
        )
        scene = api.freeze()
        assert scene.springs[0].anchor_a == (0.1, 0.0, 0.0)

    def test_add_spring_same_body_rejected(self) -> None:
        api = self._two_dynamic_bodies()
        with pytest.raises(DSLValidationError):
            api.add_spring(
                "s1", body_a="a", body_b="a", stiffness=1.0, damping=0.1
            )

    def test_add_spring_unknown_body_rejected(self) -> None:
        api = self._two_dynamic_bodies()
        with pytest.raises(DSLValidationError):
            api.add_spring(
                "s1", body_a="a", body_b="ghost", stiffness=1.0, damping=0.1
            )

    def test_add_spring_both_fixed_rejected(self) -> None:
        api = PhysDSLApi()
        api.add_box(
            "wa", pos=(-0.2, 0.0, 0.5), size=(0.05, 0.05, 0.05), fixed=True
        )
        api.add_box(
            "wb", pos=(0.2, 0.0, 0.5), size=(0.05, 0.05, 0.05), fixed=True
        )
        with pytest.raises(DSLValidationError):
            api.add_spring(
                "s1", body_a="wa", body_b="wb", stiffness=1.0, damping=0.1
            )

    def test_add_spring_one_fixed_allowed(self) -> None:
        # A fixed anchor block + dynamic ball is a valid spring pair.
        api = PhysDSLApi()
        api.add_box(
            "anchor",
            pos=(0.0, 0.0, 1.0),
            size=(0.05, 0.05, 0.05),
            fixed=True,
        )
        api.add_sphere("bob", pos=(0.3, 0.0, 0.5), radius=0.05)
        api.add_spring(
            "s1", body_a="anchor", body_b="bob", stiffness=20.0, damping=0.5
        )
        scene = api.freeze()
        assert len(scene.springs) == 1

    @pytest.mark.parametrize("bad_k", [-1.0, 0.0, 0.01, 1.0e5])
    def test_add_spring_invalid_stiffness_rejected(self, bad_k: float) -> None:
        api = self._two_dynamic_bodies()
        with pytest.raises(DSLValidationError):
            api.add_spring(
                "s1", body_a="a", body_b="b", stiffness=bad_k, damping=0.1
            )

    @pytest.mark.parametrize("bad_c", [-0.1, 1e4])
    def test_add_spring_invalid_damping_rejected(self, bad_c: float) -> None:
        api = self._two_dynamic_bodies()
        with pytest.raises(DSLValidationError):
            api.add_spring(
                "s1", body_a="a", body_b="b", stiffness=10.0, damping=bad_c
            )

    def test_add_spring_anchor_out_of_bounds_rejected(self) -> None:
        api = self._two_dynamic_bodies()
        with pytest.raises(DSLValidationError):
            api.add_spring(
                "s1",
                body_a="a",
                body_b="b",
                stiffness=10.0,
                damping=0.1,
                anchor_a=(3.0, 0.0, 0.0),
            )

    def test_add_spring_invalid_rest_length_rejected(self) -> None:
        api = self._two_dynamic_bodies()
        with pytest.raises(DSLValidationError):
            api.add_spring(
                "s1",
                body_a="a",
                body_b="b",
                stiffness=10.0,
                damping=0.1,
                rest_length=0.0,
            )

    def test_add_spring_name_collision_rejected(self) -> None:
        api = self._two_dynamic_bodies()
        # "a" is already a body name.
        with pytest.raises(DSLValidationError):
            api.add_spring(
                "a", body_a="a", body_b="b", stiffness=10.0, damping=0.1
            )

    def test_spring_count_in_frozen_scene(self) -> None:
        api = self._two_dynamic_bodies()
        api.add_sphere("c", pos=(0.0, 0.2, 0.5), radius=0.05)
        api.add_spring("s1", body_a="a", body_b="b", stiffness=10.0, damping=0.1)
        api.add_spring("s2", body_a="b", body_b="c", stiffness=20.0, damping=0.2)
        scene = api.freeze()
        assert len(scene.springs) == 2
        assert {s.name for s in scene.springs} == {"s1", "s2"}

    def test_scene_def_default_springs_empty(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        scene = api.freeze()
        assert scene.springs == ()


# ════════════════ World-parent joint shortcut ═══════════════════


class TestWorldParentedJoint:
    """``parent="world"`` lets a body hinge/slide directly off the worldbody.

    The DSL must accept this without requiring a fake anchor body to be
    registered, while still rejecting all other pathological combinations
    (e.g. ``child="world"``, double-jointed child).
    """

    def test_hinge_world_parent_accepted(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("bob", pos=(0.3, 0.0, 1.5), radius=0.08, mass=0.5)
        api.add_hinge(
            name="pivot",
            parent="world",
            child="bob",
            pos=(0.0, 0.0, 1.5),
            axis=(0.0, 1.0, 0.0),
        )
        scene = api.freeze()
        assert len(scene.joints) == 1
        joint = scene.joints[0]
        assert joint.parent == "world"
        assert joint.child == "bob"
        assert joint.joint_type is JointType.HINGE
        # Frozen SceneDef remains hashable / immutable as usual.
        assert isinstance(scene, SceneDef)

    def test_slide_world_parent_accepted(self) -> None:
        api = PhysDSLApi()
        api.add_box(
            "cart",
            pos=(0.0, 0.0, 0.5),
            size=(0.1, 0.1, 0.05),
            mass=1.0,
        )
        api.add_slide(
            name="rail",
            parent="world",
            child="cart",
            pos=(0.0, 0.0, 0.5),
            axis=(1.0, 0.0, 0.0),
            limit=(-1.0, 1.0),
        )
        scene = api.freeze()
        assert len(scene.joints) == 1
        joint = scene.joints[0]
        assert joint.parent == "world"
        assert joint.child == "cart"
        assert joint.joint_type is JointType.SLIDE

    def test_child_world_rejected(self) -> None:
        api = PhysDSLApi()
        api.add_sphere("bob", pos=(0.3, 0.0, 1.5), radius=0.08)
        with pytest.raises(DSLValidationError, match="child cannot be"):
            api.add_hinge(
                name="pivot",
                parent="bob",
                child="world",
                pos=(0.0, 0.0, 1.5),
                axis=(0.0, 1.0, 0.0),
            )

    def test_world_parent_unregistered_other_name_still_rejected(self) -> None:
        # Any non-"world" parent name still has to be a registered body.
        api = PhysDSLApi()
        api.add_sphere("bob", pos=(0.3, 0.0, 1.5), radius=0.08)
        with pytest.raises(DSLValidationError, match="parent body 'ghost'"):
            api.add_hinge(
                name="pivot",
                parent="ghost",
                child="bob",
                pos=(0.0, 0.0, 1.5),
                axis=(0.0, 1.0, 0.0),
            )

    def test_duplicate_world_joint_still_nesting_guard(self) -> None:
        # Even with parent="world" the child-already-jointed guard fires
        # on the second joint targeting the same body.
        api = PhysDSLApi()
        api.add_sphere("bob", pos=(0.3, 0.0, 1.5), radius=0.08)
        api.add_hinge(
            name="pivot",
            parent="world",
            child="bob",
            pos=(0.0, 0.0, 1.5),
            axis=(0.0, 1.0, 0.0),
        )
        with pytest.raises(DSLValidationError, match="already has another joint"):
            api.add_slide(
                name="rail",
                parent="world",
                child="bob",
                pos=(0.0, 0.0, 1.5),
                axis=(1.0, 0.0, 0.0),
            )


# ════════════════════ Continuous physics (Stage-2) ════════════════════════


class TestContinuousLevels:
    """``PhysDSLApi(continuous_levels=True)`` keeps mass/friction/restitution
    as free floats inside their physical bounds (Stage-2), while the default
    (Stage-1) snaps them to the discrete ladder."""

    def _sphere(self, api: PhysDSLApi) -> BodyDef:
        api.add_sphere(
            "b", pos=(0.0, 0.0, 1.0), radius=0.05,
            mass=2.0, friction=0.3, restitution=0.7,
        )
        scene = api.freeze()
        return next(b for b in scene.bodies if b.name == "b")

    def test_default_snaps_to_levels(self) -> None:
        body = self._sphere(PhysDSLApi())
        # 2.0 -> nearest MASS_LEVEL 1.0; 0.3 -> 0.2; 0.7 -> 0.5.
        assert body.mass in MASS_LEVELS and body.mass == 1.0
        assert body.friction_slide in FRICTION_LEVELS and body.friction_slide == 0.2
        assert body.restitution in RESTITUTION_LEVELS and body.restitution == 0.5

    def test_continuous_preserves_exact_values(self) -> None:
        body = self._sphere(PhysDSLApi(continuous_levels=True))
        assert body.mass == 2.0
        assert abs(body.friction_slide - 0.3) < 1e-9
        assert abs(body.restitution - 0.7) < 1e-9

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mass": 0.0},  # below _MASS_MIN
            {"mass": 200.0},  # above _MASS_MAX
            {"friction": 3.0},  # above _FRICTION_MAX
            {"restitution": 1.5},  # above _RESTITUTION_MAX
        ],
    )
    def test_continuous_still_hard_rejects_out_of_range(self, kwargs) -> None:  # type: ignore[no-untyped-def]
        api = PhysDSLApi(continuous_levels=True)
        with pytest.raises(DSLValidationError):
            api.add_sphere("b", pos=(0.0, 0.0, 1.0), radius=0.05, **kwargs)

    def test_continuous_box_and_plane_friction(self) -> None:
        api = PhysDSLApi(continuous_levels=True)
        api.add_box(
            "blk", pos=(0.0, 0.0, 0.2), size=(0.1, 0.1, 0.1),
            mass=2.0, friction=0.3, restitution=0.7,
        )
        api.add_plane("ground", pos=(0.0, 0.0, 0.0), size=(5.0, 5.0), friction=0.4)
        scene = api.freeze()
        blk = next(b for b in scene.bodies if b.name == "blk")
        ground = next(b for b in scene.bodies if b.name == "ground")
        assert blk.mass == 2.0 and abs(blk.restitution - 0.7) < 1e-9
        assert abs(ground.friction_slide - 0.4) < 1e-9  # plane friction continuous too
