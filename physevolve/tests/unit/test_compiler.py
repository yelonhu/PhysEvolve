"""Unit tests for :mod:`physevolve.engine.compiler`.

Tests ``engine/compiler.py``: determinism,
golden-XML snapshot, MuJoCo round-trip, contact-overflow guard,
fixed/dynamic joint handling, gravity and timestep forwarding.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

from physevolve.engine.compiler import compile_to_mjcf
from physevolve.engine.physdsl import PhysDSLApi
from physevolve.tests.golden_xml import (
    hinge_pendulum,
    single_sphere,
    spring_two_boxes,
    two_boxes_fixed_and_dynamic,
    world_hinge_pendulum,
)

GOLDEN_DIR = Path(__file__).parent.parent / "golden_xml"

FIXTURES = [
    ("single_sphere", single_sphere),
    ("two_boxes_fixed_and_dynamic", two_boxes_fixed_and_dynamic),
    ("hinge_pendulum", hinge_pendulum),
    ("spring_two_boxes", spring_two_boxes),
    ("world_hinge_pendulum", world_hinge_pendulum),
]


# ══════════════════════════ Determinism ═══════════════════════════════════


class TestDeterminism:
    @pytest.mark.parametrize("name,mod", FIXTURES)
    def test_compile_is_pure_function(self, name: str, mod) -> None:
        scene = mod.build_scene()
        xml_a = compile_to_mjcf(scene)
        xml_b = compile_to_mjcf(scene)
        assert xml_a == xml_b, f"{name}: compile_to_mjcf is not deterministic"

    def test_independent_builds_byte_equal(self) -> None:
        # Same DSL script, two PhysDSLApi instances → identical XML.
        xml_a = compile_to_mjcf(single_sphere.build_scene())
        xml_b = compile_to_mjcf(single_sphere.build_scene())
        assert xml_a == xml_b


# ══════════════════════ Golden XML snapshots ══════════════════════════════


class TestGoldenSnapshots:
    @pytest.mark.parametrize("name,mod", FIXTURES)
    def test_compile_matches_golden(self, name: str, mod) -> None:
        expected = (GOLDEN_DIR / f"{name}.xml").read_text(encoding="utf-8")
        actual = compile_to_mjcf(mod.build_scene())
        assert actual == expected, (
            f"{name} XML drifted from golden; re-seed only after a reviewed "
            "intentional change to compiler.py"
        )


# ═══════════════════════ MuJoCo round-trip ════════════════════════════════


class TestMujocoRoundTrip:
    @pytest.mark.parametrize("name,mod", FIXTURES)
    def test_mjcf_parses_in_mujoco(self, name: str, mod) -> None:
        xml = compile_to_mjcf(mod.build_scene())
        model = mujoco.MjModel.from_xml_string(xml)
        assert model.nbody >= 2  # worldbody + at least one scene body.


# ═══════════════════════ Contact overflow guard ═══════════════════════════


class TestContactGuard:
    @pytest.mark.parametrize("name,mod", FIXTURES)
    def test_contact_guard_always_present(self, name: str, mod) -> None:
        xml = compile_to_mjcf(mod.build_scene())
        assert 'nconmax="500"' in xml
        assert 'njmax="1000"' in xml


# ════════════════════ Joint emission rules ════════════════════════════════


class TestJointEmission:
    def test_fixed_body_has_no_joint(self) -> None:
        api = PhysDSLApi()
        api.add_box(
            "wall",
            pos=(0.0, 0.0, 0.3),
            size=(0.1, 0.1, 0.1),
            fixed=True,
        )
        xml = compile_to_mjcf(api.freeze())
        # Find the <body name="wall" ... > block and assert no <joint>
        # or <freejoint> appears before </body>.
        start = xml.index('<body name="wall"')
        end = xml.index("</body>", start)
        body_block = xml[start:end]
        assert "<freejoint" not in body_block
        assert "<joint" not in body_block

    def test_dynamic_body_gets_freejoint(self) -> None:
        xml = compile_to_mjcf(single_sphere.build_scene())
        start = xml.index('<body name="ball"')
        end = xml.index("</body>", start)
        body_block = xml[start:end]
        assert "<freejoint/>" in body_block

    def test_hinge_child_has_joint_not_freejoint(self) -> None:
        xml = compile_to_mjcf(hinge_pendulum.build_scene())
        start = xml.index('<body name="bob"')
        end = xml.index("</body>", start)
        body_block = xml[start:end]
        assert '<joint name="pivot" type="hinge"' in body_block
        assert "<freejoint" not in body_block


# ═══════════════════ Env (gravity / timestep) forwarding ══════════════════


class TestEnvForwarding:
    def test_default_gravity_and_timestep(self) -> None:
        xml = compile_to_mjcf(single_sphere.build_scene())
        assert 'gravity="0 0 -9.81"' in xml
        assert 'timestep="0.002"' in xml

    def test_non_default_gravity_propagates(self) -> None:
        api = PhysDSLApi()
        api.set_gravity((0.0, 0.0, -1.62))  # lunar
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        xml = compile_to_mjcf(api.freeze())
        assert 'gravity="0 0 -1.62"' in xml

    def test_non_default_duration_does_not_leak_into_option(self) -> None:
        # `duration` is not an MJCF concept; it must not appear as an
        # attribute of <option/>.
        api = PhysDSLApi()
        api.add_sphere("ball", pos=(0.0, 0.0, 0.5), radius=0.05)
        api.set_duration(5.0)
        xml = compile_to_mjcf(api.freeze())
        assert "duration=" not in xml


# ═══════════════════════════ Visual block ═════════════════════════════════


class TestVisualBlock:
    @pytest.mark.parametrize("name,mod", FIXTURES)
    def test_visual_offscreen_size(self, name: str, mod) -> None:
        xml = compile_to_mjcf(mod.build_scene())
        assert 'offwidth="480"' in xml
        assert 'offheight="480"' in xml
        assert 'name="skybox"' in xml
        assert 'rgb1="0.45 0.45 0.45"' in xml
        assert "<headlight " in xml
        assert 'dir="0 0.5 -1"' in xml
        assert 'diffuse="0.95 0.95 0.95"' in xml
        assert 'specular="0.2 0.2 0.2"' in xml


# ═══════════════════════════ Camera block ═════════════════════════════════


class TestCamera:
    @pytest.mark.parametrize("name,mod", FIXTURES)
    def test_camera_has_xyaxes(self, name: str, mod) -> None:
        xml = compile_to_mjcf(mod.build_scene())
        assert "<camera " in xml
        assert "xyaxes=" in xml


# ═══════════════════════ Parent-child MJCF nesting ═════════════════════════


class TestJointNesting:
    """Joint children must be nested inside their parent."""

    def test_hinge_child_body_is_nested_in_parent(self) -> None:
        xml = compile_to_mjcf(hinge_pendulum.build_scene())
        # The bob <body> must appear WITHIN the anchor <body> ... </body> span.
        anchor_open = xml.index('<body name="anchor"')
        anchor_close = xml.index("</body>", anchor_open)
        nested_region = xml[anchor_open:anchor_close]
        assert '<body name="bob"' in nested_region, (
            "bob <body> was not nested under anchor; MJCF hinge semantics "
            "require parent-child nesting"
        )

    def test_hinge_joint_pos_is_body_local(self) -> None:
        """Joint pos in MJCF is in the child body's local frame.

        The DSL authored ``joint.pos=(0,0,1.5)`` (world frame, same as
        the anchor). In the bob's local frame that should be
        ``(-0.3, 0, 0)``. A compiler that forgot to rebase would
        emit ``(0, 0, 1.5)`` and MuJoCo would refuse to move the bob.
        """

        xml = compile_to_mjcf(hinge_pendulum.build_scene())
        assert 'type="hinge" pos="-0.3 0 0"' in xml

    def test_springless_scene_emits_no_tendon_block(self) -> None:
        # Regression: the tendon sibling block must NOT appear when the
        # scene contains no springs. Keeps existing golden XML byte-equal.
        xml = compile_to_mjcf(single_sphere.build_scene())
        assert "<tendon>" not in xml
        assert "<site " not in xml

    def test_spring_emits_tendon_and_sites(self) -> None:
        xml = compile_to_mjcf(spring_two_boxes.build_scene())
        assert "<tendon>" in xml
        assert '<spatial name="tether"' in xml
        assert 'stiffness="50"' in xml
        assert 'damping="1.5"' in xml
        assert 'springlength="0.4"' in xml
        assert xml.count('<site name="spring__tether__') == 2
        assert xml.count('<site site="spring__tether__') == 2

    def test_spring_optional_springlength_omitted(self) -> None:
        from physevolve.engine.physdsl import PhysDSLApi
        api = PhysDSLApi()
        api.add_sphere("a", pos=(-0.2, 0.0, 0.5), radius=0.05)
        api.add_sphere("b", pos=(0.2, 0.0, 0.5), radius=0.05)
        api.add_spring("s1", body_a="a", body_b="b", stiffness=10.0, damping=0.5)
        xml = compile_to_mjcf(api.freeze())
        assert "springlength" not in xml

    def test_spring_scene_parses_in_mujoco(self) -> None:
        xml = compile_to_mjcf(spring_two_boxes.build_scene())
        model = mujoco.MjModel.from_xml_string(xml)
        assert model.ntendon == 1
        assert model.nsite == 2

    def test_pendulum_bob_actually_swings(self) -> None:
        """Integration-level sanity: gravity produces non-zero hinge motion.

        Regression for incorrectly generated XML where flat-hierarchy MJCF welded
        the bob to the world and the pendulum never moved. Uses full
        gravity (-9.81) and a wide 0.5 m offset to get a clearly
        non-zero swing independent of solver damping defaults.
        """

        api = PhysDSLApi()
        api.add_box(
            "anchor",
            pos=(0.0, 0.0, 2.0),
            size=(0.05, 0.05, 0.05),
            mass=1.0,
            fixed=True,
        )
        api.add_sphere(
            "bob",
            pos=(0.5, 0.0, 2.0),
            radius=0.08,
            mass=0.5,
        )
        api.add_hinge(
            name="pivot",
            parent="anchor",
            child="bob",
            pos=(0.0, 0.0, 2.0),
            axis=(0.0, 1.0, 0.0),
        )
        xml = compile_to_mjcf(api.freeze())
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        bob_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bob")
        start = data.xpos[bob_id].copy()
        mujoco.mj_step(model, data, 200)  # 0.4 s with default timestep.
        moved = float(((data.xpos[bob_id] - start) ** 2).sum() ** 0.5)
        assert moved > 0.05, (
            f"bob only moved {moved:.4f} m under gravity — the hinge is "
            "likely compiled as a flat sibling of anchor instead of nested"
        )


# ═══════════════════ World-parent joints ════════════════════════


class TestWorldParentedJoints:
    """World-parent joints: ``parent="world"`` shortcut for hinge/slide.

    The child body must remain at the ``<worldbody>`` level (i.e. not
    nested inside any other ``<body>``) and emit its non-free joint
    inline instead of the default ``<freejoint/>``.
    """

    def test_world_hinge_body_is_not_nested(self) -> None:
        xml = compile_to_mjcf(world_hinge_pendulum.build_scene())
        # The bob <body> must appear DIRECTLY inside <worldbody> — no
        # enclosing <body name="..."> wrapper.
        bob_open = xml.index('<body name="bob"')
        # Walk backwards to find the previous open tag; it must be
        # <worldbody>, not <body name="...">.
        before_bob = xml[:bob_open]
        last_body_open = before_bob.rfind("<body ")
        # No <body ...> opens before bob — only the worldbody container.
        assert last_body_open == -1, (
            "world-parent child must stay at <worldbody> level; found a "
            "preceding <body ...> tag"
        )

    def test_world_hinge_emits_inline_joint_not_freejoint(self) -> None:
        xml = compile_to_mjcf(world_hinge_pendulum.build_scene())
        start = xml.index('<body name="bob"')
        end = xml.index("</body>", start)
        body_block = xml[start:end]
        assert '<joint name="pivot" type="hinge"' in body_block
        assert "<freejoint" not in body_block
        # Joint MUST appear before the geom per MJCF ordering.
        joint_idx = body_block.index('<joint name="pivot"')
        geom_idx = body_block.index("<geom ")
        assert joint_idx < geom_idx

    def test_world_hinge_round_trips_in_mujoco(self) -> None:
        xml = compile_to_mjcf(world_hinge_pendulum.build_scene())
        model = mujoco.MjModel.from_xml_string(xml)
        # Worldbody (1) + bob (1) = 2; one hinge joint.
        assert model.nbody == 2
        assert model.njnt == 1
        assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_HINGE
