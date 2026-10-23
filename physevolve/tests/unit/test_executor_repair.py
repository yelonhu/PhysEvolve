"""Unit tests for deterministic executor repair helpers."""

from __future__ import annotations

from physevolve.engine.schema import (
    BodyDef,
    EnvConfig,
    GeomType,
    JointType,
    SceneDef,
)
from physevolve.executor.repair import RepairPolicy, RepairTrace, repair_scene_def


def _sphere(name: str, pos: tuple[float, float, float]) -> BodyDef:
    return BodyDef(
        name=name,
        geom_type=GeomType.SPHERE,
        pos=pos,
        size=(0.1,),
        mass=1.0,
        rgba=(0.5, 0.5, 0.5, 1.0),
        friction_slide=0.5,
        restitution=0.0,
        joint_type=JointType.FREE,
        fixed=False,
    )


def test_repair_scene_def_lifts_and_separates_free_bodies() -> None:
    scene = SceneDef(
        bodies=(
            _sphere("a", (0.0, 0.0, 0.02)),
            _sphere("b", (0.0, 0.0, 0.02)),
        ),
        joints=(),
        cameras=(),
        env=EnvConfig(duration=0.5),
        workspace_bounds=(1.5, 1.5, 3.0),
    )
    trace = RepairTrace()
    repaired = repair_scene_def(
        scene,
        trace=trace,
        policy=RepairPolicy(overlap_iterations=2),
    )

    a, b = repaired.bodies
    assert a.pos[2] >= 0.12
    assert b.pos[2] >= 0.12
    assert a.pos != b.pos
    assert any(action.kind == "repair_body_pos" for action in trace.actions)
    assert any(action.kind == "separate_initial_overlap" for action in trace.actions)
    assert trace.as_dict()["tier"] == "geometry"


def test_collision_override_does_not_change_throwing_policy() -> None:
    base = RepairPolicy()

    collision = base.for_scene_kind("collision_arena")
    throwing = base.for_scene_kind("throwing_pit_board")
    motion = base.for_scene_kind("motion_lab")

    assert collision.overlap_iterations == 6
    assert collision.overlap_margin == 0.03
    assert throwing.overlap_iterations == base.overlap_iterations == 4
    assert throwing.overlap_margin == base.overlap_margin == 0.02
    assert motion.overlap_iterations == base.overlap_iterations
    assert motion.overlap_margin == base.overlap_margin
