"""Tests for the ``motion_lab`` grouped scene template (CPU, fake view).

motion_lab hosts motion_state_change + friction_or_restitution_identification
from one lane/surface scene. These tests drive its per-family ground-truth /
validation directly with a frozen ``SceneDef`` and a fake ``SimStateView``.
"""

from __future__ import annotations

from physevolve.engine.physdsl import PhysDSLApi
from physevolve.skills.scene_templates import get_scene_template_registry
from physevolve.skills.scene_templates import motion_lab as ml


class _FakeBody:
    def __init__(
        self,
        *,
        initial_vel=(0.0, 0.0, 0.0),
        final_vel=(0.0, 0.0, 0.0),
        max_speed=None,
        rebound_height=0.0,
    ):
        self.initial_vel = initial_vel
        self.final_vel = final_vel
        self.max_speed = (
            max(ml._speed(initial_vel), ml._speed(final_vel))
            if max_speed is None
            else max_speed
        )
        self.rebound_height = rebound_height


class _FakeView:
    def __init__(self, bodies):
        self._bodies = bodies

    def __contains__(self, name):
        return name in self._bodies

    def __getitem__(self, name):
        return self._bodies[name]


def _lane_scene():
    api = PhysDSLApi()
    api.add_box("blue_box", pos=(-0.7, -0.3, 0.10), size=(0.10, 0.10, 0.10),
                mass=1.0, rgba=(0.2, 0.4, 0.9, 1.0), friction=2.0,
                restitution=0.0, fixed=False)
    api.add_box("orange_box", pos=(-0.7, 0.3, 0.10), size=(0.10, 0.10, 0.10),
                mass=1.0, rgba=(0.95, 0.55, 0.15, 1.0), friction=0.05,
                restitution=0.0, fixed=False)
    api.set_velocity("blue_box", linear=(1.8, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
    api.set_velocity("orange_box", linear=(1.8, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
    api.set_duration(2.5)
    return api.freeze()


def test_registry_lists_motion_lab_menu() -> None:
    tpl = get_scene_template_registry().lookup("motion_lab")
    assert set(tpl.families) == {
        "motion_state_change",
        "friction_or_restitution_identification",
    }
    assert tpl.color_variant_bodies({}) == []


def test_state_change_stop_bucket() -> None:
    view = _FakeView(
        {"blue_box": _FakeBody(initial_vel=(1.8, 0, 0), final_vel=(0.0, 0, 0))}
    )
    manifest = {"scene_kind": "motion_lab", "motion_target": "blue_box",
                "families": ["motion_state_change"]}
    vqa = ml._state_change(manifest, view)
    assert vqa is not None
    assert vqa.correct_idx == 0  # stop
    assert vqa.options[0] == "It comes to a stop"


def test_state_change_catalog_phrasing_variants() -> None:
    view = _FakeView(
        {
            "blue_box": _FakeBody(initial_vel=(1.0, 0, 0), final_vel=(0.0, 0, 0)),
            "orange_box": _FakeBody(initial_vel=(1.8, 0, 0), final_vel=(1.5, 0, 0)),
        }
    )
    manifest = {
        "scene_kind": "motion_lab",
        "motion_target": "blue_box",
        "speed_pair": {"a": "blue_box", "b": "orange_box"},
        "families": ["motion_state_change"],
    }
    variants = ml._state_change_variants(manifest, view)
    by_phrasing = {v.phrasing: v for v in variants}

    assert {"forward", "speed_profile", "pairwise_speed"} <= set(by_phrasing)
    assert "speed of the blue box" in by_phrasing["speed_profile"].question
    assert by_phrasing["speed_profile"].options == [
        "It accelerates.",
        "It decelerates.",
        "It first accelerates and then decelerates.",
        "It remains constant.",
    ]
    assert by_phrasing["speed_profile"].correct_idx == 1
    pairwise = by_phrasing["pairwise_speed"]
    assert "relationship between the speeds" in pairwise.question
    assert pairwise.options[pairwise.correct_idx] == (
        "The orange box consistently moves faster than the blue box."
    )


def test_speed_profile_catalog_options_cover_acceleration_shapes() -> None:
    cases = {
        "accelerates": (_FakeBody(initial_vel=(0.2, 0, 0), final_vel=(0.8, 0, 0)), 0),
        "decelerates": (_FakeBody(initial_vel=(0.8, 0, 0), final_vel=(0.2, 0, 0)), 1),
        "peak_then_slows": (
            _FakeBody(initial_vel=(0.2, 0, 0), final_vel=(0.3, 0, 0), max_speed=1.0),
            2,
        ),
        "constant": (_FakeBody(initial_vel=(0.8, 0, 0), final_vel=(0.75, 0, 0)), 3),
    }
    manifest = {
        "scene_kind": "motion_lab",
        "motion_target": "target_ball",
        "families": ["motion_state_change"],
    }
    expected_options = [
        "It accelerates.",
        "It decelerates.",
        "It first accelerates and then decelerates.",
        "It remains constant.",
    ]
    for body_state, expected_idx in cases.values():
        variants = ml._state_change_variants(
            manifest, _FakeView({"target_ball": body_state})
        )
        speed_profile = {v.phrasing: v for v in variants}["speed_profile"]
        assert speed_profile.options == expected_options
        assert speed_profile.correct_idx == expected_idx


def test_pairwise_speed_catalog_options_cover_same_and_crossover() -> None:
    manifest = {
        "scene_kind": "motion_lab",
        "motion_target": "blue_box",
        "speed_pair": {"a": "blue_box", "b": "orange_box"},
        "families": ["motion_state_change"],
    }
    same_view = _FakeView(
        {
            "blue_box": _FakeBody(initial_vel=(1.0, 0, 0), final_vel=(1.0, 0, 0)),
            "orange_box": _FakeBody(initial_vel=(1.05, 0, 0), final_vel=(1.02, 0, 0)),
        }
    )
    same = {
        v.phrasing: v for v in ml._state_change_variants(manifest, same_view)
    }["pairwise_speed"]
    assert same.correct_idx == 0
    assert same.options[0] == "Both objects have the same speed"

    crossover_view = _FakeView(
        {
            "blue_box": _FakeBody(initial_vel=(1.8, 0, 0), final_vel=(0.2, 0, 0)),
            "orange_box": _FakeBody(initial_vel=(0.2, 0, 0), final_vel=(1.5, 0, 0)),
        }
    )
    crossover = {
        v.phrasing: v for v in ml._state_change_variants(manifest, crossover_view)
    }["pairwise_speed"]
    assert crossover.correct_idx == 3
    assert crossover.options[3] == (
        "Initially, the blue box moves faster, but the orange box eventually "
        "surpasses it."
    )


def test_state_change_from_rest_skipped() -> None:
    view = _FakeView(
        {"blue_box": _FakeBody(initial_vel=(0.0, 0, 0), final_vel=(0.0, 0, 0))}
    )
    manifest = {"scene_kind": "motion_lab", "motion_target": "blue_box",
                "families": ["motion_state_change"]}
    assert ml._state_change(manifest, view) is None


def test_friction_gt_higher_friction_is_slower() -> None:
    # blue_box (a) has higher friction -> smaller final speed.
    view = _FakeView(
        {
            "blue_box": _FakeBody(final_vel=(0.1, 0, 0)),
            "orange_box": _FakeBody(final_vel=(1.5, 0, 0)),
        }
    )
    manifest = {
        "scene_kind": "motion_lab",
        "compare_pair": {"a": "blue_box", "b": "orange_box",
                         "attribute": "friction", "levels": {"a": 2.0, "b": 0.05}},
        "families": ["friction_or_restitution_identification"],
    }
    vqa = ml._friction_or_restitution(manifest, view)
    assert vqa is not None
    assert vqa.correct_idx == 0  # (a, b, same, cannot_determine)
    assert vqa.options[0] == "The blue box has higher friction"


def test_restitution_gt_higher_bounces_higher() -> None:
    view = _FakeView(
        {
            "ball_a": _FakeBody(rebound_height=0.5),
            "ball_b": _FakeBody(rebound_height=0.05),
        }
    )
    manifest = {
        "scene_kind": "motion_lab",
        "compare_pair": {"a": "ball_a", "b": "ball_b",
                         "attribute": "restitution", "levels": {"a": 0.9, "b": 0.0}},
        "families": ["friction_or_restitution_identification"],
    }
    vqa = ml._friction_or_restitution(manifest, view)
    assert vqa is not None
    assert vqa.correct_idx == 0
    assert "restitution" in vqa.options[0]


def test_friction_mismatch_skipped() -> None:
    # Declared a higher friction, but a is actually faster -> skip.
    view = _FakeView(
        {
            "blue_box": _FakeBody(final_vel=(1.5, 0, 0)),
            "orange_box": _FakeBody(final_vel=(0.1, 0, 0)),
        }
    )
    manifest = {
        "scene_kind": "motion_lab",
        "compare_pair": {"a": "blue_box", "b": "orange_box",
                         "attribute": "friction", "levels": {"a": 2.0, "b": 0.05}},
        "families": ["friction_or_restitution_identification"],
    }
    assert ml._friction_or_restitution(manifest, view) is None


def test_validate_rejects_missing_body() -> None:
    scene = _lane_scene()
    manifest = {"scene_kind": "motion_lab", "motion_target": "ghost",
                "families": ["motion_state_change"]}
    reason = ml._validate_manifest(manifest, scene)
    assert reason is not None and "ghost" in reason


def test_validate_accepts_multi_family_manifest() -> None:
    scene = _lane_scene()
    manifest = {
        "scene_kind": "motion_lab",
        "motion_target": "blue_box",
        "compare_pair": {"a": "blue_box", "b": "orange_box",
                         "attribute": "friction", "levels": {"a": 2.0, "b": 0.05}},
        "families": ["motion_state_change",
                     "friction_or_restitution_identification"],
    }
    assert ml._validate_manifest(manifest, scene) is None


def test_generate_emits_feasible_families() -> None:
    scene = _lane_scene()
    view = _FakeView(
        {
            "blue_box": _FakeBody(initial_vel=(1.8, 0, 0), final_vel=(0.0, 0, 0)),
            "orange_box": _FakeBody(final_vel=(1.5, 0, 0)),
        }
    )
    manifest = {
        "scene_kind": "motion_lab",
        "motion_target": "blue_box",
        "compare_pair": {"a": "blue_box", "b": "orange_box",
                         "attribute": "friction", "levels": {"a": 2.0, "b": 0.05}},
        "families": ["motion_state_change",
                     "friction_or_restitution_identification"],
    }
    gen = list(ml._generate_questions(manifest, view, {}, {}, scene))
    assert {g.family for g in gen} == {
        "motion_state_change",
        "friction_or_restitution_identification",
    }
