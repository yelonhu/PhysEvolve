"""Tests for the ``collision_arena`` grouped scene template (CPU, fake view).

The collision_arena template hosts three active impact-centric families from one
simulation, while frozen first-contact / mass handlers are excluded from active training.
These tests drive its per-family ground-truth / validation logic directly with a
frozen ``SceneDef`` and a fake ``SimStateView`` (no MuJoCo), and assert that one
role-based manifest emits several feasible questions (multiple questions per scene).
"""

from __future__ import annotations

from physevolve.engine.physdsl import PhysDSLApi
from physevolve.engine.schema import BodyStatus
from physevolve.skills.scene_templates import collision_arena as ca
from physevolve.skills.scene_templates import get_scene_template_registry


# ─── fake SimStateView ────────────────────────────────────────────────


class _FakeBody:
    def __init__(
        self,
        *,
        initial_pos=(0.0, 0.0, 0.0),
        final_pos=(0.0, 0.0, 0.0),
        initial_vel=(0.0, 0.0, 0.0),
        final_vel=(0.0, 0.0, 0.0),
        final_status="stable",
        rebound_height=0.0,
    ):
        self.initial_pos = initial_pos
        self.final_pos = final_pos
        self.initial_vel = initial_vel
        self.final_vel = final_vel
        self.final_status = final_status
        self.rebound_height = rebound_height


class _FakeView:
    def __init__(self, bodies, *, collisions=None, times=None):
        self._bodies = bodies
        self._collisions = {frozenset(p) for p in (collisions or [])}
        self._times = {frozenset(k): v for k, v in (times or {}).items()}

    def __contains__(self, name):
        return name in self._bodies

    def __getitem__(self, name):
        return self._bodies[name]

    def collided(self, a, b):
        return frozenset((a, b)) in self._collisions

    def time_of_collision(self, a, b):
        return self._times.get(frozenset((a, b)))


# ─── scene helpers ────────────────────────────────────────────────────


def _arena_scene():
    """Freeze a 1 target + 4 dynamic movers scene (all free dynamic)."""

    api = PhysDSLApi()
    api.add_sphere("gray_target", pos=(0.0, 0.0, 0.10), radius=0.10, mass=1.0,
                   rgba=(0.7, 0.7, 0.7, 1.0), friction=0.2, restitution=0.5,
                   fixed=False)
    for name, x, y in (
        ("red_ball", -0.85, 0.0),
        ("green_ball", 0.0, -1.1),
        ("blue_ball", 1.25, 0.0),
        ("yellow_ball", 0.0, 0.65),
    ):
        api.add_sphere(name, pos=(x, y, 0.10), radius=0.08, mass=1.0,
                       rgba=(0.5, 0.5, 0.5, 1.0), friction=0.2,
                       restitution=0.5, fixed=False)
    api.set_duration(2.0)
    return api.freeze()


# ─── registry ─────────────────────────────────────────────────────────


def test_registry_lists_collision_arena_menu() -> None:
    tpl = get_scene_template_registry().lookup("collision_arena")
    assert set(tpl.families) == {
        "collision_final_state",
        "collision_final_relation",
        "motion_attribution",
    }
    assert tpl.color_variant_bodies({}) == []


# ─── first_contact ────────────────────────────────────────────────────


def test_first_contact_gt_is_earliest_collider() -> None:
    movers = ["red_ball", "green_ball", "blue_ball", "yellow_ball"]
    view = _FakeView(
        {b: _FakeBody() for b in ["gray_target", *movers]},
        times={
            ("gray_target", "red_ball"): 5,
            ("gray_target", "green_ball"): 10,
            ("gray_target", "yellow_ball"): 8,
            # blue_ball never collides.
        },
    )
    manifest = {
        "scene_kind": "collision_arena",
        "target": "gray_target",
        "movers": movers,
        "families": ["collision_first_contact"],
    }
    vqa = ca._first_contact(manifest, view)
    assert vqa is not None
    assert vqa.family == "collision_first_contact"
    assert vqa.options[vqa.correct_idx] == "The red ball"


def test_first_contact_ambiguous_order_skipped() -> None:
    movers = ["red_ball", "green_ball", "blue_ball", "yellow_ball"]
    view = _FakeView(
        {b: _FakeBody() for b in ["gray_target", *movers]},
        times={
            ("gray_target", "red_ball"): 5,
            ("gray_target", "green_ball"): 6,  # within < 2 steps of red
        },
    )
    manifest = {
        "scene_kind": "collision_arena",
        "target": "gray_target",
        "movers": movers,
        "families": ["collision_first_contact"],
    }
    assert ca._first_contact(manifest, view) is None


# ─── final_state ──────────────────────────────────────────────────────


def test_final_state_gt_maps_status() -> None:
    view = _FakeView({"gray_target": _FakeBody(final_status="fallen")})
    manifest = {"scene_kind": "collision_arena", "target": "gray_target",
                "families": ["collision_final_state"]}
    vqa = ca._final_state(manifest, view)
    assert vqa is not None
    assert vqa.correct_idx == 2  # stable, moving, fallen, ejected
    assert "Lying horizontally" in vqa.options[vqa.correct_idx]


def test_final_state_borderline_moving_skipped() -> None:
    view = _FakeView(
        {"gray_target": _FakeBody(final_status="moving", final_vel=(0.05, 0, 0))}
    )
    manifest = {"scene_kind": "collision_arena", "target": "gray_target",
                "families": ["collision_final_state"]}
    assert ca._final_state(manifest, view) is None


# ─── final_relation ───────────────────────────────────────────────────


def test_final_relation_contact_gt() -> None:
    view = _FakeView(
        {"a": _FakeBody(), "b": _FakeBody()},
        collisions=[("a", "b")],
    )
    manifest = {
        "scene_kind": "collision_arena",
        "relation_pair": {"a": "a", "b": "b", "expected": "contact"},
        "families": ["collision_final_relation"],
    }
    vqa = ca._final_relation(manifest, view)
    assert vqa is not None and vqa.correct_idx == 0


def test_final_relation_catalog_phrasing_variants() -> None:
    view = _FakeView(
        {
            "a": _FakeBody(initial_pos=(0.0, 0.0, 0.1), final_pos=(0.0, 0.0, 0.1)),
            "b": _FakeBody(initial_pos=(1.0, 0.0, 0.1), final_pos=(1.0, 0.0, 0.1)),
        }
    )
    manifest = {
        "scene_kind": "collision_arena",
        "relation_pair": {"a": "a", "b": "b", "expected": "no_contact"},
        "families": ["collision_final_relation"],
    }
    variants = ca._final_relation_variants(manifest, view)
    by_phrasing = {v.phrasing: v for v in variants}

    assert set(by_phrasing) == {
        "forward",
        "encounter",
        "between",
        "position",
        "binary_collide",
    }
    assert "encounters" in by_phrasing["encounter"].question
    assert "between" in by_phrasing["between"].question
    assert "in relation to" in by_phrasing["position"].question
    binary = by_phrasing["binary_collide"]
    assert binary.options == ["Yes", "No", "Sometimes", "Uncertain"]
    assert binary.correct_idx == 1  # no_contact -> No


def test_final_relation_uses_rgba_derived_color_phrases() -> None:
    view = _FakeView(
        {
            "left_plate": _FakeBody(
                initial_pos=(0.0, 0.0, 0.1),
                final_pos=(0.0, 0.0, 0.1),
            ),
            "right_pillar": _FakeBody(
                initial_pos=(1.0, 0.0, 0.1),
                final_pos=(1.0, 0.0, 0.1),
            ),
        }
    )
    manifest = {
        "scene_kind": "collision_arena",
        "relation_pair": {
            "a": "left_plate",
            "b": "right_pillar",
            "expected": "no_contact",
        },
        "families": ["collision_final_relation"],
    }

    variants = ca._final_relation_variants(
        manifest,
        view,
        {"left_plate": "cyan", "right_pillar": "yellow"},
    )

    assert len(variants) == 5
    for vqa in variants:
        assert "cyan plate" in vqa.question.lower()
        assert "yellow pillar" in vqa.question.lower()
        assert vqa.spec["body_a_phrase"] == "cyan plate"
        assert vqa.spec["body_b_phrase"] == "yellow pillar"
    assert all(
        "cyan plate" in option.lower() and "yellow pillar" in option.lower()
        for option in variants[0].options
    )


def test_final_relation_mismatch_skipped() -> None:
    # Declared no_contact but they actually collided -> skip (no false GT).
    view = _FakeView(
        {"a": _FakeBody(), "b": _FakeBody()},
        collisions=[("a", "b")],
    )
    manifest = {
        "scene_kind": "collision_arena",
        "relation_pair": {"a": "a", "b": "b", "expected": "no_contact"},
        "families": ["collision_final_relation"],
    }
    assert ca._final_relation(manifest, view) is None


# ─── attribution ──────────────────────────────────────────────────────


def test_attribution_gt_is_max_delta_collider() -> None:
    movers = ["red_ball", "green_ball", "blue_ball", "yellow_ball"]
    bodies = {"brown_hammer": _FakeBody()}
    # green_ball is struck hardest.
    bodies["red_ball"] = _FakeBody(initial_vel=(0, 0, 0), final_vel=(0, 0, 0))
    bodies["green_ball"] = _FakeBody(initial_vel=(0, 0, 0), final_vel=(2.0, 0, 0))
    bodies["blue_ball"] = _FakeBody(initial_vel=(0, 0, 0), final_vel=(0, 0, 0))
    bodies["yellow_ball"] = _FakeBody(initial_vel=(0, 0, 0), final_vel=(0, 0, 0))
    view = _FakeView(bodies, collisions=[("brown_hammer", "green_ball")])
    manifest = {
        "scene_kind": "collision_arena",
        "trigger": "brown_hammer",
        "movers": movers,
        "families": ["motion_attribution"],
    }
    vqa = ca._attribution(manifest, view)
    assert vqa is not None
    assert vqa.options[vqa.correct_idx] == "The green ball"


# ─── mass ─────────────────────────────────────────────────────────────


def test_mass_comparison_gt_is_heavier() -> None:
    # b is heavier; lighter a shows the larger speed change.
    view = _FakeView(
        {
            "a": _FakeBody(initial_vel=(2.5, 0, 0), final_vel=(-1.5, 0, 0)),
            "b": _FakeBody(initial_vel=(-1.0, 0, 0), final_vel=(-0.9, 0, 0)),
        },
        collisions=[("a", "b")],
    )
    manifest = {
        "scene_kind": "collision_arena",
        "mass_pair": {"a": "a", "b": "b", "true_masses": {"a": 0.5, "b": 10.0}},
        "families": ["mass_comparison_by_dynamics"],
    }
    vqa = ca._mass_comparison(manifest, view)
    assert vqa is not None
    assert vqa.correct_idx == 1  # (a, b, same, cannot_determine) -> b heavier
    assert vqa.options[1] == "The b is heavier"


# ─── validation + multi-family generation ─────────────────────────────


def test_validate_rejects_fixed_target() -> None:
    api = PhysDSLApi()
    api.add_sphere("gray_target", pos=(0.0, 0.0, 0.10), radius=0.10, mass=1.0,
                   rgba=(0.7, 0.7, 0.7, 1.0), fixed=True)
    for name, x in (("red_ball", -0.8), ("green_ball", -0.4),
                    ("blue_ball", 0.4), ("yellow_ball", 0.8)):
        api.add_sphere(name, pos=(x, 0.5, 0.10), radius=0.08, mass=1.0,
                       rgba=(0.5, 0.5, 0.5, 1.0), fixed=False)
    api.set_duration(1.0)
    scene = api.freeze()
    manifest = {
        "scene_kind": "collision_arena",
        "target": "gray_target",
        "families": ["collision_final_state"],
    }
    reason = ca._validate_manifest(manifest, scene)
    assert reason is not None and "free dynamic" in reason


def test_validate_rejects_unknown_family() -> None:
    scene = _arena_scene()
    manifest = {
        "scene_kind": "collision_arena",
        "target": "gray_target",
        "movers": ["red_ball", "green_ball", "blue_ball", "yellow_ball"],
        "families": ["not_a_family"],
    }
    reason = ca._validate_manifest(manifest, scene)
    assert reason is not None


def test_validate_accepts_multi_family_manifest() -> None:
    scene = _arena_scene()
    manifest = {
        "scene_kind": "collision_arena",
        "target": "gray_target",
        "trigger": "red_ball",
        "movers": ["green_ball", "blue_ball", "yellow_ball", "gray_target"],
        "families": ["motion_attribution", "collision_final_state"],
    }
    assert ca._validate_manifest(manifest, scene) is None


def test_generate_emits_one_question_per_feasible_family() -> None:
    scene = _arena_scene()
    movers = ["green_ball", "blue_ball", "yellow_ball", "gray_target"]
    view = _FakeView(
        {
            "gray_target": _FakeBody(final_status="fallen"),
            "red_ball": _FakeBody(),
            "green_ball": _FakeBody(initial_vel=(0, 0, 0), final_vel=(2.0, 0, 0)),
            "blue_ball": _FakeBody(),
            "yellow_ball": _FakeBody(),
        },
        collisions=[("red_ball", "green_ball")],
    )
    manifest = {
        "scene_kind": "collision_arena",
        "target": "gray_target",
        "trigger": "red_ball",
        "movers": movers,
        "families": ["motion_attribution", "collision_final_state"],
    }
    gen = list(ca._generate_questions(manifest, view, {}, {}, scene))
    families = {g.family for g in gen}
    assert families == {"motion_attribution", "collision_final_state"}
