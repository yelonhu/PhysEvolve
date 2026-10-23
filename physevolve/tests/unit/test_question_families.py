"""Tests for the PhysEvolve question-family cards.

The question-family layer (:mod:`physevolve.skills.question_families`) sits
ABOVE the verifiers. Every active family
is served by one of three grouped scene templates
(:mod:`physevolve.skills.scene_templates`): ``throwing_pit_board`` /
``collision_arena`` / ``motion_lab``. These tests guard:

* registry integrity (every family is served by its scene template; the
  template actually lists the family);
* ``active_families()`` excludes the reserved P1 stubs and the frozen families;
* the Proposer prompt injects the family name + an exemplar + the evidence rule;
* the family sampler returns only active P0 families, is deterministic for a
  fixed seed, and emphasises priority families over non-priority ones.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from physevolve.prompts.proposer_stage2 import (
    build_proposer_scene_template_prompt,
)
from physevolve.skills.question_families import (
    BRANCH_ELISION_FAMILIES,
    QUESTION_FAMILY_REGISTRY,
    active_families,
    lookup_family,
    priority_families,
    validate_registry,
)
from physevolve.skills.scene_templates import get_scene_template_registry
from physevolve.training.adaptive_curriculum_stage2 import (
    DEFAULT_FAMILY_PRIORITY_BOOST,
    DEFAULT_FAMILY_WEIGHT_MULTIPLIERS,
    make_stage2_tracker,
    pick_stage2_question_families_for_step,
    pick_stage2_question_family,
    question_family_vocab,
    stage2_family_pick_weight_snapshot,
)

# Families assigned the curriculum's priority multiplier.
_EXPECTED_PRIORITY = {
    "throwing_final_destination",
    "throwing_current_run_judgment",
    "collision_final_state",
    "motion_state_change",
}
_EXPECTED_ACTIVE = {
    "throwing_final_destination",
    "throwing_current_run_judgment",
    "counterfactual_destination",
    "inverse_intervention",
    "collision_final_state",
    "collision_final_relation",
    "motion_attribution",
    "motion_state_change",
}


def test_registry_validates_against_templates() -> None:
    # Raises on any inconsistency (unknown scene_kind, family not served).
    validate_registry()


def test_every_active_family_is_served_by_its_scene_template() -> None:
    scene_reg = get_scene_template_registry()
    for name in active_families(include_p1=False):
        family = lookup_family(name)
        assert family.scene_kind, f"{name} active family has no scene_kind"
        template = scene_reg.lookup(family.scene_kind)
        assert name in template.families, (
            f"{name} declares scene_kind {family.scene_kind!r} but the "
            f"template serves {template.families}"
        )


def test_priority_families_match_configured_set() -> None:
    assert set(priority_families()) == _EXPECTED_PRIORITY
    for name in _EXPECTED_PRIORITY:
        assert lookup_family(name).tier == "P0"


def test_active_families_exclude_p1_stubs() -> None:
    active = set(active_families(include_p1=False))
    all_fams = set(active_families(include_p1=True))
    p1 = {n for n, f in QUESTION_FAMILY_REGISTRY.items() if f.tier == "P1"}
    frozen = {n for n, f in QUESTION_FAMILY_REGISTRY.items() if f.frozen}
    assert p1, "expected at least one reserved P1 family"
    assert active == _EXPECTED_ACTIVE
    assert active.isdisjoint(p1)
    assert all_fams == active | (p1 - frozen)
    # Viewpoint / screen-location / image-option families remain P1 (need
    # renderer metadata that does not exist yet).
    for reserved in (
        "screen_left_right_location",
        "viewpoint_motion_basic",
        "image_option_temporal_order",
        "camera_depth_closest_object",
    ):
        assert reserved in p1
        assert reserved not in active


def test_branch_elision_families_are_the_active_cf_inverse_pair() -> None:
    active = set(active_families(include_p1=False))
    assert BRANCH_ELISION_FAMILIES <= active
    assert BRANCH_ELISION_FAMILIES == {
        "counterfactual_destination",
        "inverse_intervention",
    }
    assert "throwing_final_destination" in active
    assert "throwing_current_run_judgment" in active


def test_frozen_families_excluded_from_active() -> None:
    frozen = {n for n, f in QUESTION_FAMILY_REGISTRY.items() if f.frozen}
    # Frozen families are excluded from active training.
    assert {
        "motion_phenomenon_classification",
        "camera_depth_closest_object",
        "collision_first_contact",
        "mass_comparison_by_dynamics",
        "friction_or_restitution_identification",
    } <= frozen
    active = set(active_families(include_p1=True))
    assert active.isdisjoint(frozen)
    # Their scene templates remain registered.
    scene_reg = get_scene_template_registry()
    for name in frozen:
        fam = lookup_family(name)
        if fam.scene_kind:
            scene_reg.lookup(fam.scene_kind)


def test_p1_family_records_required_extension() -> None:
    for family in QUESTION_FAMILY_REGISTRY.values():
        if family.tier == "P1":
            assert family.required_extension, (
                f"P1 family {family.name} must document its required extension"
            )
            assert not family.priority


def test_prompt_injects_family_context() -> None:
    family = lookup_family("throwing_final_destination")
    prompt = build_proposer_scene_template_prompt(
        family.scene_kind, question_family=family.name
    )
    assert "## Expected Grounding" in prompt
    assert family.name in prompt
    assert "Example generated question" in prompt
    assert family.stem_exemplars[0] in prompt
    assert "Options shown to Solver" in prompt
    assert "GT source" in prompt
    assert family.evidence_rule in prompt
    # The recommended phrasing nudge must appear.
    assert family.recommended_phrasings[0] in prompt
    assert f"## Required Family: `{family.name}`" in prompt
    assert f"`SCENE_MANIFEST['families']` must include `{family.name}`" in prompt
    # Failure-mode lists must stay out of the Proposer prompt.
    assert "Avoid these common mistakes" not in prompt
    for failure_mode in family.failure_modes:
        assert failure_mode not in prompt
    # The scene-centric guidance (good scene pattern + reusable tip) appears.
    assert "reusable" in prompt.lower()
    # The Proposer always authors a SCENE_MANIFEST (never VQA_SPEC).
    assert "SCENE_MANIFEST" in prompt


def test_prompt_injects_single_card_family_context() -> None:
    family = lookup_family("collision_final_state")
    prompt = build_proposer_scene_template_prompt(
        family.scene_kind, question_family=family.name
    )
    assert family.stem_exemplars[0] in prompt
    assert "## Expected Grounding" in prompt
    assert "GT source" in prompt
    assert "SCENE_MANIFEST" in prompt
    assert "VQA_SPEC for skill" not in prompt  # retired single-question header


def test_catalog_aligned_recommended_phrasings() -> None:
    assert "fate" in lookup_family(
        "throwing_final_destination"
    ).recommended_phrasings
    assert "wrong_pit" in lookup_family(
        "throwing_current_run_judgment"
    ).recommended_phrasings
    relation = lookup_family("collision_final_relation")
    assert {
        "encounter",
        "between",
        "binary_collide",
        "position",
    } <= set(relation.recommended_phrasings)
    motion = lookup_family("motion_state_change")
    assert {"speed_profile", "pairwise_speed"} <= set(
        motion.recommended_phrasings
    )
    assert "It first accelerates and then decelerates" in motion.option_style
    assert "crossover-surpass" in motion.evidence_rule
    for family in (
        "throwing_final_destination",
        "throwing_current_run_judgment",
        "collision_final_state",
        "collision_final_relation",
        "motion_state_change",
    ):
        assert not (
            set(lookup_family(family).recommended_phrasings)
            & {"where", "judgment", "post_collision", "perception"}
        )


def test_main_training_prompt_injects_golden_wiring_card() -> None:
    family = lookup_family("collision_final_state")
    prompt = build_proposer_scene_template_prompt(
        family.scene_kind,
        question_family=family.name,
        include_fewshot=False,
    )

    assert "## Golden Wiring Card: collision_arena" in prompt
    assert "## PhysDSL API Reference" in prompt
    assert "## Scene API Focus" in prompt
    assert "## Scene Hard Constraints" in prompt
    assert "## Preflight Checks Before <propose>" in prompt
    assert "### Apparatus Wiring" in prompt
    assert "### Manifest Wiring" in prompt
    assert "### Syntax Anchors" in prompt
    assert "### Topology checklist" not in prompt
    assert "### Example grounding" not in prompt
    assert family.evidence_rule in prompt
    assert "### Good scene pattern" not in prompt
    assert "All dynamic bodies have zero initial motion" in prompt
    assert "non-zero linear or angular motion" in prompt
    assert "staggered y offsets or clear x gaps" in prompt
    assert "The required family can produce at least one VQA" in prompt
    assert "## Physics Modules" not in prompt
    assert "### hinge_striker" not in prompt
    assert "## Design Target: collision_arena" not in prompt


def test_prompt_rejects_family_scene_kind_mismatch() -> None:
    # throwing_final_destination is served by throwing_pit_board, not motion_lab.
    with pytest.raises(ValueError):
        build_proposer_scene_template_prompt(
            "motion_lab",
            question_family="throwing_final_destination",
        )


def test_prompt_without_family_renders_scene_section() -> None:
    prompt = build_proposer_scene_template_prompt("collision_arena")
    assert "## Scene Template: `collision_arena`" in prompt
    assert "SCENE_MANIFEST" in prompt
    # No family task-card header when no family is supplied.
    assert "Expected Grounding" not in prompt


def test_family_vocab_matches_active_families() -> None:
    assert question_family_vocab() == active_families(include_p1=False)


def test_sampler_returns_only_active_families() -> None:
    tracker = make_stage2_tracker()
    rng = random.Random(7)
    active = set(active_families(include_p1=False))
    for step in (1, 5, 50):
        for _ in range(200):
            pick = pick_stage2_question_family(
                rng, tracker, step=step, cold_start_uniform_steps=8
            )
            assert pick in active


def test_sampler_is_deterministic_for_fixed_seed() -> None:
    tracker = make_stage2_tracker()
    seq_a = [
        pick_stage2_question_family(
            random.Random(123), tracker, step=50, cold_start_uniform_steps=0
        )
        for _ in range(1)
    ]
    seq_b = [
        pick_stage2_question_family(
            random.Random(123), tracker, step=50, cold_start_uniform_steps=0
        )
        for _ in range(1)
    ]
    assert seq_a == seq_b


def test_scene_kind_quotas_floor_all_kinds_and_preserve_one_free_slot() -> None:
    tracker = make_stage2_tracker()
    extra_kinds: set[str] = set()
    quotas = (
        ("throwing_pit_board", 1),
        ("collision_arena", 1),
        ("motion_lab", 1),
    )

    for seed in range(64):
        picks = pick_stage2_question_families_for_step(
            random.Random(seed),
            tracker,
            step=50,
            n_picks=4,
            min_scene_kind_picks=quotas,
            cold_start_uniform_steps=0,
        )
        scene_kinds = [lookup_family(family).scene_kind for family in picks]
        assert len(scene_kinds) == 4
        assert scene_kinds.count("throwing_pit_board") >= 1
        assert scene_kinds.count("collision_arena") >= 1
        assert scene_kinds.count("motion_lab") >= 1
        extra_kinds.add(next(kind for kind in scene_kinds if scene_kinds.count(kind) == 2))

    # The fourth slot still uses the unrestricted family sampler rather than
    # being pinned to one of the three scene kinds.
    assert extra_kinds == {
        "throwing_pit_board",
        "collision_arena",
        "motion_lab",
    }


def test_scene_kind_quotas_reject_overcommitted_step() -> None:
    with pytest.raises(ValueError, match="require 5 picks"):
        pick_stage2_question_families_for_step(
            random.Random(0),
            make_stage2_tracker(),
            step=1,
            n_picks=4,
            min_scene_kind_picks=(("collision_arena", 2), ("motion_lab", 3)),
        )


def test_sampler_rejects_p1_family_in_vocab() -> None:
    tracker = make_stage2_tracker()
    rng = random.Random(0)
    with pytest.raises(NotImplementedError):
        pick_stage2_question_family(
            rng,
            tracker,
            step=50,
            families=("viewpoint_motion_basic",),
        )
    with pytest.raises(NotImplementedError):
        pick_stage2_question_family(
            rng,
            tracker,
            step=50,
            families=("camera_depth_closest_object",),
        )


def test_sampler_rejects_frozen_family_in_vocab() -> None:
    tracker = make_stage2_tracker()
    rng = random.Random(0)
    with pytest.raises(NotImplementedError):
        pick_stage2_question_family(
            rng,
            tracker,
            step=50,
            families=("friction_or_restitution_identification",),
        )


def test_priority_families_sampled_more_than_non_priority() -> None:
    # With a fresh tracker (all acc neutral -> tent equal), priority families
    # carry the boost so they should collectively outweigh non-priority.
    tracker = make_stage2_tracker()
    rng = random.Random(2024)
    counts: Counter[str] = Counter(
        pick_stage2_question_family(
            rng, tracker, step=100, cold_start_uniform_steps=0
        )
        for _ in range(8000)
    )
    prio = set(priority_families())
    active = set(active_families(include_p1=False))
    prio_share = sum(v for k, v in counts.items() if k in prio) / sum(
        counts.values()
    )
    # Priority and explicit per-family multipliers compose multiplicatively.
    b = DEFAULT_FAMILY_PRIORITY_BOOST
    raw = {
        family: (b if family in prio else 1.0)
        * DEFAULT_FAMILY_WEIGHT_MULTIPLIERS.get(family, 1.0)
        for family in active
    }
    expected = sum(raw[family] for family in prio) / sum(raw.values())
    assert abs(prio_share - expected) < 0.05


def test_family_pick_weight_snapshot_normalises() -> None:
    tracker = make_stage2_tracker()
    snap = stage2_family_pick_weight_snapshot(tracker)
    assert set(snap) == set(active_families(include_p1=False))
    assert abs(sum(snap.values()) - 1.0) < 1e-9
    # Priority families should each have a higher normalised weight than plain
    # non-priority ones under a neutral tracker. The relation family receives a
    # separate catalog multiplier and is checked independently.
    prio = set(priority_families())
    prio_w = min(w for k, w in snap.items() if k in prio)
    non_prio_w = max(
        w
        for k, w in snap.items()
        if k not in prio and k != "collision_final_relation"
    )
    assert prio_w > non_prio_w
    assert snap["collision_final_relation"] > non_prio_w
    assert snap["counterfactual_destination"] == pytest.approx(
        snap["inverse_intervention"]
    )
    assert snap["counterfactual_destination"] > snap["motion_attribution"]


def test_throwing_branch_multiplier_is_conservative() -> None:
    tracker = make_stage2_tracker()
    throwing = (
        "throwing_final_destination",
        "throwing_current_run_judgment",
        "counterfactual_destination",
        "inverse_intervention",
    )

    snap = stage2_family_pick_weight_snapshot(tracker, families=throwing)
    branch_share = (
        snap["counterfactual_destination"] + snap["inverse_intervention"]
    )

    assert DEFAULT_FAMILY_WEIGHT_MULTIPLIERS["counterfactual_destination"] == 1.5
    assert DEFAULT_FAMILY_WEIGHT_MULTIPLIERS["inverse_intervention"] == 1.5
    assert branch_share == pytest.approx(3.0 / 7.0)
    assert set(priority_families()).isdisjoint(
        {"counterfactual_destination", "inverse_intervention"}
    )


def test_throwing_branch_multiplier_supports_paper_override() -> None:
    tracker = make_stage2_tracker()
    throwing = (
        "throwing_final_destination",
        "throwing_current_run_judgment",
        "counterfactual_destination",
        "inverse_intervention",
    )
    multipliers = dict(DEFAULT_FAMILY_WEIGHT_MULTIPLIERS)
    multipliers["counterfactual_destination"] = 1.2
    multipliers["inverse_intervention"] = 1.2

    snap = stage2_family_pick_weight_snapshot(
        tracker,
        families=throwing,
        family_weight_multipliers=multipliers,
    )

    branch_share = (
        snap["counterfactual_destination"] + snap["inverse_intervention"]
    )
    assert branch_share == pytest.approx(2.4 / 6.4)
