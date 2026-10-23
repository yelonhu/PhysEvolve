"""Tests for Proposer and Solver prompt structure and formatting.
"""

from __future__ import annotations

import re

import pytest

from physevolve.engine.schema import FRICTION_LEVELS, MASS_LEVELS
from physevolve.prompts import (
    DIFFICULTY_LEVELS,
    FORWARD_TEMPLATE,
    INVERSE_TEMPLATE,
    MECHANISM_TARGETS,
    PARAMETER_TEMPLATE,
    PHYSICS_SKILL_TARGETS,
    MC_SHORT_ANSWER_INSTRUCTION,
    PROPOSER_STAGE1_TEMPLATE,
    PROPOSER_STAGE2_TEMPLATE,
    VQA_TEMPLATE,
    VQA_TRAINING_TEMPLATE,
    build_proposer_scene_template_prompt,
    build_proposer_stage1_prompt,
)


# ─── Proposer Stage-1 ────────────────────────────────────────────────


def test_proposer_stage1_has_key_surface_methods() -> None:
    tmpl = PROPOSER_STAGE1_TEMPLATE
    for method in (
        "api.add_sphere",
        "api.add_box",
        "api.add_cylinder",
        "api.add_capsule",
        "api.add_plane",
        "api.add_hinge",
        "api.add_slide",
        "api.set_velocity",
        "api.set_rotation",
        "api.set_gravity",
        "api.set_duration",
        "api.add_camera",
    ):
        assert method in tmpl, f"missing {method}"


def test_proposer_stage1_mentions_mass_and_friction_levels() -> None:
    tmpl = PROPOSER_STAGE1_TEMPLATE
    for m in MASS_LEVELS:
        assert str(m) in tmpl, f"missing mass level {m}"
    for f in FRICTION_LEVELS:
        assert str(f) in tmpl, f"missing friction level {f}"


def test_proposer_stage1_scaffolds_propose_without_required_think() -> None:
    tmpl = PROPOSER_STAGE1_TEMPLATE
    # Instruct base. Proposer prompt must require <propose>...</propose>
    # but must NOT scaffold or require <think>...</think>.
    assert "<propose>" in tmpl and "</propose>" in tmpl
    assert "<answer>" not in tmpl  # Stage-1 Proposer does not use <answer>
    assert "<think>" not in tmpl and "</think>" not in tmpl
    # ``skill_target=None`` keeps the generic prompt (legacy smoke path).
    assert build_proposer_stage1_prompt() == PROPOSER_STAGE1_TEMPLATE
    assert build_proposer_stage1_prompt(None) == PROPOSER_STAGE1_TEMPLATE


_SKILL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "dynamics_collision": ("collision",),
    "dynamics_chain_collision": ("chain", "collision"),
    "dynamics_throwing": ("projectile", "thrown"),
    "dynamics_manipulation": ("manipulation", "pushed"),
    "dynamics_motion_state": ("moving", "static"),
    "property_mass_contrast": ("mass",),
    "property_parameter_identification": ("parameter",),
    "property_bounce_restitution": ("bounc", "restitution"),
}


def test_physics_skill_targets_match_known_keywords() -> None:
    assert set(_SKILL_KEYWORDS) == set(PHYSICS_SKILL_TARGETS)
    assert len(PHYSICS_SKILL_TARGETS) == len(set(PHYSICS_SKILL_TARGETS))


def test_legacy_mechanism_targets_alias_points_at_skill_targets() -> None:
    """The deprecated MECHANISM_TARGETS name resolves to the new vocabulary."""

    assert MECHANISM_TARGETS is PHYSICS_SKILL_TARGETS


@pytest.mark.parametrize("skill", PHYSICS_SKILL_TARGETS)
def test_proposer_stage1_skill_variant_contains_keyword(skill: str) -> None:
    prompt = build_proposer_stage1_prompt(skill)
    lower = prompt.lower()
    for keyword in _SKILL_KEYWORDS[skill]:
        assert keyword in lower, (
            f"skill {skill!r} prompt missing keyword {keyword!r}"
        )
    # prompts must not scaffold <think>.
    assert "<think>" not in prompt and "</think>" not in prompt
    assert "<propose>" in prompt and "</propose>" in prompt
    assert "def build_scene(api):" in prompt
    # Hard constraints surface still present on every variant.
    for method in ("api.add_box", "api.add_sphere", "api.add_spring", "api.add_hinge"):
        assert method in prompt


@pytest.mark.parametrize("skill", PHYSICS_SKILL_TARGETS)
def test_proposer_stage1_skill_variant_has_no_copy_clause(skill: str) -> None:
    prompt = build_proposer_stage1_prompt(skill)
    assert "MUST NOT reuse the example" in prompt


def test_proposer_stage1_skill_variants_avoid_legacy_red_sphere_default() -> None:
    """Drop the red_sphere / blue_box anchor for the vast majority of variants.

    The legacy generic example survives on at most two skill variants (those
    where a sphere+box collision is still the most readable structural hint).
    The remaining variants MUST use different anchors so the Proposer cannot
    collapse onto a single body-name pattern.
    """

    legacy_hits = sum(
        1
        for skill in PHYSICS_SKILL_TARGETS
        if "red_sphere" in build_proposer_stage1_prompt(skill)
        and "blue_box" in build_proposer_stage1_prompt(skill)
    )
    assert legacy_hits <= 2, (
        f"too many skill examples reuse the legacy red_sphere/blue_box "
        f"anchor ({legacy_hits}/{len(PHYSICS_SKILL_TARGETS)})"
    )


def test_proposer_stage1_skill_examples_cover_skill_targets() -> None:
    """The physics-skill examples exercise distinct PhysDSL surfaces.

    - ``dynamics_throwing`` example must launch a body with non-zero vertical
      velocity so it traces a ballistic arc (z-component in set_velocity).
    - ``dynamics_manipulation`` example must contain at least three free
      bodies so contact-driven propagation is visible.
    - ``dynamics_motion_state`` example must contrast friction levels so
      one body clearly stops while another keeps moving.
    """

    throwing = build_proposer_stage1_prompt("dynamics_throwing")
    assert "set_velocity" in throwing
    # Non-zero z component in some set_velocity call: the simplest
    # invariant is that the example contains "linear=(...,...,X)" with
    # X != 0 — encode as "2.5)" appearing in linear payload.
    assert "linear=(2.5, 0.0, 2.5)" in throwing

    manipulation = build_proposer_stage1_prompt("dynamics_manipulation")
    assert manipulation.count("fixed=False") >= 4  # pusher + 3 cubes

    motion_state = build_proposer_stage1_prompt("dynamics_motion_state")
    assert "friction=0.05" in motion_state
    assert "friction=2.0" in motion_state


def test_proposer_stage1_unknown_skill_raises() -> None:
    with pytest.raises(ValueError, match="unknown skill_target"):
        build_proposer_stage1_prompt("not_a_real_skill")


def test_proposer_stage1_mechanism_target_kw_emits_deprecation_warning() -> None:
    """The deprecated ``mechanism_target=`` keyword still works but warns."""

    with pytest.warns(DeprecationWarning, match="mechanism_target"):
        out = build_proposer_stage1_prompt(
            mechanism_target="dynamics_collision",
        )
    assert out == build_proposer_stage1_prompt("dynamics_collision")


def test_proposer_stage1_skill_and_mechanism_conflict_raises() -> None:
    with pytest.raises(ValueError, match="not both"):
        build_proposer_stage1_prompt(
            "dynamics_collision",
            mechanism_target="dynamics_chain_collision",
        )


# ─── Adaptive difficulty ───────────────────────────────────


def test_difficulty_levels_exposed_and_well_formed() -> None:
    """Difficulty vocabulary is published and contains exactly 3 ordered levels."""

    assert DIFFICULTY_LEVELS == ("easy", "medium", "hard")


@pytest.mark.parametrize("level", DIFFICULTY_LEVELS)
def test_build_prompt_includes_difficulty_band_for_each_level(level: str) -> None:
    """Each difficulty level produces a band section with a unique marker."""

    prompt = build_proposer_stage1_prompt(
        "dynamics_collision", difficulty_level=level
    )
    assert "Difficulty band:" in prompt
    # The band text mentions the level itself (e.g. "EASY", "MEDIUM", "HARD").
    assert f"Target difficulty: {level.upper()}" in prompt
    # The legacy generic prompt (no difficulty) MUST NOT mention this marker.
    legacy = build_proposer_stage1_prompt("dynamics_collision")
    assert "Difficulty band:" not in legacy


def test_build_prompt_includes_reference_examples_when_provided() -> None:
    """Reference block renders when at least one summary is provided."""

    refs = (
        "2 spheres, 1 box (n_dynamic=3; features=collision; mass_levels=[1, 4]; friction_levels=[2])",
        "1 capsule (n_dynamic=1; features=no joints; mass_levels=[3]; friction_levels=[2])",
    )
    prompt = build_proposer_stage1_prompt(
        "dynamics_throwing",
        difficulty_level="medium",
        reference_examples=refs,
    )

    assert "Recent reference scenes" in prompt
    assert "vary them, do NOT copy" in prompt
    assert "ref1: 2 spheres, 1 box" in prompt
    assert "ref2: 1 capsule" in prompt


def test_build_prompt_renders_without_references_for_smoke_path() -> None:
    """Empty reference list keeps the reference block out of the prompt."""

    prompt_no_refs = build_proposer_stage1_prompt(
        "dynamics_collision",
        difficulty_level="easy",
    )
    prompt_with_empty_refs = build_proposer_stage1_prompt(
        "dynamics_collision",
        difficulty_level="easy",
        reference_examples=(),
    )

    assert "Recent reference scenes" not in prompt_no_refs
    assert prompt_no_refs == prompt_with_empty_refs
    # Whitespace-only entries are silently dropped (same as no refs).
    assert (
        build_proposer_stage1_prompt(
            "dynamics_collision",
            difficulty_level="easy",
            reference_examples=("   ", ""),
        )
        == prompt_no_refs
    )


def test_difficulty_level_invalid_value_raises() -> None:
    with pytest.raises(ValueError, match="unknown difficulty_level"):
        build_proposer_stage1_prompt(
            "dynamics_collision", difficulty_level="impossible"
        )


# ─── Proposer Stage-2 ────────────────────────────────────────────────


def test_proposer_stage2_advertises_scene_manifest_contract() -> None:
    """the Stage-2 Proposer authors a SCENE_MANIFEST.

    The free-form ``get_vqa`` and the per-question ``VQA_SPEC`` contracts were
    both retired. The role+rules template (``PROPOSER_STAGE2_TEMPLATE``) requires
    ``build_scene`` + a ``SCENE_MANIFEST`` literal and forbids ``get_vqa`` /
    ``VQA_SPEC``.
    """

    tmpl = PROPOSER_STAGE2_TEMPLATE
    assert "# Role" in tmpl
    assert "def build_scene(api):" in tmpl
    assert "SCENE_MANIFEST" in tmpl
    assert "z is UP" in tmpl
    assert "module-level" in tmpl and "SAME indent as `def`" in tmpl
    assert "tuple literals" in tmpl
    assert "## PhysDSL API Reference" in tmpl
    assert "api.add_box(name, pos, size, mass, rgba, friction, restitution, fixed)" in tmpl
    assert "api.set_velocity(body_name, linear=(vx, vy, vz), angular=(wx, wy, wz))" in tmpl
    assert "api.set_rotation(body_name, quaternion=(qw, qx, qy, qz))" in tmpl
    assert "api.add_camera(name, pos, target)" in tmpl
    assert "Invalid aliases that cause runtime failure" in tmpl
    assert "`color=`" in tmpl and "`vel=`" in tmpl and "`rot=`" in tmpl
    assert "`angle=`" in tmpl and "camera `fov=`" in tmpl
    assert "`api.set_friction(...)`" in tmpl
    assert "Write all `api.*` calls first" in tmpl
    assert "Put final numeric manifest values directly in the literal" in tmpl
    assert "long" in tmpl.lower() and "geometry arithmetic" in tmpl
    # The retired contracts are explicitly forbidden.
    assert "Never define `get_vqa` or `VQA_SPEC`" in tmpl
    # Stage-2 still drops the <think> scaffold (Instruct base).
    assert "<think>" not in tmpl and "</think>" not in tmpl


def test_proposer_stage2_lists_executor_hard_limits() -> None:
    """Stage-2 prompts must surface PhysDSL bounds enforced by the executor."""

    prompts = (
        PROPOSER_STAGE2_TEMPLATE,
        build_proposer_scene_template_prompt("motion_lab", include_fewshot=False),
    )
    for prompt in prompts:
        assert "### Executor Hard Limits (DSL Validation)" in prompt
        assert "|x| < 1.5" in prompt
        assert "|y| < 1.5" in prompt
        assert "0 < z < 2.5" in prompt
        assert "x=1.5" in prompt and "z=0" in prompt
        assert "Box size is half-size" in prompt
        assert "in [0.01, 1.0]" in prompt
        assert "mass in [0.01, 100.0]" in prompt
        assert "even when fixed=True" in prompt
        assert "add_plane has no mass argument" in prompt


def test_proposer_stage2_bad_outputs_cover_real_training_failures() -> None:
    tmpl = PROPOSER_STAGE2_TEMPLATE

    assert "Missing </propose>" in tmpl
    assert "missing `SCENE_MANIFEST`" in tmpl
    assert "No free dynamic body" in tmpl
    assert "platform-only scenes" in tmpl
    assert "All dynamic bodies have zero initial motion" in tmpl


def test_proposer_stage2_scene_kind_renders_scene_section() -> None:
    """Per-scene_kind prompts include scene authoring rules + a golden fewshot."""

    prompt = build_proposer_scene_template_prompt("collision_arena")
    assert "## Scene Template: `collision_arena`" in prompt
    assert "## PhysDSL API Reference" in prompt
    assert "## Scene API Focus" in prompt
    assert "## Scene Hard Constraints" in prompt
    assert "### SCENE_MANIFEST Schema" in prompt
    assert "## Validated Golden Example" in prompt
    assert "## Preflight Checks Before <propose>" in prompt
    assert "cold-start only" in prompt
    assert "manifest role wiring" in prompt
    assert "change at least 3" in prompt
    assert "verbatim AST copy" in prompt
    assert "api.add_hinge" in prompt
    assert "### Playbook: collision_arena" not in prompt
    assert "## Physics Modules" not in prompt
    assert "### Topology Skeleton: collision_arena" not in prompt
    # The golden fewshot contains a literal SCENE_MANIFEST.
    assert "SCENE_MANIFEST = {" in prompt
    assert "<propose>" in prompt and "</propose>" in prompt


def test_proposer_stage2_cold_start_selects_one_fewshot_variant() -> None:
    first = build_proposer_scene_template_prompt(
        "throwing_pit_board",
        include_fewshot=True,
        fewshot_index=0,
    )
    second = build_proposer_scene_template_prompt(
        "throwing_pit_board",
        include_fewshot=True,
        fewshot_index=1,
    )
    wrapped = build_proposer_scene_template_prompt(
        "throwing_pit_board",
        include_fewshot=True,
        fewshot_index=10,
    )

    assert "black_left_wall" in first
    assert "gray_left_wall" in second
    assert "gray_left_wall" not in first
    assert "black_left_wall" not in second
    assert first == wrapped
    assert first.count("<propose>\ndef build_scene(api):") == 1
    assert second.count("<propose>\ndef build_scene(api):") == 1
    assert '"pits": {' not in first
    assert '"pits": {' not in second

    main = build_proposer_scene_template_prompt(
        "throwing_pit_board",
        include_fewshot=False,
        fewshot_index=1,
    )
    assert "Validated Golden Example" not in main
    assert "gray_left_wall" not in main


def test_proposer_stage2_throwing_scene_kind_is_multi_family() -> None:
    prompt = build_proposer_scene_template_prompt("throwing_pit_board")
    assert "## Scene Template: `throwing_pit_board`" in prompt
    # The throwing template auto-generates several families from one scene.
    assert "throwing_final_destination" in prompt
    assert "counterfactual_destination" in prompt


def test_proposer_stage2_main_training_drops_executable_fewshot() -> None:
    """include_fewshot=False removes the copyable program but keeps the contract.

    The decisive anti-copy lever: after cold start the prompt must NOT contain
    the verbatim golden body names or a copyable skeleton program, yet it must
    still carry manifest schema, scene-specific API references, and one
    integrated Golden Wiring Card distilled from the verified fewshot.
    """

    cold = build_proposer_scene_template_prompt("collision_arena")
    main = build_proposer_scene_template_prompt(
        "collision_arena",
        include_fewshot=False,
        reference_examples=(
            "3 boxes, 1 sphere (n_dynamic=4; features=collision, hinge)",
        ),
    )
    # Cold start carries the executable fewshot program; main training drops it.
    # Key on a golden-body-specific token (the manifest skeleton in the schema
    # doc and the `def build_scene` mention in the role rules appear in both,
    # so they cannot distinguish the modes).
    assert "swing_striker" in cold
    assert "Validated Golden Example" in cold
    assert "swing_striker" not in main
    assert "gray_anchor" not in main
    assert "Validated Golden Example" not in main
    # Contract, hard constraints, and wiring-card scaffolding survive without
    # the old split layers.
    assert "### SCENE_MANIFEST Schema" in main
    assert "## PhysDSL API Reference" in main
    assert "## Scene API Focus" in main
    assert "## Scene Hard Constraints" in main
    assert "## Golden Wiring Card: collision_arena" in main
    assert "## Preflight Checks Before <propose>" in main
    assert "### Apparatus Wiring" in main
    assert "### Manifest Wiring" in main
    assert "### Syntax Anchors" in main
    assert "### Validity Checks" in main
    assert "### Playbook: collision_arena" not in main
    assert "## Physics Modules" not in main
    assert "Syntax idiom only - not a complete scene" not in main
    assert "### Topology Skeleton: collision_arena" not in main
    assert "## Design Target: collision_arena" not in main
    assert "## Success Invariants" not in main
    assert "### Topology checklist" not in main
    assert "Build order (invent all names, coordinates, sizes, and parameters)" not in main
    assert "Code fragment (inside build_scene" not in main
    assert "striker_box" not in main
    assert "hinge_arm" not in main
    assert "api.add_hinge" in main
    assert "api.set_velocity" in main
    # Reference summaries injected with the "vary, do NOT copy" framing.
    assert "## Recent Successful Scenes (vary them, do NOT copy)" in main
    assert "n_dynamic=4" in main


def test_proposer_stage2_fewshot_hybrid_appends_parameter_bands() -> None:
    """recipe: fewshot + reference summaries coexist as parameter-band nudges."""

    prompt = build_proposer_scene_template_prompt(
        "throwing_pit_board",
        include_fewshot=True,
        fewshot_index=0,
        reference_examples=(
            "3 spheres, 2 boxes (n_dynamic=5; mass_levels=[1,2]; friction_levels=[0.2])",
            "2 spheres, 1 box (n_dynamic=3; mass_levels=[1]; friction_levels=[0.5])",
        ),
    )
    assert "## Validated Golden Example" in prompt
    assert "## Recent OK Parameter Bands" in prompt
    assert "mass_levels=[1,2]" in prompt
    assert "Do NOT paste the golden program verbatim" in prompt
    assert "## Recent Successful Scenes (vary them, do NOT copy)" not in prompt


def test_proposer_stage2_fewshot_without_refs_omits_parameter_bands() -> None:
    prompt = build_proposer_scene_template_prompt(
        "throwing_pit_board",
        include_fewshot=True,
        fewshot_index=0,
        reference_examples=(),
    )
    assert "## Validated Golden Example" in prompt
    assert "## Recent OK Parameter Bands" not in prompt


def test_proposer_stage2_main_training_has_wiring_card_not_golden() -> None:
    main = build_proposer_scene_template_prompt(
        "collision_arena",
        include_fewshot=False,
    )

    assert "## Golden Wiring Card: collision_arena" in main
    assert "### Topology Skeleton: collision_arena" not in main
    assert "Build order (invent all names, coordinates, sizes, and parameters)" not in main
    assert "<propose>\ndef build_scene(api):" not in main
    assert "Code fragment (inside build_scene" not in main
    assert "swing_striker" not in main
    assert "gray_anchor" not in main


def test_proposer_stage2_main_training_restores_exact_api_contract() -> None:
    main = build_proposer_scene_template_prompt(
        "motion_lab",
        include_fewshot=False,
    )

    assert "## PhysDSL API Reference" in main
    assert "Use these exact parameter names" in main
    assert "api.add_sphere(name, pos, radius, mass, rgba, friction, restitution, fixed)" in main
    assert "api.set_velocity(body_name, linear=(vx, vy, vz), angular=(wx, wy, wz))" in main
    assert "api.set_rotation(body_name, quaternion=(qw, qx, qy, qz))" in main
    assert "api.add_hinge(name, parent, child, pos, axis=(ax, ay, az), limit=None)" in main
    assert "api.add_camera(name, pos, target)" in main
    assert "Invalid aliases that cause runtime failure" in main
    for bad_alias in (
        "`color=`",
        "`vel=`",
        "`rot=`",
        "`angle=`",
        "camera `fov=`",
        "camera `duration=`",
        "`api.set_friction(...)`",
    ):
        assert bad_alias in main


def test_proposer_stage2_main_training_has_design_cards_without_code() -> None:
    throwing = build_proposer_scene_template_prompt(
        "throwing_pit_board",
        question_family="counterfactual_destination",
        include_fewshot=False,
    )
    assert "## Required Family: `counterfactual_destination`" in throwing
    assert "must actually produce at least one executor VQA for this family" in throwing
    assert "## Golden Wiring Card: throwing_pit_board" in throwing
    assert "executor derives left/middle/right regions" in throwing
    assert "removing a wall can change a settled pit" in throwing
    assert "## Scene Hard Constraints" in throwing
    assert "api.add_box(\"left_platform\"" not in throwing
    assert "### slanted_deflector" not in throwing
    assert "api.set_rotation" in throwing

    motion = build_proposer_scene_template_prompt(
        "motion_lab",
        question_family="friction_or_restitution_identification",
        include_fewshot=False,
    )
    assert "## Required Family: `friction_or_restitution_identification`" in motion
    assert "## Golden Wiring Card: motion_lab" in motion
    assert "## Scene Hard Constraints" in motion
    assert "`compare_pair`" in motion
    assert "rebound apex" in motion
    assert "### friction_compare_pair" not in motion
    assert "### restitution_compare_pair" not in motion
    assert "green_ball" not in motion


def test_proposer_stage2_main_training_deanchors_numeric_examples() -> None:
    throwing = build_proposer_scene_template_prompt(
        "throwing_pit_board",
        include_fewshot=False,
    )
    motion = build_proposer_scene_template_prompt(
        "motion_lab",
        include_fewshot=False,
    )

    for prompt in (throwing, motion):
        assert "Prompt literal numbers are calibration hints, not defaults" in prompt
        assert "linear=(0.15, 0.0, -0.20)" not in prompt
        assert "api.set_gravity((0.0, 0.0, -1.8))" not in prompt
        assert "api.set_gravity((0.0, 0.0, -9.8))" not in prompt
        assert "pos=(0.1, 0.0, 0.5)" not in prompt


def test_proposer_stage2_main_training_reduces_grounding_duplication() -> None:
    family_prompt = build_proposer_scene_template_prompt(
        "motion_lab",
        question_family="motion_state_change",
        include_fewshot=False,
    )

    assert family_prompt.count("Example generated question") == 1
    assert family_prompt.count("GT source") == 1
    assert "It first accelerates and then decelerates" in family_prompt
    assert "crossover-surpass" in family_prompt
    assert "### Example grounding" not in family_prompt
    assert "### Good scene pattern" not in family_prompt


def test_stage2_scene_templates_add_compact_failure_hardening() -> None:
    throwing = build_proposer_scene_template_prompt(
        "throwing_pit_board",
        include_fewshot=False,
    )
    assert "## Golden Wiring Card: throwing_pit_board" in throwing
    assert "### Syntax Anchors" in throwing
    assert ">=3 fixed low platform boxes" in throwing
    assert '"platforms": [<visible fixed box names>]' in throwing
    assert '"pits": {' not in throwing
    assert 'SCENE_MANIFEST["pits"]' not in throwing
    assert "Do not include a separate pit interval map" in throwing
    assert "executor derives left/middle/right regions" in throwing
    assert "valid examples are ball_0 / ball_1 / ball_2" in throwing
    assert "invalid examples are ball_red" in throwing
    assert "`ball_red`, `ball_blue`, `green_ball`, or `red_ball`" in throwing
    assert "avoid all-positive platform chains such as 0.5, 1.5, 2.5" in throwing
    assert "golden-style centered platform band" in throwing
    assert "left at negative x" in throwing
    assert "middle near zero" in throwing
    assert "half_x values around 0.25-0.45" in throwing
    assert "Throwing is z-axis falling" in throwing
    assert "descend in negative z" in throwing
    assert "wide negative-y board camera" in throwing
    assert "cam0 pos approx" in throwing
    assert "Vary the numbers" in throwing
    assert "negative-y framing" in throwing

    collision = build_proposer_scene_template_prompt(
        "collision_arena",
        include_fewshot=False,
    )
    assert "## Golden Wiring Card: collision_arena" in collision
    assert "### Syntax Anchors" in collision
    assert "non-zero linear or angular motion" in collision
    assert "Do not leave all dynamic bodies sleeping" in collision
    assert "`trigger`" in collision
    assert "`target`" in collision
    assert "`movers`" in collision
    assert "dominant" in collision
    assert "delta-speed" in collision
    assert "staggered y offsets or clear x gaps" in collision
    assert "one dense same-line" in collision
    assert "x=2.0/2.5/3.0" in collision
    assert "First-submit geometry should already be non-overlapping" in collision
    assert "wide negative-y camera" in collision
    assert "do not zoom onto only the first contact" in collision

    motion = build_proposer_scene_template_prompt(
        "motion_lab",
        include_fewshot=False,
    )
    assert "## Golden Wiring Card: motion_lab" in motion
    assert "### Syntax Anchors" in motion
    assert "Matched comparison bodies never start at the same position" in motion
    assert "`motion_target`" in motion
    assert "`compare_pair`" in motion
    assert "Every `size` component is a half-extent and must be <= 1.0" in motion
    assert "do not write size=(2.0, ...)" in motion
    assert "Use nonzero identical initial velocity unless" in motion
    assert "not \"both stopped\"" in motion
    assert "speed-profile and pairwise-speed questions" in motion
    assert "crossover speed relation" in motion
    assert "visible final-speed /" in motion
    assert "travel-distance contrast" in motion
    assert "Avoid 90-degree / vertical ramps" in motion
    assert "wide negative-y camera" in motion
    assert "hide the comparison lane" in motion


def test_proposer_stage2_main_training_has_scene_preflight_checks() -> None:
    prompts = {
        scene_kind: build_proposer_scene_template_prompt(
            scene_kind,
            include_fewshot=False,
        )
        for scene_kind in ("throwing_pit_board", "collision_arena", "motion_lab")
    }

    for prompt in prompts.values():
        assert "## Preflight Checks Before <propose>" in prompt
        assert "The required family can produce at least one VQA" in prompt
        assert prompt.index("## Preflight Checks Before <propose>") < prompt.index(
            "## Output Format"
        )

    assert "No ball/deflector/platform overlaps at t=0" in prompts["throwing_pit_board"]
    assert "centered left/middle/right platform board" in prompts["throwing_pit_board"]
    assert "Throwing is z-axis falling" in prompts["throwing_pit_board"]
    assert "y is only depth/lane spacing" in prompts["throwing_pit_board"]
    assert "wide negative-y board shot" in prompts["throwing_pit_board"]
    assert "no x=2.0/2.5/3.0 body chains" in prompts["collision_arena"]
    assert "wide negative-y shot" in prompts["collision_arena"]
    assert "Every `size` component is a half-extent <= 1.0" in prompts["motion_lab"]
    assert "visible final-speed or travel-distance contrast" in prompts["motion_lab"]
    assert "wide negative-y shot" in prompts["motion_lab"]


# ─── Solver templates — fill round-trips ─────────────────────────────


def test_forward_template_fill_round_trip() -> None:
    filled = FORWARD_TEMPLATE.format(
        params_block="- ball: mass_level=3 (1.0 kg), friction_level=2 (0.2)",
        initial_block="- ball: pos=(0.0, 0.0, 1.0) m, initial_velocity=(0.0, 0.0, 0.0) m/s",
    )
    assert "{params_block}" not in filled
    assert "{initial_block}" not in filled
    # Solver templates require <answer> but no longer scaffold <think>.
    assert "<think>" not in filled and "</think>" not in filled
    assert "<answer>" in filled and "</answer>" in filled


def test_inverse_template_fill_round_trip() -> None:
    filled = INVERSE_TEMPLATE.format(
        params_block="- ball: mass_level=3, friction_level=2",
        final_block="- ball: final_grid=[8, 8, 0], status='stable'",
    )
    assert "{params_block}" not in filled
    assert "{final_block}" not in filled
    assert "initial_vel_grid" in filled


def test_parameter_template_fill_round_trip() -> None:
    filled = PARAMETER_TEMPLATE.format(
        initial_block="- ball: pos=(0, 0, 1), initial_velocity=(0, 0, 0)",
        final_block="- ball: final_grid=[8, 8, 0], status='stable'",
        mass_levels=list(MASS_LEVELS),
        friction_levels=list(FRICTION_LEVELS),
    )
    assert "{initial_block}" not in filled
    assert "{final_block}" not in filled
    assert "mass_level" in filled
    assert "friction_level" in filled


# ─── VQA template ────────────────────────────────────────────────────


def test_vqa_template_has_four_option_slots() -> None:
    tmpl = VQA_TEMPLATE
    for placeholder in ("{question}", "{opt_a}", "{opt_b}", "{opt_c}", "{opt_d}"):
        assert placeholder in tmpl, f"missing placeholder {placeholder}"
    # The MC layout is standard ABCD.
    assert re.search(r"A\.\s+\{opt_a\}", tmpl)
    assert re.search(r"B\.\s+\{opt_b\}", tmpl)
    assert re.search(r"C\.\s+\{opt_c\}", tmpl)
    assert re.search(r"D\.\s+\{opt_d\}", tmpl)


def test_vqa_template_fill_round_trip() -> None:
    filled = VQA_TRAINING_TEMPLATE.format(
        question="Which ball hits the ground first?",
        opt_a="The heavy ball",
        opt_b="The light ball",
        opt_c="They land simultaneously",
        opt_d="Neither ball falls",
    )
    assert "{question}" not in filled
    assert "{opt_a}" not in filled
    # VQA template requires <answer> but no longer scaffolds <think>.
    assert "<think>" not in filled and "</think>" not in filled
    assert "<answer>" in filled and "</answer>" in filled


def test_vqa_training_template_uses_short_answer() -> None:
    tmpl = VQA_TRAINING_TEMPLATE
    assert MC_SHORT_ANSWER_INSTRUCTION in tmpl
    assert "3-5 concise sentences" not in tmpl
    assert "about 120 words" not in tmpl
    assert "Answers without the <answer> block are invalid" in tmpl
    assert "<think>" not in tmpl and "</think>" not in tmpl
    assert "<answer>" in tmpl and "</answer>" in tmpl
    assert "simulated scene" not in tmpl
    assert "Task:" not in tmpl
    assert VQA_TEMPLATE is VQA_TRAINING_TEMPLATE


# ─── Output-format invariant across every Solver template ───────────


@pytest.mark.parametrize(
    "tmpl",
    [
        FORWARD_TEMPLATE,
        INVERSE_TEMPLATE,
        PARAMETER_TEMPLATE,
        VQA_TRAINING_TEMPLATE,
    ],
)
def test_solver_templates_end_with_answer_scaffold(tmpl: str) -> None:
    """Every Solver template must end with <answer>…</answer>."""

    tail = tmpl.rstrip().splitlines()[-5:]
    tail_text = "\n".join(tail)
    assert "<answer>" in tail_text
    assert "</answer>" in tail_text
