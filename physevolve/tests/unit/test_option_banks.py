"""Unit tests for option-layout sampling and answer binding."""

from __future__ import annotations

import json

from physevolve.skills.scene_templates import GeneratedVQA
from physevolve.skills.scene_templates import option_banks
from physevolve.skills.scene_templates.option_banks import (
    _YESNO_ALIASES,
    apply_option_layout,
    bind_correct_index,
    sample_option_layout,
)

# Merged fallback and layout-bank counts, locked so a vendoring
# regression cannot silently change weighted sampling.
_EXPECTED_LAYOUT_COUNTS = {
    "collision_final_relation": 411,
    "collision_final_state": 140,
    "collision_first_contact": 2,
    "counterfactual_destination": 28,
    "inverse_intervention": 51,
    "motion_state_change": 2,
    "throwing_current_run_judgment": 30,
    "throwing_final_destination": 52,
}


def test_sample_layout_deterministic() -> None:
    a = sample_option_layout(
        "counterfactual_destination",
        phrasing="counterfactual",
        seed_key=("throwing_pit_board:counterfactual_destination", "counterfactual", {}),
        salt="scene-1",
    )
    b = sample_option_layout(
        "counterfactual_destination",
        phrasing="counterfactual",
        seed_key=("throwing_pit_board:counterfactual_destination", "counterfactual", {}),
        salt="scene-1",
    )
    assert a == b
    assert len(a) == 4


def test_apply_option_layout_remaps_correct_idx() -> None:
    gen = GeneratedVQA(
        question="Which pit?",
        options=["The left pit", "The right pit", "The center pit", "None of the above"],
        correct_idx=1,
        family="throwing_final_destination",
        phrasing="forward",
        spec={"answer_semantic": "right", "target": "right"},
    )
    result = apply_option_layout(
        gen,
        scene_kind="throwing_pit_board",
        salt="test-salt",
    )
    assert result is not None
    options, correct_idx = result
    assert len(options) == 4
    assert 0 <= correct_idx < 4
    assert "right" in options[correct_idx].lower()


def test_collision_pose_distractor_aliases_bind_gt_only() -> None:
    """Broken/collapsed are distractor aliases; GT remains a canonical pose."""

    gen = GeneratedVQA(
        question="Final pose?",
        options=[
            "Standing upright",
            "Leaning",
            "Lying horizontally",
            "Floating",
        ],
        correct_idx=0,
        family="collision_final_state",
        phrasing="forward",
        spec={"answer_semantic": "standing_upright"},
    )
    result = apply_option_layout(
        gen,
        scene_kind="collision_arena",
        salt="pose-alias",
    )
    assert result is not None
    options, correct_idx = result
    assert "standing" in options[correct_idx].lower() or "upright" in options[
        correct_idx
    ].lower()


def test_throwing_neither_escape_binds() -> None:
    gen = GeneratedVQA(
        question="Where does the ball settle?",
        options=["The left pit", "The right pit", "Neither pit", "Uncertain"],
        correct_idx=2,
        family="throwing_final_destination",
        phrasing="forward",
        spec={"answer_semantic": "neither", "target": "neither"},
    )
    result = apply_option_layout(
        gen,
        scene_kind="throwing_pit_board",
        salt="escape-neither",
    )
    assert result is not None
    _options, correct_idx = result
    assert 0 <= correct_idx < 4


def test_motion_speed_profile_layout_available() -> None:
    layout = sample_option_layout(
        "motion_state_change",
        phrasing="speed_profile",
        seed_key=("motion_lab:motion_state_change", "speed_profile", {}),
        salt="spd",
    )
    assert layout is not None
    assert len(layout) == 4


def test_yes_no_judgment_keeps_template_options() -> None:
    """Layout bank must not replace Yes/No with destination layouts."""

    template_opts = ["Yes", "No", "Can not answer", "Uncertain"]
    gen = GeneratedVQA(
        question="Will the leftmost launched ball eventually drop into the left pit?",
        options=list(template_opts),
        correct_idx=1,
        family="throwing_current_run_judgment",
        phrasing="yes_no",
        spec={"answer_semantic": "no", "target": "left"},
    )
    result = apply_option_layout(
        gen,
        scene_kind="throwing_pit_board",
        salt="judgment-pollute",
    )
    assert result is not None
    options, correct_idx = result
    assert options == template_opts
    assert correct_idx == 1
    assert options[correct_idx].lower() == "no"


def test_yes_no_phrasing_variants_keep_template_options() -> None:
    """All binary phrasing variants must bypass bank remapping."""

    template_opts = ["Yes", "No", "Can not answer", "Uncertain"]
    cases = (
        (
            "throwing_current_run_judgment",
            "wrong_pit",
            "Will the ball finally drop into a different pit?",
            "throwing_pit_board",
        ),
        (
            "collision_final_relation",
            "binary_collide",
            "Will the two objects collide?",
            "collision_arena",
        ),
    )
    for family, phrasing, question, scene_kind in cases:
        gen = GeneratedVQA(
            question=question,
            options=list(template_opts),
            correct_idx=2,
            family=family,
            phrasing=phrasing,
            # Exercise the phrasing guard independently of the yes/no
            # answer-semantic guard covered above.
            spec={"answer_semantic": "uncertain"},
        )
        result = apply_option_layout(gen, scene_kind=scene_kind, salt=phrasing)
        assert result == (template_opts, 2)


def test_judgment_bank_excludes_destination_layouts() -> None:
    """The loaded judgment bank remains binary even if bypass regresses."""

    layouts = option_banks._load_banks()["throwing_current_run_judgment"]
    assert layouts
    for layout in layouts:
        normalized = {option.lower().strip().rstrip(".") for option in layout.options}
        assert {"yes", "no"} <= normalized
        assert all("pit" not in option for option in normalized)


def test_inverse_layout_binds_after_wall_placeholders_are_filled() -> None:
    gen = GeneratedVQA(
        question="What can we do to make the ball reach the left pit?",
        options=[
            "Remove the blue floating wall and other balls",
            "Remove the red floating wall and other balls",
            "Remove both floating walls and other balls",
            "None of the above",
        ],
        correct_idx=0,
        family="inverse_intervention",
        phrasing="inverse",
        spec={
            "answer_semantic": "remove_a",
            "wall_a_phrase": "blue floating wall",
            "wall_b_phrase": "red floating wall",
        },
    )
    result = apply_option_layout(
        gen,
        scene_kind="throwing_pit_board",
        salt="inverse-wall-context",
    )
    assert result is not None
    options, correct_idx = result
    assert options[correct_idx] == "Remove the blue floating wall and other balls"
    assert all("{" not in option and "}" not in option for option in options)


def test_short_alias_no_does_not_bind_into_pit() -> None:
    """\"no\" must not substring-match \"into\" / \"not\" / \"cannot\" pit text."""

    layout = [
        "into the left pit",
        "into the right pit",
        "will not drop into any pit",
        "uncertain",
    ]
    assert (
        bind_correct_index(
            layout,
            "no",
            alias_table=_YESNO_ALIASES,
            context={},
        )
        is None
    )
    yn_layout = ["Yes", "No", "Can not answer", "Uncertain"]
    assert (
        bind_correct_index(
            yn_layout,
            "no",
            alias_table=_YESNO_ALIASES,
            context={},
        )
        == 1
    )


def test_layout_bank_counts_match_snapshot() -> None:
    banks = option_banks._load_banks()
    counts = {family: len(layouts) for family, layouts in banks.items()}
    assert counts == _EXPECTED_LAYOUT_COUNTS


def test_option_bank_contains_only_layout_inputs() -> None:
    jsonl_path = option_banks._LAYOUT_BANK_JSONL
    assert jsonl_path.is_file()
    assert jsonl_path.parts[-3:] == ("data", "task_bank", "option_layout_bank.jsonl")
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert set(row) == {"family", "options", "weight"}
        assert len(row["options"]) == 4
        assert isinstance(row["weight"], int) and row["weight"] > 0
