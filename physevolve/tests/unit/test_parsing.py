"""Tests for :mod:`physevolve.inference.parsing`.

Coverage: each extractor's happy path + every failure mode that returns
``None`` (never raises), plus json5 trailing-comma tolerance.
"""

from __future__ import annotations

import pytest

from physevolve.inference.parsing import (
    extract_answer_block,
    extract_json_from_answer,
    extract_mc_letter,
    extract_propose_block,
)


# ─── extract_answer_block ────────────────────────────────────────────


def test_answer_happy_path() -> None:
    text = "Some preamble\n<answer>{\"grid\": [1, 2, 3]}</answer>\n"
    assert extract_answer_block(text) == '{"grid": [1, 2, 3]}'


def test_answer_case_insensitive() -> None:
    text = "<Answer>A</Answer>"
    assert extract_answer_block(text) == "A"


def test_answer_multiline_dotall() -> None:
    text = "<answer>\nline1\nline2\n</answer>"
    assert extract_answer_block(text) == "line1\nline2"


def test_answer_missing_returns_none() -> None:
    assert extract_answer_block("no tags here") is None


def test_answer_unclosed_tag_returns_none() -> None:
    assert extract_answer_block("<answer>oops") is None


def test_answer_non_string_returns_none() -> None:
    assert extract_answer_block(None) is None  # type: ignore[arg-type]
    assert extract_answer_block(123) is None  # type: ignore[arg-type]


def test_answer_first_occurrence_wins() -> None:
    text = "<answer>first</answer>...<answer>second</answer>"
    assert extract_answer_block(text) == "first"


# ─── extract_propose_block ───────────────────────────────────────────


def test_propose_happy_path() -> None:
    text = "<think>thinking</think><propose>def build_scene(api): pass</propose>"
    assert extract_propose_block(text) == "def build_scene(api): pass"


def test_propose_with_nested_code_ok() -> None:
    text = """\
<propose>
def build_scene(api):
    api.add_sphere("ball", pos=(0, 0, 1), radius=0.1)

def get_vqa(sim_state):
    return "?", ["a", "b", "c", "d"], 0
</propose>"""
    body = extract_propose_block(text)
    assert body is not None
    assert "def build_scene(api):" in body
    assert "def get_vqa(sim_state):" in body


def test_propose_missing_returns_none() -> None:
    assert extract_propose_block("<answer>not propose</answer>") is None


# ─── extract_json_from_answer ────────────────────────────────────────


def test_json_happy_path() -> None:
    text = '<answer>{"body_a": {"grid": [1, 2, 3], "status": "stable"}}</answer>'
    result = extract_json_from_answer(text)
    assert result == {"body_a": {"grid": [1, 2, 3], "status": "stable"}}


def test_json_trailing_comma_tolerated() -> None:
    text = '<answer>{"a": 1, "b": 2,}</answer>'
    result = extract_json_from_answer(text)
    assert result == {"a": 1, "b": 2}


def test_json_no_answer_returns_none() -> None:
    assert extract_json_from_answer("no tags") is None


def test_json_malformed_returns_none() -> None:
    assert extract_json_from_answer("<answer>{not json at all]}</answer>") is None


def test_json_list_rejected() -> None:
    # The Stage-1 prediction schemas are all dicts; lists are invalid.
    assert extract_json_from_answer("<answer>[1, 2, 3]</answer>") is None


def test_json_string_rejected() -> None:
    assert extract_json_from_answer("<answer>\"hello\"</answer>") is None


# ─── extract_mc_letter ───────────────────────────────────────────────


@pytest.mark.parametrize("letter", ["A", "B", "C", "D"])
def test_mc_letter_simple(letter: str) -> None:
    text = f"<answer>{letter}</answer>"
    assert extract_mc_letter(text) == letter


def test_mc_letter_lowercase_normalized() -> None:
    assert extract_mc_letter("<answer>b</answer>") == "B"


def test_mc_letter_inside_sentence() -> None:
    text = "<answer>The correct answer is A because momentum.</answer>"
    assert extract_mc_letter(text) == "A"


def test_mc_letter_no_answer_returns_none() -> None:
    assert extract_mc_letter("just text, no tags") is None


def test_mc_letter_junk_returns_none() -> None:
    # No stand-alone A/B/C/D word in the answer body.
    text = "<answer>xxxxxx</answer>"
    assert extract_mc_letter(text) is None


# ─── Qwen3-VL Instruct-style outputs (no <think> wrapper) ──────────────────


def test_propose_handles_instruct_no_think_with_plain_text_prefix() -> None:
    code = 'def build_scene(api):\n    api.set_duration(0.5)'
    text = f"I will set up a single sphere drop.\n<propose>{code}</propose>"

    assert extract_propose_block(text) == code


def test_answer_handles_instruct_no_think_with_plain_text_prefix() -> None:
    text = (
        "Reasoning: the ball ends near the centre.\n"
        '<answer>{"ball_a": {"grid": [8, 8, 1], "status": "stable"}}</answer>'
    )

    assert extract_json_from_answer(text) == {
        "ball_a": {"grid": [8, 8, 1], "status": "stable"},
    }


def test_mc_letter_handles_markdown_bold_letter() -> None:
    # Common Instruct VQA reply style.
    text = "<answer>The answer is **B**.</answer>"
    assert extract_mc_letter(text) == "B"
