"""Tests for tolerant structural tag parsing."""

from __future__ import annotations

from physevolve.inference.parsing import (
    extract_answer_block,
    extract_json_from_answer,
    extract_mc_letter,
    extract_propose_block,
)


def test_extract_answer_accepts_missing_stop_closing_tag() -> None:
    text = '<think>reason</think><answer>{"ball": {"grid": [1, 2, 3]}}'

    assert extract_answer_block(text) == '{"ball": {"grid": [1, 2, 3]}}'
    assert extract_json_from_answer(text) == {"ball": {"grid": [1, 2, 3]}}


def test_extract_propose_accepts_missing_stop_closing_tag() -> None:
    code = "def build_scene(api):\n    api.set_duration(0.5)"
    text = f"<think>reason</think><propose>{code}"

    assert extract_propose_block(text) == code


def test_closed_tag_wins_over_later_text() -> None:
    text = "<answer>{\"x\": 1}</answer> trailing text"

    assert extract_answer_block(text) == '{"x": 1}'


# ─── Instruct-style outputs (no <think> wrapper) ─────────────────


def test_extract_propose_handles_instruct_plain_text_prefix() -> None:
    # Instruct-base outputs do not emit <think>. Free-form rationale sits
    # in front of <propose> as ordinary prose; the extractor must still
    # find the propose block.
    code = "def build_scene(api):\n    api.set_duration(0.5)"
    text = f"Plan: drop a sphere onto a plate.\n<propose>{code}</propose>"

    assert extract_propose_block(text) == code


def test_extract_answer_handles_instruct_plain_text_prefix() -> None:
    text = (
        "Looking at the video, the ball stops near the centre.\n"
        '<answer>{"ball_a": {"grid": [8, 8, 1], "status": "stable"}}</answer>'
    )

    payload = extract_json_from_answer(text)
    assert payload == {"ball_a": {"grid": [8, 8, 1], "status": "stable"}}


def test_extract_mc_letter_handles_markdown_emphasis_around_letter() -> None:
    # Instruct VQA replies commonly look like "The answer is **B**.".
    text = "<answer>The answer is **B**.</answer>"

    assert extract_mc_letter(text) == "B"
