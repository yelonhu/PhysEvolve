"""Tests for the official PhysBench Qwen adapter glue."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation.physbench import qwen_official_adapter as adapter
from evaluation.physbench.third_party.official_eval.eval_utils.caculate_core import (
    answer_true,
    calculate_accuracy,
)


def test_strict_parser_prefers_answer_block() -> None:
    raw = "<think>A misleading thought about option B.</think><answer>\nC\n</answer>"
    assert adapter.parse_physbench_mc_answer(raw) == "C"


def test_strict_parser_does_not_search_think_prose() -> None:
    raw = "<think>A misleading thought.</think>"
    assert adapter.parse_physbench_mc_answer(raw) is None


def test_strict_parser_accepts_native_qwen_thinking_final_letter() -> None:
    raw = (
        "<think>\nOption A appears in the reasoning, but the model decides.\n"
        "</think>\n\nD<|im_end|>"
    )
    assert adapter.parse_physbench_mc_answer(raw) == "D"


def test_strict_parser_uses_last_think_block() -> None:
    raw = "<think>first</think><think>mentions A</think>\n\nC<|im_end|>"
    assert adapter.parse_physbench_mc_answer(raw) == "C"


def test_strict_parser_accepts_bare_instruct_letter() -> None:
    # Leading-letter shapes (unchanged contract).
    assert adapter.parse_physbench_mc_answer("D. The answer") == "D"
    assert adapter.parse_physbench_mc_answer("(B)") == "B"
    assert adapter.parse_physbench_mc_answer("A<|im_end|>") == "A"


def test_strict_parser_d95_mid_prose_fallback() -> None:
    # the leading-letter parser misses prose forms.
    # The embedded-letter fallback should recover them; this is the
    # primary motivation for the fallback branch.
    assert (
        adapter.parse_physbench_mc_answer("Answer: D")
        == "D"
    )
    assert (
        adapter.parse_physbench_mc_answer(
            "The answer is **B**.<|im_end|>"
        )
        == "B"
    )
    assert (
        adapter.parse_physbench_mc_answer(
            "Looking at the video, the answer is C because gravity."
        )
        == "C"
    )


def test_strict_parser_fallback_inside_answer_block() -> None:
    # <answer> body without a leading letter should still resolve via the
    # embedded fallback (Instruct may write "Final answer: B." inside the
    # block when nudged).
    raw = "<answer>Final answer: B.</answer>"
    assert adapter.parse_physbench_mc_answer(raw) == "B"


def test_strict_parser_fallback_does_not_leak_into_think_block() -> None:
    # Invariant from the original parser: we never extract a letter that
    # only appears inside <think>. The Instruct fallback must respect this.
    # When </think> exists, the post-think branch runs against the tail
    # only and returns before falling through to raw_text, so the "A" in
    # the reasoning prose is invisible.
    raw = "<think>Option A looks tempting but is wrong.</think>"
    assert adapter.parse_physbench_mc_answer(raw) is None


def test_strict_parser_returns_none_for_letter_free_output() -> None:
    raw = "I am not sure which option fits."
    assert adapter.parse_physbench_mc_answer(raw) is None


def test_qwen_content_preserves_official_prompt_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_image(path: Path) -> np.ndarray:
        return np.full((2, 2, 3), len(path.name), dtype=np.uint8)

    def fake_video(path: Path, n_frames: int = 8) -> np.ndarray:
        return np.full((n_frames, 2, 2, 3), len(path.name), dtype=np.uint8)

    monkeypatch.setattr(adapter, "load_image", fake_image)
    monkeypatch.setattr(adapter, "load_video", fake_video)

    prompt = "Look <video> choose A <image> B <image>."
    content = adapter._build_qwen_content(
        prompt=prompt,
        visual_paths=[tmp_path / "v.mp4", tmp_path / "a.png", tmp_path / "bb.png"],
        n_video_frames=4,
    )
    assert [entry["type"] for entry in content] == [
        "text",
        "video",
        "text",
        "image",
        "text",
        "image",
        "text",
    ]
    assert content[1]["video"].shape[0] == 4


def test_official_scorer_semantics(tmp_path: Path) -> None:
    preds = tmp_path / "preds.json"
    gold = tmp_path / "gold.json"
    preds.write_text(
        json.dumps(
            [
                {"idx": 1, "answer": "A"},
                {"idx": 2, "answer": "Answer: B"},
                {"idx": 3, "answer": None},
            ]
        ),
        encoding="utf-8",
    )
    gold.write_text(
        json.dumps(
            [
                {
                    "idx": 1,
                    "answer": "A",
                    "task_type": "property",
                    "sub_type": "mass",
                    "ability_type": "identify",
                },
                {
                    "idx": 2,
                    "answer": "B",
                    "task_type": "scene",
                    "sub_type": "light",
                    "ability_type": "perception",
                },
                {
                    "idx": 3,
                    "answer": "C",
                    "task_type": "dynamics",
                    "sub_type": "fluid",
                    "ability_type": "reasoning",
                },
            ]
        ),
        encoding="utf-8",
    )

    assert answer_true({"answer": "A suffix"}, {"answer": "A"})
    assert not answer_true({"answer": "Answer: B"}, {"answer": "B"})
    acc = calculate_accuracy(gold, preds)
    assert acc["overall_correct"] == 1
    # Official denominator is len(predictions), even if some answers are None.
    assert acc["overall_total"] == 3


def test_adapter_selects_family_backend_without_loading() -> None:
    """InternVL path must not construct HFBackend or load weights."""

    from evaluation.common.family_generate import InternVLFamilyBackend
    from physevolve.inference.hf_backend import HFBackend

    internvl = adapter.QwenPhysBenchAdapter(
        "models/InternVL2_5-8B",
        model_family="internvl",
    )
    assert internvl.model_family == "internvl"
    assert isinstance(internvl.backend, InternVLFamilyBackend)
    assert internvl.backend._model is None

    qwen = adapter.QwenPhysBenchAdapter(
        "Qwen/Qwen3-VL-8B-Instruct",
    )
    assert qwen.model_family == "qwen"
    assert isinstance(qwen.backend, HFBackend)
    assert qwen.backend._model is None
