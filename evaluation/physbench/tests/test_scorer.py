"""Tests for the accuracy scorer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.physbench import scorer
from evaluation.physbench.scorer import score


def _dump(path: Path, payload: list[dict]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_first_letter_variants() -> None:
    assert scorer._first_letter("A") == "A"
    assert scorer._first_letter(" A ") == "A"
    assert scorer._first_letter("A. correct") == "A"
    assert scorer._first_letter("(B)") == "B"
    assert scorer._first_letter("Answer: C") == "C"
    assert scorer._first_letter("answer: d") == "D"
    # Prose mention must NOT count as an answer.
    assert scorer._first_letter("the cat") is None
    assert scorer._first_letter(None) is None
    assert scorer._first_letter("") is None


def test_score_perfect_and_partial(tmp_path: Path) -> None:
    preds = tmp_path / "preds.json"
    gold = tmp_path / "gold.json"

    _dump(preds, [
        {"idx": 1, "answer": "A"},
        {"idx": 2, "answer": "B"},   # gold B → correct
        {"idx": 3, "answer": "Answer: c"},  # gold C → correct
        {"idx": 4, "answer": "A"},   # gold D → wrong
        {"idx": 5, "answer": None},  # null → wrong
    ])
    _dump(gold, [
        {"idx": 1, "answer": "A", "task_type": "property",
         "sub_type": "mass", "ability_type": "identify"},
        {"idx": 2, "answer": "B", "task_type": "property",
         "sub_type": "mass", "ability_type": "identify"},
        {"idx": 3, "answer": "C", "task_type": "dynamics",
         "sub_type": "collision", "ability_type": "prediction"},
        {"idx": 4, "answer": "D", "task_type": "dynamics",
         "sub_type": "collision", "ability_type": "prediction"},
        {"idx": 5, "answer": "A", "task_type": "scene",
         "sub_type": "light", "ability_type": "perception"},
    ])

    rep = score(preds, gold)

    assert rep.overall.correct == 3
    assert rep.overall.total == 5
    assert rep.by_task_type["property"].correct == 2
    assert rep.by_task_type["property"].total == 2
    assert rep.by_task_type["dynamics"].correct == 1
    assert rep.by_task_type["dynamics"].total == 2
    assert rep.by_task_type["scene"].correct == 0
    assert rep.by_task_type["scene"].total == 1

    # Round-trip dict representation.
    as_dict = rep.to_dict()
    assert as_dict["overall"]["accuracy"] == 60.0
    assert as_dict["by_task_type"]["property"]["accuracy"] == 100.0


def test_score_handles_unmatched_predictions(tmp_path: Path) -> None:
    preds = tmp_path / "preds.json"
    gold = tmp_path / "gold.json"

    _dump(preds, [
        {"idx": 1, "answer": "A"},
        {"idx": 999, "answer": "A"},   # not in gold
    ])
    _dump(gold, [
        {"idx": 1, "answer": "A", "task_type": "property",
         "sub_type": "mass", "ability_type": "identify"},
    ])

    rep = score(preds, gold)
    assert rep.n_unmatched == 1
    assert rep.overall.correct == 1
    assert rep.overall.total == 1


def test_format_report_runs(tmp_path: Path) -> None:
    preds = tmp_path / "preds.json"
    gold = tmp_path / "gold.json"
    _dump(preds, [{"idx": 1, "answer": "A"}])
    _dump(gold, [{"idx": 1, "answer": "A", "task_type": "property",
                  "sub_type": "mass", "ability_type": "identify"}])
    rep = score(preds, gold)

    text = scorer.format_report(rep, model_name="demo")
    md = scorer.format_markdown(rep, model_name="demo")

    assert "Overall" in text and "100.00%" in text
    assert "# PhysBench" in md and "demo" in md
