"""Tests for migrated PhysDSL example fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from physevolve.engine.failure import FailureReason
from physevolve.executor import run_executor


_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE_DIR = _REPO_ROOT / "data" / "fixtures" / "physdsl_examples"


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize(
    "name,min_rows",
    [
        ("proposer_fewshot.jsonl", 8),
        ("structural_regression.jsonl", 8),
    ],
)
def test_fixture_files_have_expected_schema(name: str, min_rows: int) -> None:
    rows = _load_jsonl(_FIXTURE_DIR / name)
    assert len(rows) >= min_rows
    required = {
        "scene_id",
        "category",
        "source_path",
        "proposer_text",
        "code",
        "use_tags",
    }
    for row in rows:
        assert required <= row.keys()
        assert "def build_scene(api):" in row["code"]


@pytest.mark.slow
def test_regression_examples_execute() -> None:
    rows = _load_jsonl(_FIXTURE_DIR / "structural_regression.jsonl")[:2]
    for row in rows:
        result = run_executor(row["code"], stage=int(row.get("stage", 1)), seed=0)
        assert result.status in (FailureReason.OK, FailureReason.TRIVIAL)
