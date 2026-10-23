"""Tests for smoke filesystem helpers and run-id generation."""

from __future__ import annotations

from datetime import datetime

import pytest

from physevolve.smoke.io import append_jsonl, create_run_dir, read_jsonl
from physevolve.smoke.run_config import (
    build_run_id,
    model_short_name,
)


def test_run_id_creation() -> None:
    run_id = build_run_id(
        gate="proposer",
        model="/models/Qwen3-VL-8B-Instruct",
        prompt_version="proposer_stage1_v1",
        n_samples=50,
        now=datetime(2026, 5, 1, 14, 30),
    )

    assert run_id == "20260501_1430_proposer_qwen3vl8binstruct_proposer_stage1_v1_n50"
    assert model_short_name("/models/Qwen3-VL-8B-Instruct") == "qwen3vl8binstruct"


def test_run_directory_creation_is_immutable(tmp_path) -> None:
    run_dir = create_run_dir(tmp_path, "run_001")

    assert run_dir.exists()
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, "run_001")


def test_jsonl_append_and_read(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"

    append_jsonl(path, {"b": 2})
    append_jsonl(path, {"a": 1})

    assert read_jsonl(path) == [{"b": 2}, {"a": 1}]
