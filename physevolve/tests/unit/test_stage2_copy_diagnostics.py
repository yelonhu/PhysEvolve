"""Tests for Stage-2 proposer copy diagnostics."""

from __future__ import annotations

import pytest

from physevolve.inference.parsing import extract_propose_block
from physevolve.skills.scene_templates import (
    get_scene_template_registry,
    select_fewshot_example,
)
from physevolve.training.stage2_copy_diagnostics import (
    Stage2CopyDiagnostics,
    canonical_ast_fingerprint,
    summarize_copy_diagnostics,
)


def test_canonical_ast_fingerprint_ignores_formatting() -> None:
    code_a = "def build_scene(api):\n    api.set_duration(1.0)\n"
    code_b = "def build_scene(api):\n\n    api.set_duration(1.0)\n"

    dump_a, hash_a, valid_a = canonical_ast_fingerprint(code_a)
    dump_b, hash_b, valid_b = canonical_ast_fingerprint(code_b)

    assert valid_a is True
    assert valid_b is True
    assert dump_a == dump_b
    assert hash_a == hash_b


def test_detects_exact_golden_fewshot_copy() -> None:
    template = get_scene_template_registry().lookup("motion_lab")
    code = extract_propose_block(template.fewshot_example)
    assert code is not None

    tracker = Stage2CopyDiagnostics(max_recent=4)
    diagnostic = tracker.analyze(code, scene_kind="motion_lab")

    assert diagnostic.ast_valid is True
    assert diagnostic.exact_fewshot is True
    assert diagnostic.similarity_to_fewshot == pytest.approx(1.0)


def test_detects_exact_copy_of_any_golden_variant() -> None:
    template = get_scene_template_registry().lookup("motion_lab")
    code = extract_propose_block(select_fewshot_example(template, 1))
    assert code is not None

    tracker = Stage2CopyDiagnostics(max_recent=4)
    diagnostic = tracker.analyze(code, scene_kind="motion_lab")

    assert diagnostic.ast_valid is True
    assert diagnostic.exact_fewshot is True
    assert diagnostic.similarity_to_fewshot == pytest.approx(1.0)


def test_detects_recent_ast_repeat_and_summarizes() -> None:
    code = "def build_scene(api):\n    api.set_duration(1.0)\n"
    tracker = Stage2CopyDiagnostics(max_recent=4)

    first = tracker.analyze(code, scene_kind="motion_lab")
    tracker.remember(first)
    second = tracker.analyze(code, scene_kind="motion_lab")

    summary = summarize_copy_diagnostics([first, second])

    assert first.recent_repeat is False
    assert second.recent_repeat is True
    assert second.max_similarity_to_recent == pytest.approx(1.0)
    assert summary["stage2_copy_diagnostic_count"] == 2
    assert summary["stage2_copy_recent_repeat_count"] == 1
    assert summary["stage2_copy_similarity_to_recent_max"] == pytest.approx(1.0)
