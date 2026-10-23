"""Tests for DataValidator and manifest helpers."""

from __future__ import annotations

from pathlib import Path

import physevolve.engine  # noqa: F401 - ensure EGL setup before MuJoCo use.
from physevolve.data_validation import (
    DataValidator,
    ValidationConfig,
    append_validation_record,
    record_to_dict,
    summarize_records,
)
from physevolve.data_validation.manifest import load_jsonl


_OK_CODE = """\
def build_scene(api):
    api.add_sphere('ball', pos=(0.0, 0.0, 1.0), radius=0.1,
                   mass=1.0, friction=0.2)
    api.set_velocity('ball', linear=(1.0, 0.0, 0.0))
    api.set_duration(0.5)
"""

_BAD_SEMANTIC_CODE = """\
def build_scene(api):
    api.add_sphere('ball', pos=(9.0, 0.0, 1.0), radius=0.1, mass=1.0)
"""

_TRIVIAL_CODE = """\
def build_scene(api):
    api.add_sphere('ball', pos=(0.0, 0.0, 1.0), radius=0.1,
                   mass=1.0, fixed=True)
    api.set_duration(0.5)
"""


def test_executor_failure_recorded() -> None:
    validator = DataValidator(ValidationConfig(mode="smoke", sim_timeout_s=10.0))

    record = validator.validate_code(
        _BAD_SEMANTIC_CODE,
        sample_id="bad_001",
        source="unit",
        seed=0,
    )

    assert record.format.ok is True
    assert record.executor.ok is False
    assert record.executor.status == "FAILED_SEMANTIC"
    assert record.accepted_as_candidate is False


def test_trivial_scene_not_candidate() -> None:
    validator = DataValidator(
        ValidationConfig(
            mode="candidate",
            min_quality_score=0.0,
            sim_timeout_s=10.0,
        )
    )

    record = validator.validate_code(
        _TRIVIAL_CODE,
        sample_id="trivial_001",
        source="unit",
        seed=0,
    )

    assert record.executor.status == "TRIVIAL"
    assert record.accepted_as_candidate is False


def test_ok_scene_candidate_acceptance_and_duplicate_rejection(
    tmp_path: Path,
) -> None:
    validator = DataValidator(
        ValidationConfig(
            mode="candidate",
            min_quality_score=0.0,
            candidate_root=tmp_path,
            sim_timeout_s=10.0,
        )
    )

    first = validator.validate_code(
        _OK_CODE,
        sample_id="cand_001",
        source="unit",
        seed=0,
    )
    duplicate = validator.validate_code(
        _OK_CODE,
        sample_id="cand_002",
        source="unit",
        seed=0,
    )

    assert first.accepted_as_candidate is True
    assert first.executor.status == "OK"
    assert first.artifact_paths["physdsl_code_path"].startswith("code/")
    assert duplicate.accepted_as_candidate is False

    manifest_rows = load_jsonl(tmp_path / "manifest.jsonl")
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["scene_id"] == first.executor.scene_id
    assert (tmp_path / first.artifact_paths["physdsl_code_path"]).exists()


def test_jsonl_manifest_roundtrip_and_summary(tmp_path: Path) -> None:
    validator = DataValidator(ValidationConfig(mode="smoke", sim_timeout_s=10.0))
    ok_record = validator.validate_code(
        _OK_CODE,
        sample_id="ok_001",
        source="unit",
        seed=0,
    )
    bad_record = validator.validate_raw_output(
        "<think>intent</think>missing propose",
        sample_id="bad_format",
        source="unit",
        seed=0,
    )

    path = tmp_path / "validation.jsonl"
    append_validation_record(path, ok_record)
    append_validation_record(path, bad_record)

    rows = load_jsonl(path)
    assert rows[0] == record_to_dict(ok_record)
    assert rows[1]["executor"]["status"] == "NOT_RUN"

    summary = summarize_records([ok_record, bad_record])
    assert summary["total"] == 2
    assert summary["failure_reason_distribution"]["OK"] == 1
    assert summary["failure_reason_distribution"]["NOT_RUN"] == 1
