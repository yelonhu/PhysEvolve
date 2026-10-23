"""Typed records for the offline data-validation layer.

The dataclasses in this module are intentionally small, JSON-friendly
contracts shared by naive-path validation. Physics execution remains owned
by :mod:`physevolve.executor`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


ValidationMode = Literal["smoke", "candidate"]
ValidationStage = Literal[1, 2]


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Configuration for one validation run."""

    mode: ValidationMode
    stage: ValidationStage = 1

    sandbox_timeout_s: float = 5.0
    sim_timeout_s: float = 30.0

    compute_quality: bool = True
    min_quality_score: float = 0.55

    save_artifacts: bool = False
    candidate_root: Optional[Path] = None

    save_video: bool = False
    save_latent: bool = False  # rejected at write time; probing was removed
    save_targets: bool = False

    allow_legacy_thinking_tag: bool = True

    def __post_init__(self) -> None:
        """Normalize path inputs and reject unsupported modes early."""

        if self.mode not in ("smoke", "candidate"):
            raise ValueError(f"unknown validation mode {self.mode!r}")
        if self.stage not in (1, 2):
            raise ValueError(f"unsupported validation stage {self.stage!r}")
        if not (0.0 <= self.min_quality_score <= 1.0):
            raise ValueError("min_quality_score must be in [0, 1]")
        if self.candidate_root is not None and not isinstance(
            self.candidate_root, Path
        ):
            object.__setattr__(self, "candidate_root", Path(self.candidate_root))


@dataclass(frozen=True, slots=True)
class FormatReport:
    """Result of lightweight proposer-output format validation."""

    ok: bool
    has_think: bool
    has_propose: bool
    has_build_scene: bool
    has_get_vqa: bool
    errors: tuple[str, ...] = ()
    extracted_code: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ExecutorReport:
    """JSON-friendly summary of :class:`ExecutorResult`."""

    ok: bool
    status: str
    scene_id: Optional[str]
    walltime_s: float
    error_short: Optional[str] = None


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Quality score and diagnostics for an executor-OK scene."""

    hard_pass: bool
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    reject_reasons: tuple[str, ...] = ()
    task_suitability: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    """One complete validation record for a proposer sample."""

    sample_id: str
    source: str
    seed: int

    format: FormatReport
    executor: ExecutorReport
    quality: Optional[QualityReport]

    accepted_as_candidate: bool
    artifact_paths: dict[str, str] = field(default_factory=dict)


__all__ = [
    "ExecutorReport",
    "FormatReport",
    "QualityReport",
    "ValidationConfig",
    "ValidationMode",
    "ValidationRecord",
    "ValidationStage",
]
