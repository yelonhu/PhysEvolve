"""Public validation API for proposer smoke and candidate curation."""

from __future__ import annotations

from typing import Optional

from physevolve.data_validation.format import (
    validate_code_format,
    validate_raw_format,
)
from physevolve.data_validation.manifest import (
    append_candidate_manifest,
    load_candidate_scene_ids,
    write_candidate_artifacts,
)
from physevolve.data_validation.quality import evaluate_quality
from physevolve.data_validation.schemas import (
    ExecutorReport,
    FormatReport,
    QualityReport,
    ValidationConfig,
    ValidationRecord,
)
from physevolve.engine.failure import FailureReason
from physevolve.executor import ExecutorResult, run_executor


_NOT_RUN_STATUS = "NOT_RUN"
_ERROR_SHORT_MAX = 512


class DataValidator:
    """Validate proposer outputs with the existing executor as hard gate."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self._accepted_scene_ids: set[str] = set()
        if config.candidate_root is not None:
            self._accepted_scene_ids.update(
                load_candidate_scene_ids(config.candidate_root)
            )

    def validate_raw_output(
        self,
        raw_output: str,
        *,
        sample_id: str,
        source: str,
        seed: int,
    ) -> ValidationRecord:
        """Validate a model generation containing wrapper tags."""

        fmt = validate_raw_format(
            raw_output,
            stage=self.config.stage,
            allow_legacy_thinking_tag=self.config.allow_legacy_thinking_tag,
        )
        return self._validate_after_format(
            fmt,
            sample_id=sample_id,
            source=source,
            seed=seed,
        )

    def validate_code(
        self,
        code: str,
        *,
        sample_id: str,
        source: str,
        seed: int,
    ) -> ValidationRecord:
        """Validate already-extracted PhysDSL code."""

        fmt = validate_code_format(code, stage=self.config.stage)
        return self._validate_after_format(
            fmt,
            sample_id=sample_id,
            source=source,
            seed=seed,
        )

    def _validate_after_format(
        self,
        fmt: FormatReport,
        *,
        sample_id: str,
        source: str,
        seed: int,
    ) -> ValidationRecord:
        if not fmt.ok or fmt.extracted_code is None:
            return ValidationRecord(
                sample_id=sample_id,
                source=source,
                seed=seed,
                format=fmt,
                executor=_executor_not_run(),
                quality=None,
                accepted_as_candidate=False,
                artifact_paths={},
            )

        result = run_executor(
            fmt.extracted_code,
            stage=self.config.stage,
            seed=seed,
            sandbox_timeout_s=self.config.sandbox_timeout_s,
            sim_timeout_s=self.config.sim_timeout_s,
        )
        executor = _executor_report(result)
        quality = _quality_report(result, compute=self.config.compute_quality)
        accepted = self._accepted_as_candidate(executor, quality)

        artifact_paths: dict[str, str] = {}
        if accepted and self.config.candidate_root is not None:
            if result.sim_state is None:
                raise RuntimeError("accepted candidate is missing sim_state")
            artifact_paths = write_candidate_artifacts(
                self.config.candidate_root,
                code=fmt.extracted_code,
                sim_state=result.sim_state,
                save_video=self.config.save_video,
                save_latent=self.config.save_latent,
                save_targets=self.config.save_targets,
            )

        record = ValidationRecord(
            sample_id=sample_id,
            source=source,
            seed=seed,
            format=fmt,
            executor=executor,
            quality=quality,
            accepted_as_candidate=accepted,
            artifact_paths=artifact_paths,
        )

        if accepted and executor.scene_id is not None:
            self._accepted_scene_ids.add(executor.scene_id)
            if self.config.candidate_root is not None:
                append_candidate_manifest(self.config.candidate_root, record)

        return record

    def _accepted_as_candidate(
        self,
        executor: ExecutorReport,
        quality: Optional[QualityReport],
    ) -> bool:
        if self.config.mode != "candidate":
            return False
        if not executor.ok or executor.status != FailureReason.OK.name:
            return False
        if executor.scene_id is None or executor.scene_id in self._accepted_scene_ids:
            return False
        if quality is None or not quality.hard_pass:
            return False
        return quality.score >= self.config.min_quality_score


def _executor_not_run() -> ExecutorReport:
    return ExecutorReport(
        ok=False,
        status=_NOT_RUN_STATUS,
        scene_id=None,
        walltime_s=0.0,
        error_short="format validation failed",
    )


def _executor_report(result: ExecutorResult) -> ExecutorReport:
    return ExecutorReport(
        ok=result.ok,
        status=result.status.name,
        scene_id=result.scene_id or None,
        walltime_s=float(result.walltime_s),
        error_short=_short_error(result.error_message),
    )


def _quality_report(
    result: ExecutorResult,
    *,
    compute: bool,
) -> Optional[QualityReport]:
    if not compute or not result.ok or result.sim_state is None:
        return None
    return evaluate_quality(result.sim_state)


def _short_error(message: Optional[str]) -> Optional[str]:
    if message is None:
        return None
    if len(message) <= _ERROR_SHORT_MAX:
        return message
    return message[: _ERROR_SHORT_MAX - 3] + "..."


__all__ = ["DataValidator"]
