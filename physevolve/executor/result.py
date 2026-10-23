"""Structured executor results and status constructors.

Expected validation, simulation, and rendering failures are represented by
``FailureReason`` values. Results include available scene, trajectory, task,
and repair metadata.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Optional

from physevolve.engine.failure import PROPOSER_PENALTY, FailureReason
from physevolve.engine.schema import SimState

# Error tracebacks can explode on deeply recursive user code; keep the
# stored string bounded so log sinks (wandb, disk) are safe.
_TRACEBACK_CAP_BYTES: int = 4096


@dataclass(frozen=True, slots=True)
class ExecutorResult:
    """Terminal output of one :func:`run_executor` call.

    Attributes:
        scene_id: Deterministic hash of ``(SceneDef, seed)``; empty string
            if the pipeline never reached a frozen scene.
        status: Which exit branch was taken (see docs/architecture.md).
        sim_state: Populated when ``status == FailureReason.OK`` and
            also when ``status == FailureReason.TRIVIAL`` (so TRIVIAL
            can be diagnosed offline). All other failure branches
            leave this ``None``.
        vqa_payload: Only populated on Stage 2 OK runs. Tuple of
            ``(question, options, correct_idx)`` — the authoritative
            payload assembled by the Stage-2 skill card's
            ``ground_truth_extractor`` (docs/architecture.md).
        vqa_spec: Stage-2 only. The resolved
            ``{"skill": str, "phrasing": str, "params": dict}`` triple
            taken from the Proposer's ``VQA_SPEC`` literal after
            registry validation. Used by the step builder for per-skill
            metrics + novelty signature computation.
        error_message: Short human-readable summary of the failure.
        error_traceback: Full Python traceback, capped to a few KB.
        proposer_penalty: Looked up once from :data:`PROPOSER_PENALTY`
            and frozen into the result so downstream code never re-
            consults the enum.
        repair_trace: Optional JSON-safe audit trail for deterministic
            executor/compiler repairs. ``None`` means no repair path was active.
        walltime_s: End-to-end pipeline walltime in seconds (sandbox +
            sim + render + VQA).
    """

    scene_id: str
    status: FailureReason
    sim_state: Optional[SimState]
    vqa_payload: Optional[tuple[str, list[str], int]]
    error_message: Optional[str]
    error_traceback: Optional[str]
    proposer_penalty: float
    walltime_s: float
    vqa_spec: Optional[dict] = None
    # Scene-template (multi-GT) path: a list of derived VQAs from ONE
    # simulation. Each item is
    # ``{"payload": (question, options, correct_idx), "spec": {...}}``.
    # Colour-permuted variant questions additionally carry their OWN
    # ``"sim_state"`` (a re-rendered video with recoloured balls); baseline
    # questions omit it and reuse :attr:`sim_state`. ``None`` on the
    # single-question typed-skill path; non-empty on a scene-manifest OK run
    # (and ``vqa_payload`` mirrors the first item for back-compat with code that
    # only reads one question). NOTE: ``sim_state`` is intentionally excluded
    # from JSON serialisation in ``executor_result_record`` (not JSON-safe).
    vqa_payloads: Optional[list[dict]] = None
    repair_trace: Optional[dict] = None

    @property
    def ok(self) -> bool:
        """``True`` iff ``status == FailureReason.OK``."""

        return self.status == FailureReason.OK


def _truncate_traceback(tb: Optional[str]) -> Optional[str]:
    """Cap traceback size. ``None`` in, ``None`` out."""

    if tb is None:
        return None
    encoded = tb.encode("utf-8", errors="replace")
    if len(encoded) <= _TRACEBACK_CAP_BYTES:
        return tb
    # Keep the tail: the innermost frame is where the user's bug lives.
    tail = encoded[-_TRACEBACK_CAP_BYTES:].decode("utf-8", errors="replace")
    return "...[truncated]...\n" + tail


def make_failure(
    status: FailureReason,
    scene_id: str,
    err: BaseException,
    walltime_s: float,
    repair_trace: Optional[dict] = None,
) -> ExecutorResult:
    """Construct an :class:`ExecutorResult` for a failed pipeline branch.

    Args:
        status: The failure category from docs/architecture.md. Must not be
            :attr:`FailureReason.OK`.
        scene_id: Best-effort scene ID. Empty string when the pipeline
            never reached :meth:`PhysDSLApi.freeze`.
        err: The exception that triggered this branch.
        walltime_s: Seconds since pipeline entry.

    Returns:
        A frozen :class:`ExecutorResult` with no ``sim_state`` or
        ``vqa_payload``.
    """

    if status == FailureReason.OK:
        raise ValueError("make_failure() cannot be used for OK status")

    message = f"{type(err).__name__}: {err}"
    tb_str = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    return ExecutorResult(
        scene_id=scene_id,
        status=status,
        sim_state=None,
        vqa_payload=None,
        error_message=message,
        error_traceback=_truncate_traceback(tb_str),
        proposer_penalty=PROPOSER_PENALTY[status],
        walltime_s=walltime_s,
        repair_trace=repair_trace,
    )


def make_ok(
    scene_id: str,
    sim_state: SimState,
    vqa_payload: Optional[tuple[str, list[str], int]],
    walltime_s: float,
    vqa_spec: Optional[dict] = None,
    vqa_payloads: Optional[list[dict]] = None,
    repair_trace: Optional[dict] = None,
) -> ExecutorResult:
    """Construct an :class:`ExecutorResult` for a successful pipeline run."""

    return ExecutorResult(
        scene_id=scene_id,
        status=FailureReason.OK,
        sim_state=sim_state,
        vqa_payload=vqa_payload,
        error_message=None,
        error_traceback=None,
        proposer_penalty=PROPOSER_PENALTY[FailureReason.OK],
        walltime_s=walltime_s,
        vqa_spec=vqa_spec,
        vqa_payloads=vqa_payloads,
        repair_trace=repair_trace,
    )


def make_trivial(
    scene_id: str,
    sim_state: SimState,
    walltime_s: float,
    repair_trace: Optional[dict] = None,
) -> ExecutorResult:
    """Construct an :class:`ExecutorResult` for a TRIVIAL-classified scene.

    TRIVIAL keeps the ``sim_state`` (useful for debugging / learnability
    diagnostics) but skips the Solver turn, so callers MUST treat this as
    a non-OK status.
    """

    return ExecutorResult(
        scene_id=scene_id,
        status=FailureReason.TRIVIAL,
        sim_state=sim_state,
        vqa_payload=None,
        error_message="scene classified as TRIVIAL (no motion, no collisions)",
        error_traceback=None,
        proposer_penalty=PROPOSER_PENALTY[FailureReason.TRIVIAL],
        walltime_s=walltime_s,
        repair_trace=repair_trace,
    )


__all__ = [
    "ExecutorResult",
    "make_failure",
    "make_ok",
    "make_trivial",
]
