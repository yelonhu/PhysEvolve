"""Executor termination statuses, exceptions, and Proposer failure penalties.
"""

from __future__ import annotations

from enum import Enum


class FailureReason(str, Enum):
    """Terminal status of an executor pipeline run.

    ``OK`` means the Solver turn proceeds; every other value means the
    Solver turn is skipped and the Proposer receives only the failure
    penalty (docs/architecture.md).
    """

    OK = "ok"
    FAILED_SYNTAX = "syntax"            # Python parse error
    FAILED_SEMANTIC = "semantic"        # DSL validation error (bad args, names)
    FAILED_RUNTIME = "runtime"          # build_scene raised non-validation exc
    FAILED_COMPILE = "compile"          # MJCF compile / MjModel init error
    FAILED_OVERLAP = "overlap"          # AABB interpenetration at t=0
    FAILED_NAN = "nan"                  # qpos / qvel went NaN during sim
    FAILED_TIMEOUT = "timeout"          # sim or render exceeded wall budget
    FAILED_RENDER = "render"            # mujoco.Renderer raised
    FAILED_VQA = "vqa"                  # get_vqa() raised or returned invalid
    TRIVIAL = "trivial"                 # scene static; no physics happened


# Proposer base reward per failure reason. Applied BEFORE the learnability
# term (which is only added when status == OK). All non-OK reasons are
# non-positive; TRIVIAL is zero so exploration is not over-punished.
PROPOSER_PENALTY: dict[FailureReason, float] = {
    FailureReason.OK:              0.0,
    FailureReason.FAILED_SYNTAX:   -1.0,
    FailureReason.FAILED_SEMANTIC: -1.0,
    FailureReason.FAILED_RUNTIME:  -0.8,
    FailureReason.FAILED_COMPILE:  -0.5,
    FailureReason.FAILED_OVERLAP:  -0.5,
    FailureReason.FAILED_NAN:      -0.3,
    FailureReason.FAILED_TIMEOUT:  -0.3,
    FailureReason.FAILED_RENDER:   -0.3,
    FailureReason.FAILED_VQA:      -1.0,
    FailureReason.TRIVIAL:          0.0,
}


# ─── Exceptions raised inside the engine/executor and caught at the
# ─── pipeline boundary to be re-encoded as FailureReason values.


class DSLValidationError(ValueError):
    """Raised by ``PhysDSLApi`` when a validation rule is violated.

    Caught by ``executor.pipeline`` and mapped to
    :attr:`FailureReason.FAILED_SEMANTIC`. Subclasses ``ValueError`` so
    code that catches ``ValueError`` broadly still catches it; specific
    handlers should catch ``DSLValidationError`` directly.
    """


class SimulationDivergedError(RuntimeError):
    """Raised when qpos or qvel becomes NaN during stepping.

    Maps to :attr:`FailureReason.FAILED_NAN`.
    """


class SimulationTimeoutError(TimeoutError):
    """Raised when simulation wall-clock exceeds the configured budget.

    Maps to :attr:`FailureReason.FAILED_TIMEOUT`. Subclasses built-in
    ``TimeoutError`` so ``signal.SIGALRM`` handlers can raise a single
    exception type consistently.
    """


class InitialOverlapError(ValueError):
    """Raised when two bodies interpenetrate above the AABB tolerance at t=0.

    Maps to :attr:`FailureReason.FAILED_OVERLAP`.
    """


__all__ = [
    "FailureReason",
    "PROPOSER_PENALTY",
    "DSLValidationError",
    "SimulationDivergedError",
    "SimulationTimeoutError",
    "InitialOverlapError",
]
