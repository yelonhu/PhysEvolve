"""Executor layer — Proposer source -> SimState + VQA payload.

Public surface:

* :func:`run_executor` — the one function RL code should call to run a
  single scene synchronously.
* :class:`ExecutorResult` — the return type.
* :class:`SimStateView` / :func:`sim_state_public_view` — the stripped-
  down view passed to Stage-2 ``get_vqa`` functions.
* :class:`RestrictedExecError` — raised only within the sandbox; the
  pipeline converts it to ``FailureReason.FAILED_SEMANTIC``.
* :func:`execute_parallel` / :class:`ParallelConfig` — multiprocess
  fan-out over :func:`run_executor`.
* :class:`SceneCache` — two-tier (memory + disk) cache of
  :class:`ExecutorResult` objects keyed by ``(code, stage, seed)``.
"""

from physevolve.executor.parallel import ParallelConfig, execute_parallel
from physevolve.executor.pipeline import run_executor
from physevolve.executor.result import ExecutorResult
from physevolve.executor.sandbox import RestrictedExecError
from physevolve.executor.scene_cache import SceneCache
from physevolve.executor.state_view import SimStateView, sim_state_public_view

__all__ = [
    "run_executor",
    "ExecutorResult",
    "RestrictedExecError",
    "SimStateView",
    "sim_state_public_view",
    "ParallelConfig",
    "execute_parallel",
    "SceneCache",
]
