"""Run executor jobs in an order-preserving process pool.

Workers use ``spawn`` to initialize independent MuJoCo/OpenGL contexts. A shared
disk cache can reuse results across workers. A crashed worker may be retried
once according to ``on_worker_crash``; ordinary executor failures are returned
as statuses. Per-future timeouts are reported as ``FAILED_TIMEOUT``.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from concurrent.futures import (
    BrokenExecutor,
    ProcessPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence

from physevolve.engine.failure import PROPOSER_PENALTY, FailureReason
from physevolve.executor.pipeline import run_executor
from physevolve.executor.result import ExecutorResult
from physevolve.executor.scene_cache import SceneCache, compute_cache_key

# ─── Module-level worker state ───────────────────────────────────────
#
# Each spawned worker has its own module import of ``physevolve.executor``;
# the cache-configuration globals below are populated by
# :func:`_worker_init` at startup and read by :func:`_worker_run`.

_WORKER_CACHE: Optional[SceneCache] = None


def _worker_init(
    disk_cache_dir: Optional[str],
    max_mem_items: int,
) -> None:
    """Per-worker startup hook. Runs once per spawned process.

    Responsibilities:

    * Trigger ``physevolve.engine`` import so ``MUJOCO_GL=egl`` is set
      before MuJoCo touches a GL context.
    * Instantiate a per-worker :class:`SceneCache` backed by the shared
      disk directory (if given). The in-memory tier is per-worker by
      design — cross-worker in-memory sharing would require shared
      memory manager gymnastics and would slow us down.
    """

    import physevolve.engine  # noqa: F401 — env-var side effects.

    global _WORKER_CACHE
    _WORKER_CACHE = SceneCache(
        max_mem_items=max_mem_items,
        disk_dir=Path(disk_cache_dir) if disk_cache_dir else None,
    )


def _worker_run(
    code: str,
    stage: int,
    seed: int,
    sandbox_timeout_s: float,
    sim_timeout_s: float,
    enable_cache: bool,
) -> ExecutorResult:
    """Worker entrypoint. Consults the cache, then calls ``run_executor``."""

    global _WORKER_CACHE
    if enable_cache and _WORKER_CACHE is not None:
        key = compute_cache_key(code, stage, seed)
        hit = _WORKER_CACHE.get(key)
        if hit is not None:
            return hit
    else:
        key = None

    result = run_executor(
        code,
        stage=stage,
        seed=seed,
        sandbox_timeout_s=sandbox_timeout_s,
        sim_timeout_s=sim_timeout_s,
    )

    if enable_cache and _WORKER_CACHE is not None and key is not None:
        _WORKER_CACHE.put(key, result)
    return result


# ─── Public configuration ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    """Settings for :func:`execute_parallel`.

    Fields:

    * ``n_workers``: process-pool size.
    * ``worker_timeout_s``: per-future wall-clock budget. 45 s covers
      the 30 s engine sim timeout plus overhead.
    * ``batch_chunksize``: number of scenes per ``map`` chunk. 4 keeps
      scheduler overhead low without sacrificing load-balance.
    * ``on_worker_crash``: when a worker dies mid-task, either retry
      the task once on a restarted pool or return a synthetic
      ``FAILED_RUNTIME`` immediately.
    * ``enable_scene_cache``: turn the two-tier cache on/off.
    * ``scene_cache_max_mem``: LRU bound for the per-worker in-memory
      tier. 512 per worker is cheap RAM.
    * ``scene_cache_disk_dir``: shared disk cache. ``None`` = mem only.
    * ``sandbox_timeout_s`` / ``sim_timeout_s``: forwarded to
      ``run_executor``.
    """

    n_workers: int = 32
    worker_timeout_s: float = 45.0
    batch_chunksize: int = 4
    on_worker_crash: Literal["retry_once", "fail_scene"] = "retry_once"
    enable_scene_cache: bool = True
    scene_cache_max_mem: int = 512
    scene_cache_disk_dir: Optional[Path] = field(
        default_factory=lambda: Path("cache/scenes")
    )
    sandbox_timeout_s: float = 5.0
    sim_timeout_s: float = 30.0


# ─── Failure synthesis (preserves the ExecutorResult contract) ───────


def _synthesize_crash_failure(
    scene_index: int,
    reason: FailureReason,
    error: BaseException,
    walltime_s: float,
) -> ExecutorResult:
    """Manufacture an :class:`ExecutorResult` for infrastructure failures.

    We can't reach :func:`physevolve.executor.result.make_failure`
    directly for pool-level crashes because that helper expects a raised
    exception whose traceback is already attached — here we synthesise
    a short fake one.
    """

    message = f"{type(error).__name__}: {error}"
    tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
    tb = "".join(tb_lines) if tb_lines else f"parallel executor: {message}"
    return ExecutorResult(
        scene_id="",
        status=reason,
        sim_state=None,
        vqa_payload=None,
        error_message=f"[parallel scene {scene_index}] {message}",
        error_traceback=tb,
        proposer_penalty=PROPOSER_PENALTY[reason],
        walltime_s=walltime_s,
    )


# ─── Main entry point ────────────────────────────────────────────────


def execute_parallel(
    proposer_codes: Sequence[str],
    stages: Optional[Sequence[int]] = None,
    seeds: Optional[Sequence[int]] = None,
    config: Optional[ParallelConfig] = None,
) -> list[ExecutorResult]:
    """Run many Proposer codes through the executor pipeline in parallel.

    Args:
        proposer_codes: Python source strings; one per scene. Each is
            passed verbatim to :func:`run_executor`.
        stages: Stage per scene (``1`` or ``2``). Defaults to stage 1
            for every scene.
        seeds: Seed per scene. Defaults to ``range(len(proposer_codes))``.
        config: Parallel configuration; a fresh :class:`ParallelConfig`
            with defaults is used if omitted.

    Returns:
        One :class:`ExecutorResult` per input, in the SAME order. Every
        expected failure is encoded into
        :attr:`ExecutorResult.status` — no exceptions escape.

    Concurrency guarantees:

    * Order-preserving: output index ``i`` corresponds to input index ``i``.
    * No retries on Proposer-level failures (deterministic, waste of CPU).
    * Infrastructure-level failures (pool crash, worker death) are
      retried ONCE when ``config.on_worker_crash == "retry_once"``.
    """

    if config is None:
        config = ParallelConfig()

    n = len(proposer_codes)
    if n == 0:
        return []

    if stages is None:
        stages_list = [1] * n
    else:
        if len(stages) != n:
            raise ValueError(
                f"stages length {len(stages)} != proposer_codes length {n}"
            )
        stages_list = list(stages)

    if seeds is None:
        seeds_list = list(range(n))
    else:
        if len(seeds) != n:
            raise ValueError(
                f"seeds length {len(seeds)} != proposer_codes length {n}"
            )
        seeds_list = list(seeds)

    ctx = mp.get_context("spawn")
    disk_cache_dir: Optional[str] = (
        str(config.scene_cache_disk_dir) if config.scene_cache_disk_dir else None
    )
    if disk_cache_dir is not None:
        # Create the shared cache dir once in the parent so every worker
        # finds it ready (also gives us an early error if the parent has
        # no write permission).
        Path(disk_cache_dir).mkdir(parents=True, exist_ok=True)

    results: list[Optional[ExecutorResult]] = [None] * n
    pending_indices = list(range(n))
    retries_used: dict[int, int] = {}

    def _build_pool() -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=max(1, config.n_workers),
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(disk_cache_dir, config.scene_cache_max_mem),
        )

    while pending_indices:
        pool = _build_pool()
        crashed_indices: list[int] = []
        try:
            future_to_idx = {
                pool.submit(
                    _worker_run,
                    proposer_codes[i],
                    stages_list[i],
                    seeds_list[i],
                    config.sandbox_timeout_s,
                    config.sim_timeout_s,
                    config.enable_scene_cache,
                ): i
                for i in pending_indices
            }
            for future, idx in list(future_to_idx.items()):
                try:
                    results[idx] = future.result(timeout=config.worker_timeout_s)
                except FuturesTimeoutError as e:
                    results[idx] = _synthesize_crash_failure(
                        idx, FailureReason.FAILED_TIMEOUT, e, config.worker_timeout_s
                    )
                except BrokenProcessPool as e:
                    # A pool-level crash means ALL in-flight futures for
                    # this pool are dead. Collect the indices and rebuild.
                    crashed_indices = [
                        j for f, j in future_to_idx.items() if not f.done()
                    ]
                    # Also capture the index that surfaced the exception
                    # if it isn't already pending.
                    if idx not in crashed_indices:
                        crashed_indices.append(idx)
                    # Short-circuit the per-future loop — reconvene on
                    # the outer ``while`` with crashed_indices as the
                    # new pending set.
                    for j in crashed_indices:
                        # Temporary sentinel so we know to retry these.
                        results[j] = None
                    break
                except BrokenExecutor as e:
                    # Tolerate any other pool-level death the same way.
                    crashed_indices = [
                        j for f, j in future_to_idx.items() if not f.done()
                    ]
                    if idx not in crashed_indices:
                        crashed_indices.append(idx)
                    for j in crashed_indices:
                        results[j] = None
                    break
                except Exception as e:  # noqa: BLE001 — defensive catch-all.
                    # Any other uncaught exception from the worker is a
                    # genuine bug — but at parallel-layer granularity we
                    # still encode it so batch processing doesn't die.
                    results[idx] = _synthesize_crash_failure(
                        idx, FailureReason.FAILED_RUNTIME, e, 0.0
                    )
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

        # Decide what (if anything) goes into the next pool iteration.
        if crashed_indices:
            if config.on_worker_crash == "retry_once":
                next_pending: list[int] = []
                for j in crashed_indices:
                    if retries_used.get(j, 0) >= 1:
                        # Already retried; lock in a synthetic failure.
                        results[j] = _synthesize_crash_failure(
                            j,
                            FailureReason.FAILED_RUNTIME,
                            RuntimeError(
                                "worker crashed twice (retry budget exhausted)"
                            ),
                            0.0,
                        )
                    else:
                        retries_used[j] = retries_used.get(j, 0) + 1
                        next_pending.append(j)
                pending_indices = next_pending
            else:  # "fail_scene"
                for j in crashed_indices:
                    results[j] = _synthesize_crash_failure(
                        j,
                        FailureReason.FAILED_RUNTIME,
                        RuntimeError("worker crashed; on_worker_crash=fail_scene"),
                        0.0,
                    )
                pending_indices = []
        else:
            pending_indices = []

    # Every index must now be populated; a None here is a bug in the
    # control-flow above.
    for i, r in enumerate(results):
        if r is None:  # pragma: no cover — guard against control-flow bugs.
            results[i] = _synthesize_crash_failure(
                i,
                FailureReason.FAILED_RUNTIME,
                RuntimeError("parallel executor left an index unpopulated"),
                0.0,
            )
    return [r for r in results if r is not None]


__all__ = [
    "ParallelConfig",
    "execute_parallel",
]
