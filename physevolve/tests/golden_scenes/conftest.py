"""Shared helpers for the 10-scene golden integration suite.

Each scene's ``test_{name}.py`` imports ``run_golden`` + ``assert_*``
and passes its sibling ``scene.py`` / ``expected.json``. No shared
fixtures — the executor is reasonably fast (~0.1 s per scene at the
default render config) so we keep the tests independent for clearer
diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from physevolve.engine.failure import FailureReason
from physevolve.engine.schema import world_to_grid
from physevolve.executor import ExecutorResult, run_executor


def run_golden(scene_py_path: Path) -> ExecutorResult:
    """Execute a golden scene's ``scene.py`` through the full pipeline.

    ``scene.py`` is read verbatim and fed to :func:`run_executor`. This
    mirrors how a Proposer's generated code enters the system, keeps
    the sandbox in play (so scene files get the same treatment as
    learned Proposer output), and avoids any import-time side effects
    leaking between scenes.
    """

    src = Path(scene_py_path).read_text(encoding="utf-8")
    return run_executor(src, stage=1, seed=0, sim_timeout_s=30.0)


def load_expected(test_file: str | Path) -> dict[str, Any]:
    """Load ``expected.json`` sitting next to ``test_file``."""

    here = Path(test_file).resolve().parent
    path = here / "expected.json"
    return json.loads(path.read_text(encoding="utf-8"))


def scene_path(test_file: str | Path) -> Path:
    """Return the ``scene.py`` sitting next to ``test_file``."""

    return Path(test_file).resolve().parent / "scene.py"


def _manhattan(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return sum(abs(int(ai) - int(bi)) for ai, bi in zip(a, b))


def assert_body_grid_within(
    result: ExecutorResult,
    expected: dict[str, Any],
    tolerance: int = 1,
) -> None:
    """Assert that ``result`` matches ``expected.json`` to within ``tolerance``.

    ``expected`` schema::

        {
          "status": "ok" | "trivial",
          "min_collisions": 0,
          "bodies": {
            "name": {"final_grid": [gx, gy, gz], "final_status": "fallen"}
          }
        }

    * ``status`` is checked exactly against the executor's top-level
      :class:`FailureReason`.
    * Per-body ``final_grid`` is compared via grid Manhattan ≤ ``tolerance``.
    * Per-body ``final_status`` must match exactly.
    * ``min_collisions`` is a lower bound on
      ``len(result.sim_state.collisions)``.

    Missing bodies in ``expected.bodies`` are ignored (so scenes can
    choose to only pin the ones whose final state is load-bearing).
    """

    expected_status = expected.get("status", "ok").lower()
    if expected_status == "ok":
        assert result.status == FailureReason.OK, (
            f"expected OK, got {result.status}: {result.error_message}"
        )
    elif expected_status == "trivial":
        assert result.status == FailureReason.TRIVIAL, (
            f"expected TRIVIAL, got {result.status}: {result.error_message}"
        )
        # TRIVIAL scenes still keep sim_state.
        assert result.sim_state is not None
        # No further grid checks for TRIVIAL — there are no dynamic bodies
        # whose final state is meaningful.
        return
    else:
        raise ValueError(f"unknown expected.status {expected_status!r}")

    sim = result.sim_state
    assert sim is not None

    min_collisions = int(expected.get("min_collisions", 0))
    assert len(sim.collisions) >= min_collisions, (
        f"expected at least {min_collisions} collisions, "
        f"got {len(sim.collisions)}"
    )

    for body_name, body_expected in expected.get("bodies", {}).items():
        traj = sim.trajectories.get(body_name)
        assert traj is not None, f"body {body_name!r} missing from sim_state"
        final_pos = traj.positions[-1]
        actual_grid = world_to_grid(
            (float(final_pos[0]), float(final_pos[1]), float(final_pos[2]))
        )
        want_grid = tuple(int(g) for g in body_expected["final_grid"])
        distance = _manhattan(actual_grid, want_grid)
        assert distance <= tolerance, (
            f"{body_name}: final_grid {actual_grid} differs from expected "
            f"{want_grid} by Manhattan {distance} > tolerance {tolerance}"
        )
        want_status = body_expected["final_status"].lower()
        got_status = traj.final_status.value.lower()
        assert got_status == want_status, (
            f"{body_name}: final_status {got_status!r} != expected "
            f"{want_status!r}"
        )


__all__ = [
    "run_golden",
    "load_expected",
    "scene_path",
    "assert_body_grid_within",
]
