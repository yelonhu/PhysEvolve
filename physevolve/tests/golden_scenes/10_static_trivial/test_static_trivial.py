"""Golden scene 10 — integration test.

Verifies that a scene with no dynamic bodies round-trips through the
executor as :attr:`FailureReason.TRIVIAL`.
"""

from __future__ import annotations

import physevolve.engine  # noqa: F401 — EGL env.
from physevolve.engine.failure import FailureReason
from physevolve.tests.golden_scenes.conftest import (
    assert_body_grid_within,
    load_expected,
    run_golden,
    scene_path,
)


def test_static_trivial() -> None:
    result = run_golden(scene_path(__file__))
    expected = load_expected(__file__)
    assert_body_grid_within(result, expected, tolerance=1)
    # Explicit post-condition: TRIVIAL must surface as a distinct
    # FailureReason (not OK), and the pipeline must still populate
    # sim_state for offline diagnostics.
    assert result.status == FailureReason.TRIVIAL
    assert result.sim_state is not None
