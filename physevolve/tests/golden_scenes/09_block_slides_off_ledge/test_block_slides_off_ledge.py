"""Golden scene 09 — integration test."""

from __future__ import annotations

import physevolve.engine  # noqa: F401 — EGL env.
from physevolve.engine.failure import FailureReason
from physevolve.tests.golden_scenes.conftest import (
    assert_body_grid_within,
    load_expected,
    run_golden,
    scene_path,
)


def test_block_slides_off_ledge() -> None:
    result = run_golden(scene_path(__file__))
    expected = load_expected(__file__)
    assert_body_grid_within(result, expected, tolerance=1)
    # Explicit triviality gate: scene 09 is THE boundary test — we
    # must guarantee the pipeline does NOT classify this as TRIVIAL.
    assert result.status == FailureReason.OK, (
        f"scene 09 must classify as OK (not TRIVIAL), got {result.status}"
    )
