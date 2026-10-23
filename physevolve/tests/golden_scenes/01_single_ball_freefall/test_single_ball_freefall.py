"""Golden scene 01 — integration test."""

from __future__ import annotations

import physevolve.engine  # noqa: F401 — EGL env.
from physevolve.tests.golden_scenes.conftest import (
    assert_body_grid_within,
    load_expected,
    run_golden,
    scene_path,
)


def test_single_ball_freefall() -> None:
    result = run_golden(scene_path(__file__))
    expected = load_expected(__file__)
    assert_body_grid_within(result, expected, tolerance=1)
