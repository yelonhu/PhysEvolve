"""Tests for data-validation metrics and quality scoring."""

from __future__ import annotations

import pytest

import physevolve.engine  # noqa: F401 - ensure EGL setup before MuJoCo use.
from physevolve.data_validation import compute_quality_metrics, evaluate_quality
from physevolve.executor import run_executor


_MOVING_SPHERE = """\
def build_scene(api):
    api.add_sphere('ball', pos=(0.0, 0.0, 1.0), radius=0.1,
                   mass=1.0, friction=0.2)
    api.set_velocity('ball', linear=(1.0, 0.0, 0.0))
    api.set_duration(0.5)
"""

_STATIC_FIXED_SPHERE = """\
def build_scene(api):
    api.add_sphere('ball', pos=(0.0, 0.0, 1.0), radius=0.1,
                   mass=1.0, fixed=True)
    api.set_duration(0.5)
"""


@pytest.fixture(scope="module")
def moving_sim_state():
    result = run_executor(_MOVING_SPHERE, stage=1, seed=0, sim_timeout_s=10.0)
    assert result.ok, result.error_message
    return result.sim_state


def test_quality_metrics_basic_motion(moving_sim_state) -> None:
    metrics = compute_quality_metrics(moving_sim_state)

    assert metrics["dynamic_body_count"] == 1
    assert metrics["body_count"] >= 1
    assert metrics["max_displacement"] > 0.05
    assert metrics["max_speed"] > 0.0
    assert metrics["n_frames"] == 32
    assert metrics["video_complete"] is True


def test_quality_score_is_bounded(moving_sim_state) -> None:
    report = evaluate_quality(moving_sim_state)

    assert 0.0 <= report.score <= 1.0
    assert "single_body" in report.tags
    assert report.task_suitability["forward"] is True
    assert report.task_suitability["inverse"] is True


def test_quality_rejects_static_scene() -> None:
    result = run_executor(_STATIC_FIXED_SPHERE, stage=1, seed=0, sim_timeout_s=10.0)
    assert result.sim_state is not None

    report = evaluate_quality(result.sim_state)

    assert report.hard_pass is False
    assert "max displacement below trivial threshold" in report.reject_reasons
