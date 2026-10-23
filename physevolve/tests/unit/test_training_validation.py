"""Tests for training post-run validation checks."""

from __future__ import annotations

from pathlib import Path

from physevolve.smoke.io import append_jsonl, write_json
from physevolve.training.training_validation import validate_training_run


def test_training_validation_accepts_complete_finite_run(tmp_path: Path) -> None:
    _write_summary(tmp_path, ppo_update_count=20, reward_tensor_finite=True)
    write_json(tmp_path / "run_status.json", {"completed": True, "error": None})
    (tmp_path / "global_step_20" / "actor").mkdir(parents=True)
    _append_metric_row(tmp_path, step=1)

    report = validate_training_run(tmp_path, expected_steps=20)

    assert report["ok"] is True
    assert report["errors"] == []


def test_training_validation_hard_fails_incomplete_or_nonfinite_run(tmp_path: Path) -> None:
    _write_summary(tmp_path, ppo_update_count=19, reward_tensor_finite=False)
    write_json(tmp_path / "run_status.json", {"completed": False, "error": "boom"})
    _append_metric_row(tmp_path, step=1, actor_loss="nan")

    report = validate_training_run(tmp_path, expected_steps=20)

    assert report["ok"] is False
    assert any("run_status has error" in error for error in report["errors"])
    assert any("reward_tensor_finite" in error for error in report["errors"])
    assert any("ppo_update_count=19" in error for error in report["errors"])
    assert any("missing actor checkpoint" in error for error in report["errors"])
    assert any("actor/loss" in error for error in report["errors"])


def _write_summary(
    path: Path,
    *,
    ppo_update_count: int,
    reward_tensor_finite: bool,
) -> None:
    write_json(
        path / "summary.json",
        {
            "completed": ppo_update_count >= 20,
            "executed": True,
            "ppo_update_count": ppo_update_count,
            "reward_tensor_finite": reward_tensor_finite,
        },
    )


def _append_metric_row(path: Path, *, step: int, actor_loss: object = 0.0) -> None:
    append_jsonl(
        path / "metrics.jsonl",
        {
            "step": step,
            "time/step_total_s": 1.0,
            "reward_batch": {"reward_tensor_finite": True, "reward_mean": 0.5},
            "online": {
                "all_rewards_finite": True,
                "n_executor_ok": 1,
                "selected_solver_groups": 1,
                "answer_parse_rate": 1.0,
                "schema_ok_rate": 1.0,
                "solver_backfill_shortfall": 0,
                "planned_solver_groups": 1,
                "unique_scene_ratio": 1.0,
            },
            "update": {
                "actor/loss": actor_loss,
                "actor/pg_loss": 0.0,
                "actor/kl_loss": 0.0,
                "actor/grad_norm": 1.0,
            },
        },
    )
