"""Post-run validation checks for PhysEvolve PPO runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from physevolve.smoke.io import read_jsonl, write_json

_FINITE_UPDATE_KEYS = (
    "actor/loss",
    "actor/pg_loss",
    "actor/kl_loss",
    "actor/grad_norm",
)


def validate_training_run(
    run_dir: Path,
    *,
    expected_steps: int = 20,
    slow_step_s: float = 900.0,
    kl_explosion_threshold: float = 1.0,
) -> dict[str, Any]:
    """Validate a completed training run and return hard errors plus warnings."""

    run_path = Path(run_dir)
    summary = _read_json(run_path / "summary.json")
    run_status = _read_json(run_path / "run_status.json")
    rows = read_jsonl(run_path / "metrics.jsonl")

    errors: list[str] = []
    warnings: list[str] = []

    if run_status.get("error"):
        errors.append(f"run_status has error: {run_status['error']}")

    if not bool(summary.get("reward_tensor_finite", False)):
        errors.append("summary reward_tensor_finite is false")
    for row in rows:
        step = int(row.get("step", 0) or 0)
        reward_batch = row.get("reward_batch", {})
        online = row.get("online", {})
        if reward_batch.get("reward_tensor_finite") is False:
            errors.append(f"step {step}: reward tensor is non-finite")
        if online.get("all_rewards_finite") is False:
            errors.append(f"step {step}: online rewards are non-finite")
        update = row.get("update", {})
        for key in _FINITE_UPDATE_KEYS:
            if key not in update:
                errors.append(f"step {step}: missing update metric {key}")
                continue
            if not _is_finite_number(update[key]):
                errors.append(f"step {step}: update metric {key} is non-finite")

    ppo_update_count = int(summary.get("ppo_update_count", 0) or 0)
    if ppo_update_count < int(expected_steps):
        errors.append(
            f"ppo_update_count={ppo_update_count} is less than expected {int(expected_steps)}"
        )

    checkpoint_actor = run_path / f"global_step_{int(expected_steps)}" / "actor"
    if not checkpoint_actor.exists():
        errors.append(f"missing actor checkpoint: {checkpoint_actor.as_posix()}")

    _add_soft_warnings(
        rows,
        warnings,
        slow_step_s=float(slow_step_s),
        kl_explosion_threshold=float(kl_explosion_threshold),
    )

    return {
        "ok": not errors,
        "expected_steps": int(expected_steps),
        "metrics_rows": len(rows),
        "checkpoint_actor_path": checkpoint_actor.as_posix(),
        "errors": errors,
        "warnings": warnings,
    }


def write_training_validation_report(
    run_dir: Path,
    *,
    expected_steps: int = 20,
) -> dict[str, Any]:
    """Validate a training run and write ``training_validation.json``."""

    report = validate_training_run(run_dir, expected_steps=expected_steps)
    write_json(Path(run_dir) / "training_validation.json", report)
    return report


def _add_soft_warnings(
    rows: list[dict[str, Any]],
    warnings: list[str],
    *,
    slow_step_s: float,
    kl_explosion_threshold: float,
) -> None:
    if not rows:
        warnings.append("metrics.jsonl has no training steps")
        return

    executor_zero_steps = [
        int(row.get("step", 0) or 0)
        for row in rows
        if int(row.get("online", {}).get("n_executor_ok", 0) or 0) == 0
    ]
    if len(executor_zero_steps) >= max(3, len(rows) // 4):
        warnings.append(
            f"executor OK count is zero for many steps: {executor_zero_steps}"
        )

    selected_zero_steps = [
        int(row.get("step", 0) or 0)
        for row in rows
        if int(row.get("online", {}).get("selected_solver_groups", 0) or 0) == 0
    ]
    if selected_zero_steps:
        warnings.append(f"selected solver groups are zero: {selected_zero_steps}")

    answer_rates = [_float(row.get("online", {}).get("answer_parse_rate")) for row in rows]
    if answer_rates and all(rate == 0.0 for rate in answer_rates):
        warnings.append("answer_parse_rate is zero for all steps")

    schema_rates = [_float(row.get("online", {}).get("schema_ok_rate")) for row in rows]
    if schema_rates and all(rate == 0.0 for rate in schema_rates):
        warnings.append("schema_ok_rate is zero for all steps")

    high_shortfall_steps = []
    for row in rows:
        online = row.get("online", {})
        shortfall = int(online.get("solver_backfill_shortfall", 0) or 0)
        planned = max(1, int(online.get("planned_solver_groups", 0) or 0))
        if shortfall / planned >= 0.5:
            high_shortfall_steps.append(int(row.get("step", 0) or 0))
    if len(high_shortfall_steps) >= max(2, len(rows) // 4):
        warnings.append(f"solver_backfill_shortfall remains high: {high_shortfall_steps}")

    unique_ratios = [_float(row.get("online", {}).get("unique_scene_ratio")) for row in rows]
    if unique_ratios and min(unique_ratios) < 0.25:
        warnings.append(f"unique_scene_ratio is low; min={min(unique_ratios):.3f}")

    reward_means = [_float(row.get("reward_batch", {}).get("reward_mean")) for row in rows]
    if len(reward_means) > 1 and max(reward_means) - min(reward_means) <= 1.0e-9:
        warnings.append("reward_mean is constant across steps")

    kl_values = [
        _float(row.get("update", {}).get(key))
        for row in rows
        for key in ("actor/kl_loss", "actor/ppo_kl")
        if row.get("update", {}).get(key) is not None
    ]
    if kl_values and max(kl_values) > kl_explosion_threshold:
        warnings.append(f"KL may be exploding; max={max(kl_values):.6f}")

    step_times = [_float(row.get("time/step_total_s")) for row in rows]
    if step_times and max(step_times) > slow_step_s:
        warnings.append(f"step time is slow; max={max(step_times):.3f}s")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


__all__ = ["validate_training_run", "write_training_validation_report"]
