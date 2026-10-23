"""CPU diagnostic dryrun for the naive (``stage=1``) baseline.

This launcher does NOT touch veRL, vLLM, FSDP, or any GPU. It wires up the
full naive closed loop for all three dense task modes:

  Adaptive Proposer prompt (skill picked by learnability tent)
    -> mock backend (deterministic Proposer text)
    -> executor.run_executor(stage=1)
    -> PhysEvolveOnlineStepBuilder Phase 2..7 (Forward + Inverse + Parameter)
    -> dense reward + naive_diagnostics (exact-match + collapse)
    -> token-level reward tensor + step_summary.jsonl

The mock Solver returns a *fixed, schema-valid* answer per mode that is
deliberately a summary-statistic collapse for Inverse (zero-velocity bin 3)
and Parameter (mid-rung level 3), so the dryrun exercises both the
exact-match and the collapse-detection paths in
:mod:`physevolve.verification.naive_diagnostics`, not just the plumbing.

The script exits with code ``0`` only when ALL of the following hold across
the dryrun steps:

  1. Every step's reward tensor is finite.
  2. ``exact_match_accuracy_by_task`` is present and every value is in
     ``[0, 1]``.
  3. ``collapse_distribution_by_task`` is present and non-empty (at least
     one task logged a collapse label).
  4. ``n_intervention_operations == 0`` and ``n_counterfactual_items == 0``
     — the naive single-world interface has no branch / counterfactual.
  5. ``naive_interface == "single_world_no_intervention"``.
  6. ``mean_reward_by_task`` is present and non-empty.

Use this script before any GPU launch to confirm the naive diagnostic
plumbing is healthy. It is the CPU counterpart of
``scripts/train/naive_2step_smoke.sh`` (GPU) and ``scripts/train/naive.sh``
(GPU, 100-step).

Usage::

  python scripts/smoke/run_naive_diagnostic_dryrun.py \\
    --scenes-per-batch 2 --group-size 2 --max-steps 3 \\
    --out-root /tmp/naive_diag
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physevolve.executor import run_executor  # noqa: E402
from physevolve.inference.hf_backend import GenerationResult  # noqa: E402
from physevolve.smoke.run_config import SmokeGenerationConfig  # noqa: E402
from physevolve.training.online_step_builder import (  # noqa: E402
    PhysEvolveOnlineStepBuilder,
    PhysEvolveOnlineStepBuilderConfig,
)


# Synthetic naive Proposer output: a single launched sphere. The executor
# simulates it (stage=1) and produces a real SimState so the dense scorers
# and the naive diagnostics run against genuine ground truth.
_PROPOSER_TEXT = textwrap.dedent(
    """\
    Plan: a small sphere is launched forward and upward; predict its fate.
    <propose>
    def build_scene(api):
        api.add_sphere(
            "ball",
            pos=(-0.6, 0.0, 0.30),
            radius=0.10,
            mass=0.8,
            rgba=(0.85, 0.45, 0.25, 1.0),
            friction=0.5,
            restitution=0.4,
            fixed=False,
        )
        api.set_velocity("ball", linear=(1.5, 0.0, 1.5), angular=(0.0, 0.0, 0.0))
        api.set_duration(1.5)
    </propose>
    """
).strip()


# Fixed per-mode Solver answers. Inverse is a zero-velocity collapse (all
# bins = 3 = 0.0 m/s); Parameter is the mid-rung (level 3). Forward is a
# plausible-but-wrong grid. These are chosen so the dryrun exercises the
# exact-match True (parameter), exact-match False + collapse (inverse), and
# exact-match False (forward) branches of naive_diagnostics.
_SOLVER_TEXT_BY_PROBLEM_TYPE: dict[str, str] = {
    "pred_forward": '<answer>{"ball": {"grid": [8, 8, 4], "status": "stable"}}</answer>',
    "pred_inverse": '<answer>{"ball": {"initial_vel_grid": [3, 3, 3]}}</answer>',
    "pred_parameter": '<answer>{"ball": {"mass_level": 3, "friction_level": 3}}</answer>',
}


@dataclass
class _NaiveMockBackend:
    """Mock backend routing Proposer vs Solver by rollout metadata.

    Proposer rows return ``_PROPOSER_TEXT``; Solver rows return the
    per-``problem_type`` fixed answer above. Both queues are infinite per
    step — we mint a fresh backend per builder call.
    """

    def __enter__(self) -> "_NaiveMockBackend":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def generate_batch(self, specs):
        results: list[GenerationResult] = []
        for spec in specs:
            meta = spec.rollout_metadata or {}
            role = meta.get("role", "proposer")
            if role == "proposer":
                text = _PROPOSER_TEXT
            else:
                text = _SOLVER_TEXT_BY_PROBLEM_TYPE.get(
                    meta.get("problem_type", ""),
                    _SOLVER_TEXT_BY_PROBLEM_TYPE["pred_forward"],
                )
            results.append(
                GenerationResult(
                    text=text,
                    metadata={
                        "generated_token_count": 5,
                        "video_present": spec.video is not None,
                    },
                    generated_text=text,
                    prompt_text=spec.prompt,
                    prompt_tail=spec.prompt[-32:],
                )
            )
        return results


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _gather_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Pick the subset of metrics that matters for the dryrun status."""

    return {
        "executor_call_count": metrics.get("executor_call_count"),
        "n_executor_ok": metrics.get("n_executor_ok"),
        "all_rewards_finite": metrics.get("all_rewards_finite"),
        "mean_reward_by_task": metrics.get("mean_reward_by_task"),
        "exact_match_accuracy_by_task": metrics.get("exact_match_accuracy_by_task"),
        "collapse_distribution_by_task": metrics.get("collapse_distribution_by_task"),
        "n_intervention_operations": metrics.get("n_intervention_operations"),
        "n_counterfactual_items": metrics.get("n_counterfactual_items"),
        "naive_interface": metrics.get("naive_interface"),
        "expected_problem_types": metrics.get("expected_problem_types"),
        "physevolve_problem_types_seen": metrics.get("physevolve_problem_types_seen"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/tmp/naive_diag"),
        help="root directory for dryrun artifacts (no GPU paths used)",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--scenes-per-batch", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--cold-start-uniform-steps", type=int, default=8)
    parser.add_argument("--sandbox-timeout-s", type=float, default=5.0)
    parser.add_argument("--sim-timeout-s", type=float, default=30.0)
    args = parser.parse_args(argv)

    run_id = args.run_id or f"naive_diag_{int(time.time())}"
    run_dir = args.out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = PhysEvolveOnlineStepBuilderConfig(
        scenes_per_batch=int(args.scenes_per_batch),
        group_size=int(args.group_size),
        task_modes=("forward", "inverse", "parameter"),
        response_length=512,
        sandbox_timeout_s=float(args.sandbox_timeout_s),
        sim_timeout_s=float(args.sim_timeout_s),
        cold_start_uniform_steps=int(args.cold_start_uniform_steps),
        generation=SmokeGenerationConfig(
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            stop_sequences=("</propose>", "</answer>"),
        ),
    )

    builder = PhysEvolveOnlineStepBuilder(
        config,
        run_dir=run_dir,
        backend_factory=lambda _cfg: _NaiveMockBackend(),
        executor_fn=run_executor,
    )

    summaries: list[dict[str, Any]] = []
    overall_ok = True
    failure_reasons: list[str] = []

    for step_idx in range(1, int(args.max_steps) + 1):
        step = builder(step_idx)
        summary = _gather_metrics(step.metrics)
        summary["step"] = step_idx
        summary["reward_records"] = int(step.reward_batch.tensor.shape[0])
        summaries.append(summary)

        if not summary["all_rewards_finite"]:
            overall_ok = False
            failure_reasons.append(f"step {step_idx}: reward tensor not finite")

        ema = summary.get("exact_match_accuracy_by_task") or {}
        if not ema:
            overall_ok = False
            failure_reasons.append(f"step {step_idx}: exact_match_accuracy_by_task missing")
        else:
            for task, rate in ema.items():
                if not _is_finite(rate) or not (0.0 <= float(rate) <= 1.0):
                    overall_ok = False
                    failure_reasons.append(
                        f"step {step_idx}: exact_match_accuracy_by_task[{task}]={rate} not in [0,1]"
                    )

        collapse = summary.get("collapse_distribution_by_task") or {}
        if not collapse:
            overall_ok = False
            failure_reasons.append(
                f"step {step_idx}: collapse_distribution_by_task missing/empty"
            )

        if summary.get("n_intervention_operations") != 0:
            overall_ok = False
            failure_reasons.append(
                f"step {step_idx}: n_intervention_operations="
                f"{summary.get('n_intervention_operations')!r} (expected 0)"
            )
        if summary.get("n_counterfactual_items") != 0:
            overall_ok = False
            failure_reasons.append(
                f"step {step_idx}: n_counterfactual_items="
                f"{summary.get('n_counterfactual_items')!r} (expected 0)"
            )
        if summary.get("naive_interface") != "single_world_no_intervention":
            overall_ok = False
            failure_reasons.append(
                f"step {step_idx}: naive_interface="
                f"{summary.get('naive_interface')!r} (expected single_world_no_intervention)"
            )

        mrt = summary.get("mean_reward_by_task") or {}
        if not mrt:
            overall_ok = False
            failure_reasons.append(f"step {step_idx}: mean_reward_by_task missing/empty")

    # Confirm the collapse path was actually exercised (inverse zero-velocity).
    last_collapse = summaries[-1].get("collapse_distribution_by_task") if summaries else {}
    inverse_collapse = (last_collapse or {}).get("pred_inverse", {})
    if not inverse_collapse:
        overall_ok = False
        failure_reasons.append(
            "last step: pred_inverse collapse_distribution empty (zero-velocity "
            "collapse path not exercised)"
        )

    status_payload: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": run_dir.as_posix(),
        "ok": overall_ok,
        "failure_reasons": failure_reasons,
        "last_step_collapse_distribution_by_task": last_collapse,
        "last_step_exact_match_accuracy_by_task": (
            summaries[-1].get("exact_match_accuracy_by_task") if summaries else {}
        ),
        "steps": summaries,
    }
    (run_dir / "dryrun_status.json").write_text(
        json.dumps(status_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(status_payload, indent=2))
    return 0 if overall_ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI surface
    raise SystemExit(main())
