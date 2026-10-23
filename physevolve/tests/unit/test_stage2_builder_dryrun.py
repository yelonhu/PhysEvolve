"""Stage-2 VQA step builder CPU dryrun test (docs/architecture.md).

Walks the full Stage-2 self-play loop with a mock generation backend
and a fake executor that returns OK results with a synthetic
``vqa_payload``. This stays on CPU and runs in well under a second so
it can guard the Stage-2 pipeline at every commit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from physevolve.engine.schema import (
    BodyDef,
    BodyStatus,
    BodyTrajectory,
    EnvConfig,
    GeomType,
    JointType,
    SceneDef,
    SimState,
    compute_scene_id,
)
from physevolve.executor.result import make_failure, make_ok
from physevolve.engine.failure import FailureReason
from physevolve.inference.hf_backend import GenerationResult
from physevolve.smoke.stage2_mock_backend import build_proposer_text_for_scene_kind
from physevolve.smoke.run_config import SmokeGenerationConfig
from physevolve.training.online_step_builder import GenerationSpec
from physevolve.training.online_step_builder_stage2 import (
    PhysEvolveStage2StepBuilder,
    PhysEvolveStage2StepBuilderConfig,
    _select_stage2_vqa_payloads,
)


# ─── Mocks ───────────────────────────────────────────────────────────


@dataclass
class _Stage2FakeBackend:
    """Round-robin fake backend with separate proposer + solver queues.

    The Stage-2 builder issues one batched proposer call and one batched
    solver call per step. Tests usually want to inject distinct text
    pools per phase so we route by ``role`` in ``rollout_metadata``.
    """

    proposer_outputs: list[str]
    solver_outputs: list[str]
    calls: list[list[GenerationSpec]] = field(default_factory=list, init=False)

    def __enter__(self):
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def generate_batch(self, specs):
        spec_list = list(specs)
        self.calls.append(spec_list)
        results: list[GenerationResult] = []
        for spec in spec_list:
            role = (spec.rollout_metadata or {}).get("role", "proposer")
            if role == "proposer":
                text = self.proposer_outputs.pop(0)
            else:
                text = self.solver_outputs.pop(0)
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


def _make_sim_state() -> SimState:
    body = BodyDef(
        name="ball",
        geom_type=GeomType.SPHERE,
        pos=(0.0, 0.0, 1.0),
        size=(0.1,),
        mass=1.0,
        rgba=(1.0, 0.0, 0.0, 1.0),
        friction_slide=0.2,
        restitution=0.0,
        joint_type=JointType.FREE,
    )
    scene = SceneDef(
        bodies=(body,),
        joints=(),
        cameras=(),
        env=EnvConfig(duration=8.0, n_frames=32),
        workspace_bounds=(1.5, 1.5, 3.0),
    )
    positions = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (4, 1))
    orientations = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (4, 1))
    trajectory = BodyTrajectory(
        name="ball",
        positions=positions,
        orientations=orientations,
        lin_velocities=np.zeros((4, 3), dtype=np.float64),
        ang_velocities=np.zeros((4, 3), dtype=np.float64),
        final_status=BodyStatus.STABLE,
    )
    return SimState(
        scene_id=compute_scene_id(scene, seed=1),
        scene_def=scene,
        trajectories={"ball": trajectory},
        collisions=(),
        video_frames=np.zeros((1, 2, 8, 8, 3), dtype=np.uint8),
        camera_names=("main",),
        sim_walltime_s=0.01,
        initial_positions={"ball": positions[0]},
        initial_lin_velocities={"ball": np.zeros(3, dtype=np.float64)},
    )


def _fake_executor_ok_with_vqa(*_args: Any, **_kwargs: Any):
    sim_state = _make_sim_state()
    return make_ok(
        sim_state.scene_id,
        sim_state,
        ("Where is the ball?", ["here", "there", "up", "down"], 2),
        walltime_s=0.01,
    )


def _fake_executor_ok_with_repair(*_args: Any, **_kwargs: Any):
    sim_state = _make_sim_state()
    return make_ok(
        sim_state.scene_id,
        sim_state,
        ("Where is the ball?", ["here", "there", "up", "down"], 2),
        walltime_s=0.01,
        repair_trace={
            "applied": True,
            "tier": "geometry",
            "reward_discount": 0.75,
            "actions": [
                {
                    "kind": "repair_body_pos",
                    "target": "ball",
                    "tier": "geometry",
                    "detail": {},
                }
            ],
        },
    )


def _fake_executor_ok_with_many_vqa(*_args: Any, **_kwargs: Any):
    sim_state = _make_sim_state()
    variant = replace(
        sim_state,
        scene_id="variant_scene_xyz",
        video_frames=np.ones((1, 2, 8, 8, 3), dtype=np.uint8),
    )
    payloads = [
        {
            "payload": ["Q0 preferred?", ["A", "B", "C", "D"], 2],
            "spec": {
                "family": "throwing_final_destination",
                "phrasing": "forward",
            },
        },
        {
            "payload": ["Q1 judgment?", ["A", "B", "C", "D"], 0],
            "spec": {
                "family": "throwing_current_run_judgment",
                "phrasing": "yes_no",
            },
        },
        {
            "payload": ["Q2 cf?", ["A", "B", "C", "D"], 1],
            "spec": {
                "family": "counterfactual_destination",
                "phrasing": "counterfactual",
            },
        },
        {
            "payload": ["Q3 inverse?", ["A", "B", "C", "D"], 3],
            "spec": {
                "family": "inverse_intervention",
                "phrasing": "inverse",
            },
        },
        {
            "payload": ["Q4 variant?", ["A", "B", "C", "D"], 2],
            "spec": {
                "family": "throwing_final_destination",
                "phrasing": "forward",
            },
            "sim_state": variant,
        },
        {
            "payload": ["Q5 extra?", ["A", "B", "C", "D"], 0],
            "spec": {
                "family": "throwing_current_run_judgment",
                "phrasing": "yes_no",
            },
        },
    ]
    return make_ok(
        sim_state.scene_id,
        sim_state,
        tuple(payloads[0]["payload"]),
        walltime_s=0.01,
        vqa_payloads=payloads,
    )


def _fake_executor_ok_with_wrong_family(*_args: Any, **_kwargs: Any):
    sim_state = _make_sim_state()
    payloads = [
        {
            "payload": ["Q motion?", ["A", "B", "C", "D"], 0],
            "spec": {
                "family": "motion_state_change",
                "phrasing": "forward",
            },
        }
    ]
    return make_ok(
        sim_state.scene_id,
        sim_state,
        tuple(payloads[0]["payload"]),
        walltime_s=0.01,
        vqa_payloads=payloads,
    )


def _fake_executor_ok_with_relation_vqa(*_args: Any, **_kwargs: Any):
    sim_state = _make_sim_state()
    payloads = [
        {
            "payload": ["Q relation?", ["A", "B", "C", "D"], 1],
            "spec": {
                "family": "collision_final_relation",
                "phrasing": "between",
                "expected_relation": "no_contact",
            },
        },
        {
            "payload": ["Q collide?", ["Yes", "No", "Sometimes", "Uncertain"], 1],
            "spec": {
                "family": "collision_final_relation",
                "phrasing": "binary_collide",
                "expected_relation": "no_contact",
            },
        },
    ]
    return make_ok(
        sim_state.scene_id,
        sim_state,
        tuple(payloads[0]["payload"]),
        walltime_s=0.01,
        vqa_payloads=payloads,
    )


def _fake_executor_failed_vqa(*_args: Any, **_kwargs: Any):
    return make_failure(
        FailureReason.FAILED_VQA,
        scene_id="",
        err=ValueError("get_vqa returned None"),
        walltime_s=0.01,
    )


def _fake_executor_overlap(*_args: Any, **_kwargs: Any):
    return make_failure(
        FailureReason.FAILED_OVERLAP,
        scene_id="",
        err=ValueError("initial overlap"),
        walltime_s=0.01,
    )


# ─── Static templates the fake Proposer / Solver emit ───────────────


_PROPOSER_OK = (
    "<think>plan</think><propose>\n"
    "def build_scene(api):\n"
    "    api.add_sphere('ball', pos=(0,0,1), radius=0.1, mass=1.0, "
    "rgba=(1,0,0,1), friction=0.2, restitution=0.0, fixed=False)\n"
    "\n"
    "def get_vqa(sim_state):\n"
    "    return ('Where is the ball?', ['here', 'there', 'up', 'down'], 2)\n"
    "</propose>"
)
_SOLVER_CORRECT = "<think>I think the ball lands up.</think>\n<answer>C</answer>"
_SOLVER_WRONG = "<think>guess A</think>\n<answer>A</answer>"
_PROPOSER_NO_PROPOSE = "<think>oops</think>I forgot the propose block"
_PROPOSER_NEAR_DUP = (
    "<think>plan</think><propose>\n"
    "def build_scene(api):\n"
    "    api.add_sphere('ball', pos=(0.02,0,1), radius=0.1, mass=1.0, "
    "rgba=(1,0,0,1), friction=0.2, restitution=0.0, fixed=False)\n"
    "\n"
    "def get_vqa(sim_state):\n"
    "    return ('Where is the ball?', ['here', 'there', 'up', 'down'], 2)\n"
    "</propose>"
)


# ─── Tests ───────────────────────────────────────────────────────────


def test_stage2_dryrun_full_pipeline_correct_solver(tmp_path: Path) -> None:
    """One scene, one solver rollout per group; solver picks the correct C."""

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK, _PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT, _SOLVER_CORRECT],
    )

    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=2,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )

    step = builder(1)

    assert step.step == 1
    assert step.reward_batch.metrics["reward_tensor_finite"] is True
    assert step.metrics["stage"] == 2
    assert step.metrics["executor_call_count"] == 2
    assert step.metrics["n_executor_ok"] == 2
    assert step.metrics["vqa_payload_present_count"] == 2
    assert step.metrics["vqa_payload_present_rate"] == 1.0
    # Solver always picked C (idx 2), which is the GT.
    assert step.metrics["mean_reward_by_task"]["pred_vqa"] == 1.0
    assert step.metrics["expected_problem_types"] == ["gen_vqa", "pred_vqa"]
    assert "gen_vqa" in step.metrics["physevolve_problem_types_seen"]
    assert "pred_vqa" in step.metrics["physevolve_problem_types_seen"]
    assert step.metrics["stage2_copy_diagnostic_count"] == 2
    assert step.metrics["stage2_copy_recent_repeat_count"] == 1
    assert step.metrics["stage2_copy_exact_fewshot_count"] == 0
    # VQA correct-idx distribution is reported (the executor returns
    # correct_idx=2 → letter "C" 100% of the time).
    dist = step.metrics["vqa_correct_idx_distribution"]
    assert dist["C"] == 1.0
    assert dist["A"] == 0.0


def test_stage2_dryrun_wrong_solver_gets_zero_reward(tmp_path: Path) -> None:
    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK],
        solver_outputs=[_SOLVER_WRONG],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )
    step = builder(1)
    assert step.metrics["mean_reward_by_task"]["pred_vqa"] == 0.0


def test_stage2_dryrun_reports_repair_metrics(tmp_path: Path) -> None:
    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_repair,
    )
    step = builder(1)
    assert step.metrics["stage2_repair_applied_count"] == 1
    assert step.metrics["stage2_repair_applied_ok_count"] == 1
    assert step.metrics["stage2_repair_tier_counts"]["geometry"] == 1
    assert step.metrics["stage2_repair_action_counts"]["repair_body_pos"] == 1


def test_stage2_dryrun_forwards_camera_depth_bypass_flag(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _capturing_executor(*args: Any, **kwargs: Any):
        seen.update(kwargs)
        return _fake_executor_ok_with_vqa(*args, **kwargs)

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            question_families=("collision_final_state",),
            camera_depth_bypass_enabled=True,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_capturing_executor,
    )

    step = builder(1)

    assert seen["camera_depth_bypass_enabled"] is True
    assert step.metrics["stage2_camera_depth_bypass_enabled"] is True


def test_stage2_dryrun_applies_branch_family_weight_override(tmp_path: Path) -> None:
    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_NO_PROPOSE],
        solver_outputs=[],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            question_families=(
                "throwing_final_destination",
                "throwing_current_run_judgment",
                "counterfactual_destination",
                "inverse_intervention",
            ),
            branch_family_weight_multiplier=1.2,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )

    step = builder(1)

    assert step.metrics["stage2_branch_family_weight_multiplier"] == 1.2
    pick_weights = step.metrics["vqa_family_pick_weights"]
    assert (
        pick_weights["counterfactual_destination"]
        + pick_weights["inverse_intervention"]
    ) == pytest.approx(0.375)


def test_stage2_dryrun_proposer_failure_path(tmp_path: Path) -> None:
    """A Proposer without ``<propose>`` block must not crash the step.

    The reward record gets the FAILED_SYNTAX penalty and no Solver rollout
    is issued for that scene.
    """

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_NO_PROPOSE],
        solver_outputs=[],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )
    step = builder(1)
    assert step.metrics["n_executor_ok"] == 0
    assert step.metrics["executor_call_count"] == 0
    # The reward tensor still has a row for the failed proposer.
    assert step.reward_batch.tensor.shape[0] == 1
    failures = step.metrics["failure_reason_distribution"]
    assert failures.get("syntax", 0) >= 1


def test_stage2_dryrun_failed_vqa_does_not_crash(tmp_path: Path) -> None:
    """Executor returning FAILED_VQA must be handled cleanly."""

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK],
        solver_outputs=[],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_failed_vqa,
    )
    step = builder(1)
    assert step.metrics["n_executor_ok"] == 0
    assert step.metrics["vqa_payload_present_count"] == 0
    assert step.metrics["vqa_payload_present_rate"] == 0.0
    assert step.metrics["failure_reason_distribution"].get("vqa", 0) >= 1


def test_stage2_dryrun_writes_step_summary_jsonl(tmp_path: Path) -> None:
    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )
    step = builder(1)
    step_dir = Path(step.artifacts["step_dir"])
    assert (step_dir / "raw_proposer_outputs.jsonl").exists()
    assert (step_dir / "executor_results.jsonl").exists()
    assert (step_dir / "solver_outputs.jsonl").exists()
    assert (step_dir / "step_summary.jsonl").exists()
    snapshots = list((step_dir / "prompt_snapshots").glob("proposer__stage2__*.txt"))
    assert snapshots, f"expected per-skill prompt snapshot, got {snapshots!r}"

    rows = [
        json.loads(line)
        for line in (step_dir / "reward_records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    pred_rows = [r for r in rows if r.get("problem_type") == "pred_vqa"]
    proposer_rows = [r for r in rows if r.get("problem_type") == "gen_vqa"]
    assert pred_rows, rows
    assert proposer_rows, rows
    # intrinsic/novelty shaping is OFF by default, so the
    # proposer rows carry NO intrinsic breakdown and the summary flags it off.
    assert not any("intrinsic/motion" in r for r in proposer_rows)
    assert step.metrics["intrinsic_enabled"] is False


def test_stage2_dryrun_intrinsic_opt_in(tmp_path: Path) -> None:
    """Enabling intrinsic restores the per-component breakdown on proposer rows."""

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            intrinsic_enabled=True,
            intrinsic_weight=0.1,
            intrinsic_component_weights=(0.0, 0.0, 0.0, 1.0, 0.0),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )
    step = builder(1)
    step_dir = Path(step.artifacts["step_dir"])
    rows = [
        json.loads(line)
        for line in (step_dir / "reward_records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    proposer_rows = [r for r in rows if r.get("problem_type") == "gen_vqa"]
    assert any("intrinsic/motion" in r for r in proposer_rows)
    assert step.metrics["intrinsic_enabled"] is True
    assert step.metrics["intrinsic_weight"] == 0.1
    assert step.metrics["intrinsic_component_weights"] == [0.0, 0.0, 0.0, 1.0, 0.0]


def _run_stage2_step(tmp_path: Path, chunk_size: int):
    """Run one Stage-2 step with a fixed proposer/solver script.

    ``scenes_per_batch=2 * group_size=2`` yields 4 OK scenes, each deriving one
    question, so the solver phase produces 8 video-bearing rows -- enough for a
    sub-batch chunk size to actually split the dispatch.
    """

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK] * 4,
        solver_outputs=[
            _SOLVER_CORRECT,
            _SOLVER_WRONG,
            _SOLVER_CORRECT,
            _SOLVER_WRONG,
            _SOLVER_CORRECT,
            _SOLVER_WRONG,
            _SOLVER_CORRECT,
            _SOLVER_WRONG,
        ],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=2,
            group_size=2,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            solver_rollout_chunk_size=chunk_size,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )
    return builder(1)


def test_stage2_solver_chunking_matches_unchunked(tmp_path: Path) -> None:
    """Chunked solver dispatch must not change the reward batch.

    The chunk size only governs how many rows are dispatched per
    ``generate_batch`` call; the rows, their order, scoring, and GRPO grouping
    are identical, so the reward tensor and per-task means must match the
    single-batch path exactly.
    """

    unchunked = _run_stage2_step(tmp_path / "unchunked", chunk_size=0)
    chunked = _run_stage2_step(tmp_path / "chunked", chunk_size=3)

    assert (
        chunked.reward_batch.tensor.shape == unchunked.reward_batch.tensor.shape
    )
    # 4 proposer rows + 8 solver rows.
    assert chunked.reward_batch.tensor.shape[0] == 12
    assert (
        chunked.metrics["mean_reward_by_task"]
        == unchunked.metrics["mean_reward_by_task"]
    )
    assert (
        chunked.metrics["n_solver_rollouts_per_batch"]
        == unchunked.metrics["n_solver_rollouts_per_batch"]
        == 8
    )


def test_stage2_limits_solver_questions_per_scene(tmp_path: Path) -> None:
    """Per-scene cap keeps multi-GT scenes representative, not exhaustive."""

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK, _PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT] * 6,
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=2,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            question_families=("throwing_final_destination",),
            max_solver_questions_per_scene=3,
            max_solver_groups_per_step=0,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_many_vqa,
    )

    step = builder(1)

    assert step.metrics["vqa_question_total_raw"] == 12
    assert step.metrics["vqa_question_total_before_step_cap"] == 6
    assert step.metrics["vqa_question_total_selected"] == 6
    assert step.metrics["n_solver_groups_available"] == 6
    assert step.metrics["n_solver_groups_selected"] == 6
    assert step.metrics["n_solver_rollouts_per_batch"] == 6
    assert step.metrics["solver_selection_variant_count"] == 2
    assert step.metrics["vqa_questions_per_scene"] == 6.0
    assert step.metrics["vqa_questions_per_scene_selected"] == 3.0
    assert step.metrics["stage2_raw_payload_by_family"] == {
        "counterfactual_destination": 2,
        "inverse_intervention": 2,
        "throwing_current_run_judgment": 4,
        "throwing_final_destination": 4,
    }
    assert step.metrics["stage2_preferred_family_missing_by_family"] == {}
    assert step.metrics["solver_selection_by_family"] == {
        "counterfactual_destination": 2,
        "throwing_final_destination": 4,
    }


def test_stage2_per_scene_cap_keeps_preferred_relation_family() -> None:
    """A sampled relation question survives even under a one-question cap."""

    payloads = [
        {
            "payload": ["Q state?", ["A", "B", "C", "D"], 0],
            "spec": {"family": "collision_final_state", "phrasing": "forward"},
        },
        {
            "payload": ["Q relation?", ["A", "B", "C", "D"], 1],
            "spec": {
                "family": "collision_final_relation",
                "phrasing": "between",
                "expected_relation": "no_contact",
            },
        },
        {
            "payload": ["Q depth?", ["A", "B", "C", "D"], 2],
            "spec": {"family": "camera_depth_closest_object", "phrasing": "static"},
        },
    ]

    selected = _select_stage2_vqa_payloads(
        payloads,
        preferred_family="collision_final_relation",
        max_questions=1,
        seed_key="relation-budget-regression",
    )

    assert len(selected) == 1
    assert selected[0]["spec"]["family"] == "collision_final_relation"


@pytest.mark.parametrize(
    "preferred_family",
    ("counterfactual_destination", "inverse_intervention"),
)
def test_stage2_per_scene_cap_keeps_preferred_throwing_branch_family(
    preferred_family: str,
) -> None:
    """Sampled branch families retain the existing Solver-prefer guardrail."""

    payloads = [
        {
            "payload": ["Q destination?", ["A", "B", "C", "D"], 0],
            "spec": {"family": "throwing_final_destination", "phrasing": "forward"},
        },
        {
            "payload": ["Q cf?", ["A", "B", "C", "D"], 1],
            "spec": {
                "family": "counterfactual_destination",
                "phrasing": "counterfactual",
            },
        },
        {
            "payload": ["Q inverse?", ["A", "B", "C", "D"], 2],
            "spec": {"family": "inverse_intervention", "phrasing": "inverse"},
        },
    ]

    selected = _select_stage2_vqa_payloads(
        payloads,
        preferred_family=preferred_family,
        max_questions=1,
        seed_key=f"{preferred_family}-budget-regression",
    )

    assert len(selected) == 1
    assert selected[0]["spec"]["family"] == preferred_family


@pytest.mark.parametrize(
    "preferred_family",
    ("counterfactual_destination", "inverse_intervention"),
)
def test_stage2_branch_funnel_reaches_solver(
    tmp_path: Path,
    preferred_family: str,
) -> None:
    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK],
        solver_outputs=[_SOLVER_WRONG],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            question_families=(preferred_family,),
            max_solver_questions_per_scene=1,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_many_vqa,
    )

    step = builder(1)

    assert step.metrics["stage2_raw_payload_by_family"][
        "counterfactual_destination"
    ] == 1
    assert step.metrics["stage2_raw_payload_by_family"]["inverse_intervention"] == 1
    assert step.metrics["stage2_preferred_family_missing_by_family"] == {}
    assert step.metrics["solver_selection_by_family"] == {preferred_family: 1}


def test_stage2_reports_relation_semantic_counts(tmp_path: Path) -> None:
    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK],
        solver_outputs=[_SOLVER_WRONG, _SOLVER_WRONG],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            question_families=("collision_final_relation",),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_relation_vqa,
    )

    step = builder(1)

    assert step.metrics["stage2_relation_semantic_counts"] == {"no_contact": 2}
    assert step.metrics["stage2_raw_payload_by_family"] == {
        "collision_final_relation": 2
    }
    assert step.metrics["solver_selection_by_family"] == {
        "collision_final_relation": 2
    }


def test_stage2_limits_solver_groups_per_step(tmp_path: Path) -> None:
    """Step cap bounds total Solver video rows after per-scene sampling."""

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK, _PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT] * 4,
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=2,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            question_families=("throwing_final_destination",),
            max_solver_questions_per_scene=3,
            max_solver_groups_per_step=4,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_many_vqa,
    )

    step = builder(1)

    assert step.metrics["vqa_question_total_raw"] == 12
    assert step.metrics["vqa_question_total_before_step_cap"] == 6
    assert step.metrics["vqa_question_total_selected"] == 4
    assert step.metrics["n_solver_groups_available"] == 6
    assert step.metrics["n_solver_groups_selected"] == 4
    assert step.metrics["solver_groups_dropped"] == 2
    assert step.metrics["n_solver_rollouts_per_batch"] == 4
    assert sum(step.metrics["solver_selection_by_family"].values()) == 4


def test_stage2_batch_dedup_penalises_duplicates_and_skips_solver(
    tmp_path: Path,
) -> None:
    """Identical rollouts in one GRPO group: keep 1 canonical, penalise the rest.

    With ``group_size=2`` and two byte-identical proposer outputs, the second
    rollout is an AST duplicate: it must get the overlap-aware duplicate reward
    and must NOT be sent to the Solver (so only the canonical's ``group_size``
    solver rows run). This is what restores intra-group GRPO variance.
    """

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK, _PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT, _SOLVER_WRONG],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=2,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            cold_start_fewshot_steps=0,
            copy_penalty_enabled=True,
            copy_exact_reward=-0.5,
            batch_dedup_enabled=True,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )

    step = builder(1)

    assert step.metrics["stage2_anti_copy_active"] is True
    assert step.metrics["stage2_batch_duplicate_count"] == 1
    # Only the canonical rollout reaches the Solver (group_size=2 rows).
    assert step.metrics["n_solver_rollouts_per_batch"] == 2

    rows = [
        json.loads(line)
        for line in (
            Path(step.artifacts["step_dir"]) / "reward_records.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]
    proposer_rows = [r for r in rows if r.get("problem_type") == "gen_vqa"]
    dup_rows = [r for r in proposer_rows if r.get("batch_duplicate") is True]
    assert len(dup_rows) == 1
    assert dup_rows[0]["reward"] < -0.75
    # Intra-group variance: the canonical reward differs from the duplicate.
    rewards = sorted(r["reward"] for r in proposer_rows)
    assert rewards[0] < -0.75
    assert rewards[-1] != rewards[0]


def test_stage2_batch_dedup_penalises_near_duplicates(tmp_path: Path) -> None:
    """Near-identical ASTs in one GRPO group are treated like duplicates."""

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK, _PROPOSER_NEAR_DUP],
        solver_outputs=[_SOLVER_CORRECT, _SOLVER_WRONG],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=2,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            cold_start_fewshot_steps=0,
            copy_penalty_enabled=True,
            copy_exact_reward=-0.5,
            copy_similarity_threshold=0.80,
            batch_dedup_enabled=True,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )

    step = builder(1)

    assert step.metrics["stage2_batch_duplicate_count"] == 1
    assert step.metrics["n_solver_rollouts_per_batch"] == 2


def test_stage2_batch_dedup_penalises_failed_duplicates(tmp_path: Path) -> None:
    """All-failed batches still get anti-repeat variance for duplicate ASTs."""

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK, _PROPOSER_OK],
        solver_outputs=[],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=2,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            cold_start_fewshot_steps=0,
            copy_penalty_enabled=True,
            copy_exact_reward=-0.5,
            batch_dedup_enabled=True,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_overlap,
    )

    step = builder(1)

    assert step.metrics["n_executor_ok"] == 0
    assert step.metrics["failure_reason_distribution"]["overlap"] == 2
    assert step.metrics["stage2_batch_duplicate_count"] == 1
    assert step.metrics["stage2_duplicate_failure_count"] == 1
    assert step.metrics["stage2_copy_penalty_applied_count"] == 1
    proposer_rewards = [
        float(r.reward)
        for r in step.reward_batch.records
        if r.problem_type == "gen_vqa"
    ]
    assert sorted(proposer_rewards)[0] < sorted(proposer_rewards)[-1]
    assert sorted(proposer_rewards)[-1] == -0.75


def test_stage2_preferred_family_gate_skips_wrong_family_solver(
    tmp_path: Path,
) -> None:
    """A sampled relation prompt cannot be rewarded by a motion-state fallback."""

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK],
        solver_outputs=[],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            question_families=("collision_final_relation",),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_wrong_family,
    )

    step = builder(1)

    assert step.metrics["n_executor_ok"] == 1
    assert step.metrics["stage2_preferred_family_missing_count"] == 1
    assert step.metrics["stage2_preferred_family_missing_by_family"] == {
        "collision_final_relation": 1
    }
    assert step.metrics["stage2_raw_payload_by_family"] == {
        "motion_state_change": 1
    }
    assert step.metrics["n_solver_rollouts_per_batch"] == 0
    assert step.metrics["failure_reason_distribution"]["preferred_family_missing"] == 1


def test_stage2_geometry_repair_can_seed_reference_buffer(
    tmp_path: Path,
) -> None:
    """Bounded geometry-repaired OK+VQA scenes can seed prompt references."""

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            cold_start_fewshot_steps=0,
            reference_buffer_enabled=True,
            reference_k=2,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_repair,
    )

    step = builder(1)

    assert step.metrics["stage2_geometry_repaired_ok_count"] == 1
    assert step.metrics["n_solver_rollouts_per_batch"] == 1
    sizes = step.metrics["stage2_reference_buffer_sizes"]
    assert sum(sizes.values()) == 1
    assert step.metrics["stage2_reference_recorded_repair_tier_counts"] == {
        "geometry": 1
    }


def test_stage2_cold_start_disables_copy_controls(tmp_path: Path) -> None:
    """During the cold-start window dedup/penalty are off (fewshot allowed)."""

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK, _PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT, _SOLVER_CORRECT, _SOLVER_WRONG, _SOLVER_WRONG],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=2,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            cold_start_fewshot_steps=5,
            copy_penalty_enabled=True,
            batch_dedup_enabled=True,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )

    step = builder(1)

    assert step.metrics["stage2_anti_copy_active"] is False
    assert step.metrics["stage2_batch_duplicate_count"] == 0
    assert step.metrics["stage2_copy_penalty_applied_count"] == 0
    # Both rollouts reach the Solver during cold start (no dedup).
    assert step.metrics["n_solver_rollouts_per_batch"] == 4


def test_stage2_anti_copy_decoupled_from_fewshot(tmp_path: Path) -> None:
    """recipe: anti-copy can activate while fewshot prompts remain on."""

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK, _PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT, _SOLVER_WRONG],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=2,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            cold_start_fewshot_steps=200,
            anti_copy_start_step=0,
            copy_penalty_enabled=True,
            batch_dedup_enabled=True,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )

    step = builder(1)

    assert step.metrics["stage2_anti_copy_active"] is True
    assert step.metrics["stage2_anti_copy_start_step"] == 0
    assert step.metrics["stage2_cold_start_fewshot_steps"] == 200
    assert step.metrics["stage2_learnability_weight"] == pytest.approx(1.0)


def test_stage2_learnability_zero_during_fewshot(tmp_path: Path) -> None:
    """recipe: fewshot window forces learnability_weight_during_fewshot."""

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK, _PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT, _SOLVER_WRONG],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=2,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            cold_start_fewshot_steps=200,
            anti_copy_start_step=0,
            learnability_weight=0.5,
            learnability_weight_during_fewshot=0.0,
            copy_penalty_enabled=True,
            batch_dedup_enabled=True,
            diversity_edit_distance_weight=1.5,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )

    step = builder(1)

    assert step.metrics["stage2_fewshot_active"] is True
    assert step.metrics["stage2_effective_learnability_weight"] == pytest.approx(0.0)
    assert step.metrics["stage2_learnability_weight_during_fewshot"] == pytest.approx(
        0.0
    )
    assert "stage2_proposer_group_reward_std_mean" in step.metrics


def test_effective_learnability_and_group_std_helpers() -> None:
    from physevolve.training.online_step_builder_stage2 import (
        PhysEvolveStage2StepBuilderConfig,
        _effective_learnability_weight,
        _proposer_group_reward_std,
    )
    from physevolve.training.reward_batch import RewardRecord

    cfg = PhysEvolveStage2StepBuilderConfig(
        learnability_weight=0.5,
        learnability_weight_during_fewshot=0.0,
    )
    assert _effective_learnability_weight(cfg, fewshot_active=True) == pytest.approx(
        0.0
    )
    assert _effective_learnability_weight(cfg, fewshot_active=False) == pytest.approx(
        0.5
    )
    legacy = PhysEvolveStage2StepBuilderConfig(learnability_weight=0.5)
    assert _effective_learnability_weight(
        legacy, fewshot_active=True
    ) == pytest.approx(0.5)

    records = [
        RewardRecord(
            problem_type="gen_vqa",
            reward=-0.8,
            valid_response_length=8,
            response_length=8,
        ),
        RewardRecord(
            problem_type="gen_vqa",
            reward=-0.8,
            valid_response_length=8,
            response_length=8,
        ),
        RewardRecord(
            problem_type="gen_vqa",
            reward=1.2,
            valid_response_length=8,
            response_length=8,
        ),
        RewardRecord(
            problem_type="pred_vqa",
            reward=1.0,
            valid_response_length=8,
            response_length=8,
        ),
    ]
    uid_map = {0: "g1", 1: "g1", 2: "g1"}
    std = _proposer_group_reward_std(records, uid_map)
    assert std > 0.0
    flat = _proposer_group_reward_std(
        [
            RewardRecord(
                problem_type="gen_vqa",
                reward=-0.8,
                valid_response_length=8,
                response_length=8,
            ),
            RewardRecord(
                problem_type="gen_vqa",
                reward=-0.8,
                valid_response_length=8,
                response_length=8,
            ),
        ],
        {0: "g", 1: "g"},
    )
    assert flat == pytest.approx(0.0)


def test_stage2_cold_start_records_fewshot_index_metadata(
    tmp_path: Path,
) -> None:
    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK, _PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT, _SOLVER_CORRECT],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=2,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            question_families=("throwing_final_destination",),
            cold_start_fewshot_steps=5,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )

    step = builder(1)
    step_dir = Path(step.artifacts["step_dir"])
    snapshots = sorted((step_dir / "prompt_snapshots").glob("proposer__stage2__*.txt"))
    snapshot_names = {path.name for path in snapshots}
    assert "proposer__stage2__throwing_final_destination__fs1.txt" in snapshot_names
    assert "proposer__stage2__throwing_final_destination__fs2.txt" in snapshot_names

    rows = [
        json.loads(line)
        for line in (step_dir / "raw_proposer_outputs.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [row["fewshot_index"] for row in rows] == [1, 2]


def test_stage2_grpo_groups_share_prompt_fewshot_and_video(tmp_path: Path) -> None:
    """Sampling entropy must not change the condition within a GRPO group."""

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK, _PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT] * 4,
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=2,
            response_length=16,
            generation=SmokeGenerationConfig(
                max_new_tokens=16,
                temperature=0.8,
                top_p=0.98,
            ),
            question_families=("throwing_final_destination",),
            cold_start_fewshot_steps=5,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )

    builder(1)

    proposer_specs = [
        spec
        for call in backend.calls
        for spec in call
        if (spec.rollout_metadata or {}).get("role") == "proposer"
    ]
    assert len(proposer_specs) == 2
    assert len({spec.prompt for spec in proposer_specs}) == 1
    assert {
        (spec.rollout_metadata or {}).get("fewshot_index")
        for spec in proposer_specs
    } == {1}
    assert len(
        {(spec.rollout_metadata or {}).get("group_id") for spec in proposer_specs}
    ) == 1
    assert all(spec.temperature == 0.8 and spec.top_p == 0.98 for spec in proposer_specs)

    solver_specs = [
        spec
        for call in backend.calls
        for spec in call
        if (spec.rollout_metadata or {}).get("role") == "solver"
    ]
    by_group: dict[str, list[GenerationSpec]] = {}
    for spec in solver_specs:
        group_id = str((spec.rollout_metadata or {})["group_id"])
        by_group.setdefault(group_id, []).append(spec)
    assert len(by_group) == 2
    for group in by_group.values():
        assert len(group) == 2
        assert len({spec.prompt for spec in group}) == 1
        assert len({id(spec.video) for spec in group}) == 1
        assert all(spec.temperature == 0.8 and spec.top_p == 0.98 for spec in group)


def test_stage2_mock_backend_returns_indexed_fewshot_variant() -> None:
    first = build_proposer_text_for_scene_kind(
        "throwing_pit_board",
        fewshot_index=0,
    )
    second = build_proposer_text_for_scene_kind(
        "throwing_pit_board",
        fewshot_index=1,
    )

    assert "black_left_wall" in first
    assert "gray_left_wall" in second
    assert '"pits": {' not in second


def test_stage2_reference_buffer_records_ok_scenes(tmp_path: Path) -> None:
    """OK scenes feed the ReferenceBuffer for post-cold-start prompting."""

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK, _PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT, _SOLVER_WRONG],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=2,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            cold_start_fewshot_steps=0,
            reference_buffer_enabled=True,
            reference_k=2,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )

    step = builder(1)

    sizes = step.metrics["stage2_reference_buffer_sizes"]
    assert sizes, sizes
    assert sum(sizes.values()) >= 1


def test_stage2_reference_prompt_does_not_cross_scene_kind(tmp_path: Path) -> None:
    """Stage-2 prompts must not borrow references from other scene_kind buckets."""

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            question_families=("throwing_final_destination",),
            cold_start_fewshot_steps=0,
            reference_buffer_enabled=True,
            reference_k=2,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )
    builder._reference_buffer.record(
        "collision_arena",
        _make_sim_state(),
        "collision_ref",
        summary="collision-only summary",
    )

    step = builder(1)
    prompt = (
        Path(step.artifacts["step_dir"])
        / "prompt_snapshots"
        / "proposer__stage2__throwing_final_destination.txt"
    ).read_text()

    assert "collision-only summary" not in prompt
    assert "## Recent Successful Scenes" not in prompt


def test_stage2_reference_during_fewshot_hybrid(tmp_path: Path) -> None:
    """recipe: fewshot ON still injects buffer summaries when flag is set."""

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            question_families=("throwing_final_destination",),
            cold_start_fewshot_steps=200,
            reference_buffer_enabled=True,
            reference_buffer_during_fewshot=True,
            reference_k=2,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_fake_executor_ok_with_vqa,
    )
    sim = _make_sim_state()
    builder._reference_buffer.record(
        "throwing_pit_board",
        sim,
        "throw_ref_a",
        summary="hybrid-ref-alpha n_dynamic=3",
        structural_signature="sig-alpha",
    )
    builder._reference_buffer.record(
        "throwing_pit_board",
        sim,
        "throw_ref_b",
        summary="hybrid-ref-beta n_dynamic=4",
        structural_signature="sig-beta",
    )

    step = builder(1)
    snapshots = list(
        (Path(step.artifacts["step_dir"]) / "prompt_snapshots").glob(
            "proposer__stage2__*.txt"
        )
    )
    assert snapshots
    prompt = snapshots[0].read_text()
    assert "Validated Golden Example" in prompt
    assert "Recent OK Parameter Bands" in prompt
    assert "hybrid-ref-alpha" in prompt or "hybrid-ref-beta" in prompt


def test_stage2_config_rejects_negative_chunk_size(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="solver_rollout_chunk_size"):
        PhysEvolveStage2StepBuilderConfig(solver_rollout_chunk_size=-1)


def test_stage2_config_validates_scene_kind_pick_quotas() -> None:
    config = PhysEvolveStage2StepBuilderConfig(
        scenes_per_batch=4,
        min_scene_kind_picks_per_step=(
            ("collision_arena", 1),
            ("motion_lab", 1),
        ),
    )
    assert config.min_scene_kind_picks_per_step == (
        ("collision_arena", 1),
        ("motion_lab", 1),
    )

    with pytest.raises(ValueError, match="requires 5 picks"):
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=4,
            min_scene_kind_picks_per_step=(
                ("collision_arena", 2),
                ("motion_lab", 3),
            ),
        )
    with pytest.raises(ValueError, match="no active family"):
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=2,
            question_families=("throwing_final_destination",),
            min_scene_kind_picks_per_step=(("motion_lab", 1),),
        )


def test_stage2_config_rejects_negative_failure_weight() -> None:
    import pytest

    with pytest.raises(ValueError, match="intrinsic_weight"):
        PhysEvolveStage2StepBuilderConfig(intrinsic_weight=-0.1)
    with pytest.raises(ValueError, match="intrinsic_component_weights"):
        PhysEvolveStage2StepBuilderConfig(intrinsic_component_weights=(0.0, 1.0))
    with pytest.raises(ValueError, match="intrinsic_component_weights"):
        PhysEvolveStage2StepBuilderConfig(
            intrinsic_component_weights=(0.0, 0.0, 0.0, -1.0, 0.0)
        )
    with pytest.raises(ValueError, match="proposer_failure_weight"):
        PhysEvolveStage2StepBuilderConfig(proposer_failure_weight=-0.1)
    with pytest.raises(ValueError, match="color_variants_per_scene"):
        PhysEvolveStage2StepBuilderConfig(color_variants_per_scene=-1)
    with pytest.raises(ValueError, match="max_solver_questions_per_scene"):
        PhysEvolveStage2StepBuilderConfig(max_solver_questions_per_scene=-1)
    with pytest.raises(ValueError, match="max_solver_groups_per_step"):
        PhysEvolveStage2StepBuilderConfig(max_solver_groups_per_step=-1)
    with pytest.raises(ValueError, match="cold_start_fewshot_steps"):
        PhysEvolveStage2StepBuilderConfig(cold_start_fewshot_steps=-1)
    with pytest.raises(ValueError, match="copy_similarity_threshold"):
        PhysEvolveStage2StepBuilderConfig(copy_similarity_threshold=1.5)
    with pytest.raises(ValueError, match="learnability_saturation_threshold"):
        PhysEvolveStage2StepBuilderConfig(learnability_saturation_threshold=-0.1)
    with pytest.raises(ValueError, match="learnability_saturation_weight"):
        PhysEvolveStage2StepBuilderConfig(learnability_saturation_weight=-0.1)
    with pytest.raises(ValueError, match="diversity_edit_distance_weight"):
        PhysEvolveStage2StepBuilderConfig(diversity_edit_distance_weight=-0.1)


def test_build_solver_requests_uses_per_item_sim_state() -> None:
    """Colour-permuted variant questions get their OWN (re-rendered) video.

    The variant payload item carries its own ``sim_state`` (distinct scene_id /
    video); ``build_solver_requests`` must use it for that question while
    baseline questions reuse the scene's sim_state.
    """

    import dataclasses

    from physevolve.tasks.router import TaskMode
    from physevolve.training.selfplay_batch import build_solver_requests

    base = _make_sim_state()
    variant = dataclasses.replace(
        base,
        scene_id="variant_scene_xyz",
        video_frames=np.ones((1, 2, 8, 8, 3), dtype=np.uint8),
    )
    result = make_ok(
        base.scene_id,
        base,
        ("Q0?", ["a", "b", "c", "d"], 0),
        walltime_s=0.01,
        vqa_payloads=[
            {"payload": ["Q0?", ["a", "b", "c", "d"], 0], "spec": {"family": "f"}},
            {
                "payload": ["Q1 variant?", ["a", "b", "c", "d"], 1],
                "spec": {"family": "f"},
                "sim_state": variant,
            },
        ],
    )
    reqs = build_solver_requests(
        result, task_modes=(TaskMode.VQA,), sample_id="s"
    )
    assert len(reqs) == 2
    assert reqs[0].scene_id == base.scene_id
    assert reqs[1].scene_id == "variant_scene_xyz"


def test_resolved_families_drops_branch_elision_when_disabled() -> None:
    default = PhysEvolveStage2StepBuilderConfig()
    assert "counterfactual_destination" in default.resolved_families()
    assert "inverse_intervention" in default.resolved_families()

    disabled = PhysEvolveStage2StepBuilderConfig(disable_branch_elision=True)
    families = disabled.resolved_families()
    assert "counterfactual_destination" not in families
    assert "inverse_intervention" not in families
    assert "throwing_final_destination" in families
    assert "throwing_current_run_judgment" in families

    filtered = PhysEvolveStage2StepBuilderConfig(
        question_families=(
            "throwing_final_destination",
            "counterfactual_destination",
            "inverse_intervention",
        ),
        disable_branch_elision=True,
    )
    assert filtered.resolved_families() == ("throwing_final_destination",)

    with pytest.raises(ValueError, match="disable_branch_elision"):
        PhysEvolveStage2StepBuilderConfig(
            question_families=(
                "counterfactual_destination",
                "inverse_intervention",
            ),
            disable_branch_elision=True,
        )


def test_stage2_disable_branch_elision_omits_skip_kwarg_by_default(
    tmp_path: Path,
) -> None:
    seen: dict[str, Any] = {}

    def _capturing_executor(*args: Any, **kwargs: Any):
        seen.update(kwargs)
        return _fake_executor_ok_with_vqa(*args, **kwargs)

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT],
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            question_families=("throwing_final_destination",),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_capturing_executor,
    )

    builder(1)

    assert "skip_branch_sims" not in seen
    assert seen["camera_depth_bypass_enabled"] is False


def test_stage2_disable_branch_elision_skips_branch_sims_and_payloads(
    tmp_path: Path,
) -> None:
    seen: dict[str, Any] = {}

    def _capturing_executor(*args: Any, **kwargs: Any):
        seen.update(kwargs)
        return _fake_executor_ok_with_many_vqa(*args, **kwargs)

    backend = _Stage2FakeBackend(
        proposer_outputs=[_PROPOSER_OK],
        solver_outputs=[_SOLVER_CORRECT] * 12,
    )
    builder = PhysEvolveStage2StepBuilder(
        PhysEvolveStage2StepBuilderConfig(
            scenes_per_batch=1,
            group_size=1,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            question_families=(
                "throwing_final_destination",
                "throwing_current_run_judgment",
                "counterfactual_destination",
                "inverse_intervention",
            ),
            disable_branch_elision=True,
            max_solver_questions_per_scene=5,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _cfg: backend,
        executor_fn=_capturing_executor,
    )

    step = builder(1)

    assert seen["skip_branch_sims"] is True
    pick_weights = step.metrics["vqa_family_pick_weights"]
    assert "counterfactual_destination" not in pick_weights
    assert "inverse_intervention" not in pick_weights
    assert "throwing_final_destination" in pick_weights
    assert "throwing_current_run_judgment" in pick_weights
    assert "counterfactual_destination" not in step.metrics["stage2_raw_payload_by_family"]
    assert "inverse_intervention" not in step.metrics["stage2_raw_payload_by_family"]
    assert "counterfactual_destination" not in step.metrics["solver_selection_by_family"]
    assert "inverse_intervention" not in step.metrics["solver_selection_by_family"]
