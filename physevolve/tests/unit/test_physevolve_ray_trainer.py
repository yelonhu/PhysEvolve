"""Tests for PhysEvolve's self-play veRL trainer boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from physevolve.engine.schema import (
    BodyDef,
    BodyTrajectory,
    BodyStatus,
    EnvConfig,
    GeomType,
    JointType,
    SceneDef,
    SimState,
    compute_scene_id,
)
from physevolve.executor.result import make_ok
from physevolve.inference.hf_backend import GenerationResult
from physevolve.smoke.run_config import SmokeGenerationConfig
from physevolve.training.physevolve_ray_trainer import (
    PhysEvolveRayPPOTrainer,
    _drop_gen_vqa_rows_for_frozen_proposer,
)
from physevolve.training.online_step import OnlineSelfPlayStep
from physevolve.training.online_step_builder import (
    GenerationSpec,
    PhysEvolveOnlineStepBuilder,
    PhysEvolveOnlineStepBuilderConfig,
)
from physevolve.training.reward_batch import RewardRecord, build_reward_batch
from physevolve.training.verl_smoke import (
    PhysEvolveTrainerConfig,
    prepare_physevolve_trainer_smoke,
    write_dry_run_summary,
)


@dataclass
class _FakeGenerationBackend:
    outputs: list[str]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def generate_with_metadata(self, prompt: str, video=None, **kwargs: Any) -> GenerationResult:
        text = self.outputs.pop(0)
        return GenerationResult(
            text=text,
            metadata={"generated_token_count": 5, "video_present": video is not None},
            generated_text=text,
            prompt_text=prompt,
            prompt_tail=prompt[-32:],
        )


@dataclass
class _BatchedFakeGenerationBackend:
    """Fake backend that mirrors the real veRL backend's batched contract.

    Each ``generate_batch`` call is recorded so tests can assert exactly how
    many vLLM submissions the builder makes per PPO step.
    """

    outputs: list[str]
    calls: list[list[GenerationSpec]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def generate_batch(self, specs):
        spec_list = list(specs)
        self.calls.append(spec_list)
        results: list[GenerationResult] = []
        for spec in spec_list:
            text = self.outputs.pop(0)
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


def test_physevolve_ray_trainer_fit_owns_lifecycle_and_update(tmp_path: Path) -> None:
    trainer = object.__new__(PhysEvolveRayPPOTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            total_training_steps=2,
            default_local_dir=tmp_path,
            save_freq=0,
        )
    )
    trainer.global_steps = 0
    trainer._load_checkpoint = lambda: None
    trainer._save_checkpoint = lambda: None
    trainer._physevolve_cached_step_builder = _fake_step
    trainer._update_physevolve_online_step = lambda step: {"updated": True, "step": step.step}
    trainer.checkpoint_manager = _FakeCheckpointManager()

    rows, update_count = trainer.fit()

    assert update_count == 2
    assert [row["step"] for row in rows] == [1, 2]
    assert all("time/step_total_s" in row for row in rows)
    assert all("throughput/rollouts_per_s" in row for row in rows)
    assert trainer.checkpoint_manager.events == [
        ("wake", 0),
        ("sleep", None),
        ("wake", 1),
        ("sleep", None),
    ]
    assert (tmp_path / "summary.json").exists()


def test_online_step_builder_builds_real_step_with_fake_backend(tmp_path: Path) -> None:
    proposer = (
        "<think>x</think><propose>\n"
        "def build_scene(api):\n"
        "    api.add_sphere('ball', pos=(0,0,1), radius=0.1, mass=1.0, "
        "rgba=(1,0,0,1), friction=0.2, restitution=0.0, fixed=False)\n"
        "</propose>"
    )
    solver = '<think>x</think><answer>{"ball": {"grid": [8, 8, 5], "status": "stable"}}</answer>'
    backend = _FakeGenerationBackend([proposer, solver])

    builder = PhysEvolveOnlineStepBuilder(
        PhysEvolveOnlineStepBuilderConfig(
            scenes_per_batch=1,
            task_modes=("forward",),
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _config: backend,
        executor_fn=_fake_executor_ok,
    )

    step = builder(1)

    assert step.step == 1
    assert step.reward_batch.metrics["reward_tensor_finite"] is True
    assert step.metrics["executor_call_count"] == 1
    assert step.metrics["solver_video_request_count"] == 1
    assert step.metrics["unsupported_problem_types_seen"] == []
    assert step.metrics["executor_workers_configured"] == 1
    assert step.metrics["executor_parallelism_active"] is False
    assert step.metrics["time/executor_s"] >= 0.0


def test_online_step_builder_expands_rollout_groups(tmp_path: Path) -> None:
    proposer = (
        "<think>x</think><propose>\n"
        "def build_scene(api):\n"
        "    api.add_sphere('ball', pos=(0,0,1), radius=0.1, mass=1.0, "
        "rgba=(1,0,0,1), friction=0.2, restitution=0.0, fixed=False)\n"
        "</propose>"
    )
    solver = '<think>x</think><answer>{"ball": {"grid": [8, 8, 5], "status": "stable"}}</answer>'
    backend = _FakeGenerationBackend([proposer, proposer, solver, solver])

    builder = PhysEvolveOnlineStepBuilder(
        PhysEvolveOnlineStepBuilderConfig(
            scenes_per_batch=1,
            group_size=2,
            task_modes=("forward",),
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _config: backend,
        executor_fn=_fake_executor_ok,
    )

    step = builder(1)

    assert step.reward_batch.tensor.shape == (4, 16)
    assert step.metrics["online_proposer_generation_count"] == 2
    assert step.metrics["planned_solver_groups"] == 1
    assert step.metrics["selected_solver_groups"] == 1
    assert step.metrics["n_solver_requests"] == 2
    assert step.metrics["online_generation_token_count"] == 20


def test_online_step_builder_solver_shape_independent_of_failed_proposer(tmp_path: Path) -> None:
    bad_proposer = "<think>x</think>no propose block"
    proposer = (
        "<think>x</think><propose>\n"
        "def build_scene(api):\n"
        "    api.add_sphere('ball', pos=(0,0,1), radius=0.1, mass=1.0, "
        "rgba=(1,0,0,1), friction=0.2, restitution=0.0, fixed=False)\n"
        "</propose>"
    )
    solver = '<think>x</think><answer>{"ball": {"grid": [8, 8, 5], "status": "stable"}}</answer>'
    backend = _FakeGenerationBackend([bad_proposer, proposer, solver, solver])

    builder = PhysEvolveOnlineStepBuilder(
        PhysEvolveOnlineStepBuilderConfig(
            scenes_per_batch=1,
            group_size=2,
            task_modes=("forward",),
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _config: backend,
        executor_fn=_fake_executor_ok,
    )

    step = builder(1)

    assert step.reward_batch.tensor.shape == (4, 16)
    assert step.metrics["executor_call_count"] == 1
    assert step.metrics["selected_solver_groups"] == 1
    assert step.metrics["failure_reason_distribution"]["syntax"] == 1


def test_online_step_builder_rotates_skills_and_logs_intrinsic(tmp_path: Path) -> None:
    """Per-item skill rotation and intrinsic components persisted on disk.

    Checks that a single PPO step uses
    multiple skill_targets across items, the prompt-snapshot directory
    captures each one, ``reward_records.jsonl`` carries ``intrinsic/*``
    keys for OK scenes, and ``step_summary.jsonl`` reports per-skill
    reward means.
    """

    proposer = (
        "<think>x</think><propose>\n"
        "def build_scene(api):\n"
        "    api.add_sphere('ball', pos=(0,0,1), radius=0.1, mass=1.0, "
        "rgba=(1,0,0,1), friction=0.2, restitution=0.0, fixed=False)\n"
        "</propose>"
    )
    solver = '<think>x</think><answer>{"ball": {"grid": [8, 8, 5], "status": "stable"}}</answer>'
    # 4 items x group_size 1 proposer + 4 OK scenes x 1 solver request x group_size 1 solver.
    outputs = [proposer] * 4 + [solver] * 4
    backend = _FakeGenerationBackend(outputs)

    builder = PhysEvolveOnlineStepBuilder(
        PhysEvolveOnlineStepBuilderConfig(
            scenes_per_batch=4,
            group_size=1,
            task_modes=("forward",),
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _config: backend,
        executor_fn=_fake_executor_ok,
    )

    step = builder(1)

    step_dir = Path(step.artifacts["step_dir"])
    snapshot_dir = step_dir / "prompt_snapshots"
    skill_snapshots = sorted(p.name for p in snapshot_dir.glob("proposer__*.txt"))
    assert len(skill_snapshots) >= 2, skill_snapshots
    assert all(name.startswith("proposer__") for name in skill_snapshots)

    # Skill counts and per-skill reward means are reported.
    counts = step.metrics["skill_target_counts"]
    assert sum(counts.values()) == 4
    assert step.metrics["proposer_reward_by_skill"], step.metrics

    # Intrinsic component means surface on the step summary.
    for component in ("motion", "interaction", "complexity", "diversity", "parameter"):
        assert f"intrinsic/{component}_mean" in step.metrics, component

    # reward_records.jsonl carries intrinsic/* keys for proposer rows.
    import json as _json

    rows = [
        _json.loads(line)
        for line in (step_dir / "reward_records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    proposer_rows = [r for r in rows if r.get("problem_type") == "gen_physics"]
    assert proposer_rows, rows
    assert all("skill_target" in r for r in proposer_rows)
    assert any("intrinsic/motion" in r for r in proposer_rows)

    # Rolling novelty buffer grew this step.
    assert step.metrics["novelty_buffer_size"] >= 1
    assert step.metrics["intrinsic_enabled"] is True


def test_online_step_builder_records_acc_per_skill(tmp_path: Path) -> None:
    """per-skill rolling acc tracker accumulates step-to-step.

    After a step in which OK proposers produced solver rewards, the
    tracker snapshot on the *next* step must show at least one populated
    skill, and the builder must surface that snapshot under
    ``acc_by_skill``. ``difficulty_assignments_by_skill`` must surface
    the derived band for every skill picked this step.
    """

    proposer = (
        "<think>x</think><propose>\n"
        "def build_scene(api):\n"
        "    api.add_sphere('ball', pos=(0,0,1), radius=0.1, mass=1.0, "
        "rgba=(1,0,0,1), friction=0.2, restitution=0.0, fixed=False)\n"
        "</propose>"
    )
    solver = '<think>x</think><answer>{"ball": {"grid": [8, 8, 5], "status": "stable"}}</answer>'
    # 2 steps; each step uses 2 items x 1 group_size proposer + 2 solver.
    outputs_per_step = [proposer] * 2 + [solver] * 2
    backend = _BatchedFakeGenerationBackend(outputs_per_step * 2)

    builder = PhysEvolveOnlineStepBuilder(
        PhysEvolveOnlineStepBuilderConfig(
            scenes_per_batch=2,
            group_size=1,
            task_modes=("forward",),
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
            cold_start_uniform_steps=0,
        ),
        run_dir=tmp_path,
        backend_factory=lambda _config: backend,
        executor_fn=_fake_executor_ok,
    )

    step_one = builder(1)
    step_two = builder(2)

    # At least one skill bucket must surface in the tracker snapshot
    # on step 2 because step 1 already populated it.
    snapshot = step_two.metrics["acc_by_skill"]
    assert snapshot, snapshot
    assert all(isinstance(v, float) for v in snapshot.values())

    # difficulty_assignments_by_skill reflects what derive_difficulty
    # decided for each skill picked this step.
    assignments = step_one.metrics["difficulty_assignments_by_skill"]
    assert assignments, assignments
    assert all(v in {"easy", "medium", "hard"} for v in assignments.values())

    # Snapshot files use single-segment names:
    # ``proposer__<skill>.txt`` (no difficulty in filename).
    step_dir = Path(step_one.artifacts["step_dir"])
    snapshot_files = sorted(p.name for p in (step_dir / "prompt_snapshots").glob("proposer__*.txt"))
    assert snapshot_files, snapshot_files
    for name in snapshot_files:
        assert name.count("__") == 1, name


def test_online_step_builder_emits_effective_learnable_fraction_metric(tmp_path: Path) -> None:
    """``effective_learnable_fraction`` must be finite and in ``[0, 1]``.

    The metric records high-accuracy scenes: when the Solver
    collapses to all-0 or all-1 it should still be finite (we treat
    scored_ok=0 as fraction=0.0 rather than NaN), and it must not
    exceed 1.0 by any rounding error.
    """

    proposer = (
        "<think>x</think><propose>\n"
        "def build_scene(api):\n"
        "    api.add_sphere('ball', pos=(0,0,1), radius=0.1, mass=1.0, "
        "rgba=(1,0,0,1), friction=0.2, restitution=0.0, fixed=False)\n"
        "</propose>"
    )
    solver = '<think>x</think><answer>{"ball": {"grid": [8, 8, 5], "status": "stable"}}</answer>'
    backend = _FakeGenerationBackend([proposer, proposer, solver, solver])

    builder = PhysEvolveOnlineStepBuilder(
        PhysEvolveOnlineStepBuilderConfig(
            scenes_per_batch=2,
            group_size=1,
            task_modes=("forward",),
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _config: backend,
        executor_fn=_fake_executor_ok,
    )

    step = builder(1)

    elf = step.metrics["effective_learnable_fraction"]
    assert isinstance(elf, float)
    assert np.isfinite(elf)
    assert 0.0 <= elf <= 1.0
    assert "cold_start_active" in step.metrics
    assert isinstance(step.metrics["reference_buffer_sizes"], dict)
    # pick weights are a 1-D dict keyed on skill.
    pick_weights = step.metrics["pick_weight_by_skill"]
    total = 0.0
    for w in pick_weights.values():
        assert np.isfinite(w)
        total += float(w)
    assert total == pytest.approx(1.0, abs=1e-6)
    # The default intrinsic weight is 0.3.
    assert step.metrics["intrinsic_weight"] == pytest.approx(0.3)


def test_online_step_builder_does_not_backfill_from_current_candidate(tmp_path: Path) -> None:
    bad_proposer = "<think>x</think>no propose block"
    proposer = (
        "<think>x</think><propose>\n"
        "def build_scene(api):\n"
        "    api.add_sphere('ball', pos=(0,0,1), radius=0.1, mass=1.0, "
        "rgba=(1,0,0,1), friction=0.2, restitution=0.0, fixed=False)\n"
        "</propose>"
    )
    solver = '<think>x</think><answer>{"ball": {"grid": [8, 8, 5], "status": "stable"}}</answer>'
    backend = _FakeGenerationBackend([bad_proposer, bad_proposer, bad_proposer, proposer, solver, solver])

    builder = PhysEvolveOnlineStepBuilder(
        PhysEvolveOnlineStepBuilderConfig(
            scenes_per_batch=2,
            group_size=2,
            task_modes=("forward",),
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _config: backend,
        executor_fn=_fake_executor_ok,
    )

    step = builder(1)

    assert step.metrics["planned_solver_groups"] == 2
    assert step.metrics["selected_solver_groups"] == 1
    assert step.metrics["solver_backfill_shortfall"] == 1
    assert step.metrics["solver_candidate_pool_size"] == 1
    assert step.reward_batch.tensor.shape == (6, 16)


def test_online_step_builder_uses_batched_backend_when_available(tmp_path: Path) -> None:
    """Single proposer batch + single solver batch, uid groups preserved."""

    proposer = (
        "<think>x</think><propose>\n"
        "def build_scene(api):\n"
        "    api.add_sphere('ball', pos=(0,0,1), radius=0.1, mass=1.0, "
        "rgba=(1,0,0,1), friction=0.2, restitution=0.0, fixed=False)\n"
        "</propose>"
    )
    solver = '<think>x</think><answer>{"ball": {"grid": [8, 8, 5], "status": "stable"}}</answer>'
    scenes_per_batch = 2
    group_size = 2
    task_modes = ("forward",)
    proposer_rows = scenes_per_batch * group_size
    # Every proposer OK -> one solver candidate per scene -> rolled out
    # ``group_size`` times -> scenes_per_batch * group_size solver rows.
    selected_groups = scenes_per_batch * len(task_modes)
    solver_rows = selected_groups * group_size
    outputs = [proposer] * proposer_rows + [solver] * solver_rows
    backend = _BatchedFakeGenerationBackend(outputs)

    builder = PhysEvolveOnlineStepBuilder(
        PhysEvolveOnlineStepBuilderConfig(
            scenes_per_batch=scenes_per_batch,
            group_size=group_size,
            task_modes=task_modes,
            response_length=16,
            generation=SmokeGenerationConfig(max_new_tokens=16),
        ),
        run_dir=tmp_path,
        backend_factory=lambda _config: backend,
        executor_fn=_fake_executor_ok,
    )

    step = builder(1)

    # Exactly two batched submissions per step (proposer batch + solver batch).
    assert len(backend.calls) == 2, [len(c) for c in backend.calls]
    proposer_specs, solver_specs = backend.calls
    assert len(proposer_specs) == proposer_rows
    assert len(solver_specs) == solver_rows
    assert all(spec.video is None for spec in proposer_specs)
    # Per-batch row counts must surface as W&B metrics.
    assert step.metrics["n_proposer_rollouts_per_batch"] == proposer_rows
    assert step.metrics["n_solver_rollouts_per_batch"] == solver_rows

    # GRPO uid invariant: each uid appears exactly group_size times within its phase.
    from collections import Counter

    proposer_uids = Counter(spec.rollout_metadata["uid"] for spec in proposer_specs)
    solver_uids = Counter(spec.rollout_metadata["uid"] for spec in solver_specs)
    assert all(count == group_size for count in proposer_uids.values()), proposer_uids
    assert all(count == group_size for count in solver_uids.values()), solver_uids
    # Proposer uids fan out per item; solver uids fan out per (request, candidate_idx).
    assert len(proposer_uids) == scenes_per_batch
    assert len(solver_uids) == selected_groups

    # Reward tensor shape matches the flat (proposer | solver) record order.
    assert step.reward_batch.tensor.shape == (proposer_rows + solver_rows, 16)


def test_prepare_smoke_dry_run_writes_native_config(tmp_path: Path) -> None:
    config = PhysEvolveTrainerConfig(
        model="/models/Qwen3-VL-8B-Instruct",
        out_root=tmp_path,
        run_id="physevolve_unit",
        execute=False,
        generation=SmokeGenerationConfig(
            max_new_tokens=2048,
            temperature=0.8,
            top_p=0.98,
            top_k=20,
        ),
    )

    prepared = prepare_physevolve_trainer_smoke(config)
    summary = write_dry_run_summary(config, prepared.run_dir)
    verl_config = json.loads((prepared.run_dir / "verl_config.json").read_text())

    assert summary["executed"] is False
    assert summary["ppo_update_count"] == 0
    assert (prepared.run_dir / "verl_config.json").exists()
    assert prepared.factory_config["output_dir"] == prepared.run_dir.as_posix()
    rollout = verl_config["actor_rollout_ref"]["rollout"]
    assert rollout["response_length"] == 2048
    assert rollout["temperature"] == pytest.approx(0.8)
    assert rollout["top_p"] == pytest.approx(0.98)
    assert rollout["top_k"] == 20
    generation = verl_config["physevolve"]["generation"]
    assert generation["temperature"] == pytest.approx(0.8)
    assert generation["top_p"] == pytest.approx(0.98)


def test_prepare_stage2_smoke_writes_diversity_only_intrinsic_config(
    tmp_path: Path,
) -> None:
    config = PhysEvolveTrainerConfig(
        model="/models/Qwen3-VL-8B-Instruct",
        out_root=tmp_path,
        run_id="stage2_diversity_only",
        execute=False,
        stage=2,
        task_modes=("vqa",),
        group_size=2,
        stage2_intrinsic_enabled=True,
        stage2_intrinsic_weight=0.1,
        stage2_intrinsic_component_weights=(0.0, 0.0, 0.0, 1.0, 0.0),
        stage2_branch_family_weight_multiplier=1.2,
    )

    prepared = prepare_physevolve_trainer_smoke(config)
    payload = json.loads((prepared.run_dir / "verl_config.json").read_text())

    assert payload["physevolve"]["stage2_intrinsic_enabled"] is True
    assert payload["physevolve"]["stage2_intrinsic_weight"] == pytest.approx(0.1)
    assert payload["physevolve"]["stage2_branch_family_weight_multiplier"] == 1.2
    assert payload["physevolve"]["stage2_intrinsic_component_weights"] == [
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]


def test_stage2_trainer_builder_reads_solver_budget(tmp_path: Path) -> None:
    trainer = object.__new__(PhysEvolveRayPPOTrainer)
    trainer.config = SimpleNamespace(
        data=SimpleNamespace(train_batch_size=4),
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(n=4, response_length=16),
        ),
        physevolve=SimpleNamespace(
            stage=2,
            executor_workers=3,
            base_model_path="/models/Qwen3-VL-8B-Instruct",
            video_save_freq=0,
            video_save_max_per_step=4,
            solver_rollout_chunk_size=5,
            stage2_intrinsic_enabled=True,
            stage2_intrinsic_weight=0.1,
            stage2_intrinsic_component_weights=[0.0, 0.0, 0.0, 1.0, 0.0],
            stage2_proposer_failure_weight=1.5,
            stage2_color_variants_per_scene=1,
            stage2_max_solver_questions_per_scene=3,
            stage2_max_solver_groups_per_step=24,
            stage2_min_scene_kind_picks_per_step={
                "collision_arena": 1,
                "motion_lab": 1,
            },
            stage2_camera_depth_bypass_enabled=True,
            stage2_branch_family_weight_multiplier=1.2,
            stage2_cold_start_fewshot_steps=15,
            stage2_reference_buffer_enabled=True,
            stage2_reference_k=3,
            stage2_reference_buffer_size=32,
            stage2_copy_penalty_enabled=True,
            stage2_copy_exact_reward=-0.5,
            stage2_copy_cap=0.1,
            stage2_copy_similarity_threshold=0.9,
            stage2_batch_dedup_enabled=True,
            stage2_learnability_saturation_threshold=0.85,
            stage2_learnability_saturation_weight=1.0,
            stage2_diversity_edit_distance_weight=1.0,
            generation=SimpleNamespace(
                proposer_max_new_tokens=16,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
            ),
        ),
        trainer=SimpleNamespace(default_local_dir=tmp_path),
    )

    builder = trainer._physevolve_step_builder()

    assert builder.config.solver_rollout_chunk_size == 5
    assert builder.config.intrinsic_enabled is True
    assert builder.config.intrinsic_weight == pytest.approx(0.1)
    assert builder.config.intrinsic_component_weights == (
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    assert builder.config.color_variants_per_scene == 1
    assert builder.config.max_solver_questions_per_scene == 3
    assert builder.config.max_solver_groups_per_step == 24
    assert builder.config.min_scene_kind_picks_per_step == (
        ("collision_arena", 1),
        ("motion_lab", 1),
    )
    assert builder.config.camera_depth_bypass_enabled is True
    assert builder.config.branch_family_weight_multiplier == 1.2
    assert builder.config.cold_start_fewshot_steps == 15
    assert builder.config.reference_buffer_enabled is True
    assert builder.config.reference_k == 3
    assert builder.config.copy_penalty_enabled is True
    assert builder.config.copy_exact_reward == -0.5
    assert builder.config.batch_dedup_enabled is True
    assert builder.config.learnability_saturation_weight == 1.0
    assert builder.config.diversity_edit_distance_weight == 1.0


def test_physevolve_update_adds_temperature_meta_before_logprob() -> None:
    pytest = __import__("pytest")
    pytest.importorskip("verl")
    from tensordict import TensorDict
    from verl import DataProto

    trainer = object.__new__(PhysEvolveRayPPOTrainer)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(n=2, temperature=0.6),
        ),
        algorithm=SimpleNamespace(
            use_kl_in_reward=False,
            adv_estimator="grpo",
            gamma=1.0,
            lam=1.0,
            norm_adv_by_std_in_grpo=True,
        ),
        trainer=SimpleNamespace(critic_warmup=999),
    )
    trainer.use_reference_policy = False
    trainer.use_critic = False
    seen: dict[str, float] = {}

    def _old_log_prob(batch):
        assert "temperature" not in batch.batch.keys()
        seen["temperature"] = batch.meta_info["temperature"]
        return DataProto(
            batch=TensorDict(
                {"old_log_probs": torch.zeros((2, 3))},
                batch_size=[2],
            )
        ), 0.0

    trainer._compute_old_log_prob = _old_log_prob
    batch = DataProto(
        batch=TensorDict(
            {
                "responses": torch.ones((2, 3), dtype=torch.long),
                "attention_mask": torch.ones((2, 5), dtype=torch.long),
                "input_ids": torch.ones((2, 5), dtype=torch.long),
            },
            batch_size=[2],
        ),
        non_tensor_batch={"uid": np.array(["u", "u"], dtype=object)},
    )
    step = OnlineSelfPlayStep(
        step=1,
        reward_batch=build_reward_batch(
            [
                RewardRecord("gen_physics", 0.0, 1, 3),
                RewardRecord("gen_physics", 0.0, 1, 3),
            ],
            response_length=3,
        ),
        artifacts={"rollout_batch": batch, "token_level_scores": torch.zeros((2, 3))},
    )

    trainer._update_physevolve_online_step(step)

    assert seen["temperature"] == 0.6
    assert "temperature" not in batch.batch.keys()


class _FakeCheckpointManager:
    def __init__(self) -> None:
        self.events: list[tuple[str, int | None]] = []

    def update_weights(self, step: int) -> None:
        self.events.append(("wake", int(step)))

    def sleep_replicas(self) -> None:
        self.events.append(("sleep", None))


def _fake_step(step: int) -> OnlineSelfPlayStep:
    reward_batch = build_reward_batch(
        [
            RewardRecord(
                problem_type="gen_physics",
                reward=0.0,
                valid_response_length=1,
                response_length=2,
            )
        ],
        response_length=2,
    )
    return OnlineSelfPlayStep(step=step, reward_batch=reward_batch)


def _fake_executor_ok(*args: Any, **kwargs: Any):
    sim_state = _make_sim_state()
    return make_ok(sim_state.scene_id, sim_state, None, walltime_s=0.01)


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


@dataclass
class _FakePPOBatch:
    problem_types: list[str]

    @property
    def non_tensor_batch(self) -> dict[str, Any]:
        return {"problem_type": np.array(self.problem_types, dtype=object)}

    def select_idxs(self, idxs: list[int]) -> "_FakePPOBatch":
        return _FakePPOBatch([self.problem_types[i] for i in idxs])


def test_drop_gen_vqa_rows_false_returns_same_objects() -> None:
    batch = _FakePPOBatch(["gen_vqa", "pred_vqa"])
    scores = torch.arange(2, dtype=torch.float32).unsqueeze(1)
    out_batch, out_scores, skip_actor = _drop_gen_vqa_rows_for_frozen_proposer(
        batch, scores, freeze_proposer=False
    )
    assert out_batch is batch
    assert out_scores is scores
    assert skip_actor is False


def test_drop_gen_vqa_rows_true_keeps_pred_vqa_and_aligns_scores() -> None:
    batch = _FakePPOBatch(["gen_vqa", "pred_vqa", "gen_vqa", "pred_vqa"])
    scores = torch.arange(4, dtype=torch.float32).unsqueeze(1)
    out_batch, out_scores, skip_actor = _drop_gen_vqa_rows_for_frozen_proposer(
        batch, scores, freeze_proposer=True
    )
    assert skip_actor is False
    assert list(out_batch.non_tensor_batch["problem_type"]) == ["pred_vqa", "pred_vqa"]
    assert "gen_vqa" not in list(out_batch.non_tensor_batch["problem_type"])
    assert out_scores.shape[0] == 2
    assert torch.equal(out_scores.squeeze(1), torch.tensor([1.0, 3.0]))


def test_drop_gen_vqa_rows_true_skips_actor_when_no_pred_vqa_remains() -> None:
    batch = _FakePPOBatch(["gen_vqa", "gen_vqa"])
    scores = torch.zeros(2, 3)
    out_batch, out_scores, skip_actor = _drop_gen_vqa_rows_for_frozen_proposer(
        batch, scores, freeze_proposer=True
    )
    assert skip_actor is True
    assert out_batch is batch
    assert out_scores is scores
