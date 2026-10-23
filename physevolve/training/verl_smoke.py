"""Smoke-run configuration helpers for PhysEvolve's veRL trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from physevolve.smoke.io import append_jsonl, create_run_dir, write_json, write_yaml
from physevolve.smoke.run_config import SmokeGenerationConfig, build_run_id
from physevolve.training.verl_config import (
    PHYSEVOLVE_STAGE1_PROBLEM_TYPES,
    PHYSEVOLVE_OUTPUT_ROOT,
    build_physevolve_stage1_verl_config,
)


@dataclass(frozen=True, slots=True)
class PhysEvolveTrainerConfig:
    """Configuration for a PhysEvolve veRL smoke run."""

    model: str
    max_steps: int = 2
    scenes_per_batch: int = 2
    group_size: int = 2
    prompt_length: int = 6144
    out_root: Path = PHYSEVOLVE_OUTPUT_ROOT
    run_id: str | None = None
    execute: bool = True
    n_gpus_per_node: int = 1
    rollout_enforce_eager: bool | None = None
    rollout_enable_chunked_prefill: bool | None = None
    rollout_enable_prefix_caching: bool | None = None
    rollout_gpu_memory_utilization: float = 0.5
    rollout_max_num_seqs: int | None = None
    rollout_max_num_batched_tokens: int | None = None
    ray_num_cpus: int = 16
    checkpoint_freq: int | None = None
    executor_workers: int = 4
    experiment_name: str = "training_native_2step"
    wandb_enabled: bool = False
    wandb_project: str = "physevolve-stage1"
    wandb_run_name: str | None = None
    generation: SmokeGenerationConfig = field(default_factory=SmokeGenerationConfig)
    stage: int = 1
    task_modes: tuple[str, ...] = ("forward", "inverse", "parameter")
    # Video saving: save up to ``video_save_max_per_step`` OK scene videos
    # every ``video_save_freq`` steps to ``steps/step_XXXXXX/videos/``.
    # 0 (default) means disabled.
    video_save_freq: int = 0
    video_save_max_per_step: int = 4
    # Stage-2 only: dispatch the VQA solver rollout in fixed-size chunks so the
    # per-row video preprocessing peak stays bounded. 0 disables chunking.
    solver_rollout_chunk_size: int = 8
    # Stage-2 settings. Intrinsic/novelty shaping is OFF by
    # default; the failure penalty is scaled up for GRPO signal; the executor
    # renders N colour-permuted variants per scene to break the colour shortcut.
    stage2_intrinsic_enabled: bool = False
    stage2_intrinsic_weight: float = 0.3
    stage2_intrinsic_component_weights: tuple[float, float, float, float, float] = (
        0.30,
        0.25,
        0.20,
        0.20,
        0.05,
    )
    stage2_proposer_failure_weight: float = 1.5
    stage2_color_variants_per_scene: int = 1
    # Stage-2 Solver budget: bound unique VQA question groups before each group
    # is expanded by ``group_size`` GRPO rollouts. 0 disables each cap.
    stage2_max_solver_questions_per_scene: int = 4
    stage2_max_solver_groups_per_step: int = 32
    stage2_min_scene_kind_picks_per_step: tuple[tuple[str, int], ...] = ()
    stage2_camera_depth_bypass_enabled: bool = False
    stage2_branch_family_weight_multiplier: float = 1.5
    # Stage-2 anti-copy settings: cold-start fewshot schedule +
    # ReferenceBuffer + AST copy penalty / batch dedup / saturation / diversity.
    stage2_cold_start_fewshot_steps: int = 0
    stage2_reference_buffer_enabled: bool = False
    stage2_reference_k: int = 2
    stage2_reference_buffer_size: int = 32
    stage2_reference_buffer_during_fewshot: bool = False
    stage2_copy_penalty_enabled: bool = False
    stage2_copy_exact_reward: float = -0.5
    stage2_copy_cap: float = 0.1
    stage2_copy_similarity_threshold: float = 0.95
    stage2_batch_dedup_enabled: bool = False
    stage2_learnability_saturation_threshold: float = 0.85
    stage2_learnability_saturation_weight: float = 0.0
    stage2_diversity_edit_distance_weight: float = 0.0
    stage2_anti_copy_start_step: int | None = None
    stage2_learnability_weight: float = 1.0
    stage2_learnability_weight_during_fewshot: float | None = None
    stage2_family_difficulty_beta: float = 0.0
    stage2_solver_reward_scale: float = 1.0
    # Actor KL loss + reward KL penalty share this fixed coefficient.
    kl_coef: float = 0.01
    # The vision tower is trainable in the paper recipes.
    # Opt-in Stage-2 ablations; false omits the physevolve keys.
    disable_branch_elision: bool = False
    freeze_proposer: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "out_root", Path(self.out_root))
        object.__setattr__(self, "task_modes", tuple(self.task_modes))
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.scenes_per_batch <= 0:
            raise ValueError("scenes_per_batch must be positive")
        if self.group_size <= 1:
            raise ValueError("group_size must be greater than 1")
        if self.prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        if self.checkpoint_freq is not None and self.checkpoint_freq <= 0:
            raise ValueError("checkpoint_freq must be positive")
        if self.stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2; got {self.stage}")
        if not self.task_modes:
            raise ValueError("task_modes must be non-empty")
        if not 0.0 < self.rollout_gpu_memory_utilization <= 1.0:
            raise ValueError(
                "rollout_gpu_memory_utilization must be in (0, 1]; "
                f"got {self.rollout_gpu_memory_utilization}"
            )
        if self.rollout_max_num_seqs is not None and self.rollout_max_num_seqs <= 0:
            raise ValueError("rollout_max_num_seqs must be positive")
        if (
            self.rollout_max_num_batched_tokens is not None
            and self.rollout_max_num_batched_tokens <= 0
        ):
            raise ValueError("rollout_max_num_batched_tokens must be positive")
        if self.solver_rollout_chunk_size < 0:
            raise ValueError("solver_rollout_chunk_size must be non-negative")
        if self.stage2_intrinsic_weight < 0:
            raise ValueError("stage2_intrinsic_weight must be non-negative")
        object.__setattr__(
            self,
            "stage2_intrinsic_component_weights",
            tuple(float(w) for w in self.stage2_intrinsic_component_weights),
        )
        object.__setattr__(
            self,
            "stage2_min_scene_kind_picks_per_step",
            tuple(
                (str(scene_kind), int(count))
                for scene_kind, count in self.stage2_min_scene_kind_picks_per_step
            ),
        )
        if len(self.stage2_intrinsic_component_weights) != 5:
            raise ValueError(
                "stage2_intrinsic_component_weights must contain exactly 5 values"
            )
        if any(w < 0 for w in self.stage2_intrinsic_component_weights):
            raise ValueError(
                "stage2_intrinsic_component_weights must be non-negative"
            )
        if self.stage2_proposer_failure_weight < 0:
            raise ValueError("stage2_proposer_failure_weight must be non-negative")
        if self.stage2_color_variants_per_scene < 0:
            raise ValueError("stage2_color_variants_per_scene must be non-negative")
        if self.stage2_max_solver_questions_per_scene < 0:
            raise ValueError(
                "stage2_max_solver_questions_per_scene must be non-negative"
            )
        if self.stage2_max_solver_groups_per_step < 0:
            raise ValueError("stage2_max_solver_groups_per_step must be non-negative")
        if self.stage2_branch_family_weight_multiplier <= 0:
            raise ValueError(
                "stage2_branch_family_weight_multiplier must be positive"
            )
        scene_kind_quotas = self.stage2_min_scene_kind_picks_per_step
        if any(not scene_kind for scene_kind, _count in scene_kind_quotas):
            raise ValueError("stage2 scene-kind quota names must be non-empty")
        if any(count <= 0 for _scene_kind, count in scene_kind_quotas):
            raise ValueError("stage2 scene-kind quota counts must be positive")
        if len({scene_kind for scene_kind, _count in scene_kind_quotas}) != len(
            scene_kind_quotas
        ):
            raise ValueError("stage2 scene-kind quotas must not contain duplicate names")
        if sum(count for _scene_kind, count in scene_kind_quotas) > self.scenes_per_batch:
            raise ValueError(
                "stage2_min_scene_kind_picks_per_step exceeds scenes_per_batch"
            )
        if self.kl_coef < 0.0:
            raise ValueError("kl_coef must be non-negative")

    def resolved_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        return build_run_id(
            gate="physevolve",
            model=self.model,
            prompt_version="native_verl_trainer",
            n_samples=self.max_steps,
        )

    def to_dict(self, *, run_id: str | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload["out_root"] = str(self.out_root)
        payload["generation"]["stop_sequences"] = list(self.generation.stop_sequences)
        if run_id is not None:
            payload["resolved_run_id"] = run_id
        return payload

    def to_factory_config(self, *, output_dir: Path) -> dict[str, Any]:
        payload = self.to_dict()
        payload["output_dir"] = Path(output_dir).as_posix()
        payload["prompt_length"] = int(self.prompt_length)
        payload["max_new_tokens"] = int(self.generation.max_new_tokens)
        payload["temperature"] = float(self.generation.temperature)
        payload["top_p"] = float(self.generation.top_p)
        payload["top_k"] = int(self.generation.top_k)
        payload["enable_wandb"] = bool(self.wandb_enabled)
        payload["project_name"] = str(self.wandb_project)
        payload["experiment_name"] = str(self.wandb_run_name or self.experiment_name)
        return payload


@dataclass(frozen=True, slots=True)
class PhysEvolveTrainerSmokeRun:
    """Prepared smoke-run paths and config payload."""

    run_id: str
    run_dir: Path
    factory_config: dict[str, Any]


def prepare_physevolve_trainer_smoke(config: PhysEvolveTrainerConfig) -> PhysEvolveTrainerSmokeRun:
    """Create smoke artifacts and return the config consumed by the factory."""

    run_id = config.resolved_run_id()
    run_dir = create_run_dir(config.out_root, run_id)
    factory_config = config.to_factory_config(output_dir=run_dir)
    verl_config = build_physevolve_stage1_verl_config(
        model_path=config.model,
        output_dir=run_dir,
        max_steps=config.max_steps,
        scenes_per_batch=config.scenes_per_batch,
        group_size=config.group_size,
        response_length=config.generation.max_new_tokens,
        prompt_length=config.prompt_length,
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
        top_k=config.generation.top_k,
        n_gpus_per_node=config.n_gpus_per_node,
        rollout_enforce_eager=config.rollout_enforce_eager,
        rollout_enable_chunked_prefill=config.rollout_enable_chunked_prefill,
        rollout_enable_prefix_caching=config.rollout_enable_prefix_caching,
        rollout_gpu_memory_utilization=config.rollout_gpu_memory_utilization,
        rollout_max_num_seqs=config.rollout_max_num_seqs,
        rollout_max_num_batched_tokens=config.rollout_max_num_batched_tokens,
        ray_num_cpus=config.ray_num_cpus,
        checkpoint_freq=config.checkpoint_freq,
        executor_workers=config.executor_workers,
        project_name=config.wandb_project,
        experiment_name=config.wandb_run_name or config.experiment_name,
        enable_wandb=config.wandb_enabled,
        stage=config.stage,
        task_modes=config.task_modes,
        video_save_freq=config.video_save_freq,
        video_save_max_per_step=config.video_save_max_per_step,
        solver_rollout_chunk_size=config.solver_rollout_chunk_size,
        stage2_intrinsic_enabled=config.stage2_intrinsic_enabled,
        stage2_intrinsic_weight=config.stage2_intrinsic_weight,
        stage2_intrinsic_component_weights=config.stage2_intrinsic_component_weights,
        stage2_proposer_failure_weight=config.stage2_proposer_failure_weight,
        stage2_color_variants_per_scene=config.stage2_color_variants_per_scene,
        stage2_max_solver_questions_per_scene=(
            config.stage2_max_solver_questions_per_scene
        ),
        stage2_max_solver_groups_per_step=config.stage2_max_solver_groups_per_step,
        stage2_min_scene_kind_picks_per_step=(
            config.stage2_min_scene_kind_picks_per_step
        ),
        stage2_camera_depth_bypass_enabled=(
            config.stage2_camera_depth_bypass_enabled
        ),
        stage2_branch_family_weight_multiplier=(
            config.stage2_branch_family_weight_multiplier
        ),
        stage2_cold_start_fewshot_steps=config.stage2_cold_start_fewshot_steps,
        stage2_reference_buffer_enabled=config.stage2_reference_buffer_enabled,
        stage2_reference_k=config.stage2_reference_k,
        stage2_reference_buffer_size=config.stage2_reference_buffer_size,
        stage2_reference_buffer_during_fewshot=(
            config.stage2_reference_buffer_during_fewshot
        ),
        stage2_copy_penalty_enabled=config.stage2_copy_penalty_enabled,
        stage2_copy_exact_reward=config.stage2_copy_exact_reward,
        stage2_copy_cap=config.stage2_copy_cap,
        stage2_copy_similarity_threshold=config.stage2_copy_similarity_threshold,
        stage2_batch_dedup_enabled=config.stage2_batch_dedup_enabled,
        stage2_learnability_saturation_threshold=(
            config.stage2_learnability_saturation_threshold
        ),
        stage2_learnability_saturation_weight=(
            config.stage2_learnability_saturation_weight
        ),
        stage2_diversity_edit_distance_weight=(
            config.stage2_diversity_edit_distance_weight
        ),
        stage2_anti_copy_start_step=config.stage2_anti_copy_start_step,
        stage2_learnability_weight=config.stage2_learnability_weight,
        stage2_learnability_weight_during_fewshot=(
            config.stage2_learnability_weight_during_fewshot
        ),
        stage2_family_difficulty_beta=config.stage2_family_difficulty_beta,
        stage2_solver_reward_scale=config.stage2_solver_reward_scale,
        kl_coef=config.kl_coef,
        disable_branch_elision=config.disable_branch_elision,
        freeze_proposer=config.freeze_proposer,
    )
    write_yaml(run_dir / "config.yaml", config.to_dict(run_id=run_id))
    write_json(run_dir / "verl_config.json", verl_config)
    from physevolve.training.provenance import write_run_provenance
    write_run_provenance(run_dir, model=config.model, stage=config.stage)
    return PhysEvolveTrainerSmokeRun(
        run_id=run_id,
        run_dir=run_dir,
        factory_config=factory_config,
    )


def write_dry_run_summary(config: PhysEvolveTrainerConfig, run_dir: Path) -> dict[str, Any]:
    from physevolve.training.verl_config import PHYSEVOLVE_STAGE2_PROBLEM_TYPES

    append_jsonl(run_dir / "metrics.jsonl", {"event": "dry_run"})
    expected_problem_types = (
        PHYSEVOLVE_STAGE2_PROBLEM_TYPES if config.stage == 2 else PHYSEVOLVE_STAGE1_PROBLEM_TYPES
    )
    summary = {
        "completed": False,
        "gate": "physevolve",
        "executed": False,
        "max_steps": int(config.max_steps),
        "scenes_per_batch": int(config.scenes_per_batch),
        "group_size": int(config.group_size),
        "stage": int(config.stage),
        "physevolve_problem_types_seen": list(expected_problem_types),
        "unsupported_problem_types_seen": [],
        "reward_tensor_finite": True,
        "ppo_update_count": 0,
    }
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "run_status.json",
        {"completed": False, "error": None, "executed": False},
    )
    return summary


__all__ = [
    "PhysEvolveTrainerConfig",
    "PhysEvolveTrainerSmokeRun",
    "PHYSEVOLVE_OUTPUT_ROOT",
    "prepare_physevolve_trainer_smoke",
    "write_dry_run_summary",
]
