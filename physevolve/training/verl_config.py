"""PhysEvolve-native veRL config builder for training.

This module deliberately builds the training config from PhysEvolve/veRL concepts
with a plain configuration interface. The returned object is a
plain nested dictionary so tests and launchers can inspect it without Hydra.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from physevolve.inference.hf_backend import DEFAULT_MODEL


PHYSEVOLVE_STAGE1_PROBLEM_TYPES: tuple[str, ...] = (
    "gen_physics",
    "pred_forward",
    "pred_inverse",
    "pred_parameter",
)
PHYSEVOLVE_STAGE2_PROBLEM_TYPES: tuple[str, ...] = (
    "gen_vqa",
    "pred_vqa",
)
"""Stage-2 problem-type vocabulary (docs/architecture.md).

Stage-2 self-play has a single Proposer task (``gen_vqa`` — emit
``build_scene`` + ``get_vqa``) and a single Solver task (``pred_vqa`` —
pick A/B/C/D). Stage-2 does NOT reuse the Stage-1 dense problem types;
the Stage-2 step builder explicitly substitutes this tuple.
"""
VERL_COMPAT_EXTERNAL_LIB = "physevolve.training.verl_compat"
PHYSEVOLVE_AGENT_LOOP_NAME = "physevolve_single_turn"
PHYSEVOLVE_AGENT_LOOP_CONFIG_PATH = Path(__file__).with_name("verl_agent_loops.yaml").resolve().as_posix()
PHYSEVOLVE_NULL_REWARD_FN_PATH = Path(__file__).with_name("verl_reward_fn.py").resolve().as_posix()
PHYSEVOLVE_NULL_REWARD_FN_NAME = "physevolve_null_compute_score"
PHYSEVOLVE_OUTPUT_ROOT = Path("outputs")


def build_physevolve_stage1_verl_config(
    *,
    model_path: str = DEFAULT_MODEL,
    output_dir: str | Path = PHYSEVOLVE_OUTPUT_ROOT,
    max_steps: int = 2,
    scenes_per_batch: int = 2,
    group_size: int = 2,
    response_length: int = 4096,
    prompt_length: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    n_gpus_per_node: int = 1,
    agent_num_workers: int = 1,
    rollout_enforce_eager: bool | None = None,
    rollout_enable_chunked_prefill: bool | None = None,
    rollout_enable_prefix_caching: bool | None = None,
    rollout_gpu_memory_utilization: float = 0.5,
    rollout_max_num_seqs: int | None = None,
    rollout_max_num_batched_tokens: int | None = None,
    ray_num_cpus: int = 16,
    checkpoint_freq: int | None = None,
    project_name: str = "physevolve-stage1",
    experiment_name: str = "training_physevolve_native",
    enable_wandb: bool = False,
    executor_workers: int = 4,
    task_modes: tuple[str, ...] = ("forward", "inverse", "parameter"),
    stage: int = 1,
    problem_types: tuple[str, ...] | None = None,
    video_save_freq: int = 0,
    video_save_max_per_step: int = 4,
    solver_rollout_chunk_size: int = 8,
    stage2_intrinsic_enabled: bool = False,
    stage2_intrinsic_weight: float = 0.3,
    stage2_intrinsic_component_weights: tuple[float, float, float, float, float] = (
        0.30,
        0.25,
        0.20,
        0.20,
        0.05,
    ),
    stage2_proposer_failure_weight: float = 1.5,
    stage2_color_variants_per_scene: int = 1,
    stage2_max_solver_questions_per_scene: int = 4,
    stage2_max_solver_groups_per_step: int = 32,
    stage2_min_scene_kind_picks_per_step: tuple[tuple[str, int], ...] = (),
    stage2_camera_depth_bypass_enabled: bool = False,
    stage2_branch_family_weight_multiplier: float = 1.5,
    stage2_cold_start_fewshot_steps: int = 0,
    stage2_reference_buffer_enabled: bool = False,
    stage2_reference_k: int = 2,
    stage2_reference_buffer_size: int = 32,
    stage2_reference_buffer_during_fewshot: bool = False,
    stage2_copy_penalty_enabled: bool = False,
    stage2_copy_exact_reward: float = -0.5,
    stage2_copy_cap: float = 0.1,
    stage2_copy_similarity_threshold: float = 0.95,
    stage2_batch_dedup_enabled: bool = False,
    stage2_learnability_saturation_threshold: float = 0.85,
    stage2_learnability_saturation_weight: float = 0.0,
    stage2_diversity_edit_distance_weight: float = 0.0,
    stage2_anti_copy_start_step: int | None = None,
    stage2_learnability_weight: float = 1.0,
    stage2_learnability_weight_during_fewshot: float | None = None,
    stage2_family_difficulty_beta: float = 0.0,
    stage2_solver_reward_scale: float = 1.0,
    kl_coef: float = 0.01,
    disable_branch_elision: bool = False,
    freeze_proposer: bool = False,
) -> dict[str, Any]:
    """Return a minimal current-veRL config plus isolated PhysEvolve metadata.

    Args:
        stage: ``1`` for Stage-1 dense self-play (default), ``2`` for the
            Stage-2 VQA self-play loop. Stage-2 swaps the ``physevolve.problem_types``
            tuple to :data:`PHYSEVOLVE_STAGE2_PROBLEM_TYPES` and uses
            ``task_modes=("vqa",)`` unless ``task_modes`` is explicitly passed.
        problem_types: Override the default ``physevolve.problem_types`` tuple.
            When ``None``, Stage-1 uses :data:`PHYSEVOLVE_STAGE1_PROBLEM_TYPES`
            and Stage-2 uses :data:`PHYSEVOLVE_STAGE2_PROBLEM_TYPES`.
        rollout_gpu_memory_utilization: Fraction of each GPU vLLM reserves for
            weights + KV cache (maps to ``actor_rollout_ref.rollout.
            gpu_memory_utilization``). Lower it to leave headroom for the
            Qwen-VL video activation spike on colocated GPUs.
        rollout_max_num_seqs: Upper bound on concurrently-prefilled vLLM
            sequences (``actor_rollout_ref.rollout.max_num_seqs``). When
            ``None`` it defaults to ``scenes_per_batch * group_size`` so the
            video multimodal-merge peak stays bounded.
        rollout_max_num_batched_tokens: Cap on tokens scheduled per vLLM
            forward step (``actor_rollout_ref.rollout.max_num_batched_tokens``).
            With chunked prefill enabled this bounds the token slice that the
            Qwen-VL video ``masked_scatter`` operates on, directly shrinking
            its activation peak. When ``None`` the key is omitted so veRL's
            default (8192) applies unchanged.
        disable_branch_elision: Stage-2 ablation that drops counterfactual /
            inverse families, skips executor branch sims, and drops those
            payloads on the Solver side. Omitted from ``physevolve`` when false.
        freeze_proposer: Stage-2 ablation that still generates ``gen_vqa``
            scenes but drops those rows from the PPO update. Omitted from
            ``physevolve`` when false.
    """

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if scenes_per_batch <= 0:
        raise ValueError("scenes_per_batch must be positive")
    if group_size <= 1:
        raise ValueError("group_size must be greater than 1 for GRPO smoke")
    if agent_num_workers <= 0:
        raise ValueError("agent_num_workers must be positive")
    if response_length <= 0 or prompt_length <= 0:
        raise ValueError("prompt_length and response_length must be positive")
    if n_gpus_per_node <= 0:
        raise ValueError("n_gpus_per_node must be positive")
    if not 0.0 < float(rollout_gpu_memory_utilization) <= 1.0:
        raise ValueError(
            "rollout_gpu_memory_utilization must be in (0, 1]; "
            f"got {rollout_gpu_memory_utilization}"
        )
    if rollout_max_num_seqs is not None and int(rollout_max_num_seqs) <= 0:
        raise ValueError(
            f"rollout_max_num_seqs must be positive; got {rollout_max_num_seqs}"
        )
    if (
        rollout_max_num_batched_tokens is not None
        and int(rollout_max_num_batched_tokens) <= 0
    ):
        raise ValueError(
            "rollout_max_num_batched_tokens must be positive; "
            f"got {rollout_max_num_batched_tokens}"
        )
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2; got {stage}")
    if solver_rollout_chunk_size < 0:
        raise ValueError("solver_rollout_chunk_size must be non-negative")
    if stage2_intrinsic_weight < 0:
        raise ValueError("stage2_intrinsic_weight must be non-negative")
    stage2_intrinsic_component_weights = tuple(
        float(w) for w in stage2_intrinsic_component_weights
    )
    if len(stage2_intrinsic_component_weights) != 5:
        raise ValueError(
            "stage2_intrinsic_component_weights must contain exactly 5 values"
        )
    if any(w < 0 for w in stage2_intrinsic_component_weights):
        raise ValueError("stage2_intrinsic_component_weights must be non-negative")
    if stage2_proposer_failure_weight < 0:
        raise ValueError("stage2_proposer_failure_weight must be non-negative")
    if stage2_color_variants_per_scene < 0:
        raise ValueError("stage2_color_variants_per_scene must be non-negative")
    if stage2_max_solver_questions_per_scene < 0:
        raise ValueError("stage2_max_solver_questions_per_scene must be non-negative")
    if stage2_max_solver_groups_per_step < 0:
        raise ValueError("stage2_max_solver_groups_per_step must be non-negative")
    if stage2_branch_family_weight_multiplier <= 0:
        raise ValueError("stage2_branch_family_weight_multiplier must be positive")
    scene_kind_quotas = tuple(
        (str(scene_kind), int(count))
        for scene_kind, count in stage2_min_scene_kind_picks_per_step
    )
    if any(not scene_kind for scene_kind, _count in scene_kind_quotas):
        raise ValueError("stage2 scene-kind quota names must be non-empty")
    if any(count <= 0 for _scene_kind, count in scene_kind_quotas):
        raise ValueError("stage2 scene-kind quota counts must be positive")
    if len({scene_kind for scene_kind, _count in scene_kind_quotas}) != len(
        scene_kind_quotas
    ):
        raise ValueError("stage2 scene-kind quotas must not contain duplicate names")
    if sum(count for _scene_kind, count in scene_kind_quotas) > scenes_per_batch:
        raise ValueError(
            "stage2_min_scene_kind_picks_per_step exceeds scenes_per_batch"
        )
    if stage2_cold_start_fewshot_steps < 0:
        raise ValueError("stage2_cold_start_fewshot_steps must be non-negative")
    if stage2_reference_k < 0:
        raise ValueError("stage2_reference_k must be non-negative")
    if stage2_reference_buffer_size <= 0:
        raise ValueError("stage2_reference_buffer_size must be positive")
    if not 0.0 <= stage2_copy_similarity_threshold <= 1.0:
        raise ValueError("stage2_copy_similarity_threshold must be in [0, 1]")
    if not 0.0 <= stage2_learnability_saturation_threshold <= 1.0:
        raise ValueError(
            "stage2_learnability_saturation_threshold must be in [0, 1]"
        )
    if stage2_learnability_saturation_weight < 0:
        raise ValueError(
            "stage2_learnability_saturation_weight must be non-negative"
        )
    if stage2_diversity_edit_distance_weight < 0:
        raise ValueError(
            "stage2_diversity_edit_distance_weight must be non-negative"
        )
    if stage2_anti_copy_start_step is not None and int(stage2_anti_copy_start_step) < 0:
        raise ValueError("stage2_anti_copy_start_step must be non-negative")
    if stage2_learnability_weight < 0:
        raise ValueError("stage2_learnability_weight must be non-negative")
    if (
        stage2_learnability_weight_during_fewshot is not None
        and float(stage2_learnability_weight_during_fewshot) < 0
    ):
        raise ValueError(
            "stage2_learnability_weight_during_fewshot must be non-negative"
        )
    if stage2_family_difficulty_beta < 0:
        raise ValueError("stage2_family_difficulty_beta must be non-negative")
    if stage2_solver_reward_scale <= 0:
        raise ValueError("stage2_solver_reward_scale must be positive")
    if float(kl_coef) < 0.0:
        raise ValueError("kl_coef must be non-negative")
    resolved_kl_coef = float(kl_coef)
    if problem_types is None:
        problem_types = (
            PHYSEVOLVE_STAGE2_PROBLEM_TYPES if stage == 2 else PHYSEVOLVE_STAGE1_PROBLEM_TYPES
        )

    out = Path(output_dir).as_posix()
    save_freq = int(checkpoint_freq if checkpoint_freq is not None else max_steps)
    ppo_mini_batch_size = _minimum_ppo_mini_batch_size(
        data_parallel_size=int(n_gpus_per_node),
        group_size=int(group_size),
    )
    if int(scenes_per_batch) < ppo_mini_batch_size:
        raise ValueError(
            f"scenes_per_batch={scenes_per_batch} is too small for "
            f"n_gpus_per_node={n_gpus_per_node} and group_size={group_size}. "
            f"Use --scenes-per-batch {ppo_mini_batch_size} or larger so that "
            f"actor.ppo_mini_batch_size * rollout.n is divisible by the actor "
            f"data-parallel size."
        )
    qwen_vl_family = _qwen_vl_model_family(model_path)
    qwen3vl_detected = qwen_vl_family == "qwen3_vl"
    qwen_vl_detected = qwen_vl_family is not None
    resolved_enforce_eager = (
        qwen_vl_detected if rollout_enforce_eager is None else bool(rollout_enforce_eager)
    )
    resolved_chunked_prefill = (
        True
        if rollout_enable_chunked_prefill is None
        else bool(rollout_enable_chunked_prefill)
    )
    resolved_prefix_caching = (
        True
        if rollout_enable_prefix_caching is None
        else bool(rollout_enable_prefix_caching)
    )
    # Bound the number of concurrently-prefilled vLLM sequences. The Qwen-VL
    # video multimodal merge (``masked_scatter``) peaks with the total video
    # tokens prefilled together, so cap concurrency to the rows PhysEvolve actually
    # submits per phase (``scenes_per_batch * group_size``) instead of vLLM's
    # default 1024. This is pure rollout-side throughput/memory tuning; it does
    # not change what is generated.
    resolved_max_num_seqs = (
        int(scenes_per_batch) * int(group_size)
        if rollout_max_num_seqs is None
        else int(rollout_max_num_seqs)
    )
    resolved_gpu_memory_utilization = float(rollout_gpu_memory_utilization)
    trainer_logger = ["console", "wandb"] if enable_wandb else ["console"]
    config: dict[str, Any] = {
        "data": {
            "train_batch_size": int(scenes_per_batch),
            "val_batch_size": int(scenes_per_batch),
            "max_prompt_length": int(prompt_length),
            "max_response_length": int(response_length),
            "image_key": "images",
            "video_key": "videos",
            "trust_remote_code": True,
        },
        "actor_rollout_ref": {
            "model": {
                "path": str(model_path),
                "tokenizer": str(model_path),
                "processor": str(model_path),
                "trust_remote_code": True,
                "external_lib": VERL_COMPAT_EXTERNAL_LIB,
                "enable_gradient_checkpointing": True,
            },
            "actor": {
                "strategy": "fsdp2",
                # veRL multiplies this by rollout.n before dispatching actor
                # training; the resulting global mini-batch must divide the
                # actor FSDP data-parallel size.
                "ppo_mini_batch_size": ppo_mini_batch_size,
                "ppo_micro_batch_size_per_gpu": 1,
                "ppo_max_token_len_per_gpu": 65536,
                "use_dynamic_bsz": True,
                "use_kl_loss": True,
                "kl_loss_coef": resolved_kl_coef,
                "kl_loss_type": "low_var_kl",
                "clip_ratio": 0.2,
                "entropy_coeff": 0.0,
                "ppo_epochs": 1,
                "optim": {
                    "lr": 1.0e-6,
                    "total_training_steps": int(max_steps),
                },
            },
            "rollout": {
                "name": "vllm",
                "mode": "async",
                "tensor_model_parallel_size": int(n_gpus_per_node),
                "load_format": "auto",
                # In PhysEvolve's online path this is the GRPO group-size
                # invariant used by update/advantage. Rollout rows are already
                # expanded by PhysEvolveOnlineStepBuilder, not by veRL repeat().
                "n": int(group_size),
                "prompt_length": int(prompt_length),
                "response_length": int(response_length),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "max_model_len": int(prompt_length + response_length),
                # vLLM's Qwen-VL multimodal path can be fragile with CUDA graph
                # compilation in async rollout. Default Qwen-VL to eager while
                # keeping explicit overrides available for upstream bisects.
                "enforce_eager": resolved_enforce_eager,
                "enable_chunked_prefill": resolved_chunked_prefill,
                "enable_prefix_caching": resolved_prefix_caching,
                # Static fraction of each GPU vLLM reserves for weights + KV
                # cache. Exposed so colocated runs can leave headroom for the
                # transient Qwen-VL video activation spike.
                "gpu_memory_utilization": resolved_gpu_memory_utilization,
                # Cap concurrent prefill so the video multimodal merge peak
                # stays bounded (default vLLM value is 1024).
                "max_num_seqs": resolved_max_num_seqs,
                "do_sample": True,
                "agent": {
                    "num_workers": int(agent_num_workers),
                    "default_agent_loop": PHYSEVOLVE_AGENT_LOOP_NAME,
                    "agent_loop_config_path": PHYSEVOLVE_AGENT_LOOP_CONFIG_PATH,
                    "agent_loop_manager_class": None,
                },
            },
        },
        "critic": {
            "strategy": "fsdp2",
            "optim": {"lr": 1.0e-6, "total_training_steps": int(max_steps)},
        },
        "algorithm": {
            "adv_estimator": "grpo",
            "gamma": 1.0,
            "lam": 1.0,
            "use_kl_in_reward": True,
            "kl_penalty": "kl",
            "kl_ctrl": {"type": "fixed", "kl_coef": resolved_kl_coef},
        },
        "reward_model": {"enable": False},
        "reward": {
            # AgentLoopWorker invokes a reward loop after every generation.
            # PhysEvolve computes its real physics rewards outside that loop, so
            # use a deterministic zero-score function as infrastructure glue.
            "num_workers": 1,
            "custom_reward_function": {
                "path": PHYSEVOLVE_NULL_REWARD_FN_PATH,
                "name": PHYSEVOLVE_NULL_REWARD_FN_NAME,
            },
            "reward_model": {
                "enable": False,
                "enable_resource_pool": False,
            },
        },
        "trainer": {
            "project_name": project_name,
            "experiment_name": experiment_name,
            "total_training_steps": int(max_steps),
            "total_epochs": int(max_steps),
            "save_freq": save_freq,
            "test_freq": 0,
            "val_before_train": False,
            "nnodes": 1,
            "n_gpus_per_node": int(n_gpus_per_node),
            "default_local_dir": out,
            "logger": trainer_logger,
            "balance_batch": False,
            "critic_warmup": 0,
        },
        "ray_init": {"num_cpus": int(ray_num_cpus)},
        "physevolve": {
            "stage": int(stage),
            "problem_types": list(problem_types),
            "task_modes": list(task_modes),
            "base_model_path": str(model_path),
            "executor_workers": int(executor_workers),
            "video_save_freq": int(video_save_freq),
            "video_save_max_per_step": int(video_save_max_per_step),
            "solver_rollout_chunk_size": int(solver_rollout_chunk_size),
            "stage2_intrinsic_enabled": bool(stage2_intrinsic_enabled),
            "stage2_intrinsic_weight": float(stage2_intrinsic_weight),
            "stage2_intrinsic_component_weights": list(
                stage2_intrinsic_component_weights
            ),
            "stage2_proposer_failure_weight": float(stage2_proposer_failure_weight),
            "stage2_color_variants_per_scene": int(stage2_color_variants_per_scene),
            "stage2_max_solver_questions_per_scene": int(
                stage2_max_solver_questions_per_scene
            ),
            "stage2_max_solver_groups_per_step": int(
                stage2_max_solver_groups_per_step
            ),
            "stage2_min_scene_kind_picks_per_step": dict(scene_kind_quotas),
            "stage2_camera_depth_bypass_enabled": bool(
                stage2_camera_depth_bypass_enabled
            ),
            "stage2_branch_family_weight_multiplier": float(
                stage2_branch_family_weight_multiplier
            ),
            "stage2_cold_start_fewshot_steps": int(stage2_cold_start_fewshot_steps),
            "stage2_reference_buffer_enabled": bool(stage2_reference_buffer_enabled),
            "stage2_reference_k": int(stage2_reference_k),
            "stage2_reference_buffer_size": int(stage2_reference_buffer_size),
            "stage2_reference_buffer_during_fewshot": bool(
                stage2_reference_buffer_during_fewshot
            ),
            "stage2_copy_penalty_enabled": bool(stage2_copy_penalty_enabled),
            "stage2_copy_exact_reward": float(stage2_copy_exact_reward),
            "stage2_copy_cap": float(stage2_copy_cap),
            "stage2_copy_similarity_threshold": float(
                stage2_copy_similarity_threshold
            ),
            "stage2_batch_dedup_enabled": bool(stage2_batch_dedup_enabled),
            "stage2_learnability_saturation_threshold": float(
                stage2_learnability_saturation_threshold
            ),
            "stage2_learnability_saturation_weight": float(
                stage2_learnability_saturation_weight
            ),
            "stage2_diversity_edit_distance_weight": float(
                stage2_diversity_edit_distance_weight
            ),
            "stage2_anti_copy_start_step": (
                None
                if stage2_anti_copy_start_step is None
                else int(stage2_anti_copy_start_step)
            ),
            "stage2_learnability_weight": float(stage2_learnability_weight),
            "stage2_learnability_weight_during_fewshot": (
                None
                if stage2_learnability_weight_during_fewshot is None
                else float(stage2_learnability_weight_during_fewshot)
            ),
            "stage2_family_difficulty_beta": float(stage2_family_difficulty_beta),
            "stage2_solver_reward_scale": float(stage2_solver_reward_scale),
            "kl_coef": resolved_kl_coef,
            "rollout_safety": {
                "model_family": qwen_vl_family,
                "qwen_vl_detected": qwen_vl_detected,
                "qwen3vl_detected": qwen3vl_detected,
                "enforce_eager": resolved_enforce_eager,
                "enable_chunked_prefill": resolved_chunked_prefill,
                "enable_prefix_caching": resolved_prefix_caching,
            },
            "generation": {
                "proposer_max_new_tokens": int(response_length),
                "solver_max_new_tokens": int(response_length),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "stop": {
                    "proposer": ["</propose>"],
                    "solver": ["</answer>"],
                },
            },
        },
    }
    # Only pin max_num_batched_tokens when explicitly requested; leaving it
    # unset keeps veRL's default (8192) so Stage-1 behaviour is unchanged.
    if rollout_max_num_batched_tokens is not None:
        config["actor_rollout_ref"]["rollout"]["max_num_batched_tokens"] = int(
            rollout_max_num_batched_tokens
        )
    if disable_branch_elision:
        config["physevolve"]["disable_branch_elision"] = True
    if freeze_proposer:
        config["physevolve"]["freeze_proposer"] = True
    validate_physevolve_verl_config(config)
    return config


def _is_qwen3_vl_model_path(model_path: str | Path) -> bool:
    """Return whether a model path/config identifies Qwen3-VL."""

    return _qwen_vl_model_family(model_path) == "qwen3_vl"


def _qwen_vl_model_family(model_path: str | Path) -> str | None:
    """Return the Qwen-VL model family, preferring local HF config metadata."""

    config_family = _qwen_vl_family_from_config(model_path)
    if config_family is not None:
        return config_family
    return _qwen_vl_family_from_path(model_path)


def _qwen_vl_family_from_config(model_path: str | Path) -> str | None:
    """Read `config.json` when available to avoid path-name heuristics."""

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    model_type = str(payload.get("model_type", "")).lower()
    return _normalize_qwen_vl_family(model_type)


def _qwen_vl_family_from_path(model_path: str | Path) -> str | None:
    normalized = str(model_path).lower().replace("_", "-")
    if "qwen3-vl" in normalized or "qwen3vl" in normalized:
        return "qwen3_vl"
    if (
        "qwen2.5-vl" in normalized
        or "qwen2-5-vl" in normalized
        or "qwen25-vl" in normalized
        or "qwen25vl" in normalized
    ):
        return "qwen2_5_vl"
    if "qwen2-vl" in normalized or "qwen2vl" in normalized:
        return "qwen2_vl"
    return None


def _normalize_qwen_vl_family(model_type: str) -> str | None:
    normalized = model_type.lower().replace("-", "_")
    if normalized == "qwen3_vl":
        return "qwen3_vl"
    if normalized == "qwen2_5_vl":
        return "qwen2_5_vl"
    if normalized == "qwen2_vl":
        return "qwen2_vl"
    return None


def _minimum_ppo_mini_batch_size(*, data_parallel_size: int, group_size: int) -> int:
    """Return the smallest prompt-group mini-batch satisfying veRL DP divisibility."""

    if data_parallel_size <= 0:
        raise ValueError("data_parallel_size must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return math.lcm(int(data_parallel_size), int(group_size)) // int(group_size)


def worker_config_view(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the veRL worker-visible config with PhysEvolve-only metadata removed."""

    payload = deepcopy(dict(config))
    payload.pop("physevolve", None)
    return payload


def validate_physevolve_verl_config(config: Mapping[str, Any]) -> None:
    """Validate training sentinel constraints that must not silently drift."""

    physevolve = _mapping_at(config, "physevolve")
    stage = int(physevolve.get("stage", 1))
    if stage not in (1, 2):
        raise ValueError(f"physevolve.stage must be 1 or 2; got {stage}")
    problem_types = tuple(str(x) for x in physevolve.get("problem_types", ()))
    expected_problem_types = (
        PHYSEVOLVE_STAGE2_PROBLEM_TYPES if stage == 2 else PHYSEVOLVE_STAGE1_PROBLEM_TYPES
    )
    if problem_types != expected_problem_types:
        raise ValueError(
            f"physevolve.problem_types must be {expected_problem_types} for stage {stage}; "
            f"got {problem_types}"
        )
    rollout = _mapping_at(_mapping_at(config, "actor_rollout_ref"), "rollout")
    if rollout.get("name") != "vllm":
        raise ValueError("training requires actor_rollout_ref.rollout.name='vllm'")
    if rollout.get("mode") != "async":
        raise ValueError("training requires actor_rollout_ref.rollout.mode='async'")
    if int(rollout.get("n", 0)) <= 1:
        raise ValueError("training requires rollout.n/group_size > 1")
    agent = _mapping_at(rollout, "agent")
    if int(agent.get("num_workers", 0)) <= 0:
        raise ValueError("training requires rollout.agent.num_workers > 0")
    if str(agent.get("default_agent_loop", "")) != PHYSEVOLVE_AGENT_LOOP_NAME:
        raise ValueError(
            f"training requires rollout.agent.default_agent_loop='{PHYSEVOLVE_AGENT_LOOP_NAME}'"
        )
    agent_loop_path = str(agent.get("agent_loop_config_path", ""))
    if not agent_loop_path:
        raise ValueError("training requires rollout.agent.agent_loop_config_path")
    if not Path(agent_loop_path).exists():
        raise ValueError(f"training agent loop config does not exist: {agent_loop_path}")

    reward = _mapping_at(config, "reward")
    reward_fn = _mapping_at(reward, "custom_reward_function")
    reward_fn_path = str(reward_fn.get("path", ""))
    if reward_fn_path != PHYSEVOLVE_NULL_REWARD_FN_PATH:
        raise ValueError("training requires PhysEvolve null reward function for agent-loop scoring")
    if str(reward_fn.get("name", "")) != PHYSEVOLVE_NULL_REWARD_FN_NAME:
        raise ValueError("training requires PhysEvolve null reward function name")
    if not Path(reward_fn_path).exists():
        raise ValueError(f"PhysEvolve null reward function does not exist: {reward_fn_path}")

    model = _mapping_at(_mapping_at(config, "actor_rollout_ref"), "model")
    if not str(model.get("path", "")):
        raise ValueError("actor_rollout_ref.model.path is required")

    actor = _mapping_at(_mapping_at(config, "actor_rollout_ref"), "actor")
    trainer = _mapping_at(config, "trainer")
    data = _mapping_at(config, "data")
    ppo_mini_batch_size = _positive_int_at(
        actor,
        "ppo_mini_batch_size",
        context="actor_rollout_ref.actor",
    )
    group_size = _positive_int_at(rollout, "n", context="actor_rollout_ref.rollout")
    data_parallel_size = _positive_int_at(trainer, "n_gpus_per_node", context="trainer")
    train_batch_size = _positive_int_at(data, "train_batch_size", context="data")
    actor_global_mini_batch_size = ppo_mini_batch_size * group_size
    if actor_global_mini_batch_size % data_parallel_size != 0:
        raise ValueError(
            "actor mini-batch must be divisible by data-parallel size: "
            f"actor.ppo_mini_batch_size={ppo_mini_batch_size}, "
            f"rollout.n={group_size}, trainer.n_gpus_per_node={data_parallel_size}, "
            f"global mini-batch={actor_global_mini_batch_size}"
        )
    if train_batch_size < ppo_mini_batch_size:
        raise ValueError(
            "data.train_batch_size/scenes_per_batch must be at least "
            f"actor.ppo_mini_batch_size={ppo_mini_batch_size}; got {train_batch_size}"
        )


def _mapping_at(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{key} must be a mapping")
    return value


def _positive_int_at(config: Mapping[str, Any], key: str, *, context: str) -> int:
    """Read a required positive integer from a config mapping."""

    value = config.get(key)
    if value is None:
        raise ValueError(f"{context}.{key} is required")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{context}.{key} must be positive")
    return parsed


__all__ = [
    "PHYSEVOLVE_AGENT_LOOP_CONFIG_PATH",
    "PHYSEVOLVE_AGENT_LOOP_NAME",
    "PHYSEVOLVE_NULL_REWARD_FN_NAME",
    "PHYSEVOLVE_NULL_REWARD_FN_PATH",
    "PHYSEVOLVE_OUTPUT_ROOT",
    "PHYSEVOLVE_STAGE1_PROBLEM_TYPES",
    "PHYSEVOLVE_STAGE2_PROBLEM_TYPES",
    "VERL_COMPAT_EXTERNAL_LIB",
    "build_physevolve_stage1_verl_config",
    "validate_physevolve_verl_config",
    "worker_config_view",
]
