"""Tests for PhysEvolve-native training veRL config builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from physevolve.training.verl_config import (
    PHYSEVOLVE_AGENT_LOOP_CONFIG_PATH,
    PHYSEVOLVE_AGENT_LOOP_NAME,
    PHYSEVOLVE_NULL_REWARD_FN_NAME,
    PHYSEVOLVE_NULL_REWARD_FN_PATH,
    PHYSEVOLVE_OUTPUT_ROOT,
    PHYSEVOLVE_STAGE1_PROBLEM_TYPES,
    PHYSEVOLVE_STAGE2_PROBLEM_TYPES,
    VERL_COMPAT_EXTERNAL_LIB,
    build_physevolve_stage1_verl_config,
    validate_physevolve_verl_config,
    worker_config_view,
)


def test_builder_uses_physevolve_problem_types() -> None:
    cfg = build_physevolve_stage1_verl_config(
        model_path="/models/Qwen3-VL-8B-Instruct",
        output_dir="/tmp/nz",
        group_size=2,
    )

    assert tuple(cfg["physevolve"]["problem_types"]) == PHYSEVOLVE_STAGE1_PROBLEM_TYPES
    rollout_cfg = cfg["actor_rollout_ref"]["rollout"]
    assert rollout_cfg["name"] == "vllm"
    assert rollout_cfg["mode"] == "async"
    assert rollout_cfg["tensor_model_parallel_size"] == 1
    assert rollout_cfg["load_format"] == "auto"
    assert rollout_cfg["n"] == 2
    assert rollout_cfg["enforce_eager"] is True
    assert rollout_cfg["enable_chunked_prefill"] is True
    assert rollout_cfg["enable_prefix_caching"] is True
    assert cfg["physevolve"]["rollout_safety"]["model_family"] == "qwen3_vl"
    assert cfg["physevolve"]["rollout_safety"]["qwen_vl_detected"] is True
    assert rollout_cfg["agent"]["num_workers"] == 1
    assert cfg["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 1
    assert rollout_cfg["agent"]["default_agent_loop"] == PHYSEVOLVE_AGENT_LOOP_NAME
    assert rollout_cfg["agent"]["agent_loop_config_path"] == PHYSEVOLVE_AGENT_LOOP_CONFIG_PATH
    assert Path(rollout_cfg["agent"]["agent_loop_config_path"]).exists()
    assert cfg["reward"]["custom_reward_function"] == {
        "path": PHYSEVOLVE_NULL_REWARD_FN_PATH,
        "name": PHYSEVOLVE_NULL_REWARD_FN_NAME,
    }
    assert Path(cfg["reward"]["custom_reward_function"]["path"]).exists()
    assert cfg["reward"]["reward_model"]["enable"] is False


def test_builder_defaults_to_canonical_output_root() -> None:
    cfg = build_physevolve_stage1_verl_config(group_size=2)

    assert cfg["trainer"]["default_local_dir"] == PHYSEVOLVE_OUTPUT_ROOT.as_posix()
    assert cfg["trainer"]["logger"] == ["console"]
    assert cfg["physevolve"]["stage2_branch_family_weight_multiplier"] == 1.5


def test_builder_kl_coef_sets_actor_and_reward_penalty() -> None:
    default_cfg = build_physevolve_stage1_verl_config(group_size=2)
    assert default_cfg["actor_rollout_ref"]["actor"]["kl_loss_coef"] == 0.01
    assert default_cfg["algorithm"]["kl_ctrl"]["kl_coef"] == 0.01
    assert default_cfg["physevolve"]["kl_coef"] == 0.01

    cfg = build_physevolve_stage1_verl_config(group_size=2, kl_coef=0.02)
    assert cfg["actor_rollout_ref"]["actor"]["kl_loss_coef"] == 0.02
    assert cfg["algorithm"]["kl_ctrl"]["kl_coef"] == 0.02
    assert cfg["physevolve"]["kl_coef"] == 0.02

    with pytest.raises(ValueError, match="kl_coef"):
        build_physevolve_stage1_verl_config(group_size=2, kl_coef=-0.1)


def test_builder_stage2_ablation_flags_are_opt_in() -> None:
    default_cfg = build_physevolve_stage1_verl_config(
        group_size=2, stage=2, task_modes=("vqa",)
    )
    physevolve = default_cfg["physevolve"]
    assert "disable_branch_elision" not in physevolve
    assert "freeze_proposer" not in physevolve

    branch_cfg = build_physevolve_stage1_verl_config(
        group_size=2,
        stage=2,
        task_modes=("vqa",),
        disable_branch_elision=True,
    )
    assert branch_cfg["physevolve"]["disable_branch_elision"] is True
    assert "freeze_proposer" not in branch_cfg["physevolve"]
    validate_physevolve_verl_config(branch_cfg)

    proposer_cfg = build_physevolve_stage1_verl_config(
        group_size=2,
        stage=2,
        task_modes=("vqa",),
        freeze_proposer=True,
    )
    assert proposer_cfg["physevolve"]["freeze_proposer"] is True
    assert "disable_branch_elision" not in proposer_cfg["physevolve"]
    validate_physevolve_verl_config(proposer_cfg)


def test_builder_enables_wandb_logger() -> None:
    cfg = build_physevolve_stage1_verl_config(
        group_size=2,
        project_name="nz-project",
        experiment_name="training-20",
        enable_wandb=True,
    )

    assert cfg["trainer"]["logger"] == ["console", "wandb"]
    assert cfg["trainer"]["project_name"] == "nz-project"
    assert cfg["trainer"]["experiment_name"] == "training-20"


def test_builder_stage2_solver_budget_knobs() -> None:
    cfg = build_physevolve_stage1_verl_config(
        stage=2,
        task_modes=("vqa",),
        group_size=2,
        stage2_intrinsic_enabled=True,
        stage2_intrinsic_weight=0.1,
        stage2_intrinsic_component_weights=(0.0, 0.0, 0.0, 1.0, 0.0),
        stage2_color_variants_per_scene=1,
        stage2_max_solver_questions_per_scene=3,
        stage2_max_solver_groups_per_step=24,
        stage2_min_scene_kind_picks_per_step=(
            ("collision_arena", 1),
            ("motion_lab", 1),
        ),
        stage2_camera_depth_bypass_enabled=True,
        stage2_branch_family_weight_multiplier=1.2,
    )

    assert tuple(cfg["physevolve"]["problem_types"]) == PHYSEVOLVE_STAGE2_PROBLEM_TYPES
    assert cfg["physevolve"]["stage2_intrinsic_enabled"] is True
    assert cfg["physevolve"]["stage2_intrinsic_weight"] == pytest.approx(0.1)
    assert cfg["physevolve"]["stage2_intrinsic_component_weights"] == [
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    assert cfg["physevolve"]["stage2_color_variants_per_scene"] == 1
    assert cfg["physevolve"]["stage2_max_solver_questions_per_scene"] == 3
    assert cfg["physevolve"]["stage2_max_solver_groups_per_step"] == 24
    assert cfg["physevolve"]["stage2_min_scene_kind_picks_per_step"] == {
        "collision_arena": 1,
        "motion_lab": 1,
    }
    assert cfg["physevolve"]["stage2_camera_depth_bypass_enabled"] is True
    assert cfg["physevolve"]["stage2_branch_family_weight_multiplier"] == 1.2
    validate_physevolve_verl_config(cfg)


def test_builder_stage2_anti_copy_knobs() -> None:
    cfg = build_physevolve_stage1_verl_config(
        stage=2,
        task_modes=("vqa",),
        group_size=2,
        stage2_cold_start_fewshot_steps=15,
        stage2_anti_copy_start_step=0,
        stage2_learnability_weight=0.5,
        stage2_learnability_weight_during_fewshot=0.0,
        stage2_family_difficulty_beta=1.0,
        stage2_solver_reward_scale=1.5,
        stage2_reference_buffer_enabled=True,
        stage2_reference_k=3,
        stage2_copy_penalty_enabled=True,
        stage2_copy_exact_reward=-0.5,
        stage2_copy_cap=0.1,
        stage2_copy_similarity_threshold=0.9,
        stage2_batch_dedup_enabled=True,
        stage2_learnability_saturation_threshold=0.85,
        stage2_learnability_saturation_weight=1.0,
        stage2_diversity_edit_distance_weight=1.0,
    )

    physevolve = cfg["physevolve"]
    assert physevolve["stage2_cold_start_fewshot_steps"] == 15
    assert physevolve["stage2_anti_copy_start_step"] == 0
    assert physevolve["stage2_learnability_weight"] == 0.5
    assert physevolve["stage2_learnability_weight_during_fewshot"] == 0.0
    assert physevolve["stage2_family_difficulty_beta"] == 1.0
    assert physevolve["stage2_solver_reward_scale"] == 1.5
    assert physevolve["stage2_reference_buffer_enabled"] is True
    assert physevolve["stage2_reference_k"] == 3
    assert physevolve["stage2_copy_penalty_enabled"] is True
    assert physevolve["stage2_copy_exact_reward"] == -0.5
    assert physevolve["stage2_copy_cap"] == 0.1
    assert physevolve["stage2_copy_similarity_threshold"] == 0.9
    assert physevolve["stage2_batch_dedup_enabled"] is True
    assert physevolve["stage2_learnability_saturation_threshold"] == 0.85
    assert physevolve["stage2_learnability_saturation_weight"] == 1.0
    assert physevolve["stage2_diversity_edit_distance_weight"] == 1.0
    validate_physevolve_verl_config(cfg)


def test_builder_rejects_invalid_anti_copy_knobs() -> None:
    with pytest.raises(ValueError, match="stage2_copy_similarity_threshold"):
        build_physevolve_stage1_verl_config(
            group_size=2, stage2_copy_similarity_threshold=1.5
        )
    with pytest.raises(ValueError, match="stage2_learnability_saturation_weight"):
        build_physevolve_stage1_verl_config(
            group_size=2, stage2_learnability_saturation_weight=-1.0
        )
    with pytest.raises(ValueError, match="stage2_diversity_edit_distance_weight"):
        build_physevolve_stage1_verl_config(
            group_size=2, stage2_diversity_edit_distance_weight=-1.0
        )


def test_worker_config_view_strips_physevolve_namespace() -> None:
    cfg = build_physevolve_stage1_verl_config()
    worker_cfg = worker_config_view(cfg)

    assert "physevolve" not in worker_cfg
    assert "actor_rollout_ref" in worker_cfg


def test_builder_rollout_tensor_parallel_tracks_gpu_count() -> None:
    cfg = build_physevolve_stage1_verl_config(n_gpus_per_node=2, scenes_per_batch=2)

    rollout_cfg = cfg["actor_rollout_ref"]["rollout"]
    assert rollout_cfg["tensor_model_parallel_size"] == 2


def test_builder_actor_mini_batch_tracks_data_parallel_size() -> None:
    cfg = build_physevolve_stage1_verl_config(
        n_gpus_per_node=8,
        group_size=2,
        scenes_per_batch=4,
    )

    actor_cfg = cfg["actor_rollout_ref"]["actor"]
    rollout_cfg = cfg["actor_rollout_ref"]["rollout"]
    assert actor_cfg["ppo_mini_batch_size"] == 4
    assert actor_cfg["ppo_mini_batch_size"] * rollout_cfg["n"] == 8
    validate_physevolve_verl_config(cfg)


def test_builder_rejects_actor_mini_batch_smaller_than_gpu_count() -> None:
    with pytest.raises(ValueError, match="Use --scenes-per-batch 4"):
        build_physevolve_stage1_verl_config(
            n_gpus_per_node=8,
            group_size=2,
            scenes_per_batch=2,
        )


def test_builder_accepts_multiple_agent_workers() -> None:
    cfg = build_physevolve_stage1_verl_config(agent_num_workers=8)

    assert cfg["actor_rollout_ref"]["rollout"]["agent"]["num_workers"] == 8
    validate_physevolve_verl_config(cfg)


def test_builder_defaults_qwen3vl_rollout_to_eager() -> None:
    cfg = build_physevolve_stage1_verl_config(
        model_path="/models/Qwen3-VL-8B-Instruct",
        group_size=2,
    )

    rollout_cfg = cfg["actor_rollout_ref"]["rollout"]
    assert rollout_cfg["enforce_eager"] is True
    assert cfg["physevolve"]["rollout_safety"] == {
        "model_family": "qwen3_vl",
        "qwen_vl_detected": True,
        "qwen3vl_detected": True,
        "enforce_eager": True,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
    }


def test_builder_allows_rollout_safety_overrides() -> None:
    cfg = build_physevolve_stage1_verl_config(
        model_path="/models/Qwen3-VL-8B-Instruct",
        group_size=2,
        rollout_enforce_eager=False,
        rollout_enable_chunked_prefill=False,
        rollout_enable_prefix_caching=False,
    )

    rollout_cfg = cfg["actor_rollout_ref"]["rollout"]
    assert rollout_cfg["enforce_eager"] is False
    assert rollout_cfg["enable_chunked_prefill"] is False
    assert rollout_cfg["enable_prefix_caching"] is False


def test_builder_defaults_qwen25vl_rollout_to_eager() -> None:
    cfg = build_physevolve_stage1_verl_config(
        model_path="/models/Qwen2.5-VL-7B-Instruct",
        group_size=2,
    )

    rollout_cfg = cfg["actor_rollout_ref"]["rollout"]
    safety = cfg["physevolve"]["rollout_safety"]
    assert rollout_cfg["enforce_eager"] is True
    assert safety["model_family"] == "qwen2_5_vl"
    assert safety["qwen_vl_detected"] is True
    assert safety["qwen3vl_detected"] is False


def test_builder_detects_qwen25vl_from_local_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "renamed-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type": "qwen2_5_vl", "architectures": ["Qwen2_5_VLForConditionalGeneration"]}',
        encoding="utf-8",
    )

    cfg = build_physevolve_stage1_verl_config(
        model_path=model_dir,
        group_size=2,
    )

    safety = cfg["physevolve"]["rollout_safety"]
    assert safety["model_family"] == "qwen2_5_vl"
    assert safety["qwen_vl_detected"] is True
    assert cfg["actor_rollout_ref"]["rollout"]["enforce_eager"] is True


def test_builder_keeps_non_qwenvl_rollout_graph_default() -> None:
    cfg = build_physevolve_stage1_verl_config(
        model_path="/models/Qwen2.5-7B-Instruct",
        group_size=2,
    )

    rollout_cfg = cfg["actor_rollout_ref"]["rollout"]
    assert rollout_cfg["enforce_eager"] is False
    assert cfg["physevolve"]["rollout_safety"]["model_family"] is None
    assert cfg["physevolve"]["rollout_safety"]["qwen_vl_detected"] is False
    assert cfg["physevolve"]["rollout_safety"]["qwen3vl_detected"] is False


def test_builder_model_config_excludes_legacy_pretrained_tokenizer() -> None:
    cfg = build_physevolve_stage1_verl_config(
        model_path="/models/Qwen3-VL-8B-Instruct",
        group_size=2,
    )
    model_cfg = cfg["actor_rollout_ref"]["model"]

    assert model_cfg["path"] == "/models/Qwen3-VL-8B-Instruct"
    assert model_cfg["trust_remote_code"] is True
    assert model_cfg["external_lib"] == VERL_COMPAT_EXTERNAL_LIB
    assert model_cfg["enable_gradient_checkpointing"] is True
    assert "pretrained_tokenizer" not in model_cfg


def test_validate_rejects_group_size_one() -> None:
    with pytest.raises(ValueError, match="group_size"):
        build_physevolve_stage1_verl_config(group_size=1)


def test_builder_rejects_negative_stage2_solver_budget() -> None:
    with pytest.raises(ValueError, match="stage2_intrinsic_weight"):
        build_physevolve_stage1_verl_config(
            group_size=2,
            stage2_intrinsic_weight=-0.1,
        )
    with pytest.raises(ValueError, match="stage2_intrinsic_component_weights"):
        build_physevolve_stage1_verl_config(
            group_size=2,
            stage2_intrinsic_component_weights=(0.0, 1.0),
        )
    with pytest.raises(ValueError, match="stage2_intrinsic_component_weights"):
        build_physevolve_stage1_verl_config(
            group_size=2,
            stage2_intrinsic_component_weights=(0.0, 0.0, 0.0, -1.0, 0.0),
        )
    with pytest.raises(ValueError, match="stage2_max_solver_questions_per_scene"):
        build_physevolve_stage1_verl_config(
            group_size=2,
            stage2_max_solver_questions_per_scene=-1,
        )
    with pytest.raises(ValueError, match="stage2_max_solver_groups_per_step"):
        build_physevolve_stage1_verl_config(
            group_size=2,
            stage2_max_solver_groups_per_step=-1,
        )
    with pytest.raises(ValueError, match="stage2_branch_family_weight_multiplier"):
        build_physevolve_stage1_verl_config(
            group_size=2,
            stage2_branch_family_weight_multiplier=0.0,
        )
    with pytest.raises(
        ValueError, match="stage2_min_scene_kind_picks_per_step"
    ):
        build_physevolve_stage1_verl_config(
            group_size=2,
            scenes_per_batch=4,
            stage2_min_scene_kind_picks_per_step=(
                ("collision_arena", 2),
                ("motion_lab", 3),
            ),
        )
