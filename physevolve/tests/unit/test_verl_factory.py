"""Tests for the PhysEvolve-native veRL trainer factory."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import physevolve.training.verl_factory as factory


def _fake_base_config():
    return OmegaConf.create(
        {
            "actor_rollout_ref": {"hybrid_engine": True},
            "trainer": {"device": "cuda"},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "data": {"dataloader_num_workers": 0},
        }
    )


def test_factory_config_has_no_legacy_problem_types(monkeypatch) -> None:
    monkeypatch.setattr(factory, "_load_current_verl_base_config", _fake_base_config)

    cfg = factory.build_factory_verl_config(
        {
            "model": "/models/Qwen3-VL-8B-Instruct",
            "max_steps": 2,
            "scenes_per_batch": 2,
            "group_size": 2,
        }
    )
    payload = OmegaConf.to_container(cfg, resolve=True)

    assert payload["physevolve"]["problem_types"] == [
        "gen_physics",
        "pred_forward",
        "pred_inverse",
        "pred_parameter",
    ]


def test_factory_config_accepts_config_keyword(monkeypatch) -> None:
    monkeypatch.setattr(factory, "_load_current_verl_base_config", _fake_base_config)

    cfg = factory.build_factory_verl_config(
        config={
            "model": "/models/Qwen3-VL-8B-Instruct",
            "max_steps": 2,
            "scenes_per_batch": 2,
            "group_size": 3,
        }
    )
    payload = OmegaConf.to_container(cfg, resolve=True)

    assert payload["actor_rollout_ref"]["model"]["path"] == "/models/Qwen3-VL-8B-Instruct"
    assert (
        payload["actor_rollout_ref"]["model"]["external_lib"]
        == factory.VERL_COMPAT_EXTERNAL_LIB
    )
    assert payload["actor_rollout_ref"]["rollout"]["n"] == 3


def test_factory_forwards_stage2_solver_budget(monkeypatch) -> None:
    monkeypatch.setattr(factory, "_load_current_verl_base_config", _fake_base_config)

    cfg = factory.build_factory_verl_config(
        {
            "model": "/models/Qwen3-VL-8B-Instruct",
            "stage": 2,
            "task_modes": ("vqa",),
            "max_steps": 2,
            "scenes_per_batch": 2,
            "group_size": 2,
            "stage2_intrinsic_enabled": True,
            "stage2_intrinsic_weight": 0.1,
            "stage2_intrinsic_component_weights": (0.0, 0.0, 0.0, 1.0, 0.0),
            "stage2_color_variants_per_scene": 1,
            "stage2_max_solver_questions_per_scene": 3,
            "stage2_max_solver_groups_per_step": 24,
            "stage2_min_scene_kind_picks_per_step": (
                ("collision_arena", 1),
                ("motion_lab", 1),
            ),
            "stage2_camera_depth_bypass_enabled": True,
            "stage2_branch_family_weight_multiplier": 1.2,
        }
    )
    payload = OmegaConf.to_container(cfg, resolve=True)

    assert payload["physevolve"]["problem_types"] == ["gen_vqa", "pred_vqa"]
    assert payload["physevolve"]["stage2_intrinsic_enabled"] is True
    assert payload["physevolve"]["stage2_intrinsic_weight"] == pytest.approx(0.1)
    assert payload["physevolve"]["stage2_intrinsic_component_weights"] == [
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    assert payload["physevolve"]["stage2_color_variants_per_scene"] == 1
    assert payload["physevolve"]["stage2_max_solver_questions_per_scene"] == 3
    assert payload["physevolve"]["stage2_max_solver_groups_per_step"] == 24
    assert payload["physevolve"]["stage2_min_scene_kind_picks_per_step"] == {
        "collision_arena": 1,
        "motion_lab": 1,
    }
    assert payload["physevolve"]["stage2_camera_depth_bypass_enabled"] is True
    assert payload["physevolve"]["stage2_branch_family_weight_multiplier"] == 1.2


def test_factory_forwards_opt_in_stage2_ablation_flags(monkeypatch) -> None:
    monkeypatch.setattr(factory, "_load_current_verl_base_config", _fake_base_config)

    base_data = {
        "model": "/models/Qwen3-VL-8B-Instruct",
        "stage": 2,
        "task_modes": ("vqa",),
        "max_steps": 2,
        "scenes_per_batch": 2,
        "group_size": 2,
    }
    default_cfg = factory.build_factory_verl_config(base_data)
    default_physevolve = OmegaConf.to_container(default_cfg, resolve=True)["physevolve"]
    assert "disable_branch_elision" not in default_physevolve
    assert "freeze_proposer" not in default_physevolve

    branch_cfg = factory.build_factory_verl_config(
        {**base_data, "disable_branch_elision": True}
    )
    branch_physevolve = OmegaConf.to_container(branch_cfg, resolve=True)["physevolve"]
    assert branch_physevolve["disable_branch_elision"] is True
    assert "freeze_proposer" not in branch_physevolve

    proposer_cfg = factory.build_factory_verl_config(
        {**base_data, "freeze_proposer": True}
    )
    proposer_physevolve = OmegaConf.to_container(proposer_cfg, resolve=True)["physevolve"]
    assert proposer_physevolve["freeze_proposer"] is True
    assert "disable_branch_elision" not in proposer_physevolve


def test_factory_forwards_stage2_anti_copy_knobs(monkeypatch) -> None:
    monkeypatch.setattr(factory, "_load_current_verl_base_config", _fake_base_config)

    cfg = factory.build_factory_verl_config(
        {
            "model": "/models/Qwen3-VL-8B-Instruct",
            "stage": 2,
            "task_modes": ("vqa",),
            "max_steps": 2,
            "scenes_per_batch": 2,
            "group_size": 2,
            "stage2_cold_start_fewshot_steps": 15,
            "stage2_reference_buffer_enabled": True,
            "stage2_reference_k": 3,
            "stage2_copy_penalty_enabled": True,
            "stage2_batch_dedup_enabled": True,
            "stage2_learnability_saturation_weight": 1.0,
            "stage2_diversity_edit_distance_weight": 1.0,
        }
    )
    physevolve = OmegaConf.to_container(cfg, resolve=True)["physevolve"]

    assert physevolve["stage2_cold_start_fewshot_steps"] == 15
    assert physevolve["stage2_reference_buffer_enabled"] is True
    assert physevolve["stage2_reference_k"] == 3
    assert physevolve["stage2_copy_penalty_enabled"] is True
    assert physevolve["stage2_batch_dedup_enabled"] is True
    assert physevolve["stage2_learnability_saturation_weight"] == 1.0
    assert physevolve["stage2_diversity_edit_distance_weight"] == 1.0


def test_factory_config_forwards_wandb_and_output_defaults(monkeypatch) -> None:
    monkeypatch.setattr(factory, "_load_current_verl_base_config", _fake_base_config)

    cfg = factory.build_factory_verl_config(
        {
            "model": "/models/Qwen3-VL-8B-Instruct",
            "max_steps": 2,
            "scenes_per_batch": 2,
            "group_size": 2,
            "enable_wandb": True,
            "wandb_project": "nz-project",
            "wandb_run_name": "training-20",
            "checkpoint_freq": 20,
        }
    )
    payload = OmegaConf.to_container(cfg, resolve=True)

    assert payload["trainer"]["logger"] == ["console", "wandb"]
    assert payload["trainer"]["project_name"] == "nz-project"
    assert payload["trainer"]["experiment_name"] == "training-20"
    assert payload["trainer"]["default_local_dir"] == factory.PHYSEVOLVE_OUTPUT_ROOT.as_posix()
    assert payload["trainer"]["save_freq"] == 20


def test_factory_prefers_smoke_output_dir_over_out_root(monkeypatch) -> None:
    monkeypatch.setattr(factory, "_load_current_verl_base_config", _fake_base_config)

    cfg = factory.build_factory_verl_config(
        {
            "model": "/models/Qwen3-VL-8B-Instruct",
            "out_root": "runs/selfplay_smoke",
            "output_dir": "runs/selfplay_smoke/run_123",
            "max_steps": 2,
            "scenes_per_batch": 2,
            "group_size": 2,
        }
    )
    payload = OmegaConf.to_container(cfg, resolve=True)

    assert payload["trainer"]["default_local_dir"] == "runs/selfplay_smoke/run_123"


def test_factory_config_rejects_ambiguous_arguments(monkeypatch) -> None:
    monkeypatch.setattr(factory, "_load_current_verl_base_config", _fake_base_config)

    with pytest.raises(TypeError, match="either config_or_path or config"):
        factory.build_factory_verl_config({"model": "/models/one"}, config={"model": "/models/two"})


def test_dataloader_stub_lengths_cover_configured_batches() -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "train_batch_size": 2,
                "val_batch_size": 3,
                "dataloader_num_workers": 0,
            }
        }
    )

    assert factory._dataloader_stub_lengths(cfg) == (2, 3)  # noqa: SLF001


def test_dataloader_stub_lengths_prefer_generation_batch_size() -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "gen_batch_size": 4,
                "train_batch_size": 2,
                "val_batch_size": None,
                "dataloader_num_workers": 0,
            }
        }
    )

    assert factory._dataloader_stub_lengths(cfg) == (4, 1)  # noqa: SLF001


def test_tiny_dataset_uses_requested_length() -> None:
    dataset = factory._TinyDataset(length=3)  # noqa: SLF001

    assert len(dataset) == 3
    assert dataset[2] == {}
    with pytest.raises(IndexError):
        _ = dataset[3]


def test_missing_async_rollout_manager_fails_fast() -> None:
    trainer = SimpleNamespace(fit=lambda: None)
    with pytest.raises(RuntimeError, match="async_rollout_manager"):
        factory.assert_initialized_trainer_interface(trainer)


def test_returned_trainer_interface_check_accepts_required_methods() -> None:
    trainer = SimpleNamespace(
        async_rollout_manager=SimpleNamespace(generate_sequences=lambda batch: batch),
        fit=lambda: None,
    )

    factory.assert_initialized_trainer_interface(trainer)


def test_factory_preflight_does_not_start_ray_or_load_model(monkeypatch) -> None:
    monkeypatch.setattr(factory, "_load_current_verl_base_config", _fake_base_config)
    monkeypatch.setattr(
        factory,
        "_preflight_imports",
        lambda: {"verl": True, "ray_trainer": True, "main_ppo": True, "agent_loop": True},
    )
    monkeypatch.setattr(
        factory,
        "_preflight_trl_value_head_compat",
        lambda: {"ok": True, "trl_available": True, "has_value_head": True, "patched": True},
    )

    report = factory.preflight_physevolve_verl_factory(config={"model": "/models/Qwen3"})

    assert report["ok"] is True
    assert report["loads_model"] is False
    assert report["starts_ray"] is False
    assert report["unsupported_problem_types_seen"] == []
    assert report["model_external_lib"] == factory.VERL_COMPAT_EXTERNAL_LIB
    assert report["trl_value_head_compat"]["ok"] is True
    assert report["rollout"]["agent_workers"] == 1
    assert report["trainer"]["logger"] == ["console"]
    assert report["trainer"]["save_freq"] == 2
    assert report["executor"] == {
        "executor_workers_configured": 4,
        "executor_parallelism_active": False,
    }
    assert report["agent_loop_registration"]["ok"] is True
    assert report["agent_loop_registration"]["found"] is True
    assert report["agent_loop_registration"]["target_importable"] is True
    assert (
        report["agent_loop_registration"]["target"]
        == "physevolve.training.verl_agent_loop.PhysEvolveSingleTurnAgentLoop"
    )
