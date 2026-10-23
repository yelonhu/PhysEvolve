"""Factory for a PhysEvolve-native initialized veRL trainer.

The factory starts from
`build_physevolve_stage1_verl_config()`, merges onto the current veRL PPO schema, and
returns a trainer with the interface required by training.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from physevolve.inference.hf_backend import DEFAULT_MODEL
from physevolve.training.verl_config import (
    PHYSEVOLVE_AGENT_LOOP_CONFIG_PATH,
    PHYSEVOLVE_AGENT_LOOP_NAME,
    PHYSEVOLVE_OUTPUT_ROOT,
    PHYSEVOLVE_STAGE1_PROBLEM_TYPES,
    PHYSEVOLVE_STAGE2_PROBLEM_TYPES,
    VERL_COMPAT_EXTERNAL_LIB,
    build_physevolve_stage1_verl_config,
    validate_physevolve_verl_config,
)


class _TinyDataset:
    """Minimal dataset used only so RayPPOTrainer can construct dataloaders."""

    def __init__(self, length: int = 1) -> None:
        self._length = max(1, int(length))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self._length:
            raise IndexError(index)
        return {}


def create_initialized_physevolve_verl_trainer(
    config_or_path: Any = None,
    *,
    config: Any = None,
) -> Any:
    """Create, initialize, and return a PhysEvolve-native veRL trainer.

    This function performs heavyweight work: it may initialize Ray, load the
    tokenizer/processor, initialize workers, and start vLLM rollout servers. Use
    `preflight_physevolve_verl_factory()` for import/config/interface checks only.
    """

    cfg = build_factory_verl_config(config_or_path, config=config)
    _assert_supported_problem_types(cfg)

    import ray
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import TaskRunner
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.fs import copy_to_local
    from verl.utils.dataset.rl_dataset import collate_fn

    from physevolve.training.physevolve_ray_trainer import PhysEvolveRayPPOTrainer

    import physevolve.training.verl_agent_loop  # noqa: F401 - registers agent loop.

    if not ray.is_initialized():
        ray.init(num_cpus=int(OmegaConf.select(cfg, "ray_init.num_cpus", default=16)))

    from physevolve.smoke.io import write_json
    write_json(Path(str(cfg.trainer.default_local_dir)) / "resolved_verl_config.json",
               OmegaConf.to_container(cfg, resolve=True))

    runner = TaskRunner()
    actor_rollout_cls, ray_worker_group_cls = runner.add_actor_rollout_worker(cfg)
    runner.add_critic_worker(cfg)
    runner.add_reward_model_resource_pool(cfg)
    runner.add_teacher_model_resource_pool(cfg)
    runner.add_ref_policy_worker(cfg, actor_rollout_cls)

    local_path = copy_to_local(
        cfg.actor_rollout_ref.model.path,
        use_shm=cfg.actor_rollout_ref.model.get("use_shm", False),
    )
    trust_remote_code = cfg.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    resource_pool_manager = runner.init_resource_pool_mgr(cfg)
    train_stub_len, val_stub_len = _dataloader_stub_lengths(cfg)
    trainer = PhysEvolveRayPPOTrainer(
        config=cfg,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=runner.role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        train_dataset=_TinyDataset(train_stub_len),
        val_dataset=_TinyDataset(val_stub_len),
        collate_fn=collate_fn,
    )
    trainer.init_workers()
    assert_initialized_trainer_interface(trainer)
    return trainer


def preflight_physevolve_verl_factory(
    config_or_path: Any = None,
    *,
    config: Any = None,
) -> dict[str, Any]:
    """Check config/import/interface boundaries without loading a model."""

    cfg = build_factory_verl_config(config_or_path, config=config)
    _assert_supported_problem_types(cfg)
    imports = _preflight_imports()
    trl_value_head_compat = _preflight_trl_value_head_compat()
    agent_loop_registration = _preflight_agent_loop_registration(cfg)
    return {
        "ok": bool(
            all(imports.values())
            and trl_value_head_compat.get("ok")
            and agent_loop_registration.get("ok")
        ),
        "mode": "preflight",
        "imports": imports,
        "physevolve_problem_types": list(cfg.physevolve.problem_types),
        "unsupported_problem_types_seen": [],
        "rollout": {
            "name": cfg.actor_rollout_ref.rollout.name,
            "mode": cfg.actor_rollout_ref.rollout.mode,
            "n": int(cfg.actor_rollout_ref.rollout.n),
            "enforce_eager": bool(cfg.actor_rollout_ref.rollout.enforce_eager),
            "enable_chunked_prefill": bool(
                cfg.actor_rollout_ref.rollout.enable_chunked_prefill
            ),
            "enable_prefix_caching": bool(
                cfg.actor_rollout_ref.rollout.enable_prefix_caching
            ),
            "agent_loop": cfg.actor_rollout_ref.rollout.agent.default_agent_loop,
            "agent_workers": int(cfg.actor_rollout_ref.rollout.agent.num_workers),
            "agent_loop_config_path": cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path,
        },
        "trainer": {
            "default_local_dir": str(cfg.trainer.default_local_dir),
            "logger": list(cfg.trainer.logger),
            "project_name": str(cfg.trainer.project_name),
            "experiment_name": str(cfg.trainer.experiment_name),
            "save_freq": int(cfg.trainer.save_freq),
        },
        "executor": {
            "executor_workers_configured": int(cfg.physevolve.executor_workers),
            "executor_parallelism_active": False,
        },
        "model_external_lib": cfg.actor_rollout_ref.model.get("external_lib"),
        "expected_model_external_lib": VERL_COMPAT_EXTERNAL_LIB,
        "trl_value_head_compat": trl_value_head_compat,
        "agent_loop_registration": agent_loop_registration,
        "trainer_interface_required": [
            "async_rollout_manager.generate_sequences",
            "PhysEvolveRayPPOTrainer.fit",
        ],
        "loads_model": False,
        "starts_ray": False,
    }


def build_factory_verl_config(config_or_path: Any = None, *, config: Any = None) -> Any:
    """Build current-veRL OmegaConf config from the PhysEvolve-native builder."""

    from omegaconf import OmegaConf

    native = _native_config(_resolve_config_arg(config_or_path, config))
    validate_physevolve_verl_config(native)
    base = _load_current_verl_base_config()
    cfg = OmegaConf.merge(base, native)
    OmegaConf.resolve(cfg)
    validate_physevolve_verl_config(OmegaConf.to_container(cfg, resolve=True))
    return cfg


def _resolve_config_arg(config_or_path: Any = None, config: Any = None) -> Any:
    """Normalize positional and keyword config entrypoints."""

    if config_or_path is not None and config is not None:
        raise TypeError("pass either config_or_path or config, not both")
    return config if config is not None else config_or_path


def _dataloader_stub_lengths(cfg: Any) -> tuple[int, int]:
    """Return train/validation stub lengths that satisfy veRL dataloader checks."""

    train_batch_size = _positive_config_int(cfg, "data.gen_batch_size")
    if train_batch_size is None:
        train_batch_size = _positive_config_int(cfg, "data.train_batch_size", default=1)
    val_batch_size = _positive_config_int(cfg, "data.val_batch_size", default=1)
    return train_batch_size, val_batch_size


def _positive_config_int(cfg: Any, key: str, default: int | None = None) -> int | None:
    from omegaconf import OmegaConf

    value = OmegaConf.select(cfg, key, default=default)
    if value is None:
        if default is None:
            return None
        return max(1, int(default))
    return max(1, int(value))


def assert_initialized_trainer_interface(trainer: Any) -> None:
    """Fail fast if a returned trainer cannot support training."""

    manager = getattr(trainer, "async_rollout_manager", None)
    if manager is None or not callable(getattr(manager, "generate_sequences", None)):
        raise RuntimeError(
            "initialized trainer must expose async_rollout_manager.generate_sequences"
        )
    if not callable(getattr(trainer, "fit", None)):
        raise RuntimeError("initialized trainer must expose fit()")


def _native_config(config_or_path: Any) -> dict[str, Any]:
    if config_or_path is None:
        return build_physevolve_stage1_verl_config()
    if is_dataclass(config_or_path):
        data = asdict(config_or_path)
    elif isinstance(config_or_path, (str, Path)):
        from omegaconf import OmegaConf

        data = OmegaConf.to_container(OmegaConf.load(config_or_path), resolve=True)
    elif isinstance(config_or_path, Mapping):
        data = dict(config_or_path)
    else:
        raise TypeError(
            "config_or_path must be None, a dataclass, mapping, or path; "
            f"got {type(config_or_path).__name__}"
        )

    if "physevolve" in data and "actor_rollout_ref" in data:
        model_path = str(data.get("physevolve", {}).get("base_model_path", DEFAULT_MODEL))
        seed = build_physevolve_stage1_verl_config(
            model_path=model_path or DEFAULT_MODEL
        )
        return _deep_merge(seed, data)

    model_path = str(data.get("model", data.get("model_path", DEFAULT_MODEL)))
    # Determine stage and task_modes from the flat dict.  These fields are
    # serialised by PhysEvolveTrainerConfig.to_dict() (asdict) and must be
    # forwarded here; forgetting them silently falls back to Stage-1 defaults.
    stage = int(data.get("stage", 1))
    raw_task_modes = data.get("task_modes")
    if raw_task_modes is not None:
        task_modes: tuple[str, ...] = tuple(str(m) for m in raw_task_modes)
    else:
        task_modes = ("vqa",) if stage == 2 else ("forward", "inverse", "parameter")
    return build_physevolve_stage1_verl_config(
        model_path=model_path or DEFAULT_MODEL,
        output_dir=data.get("output_dir", data.get("out_root", PHYSEVOLVE_OUTPUT_ROOT)),
        max_steps=int(data.get("max_steps", 2)),
        scenes_per_batch=int(data.get("scenes_per_batch", 2)),
        group_size=int(data.get("group_size", 2)),
        response_length=int(
            data.get("max_new_tokens", data.get("response_length", 4096))
        ),
        prompt_length=int(data.get("prompt_length", 4096)),
        temperature=float(data.get("temperature", 0.6)),
        top_p=float(data.get("top_p", 0.95)),
        top_k=int(data.get("top_k", 20)),
        n_gpus_per_node=int(data.get("n_gpus_per_node", 1)),
        agent_num_workers=int(data.get("agent_num_workers", data.get("num_workers", 1))),
        rollout_enforce_eager=_optional_bool(data, "rollout_enforce_eager"),
        rollout_enable_chunked_prefill=_optional_bool(
            data, "rollout_enable_chunked_prefill"
        ),
        rollout_enable_prefix_caching=_optional_bool(
            data, "rollout_enable_prefix_caching"
        ),
        rollout_gpu_memory_utilization=float(
            data.get("rollout_gpu_memory_utilization", 0.5)
        ),
        rollout_max_num_seqs=_optional_int(data, "rollout_max_num_seqs"),
        rollout_max_num_batched_tokens=_optional_int(
            data, "rollout_max_num_batched_tokens"
        ),
        ray_num_cpus=int(data.get("ray_num_cpus", 16)),
        checkpoint_freq=_optional_int(data, "checkpoint_freq"),
        executor_workers=int(data.get("executor_workers", 4)),
        video_save_freq=int(data.get("video_save_freq", 0)),
        video_save_max_per_step=int(data.get("video_save_max_per_step", 4)),
        solver_rollout_chunk_size=int(data.get("solver_rollout_chunk_size", 8)),
        stage2_intrinsic_enabled=bool(data.get("stage2_intrinsic_enabled", False)),
        stage2_intrinsic_weight=float(data.get("stage2_intrinsic_weight", 0.3)),
        stage2_intrinsic_component_weights=_stage2_component_weights_from_data(data),
        stage2_proposer_failure_weight=float(
            data.get("stage2_proposer_failure_weight", 1.5)
        ),
        stage2_color_variants_per_scene=int(
            data.get("stage2_color_variants_per_scene", 1)
        ),
        stage2_max_solver_questions_per_scene=int(
            data.get("stage2_max_solver_questions_per_scene", 4)
        ),
        stage2_max_solver_groups_per_step=int(
            data.get("stage2_max_solver_groups_per_step", 32)
        ),
        stage2_min_scene_kind_picks_per_step=_stage2_scene_kind_pick_quotas_from_data(
            data
        ),
        stage2_camera_depth_bypass_enabled=bool(
            data.get("stage2_camera_depth_bypass_enabled", False)
        ),
        stage2_branch_family_weight_multiplier=float(
            data.get("stage2_branch_family_weight_multiplier", 1.5)
        ),
        stage2_cold_start_fewshot_steps=int(
            data.get("stage2_cold_start_fewshot_steps", 0)
        ),
        stage2_reference_buffer_enabled=bool(
            data.get("stage2_reference_buffer_enabled", False)
        ),
        stage2_reference_k=int(data.get("stage2_reference_k", 2)),
        stage2_reference_buffer_size=int(
            data.get("stage2_reference_buffer_size", 32)
        ),
        stage2_reference_buffer_during_fewshot=bool(
            data.get("stage2_reference_buffer_during_fewshot", False)
        ),
        stage2_copy_penalty_enabled=bool(
            data.get("stage2_copy_penalty_enabled", False)
        ),
        stage2_copy_exact_reward=float(data.get("stage2_copy_exact_reward", -0.5)),
        stage2_copy_cap=float(data.get("stage2_copy_cap", 0.1)),
        stage2_copy_similarity_threshold=float(
            data.get("stage2_copy_similarity_threshold", 0.95)
        ),
        stage2_batch_dedup_enabled=bool(
            data.get("stage2_batch_dedup_enabled", False)
        ),
        stage2_learnability_saturation_threshold=float(
            data.get("stage2_learnability_saturation_threshold", 0.85)
        ),
        stage2_learnability_saturation_weight=float(
            data.get("stage2_learnability_saturation_weight", 0.0)
        ),
        stage2_diversity_edit_distance_weight=float(
            data.get("stage2_diversity_edit_distance_weight", 0.0)
        ),
        stage2_anti_copy_start_step=(
            None
            if data.get("stage2_anti_copy_start_step", None) is None
            else int(data["stage2_anti_copy_start_step"])
        ),
        stage2_learnability_weight=float(
            data.get("stage2_learnability_weight", 1.0)
        ),
        stage2_learnability_weight_during_fewshot=(
            None
            if data.get("stage2_learnability_weight_during_fewshot", None) is None
            else float(data["stage2_learnability_weight_during_fewshot"])
        ),
        stage2_family_difficulty_beta=float(
            data.get("stage2_family_difficulty_beta", 0.0)
        ),
        stage2_solver_reward_scale=float(
            data.get("stage2_solver_reward_scale", 1.0)
        ),
        kl_coef=float(data.get("kl_coef", 0.01)),
        disable_branch_elision=bool(data.get("disable_branch_elision", False)),
        freeze_proposer=bool(data.get("freeze_proposer", False)),
        stage=stage,
        task_modes=task_modes,
        project_name=str(data.get("wandb_project", data.get("project_name", "physevolve-stage1"))),
        experiment_name=str(
            data.get("wandb_run_name")
            or data.get("experiment_name", "training_native_2step")
        ),
        enable_wandb=bool(
            data.get("enable_wandb", data.get("wandb_enabled", data.get("wandb", False)))
        ),
    )


def _optional_bool(data: Mapping[str, Any], key: str) -> bool | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _optional_int(data: Mapping[str, Any], key: str) -> int | None:
    """Read an optional integer from a config mapping."""

    value = data.get(key)
    if value is None:
        return None
    return int(value)


def _stage2_component_weights_from_data(
    data: Mapping[str, Any],
) -> tuple[float, float, float, float, float]:
    """Read Stage-2 intrinsic component weights from flat smoke config data."""

    raw = data.get(
        "stage2_intrinsic_component_weights",
        (0.30, 0.25, 0.20, 0.20, 0.05),
    )
    if isinstance(raw, str):
        values = tuple(float(part.strip()) for part in raw.split(","))
    else:
        values = tuple(float(part) for part in raw)
    if len(values) != 5:
        raise ValueError(
            "stage2_intrinsic_component_weights must contain exactly 5 values"
        )
    return values  # type: ignore[return-value]


def _stage2_scene_kind_pick_quotas_from_data(
    data: Mapping[str, Any],
) -> tuple[tuple[str, int], ...]:
    """Read Stage-2 scene-kind pick minimums from flat smoke config data."""

    raw = data.get("stage2_min_scene_kind_picks_per_step", ())
    if isinstance(raw, str):
        pairs: list[tuple[str, int]] = []
        for entry in raw.split(","):
            scene_kind, separator, raw_count = entry.strip().partition("=")
            if not separator:
                raise ValueError(
                    "stage2_min_scene_kind_picks_per_step entries must use name=count"
                )
            pairs.append((scene_kind.strip(), int(raw_count.strip())))
        return tuple(pairs)
    if isinstance(raw, Mapping):
        return tuple((str(key), int(count)) for key, count in raw.items())
    return tuple((str(key), int(count)) for key, count in raw)


def _load_current_verl_base_config() -> Any:
    from omegaconf import OmegaConf

    try:
        import verl
    except Exception as exc:  # pragma: no cover - environment specific.
        raise RuntimeError("veRL must be importable for trainer factory preflight") from exc

    root = Path(verl.__file__).resolve().parent
    base_path = root / "trainer" / "config" / "_generated_ppo_trainer.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"current veRL generated PPO config not found: {base_path}")
    return OmegaConf.load(base_path)


def _preflight_imports() -> dict[str, bool]:
    modules = {
        "verl": "verl",
        "ray_trainer": "verl.trainer.ppo.ray_trainer",
        "main_ppo": "verl.trainer.main_ppo",
        "agent_loop": "verl.experimental.agent_loop.agent_loop",
        "qwen3_vl": "transformers.models.qwen3_vl",
    }
    out: dict[str, bool] = {}
    for key, module_name in modules.items():
        try:
            __import__(module_name)
        except Exception:
            out[key] = False
        else:
            out[key] = True
    return out


def _preflight_trl_value_head_compat() -> dict[str, Any]:
    """Check the TRL compatibility shim without loading model weights."""

    try:
        from physevolve.training.verl_compat import trl_value_head_compat_status
    except Exception as exc:  # pragma: no cover - import environment specific.
        return {
            "ok": False,
            "trl_available": False,
            "has_value_head": False,
            "patched": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    status = trl_value_head_compat_status()
    return {"ok": bool(status.get("has_value_head") or not status.get("trl_available")), **status}


def _preflight_agent_loop_registration(cfg: Any) -> dict[str, Any]:
    """Check that Ray agent-loop workers can load PhysEvolve's custom loop."""

    from omegaconf import OmegaConf

    path = str(
        OmegaConf.select(
            cfg,
            "actor_rollout_ref.rollout.agent.agent_loop_config_path",
            default="",
        )
        or ""
    )
    expected_name = str(
        OmegaConf.select(
            cfg,
            "actor_rollout_ref.rollout.agent.default_agent_loop",
            default=PHYSEVOLVE_AGENT_LOOP_NAME,
        )
        or ""
    )
    status: dict[str, Any] = {
        "ok": False,
        "expected_name": expected_name,
        "expected_config_path": PHYSEVOLVE_AGENT_LOOP_CONFIG_PATH,
        "config_path": path,
        "config_exists": False,
        "found": False,
        "target": None,
        "target_importable": False,
        "error": None,
    }
    if not path:
        status["error"] = "agent_loop_config_path is empty"
        return status

    config_path = Path(path)
    status["config_exists"] = config_path.exists()
    if not status["config_exists"]:
        status["error"] = f"agent_loop_config_path does not exist: {path}"
        return status

    try:
        configs = OmegaConf.load(config_path)
        for entry in configs:
            if str(entry.get("name", "")) != expected_name:
                continue
            status["found"] = True
            target = str(entry.get("_target_", ""))
            status["target"] = target
            _import_target(target)
            status["target_importable"] = True
            status["ok"] = True
            return status
        status["error"] = f"agent loop {expected_name!r} not found in {path}"
    except Exception as exc:  # pragma: no cover - depends on optional veRL import surface.
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def _import_target(target: str) -> object:
    """Import a Hydra `_target_` path without instantiating the class."""

    module_name, _, attr_path = target.rpartition(".")
    if not module_name or not attr_path:
        raise ValueError(f"invalid Hydra target path: {target!r}")
    obj: object = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def _assert_supported_problem_types(cfg: Any) -> None:
    from omegaconf import OmegaConf

    payload = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
    physevolve_block = payload.get("physevolve", {})
    stage = int(physevolve_block.get("stage", 1))
    problem_types = tuple(physevolve_block.get("problem_types", ()))
    expected = (
        PHYSEVOLVE_STAGE2_PROBLEM_TYPES if stage == 2 else PHYSEVOLVE_STAGE1_PROBLEM_TYPES
    )
    if problem_types != expected:
        raise ValueError(
            f"physevolve.problem_types must be {expected} for stage {stage}; "
            f"got {problem_types}"
        )


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


__all__ = [
    "assert_initialized_trainer_interface",
    "build_factory_verl_config",
    "create_initialized_physevolve_verl_trainer",
    "preflight_physevolve_verl_factory",
]
