"""self-play veRL trainer for PhysEvolve online self-play."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from time import perf_counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4

import numpy as np

from physevolve.inference.hf_backend import DEFAULT_MODEL, GenerationResult
from physevolve.integrations.video_bridge import VLLM_VIDEO_METADATA_KEYS, VideoBundle
from physevolve.smoke.io import append_jsonl, write_json
from physevolve.smoke.run_config import SmokeGenerationConfig
from physevolve.training.online_step_builder_stage2 import (
    PhysEvolveStage2StepBuilder,
    PhysEvolveStage2StepBuilderConfig,
)
from physevolve.training.online_step_builder import (
    GenerationSpec,
    PhysEvolveOnlineStepBuilder,
    PhysEvolveOnlineStepBuilderConfig,
)
from physevolve.training.rollout_invariants import (
    response_lengths_from_mask,
    validate_multimodal_token_alignment,
    validate_update_batch,
)
from physevolve.training.verl_config import PHYSEVOLVE_OUTPUT_ROOT

try:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer as _RayPPOTrainer

    _VERL_TRAINER_AVAILABLE = True
except Exception:  # pragma: no cover - import-only fallback for lightweight tests.
    _RayPPOTrainer = object  # type: ignore[assignment, misc]
    _VERL_TRAINER_AVAILABLE = False


class PhysEvolveRayPPOTrainer(_RayPPOTrainer):  # type: ignore[misc, valid-type]
    """Run PhysEvolve self-play inside veRL's PPO trainer boundary."""

    def fit(self) -> tuple[list[dict[str, Any]], int]:  # type: ignore[override]
        """Execute PhysEvolve online self-play and PPO updates.

        This mirrors the shared-policy design: the trainer owns generation, reward
        attachment, advantage computation, actor update, checkpoints, and vLLM
        lifecycle. PhysEvolve only replaces the source of rollout batches.
        """

        self.global_steps = int(getattr(self, "global_steps", 0))
        self._load_checkpoint()

        rows: list[dict[str, Any]] = []
        update_count = 0
        builder = self._physevolve_step_builder()
        max_steps = int(_select(self.config, "trainer.total_training_steps", 1) or 1)
        tracking_logger = self._physevolve_tracking_logger()

        for step in range(self.global_steps + 1, max_steps + 1):
            step_start = perf_counter()
            self.global_steps = step - 1
            self._physevolve_rollout_wake(self.global_steps)
            try:
                online_step = builder(step)
            finally:
                self._physevolve_rollout_sleep()

            update_metrics = self._update_physevolve_online_step(online_step)
            if bool(update_metrics.get("updated", True)):
                update_count += 1
            self.global_steps = step
            checkpoint_s = 0.0
            if int(_select(self.config, "trainer.save_freq", 0) or 0) > 0:
                if step % int(self.config.trainer.save_freq) == 0:
                    checkpoint_start = perf_counter()
                    self._save_checkpoint()
                    checkpoint_s = perf_counter() - checkpoint_start
            step_total_s = perf_counter() - step_start
            throughput_metrics = _throughput_metrics(online_step.metrics, step_total_s)
            timing_metrics = {
                "time/step_total_s": float(step_total_s),
                "time/proposer_rollout_s": float(
                    online_step.metrics.get("time/proposer_rollout_s", 0.0)
                ),
                "time/executor_s": float(online_step.metrics.get("time/executor_s", 0.0)),
                "time/solver_rollout_s": float(
                    online_step.metrics.get("time/solver_rollout_s", 0.0)
                ),
                "time/old_logprob_s": float(update_metrics.get("time/old_logprob_s", 0.0)),
                "time/ref_logprob_s": float(update_metrics.get("time/ref_logprob_s", 0.0)),
                "time/update_actor_s": float(
                    update_metrics.get("time/update_actor_s", 0.0)
                ),
                "time/checkpoint_s": float(checkpoint_s),
            }
            row = {
                "step": int(step),
                **timing_metrics,
                **throughput_metrics,
                "reward_batch": online_step.reward_batch.metrics,
                "online": online_step.metrics,
                "update": update_metrics,
            }
            rows.append(row)
            append_jsonl(self._physevolve_run_dir() / "metrics.jsonl", row)
            if tracking_logger is not None:
                tracking_logger.log(data=_flatten_tracking_metrics(row), step=step)

        self.physevolve_fit_rows = rows
        self.physevolve_ppo_update_count = update_count
        self._write_physevolve_summary(rows, update_count)
        if tracking_logger is not None:
            del tracking_logger
        return rows, update_count

    def _physevolve_step_builder(self):
        """Return a step builder appropriate for the configured stage.

        Stage-1 (default) returns :class:`PhysEvolveOnlineStepBuilder`; Stage-2
        returns :class:`PhysEvolveStage2StepBuilder`. Both honour the same
        ``__call__(step) -> OnlineSelfPlayStep`` interface so the caller
        cannot tell them apart.
        """

        builder = getattr(self, "_physevolve_cached_step_builder", None)
        if builder is not None:
            return builder

        stage = int(_select(self.config, "physevolve.stage", 1) or 1)
        scenes_per_batch = int(_select(self.config, "data.train_batch_size", 1) or 1)
        group_size = int(_select(self.config, "actor_rollout_ref.rollout.n", 1) or 1)
        response_length = int(
            _select(self.config, "actor_rollout_ref.rollout.response_length", 4096) or 4096
        )
        executor_workers = int(_select(self.config, "physevolve.executor_workers", 1) or 1)
        generation = SmokeGenerationConfig(
            max_new_tokens=int(
                _select(self.config, "physevolve.generation.proposer_max_new_tokens", 4096)
                or 4096
            ),
            temperature=float(_select(self.config, "physevolve.generation.temperature", 0.6) or 0.6),
            top_p=float(_select(self.config, "physevolve.generation.top_p", 0.95) or 0.95),
            top_k=int(_select(self.config, "physevolve.generation.top_k", 20) or 20),
            stop_sequences=(
                "</propose>",
                "</answer>",
            ),
        )
        model = str(_select(self.config, "physevolve.base_model_path", DEFAULT_MODEL))
        video_save_freq = int(
            _select(self.config, "physevolve.video_save_freq", 0) or 0
        )
        video_save_max_per_step = int(
            _select(self.config, "physevolve.video_save_max_per_step", 4) or 4
        )

        if stage == 2:
            solver_rollout_chunk_size = int(
                _select(self.config, "physevolve.solver_rollout_chunk_size", 8) or 0
            )
            intrinsic_enabled = bool(
                _select(self.config, "physevolve.stage2_intrinsic_enabled", False)
            )
            intrinsic_weight_raw = _select(
                self.config, "physevolve.stage2_intrinsic_weight", 0.3
            )
            intrinsic_weight = float(
                0.3 if intrinsic_weight_raw is None else intrinsic_weight_raw
            )
            intrinsic_component_weights = _stage2_component_weights(
                _select(
                    self.config,
                    "physevolve.stage2_intrinsic_component_weights",
                    (0.30, 0.25, 0.20, 0.20, 0.05),
                )
            )
            proposer_failure_weight = float(
                _select(self.config, "physevolve.stage2_proposer_failure_weight", 1.5)
                or 1.5
            )
            color_variants_per_scene = int(
                _select(self.config, "physevolve.stage2_color_variants_per_scene", 1) or 0
            )
            max_solver_questions_per_scene = int(
                _select(
                    self.config,
                    "physevolve.stage2_max_solver_questions_per_scene",
                    4,
                )
                or 0
            )
            max_solver_groups_per_step = int(
                _select(
                    self.config,
                    "physevolve.stage2_max_solver_groups_per_step",
                    32,
                )
                or 0
            )
            min_scene_kind_picks_per_step = _stage2_scene_kind_pick_quotas(
                _select(
                    self.config,
                    "physevolve.stage2_min_scene_kind_picks_per_step",
                    {},
                )
            )
            camera_depth_bypass_enabled = bool(
                _select(
                    self.config,
                    "physevolve.stage2_camera_depth_bypass_enabled",
                    False,
                )
            )
            disable_branch_elision = bool(
                _select(self.config, "physevolve.disable_branch_elision", False)
            )
            branch_family_weight_multiplier = float(
                _select(
                    self.config,
                    "physevolve.stage2_branch_family_weight_multiplier",
                    1.5,
                )
                or 1.5
            )
            cold_start_fewshot_steps = int(
                _select(self.config, "physevolve.stage2_cold_start_fewshot_steps", 0) or 0
            )
            reference_buffer_enabled = bool(
                _select(self.config, "physevolve.stage2_reference_buffer_enabled", False)
            )
            reference_k = int(
                _select(self.config, "physevolve.stage2_reference_k", 2) or 0
            )
            reference_buffer_size = int(
                _select(self.config, "physevolve.stage2_reference_buffer_size", 32) or 32
            )
            reference_buffer_during_fewshot = bool(
                _select(
                    self.config,
                    "physevolve.stage2_reference_buffer_during_fewshot",
                    False,
                )
            )
            copy_penalty_enabled = bool(
                _select(self.config, "physevolve.stage2_copy_penalty_enabled", False)
            )
            copy_exact_reward = float(
                _select(self.config, "physevolve.stage2_copy_exact_reward", -0.5)
            )
            copy_cap = float(
                _select(self.config, "physevolve.stage2_copy_cap", 0.1)
            )
            copy_similarity_threshold = float(
                _select(self.config, "physevolve.stage2_copy_similarity_threshold", 0.95)
            )
            batch_dedup_enabled = bool(
                _select(self.config, "physevolve.stage2_batch_dedup_enabled", False)
            )
            learnability_saturation_threshold = float(
                _select(
                    self.config,
                    "physevolve.stage2_learnability_saturation_threshold",
                    0.85,
                )
            )
            learnability_saturation_weight = float(
                _select(
                    self.config,
                    "physevolve.stage2_learnability_saturation_weight",
                    0.0,
                )
                or 0.0
            )
            diversity_edit_distance_weight = float(
                _select(
                    self.config,
                    "physevolve.stage2_diversity_edit_distance_weight",
                    0.0,
                )
                or 0.0
            )
            anti_copy_raw = _select(
                self.config, "physevolve.stage2_anti_copy_start_step", None
            )
            anti_copy_start_step = (
                None if anti_copy_raw is None else int(anti_copy_raw)
            )
            learnability_weight = float(
                _select(self.config, "physevolve.stage2_learnability_weight", 1.0) or 1.0
            )
            learnability_during_raw = _select(
                self.config,
                "physevolve.stage2_learnability_weight_during_fewshot",
                None,
            )
            learnability_weight_during_fewshot = (
                None
                if learnability_during_raw is None
                else float(learnability_during_raw)
            )
            family_difficulty_beta = float(
                _select(self.config, "physevolve.stage2_family_difficulty_beta", 0.0)
                or 0.0
            )
            solver_reward_scale = float(
                _select(self.config, "physevolve.stage2_solver_reward_scale", 1.0) or 1.0
            )
            stage2_config = PhysEvolveStage2StepBuilderConfig(
                model=model,
                scenes_per_batch=scenes_per_batch,
                group_size=group_size,
                response_length=response_length,
                executor_workers=executor_workers,
                generation=generation,
                video_save_freq=video_save_freq,
                video_save_max_per_step=video_save_max_per_step,
                solver_rollout_chunk_size=solver_rollout_chunk_size,
                intrinsic_enabled=intrinsic_enabled,
                intrinsic_weight=intrinsic_weight,
                intrinsic_component_weights=intrinsic_component_weights,
                proposer_failure_weight=proposer_failure_weight,
                color_variants_per_scene=color_variants_per_scene,
                max_solver_questions_per_scene=max_solver_questions_per_scene,
                max_solver_groups_per_step=max_solver_groups_per_step,
                min_scene_kind_picks_per_step=min_scene_kind_picks_per_step,
                camera_depth_bypass_enabled=camera_depth_bypass_enabled,
                disable_branch_elision=disable_branch_elision,
                branch_family_weight_multiplier=branch_family_weight_multiplier,
                cold_start_fewshot_steps=cold_start_fewshot_steps,
                reference_buffer_enabled=reference_buffer_enabled,
                reference_k=reference_k,
                reference_buffer_size=reference_buffer_size,
                reference_buffer_during_fewshot=reference_buffer_during_fewshot,
                copy_penalty_enabled=copy_penalty_enabled,
                copy_exact_reward=copy_exact_reward,
                copy_cap=copy_cap,
                copy_similarity_threshold=copy_similarity_threshold,
                batch_dedup_enabled=batch_dedup_enabled,
                learnability_saturation_threshold=learnability_saturation_threshold,
                learnability_saturation_weight=learnability_saturation_weight,
                diversity_edit_distance_weight=diversity_edit_distance_weight,
                anti_copy_start_step=anti_copy_start_step,
                learnability_weight=learnability_weight,
                learnability_weight_during_fewshot=learnability_weight_during_fewshot,
                family_difficulty_beta=family_difficulty_beta,
                solver_reward_scale=solver_reward_scale,
            )
            builder = PhysEvolveStage2StepBuilder(
                stage2_config,
                run_dir=self._physevolve_run_dir(),
                backend_factory=lambda _config: _PhysEvolveTrainerRolloutBackend(self),
            )
        else:
            stage1_config = PhysEvolveOnlineStepBuilderConfig(
                model=model,
                scenes_per_batch=scenes_per_batch,
                task_modes=tuple(
                    _select(self.config, "physevolve.task_modes", ("forward", "inverse", "parameter"))
                ),
                group_size=group_size,
                response_length=response_length,
                executor_workers=executor_workers,
                generation=generation,
                video_save_freq=video_save_freq,
                video_save_max_per_step=video_save_max_per_step,
            )
            builder = PhysEvolveOnlineStepBuilder(
                stage1_config,
                run_dir=self._physevolve_run_dir(),
                backend_factory=lambda _config: _PhysEvolveTrainerRolloutBackend(self),
            )
        self._physevolve_cached_step_builder = builder
        return builder

    def _update_physevolve_online_step(self, step: Any) -> dict[str, Any]:
        batch = step.artifacts.get("rollout_batch")
        scores = step.artifacts.get("token_level_scores")
        if batch is None or scores is None:
            raise TypeError("PhysEvolve self-play step must contain rollout_batch and token_level_scores")

        freeze_proposer = bool(_select(self.config, "physevolve.freeze_proposer", False))
        batch, scores, skip_actor = _drop_gen_vqa_rows_for_frozen_proposer(
            batch, scores, freeze_proposer=freeze_proposer
        )
        if skip_actor:
            return {
                "updated": False,
                "backend": "physevolve_ray_trainer",
                "step": step.step,
                "physevolve/freeze_proposer_skipped": 1.0,
            }

        from verl.trainer.ppo.ray_trainer import (
            apply_kl_penalty,
            compute_advantage,
            compute_response_mask,
            reduce_metrics,
        )

        validate_update_batch(
            batch,
            scores,
            group_size=int(self.config.actor_rollout_ref.rollout.n),
        )
        multimodal_alignment = validate_multimodal_token_alignment(
            batch,
            getattr(self, "processor", None),
        )
        batch.batch["token_level_scores"] = scores
        if "temperature" not in batch.meta_info:
            batch.meta_info["temperature"] = float(
                _select(self.config, "actor_rollout_ref.rollout.temperature", 1.0) or 1.0
            )
        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)
        batch.meta_info["global_token_num"] = batch.batch["attention_mask"].sum(dim=-1).tolist()

        # Align the batch length to the actor data-parallel size before handing
        # it to veRL's PPO update.  veRL's chunk_tensordict asserts
        # ``len(td) % dp_size == 0``; Stage-2 produces a variable-length batch
        # (proposer rows + solver rows for each OK scene) that may not satisfy
        # this when the executor success count is not a multiple of dp_size.
        # We drop at most (dp_size - 1) tail rows.  Because rows come in groups
        # of group_size and dp_size is always a multiple of group_size in our
        # configs, the dropped rows always form complete GRPO groups.
        dp_size = int(_select(self.config, "trainer.n_gpus_per_node", 1) or 1)
        _batch_len = int(batch.batch.batch_size[0])
        _remainder = _batch_len % dp_size
        if _remainder:
            batch = batch.select_idxs(list(range(_batch_len - _remainder)))

        metrics: dict[str, Any] = {}
        if multimodal_alignment:
            metrics["physevolve/video_tokens"] = multimodal_alignment["video_tokens"]
            metrics["physevolve/video_feature_tokens"] = multimodal_alignment["video_feature_tokens"]
        old_logprob_start = perf_counter()
        old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
        metrics["time/old_logprob_s"] = perf_counter() - old_logprob_start
        batch = batch.union(old_log_prob)
        if "entropys" in batch.batch.keys():
            batch.batch.pop("entropys")
        metrics["perf/mfu/actor_infer"] = old_log_prob_mfu

        metrics["time/ref_logprob_s"] = 0.0
        if getattr(self, "use_reference_policy", False):
            ref_logprob_start = perf_counter()
            ref_log_prob = self._compute_ref_log_prob(batch)
            metrics["time/ref_logprob_s"] = perf_counter() - ref_logprob_start
            batch = batch.union(ref_log_prob)

        if getattr(self, "use_critic", False):
            values = self._compute_values(batch)
            batch = batch.union(values)

        algorithm = self.config.algorithm
        if bool(getattr(algorithm, "use_kl_in_reward", False)):
            batch, kl_metrics = apply_kl_penalty(
                batch,
                kl_ctrl=self.kl_ctrl_in_reward,
                kl_penalty=getattr(algorithm, "kl_penalty", "kl"),
            )
            metrics.update(kl_metrics)
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        batch = compute_advantage(
            batch,
            adv_estimator=algorithm.adv_estimator,
            gamma=float(getattr(algorithm, "gamma", 1.0)),
            lam=float(getattr(algorithm, "lam", 1.0)),
            num_repeat=int(self.config.actor_rollout_ref.rollout.n),
            norm_adv_by_std_in_grpo=bool(
                getattr(algorithm, "norm_adv_by_std_in_grpo", True)
            ),
            config=algorithm,
        )

        if getattr(self, "use_critic", False):
            critic_output = self._update_critic(batch)
            metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

        metrics["time/update_actor_s"] = 0.0
        if int(_select(self.config, "trainer.critic_warmup", 0) or 0) <= int(step.step):
            batch.meta_info["multi_turn"] = bool(
                _select(self.config, "actor_rollout_ref.rollout.multi_turn.enable", False)
            )
            update_actor_start = perf_counter()
            actor_output = self._update_actor(batch)
            metrics["time/update_actor_s"] = perf_counter() - update_actor_start
            metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

        return {"updated": True, "backend": "physevolve_ray_trainer", "step": step.step, **metrics}

    def _physevolve_tracking_logger(self) -> Any | None:
        """Create veRL Tracking only when W&B logging is explicitly enabled."""

        backends = _logger_backends(_select(self.config, "trainer.logger", ("console",)))
        if "wandb" not in backends:
            return None
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        return Tracking(
            project_name=str(_select(self.config, "trainer.project_name", "physevolve-stage1")),
            experiment_name=str(_select(self.config, "trainer.experiment_name", "training_physevolve_native")),
            default_backend=backends,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

    def _physevolve_rollout_wake(self, global_steps: int) -> None:
        cm = getattr(self, "checkpoint_manager", None)
        if cm is not None:
            cm.update_weights(global_steps)

    def _physevolve_rollout_sleep(self) -> None:
        cm = getattr(self, "checkpoint_manager", None)
        if cm is not None:
            cm.sleep_replicas()

    def _physevolve_run_dir(self) -> Path:
        path = Path(str(_select(self.config, "trainer.default_local_dir", PHYSEVOLVE_OUTPUT_ROOT)))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_physevolve_summary(self, rows: list[dict[str, Any]], update_count: int) -> None:
        summary = {
            "completed": update_count >= int(_select(self.config, "trainer.total_training_steps", 1) or 1),
            "gate": "physevolve",
            "executed": True,
            "max_steps": int(_select(self.config, "trainer.total_training_steps", 1) or 1),
            "scenes_per_batch": int(_select(self.config, "data.train_batch_size", 1) or 1),
            "group_size": int(_select(self.config, "actor_rollout_ref.rollout.n", 1) or 1),
            "reward_tensor_finite": all(
                bool(row.get("reward_batch", {}).get("reward_tensor_finite", True))
                for row in rows
            ),
            "ppo_update_count": int(update_count),
        }
        write_json(self._physevolve_run_dir() / "summary.json", summary)
        write_json(
            self._physevolve_run_dir() / "run_status.json",
            {"completed": summary["completed"], "error": None, "executed": True},
        )


@dataclass(slots=True)
class _PhysEvolveTrainerRolloutBackend(AbstractContextManager["_PhysEvolveTrainerRolloutBackend"]):
    trainer: PhysEvolveRayPPOTrainer
    agent_name: str = "physevolve_single_turn"
    _last_rollout: Any = None

    def __enter__(self) -> "_PhysEvolveTrainerRolloutBackend":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def generate_with_metadata(
        self,
        prompt: str,
        video=None,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        stop_sequences: list[str],
        rollout_metadata: dict[str, Any] | None = None,
    ) -> GenerationResult:
        """Single-row generation, kept as a thin adapter over ``generate_batch``.

        PhysEvolve's online step builder calls ``generate_batch`` directly through
        ``call_generate_batch``. This adapter remains for external callers and
        for parity with the rest of the GenerationBackend protocol.
        """

        spec = GenerationSpec(
            prompt=prompt,
            video=video,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=None if top_k is None else int(top_k),
            stop_sequences=tuple(stop_sequences),
            rollout_metadata=dict(rollout_metadata or {}),
        )
        return self.generate_batch([spec])[0]

    def generate_batch(self, specs: Sequence[GenerationSpec]) -> list[GenerationResult]:
        """Submit all ``specs`` as a single multi-row rollout DataProto.

        veRL's ``AgentLoopWorker.generate_sequences`` already fans every row
        out as a concurrent ``asyncio.create_task``; sending one combined
        DataProto lets the underlying vLLM async server actually batch these
        proposer/solver prompts instead of running them one at a time.
        """

        if not specs:
            return []
        data = self._make_dataproto_batch(specs)
        first = specs[0]
        data.meta_info.update(
            {
                "do_sample": float(first.temperature) > 0,
                "validate": False,
                "eos_token_id": getattr(self.trainer.tokenizer, "eos_token_id", None),
                "pad_token_id": getattr(self.trainer.tokenizer, "pad_token_id", None),
            }
        )
        rollout = self._generate_padded(data)
        produced = int(rollout.batch.batch_size[0])
        if produced != len(specs):
            raise RuntimeError(
                "veRL rollout row count mismatch: "
                f"expected {len(specs)}, got {produced}. "
                "Check pad_dataproto_to_divisor / unpad_dataproto alignment."
            )
        self._last_rollout = rollout

        results: list[GenerationResult] = []
        valid_lengths = response_lengths_from_mask(rollout)
        responses = rollout.batch["responses"]
        for idx, spec in enumerate(specs):
            valid = max(1, int(valid_lengths[idx]))
            ids = responses[idx][:valid]
            text = self.trainer.tokenizer.decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            stop_list = list(spec.stop_sequences)
            text, stop_meta = _truncate_at_stop(text, stop_list)
            results.append(
                GenerationResult(
                    text=text,
                    metadata={
                        "backend": "physevolve_ray_trainer",
                        "generated_token_count": valid,
                        "max_new_tokens": int(spec.max_new_tokens),
                        "temperature": float(spec.temperature),
                        "top_p": float(spec.top_p),
                        "top_k": None if spec.top_k is None else int(spec.top_k),
                        "video_present": spec.video is not None,
                        **stop_meta,
                    },
                    generated_text=text,
                    prompt_text=spec.prompt,
                    prompt_tail=spec.prompt[-512:],
                )
            )
        return results

    def consume_last_rollout(self) -> Any:
        rollout = self._last_rollout
        self._last_rollout = None
        return rollout

    def _make_dataproto_batch(self, specs: Sequence[GenerationSpec]):
        """Build a single multi-row DataProto from a sequence of specs."""

        import torch
        from tensordict import TensorDict
        from verl import DataProto

        n = len(specs)
        prompts: list[str] = []
        agent_names: list[str] = []
        indices: list[int] = []
        uids: list[str] = []
        group_ids: list[str] = []
        roles: list[str] = []
        problem_types: list[str] = []
        scene_ids: list[Any] = []
        data_sources: list[str] = []
        reward_models: list[Any] = []
        stop_sequences_list: list[Any] = []
        max_tokens_list: list[int] = []
        bundles: list[Any] = []
        for spec in specs:
            metadata = dict(spec.rollout_metadata or {})
            uid = str(metadata.get("uid") or uuid4().hex)
            group_id = str(metadata.get("group_id") or uid)
            prompts.append(spec.prompt)
            agent_names.append(self.agent_name)
            indices.append(int(metadata.get("index", 0)))
            uids.append(uid)
            group_ids.append(group_id)
            roles.append(str(metadata.get("role") or ""))
            problem_types.append(str(metadata.get("problem_type") or ""))
            scene_ids.append(metadata.get("scene_id"))
            data_sources.append("physevolve")
            reward_models.append({"style": "rule", "ground_truth": ""})
            stop_sequences_list.append(tuple(spec.stop_sequences))
            max_tokens_list.append(int(spec.max_new_tokens))
            bundles.append(_video_bundle(spec.video))

        batch = TensorDict(
            {
                "input_ids": torch.zeros((n, 1), dtype=torch.long),
                "attention_mask": torch.ones((n, 1), dtype=torch.long),
                "position_ids": torch.zeros((n, 1), dtype=torch.long),
            },
            batch_size=[n],
        )
        non_tensor = {
            "raw_prompt": np.array(prompts, dtype=object),
            "agent_name": np.array(agent_names, dtype=object),
            "index": np.array(indices, dtype=np.int64),
            "uid": np.array(uids, dtype=object),
            "group_id": np.array(group_ids, dtype=object),
            "role": np.array(roles, dtype=object),
            "problem_type": np.array(problem_types, dtype=object),
            "scene_id": _object_array_many(scene_ids),
            "data_source": np.array(data_sources, dtype=object),
            "reward_model": _object_array_many(reward_models),
            "stop_sequences": _object_array_many(stop_sequences_list),
            "max_tokens": np.array(max_tokens_list, dtype=np.int64),
            "nz_video_bundle": _object_array_many(bundles),
        }
        return DataProto(batch=batch, non_tensor_batch=non_tensor)

    def _generate_padded(self, data: Any) -> Any:
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        divisor = _agent_worker_count(self.trainer)
        padded, pad_size = pad_dataproto_to_divisor(data, divisor)
        output = self.trainer.async_rollout_manager.generate_sequences(padded)
        rollout = unpad_dataproto(output, pad_size)
        for key, value in data.non_tensor_batch.items():
            if key not in rollout.non_tensor_batch:
                rollout.non_tensor_batch[key] = value
        return rollout


def _video_bundle(video: Any) -> VideoBundle | tuple[Any, dict[str, Any]] | None:
    if video is None:
        return None
    if isinstance(video, VideoBundle):
        _validate_video_metadata(video.metadata)
        return video
    if isinstance(video, tuple) and len(video) == 2:
        _validate_video_metadata(video[1])
        return video
    raise TypeError(
        "veRL training rollouts require video to be VideoBundle, "
        f"(frames, metadata), or None; got {type(video).__name__}"
    )


def _validate_video_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise TypeError(f"video metadata must be a dict; got {type(metadata).__name__}")
    missing = [key for key in VLLM_VIDEO_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"video metadata missing required keys: {missing}")


def _agent_worker_count(trainer: Any) -> int:
    try:
        value = trainer.config.actor_rollout_ref.rollout.agent.num_workers
    except Exception:
        value = 1
    return max(1, int(value))


def _throughput_metrics(online_metrics: dict[str, Any], step_total_s: float) -> dict[str, float]:
    """Return rollout and token throughput for one PhysEvolve PPO step."""

    elapsed = max(float(step_total_s), 1.0e-9)
    rollouts = int(online_metrics.get("online_proposer_generation_count", 0) or 0)
    rollouts += int(online_metrics.get("n_solver_requests", 0) or 0)
    tokens = int(online_metrics.get("online_generation_token_count", 0) or 0)
    return {
        "throughput/rollouts_per_s": float(rollouts / elapsed),
        "throughput/tokens_per_s": float(tokens / elapsed) if tokens > 0 else 0.0,
    }


def _logger_backends(value: Any) -> list[str]:
    """Normalize OmegaConf/list/string logger config to a list."""

    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value]
    except TypeError:
        return ["console"]


def _flatten_tracking_metrics(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested per-step metrics before sending them to W&B."""

    out: dict[str, Any] = {}

    def _visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                _visit(f"{prefix}/{key}" if prefix else str(key), child)
            return
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[prefix] = value

    _visit("", row)
    return out


def _object_array_many(values: Sequence[Any]) -> np.ndarray:
    """Build a 1-D object ndarray without numpy unpacking nested tuples."""

    arr = np.empty(len(values), dtype=object)
    for idx, item in enumerate(values):
        arr[idx] = item
    return arr


def _truncate_at_stop(text: str, stop_sequences: list[str]) -> tuple[str, dict[str, Any]]:
    best: int | None = None
    best_stop = None
    for stop in stop_sequences:
        if not stop:
            continue
        idx = text.find(stop)
        if idx >= 0 and (best is None or idx < best):
            best = idx
            best_stop = stop
    if best is None or best_stop is None:
        return text, {"hit_stop_sequence": False, "stop_sequence": None, "stop_index": None}
    return text[: best + len(best_stop)], {
        "hit_stop_sequence": True,
        "stop_sequence": best_stop,
        "stop_index": best,
    }


def _drop_gen_vqa_rows_for_frozen_proposer(
    batch: Any,
    scores: Any,
    *,
    freeze_proposer: bool,
) -> tuple[Any, Any, bool]:
    """Drop ``gen_vqa`` PPO rows when the Proposer is frozen.

    Returns ``(batch, scores, skip_actor)``. When ``freeze_proposer`` is
    False the same objects are returned. When True, rows whose
    ``problem_type`` is ``gen_vqa`` are removed and ``scores`` is sliced
    to match. If no ``pred_vqa`` rows remain, ``skip_actor`` is True so
    the caller can skip the actor update instead of crashing.
    """

    if not freeze_proposer:
        return batch, scores, False
    non_tensor = getattr(batch, "non_tensor_batch", None)
    if not isinstance(non_tensor, Mapping):
        return batch, scores, False
    problem_types = non_tensor.get("problem_type")
    if problem_types is None:
        return batch, scores, False
    keep = [idx for idx, kind in enumerate(problem_types) if kind != "gen_vqa"]
    if not keep:
        return batch, scores, True
    filtered_batch = batch.select_idxs(keep)
    if hasattr(scores, "index_select"):
        import torch

        index = torch.as_tensor(keep, device=scores.device, dtype=torch.long)
        filtered_scores = scores.index_select(0, index)
    else:
        filtered_scores = scores[keep]
    return filtered_batch, filtered_scores, False


def _select(config: Any, key: str, default: Any = None) -> Any:
    try:
        from omegaconf import OmegaConf

        return OmegaConf.select(config, key, default=default)
    except Exception:
        current = config
        for part in key.split("."):
            if current is None:
                return default
            if isinstance(current, dict):
                current = current.get(part, default)
            else:
                current = getattr(current, part, default)
        return current


def _stage2_component_weights(value: Any) -> tuple[float, float, float, float, float]:
    """Normalize Stage-2 intrinsic component weights from OmegaConf/list/string."""

    if value is None:
        return (0.30, 0.25, 0.20, 0.20, 0.05)
    if isinstance(value, str):
        values = tuple(float(part.strip()) for part in value.split(","))
    else:
        values = tuple(float(part) for part in value)
    if len(values) != 5:
        raise ValueError(
            "stage2_intrinsic_component_weights must contain exactly 5 values"
        )
    return values  # type: ignore[return-value]


def _stage2_scene_kind_pick_quotas(value: Any) -> tuple[tuple[str, int], ...]:
    """Normalize scene-kind pick minimums from OmegaConf/mapping/list values."""

    if value is None:
        return ()
    if isinstance(value, str):
        pairs: list[tuple[str, int]] = []
        for entry in value.split(","):
            scene_kind, separator, raw_count = entry.strip().partition("=")
            if not separator:
                raise ValueError(
                    "stage2_min_scene_kind_picks_per_step entries must use name=count"
                )
            pairs.append((scene_kind.strip(), int(raw_count.strip())))
        return tuple(pairs)
    if isinstance(value, Mapping) or callable(getattr(value, "items", None)):
        return tuple((str(key), int(count)) for key, count in value.items())
    return tuple((str(key), int(count)) for key, count in value)


__all__ = ["PhysEvolveRayPPOTrainer"]
