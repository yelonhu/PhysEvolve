"""PhysEvolve training entry: CPU config checks or real veRL self-play.

The paper launcher selects stage=2 with gen_vqa/pred_vqa tasks. Training
starts from a base Instruct VLM and does not consume a Naive checkpoint.
Use --dry-run to inspect configuration artifacts, --factory-preflight to
check the installed runtime, or --max-steps 2 for a real GPU smoke.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physevolve.inference.hf_backend import DEFAULT_MODEL  # noqa: E402
from physevolve.smoke import SmokeGenerationConfig  # noqa: E402
from physevolve.training.training_validation import write_training_validation_report  # noqa: E402
from physevolve.training.verl_factory import preflight_physevolve_verl_factory  # noqa: E402
from physevolve.training.verl_smoke import (  # noqa: E402
    PHYSEVOLVE_OUTPUT_ROOT,
    PhysEvolveTrainerConfig,
    prepare_physevolve_trainer_smoke,
    write_dry_run_summary,
)


def _parse_stage2_component_weights(raw: str) -> tuple[float, float, float, float, float]:
    """Parse motion,interaction,complexity,diversity,parameter weights."""

    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "--stage2-intrinsic-component-weights must contain exactly 5 "
            "comma-separated floats: motion,interaction,complexity,diversity,parameter"
        )
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--stage2-intrinsic-component-weights values must be floats"
        ) from exc
    if any(value < 0.0 for value in values):
        raise argparse.ArgumentTypeError(
            "--stage2-intrinsic-component-weights values must be non-negative"
        )
    return values  # type: ignore[return-value]


def _parse_scene_kind_pick_quotas(raw: str) -> tuple[tuple[str, int], ...]:
    """Parse comma-separated ``scene_kind=count`` minimums."""

    if not raw.strip():
        return ()
    quotas: list[tuple[str, int]] = []
    seen: set[str] = set()
    for entry in raw.split(","):
        scene_kind, separator, raw_count = entry.strip().partition("=")
        scene_kind = scene_kind.strip()
        if not separator or not scene_kind:
            raise argparse.ArgumentTypeError(
                "scene-kind quotas must use comma-separated name=count entries"
            )
        try:
            count = int(raw_count.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"scene-kind quota for {scene_kind!r} must be an integer"
            ) from exc
        if count <= 0:
            raise argparse.ArgumentTypeError(
                f"scene-kind quota for {scene_kind!r} must be positive"
            )
        if scene_kind in seen:
            raise argparse.ArgumentTypeError(
                f"duplicate scene-kind quota {scene_kind!r}"
            )
        seen.add(scene_kind)
        quotas.append((scene_kind, count))
    return tuple(quotas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model path")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--scenes-per-batch", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=6144,
        help=(
            "FSDP prompt length after HF processor expansion. Qwen-VL "
            "video placeholders are much longer than compact vLLM prompt ids."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PHYSEVOLVE_OUTPUT_ROOT,
        help="root directory for Stage-2 self-play runs",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write artifacts and native veRL config without launching Ray/vLLM",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="CUDA_VISIBLE_DEVICES value; use 'all' to keep current environment",
    )
    parser.add_argument("--n-gpus-per-node", type=int, default=8)
    parser.add_argument(
        "--rollout-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override vLLM rollout eager mode. Defaults to enabled for "
            "Qwen-VL models."
        ),
    )
    parser.add_argument(
        "--rollout-enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--rollout-enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--rollout-gpu-memory-utilization",
        type=float,
        default=0.5,
        help=(
            "Fraction of each GPU vLLM reserves for weights + KV cache. Lower "
            "it (e.g. 0.4) to leave headroom for the Qwen-VL video activation "
            "spike on colocated GPUs."
        ),
    )
    parser.add_argument(
        "--rollout-max-num-seqs",
        type=int,
        default=None,
        help=(
            "Cap on concurrently-prefilled vLLM sequences. Defaults to "
            "scenes_per_batch * group_size so the video merge peak stays bounded."
        ),
    )
    parser.add_argument(
        "--rollout-max-num-batched-tokens",
        type=int,
        default=None,
        help=(
            "Cap tokens scheduled per vLLM forward step (chunked prefill). "
            "Lower it (e.g. 4096) to shrink the Qwen-VL video masked_scatter "
            "peak. Unset keeps veRL's default (8192)."
        ),
    )
    parser.add_argument("--ray-num-cpus", type=int, default=64)
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help="Save checkpoint every N steps. Default: save only at final step.",
    )
    parser.add_argument("--executor-workers", type=int, default=4)
    parser.add_argument(
        "--solver-rollout-chunk-size",
        type=int,
        default=8,
        help=(
            "Dispatch the Stage-2 VQA solver rollout in fixed-size chunks so "
            "the per-row video preprocessing peak stays bounded (multi-question "
            "scenes can otherwise OOM a single batch). 0 disables chunking."
        ),
    )
    parser.add_argument(
        "--video-save-freq",
        type=int,
        default=0,
        help=(
            "Save up to --video-save-max MP4 videos per step every N steps to "
            "steps/step_XXXXXX/videos/. 0 disables video saving (default)."
        ),
    )
    parser.add_argument(
        "--video-save-max",
        type=int,
        default=4,
        help="Max OK-scene videos to save per qualifying step (default 4).",
    )
    parser.add_argument(
        "--stage2-intrinsic-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the Proposer intrinsic/novelty shaping term. OFF by default"
            ": the OK reward is the pure learnability tent."
        ),
    )
    parser.add_argument(
        "--stage2-intrinsic-weight",
        type=float,
        default=0.3,
        help=(
            "Outer scale for the Stage-2 OK-branch intrinsic term. Use a low "
            "value such as 0.10 for diversity-only shaping."
        ),
    )
    parser.add_argument(
        "--stage2-intrinsic-component-weights",
        type=_parse_stage2_component_weights,
        default=(0.30, 0.25, 0.20, 0.20, 0.05),
        metavar="M,I,C,D,P",
        help=(
            "Five comma-separated non-negative weights for motion, interaction, "
            "complexity, diversity, parameter. Example diversity-only: 0,0,0,1,0."
        ),
    )
    parser.add_argument(
        "--stage2-proposer-failure-weight",
        type=float,
        default=1.5,
        help=(
            "Scale on the Proposer failure penalty (>1 gives intra-group GRPO "
            "signal even when synthetic questions are all answered correctly)."
        ),
    )
    parser.add_argument(
        "--stage2-color-variants-per-scene",
        type=int,
        default=1,
        help=(
            "Extra colour-permuted variants rendered per OK scene to break the "
            "colour->answer shortcut. 0 disables extra variants."
        ),
    )
    parser.add_argument(
        "--stage2-max-solver-questions-per-scene",
        type=int,
        default=4,
        help=(
            "Max unique VQA questions from one OK scene sent to the Solver "
            "before GRPO expansion. 0 disables the per-scene cap."
        ),
    )
    parser.add_argument(
        "--stage2-max-solver-groups-per-step",
        type=int,
        default=32,
        help=(
            "Max unique VQA question groups sent to the Solver per step before "
            "GRPO expansion. 0 disables the step cap."
        ),
    )
    parser.add_argument(
        "--stage2-min-scene-kind-picks-per-step",
        type=_parse_scene_kind_pick_quotas,
        default=(),
        metavar="KIND=N,...",
        help=(
            "Per-step scene-kind minimums inside --scenes-per-batch. "
            "Example: throwing_pit_board=1,collision_arena=1,motion_lab=1."
        ),
    )
    parser.add_argument(
        "--stage2-camera-depth-bypass-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Attach at most one camera-depth VQA to eligible collision/motion "
            "scenes. The Proposer camera-depth family remains frozen."
        ),
    )
    parser.add_argument(
        "--stage2-branch-family-weight-multiplier",
        type=float,
        default=1.5,
        help=(
            "Paired sampling multiplier for counterfactual_destination and "
            "inverse_intervention. Default 1.5 preserves recipe."
        ),
    )
    # ── Anti-copy settings ────────────────────────────────────
    parser.add_argument(
        "--stage2-cold-start-fewshot-steps",
        type=int,
        default=0,
        help=(
            "Show the executable golden fewshot only for steps <= N (teach "
            "format), then drop it and use the ReferenceBuffer. All copy "
            "penalties / batch dedup are also off during this cold-start window."
        ),
    )
    parser.add_argument(
        "--stage2-reference-buffer-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Inject coordinate-free recent-scene summaries after cold start.",
    )
    parser.add_argument("--stage2-reference-k", type=int, default=2)
    parser.add_argument("--stage2-reference-buffer-size", type=int, default=32)
    parser.add_argument(
        "--stage2-reference-buffer-during-fewshot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Inject ReferenceBuffer parameter-band summaries even while the "
            "executable golden fewshot is shown (recipe hybrid)."
        ),
    )
    parser.add_argument(
        "--stage2-copy-penalty-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable AST anti-copy reward (exact-copy / recent-repeat / near-copy cap).",
    )
    parser.add_argument("--stage2-copy-exact-reward", type=float, default=-0.5)
    parser.add_argument("--stage2-copy-cap", type=float, default=0.1)
    parser.add_argument(
        "--stage2-copy-similarity-threshold", type=float, default=0.95
    )
    parser.add_argument(
        "--stage2-batch-dedup-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Intra-GRPO-group AST dedup: duplicates get the copy reward and skip the Solver.",
    )
    parser.add_argument(
        "--stage2-learnability-saturation-threshold", type=float, default=0.85
    )
    parser.add_argument(
        "--stage2-learnability-saturation-weight",
        type=float,
        default=0.0,
        help="Penalise mean_acc above the threshold (0 disables).",
    )
    parser.add_argument(
        "--stage2-diversity-edit-distance-weight",
        type=float,
        default=0.0,
        help="AST edit-distance diversity bonus weight (learnability magnitude; 0 disables).",
    )
    parser.add_argument(
        "--stage2-anti-copy-start-step",
        type=int,
        default=None,
        help=(
            "Step after which anti-copy / saturation / diversity activate. "
            "Decoupled from --stage2-cold-start-fewshot-steps. None = legacy "
            "(same as fewshot_steps). 0 = active from step 1 (recipe)."
        ),
    )
    parser.add_argument(
        "--stage2-learnability-weight",
        type=float,
        default=1.0,
        help="Scale on the Proposer learnability tent (recipe uses 0.5).",
    )
    parser.add_argument(
        "--stage2-learnability-weight-during-fewshot",
        type=float,
        default=None,
        help=(
            "Override learnability weight while executable fewshot is shown "
            "(step <= cold_start_fewshot_steps). None = use "
            "--stage2-learnability-weight. recipe uses 0.0."
        ),
    )
    parser.add_argument(
        "--stage2-family-difficulty-beta",
        type=float,
        default=0.0,
        help=(
            "Family rolling-acc difficulty weight beta: reward *= "
            "clip(1+beta*(0.5-acc), 0.5, 1.5). 0 disables (legacy)."
        ),
    )
    parser.add_argument(
        "--stage2-solver-reward-scale",
        type=float,
        default=1.0,
        help="Multiply pred_vqa reward rows before GRPO (recipe uses 1.5).",
    )
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=0.01,
        help=(
            "Fixed KL coefficient for actor KL loss and reward KL penalty "
            "(default 0.01). recipe anti-forgetting uses 0.02."
        ),
    )
    parser.add_argument(
        "--disable-branch-elision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Stage-2 ablation: drop counterfactual_destination and "
            "inverse_intervention from sampling, skip executor branch sims, "
            "and drop those families from Solver payloads. Disabled by "
            "default; enabling it writes physevolve.disable_branch_elision=true."
        ),
    )
    parser.add_argument(
        "--freeze-proposer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Stage-2 ablation: still generate gen_vqa scenes with the current "
            "actor, but drop gen_vqa rows from the PPO update so only pred_vqa "
            "trains. Disabled by default; enabling it writes "
            "physevolve.freeze_proposer=true."
        ),
    )
    parser.add_argument("--experiment-name", default="stage2_vqa_smoke")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="enable W&B logging through veRL Tracking",
    )
    parser.add_argument(
        "--wandb-project",
        default="physevolve-stage2",
        help="Stage-2 runs land in a distinct W&B project by default.",
    )
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--trainer-factory",
        default=None,
        help=(
            "Optional 'module:function' that returns an initialized veRL "
            "trainer. Required for real (non-dry-run) Stage-2 execution."
        ),
    )
    parser.add_argument(
        "--factory-preflight",
        action="store_true",
        help="check native factory config/import/interface without loading the model",
    )
    args = parser.parse_args(argv)

    if args.gpu.lower() != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = PhysEvolveTrainerConfig(
        model=args.model,
        max_steps=args.max_steps,
        scenes_per_batch=args.scenes_per_batch,
        group_size=args.group_size,
        prompt_length=args.prompt_length,
        out_root=args.out_root,
        run_id=args.run_id,
        execute=not args.dry_run,
        n_gpus_per_node=args.n_gpus_per_node,
        rollout_enforce_eager=args.rollout_enforce_eager,
        rollout_enable_chunked_prefill=args.rollout_enable_chunked_prefill,
        rollout_enable_prefix_caching=args.rollout_enable_prefix_caching,
        rollout_gpu_memory_utilization=args.rollout_gpu_memory_utilization,
        rollout_max_num_seqs=args.rollout_max_num_seqs,
        rollout_max_num_batched_tokens=args.rollout_max_num_batched_tokens,
        ray_num_cpus=args.ray_num_cpus,
        checkpoint_freq=args.checkpoint_freq,
        executor_workers=args.executor_workers,
        solver_rollout_chunk_size=args.solver_rollout_chunk_size,
        stage2_intrinsic_enabled=args.stage2_intrinsic_enabled,
        stage2_intrinsic_weight=args.stage2_intrinsic_weight,
        stage2_intrinsic_component_weights=args.stage2_intrinsic_component_weights,
        stage2_proposer_failure_weight=args.stage2_proposer_failure_weight,
        stage2_color_variants_per_scene=args.stage2_color_variants_per_scene,
        stage2_max_solver_questions_per_scene=(
            args.stage2_max_solver_questions_per_scene
        ),
        stage2_max_solver_groups_per_step=args.stage2_max_solver_groups_per_step,
        stage2_min_scene_kind_picks_per_step=(
            args.stage2_min_scene_kind_picks_per_step
        ),
        stage2_camera_depth_bypass_enabled=(
            args.stage2_camera_depth_bypass_enabled
        ),
        stage2_branch_family_weight_multiplier=(
            args.stage2_branch_family_weight_multiplier
        ),
        stage2_cold_start_fewshot_steps=args.stage2_cold_start_fewshot_steps,
        stage2_reference_buffer_enabled=args.stage2_reference_buffer_enabled,
        stage2_reference_k=args.stage2_reference_k,
        stage2_reference_buffer_size=args.stage2_reference_buffer_size,
        stage2_reference_buffer_during_fewshot=(
            args.stage2_reference_buffer_during_fewshot
        ),
        stage2_copy_penalty_enabled=args.stage2_copy_penalty_enabled,
        stage2_copy_exact_reward=args.stage2_copy_exact_reward,
        stage2_copy_cap=args.stage2_copy_cap,
        stage2_copy_similarity_threshold=args.stage2_copy_similarity_threshold,
        stage2_batch_dedup_enabled=args.stage2_batch_dedup_enabled,
        stage2_learnability_saturation_threshold=(
            args.stage2_learnability_saturation_threshold
        ),
        stage2_learnability_saturation_weight=(
            args.stage2_learnability_saturation_weight
        ),
        stage2_diversity_edit_distance_weight=(
            args.stage2_diversity_edit_distance_weight
        ),
        stage2_anti_copy_start_step=args.stage2_anti_copy_start_step,
        stage2_learnability_weight=args.stage2_learnability_weight,
        stage2_learnability_weight_during_fewshot=(
            args.stage2_learnability_weight_during_fewshot
        ),
        stage2_family_difficulty_beta=args.stage2_family_difficulty_beta,
        stage2_solver_reward_scale=args.stage2_solver_reward_scale,
        kl_coef=args.kl_coef,
        disable_branch_elision=args.disable_branch_elision,
        freeze_proposer=args.freeze_proposer,
        video_save_freq=args.video_save_freq,
        video_save_max_per_step=args.video_save_max,
        experiment_name=args.experiment_name,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        generation=SmokeGenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stop_sequences=("</propose>", "</answer>"),
        ),
        stage=2,
        task_modes=("vqa",),
    )

    if args.factory_preflight:
        report = preflight_physevolve_verl_factory(config)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    prepared = prepare_physevolve_trainer_smoke(config)
    summary = None
    validation_report = None
    if config.execute:
        if not args.trainer_factory:
            raise RuntimeError(
                "Real Stage-2 execution requires --trainer-factory module:function. "
                "Use --dry-run for config/artifact checks."
            )
        trainer = _load_trainer_factory(args.trainer_factory)(
            config=prepared.factory_config,
        )
        trainer.fit()
        summary_path = prepared.run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if config.max_steps >= 20:
            validation_report = write_training_validation_report(
                prepared.run_dir,
                expected_steps=config.max_steps,
            )
    else:
        summary = write_dry_run_summary(config, prepared.run_dir)
    print(f"Stage-2 VQA smoke run written to: {prepared.run_dir}")
    print(f"Run status: {prepared.run_dir / 'run_status.json'}")
    print(f"Summary: {prepared.run_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if validation_report is not None:
        print(f"Stage-2 validation: {prepared.run_dir / 'training_validation.json'}")
        print(json.dumps(validation_report, indent=2, sort_keys=True))
    validation_ok = validation_report is None or bool(validation_report.get("ok"))
    return 0 if (summary.get("completed") and validation_ok) or not config.execute else 1


def _load_trainer_factory(spec: str):
    """Load a trainer factory from 'module:function'."""

    if ":" not in spec:
        raise ValueError("--trainer-factory must be in 'module:function' format")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, func_name)
    if not callable(factory):
        raise TypeError(f"trainer factory {spec!r} is not callable")
    return factory


if __name__ == "__main__":
    raise SystemExit(main())
