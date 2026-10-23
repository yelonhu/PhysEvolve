"""Two-step smoke entry for the Naive baseline.

Uses the shared training launcher with Stage-1 Forward/Inverse/Parameter tasks.
Defaults match ``scripts/train/naive.sh``; ``--max-steps`` defaults to two.
Use ``scripts/train/physevolve.sh`` for Stage-2 training.
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

# Defaults for scripts/train/naive.sh.
V3_SCENES_PER_BATCH = 4
V3_GROUP_SIZE = 4
V3_N_GPUS_PER_NODE = 8
V3_RAY_NUM_CPUS = 64
V3_CHECKPOINT_FREQ = 10
V3_EXPERIMENT_NAME = "training_100step_v3_instruct"
V3_WANDB_RUN_NAME = "training_100step_v3_instruct"

from physevolve.inference.hf_backend import DEFAULT_MODEL  # noqa: E402
from physevolve.smoke import SmokeGenerationConfig  # noqa: E402
from physevolve.training.training_validation import write_training_validation_report  # noqa: E402
from physevolve.training.verl_smoke import (  # noqa: E402
    PHYSEVOLVE_OUTPUT_ROOT,
    PhysEvolveTrainerConfig,
    prepare_physevolve_trainer_smoke,
    write_dry_run_summary,
)
from physevolve.training.verl_factory import preflight_physevolve_verl_factory  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model path")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2,
        help="PPO steps. Production 100-step runs pass 100 explicitly.",
    )
    parser.add_argument(
        "--scenes-per-batch",
        type=int,
        default=V3_SCENES_PER_BATCH,
        help=f"Online scenes per PPO step (v3 default: {V3_SCENES_PER_BATCH}).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=V3_GROUP_SIZE,
        help=f"GRPO group size / rollout.n (v3 default: {V3_GROUP_SIZE}).",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=6144,
        help=(
            "FSDP prompt length after HF processor expansion. Qwen3-VL video "
            "placeholders are much longer than compact vLLM prompt ids."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PHYSEVOLVE_OUTPUT_ROOT,
        help="root directory for training self-play validation runs",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write artifacts and native veRL config without launching Ray/vLLM",
    )
    parser.add_argument(
        "--gpu",
        default="all",
        help="CUDA_VISIBLE_DEVICES value; use 'all' to keep current environment",
    )
    parser.add_argument(
        "--n-gpus-per-node",
        type=int,
        default=V3_N_GPUS_PER_NODE,
        help=f"vLLM tensor parallel / FSDP world size (v3 default: {V3_N_GPUS_PER_NODE}).",
    )
    parser.add_argument(
        "--rollout-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override vLLM rollout eager mode. Defaults to enabled for Qwen3-VL "
            "models to avoid CUDA graph/deepstack crashes."
        ),
    )
    parser.add_argument(
        "--rollout-enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override vLLM rollout chunked prefill.",
    )
    parser.add_argument(
        "--rollout-enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override vLLM rollout prefix caching.",
    )
    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=V3_RAY_NUM_CPUS,
        help=f"Ray cluster CPU budget (v3 default: {V3_RAY_NUM_CPUS}).",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=V3_CHECKPOINT_FREQ,
        help=(
            f"Save checkpoint every N steps (v3 default: {V3_CHECKPOINT_FREQ}). "
            "Pass 0 or use a wrapper to disable intermediate saves."
        ),
    )
    parser.add_argument("--executor-workers", type=int, default=4)
    parser.add_argument(
        "--experiment-name",
        default=V3_EXPERIMENT_NAME,
        help="Run directory / logger experiment label (v3 default).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="enable W&B logging through veRL Tracking",
    )
    parser.add_argument("--wandb-project", default="physevolve-stage1")
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help=(
            "W&B run name; when --wandb is set and this is omitted, defaults to "
            f"{V3_WANDB_RUN_NAME!r}."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--trainer-factory",
        default=None,
        help=(
            "Optional 'module:function' that returns an initialized veRL trainer. "
            "Required for real non-dry-run training execution."
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

    wandb_run_name = args.wandb_run_name
    if wandb_run_name is None and args.wandb:
        wandb_run_name = V3_WANDB_RUN_NAME

    checkpoint_freq = args.checkpoint_freq
    if checkpoint_freq is not None and checkpoint_freq <= 0:
        checkpoint_freq = None

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
        ray_num_cpus=args.ray_num_cpus,
        checkpoint_freq=checkpoint_freq,
        executor_workers=args.executor_workers,
        experiment_name=args.experiment_name,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=wandb_run_name,
        generation=SmokeGenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stop_sequences=("</propose>", "</answer>"),
        ),
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
                "Real training execution requires --trainer-factory module:function. "
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
    print(f"training native veRL smoke run written to: {prepared.run_dir}")
    print(f"Run status: {prepared.run_dir / 'run_status.json'}")
    print(f"Summary: {prepared.run_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if validation_report is not None:
        print(f"training validation: {prepared.run_dir / 'training_validation.json'}")
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
