#!/usr/bin/env python3
"""Run PhysBench through the vendored official evaluator path.

This script keeps PhysBench's official dataset iteration, prompt suffix,
result schema, and scoring semantics as the primary path. PhysEvolve only
supplies a Qwen3-VL ``qa(image, prompt, mode)`` adapter.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_MODEL_PATH = Path(
    "Qwen/Qwen3-VL-8B-Instruct"
)
DEFAULT_DATASET_PATH = Path("datasets/PhysBench")
DEFAULT_ANSWERS_DIR = (
    _REPO_ROOT / "evaluation" / "physbench" / "third_party" / "physbench_answers"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--answers-dir", type=Path, default=DEFAULT_ANSWERS_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Optionally sample this many items. By default the full split is evaluated.",
    )
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--no-stratify", action="store_true")
    parser.add_argument("--n-video-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument(
        "--model-family",
        choices=("qwen", "internvl", "llama32_vision", "pixtral"),
        default=None,
        help="Generate family. Default: detect from --model-path.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-audit", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument(
        "--merge-shards",
        type=Path,
        default=None,
        help="Merge predictions.shard*.json inside this directory and score.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def _attach_run_file_log(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    root = logging.getLogger()
    for handler in root.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) is not None
            and str(getattr(handler, "baseFilename", "")) == str(log_path.resolve())
        ):
            return
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def _load_stratify_map(answers_dir: Path, split: str) -> dict[int, str]:
    answer_file = answers_dir / f"{split}_answer.json"
    if not answer_file.exists():
        return {}
    rows = json.loads(answer_file.read_text(encoding="utf-8"))
    return {
        int(row["idx"]): str(row.get("task_type", "unknown"))
        for row in rows
        if "idx" in row
    }


def _select_items(
    *,
    dataset_path: Path,
    answers_dir: Path,
    split: str,
    n_samples: Optional[int],
    seed: int,
    stratify: bool,
) -> list[Any]:
    from evaluation.physbench.data_loader import load_items, sample_items

    loaded = load_items(dataset_path, split=split)
    stratify_map = _load_stratify_map(answers_dir, split) if stratify else {}
    sampled = sample_items(
        loaded,
        n=n_samples,
        seed=seed,
        stratify_by=stratify_map or None,
    )
    return sampled


def _filter_evaluator_dataset(evaluator: Any, idx_values: set[int]) -> None:
    evaluator.dataset = [
        item for item in evaluator.dataset if int(item["idx"]) in idx_values
    ]


def _apply_sample(
    evaluator: Any,
    *,
    dataset_path: Path,
    answers_dir: Path,
    split: str,
    n_samples: Optional[int],
    seed: int,
    stratify: bool,
) -> list[Any]:
    sampled = _select_items(
        dataset_path=dataset_path,
        answers_dir=answers_dir,
        split=split,
        n_samples=n_samples,
        seed=seed,
        stratify=stratify,
    )
    _filter_evaluator_dataset(evaluator, {item.idx for item in sampled})
    return sampled


def _apply_output_resume(
    evaluator: Any,
    predictions_file: Path,
    *,
    resume: bool,
) -> None:
    evaluator.result_file = str(predictions_file)
    if not resume or not predictions_file.exists():
        evaluator.model_answers = []
        return
    rows = json.loads(predictions_file.read_text(encoding="utf-8"))
    existing_ids = {
        int(row["idx"])
        for row in rows
        if isinstance(row, dict) and row.get("idx") is not None and row.get("answer")
    }
    valid_ids = {int(item["idx"]) for item in evaluator.dataset}
    evaluator.model_answers = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("idx") is not None
        and int(row["idx"]) in valid_ids
        and row.get("answer")
    ]
    evaluator.dataset = [
        item for item in evaluator.dataset if int(item["idx"]) not in existing_ids
    ]


def _write_score_outputs(
    *,
    predictions_file: Path,
    answers_dir: Path,
    split: str,
    model_tag: str,
    summary_file: Path,
    report_file: Path,
) -> None:
    from evaluation.physbench.third_party.official_eval.eval_utils.caculate_core import (
        calculate_accuracy,
        format_markdown,
        print_accuracies,
    )

    answer_file = answers_dir / f"{split}_answer.json"
    if not answer_file.exists():
        logging.getLogger("physbench_official_eval").warning(
            "answer file %s missing; skipped official local scoring", answer_file
        )
        return
    accuracies = calculate_accuracy(answer_file, predictions_file)
    print_accuracies(accuracies, name=model_tag)
    summary_file.write_text(
        json.dumps(accuracies, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_file.write_text(format_markdown(accuracies, name=model_tag), encoding="utf-8")


def _write_shard_config(
    config_file: Path,
    *,
    args: argparse.Namespace,
    model_tag: str,
    expected_idx: list[int],
) -> None:
    payload = {
        "model_path": str(args.model_path),
        "model_tag": model_tag,
        "dataset_path": str(args.dataset_path),
        "answers_dir": str(args.answers_dir),
        "split": args.split,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "stratify": not args.no_stratify,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "expected_idx": expected_idx,
    }
    family = str(getattr(args, "model_family", None) or "")
    if family and family != "qwen":
        payload["model_family"] = family
    config_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _merge_only(args: argparse.Namespace, log: logging.Logger) -> int:
    from evaluation.physbench.sharding import (
        discover_shard_files,
        merge_predictions,
    )

    out_dir = Path(args.merge_shards)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"--merge-shards directory not found: {out_dir}")
    _attach_run_file_log(out_dir)

    shard_files = discover_shard_files(out_dir)
    if not shard_files:
        raise FileNotFoundError(
            f"no predictions.shard*.json files under {out_dir}"
        )
    log.info("found %d shard files under %s", len(shard_files), out_dir)

    config_files = sorted(out_dir.glob("official_eval_config.shard*.json"))
    expected_idx: list[int] | None = None
    model_tag = args.model_tag or args.model_path.name or "qwen3-vl-local"
    split = args.split
    if config_files:
        cfg = json.loads(config_files[0].read_text(encoding="utf-8"))
        expected_idx = [int(idx) for idx in cfg.get("expected_idx", [])]
        model_tag = str(cfg.get("model_tag") or model_tag)
        split = str(cfg.get("split") or split)
        log.info("loaded merge metadata from %s", config_files[0])
    else:
        log.warning(
            "no official_eval_config.shard*.json files under %s; "
            "merging without expected-idx validation",
            out_dir,
        )

    predictions_file = out_dir / "predictions.json"
    report = merge_predictions(
        shard_files,
        out_file=predictions_file,
        expected_idx=expected_idx,
        strict=True,
    )
    log.info(
        "merged %d rows -> %d unique idx (missing=%d extra=%d) -> %s",
        report.n_input_rows,
        report.n_unique_idx,
        report.n_missing,
        report.n_extra,
        report.output_file,
    )

    _write_score_outputs(
        predictions_file=predictions_file,
        answers_dir=args.answers_dir,
        split=split,
        model_tag=model_tag,
        summary_file=out_dir / "summary.json",
        report_file=out_dir / "report.md",
    )
    log.info("wrote %s and %s", out_dir / "summary.json", out_dir / "report.md")
    return 0


def main() -> int:
    args = _parse_args()
    _configure_logging(args.verbose)
    log = logging.getLogger("physbench_official_eval")

    if args.merge_shards is not None:
        return _merge_only(args, log)

    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1; got {args.num_shards}")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError(
            f"--shard-id must be in [0, {args.num_shards}); got {args.shard_id}"
        )

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        log.info("CUDA_VISIBLE_DEVICES=%s", os.environ["CUDA_VISIBLE_DEVICES"])

    from evaluation.common.family_generate import resolve_eval_family

    args.model_family = resolve_eval_family(args.model_path, args.model_family)

    model_tag = args.model_tag or args.model_path.name or "qwen3-vl-local"
    safe_tag = model_tag.replace("/", "_")
    sample_label = "all" if args.n_samples is None else str(args.n_samples)
    out_dir = args.out_dir or (
        _REPO_ROOT
        / "outputs" / "evaluation"
        / "physbench"
        / f"{safe_tag}_{args.split}_{sample_label}_official"
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _attach_run_file_log(out_dir)

    predictions_file = out_dir / "predictions.json"
    summary_file = out_dir / "summary.json"
    report_file = out_dir / "report.md"
    audit_file = None if args.no_audit else out_dir / "raw_audit.jsonl"
    if args.num_shards > 1:
        from evaluation.physbench.sharding import shard_filename

        predictions_file = shard_filename(
            predictions_file,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )
        if audit_file is not None:
            audit_file = shard_filename(
                audit_file,
                shard_id=args.shard_id,
                num_shards=args.num_shards,
            )
    if args.no_resume and audit_file is not None and audit_file.exists():
        audit_file.unlink()

    from evaluation.physbench.qwen_official_adapter import QwenPhysBenchAdapter
    from evaluation.physbench.third_party.official_eval.eval_utils.task_evaluator import (
        PhysionBenchEvaluator,
        task_split,
    )

    mode = task_split.get(model_tag, "general")
    log.info("output directory: %s", out_dir)
    log.info("loading model from %s", args.model_path)
    with QwenPhysBenchAdapter(
        args.model_path,
        n_video_frames=args.n_video_frames,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        audit_file=audit_file,
        model_family=args.model_family,
    ) as adapter:
        evaluator = PhysionBenchEvaluator(
            model=adapter,
            mode=mode,
            dataset_path=str(args.dataset_path),
            model_name=model_tag,
            resume=False,
            sample_ratio=None,
            split=args.split,
        )
        sampled = _apply_sample(
            evaluator,
            dataset_path=args.dataset_path,
            answers_dir=args.answers_dir,
            split=args.split,
            n_samples=args.n_samples,
            seed=args.seed,
            stratify=not args.no_stratify,
        )
        full_idx = [int(item.idx) for item in sampled]
        if args.num_shards > 1:
            from evaluation.physbench.sharding import shard_filename, shard_items

            shard = shard_items(
                sampled,
                num_shards=args.num_shards,
                shard_id=args.shard_id,
            )
            _filter_evaluator_dataset(evaluator, {int(item.idx) for item in shard})
            config_file = shard_filename(
                out_dir / "official_eval_config.json",
                shard_id=args.shard_id,
                num_shards=args.num_shards,
            )
            _write_shard_config(
                config_file,
                args=args,
                model_tag=model_tag,
                expected_idx=full_idx,
            )
            log.info(
                "shard %d/%d -> %d items (full sample: %d)",
                args.shard_id,
                args.num_shards,
                len(shard),
                len(sampled),
            )
        _apply_output_resume(
            evaluator,
            predictions_file,
            resume=not args.no_resume,
        )
        log.info("official evaluator will run %d remaining items", len(evaluator.dataset))
        evaluator.test()

    if args.num_shards == 1:
        _write_score_outputs(
            predictions_file=predictions_file,
            answers_dir=args.answers_dir,
            split=args.split,
            model_tag=model_tag,
            summary_file=summary_file,
            report_file=report_file,
        )
        log.info("wrote %s, %s, %s", predictions_file, summary_file, report_file)
    else:
        log.info(
            "wrote shard predictions at %s; run --merge-shards %s to score",
            predictions_file,
            out_dir,
        )
    if audit_file is not None:
        log.info("wrote raw audit at %s", audit_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
