"""Zero-shot local VLM evaluation for the CausalPhys reference benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .common import (
    DEFAULT_MODEL_PATH,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    GenerationSettings,
    PredictionStore,
    accuracy,
    append_jsonl,
    build_image_content,
    build_video_content,
    generate_reference_answer,
    load_prediction_map,
    load_shard_config,
    merge_prediction_shards,
    parse_physbench_mc_answer,
    prepare_jsonl,
    safe_model_tag,
    shard_filename,
    shard_items,
    write_json_atomic,
    write_text_atomic,
)
from .family_generate import open_eval_backend, resolve_eval_family


DEFAULT_DATASET_DIR = Path(
    os.environ.get(
        "DATASET_DIR",
        "datasets/causalphys",
    )
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_STYLES = ("short", "cot")
_SHORT_PROMPT_SUFFIX = (
    "\n\nAnswer with exactly one option letter from A, B, C, or D. "
    "Put that letter inside <answer>...</answer>."
)
_COT_PROMPT_SUFFIX = (
    "\n\nFirst provide a brief, step-by-step rationale grounded only in the "
    "visual evidence and the question. Be explicit, concise, and avoid "
    "speculation; use at most 8 sentences. Then give exactly one final option "
    "letter from A, B, C, or D."
    "\n\nUse this exact output structure:"
    "\n<rationale>Your concise rationale.</rationale>"
    "\n<answer>A</answer>"
)
_NATURAL_PART_RE = re.compile(r"(\d+)")
_RATIONALE_BLOCK_RE = re.compile(
    r"<rationale>(.*?)</rationale>",
    re.IGNORECASE | re.DOTALL,
)
_ANSWER_BLOCK_RE = re.compile(
    r"<answer>(.*?)</answer>",
    re.IGNORECASE | re.DOTALL,
)
_ANSWER_OPEN_RE = re.compile(
    r"<answer>([^<]*)$",
    re.IGNORECASE | re.DOTALL,
)
_RESULT_BLOCK_RE = re.compile(
    r"<result>(.*?)</result>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
_FINAL_ANSWER_CUE_RE = re.compile(
    r"\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*$",
    re.IGNORECASE,
)
_TERMINAL_MC_RE = re.compile(
    r"\b([ABCD])\b(?:\s*[.)!]*)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class InvalidGold:
    """One annotation excluded because its local answer is not A-D."""

    sample_key: str
    category: str
    subcategory: str
    raw_gold: Any


@dataclass(frozen=True, slots=True)
class CausalPhysItem:
    """One scoreable CausalPhys annotation."""

    idx: int
    sample_key: str
    annotation_path: Path
    category: str
    subcategory: str
    question: str
    gold_answer: str
    media_paths: tuple[Path, ...]
    media_kind: str


@dataclass(frozen=True, slots=True)
class CausalPhysCorpus:
    """Indexed CausalPhys items plus data-quality diagnostics."""

    dataset_dir: Path
    n_annotations: int
    items: tuple[CausalPhysItem, ...]
    invalid_gold: tuple[InvalidGold, ...]


@dataclass(frozen=True, slots=True)
class CausalPhysResponse:
    """Parsed final answer and optional model-authored rationale."""

    answer: str | None
    rationale: str | None


def load_causalphys_corpus(
    dataset_dir: Path,
    *,
    max_images: int = 12,
) -> CausalPhysCorpus:
    """Index CausalPhys annotations without writing extracted video frames."""

    root = Path(dataset_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CausalPhys dataset directory not found: {root}")

    annotation_paths = sorted(root.glob("*/*/annotations/*.json"))
    if not annotation_paths:
        raise FileNotFoundError(
            f"no */*/annotations/*.json files found under {root}"
        )

    items: list[CausalPhysItem] = []
    invalid: list[InvalidGold] = []
    for idx, annotation_path in enumerate(annotation_paths):
        relative = annotation_path.relative_to(root)
        category, subcategory = relative.parts[0], relative.parts[1]
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        question = str(annotation.get("question", "")).strip()
        if not question:
            raise ValueError(f"annotation has no question: {annotation_path}")

        raw_gold = annotation.get("ground_truth_answer")
        normalized_gold = (
            raw_gold.strip().upper() if isinstance(raw_gold, str) else ""
        )
        sample_key = relative.as_posix()
        if normalized_gold not in {"A", "B", "C", "D"}:
            invalid.append(
                InvalidGold(
                    sample_key=sample_key,
                    category=category,
                    subcategory=subcategory,
                    raw_gold=raw_gold,
                )
            )
            continue

        media_paths, media_kind = _resolve_causalphys_media(
            annotation,
            annotation_path=annotation_path,
            max_images=max_images,
        )
        items.append(
            CausalPhysItem(
                idx=idx,
                sample_key=sample_key,
                annotation_path=annotation_path,
                category=category,
                subcategory=subcategory,
                question=question,
                gold_answer=normalized_gold,
                media_paths=media_paths,
                media_kind=media_kind,
            )
        )

    return CausalPhysCorpus(
        dataset_dir=root,
        n_annotations=len(annotation_paths),
        items=tuple(items),
        invalid_gold=tuple(invalid),
    )


def _resolve_causalphys_media(
    annotation: dict[str, Any],
    *,
    annotation_path: Path,
    max_images: int,
) -> tuple[tuple[Path, ...], str]:
    """Resolve an annotation's path/paths field into ordered local media."""

    raw_paths = annotation.get("paths", annotation.get("path"))
    if isinstance(raw_paths, str):
        entries = [raw_paths]
    elif isinstance(raw_paths, list) and all(
        isinstance(entry, str) and entry for entry in raw_paths
    ):
        entries = list(raw_paths)
    else:
        raise ValueError(
            f"annotation has invalid path/paths field: {annotation_path}"
        )

    data_dir = annotation_path.parent.parent / "data"
    resolved = tuple(
        (
            Path(entry)
            if Path(entry).is_absolute()
            else data_dir / entry
        ).resolve()
        for entry in entries
    )
    for path in resolved:
        if not path.exists():
            raise FileNotFoundError(
                f"CausalPhys media not found: {path} "
                f"(annotation: {annotation_path})"
            )

    if len(resolved) == 1 and resolved[0].is_dir():
        directory = resolved[0]
        children = sorted(directory.iterdir(), key=_natural_path_key)
        if not children:
            raise ValueError(f"empty image directory: {directory}")
        if any(
            not path.is_file()
            or path.suffix.lower() not in IMAGE_EXTENSIONS
            for path in children
        ):
            raise ValueError(
                f"image sequence directory contains unsupported entries: "
                f"{directory}"
            )
        resolved = tuple(children)

    suffixes = {path.suffix.lower() for path in resolved}
    if suffixes <= IMAGE_EXTENSIONS:
        if len(resolved) > max_images:
            raise ValueError(
                f"image sequence has {len(resolved)} entries; "
                f"maximum is {max_images}: {annotation_path}"
            )
        return resolved, "images"
    if len(resolved) == 1 and suffixes <= VIDEO_EXTENSIONS:
        return resolved, "video"
    raise ValueError(
        f"mixed or unsupported CausalPhys media {sorted(suffixes)}: "
        f"{annotation_path}"
    )


def _natural_path_key(path: Path) -> tuple[Any, ...]:
    """Return a numeric-aware filename sort key."""

    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in _NATURAL_PART_RE.split(path.name)
    )


def select_causalphys_items(
    corpus: CausalPhysCorpus,
    *,
    n_samples: int | None,
    seed: int,
    categories: Sequence[str] = (),
    subcategories: Sequence[str] = (),
) -> list[CausalPhysItem]:
    """Filter and deterministically sample scoreable CausalPhys items."""

    category_filter = {value.casefold() for value in categories}
    subcategory_filter = {value.casefold() for value in subcategories}
    selected = [
        item
        for item in corpus.items
        if (
            not category_filter
            or item.category.casefold() in category_filter
        )
        and (
            not subcategory_filter
            or item.subcategory.casefold() in subcategory_filter
        )
    ]
    if n_samples is not None:
        if n_samples < 1:
            raise ValueError(f"--n-samples must be >= 1; got {n_samples}")
        if n_samples < len(selected):
            rng = random.Random(seed)
            selected = rng.sample(selected, n_samples)
    return sorted(selected, key=lambda item: item.idx)


def build_causalphys_prompt(
    item: CausalPhysItem,
    *,
    prompt_style: str = "cot",
) -> str:
    """Build a gold-free short or Appendix-C.1-style prompt."""

    if prompt_style == "short":
        return item.question + _SHORT_PROMPT_SUFFIX
    if prompt_style == "cot":
        return item.question + _COT_PROMPT_SUFFIX
    raise ValueError(
        f"unsupported CausalPhys prompt style {prompt_style!r}; "
        f"expected one of {PROMPT_STYLES}"
    )


def build_causalphys_content(
    item: CausalPhysItem,
    *,
    prompt_style: str = "cot",
    n_video_frames: int,
    max_images: int,
) -> list[dict[str, Any]]:
    """Decode an item's media and build ordered Qwen multimodal content."""

    prompt = build_causalphys_prompt(item, prompt_style=prompt_style)
    if item.media_kind == "video":
        return build_video_content(
            item.media_paths[0],
            prompt,
            n_frames=n_video_frames,
        )
    return build_image_content(
        item.media_paths,
        prompt,
        max_images=max_images,
    )


def parse_causalphys_response(raw_text: str) -> CausalPhysResponse:
    """Parse the final A-D choice without reading letters from rationale.

    Canonical ``<answer>`` wins, followed by the official repository's
    historical ``<result>`` wrapper. Untagged compatibility is deliberately
    restricted to a stand-alone letter at the end of the post-rationale tail.
    """

    answer_match = _last_match(_ANSWER_BLOCK_RE, raw_text)
    if answer_match is None:
        answer_match = _ANSWER_OPEN_RE.search(raw_text)
    result_match = _last_match(_RESULT_BLOCK_RE, raw_text)
    final_match = answer_match
    answer = None
    if answer_match is not None:
        answer = parse_physbench_mc_answer(
            f"<answer>{answer_match.group(1)}</answer>"
        )
    if answer is None and result_match is not None:
        final_match = result_match
        answer = parse_physbench_mc_answer(
            f"<answer>{result_match.group(1)}</answer>"
        )

    rationale_match = _RATIONALE_BLOCK_RE.search(raw_text)
    rationale = (
        _clean_rationale(rationale_match.group(1))
        if rationale_match is not None
        else None
    )

    if answer is None:
        if rationale_match is not None:
            tail = raw_text[rationale_match.end() :]
        else:
            tail = raw_text
        answer = _terminal_mc_letter(tail)

    if rationale is None:
        rationale_end = final_match.start() if final_match is not None else None
        if rationale_end is None:
            terminal_match = _TERMINAL_MC_RE.search(
                raw_text.replace("<|im_end|>", "").strip()
            )
            rationale_end = (
                terminal_match.start() if terminal_match is not None else None
            )
        if rationale_end is not None:
            rationale = _clean_rationale(raw_text[:rationale_end])

    return CausalPhysResponse(answer=answer, rationale=rationale)


def _last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    """Return the last regex match so the final tagged answer wins."""

    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def _terminal_mc_letter(text: str) -> str | None:
    """Return an A-D token only when it terminates the response tail."""

    cleaned = text.replace("<|im_end|>", "").strip()
    match = _TERMINAL_MC_RE.search(cleaned)
    return match.group(1).upper() if match is not None else None


def _clean_rationale(text: str) -> str | None:
    """Remove structural wrappers and return non-empty rationale text."""

    cleaned = text.replace("<|im_end|>", "")
    cleaned = _THINK_TAG_RE.sub("", cleaned).strip()
    cleaned = _FINAL_ANSWER_CUE_RE.sub("", cleaned).strip()
    return cleaned or None


def score_causalphys(
    items: Sequence[CausalPhysItem],
    predictions: dict[int, str | None],
    *,
    corpus: CausalPhysCorpus,
    model_tag: str,
    prompt_style: str,
    max_new_tokens: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Compute local exact-match ACC and write JSON/Markdown summaries."""

    overall = {"correct": 0, "total": 0, "parsed": 0}
    category_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0, "parsed": 0}
    )
    subcategory_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0, "parsed": 0}
    )

    for item in items:
        raw_answer = predictions.get(item.idx)
        answer = (
            raw_answer.strip().upper()
            if isinstance(raw_answer, str)
            else None
        )
        parsed = answer in {"A", "B", "C", "D"}
        correct = parsed and answer == item.gold_answer
        for stats in (
            overall,
            category_stats[item.category],
            subcategory_stats[f"{item.category}/{item.subcategory}"],
        ):
            stats["total"] += 1
            stats["parsed"] += int(parsed)
            stats["correct"] += int(correct)

    summary = {
        "benchmark": "CausalPhys",
        "track": "reference_appendix",
        "model_tag": model_tag,
        "prompt_style": prompt_style,
        "max_new_tokens": max_new_tokens,
        "dataset_dir": str(corpus.dataset_dir),
        "n_annotations": corpus.n_annotations,
        "n_valid_gold": len(corpus.items),
        "n_invalid_gold": len(corpus.invalid_gold),
        "invalid_gold": [
            {
                "sample_key": row.sample_key,
                "raw_gold": row.raw_gold,
            }
            for row in corpus.invalid_gold
        ],
        "overall": _stats_payload(overall),
        "by_category": {
            key: _stats_payload(value)
            for key, value in sorted(category_stats.items())
        },
        "by_subcategory": {
            key: _stats_payload(value)
            for key, value in sorted(subcategory_stats.items())
        },
    }
    write_json_atomic(Path(out_dir) / "summary.json", summary)
    write_text_atomic(
        Path(out_dir) / "report.md",
        _causalphys_report(summary),
    )
    return summary


def _stats_payload(stats: dict[str, int]) -> dict[str, int | float]:
    """Convert running counts into a serializable metric payload."""

    return {
        "accuracy": accuracy(stats["correct"], stats["total"]),
        "parse_rate": accuracy(stats["parsed"], stats["total"]),
        "correct": stats["correct"],
        "parsed": stats["parsed"],
        "total": stats["total"],
    }


def _causalphys_report(summary: dict[str, Any]) -> str:
    """Render a compact Markdown report for a CausalPhys run."""

    overall = summary["overall"]
    lines = [
        "# CausalPhys zero-shot VLM reference evaluation",
        "",
        f"- Model: `{summary['model_tag']}`",
        f"- Prompt style: `{summary['prompt_style']}`",
        f"- Max new tokens: {summary['max_new_tokens']}",
        "- PhysEvolve paper benchmark",
        (
            f"- Overall ACC: {overall['accuracy'] * 100:.2f}% "
            f"({overall['correct']}/{overall['total']})"
        ),
        (
            f"- Parse rate: {overall['parse_rate'] * 100:.2f}% "
            f"({overall['parsed']}/{overall['total']})"
        ),
        f"- Invalid local gold excluded: {summary['n_invalid_gold']}",
        "",
        "## Category breakdown",
        "",
        "| Category | ACC | Correct / Total | Parse rate |",
        "|---|---:|---:|---:|",
    ]
    for name, stats in summary["by_category"].items():
        lines.append(
            f"| {name} | {stats['accuracy'] * 100:.2f}% | "
            f"{stats['correct']} / {stats['total']} | "
            f"{stats['parse_rate'] * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Subcategory breakdown",
            "",
            "| Subcategory | ACC | Correct / Total | Parse rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, stats in summary["by_subcategory"].items():
        lines.append(
            f"| {name} | {stats['accuracy'] * 100:.2f}% | "
            f"{stats['correct']} / {stats['total']} | "
            f"{stats['parse_rate'] * 100:.2f}% |"
        )
    return "\n".join(lines) + "\n"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the CausalPhys evaluator command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument(
        "--model-family",
        choices=("qwen", "internvl", "llama32_vision", "pixtral"),
        default=None,
        help="Generate family. Default: detect from --model-path.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--subcategory", action="append", default=[])
    parser.add_argument("--n-video-frames", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument(
        "--prompt-style",
        choices=PROMPT_STYLES,
        default="cot",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--merge-shards", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-audit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-media", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def _configure_logging(verbose: bool) -> None:
    """Configure concise console logging."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def _validate_shard_args(num_shards: int, shard_id: int) -> None:
    """Validate deterministic shard coordinates."""

    if num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1; got {num_shards}")
    if not 0 <= shard_id < num_shards:
        raise ValueError(
            f"--shard-id must be in [0, {num_shards}); got {shard_id}"
        )


def _default_out_dir(
    model_tag: str,
    *,
    n_samples: int | None,
    prompt_style: str,
    max_new_tokens: int,
) -> Path:
    """Return the default CausalPhys reference result directory."""

    sample_label = "full" if n_samples is None else f"n{n_samples}"
    return (
        _REPO_ROOT
        / "outputs" / "evaluation"
        / "CausalPhys"
        / (
            f"{safe_model_tag(model_tag)}_{sample_label}_"
            f"zero_shot_{prompt_style}_tok{max_new_tokens}"
        )
    )


def _run_signature(
    args: argparse.Namespace,
    *,
    model_tag: str,
) -> dict[str, Any]:
    """Build merge-time metadata that must agree across shards."""

    payload: dict[str, Any] = {
        "benchmark": "causalphys",
        "model_path": str(args.model_path.resolve()) if args.model_path.exists() else str(args.model_path),
        "model_tag": model_tag,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "n_samples": args.n_samples,
        "seed": args.seed,
        "categories": list(args.category),
        "subcategories": list(args.subcategory),
        "prompt_style": args.prompt_style,
        "n_video_frames": args.n_video_frames,
        "max_images": args.max_images,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    family = resolve_eval_family(args.model_path, args.model_family)
    if family != "qwen":
        payload["model_family"] = family
    return payload


def _select_from_signature(
    signature: dict[str, Any],
) -> tuple[CausalPhysCorpus, list[CausalPhysItem]]:
    """Reconstruct the exact selected item set recorded by shard configs."""

    corpus = load_causalphys_corpus(
        Path(signature["dataset_dir"]),
        max_images=int(signature["max_images"]),
    )
    selected = select_causalphys_items(
        corpus,
        n_samples=signature.get("n_samples"),
        seed=int(signature["seed"]),
        categories=signature.get("categories", []),
        subcategories=signature.get("subcategories", []),
    )
    return corpus, selected


def _merge_only(args: argparse.Namespace) -> int:
    """Merge completed shards and run the local CausalPhys scorer once."""

    out_dir = Path(args.merge_shards)
    config = load_shard_config(
        out_dir,
        prefix="causalphys_eval_config",
    )
    signature = config["run_signature"]
    corpus, selected = _select_from_signature(signature)
    expected_idx = [item.idx for item in selected]
    if expected_idx != [int(value) for value in config["expected_idx"]]:
        raise ValueError("dataset selection no longer matches shard metadata")
    merge_prediction_shards(
        out_dir,
        expected_idx=expected_idx,
        expected_num_shards=int(config["num_shards"]),
    )
    score_causalphys(
        selected,
        load_prediction_map(out_dir / "predictions.json"),
        corpus=corpus,
        model_tag=str(signature["model_tag"]),
        prompt_style=str(signature["prompt_style"]),
        max_new_tokens=int(signature["max_new_tokens"]),
        out_dir=out_dir,
    )
    return 0


def _dry_run(
    args: argparse.Namespace,
    *,
    corpus: CausalPhysCorpus,
    selected: Sequence[CausalPhysItem],
    shard: Sequence[CausalPhysItem],
) -> int:
    """Print a no-model validation summary and optionally decode one item."""

    media_validation: dict[str, Any] | None = None
    if args.validate_media and shard:
        content = build_causalphys_content(
            shard[0],
            prompt_style=args.prompt_style,
            n_video_frames=args.n_video_frames,
            max_images=args.max_images,
        )
        media_validation = {
            "sample_key": shard[0].sample_key,
            "content_types": [entry["type"] for entry in content],
        }
    payload = {
        "benchmark": "CausalPhys",
        "dry_run": True,
        "dataset_dir": str(corpus.dataset_dir),
        "n_annotations": corpus.n_annotations,
        "n_valid_gold": len(corpus.items),
        "n_invalid_gold": len(corpus.invalid_gold),
        "prompt_style": args.prompt_style,
        "max_new_tokens": args.max_new_tokens,
        "n_selected": len(selected),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "n_shard_items": len(shard),
        "first_sample_keys": [item.sample_key for item in shard[:3]],
        "media_validation": media_validation,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _single_run_seed_answers(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    signature: dict[str, Any],
    selected: Sequence[CausalPhysItem],
    shard: Sequence[CausalPhysItem],
    has_shard_history: bool,
) -> dict[int, str]:
    """Load compatible non-null answers when migrating one run to shards."""

    if args.no_resume or args.num_shards <= 1 or has_shard_history:
        return {}

    source_predictions = out_dir / "predictions.json"
    source_journal = source_predictions.with_suffix(".jsonl")
    if not source_predictions.exists() and not source_journal.exists():
        return {}

    source_config_path = out_dir / "causalphys_eval_config.json"
    if not source_config_path.exists():
        raise ValueError(
            "cannot migrate single-run CausalPhys predictions without "
            f"metadata: {source_config_path}"
        )
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    expected_source_config = {
        "run_signature": signature,
        "num_shards": 1,
        "shard_id": 0,
        "expected_idx": [item.idx for item in selected],
    }
    if source_config != expected_source_config:
        raise ValueError(
            "single-run CausalPhys metadata is incompatible with the "
            "requested sharded resume"
        )

    source_store = PredictionStore(source_predictions, resume=True)
    answers: dict[int, str] = {}
    for item in shard:
        answer = source_store.answer_for(item.idx)
        if answer is not None:
            answers[item.idx] = answer
    return answers


def _run_inference(
    args: argparse.Namespace,
    *,
    corpus: CausalPhysCorpus,
    selected: Sequence[CausalPhysItem],
    shard: Sequence[CausalPhysItem],
    model_tag: str,
    out_dir: Path,
) -> int:
    """Run one CausalPhys shard with resumable local HF inference."""

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = out_dir / "predictions.json"
    audit_file = out_dir / "raw_audit.jsonl"
    config_file = out_dir / "causalphys_eval_config.json"
    if args.num_shards > 1:
        predictions_file = shard_filename(
            predictions_file,
            args.shard_id,
            args.num_shards,
        )
        audit_file = shard_filename(
            audit_file,
            args.shard_id,
            args.num_shards,
        )
        config_file = shard_filename(
            config_file,
            args.shard_id,
            args.num_shards,
        )

    resume = not args.no_resume
    signature = _run_signature(args, model_tag=model_tag)
    config_payload = {
        "run_signature": signature,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "expected_idx": [item.idx for item in selected],
    }
    journal_file = predictions_file.with_suffix(".jsonl")
    has_shard_history = predictions_file.exists() or journal_file.exists()
    seed_answers = _single_run_seed_answers(
        args=args,
        out_dir=out_dir,
        signature=signature,
        selected=selected,
        shard=shard,
        has_shard_history=has_shard_history,
    )
    if resume and has_shard_history:
        if not config_file.exists():
            raise ValueError(
                "cannot resume CausalPhys predictions without matching "
                f"run metadata: {config_file}; use --no-resume or a new out-dir"
            )
        existing_config = json.loads(config_file.read_text(encoding="utf-8"))
        if existing_config != config_payload:
            raise ValueError(
                "CausalPhys resume metadata differs from the requested "
                "prompt/generation protocol; use --no-resume or a new out-dir"
            )
    write_json_atomic(config_file, config_payload)

    store = PredictionStore(predictions_file, resume=resume)
    for idx, answer in seed_answers.items():
        if not store.is_completed(idx):
            store.append(idx, answer)
    if not args.no_audit:
        prepare_jsonl(audit_file, resume=resume)
    remaining = [
        item for item in shard if not store.is_completed(item.idx)
    ]
    log = logging.getLogger("causalphys_vlm_eval")
    if seed_answers:
        log.info(
            "seeded %d non-null answers from the compatible single-run journal",
            len(seed_answers),
        )
    log.info(
        "loading %s for %d remaining items on shard %d/%d",
        args.model_path,
        len(remaining),
        args.shard_id,
        args.num_shards,
    )
    settings = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    if not remaining:
        log.info("all shard items already have parsed answers; skipped model load")
        store.finalize()
        if args.num_shards == 1:
            score_causalphys(
                selected,
                load_prediction_map(predictions_file),
                corpus=corpus,
                model_tag=model_tag,
                prompt_style=args.prompt_style,
                max_new_tokens=args.max_new_tokens,
                out_dir=out_dir,
            )
        return 0

    with open_eval_backend(
        str(args.model_path), family=resolve_eval_family(args.model_path, args.model_family)
    ) as backend:
        for position, item in enumerate(remaining, start=1):
            started = time.perf_counter()
            raw_text = ""
            answer: str | None = None
            rationale: str | None = None
            metadata: dict[str, Any] = {}
            error: str | None = None
            try:
                content = build_causalphys_content(
                    item,
                    prompt_style=args.prompt_style,
                    n_video_frames=args.n_video_frames,
                    max_images=args.max_images,
                )
                result = generate_reference_answer(backend, content, settings)
                raw_text = result.text
                metadata = result.metadata
                parsed_response = parse_causalphys_response(raw_text)
                answer = parsed_response.answer
                rationale = parsed_response.rationale
            except Exception as exc:  # noqa: BLE001 - preserve long eval.
                error = f"{type(exc).__name__}: {exc}"
                log.exception("item failed: %s", item.sample_key)
            store.append(item.idx, answer)
            if not args.no_audit:
                append_jsonl(
                    audit_file,
                    {
                        "idx": item.idx,
                        "sample_key": item.sample_key,
                        "answer": answer,
                        "gold_answer": item.gold_answer,
                        "rationale": rationale,
                        "prompt_style": args.prompt_style,
                        "raw_text": raw_text,
                        "elapsed_s": time.perf_counter() - started,
                        "metadata": metadata,
                        "media_paths": [
                            str(path) for path in item.media_paths
                        ],
                        "error": error,
                    },
                )
            if position % 25 == 0 or position == len(remaining):
                log.info("completed %d/%d remaining items", position, len(remaining))
    store.finalize()

    if args.num_shards == 1:
        score_causalphys(
            selected,
            load_prediction_map(predictions_file),
            corpus=corpus,
            model_tag=model_tag,
            prompt_style=args.prompt_style,
            max_new_tokens=args.max_new_tokens,
            out_dir=out_dir,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run CausalPhys zero-shot evaluation, dry-run, or shard merge."""

    args = _parse_args(argv)
    _configure_logging(args.verbose)
    if args.merge_shards is not None:
        return _merge_only(args)

    _validate_shard_args(args.num_shards, args.shard_id)
    if args.n_video_frames < 1:
        raise ValueError("--n-video-frames must be >= 1")
    if args.max_images < 1:
        raise ValueError("--max-images must be >= 1")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    corpus = load_causalphys_corpus(
        args.dataset_dir,
        max_images=args.max_images,
    )
    selected = select_causalphys_items(
        corpus,
        n_samples=args.n_samples,
        seed=args.seed,
        categories=args.category,
        subcategories=args.subcategory,
    )
    if not selected:
        raise ValueError("CausalPhys selection is empty")
    shard = shard_items(
        selected,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )
    if args.dry_run:
        return _dry_run(
            args,
            corpus=corpus,
            selected=selected,
            shard=shard,
        )

    model_tag = args.model_tag or args.model_path.name
    out_dir = args.out_dir or _default_out_dir(
        model_tag,
        n_samples=args.n_samples,
        prompt_style=args.prompt_style,
        max_new_tokens=args.max_new_tokens,
    )
    return _run_inference(
        args,
        corpus=corpus,
        selected=selected,
        shard=shard,
        model_tag=model_tag,
        out_dir=Path(out_dir),
    )


if __name__ == "__main__":
    raise SystemExit(main())
