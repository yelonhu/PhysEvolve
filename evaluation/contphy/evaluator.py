"""Zero-shot local VLM evaluation for ContPhy test questions."""

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

import numpy as np

from physevolve.inference.hf_backend import (
    presampled_video_metadata,
)

from evaluation.physbench.visual_loader import load_video
from evaluation.common.common import (
    DEFAULT_MODEL_PATH,
    GenerationSettings,
    PredictionStore,
    accuracy,
    append_jsonl,
    generate_reference_answer,
    load_prediction_map,
    load_shard_config,
    merge_prediction_shards,
    parse_short_answer,
    prepare_jsonl,
    safe_model_tag,
    shard_filename,
    shard_items,
    write_json_atomic,
    write_text_atomic,
)
from evaluation.common.family_generate import (
    open_eval_backend,
    resolve_eval_family,
)


DEFAULT_DATASET_DIR = Path(
    os.environ.get(
        "DATASET_DIR",
        "datasets/contphy",
    )
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLING_PROTOCOL_VERSION = "contphy_test_presampled_v1"
_CONTPHY_PRESAMPLED_FPS = 2.0
_BROKEN_PROMPT_TOKEN_CEILING = 250
_PARSER_NAME = "parse_contphy_answer"
_CONFIG_PREFIX = "contphy_eval_config"
_DEFAULT_N_VIDEO_FRAMES = 10
VALID_LETTERS = frozenset("ABCDE")
CHOICE_ANSWER_TYPES = frozenset({"multiple_choice", "single_choice"})
ANSWER_TYPES = ("multiple_choice", "single_choice", "open_ended")
_RGB_SKIP_TOKENS = ("seg", "depth", "mask")
_PREFERRED_RGB_NAME = "output_Full.mp4"
_CHOICE_LETTER_RE = re.compile(r"\b([A-E])\b")

# Zip-native top-level domains only. Nested copies (fluid/fluid, rope/rope,
# ball/soft) are never scanned.
DOMAIN_SPECS: tuple[tuple[str, Path], ...] = (
    ("cloth", Path("cloth") / "question" / "cloth_final2_test_v1.json"),
    ("fluid", Path("fluid") / "question" / "fluid_final_test_v1.json"),
    ("rope", Path("rope") / "question" / "rope_final2_test_v1.json"),
    ("soft", Path("soft") / "question" / "soft_final_test_v1.json"),
)
DOMAIN_ORDER = tuple(domain for domain, _relative in DOMAIN_SPECS)
EXPECTED_COUNTS = {
    "cloth": 450,
    "fluid": 450,
    "rope": 600,
    "soft": 450,
}
EXPECTED_N = sum(EXPECTED_COUNTS.values())
REQUIRED_JSON_KEYS = (
    "question",
    "answer",
    "answer_type",
    "qtype",
    "question_family",
    "question_id",
    "split",
    "video_id",
)

# Transcribed from ContPhy arXiv:2402.06119v1 appendix Table 8 (ar5iv HTML
# labels the same block Table 10). The evaluator does not import ContPhy
# repos. Open-ended is one JSON ``answer_type``, so the three Table 8
# open-ended rows (number / yes-no / yes-no-cna) are collapsed into one
# short-answer suffix.
_POST_PROMPTS: dict[str, str] = {
    "multiple_choice": (
        "\nPlease answer with all correct choices listed in alphabet order, "
        'divided by spaces. For example, you can respond with "A B C". '
        "Please do NOT add any other text in your response."
    ),
    "single_choice": (
        "\nPlease answer with the correct choice. For example, you can "
        'respond with "A". Please do NOT add any other text in your response.'
    ),
    "open_ended": (
        '\nPlease answer with "yes", "no", "can not answer", or a number. '
        'For example, you can respond with "yes". Please do NOT add any '
        "other text in your response."
    ),
}


@dataclass(frozen=True, slots=True)
class ContPhyItem:
    """One scoreable ContPhy test question with a resolved RGB video."""

    idx: int
    sample_key: str
    domain: str
    question: str
    gold_answer: str
    answer_type: str
    qtype: str
    question_family: str
    question_id: str
    video_id: str
    trial_dir: Path
    video_path: Path
    video_name: str


@dataclass(frozen=True, slots=True)
class ContPhyCorpus:
    """Indexed ContPhy test JSON plus resolved RGB trial videos."""

    dataset_dir: Path
    json_paths: tuple[Path, ...]
    n_json_rows: int
    items: tuple[ContPhyItem, ...]
    n_by_domain: dict[str, int]


class ContPhyPackingError(RuntimeError):
    """Raised when processor token counts still look like fps=24 resampling."""


class ContPhyMissingVideoError(FileNotFoundError):
    """Raised when a trial directory has no usable RGB mp4."""


class ContPhyRGBSelectionError(FileNotFoundError):
    """Raised when a trial directory exists but RGB picking finds zero files."""


def post_prompt_for_answer_type(answer_type: str) -> str:
    """Return the Table 8 suffix for a ContPhy ``answer_type``."""

    try:
        return _POST_PROMPTS[answer_type]
    except KeyError as exc:
        raise ValueError(
            f"unsupported ContPhy answer_type {answer_type!r}; "
            f"expected one of {ANSWER_TYPES}"
        ) from exc


def json_path_for(dataset_dir: Path, domain: str) -> Path:
    """Return the zip-native test JSON for one top-level domain."""

    for name, relative in DOMAIN_SPECS:
        if name == domain:
            return Path(dataset_dir).resolve() / relative
    raise ValueError(
        f"unsupported ContPhy domain {domain!r}; expected one of {DOMAIN_ORDER}"
    )


def trial_dir_for(dataset_dir: Path, domain: str, video_id: str) -> Path:
    """Return ``{dataset_dir}/{domain}/video/{video_id}``."""

    if domain not in DOMAIN_ORDER:
        raise ValueError(
            f"unsupported ContPhy domain {domain!r}; expected one of {DOMAIN_ORDER}"
        )
    return (Path(dataset_dir).resolve() / domain / "video" / video_id).resolve()


def select_rgb_video(trial_dir: Path) -> Path:
    """Pick the RGB positive mp4 from a trial directory.

    Lists ``*.mp4`` in ``trial_dir`` (non-recursive), drops names containing
    ``seg`` / ``depth`` / ``mask``, then prefers ``output_Full.mp4`` and
    otherwise takes the lexicographic first remaining file.
    """

    root = Path(trial_dir)
    if not root.is_dir():
        raise ContPhyMissingVideoError(f"ContPhy trial directory is missing: {root}")
    rgb_videos: list[Path] = []
    for path in root.glob("*.mp4"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(token in name for token in _RGB_SKIP_TOKENS):
            continue
        rgb_videos.append(path)
    if not rgb_videos:
        raise ContPhyRGBSelectionError(
            f"no RGB mp4 in {root} after dropping seg/depth/mask"
        )
    preferred = [path for path in rgb_videos if path.name == _PREFERRED_RGB_NAME]
    if preferred:
        return preferred[0]
    return sorted(rgb_videos, key=lambda path: path.name)[0]


def normalize_choice_gold(raw: str) -> str:
    """Sort unique A-E letters from a gold string into ``A B C`` form."""

    letters = _CHOICE_LETTER_RE.findall(str(raw).upper())
    if not letters:
        raise ValueError(f"choice gold has no A-E letters: {raw!r}")
    return " ".join(sorted(set(letters)))


def build_contphy_prompt(item: ContPhyItem) -> str:
    """Build question + Table 8 suffix without writing gold into the prompt."""

    return f"{item.question}{post_prompt_for_answer_type(item.answer_type)}"


def assert_prompt_does_not_leak_gold(prompt: str, item: ContPhyItem) -> None:
    """Check the prompt is question plus suffix and does not end with gold."""

    expected = build_contphy_prompt(item)
    if prompt != expected:
        raise ValueError("ContPhy prompt is not question plus Table 8 suffix")
    suffix = post_prompt_for_answer_type(item.answer_type)
    if not prompt.endswith(suffix):
        raise ValueError("ContPhy prompt must end with the Table 8 suffix")
    if prompt.rstrip().endswith(str(item.gold_answer).strip()):
        raise ValueError("dry-run prompt leaked gold as a trailing completion")


def _require_json_keys(row: dict[str, Any], *, source: Path, row_index: int) -> None:
    """Refuse a question object that is missing required annotation keys."""

    missing = [key for key in REQUIRED_JSON_KEYS if key not in row]
    if missing:
        raise ValueError(
            f"{source} row {row_index} is missing keys {missing}"
        )


def _load_domain_rows(json_path: Path) -> list[dict[str, Any]]:
    """Load one test JSON list and refuse non-list payloads."""

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(
            f"{json_path} must contain a JSON list; got {type(payload)!r}"
        )
    return payload


def load_contphy_corpus(
    dataset_dir: Path,
    *,
    strict_official_counts: bool = True,
) -> ContPhyCorpus:
    """Load zip-native test JSON and fail fast if any RGB video is missing."""

    root = Path(dataset_dir).resolve()
    items: list[ContPhyItem] = []
    json_paths: list[Path] = []
    missing_videos: list[str] = []
    n_by_domain: dict[str, int] = {domain: 0 for domain in DOMAIN_ORDER}
    idx = 0

    for domain, relative in DOMAIN_SPECS:
        json_path = root / relative
        if not json_path.is_file():
            raise FileNotFoundError(f"ContPhy test JSON not found: {json_path}")
        json_paths.append(json_path)
        rows = _load_domain_rows(json_path)
        if strict_official_counts and len(rows) != EXPECTED_COUNTS[domain]:
            raise ValueError(
                f"{json_path} must contain {EXPECTED_COUNTS[domain]} test "
                f"rows; got {len(rows)}"
            )
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(
                    f"{json_path} row {row_index} must be an object; "
                    f"got {type(row)!r}"
                )
            _require_json_keys(row, source=json_path, row_index=row_index)
            split = str(row.get("split") or "").strip()
            if split != "test":
                raise ValueError(
                    f"{json_path} row {row_index} has split {split!r}; "
                    "expected 'test'"
                )
            question = str(row.get("question") or "").strip()
            if not question:
                raise ValueError(f"{json_path} row {row_index} has an empty question")
            answer_type = str(row.get("answer_type") or "").strip()
            if answer_type not in _POST_PROMPTS:
                raise ValueError(
                    f"{json_path} row {row_index} has unsupported "
                    f"answer_type {row.get('answer_type')!r}"
                )
            raw_gold = str(row.get("answer") or "").strip()
            if not raw_gold:
                raise ValueError(f"{json_path} row {row_index} has an empty answer")
            gold_answer = (
                normalize_choice_gold(raw_gold)
                if answer_type in CHOICE_ANSWER_TYPES
                else raw_gold
            )
            question_id = str(row.get("question_id") or "").strip()
            video_id = str(row.get("video_id") or "").strip()
            qtype = str(row.get("qtype") or "").strip()
            question_family = str(row.get("question_family") or "").strip()
            if not question_id or not video_id or not qtype:
                raise ValueError(
                    f"{json_path} row {row_index} is missing "
                    "question_id/video_id/qtype"
                )
            trial_dir = trial_dir_for(root, domain, video_id)
            try:
                video_path = select_rgb_video(trial_dir)
            except (ContPhyMissingVideoError, ContPhyRGBSelectionError):
                missing_videos.append(f"{domain}/video/{video_id}")
                continue
            items.append(
                ContPhyItem(
                    idx=idx,
                    sample_key=f"{domain}:{question_id}",
                    domain=domain,
                    question=question,
                    gold_answer=gold_answer,
                    answer_type=answer_type,
                    qtype=qtype,
                    question_family=question_family,
                    question_id=question_id,
                    video_id=video_id,
                    trial_dir=trial_dir,
                    video_path=video_path,
                    video_name=video_path.name,
                )
            )
            n_by_domain[domain] += 1
            idx += 1

    if missing_videos:
        preview = ", ".join(missing_videos[:5])
        extra = "" if len(missing_videos) <= 5 else f" (+{len(missing_videos) - 5} more)"
        raise ContPhyMissingVideoError(
            f"{len(missing_videos)} ContPhy videos are missing under {root}; "
            f"first: {preview}{extra}"
        )

    if strict_official_counts:
        for domain, expected in EXPECTED_COUNTS.items():
            if n_by_domain[domain] != expected:
                raise ValueError(
                    f"ContPhy domain {domain} must index {expected} items; "
                    f"got {n_by_domain[domain]}"
                )
        if len(items) != EXPECTED_N:
            raise ValueError(
                f"ContPhy must index {EXPECTED_N} scoreable items; got {len(items)}"
            )

    return ContPhyCorpus(
        dataset_dir=root,
        json_paths=tuple(json_paths),
        n_json_rows=len(items),
        items=tuple(items),
        n_by_domain=dict(n_by_domain),
    )


def select_contphy_items(
    corpus: ContPhyCorpus,
    *,
    n_samples: int | None,
    seed: int,
    domains: Sequence[str] = (),
) -> list[ContPhyItem]:
    """Filter by domain and deterministically sample scoreable items."""

    domain_filter = {value.casefold() for value in domains}
    unknown = domain_filter - {name.casefold() for name in DOMAIN_ORDER}
    if unknown:
        raise ValueError(
            f"unsupported --domain values {sorted(unknown)}; "
            f"expected one of {DOMAIN_ORDER}"
        )
    selected = [
        item
        for item in corpus.items
        if not domain_filter or item.domain.casefold() in domain_filter
    ]
    if n_samples is not None:
        if n_samples < 1:
            raise ValueError(f"--n-samples must be >= 1; got {n_samples}")
        if n_samples < len(selected):
            rng = random.Random(seed)
            selected = rng.sample(selected, n_samples)
    return sorted(selected, key=lambda item: item.idx)


def load_contphy_frames(
    item: ContPhyItem,
    *,
    n_video_frames: int,
) -> np.ndarray:
    """Uniformly sample ``n_video_frames`` RGB frames, padding short clips."""

    if n_video_frames < 1:
        raise ValueError(f"n_video_frames must be >= 1; got {n_video_frames}")
    frames = load_video(item.video_path, n_frames=n_video_frames)
    if frames.shape[0] != n_video_frames:
        raise ValueError(
            f"decoded {frames.shape[0]} frames from {item.video_path}; "
            f"expected {n_video_frames}"
        )
    return frames


def build_contphy_multimodal_content(
    frames: np.ndarray,
    item: ContPhyItem,
) -> list[dict[str, Any]]:
    """Build user-only content with pre-sampled video metadata.

    The delivered clip is already uniformly sampled (and last-frame padded).
    Metadata ``fps`` is the Qwen3-VL timestamp rate for that clip (training
    default 2.0), not native ``CAP_PROP_FPS``. ``do_sample_frames`` is
    omitted from the dict; HFBackend passes it as a processor keyword.
    """

    metadata = presampled_video_metadata(
        frames,
        fps=_CONTPHY_PRESAMPLED_FPS,
    )
    return [
        {"type": "video", "video": (frames, metadata)},
        {"type": "text", "text": build_contphy_prompt(item)},
    ]


def frames_and_metadata_from_content(
    content: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Unpack ContPhy ``(frames, video_metadata)`` video content."""

    if not content or content[0].get("type") != "video":
        raise TypeError("ContPhy content must start with a video entry")
    video = content[0]["video"]
    if not (
        isinstance(video, tuple)
        and len(video) == 2
        and isinstance(video[1], dict)
    ):
        raise TypeError(
            "ContPhy video content must be (frames, metadata); "
            f"got {type(video)!r}"
        )
    frames, metadata = video
    if not isinstance(frames, np.ndarray):
        raise TypeError(
            f"ContPhy video frames must be np.ndarray; got {type(frames)!r}"
        )
    return frames, metadata


def assert_prompt_tokens_not_resampled(
    metadata: dict[str, Any],
    *,
    family: str = "qwen",
) -> None:
    """Stop a Qwen shard whose prompt is still the ~211-token resampled clip.

    InternVL / Llama / Pixtral do not use Qwen3-VL fps=24 packing, so this
    gate is skipped for those families.
    """

    if family != "qwen":
        return
    prompt_token_count = metadata.get("prompt_token_count")
    if prompt_token_count is None:
        return
    if int(prompt_token_count) <= _BROKEN_PROMPT_TOKEN_CEILING:
        raise ContPhyPackingError(
            f"prompt_token_count={prompt_token_count} looks like Qwen3-VL "
            "fps=24 resampling (Physion++/IntPhys voided dumps were ~211). "
            "Stop this shard; do not keep writing answers."
        )


def parse_choice_letters(raw_text: str) -> str | None:
    """Extract unique A-E letters, sorted, joined by spaces.

    PhysBench's A-D single-letter parser cannot return ``A B`` or ``E``.
    """

    source = parse_short_answer(raw_text)
    if source is None:
        return None
    letters = _CHOICE_LETTER_RE.findall(source.upper())
    if not letters:
        return None
    return " ".join(sorted(set(letters)))


def parse_open_ended_answer(raw_text: str) -> str | None:
    """Normalize a short answer with ``parse_short_answer`` then lower/strip."""

    parsed = parse_short_answer(raw_text)
    if parsed is None:
        return None
    cleaned = parsed.lower().strip()
    return cleaned or None


def parse_contphy_answer(raw_text: str, answer_type: str) -> str | None:
    """Parse one model completion according to ``answer_type``."""

    if answer_type in CHOICE_ANSWER_TYPES:
        return parse_choice_letters(raw_text)
    if answer_type == "open_ended":
        return parse_open_ended_answer(raw_text)
    raise ValueError(f"unsupported ContPhy answer_type {answer_type!r}")


def _gold_for_scoring(item: ContPhyItem) -> str:
    """Return the normalized gold string compared against a parsed answer."""

    if item.answer_type == "open_ended":
        return item.gold_answer.lower().strip()
    return item.gold_answer


def score_contphy(
    items: Sequence[ContPhyItem],
    predictions: dict[int, str | None],
    *,
    corpus: ContPhyCorpus,
    model_tag: str,
    n_video_frames: int,
    max_new_tokens: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Compute local exact-match ACC and write JSON/Markdown summaries."""

    overall = _empty_score_counts()
    domain_stats: dict[str, dict[str, int]] = defaultdict(_empty_score_counts)
    qtype_stats: dict[str, dict[str, int]] = defaultdict(_empty_score_counts)
    answer_type_stats: dict[str, dict[str, int]] = defaultdict(
        _empty_score_counts
    )

    for item in items:
        raw_answer = predictions.get(item.idx)
        parsed = raw_answer.strip() if isinstance(raw_answer, str) else None
        if parsed == "":
            parsed = None
        gold = _gold_for_scoring(item)
        correct = parsed is not None and parsed == gold
        for stats in (
            overall,
            domain_stats[item.domain],
            qtype_stats[item.qtype],
            answer_type_stats[item.answer_type],
        ):
            stats["total"] += 1
            stats["parsed"] += int(parsed is not None)
            stats["correct"] += int(correct)

    summary = {
        "benchmark": "ContPhy",
        "task": "contphy_test",
        "track": "reference_appendix",
        "protocol": _SAMPLING_PROTOCOL_VERSION,
        "official_readout_comparable": False,
        "dynamic_per_option_scoring": False,
        "model_tag": model_tag,
        "dataset_dir": str(corpus.dataset_dir),
        "json_paths": [str(path) for path in corpus.json_paths],
        "n_json_rows": corpus.n_json_rows,
        "n_corpus_items": len(corpus.items),
        "n_by_domain": dict(corpus.n_by_domain),
        "n_selected": len(items),
        "n_video_frames": n_video_frames,
        "video_fps": _CONTPHY_PRESAMPLED_FPS,
        "do_sample_frames": False,
        "sampling_protocol_version": _SAMPLING_PROTOCOL_VERSION,
        "domains": list(DOMAIN_ORDER),
        "post_prompts": dict(_POST_PROMPTS),
        "parser": _PARSER_NAME,
        "max_new_tokens": max_new_tokens,
        "overall": _stats_payload(overall),
        "by_domain": {
            key: _stats_payload(value)
            for key, value in sorted(domain_stats.items())
        },
        "by_qtype": {
            key: _stats_payload(value)
            for key, value in sorted(qtype_stats.items())
        },
        "by_answer_type": {
            key: _stats_payload(value)
            for key, value in sorted(answer_type_stats.items())
        },
    }
    write_json_atomic(Path(out_dir) / "summary.json", summary)
    write_text_atomic(Path(out_dir) / "report.md", _render_report(summary))
    return summary


def _empty_score_counts() -> dict[str, int]:
    """Return mutable item-level metric counters."""

    return {"correct": 0, "parsed": 0, "total": 0}


def _stats_payload(stats: dict[str, int]) -> dict[str, int | float]:
    """Convert running counts into a serializable metric payload."""

    return {
        "accuracy": accuracy(stats["correct"], stats["total"]),
        "parse_rate": accuracy(stats["parsed"], stats["total"]),
        "correct": stats["correct"],
        "parsed": stats["parsed"],
        "total": stats["total"],
    }


def _render_report(summary: dict[str, Any]) -> str:
    """Render a compact Markdown report for a ContPhy run."""

    overall = summary["overall"]
    lines = [
        "# ContPhy zero-shot VLM reference evaluation",
        "",
        f"- Model: `{summary['model_tag']}`",
        "- Task: ContPhy test (one generation per JSON row)",
        "- PhysEvolve paper benchmark",
        "- Official GPT-4V multi-question batching / per-option dynamic MC: no",
        "- Official readout comparable: no",
        (
            f"- Overall ACC: {overall['accuracy'] * 100:.2f}% "
            f"({overall['correct']}/{overall['total']})"
        ),
        (
            f"- Parse rate: {overall['parse_rate'] * 100:.2f}% "
            f"({overall['parsed']}/{overall['total']})"
        ),
        (
            f"- Video sampling: {summary['n_video_frames']} frames, "
            f"metadata.fps={summary['video_fps']}, do_sample_frames=false"
        ),
        f"- Parser: `{summary['parser']}` (unparsed counts as incorrect)",
        "",
        "## Domain breakdown",
        "",
        "| Domain | ACC | Correct / Total | Parse rate |",
        "|---|---:|---:|---:|",
    ]
    for name, stats in summary["by_domain"].items():
        lines.append(_table_row(name, stats))
    lines.extend(
        [
            "",
            "## Answer type",
            "",
            "| Answer type | ACC | Correct / Total | Parse rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, stats in summary["by_answer_type"].items():
        lines.append(_table_row(name, stats))
    lines.extend(
        [
            "",
            "## Question type (``qtype``)",
            "",
            "| qtype | ACC | Correct / Total | Parse rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, stats in summary["by_qtype"].items():
        lines.append(_table_row(name, stats))
    return "\n".join(lines) + "\n"


def _table_row(name: str, stats: dict[str, Any]) -> str:
    """Format one ACC table row."""

    return (
        f"| {name} | {stats['accuracy'] * 100:.2f}% | "
        f"{stats['correct']} / {stats['total']} | "
        f"{stats['parse_rate'] * 100:.2f}% |"
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the ContPhy evaluator command line."""

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
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument(
        "--n-video-frames",
        type=int,
        default=_DEFAULT_N_VIDEO_FRAMES,
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
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
    n_video_frames: int,
) -> Path:
    """Return the default ContPhy reference result directory."""

    sample_label = "full" if n_samples is None else f"n{n_samples}"
    protocol = f"contphy_test_f{n_video_frames}_presampled"
    return (
        _REPO_ROOT
        / "outputs" / "evaluation"
        / "contphy"
        / f"{safe_model_tag(model_tag)}_{sample_label}_{protocol}"
    )


def _run_signature(
    args: argparse.Namespace,
    *,
    model_tag: str,
) -> dict[str, Any]:
    """Build merge-time metadata that must agree across shards."""

    payload: dict[str, Any] = {
        "benchmark": "contphy",
        "task": "contphy_test",
        "protocol": _SAMPLING_PROTOCOL_VERSION,
        "model_path": str(args.model_path.resolve()) if args.model_path.exists() else str(args.model_path),
        "model_tag": model_tag,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "n_samples": args.n_samples,
        "seed": args.seed,
        "domains": list(args.domain),
        "n_video_frames": args.n_video_frames,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "post_prompts": dict(_POST_PROMPTS),
        "parser": _PARSER_NAME,
        "sampling_protocol_version": _SAMPLING_PROTOCOL_VERSION,
        "video_fps": _CONTPHY_PRESAMPLED_FPS,
        "do_sample_frames": False,
        "domain_order": list(DOMAIN_ORDER),
    }
    family = resolve_eval_family(args.model_path, args.model_family)
    if family != "qwen":
        payload["model_family"] = family
    return payload


def _select_from_signature(
    signature: dict[str, Any],
) -> tuple[ContPhyCorpus, list[ContPhyItem]]:
    """Reconstruct the exact selected item set recorded by shard configs."""

    corpus = load_contphy_corpus(Path(signature["dataset_dir"]))
    selected = select_contphy_items(
        corpus,
        n_samples=signature.get("n_samples"),
        seed=int(signature["seed"]),
        domains=signature.get("domains", []),
    )
    return corpus, selected


def _merge_only(args: argparse.Namespace) -> int:
    """Merge completed shards and run the local ContPhy scorer once."""

    out_dir = Path(args.merge_shards)
    config = load_shard_config(out_dir, prefix=_CONFIG_PREFIX)
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
    score_contphy(
        selected,
        load_prediction_map(out_dir / "predictions.json"),
        corpus=corpus,
        model_tag=str(signature["model_tag"]),
        n_video_frames=int(signature["n_video_frames"]),
        max_new_tokens=int(signature["max_new_tokens"]),
        out_dir=out_dir,
    )
    return 0


def _validate_media_item(
    item: ContPhyItem,
    *,
    n_video_frames: int,
) -> dict[str, Any]:
    """Decode one item and check packing, RGB filename, and gold-free prompt."""

    name_lower = item.video_name.lower()
    if any(token in name_lower for token in _RGB_SKIP_TOKENS):
        raise ValueError(
            f"selected video {item.video_name!r} contains seg/depth/mask"
        )
    frames = load_contphy_frames(item, n_video_frames=n_video_frames)
    content = build_contphy_multimodal_content(frames, item)
    unpacked_frames, metadata = frames_and_metadata_from_content(content)
    video_shape = list(unpacked_frames.shape)
    if video_shape[0] != n_video_frames:
        raise ValueError(
            "ContPhy dry-run video_shape[0] must equal "
            f"n_video_frames={n_video_frames}; got {video_shape}"
        )
    if float(metadata["fps"]) != float(_CONTPHY_PRESAMPLED_FPS):
        raise ValueError(
            "ContPhy video_metadata.fps must be "
            f"{_CONTPHY_PRESAMPLED_FPS}; got {metadata['fps']}"
        )
    if "do_sample_frames" in metadata:
        raise ValueError(
            "ContPhy video_metadata must omit do_sample_frames; "
            "HFBackend passes it as a processor keyword"
        )
    prompt = build_contphy_prompt(item)
    assert_prompt_does_not_leak_gold(prompt, item)
    return {
        "sample_key": item.sample_key,
        "domain": item.domain,
        "video_id": item.video_id,
        "video_path": str(item.video_path),
        "video_name": item.video_name,
        "answer_type": item.answer_type,
        "qtype": item.qtype,
        "prompt": prompt,
        "gold_in_trailing_completion": False,
        "content_types": [entry["type"] for entry in content],
        "video_shape": video_shape,
        "video_fps": metadata["fps"],
        "do_sample_frames_in_metadata": "do_sample_frames" in metadata,
    }


def _dry_run(
    args: argparse.Namespace,
    *,
    corpus: ContPhyCorpus,
    selected: Sequence[ContPhyItem],
    shard: Sequence[ContPhyItem],
) -> int:
    """Print a no-model validation summary and optionally decode one item."""

    media_validation: dict[str, Any] | None = None
    if args.validate_media and shard:
        media_validation = _validate_media_item(
            shard[0],
            n_video_frames=args.n_video_frames,
        )
    payload = {
        "benchmark": "ContPhy",
        "task": "contphy_test",
        "dry_run": True,
        "dataset_dir": str(corpus.dataset_dir),
        "json_paths": [str(path) for path in corpus.json_paths],
        "n_json_rows": corpus.n_json_rows,
        "n_corpus_items": len(corpus.items),
        "n_by_domain": dict(corpus.n_by_domain),
        "n_selected": len(selected),
        "protocol": _SAMPLING_PROTOCOL_VERSION,
        "post_prompts": dict(_POST_PROMPTS),
        "parser": _PARSER_NAME,
        "n_video_frames": args.n_video_frames,
        "video_fps": _CONTPHY_PRESAMPLED_FPS,
        "do_sample_frames": False,
        "max_new_tokens": args.max_new_tokens,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "n_shard_items": len(shard),
        "first_sample_keys": [item.sample_key for item in shard[:3]],
        "media_validation": media_validation,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _run_inference(
    args: argparse.Namespace,
    *,
    corpus: ContPhyCorpus,
    selected: Sequence[ContPhyItem],
    shard: Sequence[ContPhyItem],
    model_tag: str,
    out_dir: Path,
) -> int:
    """Run one ContPhy shard with resumable local HF inference."""

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = out_dir / "predictions.json"
    audit_file = out_dir / "raw_audit.jsonl"
    config_file = out_dir / f"{_CONFIG_PREFIX}.json"
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
    has_history = predictions_file.exists() or journal_file.exists()
    if resume and has_history:
        if not config_file.exists():
            raise ValueError(
                "cannot resume ContPhy predictions without matching "
                f"run metadata: {config_file}; use --no-resume or a new out-dir"
            )
        existing_config = json.loads(config_file.read_text(encoding="utf-8"))
        if existing_config != config_payload:
            raise ValueError(
                "ContPhy resume metadata differs from the requested "
                "prompt/generation protocol; use --no-resume or a new out-dir"
            )
    else:
        write_json_atomic(config_file, config_payload)

    store = PredictionStore(predictions_file, resume=resume)
    if not args.no_audit:
        prepare_jsonl(audit_file, resume=resume)
    remaining = [item for item in shard if not store.is_completed(item.idx)]
    log = logging.getLogger("contphy_vlm_eval")
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
            score_contphy(
                selected,
                load_prediction_map(predictions_file),
                corpus=corpus,
                model_tag=model_tag,
                n_video_frames=args.n_video_frames,
                max_new_tokens=args.max_new_tokens,
                out_dir=out_dir,
            )
        return 0

    with open_eval_backend(
        str(args.model_path),
        family=resolve_eval_family(args.model_path, args.model_family),
    ) as backend:
        for position, item in enumerate(remaining, start=1):
            started = time.perf_counter()
            raw_text = ""
            answer: str | None = None
            metadata: dict[str, Any] = {}
            error: str | None = None
            video_shape: list[int] | None = None
            video_fps: float | None = None
            try:
                frames = load_contphy_frames(
                    item,
                    n_video_frames=args.n_video_frames,
                )
                content = build_contphy_multimodal_content(frames, item)
                unpacked, video_metadata = frames_and_metadata_from_content(
                    content
                )
                video_shape = list(unpacked.shape)
                video_fps = float(video_metadata["fps"])
                result = generate_reference_answer(backend, content, settings)
                raw_text = result.text
                metadata = result.metadata
                assert_prompt_tokens_not_resampled(
                    metadata,
                    family=resolve_eval_family(
                        args.model_path, args.model_family
                    ),
                )
                answer = parse_contphy_answer(raw_text, item.answer_type)
            except ContPhyPackingError:
                raise
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
                        "domain": item.domain,
                        "qtype": item.qtype,
                        "answer_type": item.answer_type,
                        "raw_text": raw_text,
                        "elapsed_s": time.perf_counter() - started,
                        "metadata": metadata,
                        "video_path": str(item.video_path),
                        "video_name": item.video_name,
                        "video_shape": video_shape,
                        "video_fps": video_fps,
                        "error": error,
                    },
                )
            if position % 25 == 0 or position == len(remaining):
                log.info(
                    "completed %d/%d remaining items",
                    position,
                    len(remaining),
                )
    store.finalize()

    if args.num_shards == 1:
        score_contphy(
            selected,
            load_prediction_map(predictions_file),
            corpus=corpus,
            model_tag=model_tag,
            n_video_frames=args.n_video_frames,
            max_new_tokens=args.max_new_tokens,
            out_dir=out_dir,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run ContPhy zero-shot evaluation, dry-run, or shard merge."""

    args = _parse_args(argv)
    _configure_logging(args.verbose)
    if args.merge_shards is not None:
        return _merge_only(args)

    _validate_shard_args(args.num_shards, args.shard_id)
    if args.n_video_frames < 1:
        raise ValueError("--n-video-frames must be >= 1")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    corpus = load_contphy_corpus(args.dataset_dir)
    selected = select_contphy_items(
        corpus,
        n_samples=args.n_samples,
        seed=args.seed,
        domains=args.domain,
    )
    if not selected:
        raise ValueError("ContPhy selection is empty")
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
        n_video_frames=args.n_video_frames,
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
