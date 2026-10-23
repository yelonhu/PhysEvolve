"""Shared mechanics for zero-shot VLM reference-benchmark evaluation.

This module deliberately reuses PhysEvolve's HuggingFace backend, PhysBench
visual decoders, answer parsers, and deterministic shard file contract.  It
does not import or invoke any benchmark-specific official scorer.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from evaluation.physbench.qwen_official_adapter import parse_physbench_mc_answer
from evaluation.physbench.sharding import (
    MergeReport,
    discover_shard_files,
    merge_predictions,
    shard_filename,
    shard_items,
)
from evaluation.physbench.visual_loader import load_image, load_video
from physevolve.inference.hf_backend import (
    DEFAULT_MODEL,
    GenerationResult,
)
from physevolve.inference.parsing import extract_answer_block


DEFAULT_MODEL_PATH = Path(DEFAULT_MODEL)
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})

class MultimodalGenerator(Protocol):
    """Generate surface shared by HFBackend and per-family eval backends."""

    def generate_multimodal_with_metadata(
        self,
        content: list[dict[str, Any]] | None = None,
        *,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> GenerationResult:
        ...


_ANSWER_OPEN_RE = re.compile(r"<answer>(.*)$", re.IGNORECASE | re.DOTALL)
_RESULT_RE = re.compile(r"<result>(.*?)</result>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Deterministic generation settings shared by reference evaluators."""

    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None


def build_image_content(
    image_paths: Sequence[str | Path],
    prompt: str,
    *,
    max_images: int = 12,
) -> list[dict[str, Any]]:
    """Load an ordered image sequence and append its text prompt."""

    paths = tuple(Path(path) for path in image_paths)
    if not paths:
        raise ValueError("at least one image path is required")
    if len(paths) > max_images:
        raise ValueError(
            f"image sequence has {len(paths)} entries; maximum is {max_images}"
        )

    content: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported image extension: {path}")
        content.append({"type": "image", "image": load_image(path)})
    content.append({"type": "text", "text": prompt})
    return content


def build_video_content(
    video_path: str | Path,
    prompt: str,
    *,
    n_frames: int,
) -> list[dict[str, Any]]:
    """Load a video and append its text prompt."""

    path = Path(video_path)
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"unsupported video extension: {path}")
    frames = load_video(path, n_frames=n_frames)
    return video_content_from_frames(frames, prompt)


def build_video_prefix_content(
    video_path: str | Path,
    prompt: str,
    *,
    n_frames: int,
    end_frame_exclusive: int,
) -> list[dict[str, Any]]:
    """Load a uniformly sampled video prefix and append its text prompt."""

    path = Path(video_path)
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"unsupported video extension: {path}")
    frames = load_video_prefix(
        path,
        n_frames=n_frames,
        end_frame_exclusive=end_frame_exclusive,
    )
    return video_content_from_frames(frames, prompt)


def load_video_prefix(
    video_path: str | Path,
    *,
    n_frames: int,
    end_frame_exclusive: int,
) -> np.ndarray:
    """Sample RGB frames only from ``[0, end_frame_exclusive)``.

    This benchmark-facing helper mirrors the existing PhysBench bucket-start
    sampling and last-frame padding contract without changing PhysBench's
    public loader. It is intentionally strict about the supplied cutoff so a
    malformed benchmark manifest cannot expose future answer frames.
    """

    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1; got {n_frames}")
    if end_frame_exclusive < 1:
        raise ValueError(
            "end_frame_exclusive must be >= 1; "
            f"got {end_frame_exclusive}"
        )

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"video not found: {path}")

    import cv2  # local import — heavy dependency

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cv2 could not open video: {path}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < end_frame_exclusive:
            raise ValueError(
                f"video has {total_frames} frames but cutoff is "
                f"{end_frame_exclusive}: {path}"
            )

        indices = uniform_prefix_frame_indices(
            end_frame_exclusive,
            n_frames,
        )
        frames: list[np.ndarray] = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(
                    f"failed to decode frame {frame_index} of {path}"
                )
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb.astype(np.uint8, copy=False))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"no video frames decoded from {path}")
    while len(frames) < n_frames:
        frames.append(frames[-1].copy())
    return np.ascontiguousarray(np.stack(frames[:n_frames]), dtype=np.uint8)


def uniform_prefix_frame_indices(
    end_frame_exclusive: int,
    n_frames: int,
) -> list[int]:
    """Return deterministic bucket-start indices inside an observed prefix."""

    if end_frame_exclusive < 1:
        raise ValueError(
            "end_frame_exclusive must be >= 1; "
            f"got {end_frame_exclusive}"
        )
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1; got {n_frames}")
    if n_frames >= end_frame_exclusive:
        return list(range(end_frame_exclusive))
    step = end_frame_exclusive / n_frames
    return [
        min(end_frame_exclusive - 1, int(step * index))
        for index in range(n_frames)
    ]


def video_content_from_frames(
    frames: np.ndarray,
    prompt: str,
) -> list[dict[str, Any]]:
    """Build model content from already-decoded video frames."""

    return [
        {"type": "video", "video": frames},
        {"type": "text", "text": prompt},
    ]


def generate_reference_answer(
    backend: MultimodalGenerator,
    content: list[dict[str, Any]],
    settings: GenerationSettings,
) -> GenerationResult:
    """Run one reference-benchmark generation with the shared backend."""

    return backend.generate_multimodal_with_metadata(
        content,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        stop_sequences=["</answer>", "<|im_end|>"],
    )


def parse_short_answer(raw_text: str) -> str | None:
    """Extract a short free-form value from ``<answer>`` model output.

    The canonical closed ``<answer>`` block is delegated to PhysEvolve's
    existing parser.  Compatibility fallbacks accept an unclosed block,
    CausalPhys's historical ``<result>`` wrapper, or a terse bare response
    after the final ``</think>`` block.
    """

    block = extract_answer_block(raw_text)
    if block is not None:
        return _clean_short_answer(block)

    result_match = _RESULT_RE.search(raw_text)
    if result_match is not None:
        return _clean_short_answer(result_match.group(1))

    open_match = _ANSWER_OPEN_RE.search(raw_text)
    if open_match is not None:
        return _clean_short_answer(open_match.group(1))

    think_matches = list(_THINK_CLOSE_RE.finditer(raw_text))
    candidate = (
        raw_text[think_matches[-1].end() :]
        if think_matches
        else raw_text
    )
    return _clean_short_answer(candidate)


def _clean_short_answer(text: str) -> str | None:
    """Strip model special tokens and reject empty short answers."""

    cleaned = text.replace("<|im_end|>", "").strip()
    return cleaned or None


class PredictionStore:
    """Append-only resumable journal with canonical JSON finalization.

    Per-item appends go to ``*.jsonl`` so long CLEVRER runs do not rewrite a
    growing JSON array after every generation.  :meth:`finalize` materializes
    the ``[{"idx", "answer"}, ...]`` array consumed by the shared strict merge.
    """

    def __init__(self, predictions_file: Path, *, resume: bool = True) -> None:
        self.predictions_file = Path(predictions_file)
        self.journal_file = self.predictions_file.with_suffix(".jsonl")
        self.predictions_file.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            self.predictions_file.unlink(missing_ok=True)
            self.journal_file.unlink(missing_ok=True)
        self._answers: dict[int, str | None] = {}
        if resume:
            self._load_existing()

    @property
    def completed_ids(self) -> set[int]:
        """Return IDs with a non-null parsed answer."""

        return {
            idx for idx, answer in self._answers.items() if answer is not None
        }

    def is_completed(self, idx: int) -> bool:
        """Return whether ``idx`` already has a non-null parsed answer."""

        return self._answers.get(int(idx)) is not None

    def answer_for(self, idx: int) -> str | None:
        """Return the latest parsed answer for ``idx``, if one exists."""

        return self._answers.get(int(idx))

    def append(self, idx: int, answer: str | None) -> None:
        """Append one prediction and update the in-memory latest value."""

        normalized_idx = int(idx)
        if answer is not None or normalized_idx not in self._answers:
            self._answers[normalized_idx] = answer
        append_jsonl(
            self.journal_file,
            {"idx": normalized_idx, "answer": answer},
        )

    def finalize(self) -> Path:
        """Write the canonical sorted prediction array atomically."""

        rows = [
            {"idx": idx, "answer": self._answers[idx]}
            for idx in sorted(self._answers)
        ]
        write_json_atomic(self.predictions_file, rows)
        return self.predictions_file

    def _load_existing(self) -> None:
        """Load a prior canonical file and append journal for resume."""

        if self.predictions_file.exists():
            payload = json.loads(
                self.predictions_file.read_text(encoding="utf-8")
            )
            if not isinstance(payload, list):
                raise ValueError(
                    f"prediction file must contain a JSON list: "
                    f"{self.predictions_file}"
                )
            self._merge_rows(payload)

        if self.journal_file.exists():
            lines = self.journal_file.read_text(encoding="utf-8").splitlines()
            for line_idx, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if line_idx == len(lines) - 1:
                        logging.getLogger(__name__).warning(
                            "ignoring truncated final journal line in %s",
                            self.journal_file,
                        )
                        continue
                    raise
                self._merge_rows([row])

    def _merge_rows(self, rows: Iterable[Any]) -> None:
        """Merge compatible rows using the shared later-non-null rule."""

        for row in rows:
            if not isinstance(row, dict) or row.get("idx") is None:
                continue
            idx = int(row["idx"])
            answer = row.get("answer")
            if answer is not None or idx not in self._answers:
                self._answers[idx] = answer


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one UTF-8 JSON row, creating its parent directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def prepare_jsonl(path: Path, *, resume: bool) -> None:
    """Prepare an append-only audit file for a new or resumed run."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        path.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON through a sibling temporary file and replace atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    """Write text through a sibling temporary file and replace atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_prediction_map(path: Path) -> dict[int, str | None]:
    """Read a canonical prediction array into an ID-keyed mapping."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"prediction file must contain a JSON list: {path}")
    return {
        int(row["idx"]): row.get("answer")
        for row in payload
        if isinstance(row, dict) and row.get("idx") is not None
    }


def load_shard_config(out_dir: Path, *, prefix: str) -> dict[str, Any]:
    """Load and cross-check benchmark shard configs from an output directory."""

    files = sorted(Path(out_dir).glob(f"{prefix}.shard*.json"))
    if not files:
        raise FileNotFoundError(
            f"no {prefix}.shard*.json files under {out_dir}"
        )
    configs = [
        json.loads(path.read_text(encoding="utf-8")) for path in files
    ]
    first = configs[0]
    signature = first.get("run_signature")
    expected_idx = first.get("expected_idx")
    num_shards = int(first.get("num_shards", 0))
    for path, config in zip(files[1:], configs[1:]):
        if config.get("run_signature") != signature:
            raise ValueError(f"incompatible run signature in {path}")
        if config.get("expected_idx") != expected_idx:
            raise ValueError(f"incompatible expected_idx in {path}")
        if int(config.get("num_shards", 0)) != num_shards:
            raise ValueError(f"incompatible num_shards in {path}")
    if num_shards > 0 and len(files) != num_shards:
        raise ValueError(
            f"expected {num_shards} shard configs, found {len(files)}"
        )
    return first


def merge_prediction_shards(
    out_dir: Path,
    *,
    expected_idx: Sequence[int],
    expected_num_shards: int | None = None,
) -> MergeReport:
    """Strictly merge canonical prediction shards in ``out_dir``."""

    shard_files = discover_shard_files(Path(out_dir))
    if not shard_files:
        raise FileNotFoundError(
            f"no predictions.shard*.json files under {out_dir}"
        )
    if (
        expected_num_shards is not None
        and len(shard_files) != expected_num_shards
    ):
        raise ValueError(
            f"expected {expected_num_shards} prediction shards, "
            f"found {len(shard_files)}"
        )
    return merge_predictions(
        shard_files,
        out_file=Path(out_dir) / "predictions.json",
        expected_idx=expected_idx,
        strict=True,
    )


def safe_model_tag(tag: str) -> str:
    """Return a filesystem-safe model tag."""

    return tag.replace("/", "_").replace("\\", "_").strip() or "model"


def accuracy(correct: int, total: int) -> float:
    """Return ``correct / total`` with a zero-safe denominator."""

    return float(correct) / float(total) if total else 0.0


__all__ = [
    "DEFAULT_MODEL_PATH",
    "GenerationSettings",
    "IMAGE_EXTENSIONS",
    "MultimodalGenerator",
    "PredictionStore",
    "VIDEO_EXTENSIONS",
    "accuracy",
    "append_jsonl",
    "build_image_content",
    "build_video_content",
    "build_video_prefix_content",
    "generate_reference_answer",
    "load_video_prefix",
    "load_prediction_map",
    "load_shard_config",
    "merge_prediction_shards",
    "parse_physbench_mc_answer",
    "parse_short_answer",
    "prepare_jsonl",
    "safe_model_tag",
    "shard_filename",
    "shard_items",
    "uniform_prefix_frame_indices",
    "video_content_from_frames",
    "write_json_atomic",
    "write_text_atomic",
]
