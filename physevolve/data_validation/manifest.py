"""JSONL manifests, summaries, and candidate artifact writing."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from physevolve.data_validation.schemas import ValidationRecord
from physevolve.engine.schema import SimState


def record_to_dict(record: ValidationRecord) -> dict[str, Any]:
    """Convert a validation record into a JSON-safe dictionary."""

    return _json_safe(asdict(record))


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    """Append one JSON object as a line to ``path``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(item), sort_keys=True) + "\n")


def append_validation_record(path: Path, record: ValidationRecord) -> None:
    """Append one :class:`ValidationRecord` to a validation JSONL file."""

    append_jsonl(path, record_to_dict(record))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file. Missing files return an empty list."""

    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def summarize_records(records: Iterable[ValidationRecord]) -> dict[str, Any]:
    """Aggregate smoke/candidate validation records into summary metrics."""

    rows = list(records)
    total = len(rows)
    format_ok = sum(1 for row in rows if row.format.ok)
    executor_ok = sum(1 for row in rows if row.executor.ok)
    trivial = sum(1 for row in rows if row.executor.status == "TRIVIAL")
    accepted = sum(1 for row in rows if row.accepted_as_candidate)

    failure_distribution: dict[str, int] = {}
    scene_ids: set[str] = set()
    walltimes: list[float] = []
    for row in rows:
        failure_distribution[row.executor.status] = (
            failure_distribution.get(row.executor.status, 0) + 1
        )
        if row.executor.scene_id:
            scene_ids.add(row.executor.scene_id)
        walltimes.append(float(row.executor.walltime_s))

    return {
        "total": total,
        "accepted": accepted,
        "format_ok_rate": _rate(format_ok, total),
        "executor_ok_rate": _rate(executor_ok, total),
        "trivial_rate": _rate(trivial, total),
        "failure_reason_distribution": dict(sorted(failure_distribution.items())),
        "unique_scene_ratio": _rate(len(scene_ids), total),
        "median_walltime_s": _percentile(walltimes, 50.0),
        "p95_walltime_s": _percentile(walltimes, 95.0),
    }


def write_summary(path: Path, records: Iterable[ValidationRecord]) -> None:
    """Write a summary JSON file atomically."""

    _atomic_write_text(
        Path(path),
        json.dumps(summarize_records(records), indent=2, sort_keys=True) + "\n",
    )


def candidate_manifest_path(candidate_root: Path) -> Path:
    """Return the manifest path for a candidate pool root."""

    return Path(candidate_root) / "manifest.jsonl"


def load_candidate_scene_ids(candidate_root: Path) -> set[str]:
    """Read accepted scene IDs from an existing candidate manifest."""

    scene_ids: set[str] = set()
    for row in load_jsonl(candidate_manifest_path(candidate_root)):
        scene_id = row.get("scene_id")
        if isinstance(scene_id, str) and scene_id:
            scene_ids.add(scene_id)
    return scene_ids


def write_candidate_artifacts(
    candidate_root: Path,
    *,
    code: str,
    sim_state: SimState,
    save_video: bool = False,
    save_latent: bool = False,
    save_targets: bool = False,
) -> dict[str, str]:
    """Persist accepted candidate artifacts and return relative paths."""

    root = Path(candidate_root)
    scene_id = sim_state.scene_id
    paths: dict[str, str] = {}

    code_rel = Path("code") / f"{scene_id}.py"
    _atomic_write_text(root / code_rel, code.rstrip() + "\n")
    paths["physdsl_code_path"] = code_rel.as_posix()

    if save_video:
        video_rel = Path("videos") / f"{scene_id}.npz"
        _atomic_savez(
            root / video_rel,
            video_frames=sim_state.video_frames,
            camera_names=np.array(sim_state.camera_names, dtype="U64"),
        )
        paths["video_path"] = video_rel.as_posix()

    if save_latent:
        raise ValueError(
            "save_latent was removed with physevolve.probing; "
            "leave save_latent=False"
        )

    if save_targets:
        paths.update(_write_task_targets(root, sim_state))

    return paths


def append_candidate_manifest(
    candidate_root: Path,
    record: ValidationRecord,
) -> None:
    """Append one accepted candidate line to ``manifest.jsonl``."""

    if record.quality is None:
        raise ValueError("candidate records must include a quality report")
    if record.executor.scene_id is None:
        raise ValueError("candidate records must include a scene_id")

    artifact_paths = dict(record.artifact_paths)
    entry = {
        "candidate_id": record.sample_id,
        "scene_id": record.executor.scene_id,
        "source": record.source,
        "seed": record.seed,
        "quality_score": record.quality.score,
        "quality_tags": list(record.quality.tags),
        "metrics": record.quality.metrics,
        "task_suitability": record.quality.task_suitability,
        **artifact_paths,
    }
    append_jsonl(candidate_manifest_path(candidate_root), entry)


def _write_task_targets(root: Path, sim_state: SimState) -> dict[str, str]:
    """Write per-mode task targets using the existing task router."""

    from physevolve.tasks.router import TaskMode, assemble_observation

    out: dict[str, str] = {}
    for mode in (TaskMode.FORWARD, TaskMode.INVERSE, TaskMode.PARAMETER):
        observation = assemble_observation(sim_state, mode)
        rel = Path("targets") / f"{sim_state.scene_id}_{mode.value}.json"
        _atomic_write_text(
            root / rel,
            json.dumps(observation.ground_truth, indent=2, sort_keys=True) + "\n",
        )
        out[f"{mode.value}_target_path"] = rel.as_posix()
    return out


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        finally:
            raise


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent)
    )
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        finally:
            raise


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _percentile(values: list[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "append_candidate_manifest",
    "append_jsonl",
    "append_validation_record",
    "candidate_manifest_path",
    "load_candidate_scene_ids",
    "load_jsonl",
    "record_to_dict",
    "summarize_records",
    "write_candidate_artifacts",
    "write_summary",
]
