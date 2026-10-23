"""Shared constants and helpers for Stage-2 golden scene export/render."""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_SCENES_ROOT = _REPO_ROOT / "data" / "golden_scenes"
STAGE2_DIR = GOLDEN_SCENES_ROOT / "stage2"
MANIFEST_PATH = GOLDEN_SCENES_ROOT / "manifest.json"

# Fixed export order (matches data/golden_scenes/README.md). The three active
# grouped scene_kinds come first; frozen scene_kinds (excluded from active training) are
# the tail.
ORDERED_SCENE_KINDS: tuple[str, ...] = (
    "throwing_pit_board",
    "collision_arena",
    "motion_lab",
    # Frozen (not in active training):
    "motion_phenomenon_classification",
    "camera_depth_closest_object",
)

_UPSTREAM_BY_SCENE_KIND: dict[str, str] = {
    "throwing_pit_board": "physevolve/skills/scene_templates/throwing_pit_board.py",
    "collision_arena": "physevolve/skills/scene_templates/collision_arena.py",
    "motion_lab": "physevolve/skills/scene_templates/motion_lab.py",
    "motion_phenomenon_classification": "physevolve/skills/scene_templates/motion_phenomenon.py",
    "camera_depth_closest_object": "physevolve/skills/scene_templates/camera_depth.py",
}
_DEFAULT_UPSTREAM = "physevolve/skills/scene_templates/__init__.py"

_PROPOSE_OPEN_RE = re.compile(r"<propose>", re.IGNORECASE)
_MC_LETTERS = "ABCD"


def upstream_source_path(scene_kind: str) -> str:
    """Return the repo-relative path of the template module owning ``scene_kind``."""

    return _UPSTREAM_BY_SCENE_KIND.get(scene_kind, _DEFAULT_UPSTREAM)


def stage2_filename(order: int, scene_kind: str) -> str:
    """Return ``{NN}_{scene_kind}.py`` for ``order`` in 1..99."""

    return f"{order:02d}_{scene_kind}.py"


def extract_plan_line(fewshot: str) -> Optional[str]:
    """Return the Plan preamble before ``<propose>``, if any."""

    match = _PROPOSE_OPEN_RE.search(fewshot)
    if match is None:
        return None
    preamble = fewshot[: match.start()].strip()
    if not preamble:
        return None
    if preamble.lower().startswith("plan:"):
        return preamble[5:].strip()
    return preamble


def build_golden_scene_source(
    *,
    scene_kind: str,
    families: tuple[str, ...],
    fewshot: str,
    propose_code: str,
) -> str:
    """Format a standalone runnable golden scene ``.py`` file."""

    upstream = upstream_source_path(scene_kind)
    plan = extract_plan_line(fewshot)
    families_line = ", ".join(families)
    lines = [
        f'"""Golden scene: {scene_kind}',
        "",
        f"Source: {upstream}",
        f"Families: {families_line}",
    ]
    if plan:
        lines.append(f"Plan: {plan}")
    lines.extend(['"""', ""])
    lines.append(propose_code.rstrip())
    lines.append("")
    return "\n".join(lines)


def manifest_entry(
    *,
    order: int,
    scene_kind: str,
    families: tuple[str, ...],
    relative_path: str,
) -> dict[str, Any]:
    """Build one manifest.json scene record."""

    return {
        "order": order,
        "scene_kind": scene_kind,
        "families": list(families),
        "upstream_source": upstream_source_path(scene_kind),
        "relative_path": relative_path,
        "filename": stage2_filename(order, scene_kind),
    }


def make_run_dir(
    out_dir: Path,
    stem: str,
    seed: int,
    *,
    stamp: Optional[str] = None,
) -> Path:
    """Create and return ``{out_dir}/{stem}_seed{seed}_{stamp}/``."""

    if stamp is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir.resolve() / f"{stem}_seed{seed}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def vqa_item_to_record(item: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Convert one executor ``vqa_payloads`` item to a JSON-serializable record."""

    spec = dict(item.get("spec") or {})
    question, options, correct_idx = item["payload"]
    ci = int(correct_idx)
    letter = _MC_LETTERS[ci] if 0 <= ci < len(_MC_LETTERS) else "?"
    return {
        "index": int(index),
        "family": str(spec.get("family", "")),
        "phrasing": str(spec.get("phrasing", "")),
        "question": str(question),
        "options": [str(o) for o in options],
        "correct_idx": ci,
        "correct_letter": letter,
        "spec": spec,
    }


def branches_document(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Build ``branches.json`` content from an executor manifest trace."""

    return {
        "scene_kind": trace.get("scene_kind"),
        "baseline_outcomes": trace.get("baseline_outcomes") or {},
        "branches": trace.get("branches") or [],
        "body_colors": trace.get("body_colors") or {},
    }


def write_run_artifacts(
    *,
    run_dir: Path,
    result: Any,
    trace: Mapping[str, Any],
    scene_path: Path,
    seed: int,
    status: str,
    write_video: bool = True,
    camera_idx: int = 0,
    save_video_fn: Optional[Callable[..., None]] = None,
) -> Path:
    """Write summary, branches, baseline MP4, and per-family VQA JSON (+ video copies).

    Args:
        run_dir: Target directory (already created).
        result: :class:`~physevolve.executor.result.ExecutorResult` with
            ``scene_id``, ``vqa_payloads``, and optional ``sim_state``.
        trace: Executor manifest trace dict (may be empty).
        scene_path: Source golden scene ``.py`` path (for summary metadata).
        seed: Simulation seed used for the run.
        status: Executor status string (e.g. ``ok``).
        write_video: If False, skip ``baseline.mp4`` and family ``video.mp4`` files.
        camera_idx: Camera index passed to ``save_video_mp4``.
        save_video_fn: Injectable video writer (defaults to
            :func:`physevolve.training.selfplay_batch.save_video_mp4`).

    Returns:
        ``run_dir`` (for convenience).
    """

    if save_video_fn is None:
        from physevolve.training.selfplay_batch import save_video_mp4

        save_video_fn = save_video_mp4

    payloads: Sequence[Mapping[str, Any]] = getattr(result, "vqa_payloads", None) or []
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in payloads:
        spec = item.get("spec") or {}
        family = str(spec.get("family") or "unknown_family")
        by_family[family].append(item)

    families_sorted = sorted(by_family)
    summary = {
        "scene_id": getattr(result, "scene_id", ""),
        "status": status,
        "seed": int(seed),
        "scene_path": str(scene_path.resolve()),
        "scene_stem": scene_path.stem,
        "vqa_count": len(payloads),
        "families": families_sorted,
        "walltime_s": float(getattr(result, "walltime_s", 0.0) or 0.0),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    branches_path = run_dir / "branches.json"
    branches_path.write_text(
        json.dumps(branches_document(trace), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    baseline_mp4 = run_dir / "baseline.mp4"
    baseline_frames = None
    sim_state = getattr(result, "sim_state", None)
    if sim_state is not None:
        baseline_frames = getattr(sim_state, "video_frames", None)

    if write_video:
        if baseline_frames is None or getattr(baseline_frames, "size", 0) == 0:
            raise ValueError("write_video=True but result.sim_state.video_frames is missing")
        save_video_fn(baseline_frames, baseline_mp4, camera_idx=int(camera_idx))

    for family in families_sorted:
        family_dir = run_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        items = by_family[family]
        for fam_idx, item in enumerate(items):
            record = vqa_item_to_record(item, fam_idx)
            q_path = family_dir / f"q{fam_idx:02d}.json"
            q_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            variant_state = item.get("sim_state")
            if write_video and variant_state is not None:
                variant_frames = getattr(variant_state, "video_frames", None)
                if variant_frames is not None and getattr(variant_frames, "size", 0) > 0:
                    save_video_fn(
                        variant_frames,
                        family_dir / f"q{fam_idx:02d}.mp4",
                        camera_idx=int(camera_idx),
                    )
        if write_video and baseline_mp4.is_file():
            shutil.copy2(baseline_mp4, family_dir / "video.mp4")

    return run_dir
