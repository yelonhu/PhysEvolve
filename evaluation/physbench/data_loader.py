"""PhysBench ``test.json`` loading + deterministic selection.

PhysBench distributes a single ``test.json`` that mixes the 200-item
``val`` split and the 10,002-item ``test`` split, distinguished by the
``split`` field. Each row looks like::

    {
      "scene": str | null,
      "object": list[str] | null,
      "source": "real" | "simulation" | "web",
      "file_name": list[str],     # ordered visual inputs
      "description": str | list | null,
      "question": str,            # contains <video> and <image> tags
      "mode": "image-only" | "image&video" | "general",
      "idx": int,
      "split": "val" | "test"
    }

Note that ``answer`` / ``task_type`` / ``sub_type`` / ``ability_type`` are
**not** in ``test.json`` itself — they live in the separate answer files
(``val_answer.json`` / ``test_answer.json``) keyed by ``idx``. Our loader
therefore produces items without those fields and the scorer joins them
at report time.

The default evaluation path uses the full requested split. Sampling is
available only when callers explicitly pass a target ``n``; in that mode
it is seeded and (by default) stratified on the task-type when answer
metadata is available.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


# ─── Schema ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PhysBenchItem:
    """A single PhysBench question with resolved visual-file paths.

    Attributes:
        idx: The benchmark-wide integer identifier. Stable across runs.
        question: Raw question string; contains ``<video>`` and
            ``<image>`` tags that map positionally to ``file_names``.
        mode: The upstream ``mode`` field (``image-only`` / ``image&video``
            / ``general``). Informational — not used for dispatch here.
        split: ``val`` or ``test``.
        file_names: Ordered visual inputs. The tag order inside
            ``question`` matches the order of these filenames.
        source: Upstream ``source`` tag. Retained for debugging only.
        scene: Optional free-text scene descriptor from upstream.
    """

    idx: int
    question: str
    mode: str
    split: str
    file_names: tuple[str, ...]
    source: Optional[str] = None
    scene: Optional[str] = None


# ─── Loading ─────────────────────────────────────────────────────────


def load_items(
    dataset_path: Path,
    split: Literal["val", "test"] = "val",
) -> list[PhysBenchItem]:
    """Read ``{dataset_path}/test.json`` and return items for ``split``.

    Args:
        dataset_path: Directory containing ``test.json``, ``image/`` and
            ``video/`` subfolders (matches the HuggingFace dataset layout
            produced by the official ``huggingface-cli download`` step).
        split: Which split to filter on.

    Returns:
        A list of :class:`PhysBenchItem` in their original on-disk order
        (also equal to ascending ``idx`` order — we do not resort).
    """

    manifest = Path(dataset_path) / "test.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"PhysBench manifest not found: {manifest}. "
            "Point --dataset-path at the directory containing test.json."
        )

    with manifest.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    items: list[PhysBenchItem] = []
    for row in raw:
        if row.get("split") != split:
            continue
        items.append(
            PhysBenchItem(
                idx=int(row["idx"]),
                question=str(row["question"]),
                mode=str(row.get("mode", "general")),
                split=str(row["split"]),
                file_names=tuple(row.get("file_name", []) or []),
                source=row.get("source"),
                scene=row.get("scene"),
            )
        )
    return items


# ─── Sampling ────────────────────────────────────────────────────────


def sample_items(
    items: list[PhysBenchItem],
    n: Optional[int],
    *,
    seed: int = 20260424,
    stratify_by: Optional[dict[int, str]] = None,
) -> list[PhysBenchItem]:
    """Return the full split or deterministically sample ``n`` items.

    Args:
        items: Candidate pool, typically the full ``load_items`` output.
        n: Target sample size. ``None`` means "use the full split". If
            ``n >= len(items)``, all items are returned.
        seed: RNG seed. Changing it changes the sample.
        stratify_by: Optional mapping ``{idx: task_type}`` used to bucket
            items and sample proportionally from each bucket. Typically
            built by joining the answer-file against the loaded items.
            When ``None``, falls back to uniform sampling.

    Returns:
        Sampled list in ascending-idx order so downstream log / resume
        semantics are stable.
    """

    if n is None:
        return sorted(items, key=lambda it: it.idx)
    if n <= 0:
        raise ValueError(f"n must be positive or None; got {n}")

    rng = random.Random(seed)
    if n >= len(items):
        return sorted(items, key=lambda it: it.idx)

    if not stratify_by:
        chosen = rng.sample(items, n)
        return sorted(chosen, key=lambda it: it.idx)

    # Bucket by task_type; any item with no mapping ends up in "__other__".
    buckets: dict[str, list[PhysBenchItem]] = defaultdict(list)
    for it in items:
        buckets[stratify_by.get(it.idx, "__other__")].append(it)

    # Proportional allocation with at-least-one-per-bucket (where the
    # bucket is non-empty). Rounding leftover goes to the largest bucket.
    total = len(items)
    alloc: dict[str, int] = {}
    for tt, bucket in buckets.items():
        share = max(1, round(n * len(bucket) / total))
        alloc[tt] = min(share, len(bucket))

    # Correct for over- or under-shoot.
    while sum(alloc.values()) > n:
        largest = max(alloc, key=lambda k: alloc[k])
        alloc[largest] -= 1
    while sum(alloc.values()) < n:
        # Add to the bucket with the most remaining headroom.
        headroom = {k: len(buckets[k]) - alloc[k] for k in alloc}
        target = max(headroom, key=lambda k: headroom[k])
        if headroom[target] <= 0:
            break
        alloc[target] += 1

    chosen: list[PhysBenchItem] = []
    for tt, bucket in buckets.items():
        take = alloc.get(tt, 0)
        if take <= 0:
            continue
        chosen.extend(rng.sample(bucket, take))
    return sorted(chosen, key=lambda it: it.idx)


# ─── File-path resolution ────────────────────────────────────────────


def resolve_visual_paths(
    item: PhysBenchItem, dataset_path: Path
) -> list[Path]:
    """Resolve each entry in ``item.file_names`` to an absolute path.

    Images live under ``{dataset_path}/image/``, videos under
    ``{dataset_path}/video/``. Extension-based dispatch mirrors
    ``task_evaluator.PhysionBenchEvaluator._process_visual_path``.
    """

    dataset_path = Path(dataset_path)
    out: list[Path] = []
    for name in item.file_names:
        lower = name.lower()
        if lower.endswith(".mp4"):
            out.append(dataset_path / "video" / name)
        elif lower.endswith((".jpg", ".jpeg", ".png")):
            out.append(dataset_path / "image" / name)
        else:
            raise ValueError(
                f"unknown visual-file extension for item idx={item.idx}: {name!r}"
            )
    return out


__all__ = [
    "PhysBenchItem",
    "load_items",
    "resolve_visual_paths",
    "sample_items",
]
