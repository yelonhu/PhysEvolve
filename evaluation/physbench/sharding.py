"""Deterministic sharding + merge for PhysBench evaluation runs.

The sharding strategy is intentionally trivial so it can be reproduced
by hand from a shell script:

* Sort the (already deterministically sampled) items by ``idx``. This
  matches what :func:`evaluation.physbench.data_loader.sample_items`
  produces — the public output is always sorted by ``idx``.
* Slice with ``items[shard_id::num_shards]``. Every shard gets a
  disjoint subset and the union over all ``shard_id`` values is the
  original list.

We never re-sample inside a shard — sharding only slices the *fixed*
post-sample list — so 1-shard and N-shard runs evaluate the identical
set of ``idx`` values.

The merge step gathers per-shard ``predictions.shard{NN}.json`` files
into a single ``predictions.json`` in the canonical schema
``[{"idx", "answer"}, ...]``. It validates:

* every shard contributes only its assigned ``idx`` values;
* the union exactly equals the expected ``idx`` set passed in;
* duplicate ``idx`` rows are detected and resolved by keeping the
  later non-null answer (a downstream re-run can overwrite a prior
  failure without manual cleanup).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .data_loader import PhysBenchItem


# ─── Slicing ─────────────────────────────────────────────────────────


def shard_items(
    items: Sequence[PhysBenchItem],
    *,
    num_shards: int,
    shard_id: int,
) -> list[PhysBenchItem]:
    """Return the ``shard_id`` slice of ``items``.

    Args:
        items: Already-sampled, sort-by-``idx`` items.
        num_shards: Total number of shards. Must be ``>= 1``.
        shard_id: ``0 <= shard_id < num_shards``.

    Returns:
        ``items[shard_id::num_shards]`` (as a list). Disjoint across
        ``shard_id``; their union equals ``list(items)``.
    """

    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1; got {num_shards}")
    if not 0 <= shard_id < num_shards:
        raise ValueError(
            f"shard_id must be in [0, {num_shards}); got {shard_id}"
        )
    return [it for i, it in enumerate(items) if i % num_shards == shard_id]


def shard_filename(predictions_file: Path, shard_id: int, num_shards: int) -> Path:
    """Return ``predictions.shard{NN}of{NN}.json``-style path.

    The encoding ``shardNNofNN`` makes both id and total visible from
    the filename alone — useful for debugging mismatched merges (e.g.
    half the files were produced under 4 shards and half under 8).
    """

    predictions_file = Path(predictions_file)
    stem = predictions_file.stem
    suffix = predictions_file.suffix or ".json"
    pad = max(2, len(str(num_shards - 1)))
    return predictions_file.with_name(
        f"{stem}.shard{shard_id:0{pad}d}of{num_shards:0{pad}d}{suffix}"
    )


# ─── Merging ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MergeReport:
    """Diagnostic counts for :func:`merge_predictions`."""

    n_input_rows: int
    n_unique_idx: int
    n_missing: int
    n_extra: int
    missing_idx: tuple[int, ...]
    extra_idx: tuple[int, ...]
    output_file: Path


def merge_predictions(
    shard_files: Sequence[Path],
    out_file: Path,
    *,
    expected_idx: Sequence[int] | None = None,
    strict: bool = True,
) -> MergeReport:
    """Concatenate shard prediction files into one canonical JSON.

    Args:
        shard_files: Per-shard ``predictions.shard{NN}of{NN}.json``
            files (or any compatible JSON array of ``{"idx","answer"}``).
            Files are processed in order; for a given ``idx``, later
            non-null answers win over earlier ones, and a null answer
            never replaces an existing non-null one.
        out_file: Destination ``predictions.json``.
        expected_idx: If provided, the merged set is validated against
            this exact set of ``idx`` values. Missing or extra ids are
            recorded in the returned :class:`MergeReport`; when
            ``strict=True`` either condition raises.
        strict: When ``True``, any missing or extra id raises
            :class:`ValueError`. When ``False``, the merge always writes
            the file and the caller can inspect the report.

    Returns:
        A :class:`MergeReport` describing what was merged. The output
        file always contains rows sorted by ``idx``.
    """

    merged: dict[int, Any] = {}
    n_input = 0
    for shard_file in shard_files:
        path = Path(shard_file)
        if not path.exists():
            if strict:
                raise FileNotFoundError(f"shard file missing: {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(
                f"shard file {path} must contain a JSON list; got "
                f"{type(data).__name__}"
            )
        for row in data:
            if not isinstance(row, dict):
                continue
            idx = row.get("idx")
            if idx is None:
                continue
            n_input += 1
            idx = int(idx)
            answer = row.get("answer")
            if idx not in merged or (answer is not None):
                # Later non-null wins; a null never overwrites a non-null.
                if idx in merged and merged[idx] is not None and answer is None:
                    continue
                merged[idx] = answer

    expected_set = set(int(x) for x in (expected_idx or ()))
    have_set = set(merged.keys())

    missing = tuple(sorted(expected_set - have_set)) if expected_set else ()
    extra = tuple(sorted(have_set - expected_set)) if expected_set else ()

    if strict and expected_set:
        if missing:
            raise ValueError(
                f"merge missing {len(missing)} idx values: "
                f"{list(missing[:8])}{'...' if len(missing) > 8 else ''}"
            )
        if extra:
            raise ValueError(
                f"merge contains {len(extra)} unexpected idx values: "
                f"{list(extra[:8])}{'...' if len(extra) > 8 else ''}"
            )

    rows = [{"idx": idx, "answer": merged[idx]} for idx in sorted(merged)]
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return MergeReport(
        n_input_rows=n_input,
        n_unique_idx=len(merged),
        n_missing=len(missing),
        n_extra=len(extra),
        missing_idx=missing,
        extra_idx=extra,
        output_file=out_file,
    )


def discover_shard_files(
    out_dir: Path,
    *,
    base_name: str = "predictions",
) -> list[Path]:
    """Find ``predictions.shard*.json`` files inside ``out_dir``.

    Returns the files sorted by their shard-id substring so the merge
    order is deterministic regardless of OS ``readdir`` order.
    """

    out_dir = Path(out_dir)
    files = sorted(out_dir.glob(f"{base_name}.shard*.json"))
    return files


__all__ = [
    "MergeReport",
    "discover_shard_files",
    "merge_predictions",
    "shard_filename",
    "shard_items",
]
