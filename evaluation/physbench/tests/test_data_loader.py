"""Tests for ``evaluation.physbench.data_loader``.

These tests are model- and dataset-free — they use a tiny in-memory
``test.json`` fixture so they can run on any machine in under a second.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.physbench.data_loader import (
    PhysBenchItem,
    load_items,
    resolve_visual_paths,
    sample_items,
)


def _write_manifest(tmp_path: Path, rows: list[dict]) -> Path:
    """Create a tiny ``test.json`` under ``tmp_path`` and return its parent."""

    (tmp_path / "test.json").write_text(
        json.dumps(rows), encoding="utf-8",
    )
    return tmp_path


def _row(idx: int, split: str = "val", mode: str = "general") -> dict:
    return {
        "scene": "",
        "object": ["box"],
        "source": "web",
        "file_name": [f"a{idx}.mp4", f"b{idx}.png"],
        "description": None,
        "question": "Watch <video> then <image>?",
        "mode": mode,
        "idx": idx,
        "split": split,
    }


def test_load_items_filters_by_split(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        [_row(0, "val"), _row(1, "val"), _row(2, "test"), _row(3, "val")],
    )
    val_items = load_items(manifest, split="val")
    test_items = load_items(manifest, split="test")

    assert [it.idx for it in val_items] == [0, 1, 3]
    assert [it.idx for it in test_items] == [2]
    assert all(isinstance(it, PhysBenchItem) for it in val_items)


def test_load_items_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_items(tmp_path, split="val")


def test_sample_items_deterministic_without_stratify() -> None:
    items = [PhysBenchItem(
        idx=i, question="q", mode="general", split="val",
        file_names=("x.png",),
    ) for i in range(20)]

    a = sample_items(items, n=5, seed=42)
    b = sample_items(items, n=5, seed=42)
    c = sample_items(items, n=5, seed=7)

    assert [x.idx for x in a] == [x.idx for x in b]
    assert [x.idx for x in a] != [x.idx for x in c]
    # Sorted by idx so resume semantics are stable.
    assert [x.idx for x in a] == sorted(x.idx for x in a)


def test_sample_items_stratified_coverage() -> None:
    items = [PhysBenchItem(
        idx=i, question="q", mode="general", split="val",
        file_names=("x.png",),
    ) for i in range(40)]

    # Four 10-item buckets.
    stratify = {i: ["property", "relationships", "scene", "dynamics"][i // 10]
                for i in range(40)}

    chosen = sample_items(items, n=8, seed=1, stratify_by=stratify)
    # 8 items across 4 buckets → roughly balanced; each bucket contributes
    # at least one pick thanks to the ``max(1, ...)`` allocator.
    by_type: dict[str, int] = {}
    for it in chosen:
        by_type[stratify[it.idx]] = by_type.get(stratify[it.idx], 0) + 1
    assert sum(by_type.values()) == 8
    assert set(by_type.keys()) == {"property", "relationships", "scene", "dynamics"}


def test_sample_items_returns_all_when_n_exceeds() -> None:
    items = [PhysBenchItem(
        idx=i, question="q", mode="general", split="val",
        file_names=("x.png",),
    ) for i in range(5)]
    chosen = sample_items(items, n=99, seed=0)
    assert {x.idx for x in chosen} == set(range(5))


def test_sample_items_none_returns_full_split_sorted() -> None:
    items = [PhysBenchItem(
        idx=i, question="q", mode="general", split="val",
        file_names=("x.png",),
    ) for i in [3, 1, 2]]
    chosen = sample_items(items, n=None, seed=0)
    assert [x.idx for x in chosen] == [1, 2, 3]


def test_sample_items_rejects_non_positive_n() -> None:
    items = [PhysBenchItem(
        idx=0, question="q", mode="general", split="val",
        file_names=("x.png",),
    )]
    with pytest.raises(ValueError):
        sample_items(items, n=0)


def test_resolve_visual_paths_dispatches_by_extension(tmp_path: Path) -> None:
    item = PhysBenchItem(
        idx=0, question="", mode="general", split="val",
        file_names=("v.mp4", "i.png", "j.jpg"),
    )
    out = resolve_visual_paths(item, tmp_path)
    assert out[0].parent.name == "video"
    assert out[1].parent.name == "image"
    assert out[2].parent.name == "image"
    assert out[0].name == "v.mp4"


def test_resolve_visual_paths_rejects_unknown_extension(tmp_path: Path) -> None:
    item = PhysBenchItem(
        idx=0, question="", mode="general", split="val",
        file_names=("v.webm",),
    )
    with pytest.raises(ValueError):
        resolve_visual_paths(item, tmp_path)
