"""Sharding + merge tests for the PhysBench batched runner.

All tests are dependency-light and run in a fraction of a second; they
exercise the deterministic-shard contract that the multi-GPU launcher
relies on for correctness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.physbench.data_loader import PhysBenchItem
from evaluation.physbench.sharding import (
    MergeReport,
    discover_shard_files,
    merge_predictions,
    shard_filename,
    shard_items,
)


def _items(n: int) -> list[PhysBenchItem]:
    return [
        PhysBenchItem(
            idx=i, question="q", mode="general", split="val",
            file_names=(f"x{i}.png",),
        )
        for i in range(n)
    ]


def test_shard_items_disjoint_and_covering() -> None:
    items = _items(20)
    parts = [shard_items(items, num_shards=4, shard_id=i) for i in range(4)]
    # Every shard pick is unique.
    ids = [it.idx for part in parts for it in part]
    assert sorted(ids) == sorted(it.idx for it in items)
    assert len(ids) == len(set(ids))
    # The slicing pattern is strided so reruns are stable.
    assert [it.idx for it in parts[0]] == [0, 4, 8, 12, 16]
    assert [it.idx for it in parts[3]] == [3, 7, 11, 15, 19]


def test_shard_items_single_shard_returns_all() -> None:
    items = _items(7)
    out = shard_items(items, num_shards=1, shard_id=0)
    assert [it.idx for it in out] == list(range(7))


def test_shard_items_rejects_invalid_arguments() -> None:
    items = _items(2)
    with pytest.raises(ValueError):
        shard_items(items, num_shards=0, shard_id=0)
    with pytest.raises(ValueError):
        shard_items(items, num_shards=2, shard_id=2)


def test_shard_filename_pads_consistently() -> None:
    p = Path("/tmp/out/predictions.json")
    # The encoding zero-pads to at least two digits so 8-way and
    # 10-way runs line up visually when listed side-by-side.
    assert shard_filename(p, shard_id=0, num_shards=4).name == \
        "predictions.shard00of04.json"
    assert shard_filename(p, shard_id=5, num_shards=10).name == \
        "predictions.shard05of10.json"
    assert shard_filename(p, shard_id=12, num_shards=100).name == \
        "predictions.shard12of100.json"


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_merge_predictions_concatenates_and_sorts(tmp_path: Path) -> None:
    _write(tmp_path / "p.shard0of2.json", [{"idx": 4, "answer": "B"},
                                            {"idx": 0, "answer": "A"}])
    _write(tmp_path / "p.shard1of2.json", [{"idx": 1, "answer": "C"},
                                            {"idx": 3, "answer": "D"}])
    out = tmp_path / "predictions.json"
    report = merge_predictions(
        [tmp_path / "p.shard0of2.json", tmp_path / "p.shard1of2.json"],
        out_file=out,
        expected_idx=[0, 1, 3, 4],
    )
    assert isinstance(report, MergeReport)
    assert report.n_unique_idx == 4
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert [r["idx"] for r in rows] == [0, 1, 3, 4]
    assert [r["answer"] for r in rows] == ["A", "C", "D", "B"]


def test_merge_predictions_later_nonnull_wins(tmp_path: Path) -> None:
    _write(tmp_path / "a.json", [{"idx": 1, "answer": None}])
    _write(tmp_path / "b.json", [{"idx": 1, "answer": "B"}])
    out = tmp_path / "predictions.json"
    merge_predictions(
        [tmp_path / "a.json", tmp_path / "b.json"],
        out_file=out,
        expected_idx=[1],
    )
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert rows == [{"idx": 1, "answer": "B"}]


def test_merge_predictions_strict_rejects_missing(tmp_path: Path) -> None:
    _write(tmp_path / "s.json", [{"idx": 1, "answer": "A"}])
    with pytest.raises(ValueError):
        merge_predictions(
            [tmp_path / "s.json"],
            out_file=tmp_path / "predictions.json",
            expected_idx=[1, 2, 3],
            strict=True,
        )


def test_merge_predictions_non_strict_reports_diffs(tmp_path: Path) -> None:
    _write(tmp_path / "s.json", [{"idx": 1, "answer": "A"},
                                  {"idx": 7, "answer": "B"}])
    report = merge_predictions(
        [tmp_path / "s.json"],
        out_file=tmp_path / "predictions.json",
        expected_idx=[1, 2],
        strict=False,
    )
    assert report.missing_idx == (2,)
    assert report.extra_idx == (7,)


def test_discover_shard_files_sorts(tmp_path: Path) -> None:
    for name in ["predictions.shard02of04.json",
                 "predictions.shard00of04.json",
                 "predictions.shard01of04.json"]:
        _write(tmp_path / name, [])
    files = discover_shard_files(tmp_path)
    assert [f.name for f in files] == [
        "predictions.shard00of04.json",
        "predictions.shard01of04.json",
        "predictions.shard02of04.json",
    ]
