"""Official PhysBench sharding glue tests.

These tests exercise the multi-GPU official-eval wrapper helpers without
loading a model. The inference contract remains covered by
``test_official_adapter.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
from types import ModuleType

import pytest

from evaluation.physbench.sharding import shard_filename, shard_items


def _load_official_script() -> ModuleType:
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts" / "eval" / "run_physbench_official_eval.py"
    spec = importlib.util.spec_from_file_location(
        "run_physbench_official_eval", script
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(dataset_path: Path, n: int = 9) -> None:
    dataset_path.mkdir(parents=True)
    rows = [
        {
            "idx": i,
            "question": "Which option is correct?",
            "mode": "general",
            "split": "val",
            "file_name": [],
        }
        for i in range(n)
    ]
    (dataset_path / "test.json").write_text(
        json.dumps(rows),
        encoding="utf-8",
    )


def test_official_sample_then_shard_filters_evaluator_dataset(tmp_path: Path) -> None:
    official = _load_official_script()
    dataset_path = tmp_path / "dataset"
    _write_manifest(dataset_path)

    evaluator = argparse.Namespace(
        dataset=[{"idx": i} for i in range(9)],
    )
    sampled = official._apply_sample(
        evaluator,
        dataset_path=dataset_path,
        answers_dir=tmp_path / "answers",
        split="val",
        n_samples=None,
        seed=20260424,
        stratify=False,
    )
    shard = shard_items(sampled, num_shards=3, shard_id=1)
    official._filter_evaluator_dataset(evaluator, {item.idx for item in shard})

    assert [item.idx for item in sampled] == list(range(9))
    assert [item["idx"] for item in evaluator.dataset] == [1, 4, 7]


def test_official_merge_shards_scores_with_official_scorer(tmp_path: Path) -> None:
    official = _load_official_script()
    out_dir = tmp_path / "out"
    base = out_dir / "predictions.json"
    shard0 = shard_filename(base, shard_id=0, num_shards=2)
    shard1 = shard_filename(base, shard_id=1, num_shards=2)
    shard0.parent.mkdir(parents=True)
    shard0.write_text(
        json.dumps([{"idx": 0, "answer": "A"}, {"idx": 2, "answer": "C"}]),
        encoding="utf-8",
    )
    shard1.write_text(
        json.dumps([{"idx": 1, "answer": "B"}, {"idx": 3, "answer": "D"}]),
        encoding="utf-8",
    )

    cfg = shard_filename(out_dir / "official_eval_config.json", 0, 2)
    cfg.write_text(
        json.dumps(
            {
                "model_tag": "fake-model",
                "split": "val",
                "expected_idx": [0, 1, 2, 3],
            }
        ),
        encoding="utf-8",
    )

    answers_dir = tmp_path / "answers"
    answers_dir.mkdir()
    (answers_dir / "val_answer.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "answer": "A",
                    "task_type": "property",
                    "sub_type": "mass",
                    "ability_type": "identify",
                },
                {
                    "idx": 1,
                    "answer": "B",
                    "task_type": "relationships",
                    "sub_type": "distance",
                    "ability_type": "comparison",
                },
                {
                    "idx": 2,
                    "answer": "A",
                    "task_type": "scene",
                    "sub_type": "light",
                    "ability_type": "perception",
                },
                {
                    "idx": 3,
                    "answer": "D",
                    "task_type": "dynamics",
                    "sub_type": "collision",
                    "ability_type": "prediction",
                },
            ]
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        merge_shards=out_dir,
        answers_dir=answers_dir,
        split="val",
        model_tag=None,
        model_path=Path("unused"),
        model_family=None,
    )
    assert official._merge_only(args, logging.getLogger("test")) == 0

    rows = json.loads((out_dir / "predictions.json").read_text(encoding="utf-8"))
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert [row["idx"] for row in rows] == [0, 1, 2, 3]
    assert summary["overall_correct"] == 3
    assert summary["overall_total"] == 4
    assert "fake-model" in (out_dir / "report.md").read_text(encoding="utf-8")


def _shard_config_args(*, model_path: Path, model_family: str | None) -> argparse.Namespace:
    return argparse.Namespace(
        model_path=model_path,
        model_family=model_family,
        dataset_path=Path("datasets/PhysBench"),
        answers_dir=Path("evaluation/physbench/third_party/physbench_answers"),
        split="test",
        n_samples=None,
        seed=20260424,
        no_stratify=False,
        num_shards=4,
        shard_id=0,
    )


def test_official_shard_config_records_family_only_for_non_qwen(
    tmp_path: Path,
) -> None:
    """Qwen resume metadata must stay unchanged; InternVL shards record family."""

    official = _load_official_script()
    qwen_file = tmp_path / "qwen.json"
    internvl_file = tmp_path / "internvl.json"
    official._write_shard_config(
        qwen_file,
        args=_shard_config_args(
            model_path=Path(
                "Qwen/Qwen3-VL-8B-Instruct"
            ),
            model_family="qwen",
        ),
        model_tag="Qwen3-VL-8B-Instruct",
        expected_idx=[1, 2],
    )
    official._write_shard_config(
        internvl_file,
        args=_shard_config_args(
            model_path=Path("models/InternVL2_5-8B"),
            model_family="internvl",
        ),
        model_tag="InternVL2_5-8B",
        expected_idx=[1, 2],
    )
    qwen_payload = json.loads(qwen_file.read_text(encoding="utf-8"))
    internvl_payload = json.loads(internvl_file.read_text(encoding="utf-8"))
    assert "model_family" not in qwen_payload
    assert internvl_payload["model_family"] == "internvl"
