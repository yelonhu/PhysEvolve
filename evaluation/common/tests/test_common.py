"""CPU-only tests for shared reference VLM evaluation mechanics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation.common import common


def test_build_content_reuses_visual_loaders(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Image/video content should preserve visual-first prompt ordering."""

    monkeypatch.setattr(
        common,
        "load_image",
        lambda path: np.zeros((2, 3, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        common,
        "load_video",
        lambda path, n_frames: np.zeros(
            (n_frames, 2, 3, 3),
            dtype=np.uint8,
        ),
    )
    monkeypatch.setattr(
        common,
        "load_video_prefix",
        lambda path, n_frames, end_frame_exclusive: np.zeros(
            (n_frames, 2, 3, 3),
            dtype=np.uint8,
        ),
    )

    images = common.build_image_content(
        [tmp_path / "a.png", tmp_path / "b.jpg"],
        "question",
    )
    video = common.build_video_content(
        tmp_path / "clip.webm",
        "question",
        n_frames=3,
    )
    prefix_video = common.build_video_prefix_content(
        tmp_path / "clip.mp4",
        "question",
        n_frames=4,
        end_frame_exclusive=11,
    )

    assert [entry["type"] for entry in images] == ["image", "image", "text"]
    assert [entry["type"] for entry in video] == ["video", "text"]
    assert video[0]["video"].shape == (3, 2, 3, 3)
    assert prefix_video[0]["video"].shape == (4, 2, 3, 3)


def test_uniform_prefix_indices_never_cross_cutoff() -> None:
    """Bucket-start prefix sampling must not expose future answer frames."""

    indices = common.uniform_prefix_frame_indices(310, 16)

    assert len(indices) == 16
    assert indices == sorted(indices)
    assert indices[0] == 0
    assert max(indices) < 310
    assert common.uniform_prefix_frame_indices(3, 5) == [0, 1, 2]


def test_parse_short_answer_contract() -> None:
    """The parser should prefer tags and avoid text before a think close."""

    assert common.parse_short_answer("<answer> no </answer>") == "no"
    assert common.parse_short_answer("<result>three</result>") == "three"
    assert common.parse_short_answer("<think>red</think><answer>blue") == "blue"
    assert common.parse_short_answer("<think>A</think>\nmetal<|im_end|>") == "metal"


def test_prediction_store_resume_and_finalize(tmp_path: Path) -> None:
    """Append journals should resume non-null answers and materialize JSON."""

    path = tmp_path / "predictions.json"
    store = common.PredictionStore(path, resume=False)
    store.append(2, None)
    store.append(1, "A")
    store.append(2, "B")
    store.finalize()

    resumed = common.PredictionStore(path, resume=True)
    assert resumed.completed_ids == {1, 2}
    assert resumed.answer_for(1) == "A"
    assert resumed.answer_for(99) is None
    assert common.load_prediction_map(path) == {1: "A", 2: "B"}


def test_strict_shard_config_and_merge(tmp_path: Path) -> None:
    """Reference evaluators should retain PhysBench's strict merge contract."""

    signature = {"benchmark": "fixture", "dataset_dir": "/tmp/data"}
    for shard_id, rows in enumerate(
        (
            [{"idx": 0, "answer": "A"}],
            [{"idx": 1, "answer": "B"}],
        )
    ):
        prediction_path = common.shard_filename(
            tmp_path / "predictions.json",
            shard_id,
            2,
        )
        prediction_path.write_text(json.dumps(rows), encoding="utf-8")
        config_path = common.shard_filename(
            tmp_path / "fixture_config.json",
            shard_id,
            2,
        )
        config_path.write_text(
            json.dumps(
                {
                    "run_signature": signature,
                    "num_shards": 2,
                    "shard_id": shard_id,
                    "expected_idx": [0, 1],
                }
            ),
            encoding="utf-8",
        )

    config = common.load_shard_config(tmp_path, prefix="fixture_config")
    report = common.merge_prediction_shards(
        tmp_path,
        expected_idx=config["expected_idx"],
        expected_num_shards=2,
    )

    assert report.n_unique_idx == 2
    assert common.load_prediction_map(tmp_path / "predictions.json") == {
        0: "A",
        1: "B",
    }
