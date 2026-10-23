"""Unit tests for golden-scene render artifact helpers (no MuJoCo)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SMOKE_DIR = _REPO_ROOT / "scripts" / "smoke"
for _entry in (str(_REPO_ROOT), str(_SMOKE_DIR)):
    while _entry in sys.path:
        sys.path.remove(_entry)
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SMOKE_DIR))

from golden_scenes_common import (  # noqa: E402
    branches_document,
    make_run_dir,
    vqa_item_to_record,
    write_run_artifacts,
)


def test_vqa_item_to_record_reads_spec_family() -> None:
    """Family and phrasing come from item['spec'], not top-level keys."""

    item = {
        "payload": ("Which pit?", ["A", "B", "C", "D"], 2),
        "spec": {
            "family": "throwing_final_destination",
            "phrasing": "forward",
            "scene_kind": "throwing_pit_board",
        },
    }
    rec = vqa_item_to_record(item, 0)
    assert rec["family"] == "throwing_final_destination"
    assert rec["phrasing"] == "forward"
    assert rec["correct_letter"] == "C"
    assert rec["index"] == 0


def test_branches_document_shape() -> None:
    """branches.json content mirrors executor trace fields."""

    trace = {
        "scene_kind": "throwing_pit_board",
        "baseline_outcomes": {"ball_a": {"x": 1.0}},
        "branches": [{"key": "cf:ball_a:wall", "ran": True}],
        "body_colors": {"ball_a": "green"},
    }
    doc = branches_document(trace)
    assert doc["scene_kind"] == "throwing_pit_board"
    assert doc["baseline_outcomes"]["ball_a"]["x"] == 1.0
    assert len(doc["branches"]) == 1


def test_write_run_artifacts_layout(tmp_path: Path) -> None:
    """Per-family folders receive qNN.json and a copied video.mp4."""

    run_dir = make_run_dir(tmp_path, "01_test_scene", 7, stamp="fixed_stamp")
    assert run_dir.name == "01_test_scene_seed7_fixed_stamp"

    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    written: list[Path] = []

    def _fake_save(frames_in: np.ndarray, path: Path, camera_idx: int = 0) -> None:
        del frames_in, camera_idx
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"FAKEMP4")
        written.append(path)

    result = SimpleNamespace(
        scene_id="scene_test",
        walltime_s=1.23,
        sim_state=SimpleNamespace(video_frames=frames),
        vqa_payloads=[
            {
                "payload": ("Q dest A?", ["L", "R", "M", "U"], 0),
                "spec": {
                    "family": "throwing_final_destination",
                    "phrasing": "forward",
                },
            },
            {
                "payload": ("Q dest B?", ["L", "R", "M", "U"], 1),
                "spec": {
                    "family": "throwing_final_destination",
                    "phrasing": "forward",
                },
            },
            {
                "payload": ("Q cf?", ["L", "R", "M", "None"], 3),
                "spec": {
                    "family": "counterfactual_destination",
                    "phrasing": "counterfactual",
                },
            },
        ],
    )

    write_run_artifacts(
        run_dir=run_dir,
        result=result,
        trace={
            "scene_kind": "throwing_pit_board",
            "baseline_outcomes": {},
            "branches": [],
        },
        scene_path=Path("/tmp/fake_scene.py"),
        seed=7,
        status="ok",
        write_video=True,
        save_video_fn=_fake_save,
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["vqa_count"] == 3
    assert summary["families"] == [
        "counterfactual_destination",
        "throwing_final_destination",
    ]

    dest_dir = run_dir / "throwing_final_destination"
    assert (dest_dir / "q00.json").is_file()
    assert (dest_dir / "q01.json").is_file()
    assert (dest_dir / "video.mp4").read_bytes() == b"FAKEMP4"
    q00 = json.loads((dest_dir / "q00.json").read_text(encoding="utf-8"))
    assert q00["correct_letter"] == "A"

    cf_dir = run_dir / "counterfactual_destination"
    assert (cf_dir / "q00.json").is_file()
    assert (cf_dir / "video.mp4").is_file()

    assert (run_dir / "baseline.mp4").is_file()
    assert (run_dir / "branches.json").is_file()
    assert len(written) == 1  # baseline only; family video.mp4 are shutil copies


def test_write_run_artifacts_no_video_skips_mp4(tmp_path: Path) -> None:
    """write_video=False still writes JSON artifacts."""

    run_dir = make_run_dir(tmp_path, "02_test", 0, stamp="t0")
    result = SimpleNamespace(
        scene_id="s",
        walltime_s=0.1,
        sim_state=None,
        vqa_payloads=[
            {
                "payload": ("Q?", ["a", "b", "c", "d"], 0),
                "spec": {"family": "motion_state_change", "phrasing": "p"},
            },
        ],
    )
    write_run_artifacts(
        run_dir=run_dir,
        result=result,
        trace={},
        scene_path=Path("scene.py"),
        seed=0,
        status="ok",
        write_video=False,
    )
    fam = run_dir / "motion_state_change"
    assert (fam / "q00.json").is_file()
    assert not (run_dir / "baseline.mp4").exists()
    assert not (fam / "video.mp4").exists()


def test_write_run_artifacts_requires_video_when_write_video(tmp_path: Path) -> None:
    """Missing video_frames raises when write_video=True."""

    run_dir = make_run_dir(tmp_path, "03_test", 1, stamp="t1")
    result = SimpleNamespace(
        scene_id="s",
        walltime_s=0.0,
        sim_state=None,
        vqa_payloads=[],
    )
    with pytest.raises(ValueError, match="video_frames"):
        write_run_artifacts(
            run_dir=run_dir,
            result=result,
            trace={},
            scene_path=Path("scene.py"),
            seed=1,
            status="ok",
            write_video=True,
        )
