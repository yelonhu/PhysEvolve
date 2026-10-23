"""Tests for the prompt-content builder.

We monkey-patch the visual-loader primitives so these tests don't need
real image/video files on disk — they focus on the tag-to-content
ordering contract that the Qwen3-VL processor relies on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evaluation.physbench import prompter
from evaluation.physbench.data_loader import PhysBenchItem


@pytest.fixture(autouse=True)
def _stub_visuals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``load_image`` / ``load_video`` with tiny stubs that
    return deterministic arrays encoding the source path — letting tests
    assert which visual appears in which slot."""

    def fake_image(path: Path) -> np.ndarray:
        hash_byte = (abs(hash(str(path))) % 250) + 1
        return np.full((2, 2, 3), hash_byte, dtype=np.uint8)

    def fake_video(path: Path, n_frames: int = 8) -> np.ndarray:
        hash_byte = (abs(hash(str(path))) % 250) + 1
        return np.full((n_frames, 2, 2, 3), hash_byte, dtype=np.uint8)

    monkeypatch.setattr(prompter, "load_image", fake_image)
    monkeypatch.setattr(prompter, "load_video", fake_video)


def _item(question: str, file_names: tuple[str, ...]) -> PhysBenchItem:
    return PhysBenchItem(
        idx=0, question=question, mode="general", split="val",
        file_names=file_names,
    )


def test_build_content_preserves_tag_order(tmp_path: Path) -> None:
    q = (
        "Following <video>, which happens first?\n"
        "A. <image>\nB. <image>\nC. <image>\nD. <image>"
    )
    item = _item(q, (
        "v.mp4", "a.png", "b.png", "c.png", "d.png",
    ))
    resolved = [
        tmp_path / "video" / "v.mp4",
        tmp_path / "image" / "a.png",
        tmp_path / "image" / "b.png",
        tmp_path / "image" / "c.png",
        tmp_path / "image" / "d.png",
    ]

    content = prompter.build_content(item, resolved, n_video_frames=4)

    types = [c["type"] for c in content]
    # video first (after leading text), then alternating text/image blocks.
    assert types[0] == "text"
    assert types.count("video") == 1
    assert types.count("image") == 4
    # Last chunk is text + end prompt.
    assert content[-1]["type"] == "text"
    assert prompter.END_PROMPT.strip() in content[-1]["text"]

    # Verify ordering of image blocks matches A/B/C/D order.
    image_values = [c["image"][0, 0, 0] for c in content if c["type"] == "image"]
    expected = [
        (abs(hash(str(tmp_path / "image" / f))) % 250) + 1
        for f in ("a.png", "b.png", "c.png", "d.png")
    ]
    assert image_values == expected


def test_build_content_rejects_mismatched_counts(tmp_path: Path) -> None:
    item = _item("x <image> y", ("a.png", "b.png"))
    with pytest.raises(ValueError):
        prompter.build_content(item, [tmp_path / "image" / "a.png",
                                      tmp_path / "image" / "b.png"])


def test_build_content_kind_mismatch_raises(tmp_path: Path) -> None:
    item = _item("watch <video>", ("thumb.png",))
    with pytest.raises(ValueError):
        prompter.build_content(item, [tmp_path / "image" / "thumb.png"])


def test_count_tags() -> None:
    item = _item("<video> then <image> and <image>", ("v.mp4", "a.png", "b.png"))
    assert prompter.count_tags(item) == (1, 2)


def test_build_content_only_tail_text(tmp_path: Path) -> None:
    item = _item("<image>", ("a.png",))
    content = prompter.build_content(item, [tmp_path / "image" / "a.png"])
    # One image chunk + one text chunk with only the end prompt.
    assert [c["type"] for c in content] == ["image", "text"]
    assert content[-1]["text"].endswith(prompter.END_PROMPT)
