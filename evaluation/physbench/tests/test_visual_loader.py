"""CPU-only regression tests for shared image/video loading."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.physbench import visual_loader


def _touch_video(tmp_path: Path) -> Path:
    """Create a path placeholder for backend-mocked video tests."""

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fixture")
    return path


def test_opencv_success_does_not_call_pyav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing successful OpenCV path must remain byte-identical."""

    path = _touch_video(tmp_path)
    expected = np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3)
    monkeypatch.setattr(
        visual_loader,
        "_load_video_opencv",
        lambda candidate, n_frames: expected,
    )

    def fail_pyav(candidate: Path, n_frames: int) -> np.ndarray:
        raise AssertionError("PyAV must not run after OpenCV success")

    monkeypatch.setattr(visual_loader, "_load_video_pyav", fail_pyav)

    actual = visual_loader.load_video(path, n_frames=2)

    assert actual is expected
    np.testing.assert_array_equal(actual, expected)


def test_av01_fourcc_forces_pyav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AV01 metadata must bypass unsupported OpenCV frame reads."""

    path = _touch_video(tmp_path)
    av01 = sum(ord(char) << (8 * index) for index, char in enumerate("av01"))

    class FakeCapture:
        """Minimal OpenCV capture exposing an AV01 codec tag."""

        released = False

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            assert prop == 6
            return float(av01)

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    fake_cv2 = SimpleNamespace(
        CAP_PROP_FOURCC=6,
        VideoCapture=lambda candidate: capture,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    expected = np.full((3, 2, 2, 3), 7, dtype=np.uint8)
    monkeypatch.setattr(
        visual_loader,
        "_load_video_pyav",
        lambda candidate, n_frames: expected,
    )

    actual = visual_loader.load_video(path, n_frames=3)

    assert capture.released
    assert visual_loader._decode_fourcc(av01) == "AV01"
    np.testing.assert_array_equal(actual, expected)


def test_opencv_failure_falls_back_to_pyav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any OpenCV backend failure should retry the whole clip with PyAV."""

    path = _touch_video(tmp_path)
    expected = np.ones((2, 1, 1, 3), dtype=np.uint8)

    def fail_opencv(candidate: Path, n_frames: int) -> np.ndarray:
        raise RuntimeError("frame zero failed")

    monkeypatch.setattr(visual_loader, "_load_video_opencv", fail_opencv)
    monkeypatch.setattr(
        visual_loader,
        "_load_video_pyav",
        lambda candidate, n_frames: expected,
    )

    np.testing.assert_array_equal(
        visual_loader.load_video(path, n_frames=2),
        expected,
    )


def test_pyav_uses_uniform_indices_and_last_frame_padding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PyAV must share bucket indices and deterministic padding semantics."""

    path = _touch_video(tmp_path)

    class FakeFrame:
        """Frame whose pixel value records its decode index."""

        def __init__(self, value: int) -> None:
            self.value = value

        def to_ndarray(self, *, format: str) -> np.ndarray:
            assert format == "rgb24"
            return np.full((2, 3, 3), self.value, dtype=np.uint8)

    class FakeContainer:
        """Context-managed three-frame PyAV container."""

        streams = SimpleNamespace(video=[object()])

        def __enter__(self) -> "FakeContainer":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def decode(self, *, video: int):
            assert video == 0
            return iter(FakeFrame(value) for value in range(3))

    fake_av = SimpleNamespace(open=lambda candidate: FakeContainer())
    monkeypatch.setitem(sys.modules, "av", fake_av)

    frames = visual_loader._load_video_pyav(path, n_frames=5)

    assert frames.shape == (5, 2, 3, 3)
    assert frames.dtype == np.uint8
    assert frames.flags.c_contiguous
    assert frames[:, 0, 0, 0].tolist() == [0, 1, 2, 2, 2]


def test_both_backends_report_combined_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A total decode failure should retain both backend error messages."""

    path = _touch_video(tmp_path)

    def fail_opencv(candidate: Path, n_frames: int) -> np.ndarray:
        raise RuntimeError("opencv failed")

    def fail_pyav(candidate: Path, n_frames: int) -> np.ndarray:
        raise ValueError("pyav failed")

    monkeypatch.setattr(visual_loader, "_load_video_opencv", fail_opencv)
    monkeypatch.setattr(visual_loader, "_load_video_pyav", fail_pyav)

    with pytest.raises(RuntimeError, match="OpenCV.*opencv failed.*PyAV.*pyav"):
        visual_loader.load_video(path, n_frames=2)
