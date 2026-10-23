"""Load images and sample video frames into the ``np.ndarray`` format
expected by :meth:`physevolve.inference.hf_backend.HFBackend.generate_multimodal`.

Two routines are exposed:

* :func:`load_image` — PIL-backed RGB loader returning ``(H, W, 3)`` uint8.
* :func:`load_video` — OpenCV-first frame sampler with a PyAV fallback for
  AV1 or OpenCV decode failures, returning ``(n_frames, H, W, 3)`` uint8.
  Both backends use the same uniform bucket-start frame indices.

Both loaders use PIL, OpenCV, and optional PyAV without importing upstream
PhysBench preprocessing modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np


_LOG = logging.getLogger(__name__)


# ─── Image loading ───────────────────────────────────────────────────


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk as ``(H, W, 3) uint8`` RGB.

    Args:
        path: JPG / JPEG / PNG file path.

    Returns:
        A contiguous ``uint8`` array with shape ``(H, W, 3)`` in RGB order.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the decoded array cannot be normalized to 3 channels.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")

    from PIL import Image  # local import — PIL may be lazy-installed

    with Image.open(path) as im:
        rgb = im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(
            f"image at {path} decoded with unexpected shape {arr.shape}"
        )
    return np.ascontiguousarray(arr)


# ─── Video sampling ──────────────────────────────────────────────────


def load_video(
    path: Union[str, Path],
    n_frames: int = 8,
) -> np.ndarray:
    """Uniformly sample ``n_frames`` RGB frames from a video file.

    The normal path uses OpenCV with explicit ``CAP_PROP_POS_FRAMES`` seeks.
    AV01 clips and any OpenCV open/count/read/convert failure are retried with
    PyAV. The public sampling contract and successful OpenCV path are unchanged.

    Args:
        path: Path to an ``.mp4`` file.
        n_frames: Number of frames to extract. Must be ``>= 1``. Default
            ``8``; this matches the sampling rate used by the PhysBench
            authors for their own video-mode evaluations.

    Returns:
        ``(n_frames, H, W, 3)`` ``uint8`` RGB ndarray.

    Raises:
        FileNotFoundError: if the path does not exist.
        RuntimeError: if neither OpenCV nor PyAV can decode the video.
        ValueError: if ``n_frames < 1``.
    """

    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1; got {n_frames}")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")

    try:
        return _load_video_opencv(path, n_frames)
    except Exception as opencv_error:  # noqa: BLE001 - backend fallback.
        _LOG.info(
            "OpenCV video decode unavailable for %s (%s); trying PyAV",
            path,
            opencv_error,
        )
        try:
            return _load_video_pyav(path, n_frames)
        except Exception as pyav_error:  # noqa: BLE001 - combine diagnostics.
            raise RuntimeError(
                f"both video decoders failed for {path}; "
                f"OpenCV: {type(opencv_error).__name__}: {opencv_error}; "
                f"PyAV: {type(pyav_error).__name__}: {pyav_error}"
            ) from pyav_error


def _load_video_opencv(path: Path, n_frames: int) -> np.ndarray:
    """Decode uniformly sampled frames with the existing OpenCV path."""

    import cv2  # local import — heavy dependency

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cv2 could not open video: {path}")
        codec = _decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))
        if codec == "AV01":
            raise RuntimeError("OpenCV reports AV01; PyAV/libdav1d required")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            # Fall back to a manual read-count pass if the container
            # does not expose a frame count (some .mp4 variants).
            total = _count_frames_by_read(cap)
            if total <= 0:
                raise RuntimeError(
                    f"could not determine frame count for {path}"
                )

        indices = _uniform_frame_indices(total, n_frames)

        frames: list[np.ndarray] = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                # On failure, try the previous frame; video containers
                # sometimes mis-report the last index.
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(fi) - 1))
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    raise RuntimeError(
                        f"failed to decode frame {fi} of {path}"
                    )
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb.astype(np.uint8, copy=False))
    finally:
        cap.release()

    return _stack_sampled_frames(frames, n_frames, path=path)


def _load_video_pyav(path: Path, n_frames: int) -> np.ndarray:
    """Decode AV1 or OpenCV-incompatible clips sequentially with PyAV."""

    import av  # local import — optional fallback dependency

    total = _count_frames_pyav(path, av_module=av)
    if total <= 0:
        raise RuntimeError(f"PyAV could not determine frame count for {path}")
    indices = _uniform_frame_indices(total, n_frames)
    wanted = set(indices)
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError(f"PyAV found no video stream in {path}")
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx not in wanted:
                continue
            frame_rgb = frame.to_ndarray(format="rgb24")
            frames.append(frame_rgb.astype(np.uint8, copy=False))
            if len(frames) == len(indices):
                break

    if len(frames) != len(indices):
        raise RuntimeError(
            f"PyAV decoded {len(frames)} sampled frames from {path}; "
            f"expected {len(indices)} from {total} total frames"
        )
    return _stack_sampled_frames(frames, n_frames, path=path)


# ─── Helpers ─────────────────────────────────────────────────────────


def _uniform_frame_indices(total: int, n: int) -> list[int]:
    """Pick ``n`` approximately-uniform frame indices in ``[0, total)``.

    The sampler returns the first index of each of ``n`` equal-sized
    buckets, matching the "one frame per bucket at its start" heuristic
    used by most public video-LLM benchmarks. When ``n > total`` the
    caller is expected to pad after the read loop.
    """

    if total <= 0:
        return []
    if n >= total:
        return list(range(total))
    step = total / n
    return [min(total - 1, int(step * i)) for i in range(n)]


def _decode_fourcc(value: int) -> str:
    """Decode OpenCV's integer FOURCC into an uppercase codec tag."""

    return "".join(
        chr((value >> (8 * offset)) & 0xFF) for offset in range(4)
    ).rstrip("\x00").upper()


def _count_frames_pyav(path: Path, *, av_module) -> int:
    """Count decoded frames with PyAV without retaining full video arrays."""

    with av_module.open(str(path)) as container:
        if not container.streams.video:
            return 0
        return sum(1 for _ in container.decode(video=0))


def _stack_sampled_frames(
    frames: list[np.ndarray],
    n_frames: int,
    *,
    path: Path,
) -> np.ndarray:
    """Pad sampled RGB frames and return the stable public array contract."""

    if not frames:
        raise RuntimeError(f"no video frames decoded from {path}")
    while len(frames) < n_frames:
        frames.append(frames[-1].copy())
    stacked = np.stack(frames[:n_frames], axis=0)
    return np.ascontiguousarray(stacked, dtype=np.uint8)


def _count_frames_by_read(cap) -> int:
    """Fallback frame counter for containers that do not expose CAP_PROP_FRAME_COUNT."""

    import cv2  # noqa: F401 — guaranteed imported by caller

    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    # Reset stream head so the caller can re-seek.
    cap.set(0, 0)  # CAP_PROP_POS_MSEC
    return count


__all__ = [
    "load_image",
    "load_video",
]
