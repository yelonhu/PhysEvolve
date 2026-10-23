"""Convert PhysEvolve's :class:`SimState` video payload into the
representation vLLM expects when consumed by ``AsyncRolloutRequest``.

Reference points:

* `docs/architecture.md` — frame-budget contract per mode:

    - FORWARD   → frames ``[0:16]`` (front half).
    - INVERSE   → frames ``[16:32]`` (back half).
    - PARAMETER → frames ``[0:32]`` (full).

* Canonical Qwen-VL vLLM payload shape:

    ``(np.ndarray (n_frames, H, W, 3) uint8, metadata_dict)``

  where ``metadata_dict`` carries
  ``{total_num_frames, fps, width, height, duration, frames_indices,
  do_sample_frames}``. The sentinel ``do_sample_frames=False`` tells
  Qwen-VL's vLLM preprocessor to skip its own frame-subsampling — we
  deliver the exact set the Solver should see.

This module is strictly numpy + Python; no torch / transformers / verl
imports, so probing code can consume it without touching the training
stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from physevolve.engine.schema import SimState
from physevolve.tasks.router import SolverObservation, TaskMode


# Metadata keys expected by vLLM's Qwen-VL video multimodal path.
# Kept as a tuple so downstream code can do ``key in VLLM_VIDEO_METADATA_KEYS``
# without re-allocating a set on each call.
VLLM_VIDEO_METADATA_KEYS: tuple[str, ...] = (
    "total_num_frames",
    "fps",
    "width",
    "height",
    "duration",
    "frames_indices",
    "do_sample_frames",
)


# Frame budget per TaskMode. The integers are
# the (inclusive-start, exclusive-end) indices into the 32-frame sim
# output for each mode; PARAMETER sees the whole stream.
_FRAME_BUDGET: dict[TaskMode, tuple[int, int]] = {
    TaskMode.FORWARD: (0, 16),
    TaskMode.INVERSE: (16, 32),
    TaskMode.PARAMETER: (0, 32),
}


if TYPE_CHECKING:  # pragma: no cover — type hints only.
    from numpy.typing import NDArray

    VideoArray = NDArray[np.uint8]
else:
    VideoArray = np.ndarray


@dataclass(frozen=True, slots=True)
class VideoBundle:
    """Container holding the pair vLLM's multimodal path expects.

    Kept distinct from the raw ``(ndarray, dict)`` tuple because the
    downstream rollout worker needs to be able to tell a "no video"
    sentinel (``VideoBundle(frames=empty, metadata=empty)``) from an
    accidental ``None`` during shape debugging.
    """

    frames: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Return the ``(frames, metadata)`` pair for vLLM's prompt dict."""

        return self.frames, dict(self.metadata)

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0]) if self.frames.ndim >= 1 else 0


# ─── Core API ────────────────────────────────────────────────────────


def _pick_camera(
    video_frames: np.ndarray, camera_idx: int
) -> np.ndarray:
    """Select a single camera from the ``(n_cam, n_frames, H, W, 3)`` tensor.

    Accepts either a 4-D tensor (``(n_frames, H, W, 3)``, no camera
    dimension) or a 5-D tensor (PhysEvolve engine output). Returns a
    4-D array.
    """

    if video_frames.ndim == 4:
        return video_frames
    if video_frames.ndim == 5:
        if camera_idx < 0 or camera_idx >= video_frames.shape[0]:
            raise IndexError(
                f"camera_idx={camera_idx} out of range "
                f"[0, {video_frames.shape[0]})"
            )
        return video_frames[camera_idx]
    raise ValueError(
        f"unexpected video_frames shape {video_frames.shape}; "
        "expected 4-D (n_frames, H, W, 3) or 5-D (n_cam, n_frames, H, W, 3)"
    )


def _validate_frames(frames: np.ndarray) -> None:
    """Fail loud on dtype / channel mismatches that would blow up vLLM."""

    if not isinstance(frames, np.ndarray):
        raise TypeError(
            f"frames must be np.ndarray, got {type(frames).__name__}"
        )
    if frames.ndim != 4:
        raise ValueError(
            f"frames must be 4-D (n_frames, H, W, 3); got shape {frames.shape}"
        )
    if frames.shape[-1] != 3:
        raise ValueError(
            f"frames last dim must be 3 (RGB); got {frames.shape[-1]}"
        )
    if frames.dtype != np.uint8:
        raise ValueError(f"frames dtype must be uint8; got {frames.dtype}")


def _slice_for_mode(
    frames: np.ndarray, mode: TaskMode
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(subset_frames, frames_indices)`` using the configured frame budget.

    If the sim produced fewer than 32 frames (e.g. a reduced-resolution
    smoke config), the slice is clipped rather than raising — this keeps
    the module robust for low-cost unit tests and golden-scene fixtures
    without breaking the contract for production rollouts.
    """

    total = int(frames.shape[0])
    start, end = _FRAME_BUDGET[mode]
    # Clip to the actual length; PARAMETER always keeps everything.
    start_c = max(0, min(start, total))
    end_c = max(start_c, min(end, total))
    indices = np.arange(start_c, end_c, dtype=np.int64)
    return frames[start_c:end_c], indices


def _infer_fps(sim_state: SimState) -> float:
    """Recover the effective sampling fps from the SimState duration.

    Engine defaults (see ``EnvConfig``): duration 8.0 s / 32 frames =
    4 fps. Falls back to a benign 8 fps if the math is malformed (e.g.
    a test scene with zero duration).
    """

    n_frames = int(sim_state.scene_def.env.n_frames)
    duration = float(sim_state.scene_def.env.duration)
    if n_frames <= 0 or duration <= 0.0:
        return 8.0
    return float(n_frames) / duration


def sim_state_to_vllm_video(
    sim_state: SimState,
    mode: TaskMode,
    *,
    camera_idx: int = 0,
) -> VideoBundle:
    """Convert ``SimState.video_frames`` → :class:`VideoBundle` for vLLM.

    Args:
        sim_state: Engine-produced state. Its ``video_frames`` tensor has
            shape ``(n_cam, n_frames, H, W, 3)`` uint8.
        mode: Task mode; determines which frame subset the Solver sees
            using the configured frame budget.
        camera_idx: Which camera to surface. Training uses camera 0
            exclusively; the kwarg is kept open for Stage-2 VQA scenes
            that might want a specific view.

    Returns:
        :class:`VideoBundle` holding the selected frames (contiguous
        uint8 4-D tensor) and the vLLM metadata dict (video metadata keys).
    """

    if not isinstance(mode, TaskMode):
        raise TypeError(
            f"mode must be TaskMode; got {type(mode).__name__}"
        )

    per_cam = _pick_camera(sim_state.video_frames, camera_idx)
    _validate_frames(per_cam)
    subset, indices = _slice_for_mode(per_cam, mode)
    # vLLM requires contiguous numpy storage so that the bridge to
    # torch tensors inside ``multi_modal_data`` can zero-copy.
    subset = np.ascontiguousarray(subset)

    fps = _infer_fps(sim_state)
    total = int(per_cam.shape[0])
    height = int(per_cam.shape[1])
    width = int(per_cam.shape[2])

    metadata: dict[str, Any] = {
        "total_num_frames": total,
        "fps": float(fps),
        "width": width,
        "height": height,
        "duration": float(total) / fps if fps > 0 else 0.0,
        # Convert to Python ints so OmegaConf / msgpack round-trips
        # deterministically (numpy int64 → str via repr() varies across
        # numpy versions).
        "frames_indices": [int(i) for i in indices.tolist()],
        # tell Qwen-VL's vLLM preprocessor NOT to do its own frame
        # subsampling. We already deliver the exact set.
        "do_sample_frames": False,
    }
    return VideoBundle(frames=subset, metadata=metadata)


def video_bundle_from_observation(
    observation: SolverObservation,
    *,
    total_num_frames: Optional[int] = None,
    fps: Optional[float] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> VideoBundle:
    """Build a :class:`VideoBundle` directly from a :class:`SolverObservation`.

    The observation already carries the per-mode frame slice
    (``tasks/forward.py`` etc. trim the video before returning). We
    therefore don't re-slice; we only attach the metadata the rollout
    worker needs. The caller may supply explicit ``total_num_frames``
    etc. when the raw :class:`SimState` is not in scope (reward-plugin
    call site, where only ``executor_result["solver_observation"]``
    survives).
    """

    frames = observation.video_frames
    # Observations from ``tasks/*.build_observation`` keep the leading
    # camera dimension; collapse it if present.
    if frames.ndim == 5:
        frames = frames[0]
    _validate_frames(frames)
    frames = np.ascontiguousarray(frames)

    n_frames = int(frames.shape[0])
    h = int(height) if height is not None else int(frames.shape[1])
    w = int(width) if width is not None else int(frames.shape[2])
    total = int(total_num_frames) if total_num_frames is not None else n_frames
    effective_fps = float(fps) if fps is not None else 8.0

    # Rebuild `frames_indices` relative to the full 32-frame sim output
    # so downstream logging can line up the Solver's view with the
    # ground truth.
    if observation.mode == TaskMode.FORWARD:
        indices_range = range(0, min(n_frames, 16))
    elif observation.mode == TaskMode.INVERSE:
        # ``16 + i`` so indices sit in the back half of the full stream.
        indices_range = range(16, 16 + n_frames)
    else:  # PARAMETER
        indices_range = range(0, n_frames)

    metadata: dict[str, Any] = {
        "total_num_frames": total,
        "fps": effective_fps,
        "width": w,
        "height": h,
        "duration": float(total) / effective_fps if effective_fps > 0 else 0.0,
        "frames_indices": list(indices_range),
        "do_sample_frames": False,
    }
    return VideoBundle(frames=frames, metadata=metadata)


__all__ = [
    "VideoBundle",
    "VLLM_VIDEO_METADATA_KEYS",
    "sim_state_to_vllm_video",
    "video_bundle_from_observation",
]
