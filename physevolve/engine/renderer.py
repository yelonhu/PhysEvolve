"""Render saved MuJoCo trajectory states from one or more cameras.

The simulator records ``qpos``, ``qvel``, and time while stepping. This module
replays those snapshots for rendering with the configured OpenGL backend.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import mujoco
import numpy as np

# ─── Renderer cache (module-level LRU) ─────────────────────────────────

_RENDERER_CACHE_MAX: int = 8
# Keys: (id(model), height, width). Using id() keeps us from holding a
# hard reference to the MjModel from the cache, which matters because
# the simulator owns the model's lifetime; once the caller drops the
# model the cache entry is still safe to evict (we purge on the next
# call with the same key).
_RENDERER_CACHE: "OrderedDict[tuple[int, int, int], mujoco.Renderer]" = OrderedDict()


def _get_renderer(
    model: mujoco.MjModel, height: int, width: int
) -> mujoco.Renderer:
    """Return a cached renderer, creating one if needed.

    Evicts the least-recently-used entry once the cache exceeds
    :data:`_RENDERER_CACHE_MAX` to bound EGL context memory.
    """

    key = (id(model), height, width)
    renderer = _RENDERER_CACHE.get(key)
    if renderer is None:
        renderer = mujoco.Renderer(model, height=height, width=width)
        _RENDERER_CACHE[key] = renderer
        while len(_RENDERER_CACHE) > _RENDERER_CACHE_MAX:
            _, evicted = _RENDERER_CACHE.popitem(last=False)
            # Mujoco renderers don't expose a public close() on all
            # versions; best-effort release.
            close = getattr(evicted, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
    else:
        _RENDERER_CACHE.move_to_end(key)
    return renderer


def _clear_renderer_cache() -> None:
    """Test hook: drop every cached renderer (not part of the public API)."""

    _RENDERER_CACHE.clear()


# ─── Public API ────────────────────────────────────────────────────────


def render_multiview(
    model: mujoco.MjModel,
    frame_snapshots: Sequence[tuple[np.ndarray, np.ndarray, float]],
    camera_names: Sequence[str],
    height: int = 480,
    width: int = 480,
) -> np.ndarray:
    """Render every ``(camera, frame)`` pair into a single uint8 tensor.

    Args:
        model: Compiled MuJoCo model. Cameras referenced by
            ``camera_names`` must be present in ``model``.
        frame_snapshots: One ``(qpos, qvel, time)`` tuple per frame.
            ``qpos``/``qvel`` are NumPy arrays; the renderer writes them
            into a fresh :class:`mujoco.MjData` and calls
            :func:`mujoco.mj_forward` before rendering.
        camera_names: Names of cameras to render from, in order.
        height: Output frame height in pixels.
        width: Output frame width in pixels.

    Returns:
        ``(n_cameras, n_frames, H, W, 3)`` uint8 array — the layout
        promised by :class:`SimState.video_frames`.
    """

    n_cam = len(camera_names)
    n_frames = len(frame_snapshots)
    if n_cam == 0:
        raise ValueError("render_multiview requires at least one camera name")
    if n_frames == 0:
        raise ValueError("render_multiview requires at least one frame snapshot")

    renderer = _get_renderer(model, height=height, width=width)
    scratch = mujoco.MjData(model)

    out = np.empty((n_cam, n_frames, height, width, 3), dtype=np.uint8)
    for f_idx, (qpos, qvel, t) in enumerate(frame_snapshots):
        # Replay the snapshot into the scratch data and propagate kinematics.
        np.copyto(scratch.qpos, qpos)
        np.copyto(scratch.qvel, qvel)
        scratch.time = float(t)
        mujoco.mj_forward(model, scratch)
        for cam_idx, cam_name in enumerate(camera_names):
            renderer.update_scene(scratch, camera=cam_name)
            out[cam_idx, f_idx] = renderer.render()
    return out


__all__ = ["render_multiview"]
