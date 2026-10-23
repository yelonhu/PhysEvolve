"""Two-tier cache keyed by Proposer source, stage, and seed.

Each process has a bounded in-memory LRU. The optional shared disk tier stores
results in pickle files and video frames in NumPy sidecars, using atomic
replacement for writes. Failed and successful executor results are both cached.
Keys are the first 16 hexadecimal characters of a SHA-256 digest.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
import threading
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from physevolve.engine.schema import SimState
from physevolve.executor.result import ExecutorResult


# ─── Key helper ───────────────────────────────────────────────────────


def compute_cache_key(code: str, stage: int, seed: int) -> str:
    """Return the deterministic 16-char hex key for ``(code, stage, seed)``.

    Uses SHA-256 and truncates to 16 hex chars (64 bits of entropy). A
    64-bit key is more than enough for expected cache sizes (< 1e6
    entries); collisions would require 2^32 entries by birthday bound.
    """

    payload = f"{code}\x1f{stage}\x1f{seed}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# ─── Serialization helpers ────────────────────────────────────────────


def _strip_video(result: ExecutorResult) -> tuple[ExecutorResult, Optional[np.ndarray]]:
    """Split video frames out of an OK/TRIVIAL result.

    Returns ``(result_without_video, video_frames_or_None)``. For non-OK
    results without a ``sim_state`` this is a no-op.
    """

    if result.sim_state is None:
        return result, None

    video = result.sim_state.video_frames
    # Replace video_frames with an empty placeholder so pickling the
    # result stays cheap. We reconstitute it on load.
    stripped_sim = _replace_frozen(
        result.sim_state,
        video_frames=np.empty((0, 0, 0, 0, 0), dtype=np.uint8),
    )
    stripped_result = _replace_frozen(result, sim_state=stripped_sim)
    return stripped_result, video


def _restore_video(
    result: ExecutorResult, video: Optional[np.ndarray]
) -> ExecutorResult:
    """Put the video frames back onto a previously stripped result."""

    if result.sim_state is None or video is None:
        return result
    restored_sim = _replace_frozen(result.sim_state, video_frames=video)
    return _replace_frozen(result, sim_state=restored_sim)


def _replace_frozen(obj, **changes):
    """Frozen + slotted dataclasses work with :func:`dataclasses.replace`."""

    return replace(obj, **changes)


# ─── SceneCache ───────────────────────────────────────────────────────


class SceneCache:
    """Two-tier LRU cache for :class:`ExecutorResult` instances.

    Thread-safe for the in-memory tier via an internal ``threading.Lock``.
    Disk-tier reads/writes are atomic per entry via ``os.replace``;
    concurrent writers from different processes to the same key resolve
    to last-writer-wins with no partial reads.
    """

    def __init__(
        self,
        max_mem_items: int = 512,
        disk_dir: Optional[Path] = None,
    ) -> None:
        self._max_mem_items = int(max_mem_items)
        self._disk_dir: Optional[Path] = (
            Path(disk_dir) if disk_dir is not None else None
        )
        if self._disk_dir is not None:
            self._disk_dir.mkdir(parents=True, exist_ok=True)
        self._mem: OrderedDict[str, ExecutorResult] = OrderedDict()
        self._lock = threading.Lock()
        # Hit / miss counters for diagnostics (benchmark + progress logging).
        self._mem_hits: int = 0
        self._disk_hits: int = 0
        self._misses: int = 0

    # ── Public API ────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[ExecutorResult]:
        """Return a cached :class:`ExecutorResult` or ``None`` on miss."""

        # 1. In-memory tier.
        with self._lock:
            hit = self._mem.get(key)
            if hit is not None:
                self._mem.move_to_end(key)
                self._mem_hits += 1
                return hit

        # 2. Disk tier.
        if self._disk_dir is None:
            with self._lock:
                self._misses += 1
            return None

        disk_path = self._disk_dir / f"{key}.pkl"
        if not disk_path.exists():
            with self._lock:
                self._misses += 1
            return None

        try:
            with disk_path.open("rb") as fh:
                stripped: ExecutorResult = pickle.load(fh)
        except (pickle.PickleError, EOFError, OSError):
            # Corrupt entry — delete and treat as miss. This can happen
            # if a writer crashed mid-replace (unlikely but defensive).
            try:
                disk_path.unlink(missing_ok=True)
            except OSError:
                pass
            with self._lock:
                self._misses += 1
            return None

        video = self._load_video_sidecar(key)
        result = _restore_video(stripped, video)

        # Promote to in-memory tier.
        with self._lock:
            self._insert_mem_locked(key, result)
            self._disk_hits += 1
        return result

    def put(self, key: str, result: ExecutorResult) -> None:
        """Store a result. Overwrites any existing entry for ``key``."""

        with self._lock:
            self._insert_mem_locked(key, result)

        if self._disk_dir is not None:
            stripped, video = _strip_video(result)
            self._atomic_write(key, stripped, video)

    def stats(self) -> dict[str, int]:
        """Return hit/miss counters (diagnostics only)."""

        with self._lock:
            return {
                "mem_hits": self._mem_hits,
                "disk_hits": self._disk_hits,
                "misses": self._misses,
                "mem_size": len(self._mem),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._mem)

    # ── Internal helpers ─────────────────────────────────────────────

    def _insert_mem_locked(self, key: str, result: ExecutorResult) -> None:
        """Insert into LRU; caller must hold ``self._lock``."""

        if key in self._mem:
            self._mem.move_to_end(key)
            self._mem[key] = result
            return
        self._mem[key] = result
        while len(self._mem) > self._max_mem_items:
            self._mem.popitem(last=False)

    def _atomic_write(
        self,
        key: str,
        stripped: ExecutorResult,
        video: Optional[np.ndarray],
    ) -> None:
        """Write ``{key}.pkl`` + optional ``{key}_video.npz`` atomically."""

        assert self._disk_dir is not None
        pkl_path = self._disk_dir / f"{key}.pkl"
        payload = pickle.dumps(stripped, protocol=pickle.HIGHEST_PROTOCOL)

        fd, tmp = tempfile.mkstemp(
            prefix=f".{key}.", suffix=".pkl.tmp", dir=str(self._disk_dir)
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
            os.replace(tmp, pkl_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        if video is not None and video.size > 0:
            video_path = self._disk_dir / f"{key}_video.npz"
            fd_v, tmp_v = tempfile.mkstemp(
                prefix=f".{key}.", suffix=".npz.tmp", dir=str(self._disk_dir)
            )
            try:
                with os.fdopen(fd_v, "wb") as fh:
                    np.savez_compressed(fh, video=video)
                os.replace(tmp_v, video_path)
            except Exception:
                try:
                    os.unlink(tmp_v)
                except OSError:
                    pass
                raise

    def _load_video_sidecar(self, key: str) -> Optional[np.ndarray]:
        assert self._disk_dir is not None
        video_path = self._disk_dir / f"{key}_video.npz"
        if not video_path.exists():
            return None
        try:
            with np.load(video_path) as data:
                return np.array(data["video"])
        except (OSError, KeyError, ValueError):
            return None


__all__ = [
    "SceneCache",
    "compute_cache_key",
]
