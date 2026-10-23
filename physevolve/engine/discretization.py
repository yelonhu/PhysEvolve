"""Shared velocity bins and quantization helpers.

``tasks.inverse`` uses bin indices in Solver observations, and
``verification.dense_scorer`` reconstructs velocities for replay scoring.
Keeping these definitions in the engine avoids a dependency between those layers.
"""

from __future__ import annotations

import numpy as np

# 7 evenly-spaced (in log sense) bins per axis, covering the DSL's
# |v| <= 6 m/s workspace bound. Order matters: the bin index IS the
# ground-truth label a Solver predicts.
VELOCITY_BINS: tuple[float, ...] = (-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0)


def velocity_to_bin(v: float) -> int:
    """Return the index of the closest bin in :data:`VELOCITY_BINS`.

    Ties break toward the lower bin index (numpy ``argmin`` convention),
    which is stable and deterministic.
    """

    diffs = [abs(float(v) - b) for b in VELOCITY_BINS]
    return int(np.argmin(diffs))


def bin_to_velocity(idx: int) -> float:
    """Inverse of :func:`velocity_to_bin`; raises if out of range."""

    if not (0 <= idx < len(VELOCITY_BINS)):
        raise ValueError(
            f"velocity bin idx {idx} out of [0, {len(VELOCITY_BINS)})"
        )
    return VELOCITY_BINS[idx]


__all__ = [
    "VELOCITY_BINS",
    "velocity_to_bin",
    "bin_to_velocity",
]
