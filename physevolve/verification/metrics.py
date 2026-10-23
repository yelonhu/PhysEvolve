"""Numeric reward helpers for position, status, velocity, and parameter outputs.

Functions return finite scores for valid inputs and handle malformed predictions
with the documented fallback values.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

# Locked exp() scale for position Manhattan distance.
POS_MANHATTAN_SCALE: float = 4.0
# Locked weights for aggregating position + status into body_score.
POS_WEIGHT: float = 0.7
STATUS_WEIGHT: float = 0.3
# Locked tolerance for parameter-level scoring: |pred - true| / tol in
# the (1 - x) softening.
LEVEL_TOLERANCE: int = 4


def manhattan(a: Sequence[int], b: Sequence[int]) -> int:
    """Non-negative L1 distance between two integer sequences.

    Mismatched lengths raise :class:`ValueError`; callers are expected
    to validate schema up front.
    """

    if len(a) != len(b):
        raise ValueError(
            f"manhattan length mismatch: {len(a)} vs {len(b)}"
        )
    return sum(abs(int(x) - int(y)) for x, y in zip(a, b))


def pos_score(
    pred_grid: Sequence[int],
    gt_grid: Sequence[int],
    scale: float = POS_MANHATTAN_SCALE,
) -> float:
    """``exp(-manhattan / scale)``; 1.0 when grids match exactly.

    Args:
        pred_grid: Solver-predicted integer grid coords.
        gt_grid: Ground-truth integer grid coords.
        scale: Exponential softness. Larger → more forgiving.

    Returns:
        Score in ``(0, 1]``.
    """

    d = manhattan(pred_grid, gt_grid)
    return math.exp(-d / scale)


def status_score(pred: str, gt: str) -> float:
    """1.0 iff strings match exactly after case-normalisation else 0.0."""

    return 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0


def body_score_forward(
    pred_grid: Sequence[int],
    pred_status: str,
    gt_grid: Sequence[int],
    gt_status: str,
) -> float:
    """Weighted combination used by :func:`score_forward`."""

    return (
        POS_WEIGHT * pos_score(pred_grid, gt_grid)
        + STATUS_WEIGHT * status_score(pred_status, gt_status)
    )


def level_score(
    pred: int, gt: int, tolerance: int = LEVEL_TOLERANCE
) -> float:
    """``max(0, 1 - |pred - gt| / tolerance)`` — linear partial credit.

    Tolerance is the level distance at which the score hits zero. For
    the default value of 4 (the maximum distance on a
    1..5 scale), the score is 0 only when the prediction is off by the
    full range.
    """

    return max(0.0, 1.0 - abs(int(pred) - int(gt)) / float(tolerance))


def mean(xs: Iterable[float]) -> float:
    """Arithmetic mean; empty input returns ``0.0`` (matches the spec:
    ``empty prediction → 0.0``)."""

    total = 0.0
    n = 0
    for x in xs:
        total += float(x)
        n += 1
    if n == 0:
        return 0.0
    return total / n


__all__ = [
    "POS_MANHATTAN_SCALE",
    "POS_WEIGHT",
    "STATUS_WEIGHT",
    "LEVEL_TOLERANCE",
    "manhattan",
    "pos_score",
    "status_score",
    "body_score_forward",
    "level_score",
    "mean",
]
