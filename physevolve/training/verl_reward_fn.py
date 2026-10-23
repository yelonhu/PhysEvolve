"""PhysEvolve null reward function for the veRL agent-loop reward pipeline.

PhysEvolve computes physics-based rewards externally (in PhysEvolveOnlineStepBuilder),
not inside the veRL agent-loop generation step.  The veRL infrastructure
requires a reward function to be configured so that RewardLoopWorker can
score each generated response.  This module provides a zero-score stub that
satisfies that requirement without interfering with PhysEvolve's own reward
logic, which overwrites ``token_level_scores`` after generation.
"""

from __future__ import annotations


def physevolve_null_compute_score(
    solution_str: str,
    *,
    data_source: str = "",
    ground_truth: str = "",
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Always return 0.0; PhysEvolve's physics rewards overwrite this downstream."""
    return 0.0


compute_score = physevolve_null_compute_score


__all__ = ["compute_score", "physevolve_null_compute_score"]
