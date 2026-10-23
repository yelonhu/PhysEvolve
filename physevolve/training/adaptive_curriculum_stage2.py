"""Competence-adaptive sampling of Stage-2 question families.

Each family has a rolling Solver-accuracy estimate. Sampling uses the
learnability tent ``max(floor, 4 * acc * (1 - acc))`` with configured priority
and family multipliers, following a uniform cold start. Scene-template
feasibility constraints determine which tasks can be constructed.
"""

from __future__ import annotations

import random
from typing import Mapping, Optional, Sequence

from physevolve.skills.question_families import (
    active_families,
    lookup_family,
)
from physevolve.training.adaptive_curriculum import (
    DEFAULT_ACC_WINDOW,
    DEFAULT_COLD_START_STEPS,
    WEIGHT_FLOOR,
    RollingAccTracker,
)


# Default multiplicative boost applied to the learnability-tent weight of
# priority question families (the shapes Stage-2 pushes hardest).
# 1.0 disables the boost; values > 1 emphasise priority families without
# dropping non-priority coverage.
DEFAULT_FAMILY_PRIORITY_BOOST: float = 2.0
DEFAULT_FAMILY_WEIGHT_MULTIPLIERS: Mapping[str, float] = {
    # collision_final_relation has broad in-scope relation phrasing coverage,
    # but it is not a core "priority" family. Keep priority semantics stable
    # and give it an explicit coverage boost.
    "collision_final_relation": 2.0,
    # Public throwing structure is dominated by branch questions. Use a
    # conservative boost without promoting them to priority families, so the
    # throwing quota serves causal questions more often while the unrestricted
    # slot only shifts slightly toward throwing_pit_board.
    "counterfactual_destination": 1.5,
    "inverse_intervention": 1.5,
}


def make_stage2_tracker(
    window_size: int = DEFAULT_ACC_WINDOW,
) -> RollingAccTracker:
    """Build the rolling-accuracy tracker the step builder owns."""

    return RollingAccTracker(window_size=window_size)


# ─── Question-family sampling layer ───────────────────────────────────


def question_family_vocab(*, include_p1: bool = False) -> tuple[str, ...]:
    """Return the active Stage-2 question-family vocabulary (sorted)."""

    return active_families(include_p1=include_p1)


def _validated_family_vocab(
    families: Optional[Sequence[str]] = None,
) -> tuple[str, ...]:
    """Return a non-empty active P0 family vocabulary."""

    vocab = tuple(families) if families is not None else question_family_vocab()
    if not vocab:
        raise ValueError("Stage-2 question-family vocabulary is empty")
    for name in vocab:
        family = lookup_family(name)
        if family.frozen:
            raise NotImplementedError(
                f"question family {name!r} is frozen and cannot be sampled for "
                "active training"
            )
        if family.tier == "P1":
            raise NotImplementedError(
                f"question family {name!r} is a reserved P1 stub and cannot "
                "be sampled for active training"
            )
    return vocab


def _family_weight(
    tracker: RollingAccTracker,
    family_name: str,
    *,
    priority_boost: float,
    family_weight_multipliers: Mapping[str, float] = DEFAULT_FAMILY_WEIGHT_MULTIPLIERS,
) -> float:
    """Learnability-tent weight times a priority boost for ``family_name``."""

    acc = tracker.mean(family_name)
    tent = max(WEIGHT_FLOOR, 4.0 * acc * (1.0 - acc))
    multiplier = float(family_weight_multipliers.get(family_name, 1.0))
    if lookup_family(family_name).priority:
        return tent * float(priority_boost) * multiplier
    return tent * multiplier


def pick_stage2_question_family(
    rng: random.Random,
    tracker: RollingAccTracker,
    *,
    step: int,
    families: Optional[Sequence[str]] = None,
    cold_start_uniform_steps: int = DEFAULT_COLD_START_STEPS,
    priority_boost: float = DEFAULT_FAMILY_PRIORITY_BOOST,
    family_weight_multipliers: Mapping[str, float] = DEFAULT_FAMILY_WEIGHT_MULTIPLIERS,
) -> str:
    """Sample one question family, prioritising the P0 priority shapes.

    The tracker is keyed by family name (not skill card). During the
    cold-start window every family is given the priority-aware weight too
    (so priority families are seeded faster) but non-priority families still
    receive mass; after cold start the competence-based learnability tent biases toward
    ~50% Solver accuracy on top of the priority boost.

    Args:
        rng: Caller-owned :class:`random.Random` for reproducibility.
        tracker: Per-family rolling Solver accuracy tracker.
        step: Current training step (1-indexed).
        families: Optional override of the family vocabulary; ``None`` uses
            the active P0 families. Passing a P1 family raises (P1 families
            are reserved stubs).
        cold_start_uniform_steps: While ``step <= cold_start_uniform_steps``
            sampling ignores accuracy (but still applies the priority boost).
        priority_boost: Multiplier applied to priority families' weight.
        family_weight_multipliers: Per-family catalog-coverage multipliers
            applied after the priority boost. This lets a broad non-priority
            family receive more samples without changing priority semantics.

    Returns:
        One active question-family key.

    Raises:
        ValueError: If the family vocabulary is empty.
        NotImplementedError: If a P1 (reserved) family is in the vocabulary.
    """

    vocab = _validated_family_vocab(families)

    family_list = list(vocab)
    if step <= int(cold_start_uniform_steps):
        # Cold start: ignore accuracy but keep the priority boost so the
        # priority families accumulate observations faster.
        weights = [
            (float(priority_boost) if lookup_family(name).priority else 1.0)
            * float(family_weight_multipliers.get(name, 1.0))
            for name in family_list
        ]
    else:
        weights = [
            _family_weight(
                tracker,
                name,
                priority_boost=priority_boost,
                family_weight_multipliers=family_weight_multipliers,
            )
            for name in family_list
        ]
    return rng.choices(family_list, weights=weights, k=1)[0]


def pick_stage2_question_families_for_step(
    rng: random.Random,
    tracker: RollingAccTracker,
    *,
    step: int,
    n_picks: int,
    min_scene_kind_picks: Sequence[tuple[str, int]] = (),
    families: Optional[Sequence[str]] = None,
    cold_start_uniform_steps: int = DEFAULT_COLD_START_STEPS,
    priority_boost: float = DEFAULT_FAMILY_PRIORITY_BOOST,
    family_weight_multipliers: Mapping[str, float] = DEFAULT_FAMILY_WEIGHT_MULTIPLIERS,
) -> tuple[str, ...]:
    """Sample one step while enforcing configured scene-kind minimums.

    Quota picks use the same family-level tent sampler restricted to the
    requested scene kind. All remaining slots use the original unrestricted
    sampler, so no active family is removed or globally down-weighted.
    """

    if n_picks <= 0:
        raise ValueError("n_picks must be positive")
    vocab = _validated_family_vocab(families)
    quotas = tuple((str(kind), int(count)) for kind, count in min_scene_kind_picks)
    if any(not kind for kind, _count in quotas):
        raise ValueError("scene-kind quota names must be non-empty")
    if any(count <= 0 for _kind, count in quotas):
        raise ValueError("scene-kind quota counts must be positive")
    if len({kind for kind, _count in quotas}) != len(quotas):
        raise ValueError("scene-kind quotas must not contain duplicate names")
    quota_total = sum(count for _kind, count in quotas)
    if quota_total > n_picks:
        raise ValueError(
            f"scene-kind quotas require {quota_total} picks but n_picks={n_picks}"
        )

    picks: list[str] = []
    for scene_kind, count in quotas:
        eligible = tuple(
            name for name in vocab if lookup_family(name).scene_kind == scene_kind
        )
        if not eligible:
            raise ValueError(
                f"scene-kind quota {scene_kind!r} has no family in the active vocabulary"
            )
        for _ in range(count):
            picks.append(
                pick_stage2_question_family(
                    rng,
                    tracker,
                    step=step,
                    families=eligible,
                    cold_start_uniform_steps=cold_start_uniform_steps,
                    priority_boost=priority_boost,
                    family_weight_multipliers=family_weight_multipliers,
                )
            )

    for _ in range(n_picks - quota_total):
        picks.append(
            pick_stage2_question_family(
                rng,
                tracker,
                step=step,
                families=vocab,
                cold_start_uniform_steps=cold_start_uniform_steps,
                priority_boost=priority_boost,
                family_weight_multipliers=family_weight_multipliers,
            )
        )
    rng.shuffle(picks)
    return tuple(picks)


def stage2_family_pick_weight_snapshot(
    tracker: RollingAccTracker,
    *,
    families: Optional[Sequence[str]] = None,
    priority_boost: float = DEFAULT_FAMILY_PRIORITY_BOOST,
    family_weight_multipliers: Mapping[str, float] = DEFAULT_FAMILY_WEIGHT_MULTIPLIERS,
) -> dict[str, float]:
    """Return ``{family: normalised_pick_weight}`` for the current tracker."""

    vocab = tuple(families) if families is not None else question_family_vocab()
    raw: dict[str, float] = {
        name: _family_weight(
            tracker,
            name,
            priority_boost=priority_boost,
            family_weight_multipliers=family_weight_multipliers,
        )
        for name in vocab
    }
    total = sum(raw.values())
    if total <= 0.0:
        return {name: 0.0 for name in raw}
    return {name: w / total for name, w in raw.items()}


__all__ = [
    "DEFAULT_FAMILY_PRIORITY_BOOST",
    "DEFAULT_FAMILY_WEIGHT_MULTIPLIERS",
    "make_stage2_tracker",
    "pick_stage2_question_families_for_step",
    "pick_stage2_question_family",
    "question_family_vocab",
    "stage2_family_pick_weight_snapshot",
]
