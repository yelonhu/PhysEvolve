"""Proposer reward — failure penalty + learnability + physics intrinsic terms.

Two regimes:

* **Non-OK** ``ExecutorResult``: return the fixed penalty from
  :data:`PROPOSER_PENALTY` (scaled by ``failure_weight``). No
  learnability or intrinsic terms — failure rewards use the configured penalty table.
* **OK**: compute ``1 - 2 * |mean_acc - 0.5|`` so the learnability tent
  peaks at exactly 50% Solver accuracy. Requires at least
  ``min_rollouts_for_learnability`` Solver rollouts; otherwise the OK
  reward is zero. If ``intrinsic`` is provided, add a physics-version
  intrinsic shaping term so the Proposer is also rewarded for non-trivial
  motion, interaction richness, complexity, scene novelty, and parameter
  identifiability — the five diagnostic terms measured in
  the code domain (complexity / edit-distance / Halstead / answer
  diversity) translated to physical scenes.

The intrinsic term reuses the deterministic quality metrics already
computed by :mod:`physevolve.data_validation`. Scene novelty is computed
by the caller (which owns the rolling window of recent scenes) and passed
in via :class:`IntrinsicInputs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from physevolve.data_validation.metrics import compute_quality_metrics
from physevolve.data_validation.quality import evaluate_quality
from physevolve.engine.failure import PROPOSER_PENALTY, FailureReason
from physevolve.engine.schema import SimState
from physevolve.executor.result import ExecutorResult

# Minimum Solver rollouts for a learnability reward.
DEFAULT_MIN_ROLLOUTS: int = 4

# Component order is locked: motion, interaction, complexity, diversity, parameter.
DEFAULT_INTRINSIC_WEIGHT: float = 0.3
DEFAULT_INTRINSIC_COMPONENT_WEIGHTS: tuple[float, float, float, float, float] = (
    0.30,
    0.25,
    0.20,
    0.20,
    0.05,
)

_INTRINSIC_COMPONENT_ORDER: tuple[str, ...] = (
    "motion",
    "interaction",
    "complexity",
    "diversity",
    "parameter",
)


SceneSignature = tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class IntrinsicInputs:
    """Inputs for the OK-branch physics-version intrinsic shaping term.

    Attributes:
        sim_state: Executor-produced :class:`SimState`. Used to derive the
            ``motion``, ``interaction``, ``complexity``, and
            ``parameter_identifiability`` components from existing
            deterministic quality metrics.
        signature: Stable scene signature (caller-computed via
            :func:`compute_scene_signature`). Compared against ``history``
            to derive the ``diversity`` / novelty component.
        history: Recent prior signatures (e.g. a deque snapshot owned by
            the step builder). May be empty.
        weight: Outer scale applied to the entire intrinsic sum. Defaults
            to :data:`DEFAULT_INTRINSIC_WEIGHT` (0.3) so the OK reward
            stays inside ``[0, learnability_w + weight]``.
        component_weights: Inner weights, in the order
            ``(motion, interaction, complexity, diversity, parameter)``.
            Must be five non-negative floats; they are normalised to sum
            to 1.0 inside :func:`intrinsic_reward_terms`.
    """

    sim_state: SimState
    signature: SceneSignature
    history: Sequence[SceneSignature] = ()
    weight: float = DEFAULT_INTRINSIC_WEIGHT
    component_weights: tuple[float, float, float, float, float] = (
        DEFAULT_INTRINSIC_COMPONENT_WEIGHTS
    )


@dataclass(frozen=True, slots=True)
class IntrinsicBreakdown:
    """Result of computing the intrinsic shaping term for one OK scene."""

    components: dict[str, float] = field(default_factory=dict)
    weighted_sum: float = 0.0
    weight: float = 0.0
    total: float = 0.0


def learnability(mean_accuracy: float) -> float:
    """Tent function peaking at 0.5 accuracy; bounded in ``[0, 1]``."""

    x = max(0.0, min(1.0, float(mean_accuracy)))
    return 1.0 - 2.0 * abs(x - 0.5)


def family_difficulty_weight(rolling_acc: float, beta: float) -> float:
    """Scale proposer OK reward by family rolling accuracy (hard families up).

    ``weight = clip(1 + beta * (0.5 - rolling_acc), 0.5, 1.5)``. When
    ``beta <= 0`` the weight is always 1.0 (legacy no-op).
    """

    if float(beta) <= 0.0:
        return 1.0
    acc = max(0.0, min(1.0, float(rolling_acc)))
    weight = 1.0 + float(beta) * (0.5 - acc)
    return max(0.5, min(1.5, weight))


@dataclass(frozen=True, slots=True)
class CopyControl:
    """Anti-copy / saturation / diversity adjustments for the OK reward.

    These signals are caller-derived (the Stage-2 step builder owns the
    AST copy diagnostics and the per-step intra-group dedup) and applied
    on the OK branch of :func:`compute_proposer_reward`. Defaults are a
    no-op so Stage-1 and legacy callers keep the pre-existing contract.

    Attributes:
        is_exact_copy: The program is a verbatim AST copy of the golden
            fewshot or repeats a recent OK AST inside the rolling window.
        is_batch_duplicate: The program duplicates another rollout's AST
            inside the SAME GRPO group this step (the caller keeps one
            canonical rollout and flags the rest).
        copy_penalty: Reward assigned (overriding learnability/intrinsic)
            when ``is_exact_copy`` or ``is_batch_duplicate`` is True. Use
            a negative value so copying is strictly worse than a
            non-degenerate OK scene.
        near_copy_similarity: Max AST similarity in ``[0, 1]`` against the
            golden fewshot / recent window. Above ``near_copy_threshold``
            the reward is capped at ``copy_cap``.
        near_copy_threshold: Similarity at/above which the cap applies.
        copy_cap: Upper bound on the OK reward for a near-copy (applied
            only when ``copy_cap`` is not None).
        diversity_bonus: Additive diversity reward (caller computes it as
            ``diversity_weight * (1 - near_copy_similarity)``); kept at
            learnability magnitude so structurally novel scenes get a real
            positive gradient.
        saturation_threshold: ``mean_acc`` above this is penalised.
        saturation_weight: Scale on the saturation penalty; 0 disables it.
        enabled: Master switch. When False the control is a no-op.
    """

    is_exact_copy: bool = False
    is_batch_duplicate: bool = False
    copy_penalty: float = 0.0
    near_copy_similarity: float = 0.0
    near_copy_threshold: float = 0.95
    copy_cap: Optional[float] = None
    diversity_bonus: float = 0.0
    saturation_threshold: float = 0.85
    saturation_weight: float = 0.0
    enabled: bool = False


def _saturation_penalty(mean_acc: float, control: CopyControl) -> float:
    """Return the (non-negative) learnability-saturation penalty."""

    if control.saturation_weight <= 0.0:
        return 0.0
    threshold = float(control.saturation_threshold)
    if threshold >= 1.0 or mean_acc <= threshold:
        return 0.0
    overshoot = (mean_acc - threshold) / (1.0 - threshold)
    return float(control.saturation_weight) * float(overshoot)


def apply_copy_control(
    ok_reward: float, mean_acc: float, control: Optional[CopyControl]
) -> float:
    """Apply anti-copy / saturation / diversity shaping to an OK reward.

    The order is deliberate: diversity is added and saturation subtracted
    first (they shape the "good" region), then a near-copy cap clips the
    result, and finally an exact-copy / batch-duplicate override forces
    the hard ``copy_penalty`` so degenerate rollouts sit below every
    non-degenerate OK scene and create intra-group GRPO variance.
    """

    if control is None or not control.enabled:
        return ok_reward

    reward = ok_reward + float(control.diversity_bonus)
    reward -= _saturation_penalty(mean_acc, control)

    if (
        control.copy_cap is not None
        and control.near_copy_similarity >= control.near_copy_threshold
    ):
        reward = min(reward, float(control.copy_cap))

    if control.is_exact_copy or control.is_batch_duplicate:
        reward = float(control.copy_penalty)

    return reward


def repair_reward_discount(executor_result: ExecutorResult) -> float:
    """Return the multiplicative OK reward discount for repaired scenes."""

    trace = getattr(executor_result, "repair_trace", None)
    if not isinstance(trace, dict) or not bool(trace.get("applied", False)):
        return 1.0
    try:
        value = float(trace.get("reward_discount", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, value))


def compute_scene_signature(
    sim_state: SimState,
    *,
    skill_target: Optional[str] = None,
    phrasing: Optional[str] = None,
    params_signature: Optional[str] = None,
) -> SceneSignature:
    """Return a hashable scene signature for novelty comparison.

    Built from existing :func:`compute_quality_metrics` outputs so we
    reuse the deterministic feature extractors instead of re-deriving
    them on the hot path. The signature is intentionally coarse: two
    scenes share a signature if their gross physical structure is
    indistinguishable (same geometry mix, joint mix, collision presence,
    locked-level mixes, and body-count band). That collision rule is what
    the Proposer needs to break to earn novelty reward.

    Stage-2 callers may additionally pass ``skill_target`` (now the
    scene_kind) / ``phrasing`` / ``params_signature`` (canonical-form params
    hash from
    :func:`physevolve.skills.scene_templates.render_utils.canonical_params_signature`).
    Threading these into the signature keeps question-level novelty as
    ``(scene_kind, phrasing, params_hash)``: two scenes that share gross
    structure but probe different scene_kinds no longer collapse onto the
    same signature.
    """

    metrics = compute_quality_metrics(sim_state)
    physical = _signature_from_metrics(metrics)
    if skill_target is None and phrasing is None and params_signature is None:
        return physical
    return physical + (
        ("skill", skill_target),
        ("phrasing", phrasing),
        ("params_signature", params_signature),
    )


def _signature_from_metrics(metrics: Mapping[str, Any]) -> SceneSignature:
    return (
        tuple(sorted(metrics.get("geom_type_counts", {}).items())),
        tuple(sorted(metrics.get("joint_type_counts", {}).items())),
        bool(metrics.get("has_collision", False)),
        bool(metrics.get("has_spring", False)),
        bool(metrics.get("has_hinge", False)),
        bool(metrics.get("has_slide", False)),
        _body_count_band(int(metrics.get("dynamic_body_count", 0))),
        tuple(metrics.get("mass_levels_present", ())),
        tuple(metrics.get("friction_levels_present", ())),
        tuple(metrics.get("restitution_levels_present", ())),
    )


def _body_count_band(n: int) -> str:
    if n <= 1:
        return "single"
    if n == 2:
        return "pair"
    if 3 <= n <= 5:
        return "small_group"
    if 6 <= n <= 10:
        return "medium_group"
    return "large_group"


def _parameter_identifiability(metrics: Mapping[str, Any]) -> float:
    """Reward when ≥2 distinct locked levels show up in ≥2 parameter families.

    Mass, friction, and restitution are the three "hidden" parameters the
    Solver is asked to recover in the Stage-1 parameter task. A scene where
    only one family has multiple levels is half-informative; a scene where
    two or more families do is fully informative.
    """

    families = (
        len(metrics.get("mass_levels_present", ())) > 1,
        len(metrics.get("friction_levels_present", ())) > 1,
        len(metrics.get("restitution_levels_present", ())) > 1,
    )
    n_informative = sum(1 for ok in families if ok)
    if n_informative >= 2:
        return 1.0
    if n_informative == 1:
        return 0.5
    return 0.0


def _novelty(signature: SceneSignature, history: Sequence[SceneSignature]) -> float:
    """Return 1.0 if signature is novel against history, else 0.0.

    The simple exact-match rule is intentional: signatures are coarse, so
    any duplicate inside the rolling window is a clear sign the Proposer
    is cycling on the same structural template. Soft partial matching
    would dilute the anti-collapse signal we are trying to install.
    """

    if not history:
        return 1.0
    return 0.0 if signature in history else 1.0


def intrinsic_reward_terms(intrinsic: IntrinsicInputs) -> IntrinsicBreakdown:
    """Compute the five-component physics-version intrinsic shaping term.

    Returns:
        IntrinsicBreakdown carrying per-component scores in ``[0, 1]``,
        the inner weighted sum (also in ``[0, 1]``), the outer ``weight``,
        and ``total = weight * weighted_sum``.
    """

    metrics = compute_quality_metrics(intrinsic.sim_state)
    quality = evaluate_quality(intrinsic.sim_state)
    quality_components = quality.metrics.get("quality_components", {})

    components = {
        "motion": float(quality_components.get("motion", 0.0)),
        "interaction": float(quality_components.get("interaction", 0.0)),
        "complexity": float(quality_components.get("complexity", 0.0)),
        "diversity": _novelty(intrinsic.signature, intrinsic.history),
        "parameter": _parameter_identifiability(metrics),
    }

    weights = _normalised_component_weights(intrinsic.component_weights)
    weighted_sum = sum(weights[i] * components[name] for i, name in enumerate(_INTRINSIC_COMPONENT_ORDER))
    weighted_sum = max(0.0, min(1.0, weighted_sum))
    weight = max(0.0, float(intrinsic.weight))
    total = weight * weighted_sum

    return IntrinsicBreakdown(
        components=components,
        weighted_sum=weighted_sum,
        weight=weight,
        total=total,
    )


def _normalised_component_weights(
    raw: Sequence[float],
) -> tuple[float, float, float, float, float]:
    if len(raw) != 5:
        raise ValueError(
            "intrinsic component_weights must have exactly 5 entries "
            "(motion, interaction, complexity, diversity, parameter)"
        )
    if any(float(w) < 0 for w in raw):
        raise ValueError("intrinsic component_weights must be non-negative")
    total = float(sum(raw))
    if total <= 0.0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    return tuple(float(w) / total for w in raw)  # type: ignore[return-value]


def compute_proposer_reward(
    executor_result: ExecutorResult,
    solver_rewards: Sequence[float],
    failure_weight: float = 1.0,
    learnability_weight: float = 1.0,
    min_rollouts_for_learnability: int = DEFAULT_MIN_ROLLOUTS,
    intrinsic: Optional[IntrinsicInputs] = None,
    copy_control: Optional[CopyControl] = None,
) -> float:
    """Return the Proposer's scalar reward for a single scene.

    Args:
        executor_result: Pipeline output for the scene.
        solver_rewards: Solver rewards across N rollouts on this scene.
            GRPO's group size guarantees N ≥ 4 in training.
        failure_weight: Linear scale on the per-status penalty.
        learnability_weight: Linear scale on the learnability term.
        min_rollouts_for_learnability: Threshold below which the OK
            branch returns 0 (insufficient samples for a meaningful mean
            accuracy estimate).
        intrinsic: Optional physics-version intrinsic inputs. When None,
            the OK reward is the pre-existing pure-learnability value
            (legacy contract). When provided, ``intrinsic.weight *
            normalised_components`` is added on top of learnability.

    Returns:
        A scalar reward. Non-OK branches return the negative (or zero)
        penalty exactly as before. OK branch returns
        ``learnability_weight * learnability(mean_acc) + intrinsic.total``,
        with ``intrinsic.total == 0`` when intrinsic is None.
    """

    status = executor_result.status
    if status != FailureReason.OK:
        penalty = PROPOSER_PENALTY[status]
        return failure_weight * penalty

    # Hard copy override does not need Solver rewards: a batch duplicate is
    # not even sent to the Solver, so force the penalty before the
    # insufficient-samples short-circuit below.
    if (
        copy_control is not None
        and copy_control.enabled
        and (copy_control.is_exact_copy or copy_control.is_batch_duplicate)
    ):
        return float(copy_control.copy_penalty)

    if len(solver_rewards) < min_rollouts_for_learnability:
        return 0.0

    mean_acc = sum(float(r) for r in solver_rewards) / float(len(solver_rewards))
    reward = learnability_weight * learnability(mean_acc)

    if intrinsic is not None:
        breakdown = intrinsic_reward_terms(intrinsic)
        reward += breakdown.total

    reward = apply_copy_control(reward, mean_acc, copy_control)
    return reward * repair_reward_discount(executor_result)


def compute_proposer_reward_with_intrinsic(
    executor_result: ExecutorResult,
    solver_rewards: Sequence[float],
    intrinsic: IntrinsicInputs,
    failure_weight: float = 1.0,
    learnability_weight: float = 1.0,
    min_rollouts_for_learnability: int = DEFAULT_MIN_ROLLOUTS,
    copy_control: Optional[CopyControl] = None,
) -> tuple[float, Optional[IntrinsicBreakdown]]:
    """Like :func:`compute_proposer_reward` but also returns the breakdown.

    Convenience for callers that want to log per-component scores. On the
    failure branch ``IntrinsicBreakdown`` is ``None`` because intrinsic
    shaping is skipped for failed scenes.
    """

    status = executor_result.status
    if status != FailureReason.OK:
        penalty = PROPOSER_PENALTY[status]
        return failure_weight * penalty, None

    if (
        copy_control is not None
        and copy_control.enabled
        and (copy_control.is_exact_copy or copy_control.is_batch_duplicate)
    ):
        return float(copy_control.copy_penalty), None

    if len(solver_rewards) < min_rollouts_for_learnability:
        return 0.0, None

    mean_acc = sum(float(r) for r in solver_rewards) / float(len(solver_rewards))
    base = learnability_weight * learnability(mean_acc)
    breakdown = intrinsic_reward_terms(intrinsic)
    reward = apply_copy_control(base + breakdown.total, mean_acc, copy_control)
    reward *= repair_reward_discount(executor_result)
    return reward, breakdown


__all__ = [
    "DEFAULT_INTRINSIC_COMPONENT_WEIGHTS",
    "DEFAULT_INTRINSIC_WEIGHT",
    "DEFAULT_MIN_ROLLOUTS",
    "CopyControl",
    "IntrinsicBreakdown",
    "IntrinsicInputs",
    "SceneSignature",
    "apply_copy_control",
    "compute_proposer_reward",
    "compute_proposer_reward_with_intrinsic",
    "compute_scene_signature",
    "family_difficulty_weight",
    "intrinsic_reward_terms",
    "learnability",
    "repair_reward_discount",
]
