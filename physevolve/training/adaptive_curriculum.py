"""Competence-adaptive curriculum for the Naive baseline.

``RollingAccTracker`` records per-skill Solver accuracy. ``adaptive_pick`` samples
one of eight skills with weight ``max(floor, 4 * acc * (1 - acc))``, following a
uniform cold start. ``derive_difficulty`` maps that skill's accuracy to an easy,
medium, or hard prompt. ``ReferenceBuffer`` stores summaries of validated scenes.

Samplers accept an explicit random generator; snapshots return copies.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from physevolve.data_validation.metrics import compute_quality_metrics
from physevolve.engine.schema import SimState


# ─── Tunables (exposed so callers / tests can introspect) ─────────────

DEFAULT_ACC_WINDOW: int = 64
"""Per-skill rolling window length for :class:`RollingAccTracker`."""

DEFAULT_REFERENCE_BUFFER_SIZE: int = 32
"""Per-skill rolling buffer length for :class:`ReferenceBuffer`."""

DEFAULT_REFERENCE_K: int = 2
"""Default number of reference examples drawn per Proposer prompt."""

DEFAULT_COLD_START_STEPS: int = 8
"""Number of training steps during which :func:`adaptive_pick` is uniform.

A skill with zero observations evaluates to the neutral-point accuracy
0.5 (peak weight 1.0), so once cold start ends the tent already favours
unobserved skills. The explicit ``cold_start`` window only matters when
the very first updates have already concentrated weight on a small
subset of skills and we want to guarantee broad coverage anyway.
"""

NEUTRAL_ACC: float = 0.5
"""Returned by :meth:`RollingAccTracker.mean` when the bucket is empty.

Picked deliberately at the tent peak so an uncharted skill attracts
maximum exploratory weight rather than being starved.
"""

WEIGHT_FLOOR: float = 0.05
"""Lower bound on :func:`adaptive_pick`'s per-skill weight.

Even a skill at acc=1.0 (Solver saturated) or acc=0.0 (Solver hopeless)
keeps a small exploration probability so the curriculum can recover if the
Solver's ability drifts. Setting the floor to exactly 0 would let a
single early outlier permanently exclude a skill — undesirable when
small per-skill counts make estimates noisy.
"""


# ─── Difficulty escalation thresholds ────────────────────

DEFAULT_DIFFICULTY_EASY_BELOW: float = 0.35
"""Rolling acc strictly below this value → ``"easy"`` difficulty band.

The Solver is failing more than ~65% of the time on this skill; pushing
harder scenes here would starve learnability further.
"""

DEFAULT_DIFFICULTY_HARD_AT_OR_ABOVE: float = 0.65
"""Rolling acc at or above this value → ``"hard"`` difficulty band.

The Solver clears the easier bands comfortably; this is the escalation
trigger ("the Proposer's scenes haven't gotten harder, but the Solver
already solves them") that motivated the difficulty knob in the first
place. Lowering this threshold escalates sooner but risks pushing the
Solver into the "always 0" floor; raising it delays escalation.
"""


# ─── Tracker ─────────────────────────────────────────────────────────


class RollingAccTracker:
    """Rolling mean Solver accuracy per skill.

    Updates are O(1); the implementation uses a fixed-size :class:`deque`
    per skill. The class is **not** thread-safe — the step builder owns
    a single instance and updates it serially from Phase 6.

    Example:
        >>> t = RollingAccTracker(window_size=4)
        >>> t.update("dynamics_collision", 1.0)
        >>> t.update("dynamics_collision", 0.5)
        >>> round(t.mean("dynamics_collision"), 4)
        0.75
        >>> t.mean("dynamics_throwing")  # empty skill
        0.5
    """

    __slots__ = ("_window_size", "_buckets")

    def __init__(self, window_size: int = DEFAULT_ACC_WINDOW) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive; got {window_size}")
        self._window_size = int(window_size)
        self._buckets: dict[str, deque[float]] = {}

    @property
    def window_size(self) -> int:
        return self._window_size

    def update(self, skill: str, mean_acc: float) -> None:
        """Append ``mean_acc`` to the ``skill`` bucket.

        ``mean_acc`` is clamped to ``[0, 1]`` because Solver rewards are
        themselves clamped there; an out-of-range value is a programmer
        error elsewhere, but silently clamping prevents a single noisy
        update from breaking the tent weight computation downstream.
        """

        key = str(skill)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = deque(maxlen=self._window_size)
            self._buckets[key] = bucket
        clamped = max(0.0, min(1.0, float(mean_acc)))
        bucket.append(clamped)

    def mean(self, skill: str) -> float:
        """Return the rolling mean for the skill, or :data:`NEUTRAL_ACC`.

        Empty buckets return ``0.5`` rather than ``0.0`` so the tent
        sampler treats them as maximally exploratory rather than
        permanently bad. The neutral value also keeps the cold-start
        path's pick weight identical to the warm-start expected value
        at the tent peak, avoiding a step discontinuity when cold start
        ends.
        """

        bucket = self._buckets.get(str(skill))
        if not bucket:
            return NEUTRAL_ACC
        return sum(bucket) / len(bucket)

    def count(self, skill: str) -> int:
        """Return the number of samples currently in the skill bucket."""

        bucket = self._buckets.get(str(skill))
        return 0 if bucket is None else len(bucket)

    def snapshot(self) -> dict[str, float]:
        """Return a ``{skill: mean_acc}`` snapshot of populated buckets.

        Buckets that have never been updated are omitted (they would
        report :data:`NEUTRAL_ACC` but we'd rather make "no data yet"
        visible to the W&B reader instead of hiding it behind 0.5).
        Keys are sorted for deterministic logs.
        """

        result: dict[str, float] = {}
        for skill, bucket in sorted(self._buckets.items()):
            if not bucket:
                continue
            result[skill] = sum(bucket) / len(bucket)
        return result

    def __repr__(self) -> str:
        return (
            f"RollingAccTracker(window_size={self._window_size}, "
            f"skill_count={len(self._buckets)})"
        )


# ─── Adaptive sampler (skill axis only) ──────────────────────────────


def _learnability_weight(acc: float) -> float:
    """self-play tent weight: peaks at acc=0.5, floor at acc=0/1."""

    raw = 4.0 * acc * (1.0 - acc)
    return max(WEIGHT_FLOOR, raw)


def adaptive_pick(
    rng: random.Random,
    tracker: RollingAccTracker,
    skills: Sequence[str],
    *,
    step: int,
    cold_start_uniform_steps: int = DEFAULT_COLD_START_STEPS,
) -> str:
    """Sample one ``skill`` weighted by per-skill learnability.

    Args:
        rng: Caller-owned :class:`random.Random`. The step builder seeds
            this per-item so two rollouts inside the same GRPO group
            share the picked skill (variance-reduction friendly), but
            different items see different picks.
        tracker: Acc tracker used to compute per-skill weights.
        skills: Non-empty skill vocabulary (e.g.
            :data:`PHYSICS_SKILL_TARGETS`).
        step: Current training step (1-indexed). Used only to detect
            whether we are still inside the ``cold_start`` window.
        cold_start_uniform_steps: While ``step <= cold_start_uniform_steps``,
            skills are sampled uniformly so every skill receives at
            least one seed observation before the tent biases sampling.

    Returns:
        A skill drawn from the provided vocabulary.

    Raises:
        ValueError: If the skill vocabulary is empty.
    """

    if not skills:
        raise ValueError("skills vocabulary must be non-empty")

    skill_list: list[str] = list(skills)

    if step <= int(cold_start_uniform_steps):
        return rng.choice(skill_list)

    weights = [_learnability_weight(tracker.mean(skill)) for skill in skill_list]
    return rng.choices(skill_list, weights=weights, k=1)[0]


def pick_weight_snapshot(
    tracker: RollingAccTracker,
    skills: Sequence[str],
) -> dict[str, float]:
    """Return ``{skill: normalised_pick_weight}`` for current acc state.

    Reports the normalised probability mass each skill would receive
    under :func:`adaptive_pick` if it were called right now with a
    post-cold-start ``step``. Skills with no acc data show their neutral
    weight ``4*0.5*0.5 = 1.0`` normalised by the skill count — i.e.
    uniform.
    """

    raw: dict[str, float] = {}
    total = 0.0
    for skill in skills:
        w = _learnability_weight(tracker.mean(skill))
        raw[skill] = w
        total += w

    if total <= 0.0:
        return {skill: 0.0 for skill in raw}
    return {skill: w / total for skill, w in raw.items()}


# ─── Difficulty escalation knob ──────────────────────────


def derive_difficulty(
    skill_mean_acc: Optional[float],
    *,
    easy_below: float = DEFAULT_DIFFICULTY_EASY_BELOW,
    hard_at_or_above: float = DEFAULT_DIFFICULTY_HARD_AT_OR_ABOVE,
) -> str:
    """Map a skill's rolling Solver accuracy to a difficulty band.

    This is the self-play escalation lever. Difficulty is no longer a
    sampling axis; it is a deterministic
    function of how well the Solver currently handles a skill:

    * ``mean_acc < easy_below`` → ``"easy"`` (Solver struggling; do not
      compound the problem with harder scenes).
    * ``easy_below <= mean_acc < hard_at_or_above`` → ``"medium"`` (the
      learnable band; modest complexity to maintain the gradient).
    * ``mean_acc >= hard_at_or_above`` → ``"hard"`` (Solver clears the
      easier bands; push Proposer to escalate scene complexity).
    * ``mean_acc is None`` (cold start) → ``"easy"`` (gentle entry; we
      have no signal yet, defaulting to "hard" would risk starving
      learning before any rolling acc accumulates).

    Args:
        skill_mean_acc: Output of :meth:`RollingAccTracker.mean` for the
            picked skill, or ``None`` to signal "no data". Values
            outside ``[0, 1]`` are clamped (matches tracker contract).
        easy_below: Strict upper bound for the easy band.
        hard_at_or_above: Inclusive lower bound for the hard band.

    Returns:
        One of ``"easy"`` / ``"medium"`` / ``"hard"``.

    Raises:
        ValueError: If thresholds are inconsistent
            (``hard_at_or_above <= easy_below``).
    """

    if hard_at_or_above <= easy_below:
        raise ValueError(
            "derive_difficulty requires hard_at_or_above > easy_below; "
            f"got easy_below={easy_below}, hard_at_or_above={hard_at_or_above}"
        )

    if skill_mean_acc is None:
        return "easy"

    acc = max(0.0, min(1.0, float(skill_mean_acc)))
    if acc < float(easy_below):
        return "easy"
    if acc >= float(hard_at_or_above):
        return "hard"
    return "medium"


# ─── Reference buffer ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ReferenceEntry:
    """One short reference scene description injected into Proposer prompts.

    Crucially **never** contains the raw PhysDSL source: the Proposer
    must be encouraged to vary scenes, not asked to edit them line by
    line. The summary is built from
    :func:`physevolve.data_validation.metrics.compute_quality_metrics`
    output (deterministic) so different training runs with the same
    seeds produce identical reference texts.
    """

    scene_id: str
    one_line_summary: str
    structural_signature: str = ""
    scene_kind: str = ""
    families: tuple[str, ...] = ()
    n_questions: int = 0
    repair_tier: str = "none"

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("scene_id must be a non-empty string")
        if not isinstance(self.one_line_summary, str) or not self.one_line_summary:
            raise ValueError("one_line_summary must be a non-empty string")
        if self.n_questions < 0:
            raise ValueError("n_questions must be non-negative")


def build_one_line_summary(
    sim_state: SimState,
    *,
    scene_kind: str = "",
    families: Sequence[str] = (),
    n_questions: int = 0,
    repair_tier: str = "none",
) -> str:
    """Compose a deterministic one-line summary of an OK scene.

    The summary uses only coarse aggregate metrics (counts, presence
    flags, sorted level lists). It deliberately omits absolute positions,
    velocities, body names, and raw geometric sizes so the Proposer
    cannot literally copy a reference scene into a new prompt — only
    structural mimicry is possible, and we want the Proposer to *vary*
    structure too.
    """

    metrics = compute_quality_metrics(sim_state)

    geom_counts: Mapping[str, int] = metrics.get("geom_type_counts", {}) or {}
    geom_parts = [
        f"{int(count)} {name}{'s' if int(count) != 1 else ''}"
        for name, count in sorted(geom_counts.items())
    ]
    geom_text = ", ".join(geom_parts) if geom_parts else "no bodies"

    n_dynamic = int(metrics.get("dynamic_body_count", 0))
    has_collision = bool(metrics.get("has_collision", False))
    has_spring = bool(metrics.get("has_spring", False))
    has_hinge = bool(metrics.get("has_hinge", False))
    has_slide = bool(metrics.get("has_slide", False))

    feature_flags: list[str] = []
    if has_collision:
        feature_flags.append("collision")
    if has_spring:
        feature_flags.append("spring")
    if has_hinge:
        feature_flags.append("hinge")
    if has_slide:
        feature_flags.append("slide")
    feature_text = ", ".join(feature_flags) if feature_flags else "no joints"

    mass_levels = list(metrics.get("mass_levels_present", ()))
    friction_levels = list(metrics.get("friction_levels_present", ()))
    restitution_levels = list(metrics.get("restitution_levels_present", ()))
    final_statuses: Mapping[str, int] = metrics.get("final_status_counts", {}) or {}
    status_text = ",".join(
        f"{name}:{int(count)}" for name, count in sorted(final_statuses.items())
    )
    family_text = ",".join(sorted(str(f) for f in families if str(f))) or "unspecified"
    repair_text = str(repair_tier or "none")

    return (
        f"scene_kind={scene_kind or 'unknown'}; families={family_text}; {geom_text} "
        f"(n_dynamic={n_dynamic}; features={feature_text}; "
        f"mass_levels={mass_levels}; friction_levels={friction_levels}; "
        f"restitution_levels={restitution_levels}; "
        f"status={status_text or 'unknown'}; "
        f"color_diversity={float(metrics.get('color_diversity_score', 0.0)):.2f}; "
        f"n_questions={int(n_questions)}; repair={repair_text})"
    )


def build_reference_signature(
    sim_state: SimState,
    *,
    scene_kind: str = "",
    families: Sequence[str] = (),
    n_questions: int = 0,
    repair_tier: str = "none",
) -> str:
    """Return a coarse structural key used to deduplicate references."""

    metrics = compute_quality_metrics(sim_state)
    parts = [
        f"scene={scene_kind}",
        "families=" + ",".join(sorted(str(f) for f in families if str(f))),
        "geom=" + repr(tuple(sorted((metrics.get("geom_type_counts", {}) or {}).items()))),
        "joints=" + repr(tuple(sorted((metrics.get("joint_type_counts", {}) or {}).items()))),
        f"dyn={int(metrics.get('dynamic_body_count', 0))}",
        f"collision={bool(metrics.get('has_collision', False))}",
        f"spring={bool(metrics.get('has_spring', False))}",
        f"hinge={bool(metrics.get('has_hinge', False))}",
        f"slide={bool(metrics.get('has_slide', False))}",
        "mass=" + repr(tuple(metrics.get("mass_levels_present", ()))),
        "friction=" + repr(tuple(metrics.get("friction_levels_present", ()))),
        "restitution=" + repr(tuple(metrics.get("restitution_levels_present", ()))),
        "status=" + repr(tuple(sorted((metrics.get("final_status_counts", {}) or {}).items()))),
        f"questions={int(n_questions)}",
        f"repair={repair_tier or 'none'}",
    ]
    return "|".join(parts)


class ReferenceBuffer:
    """Per-skill rolling buffer of validated OK-scene reference entries.

    The step builder calls :meth:`record` after Phase 6 (when an OK
    scene has accumulated enough Solver rollouts to be considered
    "validated") and :meth:`sample` in Phase 1a before rendering the
    next batch of Proposer prompts.

    Cross-skill fallback: if the current skill's buffer is empty and the
    caller asks for ``k`` references, :meth:`sample` borrows from other
    skills' buffers so cold-start steps still get *some* prior context.
    """

    __slots__ = ("_maxlen", "_buffers")

    def __init__(self, maxlen: int = DEFAULT_REFERENCE_BUFFER_SIZE) -> None:
        if maxlen <= 0:
            raise ValueError(f"maxlen must be positive; got {maxlen}")
        self._maxlen = int(maxlen)
        self._buffers: dict[str, deque[ReferenceEntry]] = {}

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def record(
        self,
        skill: str,
        sim_state: SimState,
        scene_id: str,
        *,
        summary: Optional[str] = None,
        scene_kind: str = "",
        families: Sequence[str] = (),
        n_questions: int = 0,
        repair_tier: str = "none",
        structural_signature: Optional[str] = None,
    ) -> ReferenceEntry:
        """Compute (or accept) a one-line summary and append it to the buffer.

        The optional ``summary`` parameter lets tests inject a known
        string without building a real :class:`SimState`. Production code
        always passes the real :class:`SimState` so the deterministic
        summary path runs.
        """

        family_tuple = tuple(str(f) for f in families if str(f))
        if summary is None:
            summary = build_one_line_summary(
                sim_state,
                scene_kind=scene_kind or str(skill),
                families=family_tuple,
                n_questions=n_questions,
                repair_tier=repair_tier,
            )
        if structural_signature is None:
            structural_signature = build_reference_signature(
                sim_state,
                scene_kind=scene_kind or str(skill),
                families=family_tuple,
                n_questions=n_questions,
                repair_tier=repair_tier,
            )
        entry = ReferenceEntry(
            scene_id=str(scene_id),
            one_line_summary=summary,
            structural_signature=str(structural_signature),
            scene_kind=str(scene_kind or skill),
            families=family_tuple,
            n_questions=int(n_questions),
            repair_tier=str(repair_tier or "none"),
        )
        bucket = self._buffers.get(skill)
        if bucket is None:
            bucket = deque(maxlen=self._maxlen)
            self._buffers[skill] = bucket
        for existing in bucket:
            same_signature = (
                entry.structural_signature
                and existing.structural_signature == entry.structural_signature
            )
            if same_signature or existing.one_line_summary == entry.one_line_summary:
                return existing
        bucket.append(entry)
        return entry

    def sample(
        self,
        rng: random.Random,
        skill: str,
        k: int = DEFAULT_REFERENCE_K,
        *,
        allow_cross_skill_fallback: bool = True,
    ) -> list[ReferenceEntry]:
        """Return up to ``k`` reference entries for ``skill``.

        Sampling is **without replacement** so the Proposer never sees
        the same reference twice in one prompt. If the current skill's
        buffer has fewer than ``k`` entries, the deficit is filled from
        a flat pool of all other skills' entries (also without
        replacement) only when ``allow_cross_skill_fallback`` is true.
        Stage-1 keeps that fallback for cold starts; Stage-2 passes false
        because these keys are scene_kind values and cross-kind summaries
        can teach the wrong apparatus. When the chosen pool is empty the
        result is the empty list — the prompt then renders without a
        reference block, which is the intended fallback for the first
        training step of a bucket.
        """

        if k <= 0:
            return []

        primary = list(self._buffers.get(skill, ()))
        if len(primary) >= k:
            return rng.sample(primary, k)

        chosen: list[ReferenceEntry] = list(primary)
        if not allow_cross_skill_fallback:
            return chosen

        deficit = k - len(chosen)
        fallback_pool: list[ReferenceEntry] = []
        for other_skill, bucket in self._buffers.items():
            if other_skill == skill:
                continue
            fallback_pool.extend(bucket)
        if not fallback_pool:
            return chosen
        fill_n = min(deficit, len(fallback_pool))
        chosen.extend(rng.sample(fallback_pool, fill_n))
        return chosen

    def snapshot_sizes(self) -> dict[str, int]:
        """Return ``{skill: buffer_size}`` for metrics logging."""

        return {skill: len(bucket) for skill, bucket in sorted(self._buffers.items())}

    def __repr__(self) -> str:
        return (
            f"ReferenceBuffer(maxlen={self._maxlen}, "
            f"skill_count={len(self._buffers)})"
        )


# ─── Helper used by the step builder when rendering prompts ──────────


def format_reference_examples(entries: Sequence[ReferenceEntry]) -> tuple[str, ...]:
    """Project :class:`ReferenceEntry` list to plain strings for the prompt.

    Kept as a tiny helper so the prompt builder accepts a generic
    ``Sequence[str]`` instead of leaking ``ReferenceEntry`` into the
    prompts module. This keeps the dependency direction one-way:
    ``training -> prompts``.
    """

    return tuple(entry.one_line_summary for entry in entries)


__all__: list[str] = [
    "DEFAULT_ACC_WINDOW",
    "DEFAULT_COLD_START_STEPS",
    "DEFAULT_DIFFICULTY_EASY_BELOW",
    "DEFAULT_DIFFICULTY_HARD_AT_OR_ABOVE",
    "DEFAULT_REFERENCE_BUFFER_SIZE",
    "DEFAULT_REFERENCE_K",
    "NEUTRAL_ACC",
    "WEIGHT_FLOOR",
    "ReferenceBuffer",
    "ReferenceEntry",
    "RollingAccTracker",
    "adaptive_pick",
    "build_reference_signature",
    "build_one_line_summary",
    "derive_difficulty",
    "format_reference_examples",
    "pick_weight_snapshot",
]


_: Any = ReferenceEntry
