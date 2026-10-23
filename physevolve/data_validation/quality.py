"""Quality scoring for probing-candidate scenes."""

from __future__ import annotations

from typing import Any

from physevolve.data_validation.metrics import compute_quality_metrics
from physevolve.data_validation.schemas import QualityReport
from physevolve.engine.schema import SimState


_TRIVIAL_DISP_THRESHOLD = 0.05
_LOW_MOTION_THRESHOLD = 0.10
_GOOD_MOTION_DISP = 0.20
_HIGH_MOTION_DISP = 2.0
_ABSURD_SPEED = 30.0
_CHAOTIC_COLLISION_COUNT = 250


def evaluate_quality(sim_state: SimState) -> QualityReport:
    """Compute deterministic quality diagnostics and bounded score."""

    metrics = compute_quality_metrics(sim_state)
    components = {
        "motion": _motion_score(metrics),
        "interaction": _interaction_score(metrics),
        "complexity": _complexity_score(metrics),
        "probe_suitability": _probe_suitability_score(metrics),
        "visual": _visual_score(metrics),
    }
    score = (
        0.25 * components["motion"]
        + 0.25 * components["interaction"]
        + 0.20 * components["complexity"]
        + 0.20 * components["probe_suitability"]
        + 0.10 * components["visual"]
    )
    score = _clamp(score)

    task_suitability = _task_suitability(metrics)
    tags = _tags(metrics, task_suitability)
    reject_reasons = _hard_reject_reasons(metrics)

    enriched_metrics = dict(metrics)
    enriched_metrics["quality_components"] = components

    return QualityReport(
        hard_pass=len(reject_reasons) == 0,
        score=score,
        metrics=enriched_metrics,
        tags=tags,
        reject_reasons=tuple(reject_reasons),
        task_suitability=task_suitability,
    )


def _hard_reject_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not bool(metrics.get("video_complete", False)):
        reasons.append("video incomplete")
    if int(metrics.get("dynamic_body_count", 0)) < 1:
        reasons.append("no dynamic bodies")
    if float(metrics.get("max_displacement", 0.0)) < _TRIVIAL_DISP_THRESHOLD:
        reasons.append("max displacement below trivial threshold")
    if float(metrics.get("max_speed", 0.0)) > _ABSURD_SPEED:
        reasons.append("max speed is likely explosive")
    if int(metrics.get("collision_count", 0)) > _CHAOTIC_COLLISION_COUNT:
        reasons.append("too many collision events")
    if float(metrics.get("out_of_frame_ratio", 0.0)) > 0.5:
        reasons.append("most dynamic bodies end out of workspace")
    return reasons


def _motion_score(metrics: dict[str, Any]) -> float:
    max_disp = float(metrics.get("max_displacement", 0.0))
    max_speed = float(metrics.get("max_speed", 0.0))
    if max_disp <= 0.0:
        return 0.0
    if max_disp < _GOOD_MOTION_DISP:
        disp_score = max_disp / _GOOD_MOTION_DISP
    elif max_disp <= _HIGH_MOTION_DISP:
        disp_score = 1.0
    else:
        disp_score = max(0.0, 1.0 - (max_disp - _HIGH_MOTION_DISP) / 2.0)
    speed_penalty = 0.4 if max_speed > _ABSURD_SPEED else 0.0
    return _clamp(disp_score - speed_penalty)


def _interaction_score(metrics: dict[str, Any]) -> float:
    score = 0.0
    if bool(metrics.get("has_collision", False)):
        score += 0.55
        score += min(0.25, 0.05 * int(metrics.get("unique_collision_pairs", 0)))
    if bool(metrics.get("has_spring", False)):
        score += 0.20
    if bool(metrics.get("has_hinge", False)):
        score += 0.15
    if bool(metrics.get("has_slide", False)):
        score += 0.15
    return _clamp(score)


def _complexity_score(metrics: dict[str, Any]) -> float:
    n_dyn = int(metrics.get("dynamic_body_count", 0))
    geom_types = len(metrics.get("geom_type_counts", {}))
    constraint_bonus = sum(
        1
        for key in ("has_spring", "has_hinge", "has_slide")
        if bool(metrics.get(key, False))
    )

    if n_dyn <= 0:
        body_score = 0.0
    elif n_dyn == 1:
        body_score = 0.35
    elif 2 <= n_dyn <= 5:
        body_score = 1.0
    elif n_dyn <= 10:
        body_score = 0.65
    else:
        body_score = 0.35

    diversity = min(1.0, max(0.0, (geom_types - 1) / 3.0))
    return _clamp(0.75 * body_score + 0.15 * diversity + 0.10 * min(1.0, constraint_bonus / 2.0))


def _probe_suitability_score(metrics: dict[str, Any]) -> float:
    suitability = _task_suitability(metrics)
    base = sum(1.0 for ok in suitability.values() if ok) / len(suitability)
    mass_contrast = len(metrics.get("mass_levels_present", ())) > 1
    friction_contrast = len(metrics.get("friction_levels_present", ())) > 1
    contrast_bonus = 0.15 if (mass_contrast or friction_contrast) else 0.0
    return _clamp(base + contrast_bonus)


def _visual_score(metrics: dict[str, Any]) -> float:
    if not bool(metrics.get("video_complete", False)):
        return 0.0
    out_ratio = float(metrics.get("out_of_frame_ratio", 0.0))
    color = float(metrics.get("color_diversity_score", 0.0))
    camera_bonus = 0.1 if int(metrics.get("n_cameras", 0)) >= 1 else 0.0
    return _clamp(0.75 * (1.0 - out_ratio) + 0.15 * color + camera_bonus)


def _task_suitability(metrics: dict[str, Any]) -> dict[str, bool]:
    max_disp = float(metrics.get("max_displacement", 0.0))
    max_speed = float(metrics.get("max_speed", 0.0))
    n_dyn = int(metrics.get("dynamic_body_count", 0))
    has_interaction = bool(metrics.get("has_collision", False)) or any(
        bool(metrics.get(key, False))
        for key in ("has_spring", "has_hinge", "has_slide")
    )
    has_parameter_contrast = (
        len(metrics.get("mass_levels_present", ())) > 1
        or len(metrics.get("friction_levels_present", ())) > 1
        or len(metrics.get("restitution_levels_present", ())) > 1
    )

    return {
        "forward": bool(
            metrics.get("video_complete", False)
            and n_dyn >= 1
            and max_disp >= _LOW_MOTION_THRESHOLD
        ),
        "inverse": bool(
            metrics.get("video_complete", False)
            and n_dyn >= 1
            and max_disp >= _LOW_MOTION_THRESHOLD
            and max_speed >= 0.10
        ),
        "parameter": bool(
            metrics.get("video_complete", False)
            and n_dyn >= 2
            and has_interaction
            and has_parameter_contrast
        ),
    }


def _tags(metrics: dict[str, Any], task_suitability: dict[str, bool]) -> tuple[str, ...]:
    tags: list[str] = []
    n_dyn = int(metrics.get("dynamic_body_count", 0))
    if n_dyn == 1:
        tags.append("single_body")
    elif n_dyn == 2:
        tags.append("two_body")
    elif n_dyn > 2:
        tags.append("multi_body")

    tags.append("collision" if metrics.get("has_collision", False) else "no_collision")
    if len(metrics.get("mass_levels_present", ())) > 1:
        tags.append("mass_contrast")
    if len(metrics.get("friction_levels_present", ())) > 1:
        tags.append("friction_contrast")
    if len(metrics.get("restitution_levels_present", ())) > 1:
        tags.append("restitution_contrast")
    for key, tag in (
        ("has_spring", "spring"),
        ("has_hinge", "hinge"),
        ("has_slide", "slide"),
    ):
        if metrics.get(key, False):
            tags.append(tag)
    tags.append(
        "high_motion"
        if float(metrics.get("max_displacement", 0.0)) >= _GOOD_MOTION_DISP
        else "low_motion"
    )
    for mode, ok in task_suitability.items():
        if ok:
            tags.append(f"{mode}_suitable")
    return tuple(tags)


def _clamp(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


__all__ = ["evaluate_quality"]
