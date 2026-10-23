"""Inactive template for coarse motion-type classification.

The verifier distinguishes circular, pendulum, and oscillatory signatures from
SimStateView statistics. The template remains registered for tests and is
excluded from the active question-family vocabulary.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from physevolve.skills.scene_templates import (
    BranchSpec,
    GeneratedVQA,
    SceneTemplate,
)
from physevolve.skills.scene_templates.render_utils import prettify

if TYPE_CHECKING:
    from physevolve.engine.schema import SceneDef
    from physevolve.executor.state_view import SimStateView


_SCENE_KIND = "motion_phenomenon_classification"
_FAMILY = "motion_phenomenon_classification"

_OPTIONS: tuple[tuple[str, str], ...] = (
    ("linear", "Linear motion (a straight-line trajectory)"),
    ("uniform_circular", "Uniform circular motion (constant-radius orbit)"),
    ("pendulum", "Pendulum motion (swing about a pivot)"),
    ("oscillatory", "Oscillatory motion (back-and-forth on a spring)"),
)
_KEY_TO_IDX: Mapping[str, int] = {key: idx for idx, (key, _) in enumerate(_OPTIONS)}
_VALID_CLASSES = frozenset(_KEY_TO_IDX)


def _target(manifest: Mapping[str, object]) -> str:
    raw = manifest.get("motion_target")
    if not isinstance(raw, str) or not raw:
        raise ValueError("manifest['motion_target'] must be a body name")
    return raw


def _expected(manifest: Mapping[str, object]) -> str:
    raw = manifest.get("expected_class")
    if not isinstance(raw, str) or raw not in _VALID_CLASSES:
        raise ValueError(
            f"manifest['expected_class'] must be one of {sorted(_VALID_CLASSES)}"
        )
    return raw


def _consistent_with(view: "SimStateView", body: str, expected: str) -> Optional[str]:
    """Return None if the trajectory is coarsely consistent with ``expected``."""

    state = view[body]
    init_pos, final_pos = state.initial_pos, state.final_pos
    init_vel, final_vel = state.initial_vel, state.final_vel
    s_init = math.sqrt(sum(float(c) * float(c) for c in init_vel))
    s_final = math.sqrt(sum(float(c) * float(c) for c in final_vel))

    if expected == "linear":
        disp = [float(final_pos[i]) - float(init_pos[i]) for i in range(3)]
        disp_norm = math.sqrt(sum(c * c for c in disp))
        if disp_norm < 0.15:
            return "linear motion requires |displacement| >= 0.15 m"
        if s_final < 0.05:
            return "linear motion requires nonzero final speed"
        cos = sum(disp[i] * float(final_vel[i]) for i in range(3)) / (
            disp_norm * max(s_final, 1e-9)
        )
        if cos < 0.6:
            return f"linear motion requires alignment cos>=0.6 (got {cos:.2f})"
        return None

    if expected == "uniform_circular":
        if not (
            math.isclose(s_init, s_final, rel_tol=0.30, abs_tol=0.20)
            and s_init >= 0.30
        ):
            return "uniform_circular requires |s_final - s_init| small and s>=0.30"
        if s_init * s_final > 0:
            cos = sum(float(init_vel[i]) * float(final_vel[i]) for i in range(3)) / (
                s_init * s_final
            )
            if cos > 0.85:
                return "uniform_circular requires final_vel rotated from initial"
        return None

    if expected == "pendulum":
        if s_init > 0.40 or s_final > 0.40:
            return "pendulum requires small speeds at endpoints"
        dz = abs(float(final_pos[2]) - float(init_pos[2]))
        if dz < 0.05 and (
            abs(float(final_pos[0]) - float(init_pos[0]))
            + abs(float(final_pos[1]) - float(init_pos[1]))
            < 0.05
        ):
            return "pendulum requires noticeable swing displacement"
        return None

    if expected == "oscillatory":
        disp = math.sqrt(
            sum((float(final_pos[i]) - float(init_pos[i])) ** 2 for i in range(3))
        )
        if disp > 0.30:
            return "oscillatory requires bounded |displacement| (<= 0.30 m)"
        if max(s_init, s_final) < 0.10:
            return "oscillatory requires nonzero motion at some endpoint"
        return None

    return f"unknown expected_class {expected!r}"


def _families(manifest: Mapping[str, object]) -> list[str]:
    raw = manifest.get("families")
    if raw is None:
        return [_FAMILY]
    if not isinstance(raw, list) or set(map(str, raw)) - {_FAMILY}:
        return []  # nothing valid declared
    return [_FAMILY]


def _validate_manifest(
    manifest: Mapping[str, object], scene_def: "SceneDef"
) -> Optional[str]:
    if manifest.get("scene_kind") != _SCENE_KIND:
        return f"scene_kind must be {_SCENE_KIND!r}"
    try:
        target = _target(manifest)
        _expected(manifest)
    except ValueError as exc:
        return str(exc)
    fixed_by_name = {b.name: bool(b.fixed) for b in scene_def.bodies}
    if target not in fixed_by_name:
        return f"motion_target {target!r} was not created by build_scene"
    if fixed_by_name[target]:
        return f"motion_target {target!r} must be a free dynamic body (fixed=False)"
    return None


def _required_branches(_manifest: Mapping[str, object]) -> list[BranchSpec]:
    return []


def _generate_questions(
    manifest: Mapping[str, object],
    baseline_view: "SimStateView",
    _branch_views: Mapping[str, "SimStateView"],
    _body_colors: Mapping[str, str],
    _scene_def: "SceneDef",
) -> Sequence[GeneratedVQA]:
    target = _target(manifest)
    expected = _expected(manifest)
    if target not in baseline_view:
        return []
    if _consistent_with(baseline_view, target, expected) is not None:
        return []
    options = [opt for _, opt in _OPTIONS]
    question = prettify(
        f"Which phenomenon best describes the motion of the {target}?"
    )
    return [
        GeneratedVQA(
            question=question,
            options=options,
            correct_idx=_KEY_TO_IDX[expected],
            family=_FAMILY,
            phrasing="categorical",
            spec={"target_body": target, "expected_class": expected},
        )
    ]


_MANIFEST_SCHEMA_DOC = """\
  SCENE_MANIFEST = {
      "scene_kind": "motion_phenomenon_classification",
      "motion_target": "<free dynamic body>",
      "expected_class": "linear|uniform_circular|pendulum|oscillatory",
      "families": ["motion_phenomenon_classification"],
  }
"""

_SCENE_CONSTRAINTS = """\
  - FROZEN scene_kind: not part of active training.
  - motion_target MUST be a FREE DYNAMIC body whose motion is unambiguously the
    declared expected_class.
"""

_FEWSHOT = """\
Plan: a ball glides on a smooth surface with moderate speed - a clean linear case.
<propose>
def build_scene(api):
    api.add_sphere("orange_ball", pos=(-0.8, 0.0, 0.10), radius=0.08, mass=1.0,
                   rgba=(0.95, 0.55, 0.15, 1.0), friction=0.05,
                   restitution=0.0, fixed=False)
    api.set_velocity("orange_ball", linear=(1.4, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
    api.set_duration(1.5)

SCENE_MANIFEST = {
    "scene_kind": "motion_phenomenon_classification",
    "motion_target": "orange_ball",
    "expected_class": "linear",
    "families": ["motion_phenomenon_classification"],
}
</propose>
"""


def make_template() -> SceneTemplate:
    return SceneTemplate(
        scene_kind=_SCENE_KIND,
        families=(_FAMILY,),
        validate_manifest=_validate_manifest,
        required_branches=_required_branches,
        generate_questions=_generate_questions,
        manifest_schema_doc=_MANIFEST_SCHEMA_DOC,
        scene_constraints=_SCENE_CONSTRAINTS,
        fewshot_example=_FEWSHOT,
    )


__all__ = ["make_template"]
