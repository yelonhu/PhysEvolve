"""Scene template: ``camera_depth_closest_object`` (camera-relative depth).

Depth items ask "Which object is the camera closest to?". This is a
PURE-GEOMETRY verifier: the executor already hands the
template the frozen ``SceneDef`` (body positions + the rendered camera's eye
``CameraDef.pos``), so the ground truth is simply the candidate body with the
min (or max, for "farthest") Euclidean distance to the camera eye -- no depth
rendering required.

The scene is intentionally STATIC (a few coloured cubes a Solver can see from a
fixed camera). Because nothing moves, the executor's triviality gate is skipped
for this scene_kind (see ``_STATIC_SCENE_KINDS`` in ``executor/pipeline.py``).

Ground truth is by body identity. Dedicated scenes use colour-bearing body names;
dynamics bypass scenes fall back to visible colour plus primitive shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from physevolve.skills.scene_templates import (
    BranchSpec,
    GeneratedVQA,
    SceneTemplate,
)

if TYPE_CHECKING:
    from physevolve.engine.schema import SceneDef
    from physevolve.executor.state_view import SimStateView


_SCENE_KIND = "camera_depth_closest_object"
_FAMILY = "camera_depth_closest_object"
# The two closest (or farthest) candidates must differ in camera distance by at
# least this margin (m) so the answer is visually unambiguous.
_MARGIN = 0.15

_PHRASINGS: Mapping[str, str] = {
    "static": "Which object is the camera closest to?",
    "farthest": "Which object is the camera farthest from?",
}


def _prettify(name: str) -> str:
    return name.replace("_", " ")


def _candidate_option(
    name: str,
    body_colors: Mapping[str, str],
    scene_def: "SceneDef",
) -> str:
    """Return a visible label for a candidate body.

    Dedicated camera-depth scenes use colour-bearing body names such as
    ``red_cube``. Dynamics bypass scenes may use neutral role names, so fall
    back to visible colour plus primitive shape there.
    """

    pretty_name = _prettify(name)
    color = str(body_colors.get(name, "") or "").strip().lower()
    if not color or color in pretty_name.lower().split():
        return f"The {pretty_name}"
    body = next((item for item in scene_def.bodies if item.name == name), None)
    if body is None:
        return f"The {pretty_name}"
    shape = {
        "box": "box",
        "sphere": "sphere",
        "cylinder": "cylinder",
        "capsule": "capsule",
    }.get(body.geom_type.value, "object")
    return f"The {color} {shape}"


def _candidates(manifest: Mapping[str, object]) -> list[str]:
    raw = manifest.get("candidate_bodies")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(
            "manifest['candidate_bodies'] must be a list of 4 body names"
        )
    out = []
    for c in raw:
        if not isinstance(c, str) or not c:
            raise ValueError(f"candidate_bodies entry {c!r} must be a body name")
        out.append(c)
    if len(set(out)) != 4:
        return out  # duplicates caught in validation below
    return out


def _phrasing(manifest: Mapping[str, object]) -> str:
    ph = manifest.get("phrasing", "static")
    if ph not in _PHRASINGS:
        raise ValueError(
            f"manifest['phrasing']={ph!r} not in {sorted(_PHRASINGS)}"
        )
    return str(ph)


def _camera_eye(
    manifest: Mapping[str, object], scene_def: "SceneDef"
) -> tuple[float, float, float]:
    """Return the queried camera's eye position.

    Uses ``manifest['camera']`` if given, else the FIRST camera -- which is the
    one folded into the Solver's video (camera 0). Every frozen scene has at
    least the auto-default camera, so ``cameras[0]`` is always present.
    """

    cam_name = manifest.get("camera")
    cams = list(scene_def.cameras)
    if not cams:
        raise ValueError("scene has no camera to measure depth from")
    if isinstance(cam_name, str) and cam_name:
        for cam in cams:
            if cam.name == cam_name:
                return tuple(float(x) for x in cam.pos)  # type: ignore[return-value]
        raise ValueError(f"camera {cam_name!r} was not created by build_scene")
    return tuple(float(x) for x in cams[0].pos)  # type: ignore[return-value]


def _distances(
    candidates: Sequence[str],
    eye: tuple[float, float, float],
    scene_def: "SceneDef",
) -> list[float]:
    by_name = {b.name: b for b in scene_def.bodies}
    out = []
    for name in candidates:
        body = by_name.get(name)
        if body is None:
            raise ValueError(f"candidate body {name!r} was not created by build_scene")
        p = body.pos
        out.append(
            ((p[0] - eye[0]) ** 2 + (p[1] - eye[1]) ** 2 + (p[2] - eye[2]) ** 2)
            ** 0.5
        )
    return out


def _validate_manifest(
    manifest: Mapping[str, object], scene_def: "SceneDef"
) -> Optional[str]:
    if manifest.get("scene_kind") != _SCENE_KIND:
        return f"scene_kind must be {_SCENE_KIND!r}"
    families = manifest.get("families")
    if families is not None:
        if not isinstance(families, list) or set(map(str, families)) - {_FAMILY}:
            return f"families must be a subset of [{_FAMILY!r}]"
    try:
        candidates = _candidates(manifest)
        phrasing = _phrasing(manifest)
        eye = _camera_eye(manifest, scene_def)
    except ValueError as exc:
        return str(exc)
    if len(set(candidates)) != 4:
        return "candidate_bodies must be 4 distinct names"
    by_name = {b.name: b for b in scene_def.bodies}
    for name in candidates:
        if name not in by_name:
            return f"candidate body {name!r} was not created by build_scene"
    dists = _distances(candidates, eye, scene_def)
    ordered = sorted(dists) if phrasing == "static" else sorted(dists, reverse=True)
    if abs(ordered[0] - ordered[1]) < _MARGIN:
        return (
            "the two extreme candidates are within "
            f"{_MARGIN} m of the camera (ambiguous closest/farthest)"
        )
    return None


def _required_branches(_manifest: Mapping[str, object]) -> list[BranchSpec]:
    return []


def _generate_questions(
    manifest: Mapping[str, object],
    _baseline_view: "SimStateView",
    _branch_views: Mapping[str, "SimStateView"],
    body_colors: Mapping[str, str],
    scene_def: "SceneDef",
) -> Sequence[GeneratedVQA]:
    candidates = _candidates(manifest)
    phrasing = _phrasing(manifest)
    eye = _camera_eye(manifest, scene_def)
    dists = _distances(candidates, eye, scene_def)
    # GT by identity: nearest for "static", farthest for "farthest".
    if phrasing == "static":
        correct_idx = min(range(4), key=lambda i: dists[i])
    else:
        correct_idx = max(range(4), key=lambda i: dists[i])
    options = [_candidate_option(name, body_colors, scene_def) for name in candidates]
    return [
        GeneratedVQA(
            question=_PHRASINGS[phrasing],
            options=options,
            correct_idx=int(correct_idx),
            family=_FAMILY,
            phrasing=phrasing,
            spec={"candidate_bodies": list(candidates)},
        )
    ]


_MANIFEST_SCHEMA_DOC = """\
  SCENE_MANIFEST = {
      "scene_kind": "camera_depth_closest_object",
      "phrasing": "static",                          # "static" (closest) | "farthest"
      "candidate_bodies": [<4 distinct visible body names>],
      "camera": "<camera name>",                     # optional; defaults to the
                                                     #   rendered camera (cam0)
  }
"""

_SCENE_CONSTRAINTS = """\
  - Place 4 clearly-coloured, well-separated bodies (e.g. red_cube / blue_cube /
    green_cube / yellow_cube) at DIFFERENT distances from the camera; they may
    be fixed (the scene is static - it is a spatial-depth question).
  - Add ONE camera with api.add_camera(name, pos, target) looking at the bodies;
    that camera's eye position is what "closest/farthest to the camera" measures.
  - Make the closest (or farthest) body unambiguous: the two extreme candidates
    must differ in camera distance by clearly more than 0.15 m.
  - Colour-name every candidate with matching rgba so the option text
    ("the red cube") matches the video.
"""

_FEWSHOT = """\
Plan: four coloured cubes sit at different distances from a fixed camera; the
red cube is nearest. The executor measures camera-to-body distance from the
scene geometry, so the answer comes from the layout, not this note.
<propose>
def build_scene(api):
    api.add_plane(name="ground", pos=(0.0, 0.0, 0.0), size=(5.0, 5.0),
                  rgba=(0.2, 0.2, 0.2, 1.0), friction=0.5)
    api.add_box("red_cube", pos=(0.0, -1.0, 0.15), size=(0.12, 0.12, 0.12),
                mass=1.0, rgba=(0.9, 0.15, 0.15, 1.0), friction=0.5,
                restitution=0.0, fixed=True)
    api.add_box("green_cube", pos=(0.6, 0.2, 0.15), size=(0.12, 0.12, 0.12),
                mass=1.0, rgba=(0.2, 0.75, 0.25, 1.0), friction=0.5,
                restitution=0.0, fixed=True)
    api.add_box("blue_cube", pos=(-0.7, 0.5, 0.15), size=(0.12, 0.12, 0.12),
                mass=1.0, rgba=(0.2, 0.3, 0.9, 1.0), friction=0.5,
                restitution=0.0, fixed=True)
    api.add_box("yellow_cube", pos=(0.1, 1.2, 0.15), size=(0.12, 0.12, 0.12),
                mass=1.0, rgba=(0.95, 0.9, 0.2, 1.0), friction=0.5,
                restitution=0.0, fixed=True)
    api.add_camera("cam0", pos=(0.0, -2.6, 1.1), target=(0.0, 0.0, 0.15))
    api.set_duration(1.0)

SCENE_MANIFEST = {
    "scene_kind": "camera_depth_closest_object",
    "phrasing": "static",
    "candidate_bodies": ["red_cube", "green_cube", "blue_cube", "yellow_cube"],
    "camera": "cam0",
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
