"""Scene template: ``motion_lab`` (surface/ramp-centric, one scene -> many VQAs).

Bodies launched, dropped, or transferred through ramps / surfaces. From ONE
simulation the template auto-derives every declared and feasible family:

* ``motion_state_change``                  - how a queried body's speed changes.
* ``friction_or_restitution_identification`` - which of two matched bodies has
  the higher friction (slides less) or restitution (bounces higher).

Same unification contract as :mod:`collision_arena`: a role-based manifest, a
declared SUBSET of the family menu, and per-family ground-truth functions that
used to live in standalone typed skill cards. The Proposer never authors
``correct_idx``.

Reserved slots: new lane/surface families plug in by adding a handler to
``_FAMILY_HANDLERS`` plus a menu entry.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Sequence

from physevolve.skills.scene_templates import (
    BranchSpec,
    GeneratedVQA,
    SceneTemplate,
    load_prompt_fewshot_examples,
)
from physevolve.skills.scene_templates.render_utils import prettify

if TYPE_CHECKING:
    from physevolve.engine.schema import SceneDef
    from physevolve.executor.state_view import SimStateView


_SCENE_KIND = "motion_lab"

_FAMILIES: tuple[str, ...] = (
    "motion_state_change",
    "friction_or_restitution_identification",
)
_MENU: frozenset[str] = frozenset(_FAMILIES)

# motion_state_change buckets (locked A/B/C/D order).
_MOTION_OPTIONS: tuple[tuple[str, str], ...] = (
    ("stop", "It comes to a stop"),
    ("slow_down", "It slows down but keeps moving"),
    ("speed_up", "It speeds up"),
    ("same_speed", "It continues at roughly the same speed"),
)
_MOTION_TO_IDX: Mapping[str, int] = {
    key: idx for idx, (key, _) in enumerate(_MOTION_OPTIONS)
}
_SPEED_PROFILE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("accelerate", "It accelerates."),
    ("decelerate", "It decelerates."),
    ("accelerate_then_decelerate", "It first accelerates and then decelerates."),
    ("constant", "It remains constant."),
)
_SPEED_PROFILE_TO_IDX: Mapping[str, int] = {
    key: idx for idx, (key, _) in enumerate(_SPEED_PROFILE_OPTIONS)
}
_STOP_THRESHOLD = 0.10
_DELTA_THRESHOLD = 0.30
_PAIRWISE_SPEED_GAP = 0.10

_VALID_ATTRIBUTES = frozenset({"friction", "restitution"})
_COMPARE_OPTION_KEYS: tuple[str, ...] = ("a", "b", "same", "cannot_determine")


def _speed(vec: Sequence[float]) -> float:
    return math.sqrt(sum(float(c) * float(c) for c in vec))


# ─── role parsing ─────────────────────────────────────────────────────


def _motion_target(manifest: Mapping[str, object]) -> str:
    raw = manifest.get("motion_target")
    if not isinstance(raw, str) or not raw:
        raise ValueError("manifest['motion_target'] must be a body name")
    return raw


def _compare_pair(
    manifest: Mapping[str, object],
) -> tuple[str, str, str, float, float]:
    raw = manifest.get("compare_pair")
    if not isinstance(raw, dict):
        raise ValueError("manifest['compare_pair'] must be a dict")
    a, b = raw.get("a"), raw.get("b")
    if not isinstance(a, str) or not a or not isinstance(b, str) or not b:
        raise ValueError("compare_pair needs body-name strings 'a' and 'b'")
    if a == b:
        raise ValueError("compare_pair 'a' and 'b' must differ")
    attribute = raw.get("attribute")
    if not isinstance(attribute, str) or attribute not in _VALID_ATTRIBUTES:
        raise ValueError("compare_pair['attribute'] must be 'friction' or 'restitution'")
    levels = raw.get("levels")
    if not isinstance(levels, dict):
        raise ValueError("compare_pair['levels'] must be {'a': float, 'b': float}")
    try:
        la, lb = float(levels["a"]), float(levels["b"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("compare_pair['levels'] requires 'a' and 'b' floats") from exc
    if la < 0 or lb < 0:
        raise ValueError("compare_pair levels must be non-negative")
    return a, b, attribute, la, lb


def _speed_pair_names(manifest: Mapping[str, object]) -> Optional[tuple[str, str]]:
    raw = manifest.get("speed_pair")
    if raw is None:
        raw = manifest.get("compare_pair")
    if not isinstance(raw, dict):
        return None
    a, b = raw.get("a"), raw.get("b")
    if not isinstance(a, str) or not a or not isinstance(b, str) or not b:
        raise ValueError("speed_pair needs body-name strings 'a' and 'b'")
    if a == b:
        raise ValueError("speed_pair 'a' and 'b' must differ")
    return a, b


def _families(manifest: Mapping[str, object]) -> list[str]:
    raw = manifest.get("families")
    if not isinstance(raw, list) or not raw:
        raise ValueError("manifest['families'] must be a non-empty list")
    out = []
    for f in raw:
        if f not in _MENU:
            raise ValueError(
                f"manifest['families'] has family {f!r} not served by "
                f"{_SCENE_KIND!r} (menu: {sorted(_MENU)})"
            )
        out.append(str(f))
    return out


# ─── per-family handlers ──────────────────────────────────────────────


def _classify_motion(view: "SimStateView", body: str) -> str:
    s_init = _speed(view[body].initial_vel)
    s_final = _speed(view[body].final_vel)
    if s_final < _STOP_THRESHOLD:
        return "stop"
    if s_final - s_init > _DELTA_THRESHOLD:
        return "speed_up"
    if s_init - s_final > _DELTA_THRESHOLD:
        return "slow_down"
    return "same_speed"


def _classify_speed_profile(view: "SimStateView", body: str) -> str:
    """Classify the speed profile over the clip."""

    state = view[body]
    s_init = _speed(state.initial_vel)
    s_final = _speed(state.final_vel)
    s_max = float(state.max_speed)
    accelerated = s_max - s_init > _DELTA_THRESHOLD
    slowed_after_peak = s_max - s_final > _DELTA_THRESHOLD
    if accelerated and slowed_after_peak:
        return "accelerate_then_decelerate"
    if s_final - s_init > _DELTA_THRESHOLD:
        return "accelerate"
    if s_init - s_final > _DELTA_THRESHOLD:
        return "decelerate"
    return "constant"


def _speed_relation(left_speed: float, right_speed: float) -> int:
    """Return 1 if left is faster, -1 if right is faster, 0 if tied."""

    if left_speed > right_speed + _PAIRWISE_SPEED_GAP:
        return 1
    if right_speed > left_speed + _PAIRWISE_SPEED_GAP:
        return -1
    return 0


def _pairwise_speed_question(
    body_a: str,
    body_b: str,
    view: "SimStateView",
) -> Optional[GeneratedVQA]:
    """Build the relationships/motion pairwise-speed phrasing."""

    a0, b0 = _speed(view[body_a].initial_vel), _speed(view[body_b].initial_vel)
    af, bf = _speed(view[body_a].final_vel), _speed(view[body_b].final_vel)
    initial_relation = _speed_relation(a0, b0)
    final_relation = _speed_relation(af, bf)
    a_label, b_label = prettify(body_a), prettify(body_b)
    if initial_relation and final_relation and initial_relation != final_relation:
        first_fast, later_fast = (
            (a_label, b_label) if initial_relation > 0 else (b_label, a_label)
        )
        crossover = (
            f"Initially, the {first_fast} moves faster, but the {later_fast} "
            "eventually surpasses it."
        )
        correct = 3
    else:
        crossover = (
            "Initially, one object moves faster, but the other eventually "
            "surpasses it."
        )
        if final_relation == 0:
            correct = 0
        elif final_relation > 0:
            correct = 1
        else:
            correct = 2
    options = [
        "Both objects have the same speed",
        f"The {a_label} consistently moves faster than the {b_label}.",
        f"The {b_label} consistently moves faster than the {a_label}.",
        crossover,
    ]
    semantics = ("same", "faster_a", "faster_b", "crossover")
    return GeneratedVQA(
        question=prettify(
            f"What is the relationship between the speeds of the {body_a} and "
            f"the {body_b} after they begin to move?"
        ),
        options=options,
        correct_idx=correct,
        family="motion_state_change",
        phrasing="pairwise_speed",
        spec={
            "body_a": body_a,
            "body_b": body_b,
            "answer_semantic": semantics[correct],
        },
    )


def _state_change(
    manifest: Mapping[str, object],
    view: "SimStateView",
    body_colors: Optional[Mapping[str, str]] = None,
) -> Optional[GeneratedVQA]:
    variants = _state_change_variants(manifest, view, body_colors)
    return variants[0] if variants else None


def _motion_label(name: str, colors: Mapping[str, str]) -> str:
    color = colors.get(name)
    if color:
        return f"{color} ball"
    return name.replace("_", " ")


def _state_change_variants(
    manifest: Mapping[str, object],
    view: "SimStateView",
    body_colors: Optional[Mapping[str, str]] = None,
) -> list[GeneratedVQA]:
    body = _motion_target(manifest)
    if body not in view:
        return []
    if _speed(view[body].initial_vel) < _STOP_THRESHOLD:
        return []  # cannot judge motion change from (near) rest
    bucket = _classify_motion(view, body)
    speed_profile = _classify_speed_profile(view, body)
    options = [opt for _, opt in _MOTION_OPTIONS]
    speed_profile_options = [opt for _, opt in _SPEED_PROFILE_OPTIONS]
    label = _motion_label(body, body_colors or {})
    out = [
        GeneratedVQA(
            question=prettify(f"How does the {label}'s motion change during the clip?"),
            options=list(options),
            correct_idx=_MOTION_TO_IDX[bucket],
            family="motion_state_change",
            phrasing="forward",
            spec={"target_body": body, "answer_semantic": bucket},
        ),
        GeneratedVQA(
            question=prettify(
                f"How does the speed of the {label} change before it hits the "
                "ground?"
            ),
            options=list(speed_profile_options),
            correct_idx=_SPEED_PROFILE_TO_IDX[speed_profile],
            family="motion_state_change",
            phrasing="speed_profile",
            spec={"target_body": body, "answer_semantic": speed_profile},
        ),
    ]
    pair = _speed_pair_names(manifest)
    if pair is not None:
        a, b = pair
        if a in view and b in view:
            out.append(_pairwise_speed_question(a, b, view))
    return out


def _classify_compare(la: float, lb: float) -> str:
    if math.isclose(la, lb, rel_tol=0.05, abs_tol=1e-3):
        return "same"
    return "a" if la > lb else "b"


def _friction_or_restitution(
    manifest: Mapping[str, object], view: "SimStateView"
) -> Optional[GeneratedVQA]:
    a, b, attribute, la, lb = _compare_pair(manifest)
    if a not in view or b not in view:
        return None
    declared = _classify_compare(la, lb)

    if attribute == "friction":
        # Higher friction -> smaller final speed.
        sa, sb = _speed(view[a].final_vel), _speed(view[b].final_vel)
        if max(sa, sb) < 0.05 and min(sa, sb) < 0.05:
            return None  # both stopped; differential not observable
        if declared == "a" and not (sa < sb - 0.10):
            return None
        if declared == "b" and not (sb < sa - 0.10):
            return None
        if declared == "same" and abs(sa - sb) > 0.30:
            return None
    else:  # restitution: compare post-impact rebound apex
        pa, pb = view[a].rebound_height, view[b].rebound_height
        if max(pa, pb) < 0.05:
            return None
        if declared == "a" and not (pa > pb + 0.05):
            return None
        if declared == "b" and not (pb > pa + 0.05):
            return None
        if declared == "same" and abs(pa - pb) > 0.10:
            return None

    options = [
        prettify(f"The {a} has higher {attribute}"),
        prettify(f"The {b} has higher {attribute}"),
        prettify(f"The {a} and the {b} have roughly the same {attribute}"),
        "It cannot be determined from the video",
    ]
    question = prettify(f"Which has higher {attribute}, the {a} or the {b}?")
    return GeneratedVQA(
        question=question,
        options=options,
        correct_idx=_COMPARE_OPTION_KEYS.index(declared),
        family="friction_or_restitution_identification",
        phrasing="comparison",
        spec={"body_a": a, "body_b": b, "attribute": attribute,
              "levels": {"a": la, "b": lb}},
    )


_FAMILY_HANDLERS: Mapping[
    str, Callable[[Mapping[str, object], "SimStateView"], Optional[GeneratedVQA]]
] = {
    "motion_state_change": _state_change,
    "friction_or_restitution_identification": _friction_or_restitution,
}


# ─── validation ───────────────────────────────────────────────────────


def _validate_family_roles(
    family: str, manifest: Mapping[str, object], fixed_by_name: Mapping[str, bool]
) -> Optional[str]:
    def _dyn(name: str, label: str) -> Optional[str]:
        if name not in fixed_by_name:
            return f"{label} {name!r} was not created by build_scene"
        if fixed_by_name[name]:
            return f"{label} {name!r} must be a free dynamic body (fixed=False)"
        return None

    try:
        if family == "motion_state_change":
            err = _dyn(_motion_target(manifest), "motion_target")
            if err is not None:
                return err
            pair = _speed_pair_names(manifest)
            if pair is not None:
                for n in pair:
                    err = _dyn(n, "speed body")
                    if err is not None:
                        return err
            return None
        if family == "friction_or_restitution_identification":
            a, b, _attr, _la, _lb = _compare_pair(manifest)
            for n in (a, b):
                err = _dyn(n, "compare body")
                if err is not None:
                    return err
    except ValueError as exc:
        return str(exc)
    return None


def _validate_manifest(
    manifest: Mapping[str, object], scene_def: "SceneDef"
) -> Optional[str]:
    if manifest.get("scene_kind") != _SCENE_KIND:
        return f"scene_kind must be {_SCENE_KIND!r}"
    try:
        families = _families(manifest)
    except ValueError as exc:
        return str(exc)
    fixed_by_name = {b.name: bool(b.fixed) for b in scene_def.bodies}
    for family in families:
        reason = _validate_family_roles(family, manifest, fixed_by_name)
        if reason is not None:
            return reason
    return None


def _required_branches(_manifest: Mapping[str, object]) -> list[BranchSpec]:
    return []


def _generate_questions(
    manifest: Mapping[str, object],
    baseline_view: "SimStateView",
    _branch_views: Mapping[str, "SimStateView"],
    body_colors: Mapping[str, str],
    _scene_def: "SceneDef",
) -> Sequence[GeneratedVQA]:
    families = _families(manifest)
    out: list[GeneratedVQA] = []
    for family in _FAMILIES:
        if family not in families:
            continue
        if family == "motion_state_change":
            out.extend(_state_change_variants(manifest, baseline_view, body_colors))
            continue
        vqa = _FAMILY_HANDLERS[family](manifest, baseline_view)
        if vqa is not None:
            out.append(vqa)
    return out


def _color_variant_bodies(manifest: Mapping[str, object]) -> list[str]:
    """Bodies whose rgba-derived colour appears in motion questions."""

    bodies: list[str] = []
    target = manifest.get("motion_target")
    if isinstance(target, str) and target:
        bodies.append(target)
    pair = manifest.get("speed_pair")
    if isinstance(pair, dict):
        for slot in ("a", "b"):
            name = pair.get(slot)
            if isinstance(name, str) and name:
                bodies.append(name)
    compare = manifest.get("compare_pair")
    if isinstance(compare, dict):
        for slot in ("a", "b"):
            name = compare.get(slot)
            if isinstance(name, str) and name:
                bodies.append(name)
    return list(dict.fromkeys(bodies))


# ─── prompt-facing docs + fewshot ─────────────────────────────────────

_MANIFEST_SCHEMA_DOC = """\
  SCENE_MANIFEST = {
      "scene_kind": "motion_lab",
      # Roles (declare only those your chosen families need):
      "motion_target": "<free dynamic body>",    # motion_state_change
      "speed_pair": {"a": "<body>", "b": "<body>"},  # optional pairwise-speed stem
      "compare_pair": {"a": "<body>", "b": "<body>",   # friction_or_restitution_identification
                       "attribute": "friction|restitution",
                       "levels": {"a": <float>, "b": <float>}},  # may be separate lanes
                                                                # or one connected apparatus
      "families": [<subset of the families this scene supports>],
  }
"""

_SCENE_CONSTRAINTS = """\
  - Every queried body MUST be a FREE DYNAMIC body (fixed=False); it slides /
    bounces on the ground plane, ramp, deck, or a connected apparatus.
  - Surface boxes are thin plates/decks/ramps, not tall blocks. Every `size` component is a half-extent and must be <= 1.0; do not write size=(2.0, ...).
  - Never spawn matched dynamic bodies at the same position. Use separate lanes
    or clear x/y spacing with visible clearance from each other and from the
    ramp/deck at t = 0; do not place spheres at the ramp center or embedded in
    a plate.
  - A valid scene needs a free dynamic body with initial motion or a gravity path,
    duration, and a camera; all-fixed or no-motion scenes are trivial.
  - motion_state_change: the queried body must START with speed >= 0.10 m/s and
    end UNAMBIGUOUSLY in one bucket (stop / slow down / speed up / same speed).
    The speed-profile phrasing uses a separate text space:
    accelerates / decelerates / first accelerates then decelerates / remains
    constant, grounded by initial, max, and final speed.
    Optional speed_pair bodies can add a pairwise-speed relation question when
    both are visible free dynamic bodies; the answer should be same speed,
    consistently faster, or a visible crossover/surpass relation.
  - friction: two bodies must differ clearly in friction (one high, one low).
    They may run in separate lanes or appear in one connected scene, but the
    higher-friction body must visibly travel less / slow more. Use nonzero
    identical initial velocity unless a clear downhill gravity path moves both;
    at least one body should remain visibly farther or faster so the comparison
    is not "both stopped".
  - restitution: DROP two identical bodies from the SAME height onto the ground,
    with a large restitution contrast and enough duration so the bounce (rebound
    apex) is captured.
  - If the Required Family is friction_or_restitution_identification, the
    manifest must include compare_pair and the video must show a visible
    two-body contrast, not only motion_target.
  - You may declare both families for one scene if the same video cleanly shows
    both motion change and attribute contrast.
"""

_AUTHORING_PLAYBOOK = """\
### Playbook: motion_lab

Motion-state recipe:
  - The target must start moving: initial speed >= 0.10 m/s.
  - To make "slow down", use high friction or a contact chain that drains speed.
  - To make "speed up", use gravity on a ramp or a downhill drop.
  - To make "stop", combine moderate initial speed with high friction or a stop
    wall. Avoid borderline final speeds near 0.10-0.20 m/s.
  - Avoid all-clip constant-speed coasting; the target should visibly accelerate,
    slow, or stop so the motion change is worth asking.

Friction recipe:
  - Matched bodies with nonzero equal starting speed + different friction separate
    because higher friction loses more energy: high friction -> shorter travel
    and lower final speed.
  - Do not encode the answer by giving one body a bigger initial velocity.
  - Matched bodies must not start at the same coordinates. Put them in parallel
    lanes or a connected apparatus with explicit positive clearance. Spawn them
    above/beside the surface, not embedded in the ramp/deck.
  - Tune duration/friction so the high-friction body slows clearly while the
    low-friction body remains visibly farther or faster at the end.

Restitution recipe:
  - Drop matched bodies from the same height; higher restitution rebounds higher.
  - Use duration >= 1.5 s so the rebound apex is visible in the 32-frame video.

Format-only scaffold (invent names and exact coordinates):
  - Bodies: ground, one fixed ramp/deck/lane pair, one free motion_target, and
    compare_pair bodies when declaring friction/restitution identification.
  - Dynamics: set initial velocity for motion_target or place it on a downhill
    gravity path; choose duration/camera so speed change or bounce is visible.
  - Manifest shape: "motion_target", "compare_pair" when required with true
    levels, and only the families supported by visible evidence.
"""

_AUTHORING_SKELETON = """\
### Topology Skeleton: motion_lab

Plan: one video shows a queried body's clear speed change, and optionally a
matched pair that reveals friction or restitution by different outcomes.

Build order (invent all names, coordinates, sizes, and parameters):
  1. Ground plus one fixed surface: ramp, deck, stop wall, or separated lanes.
  2. A free dynamic motion_target that starts moving and visibly changes speed.
  3. Compare_pair bodies when required, with matched geometry, separated starting
     positions, and matched starting speeds/heights.
  4. A camera framing the whole path so acceleration, slowing, stop, or bounce
     is visible across the 32 frames.

Anti-copy: choose either a ramp/contact/stop mechanism or an attribute pair as
needed. Do NOT paste the full ramp-plus-lane template or module fragments.
"""

_GOLDEN_DESIGN_CARD = """\
## Design Target: motion_lab

- A strong scene is a causal apparatus, not one isolated ball: ramp gravity,
  contact transfer, and surface properties explain the speed change.
- Attribute questions need matched bodies whose different outcomes reveal
  friction or restitution rather than initial-speed bias.
- Variation axes: ramp/deck layout, pad placement, stop wall, matched lanes,
  attribute levels, transfer body, duration, and camera framing.
"""

_MAIN_TRAINING_WIRING_CARD = """\
## Golden Wiring Card: motion_lab

Distilled from the validated connected ramp/contact fewshot. This is not
runnable code; invent concrete names, coordinates, shapes, levels, and colours.

### Apparatus Wiring
  1. Add ground plus one fixed surface system: ramp, deck, stop wall, rough pad,
     or matched lanes. Surfaces should be thin plates/decks/ramps with small
     `size_z`, visibly connected or clearly parallel. Every box `size`
     component is a half-extent <= 1.0.
  2. Add one free dynamic `motion_target` that starts moving or follows a
     gravity path. Its speed must change enough to fit one bucket decisively,
     and speed-profile text should be supported by visible acceleration,
     deceleration, accelerate-then-decelerate, or near-constant speed evidence.
  3. For friction comparison, add two matched dynamic bodies with equal starting
     speed but different friction; higher friction must travel less or slow more.
     Use nonzero identical initial velocity unless the setup has a clear
     gravity-driven path for both bodies. The final frame should show a visible
     travel-distance or final-speed contrast, not two stopped bodies.
  4. For restitution comparison, drop matched bodies from the same height with
     different restitution and enough duration to show rebound apex.
  5. Use duration plus a wide negative-y camera so the Solver can see the whole
     path: launch, contact, slowing/stopping, or bounce. Do not zoom onto one
     body and hide the comparison lane.

### Manifest Wiring
  - `motion_target`: free dynamic body used for motion-state questions.
  - `compare_pair`: bodies `a` and `b`, `attribute` = friction or restitution,
    and true numeric `levels` matching authored material values.
  - `families`: declare comparison only when the video shows a two-body
    contrast; declare motion_state only when the target speed bucket is clear.
    Add `speed_pair` when the video supports a pairwise-speed relation such as
    same speed, one body consistently faster, or a crossover/surpass relation.

### Syntax Anchors
  - Tilted ramp:
      api.add_box("<ramp>", pos=(x, y, z), size=(sx, sy, sz), mass=m,
                  rgba=(r, g, b, a), friction=f, restitution=e, fixed=True)
      api.set_rotation("<ramp>", quaternion=(qw, qx, qy, qz))
  - Friction compare pair:
      api.add_sphere("<body_a>", pos=(x_a, y_a, z), radius=r, mass=m,
                     rgba=(r, g, b, a), friction=f_a, restitution=e,
                     fixed=False)
      api.add_sphere("<body_b>", pos=(x_b, y_b, z), radius=r, mass=m,
                     rgba=(r, g, b, a), friction=f_b, restitution=e,
                     fixed=False)
      api.set_velocity("<body_a>", linear=(vx, vy, vz),
                       angular=(wx, wy, wz))
      api.set_velocity("<body_b>", linear=(vx, vy, vz),
                       angular=(wx, wy, wz))
  - Restitution compare pair: same drop height, different `restitution`.
  - Camera:
      api.add_camera("cam0", pos=(cam_x, cam_y, cam_z),
                     target=(target_x, target_y, target_z))

### Validity Checks
  - No `size` component exceeds 1.0; avoid `size=(2.0, ...)`. Ramps/decks should
    be thin surfaces, not tall blocks or vertical walls pretending to be ramps.
  - Matched comparison bodies never start at the same position; use lanes or
    explicit positive clearance from each other and from the ramp/deck.
  - For friction comparison, matched bodies need nonzero identical initial
    velocity or a shared downhill gravity path and a visible final-speed /
    travel-distance contrast.
  - Avoid 90-degree / vertical ramps; use a gentle tilt or a flat lane unless
    the body is clearly above the surface and already moving.
  - Do not encode the answer through different launch speeds when the question
    asks friction/restitution.
  - Avoid borderline final speeds near stop/slow thresholds.

Variation axes: ramp angle, surface layout, stop wall, transfer body, lane
spacing, friction/restitution levels, duration, and camera.
"""

_FEWSHOT = """\
Plan: a low-friction green ball rolls down a fixed ramp under gravity, strikes
an amber slider at the dock entrance, and transfers motion to a high-friction
blue ball that only nudges forward near the stop wall. The same connected
apparatus supports both the green ball motion-change question and the
blue-vs-green friction comparison.
<propose>
def build_scene(api):
    api.add_plane(
        name="ground",
        pos=(0.0, 0.0, 0.0),
        size=(5.0, 5.0),
        rgba=(0.12, 0.12, 0.13, 1.0),
        friction=0.5,
    )

    # Flush contact surfaces: the wood deck is split around the rough pad so
    # the high-friction patch is embedded instead of raised above the table.
    api.add_box(
        "deck_west",
        pos=(-0.2725, 0.0, 0.010),
        size=(0.4275, 0.30, 0.010),
        mass=10.0,
        rgba=(0.42, 0.32, 0.22, 1.0),
        friction=0.3,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "ramp_surface",
        pos=(-0.95, 0.0, 0.215),
        size=(0.72, 0.22, 0.025),
        mass=10.0,
        rgba=(0.48, 0.36, 0.24, 1.0),
        friction=0.15,
        restitution=0.0,
        fixed=True,
    )
    api.set_rotation("ramp_surface", quaternion=(0.96593, 0, 0.25882, 0))

    api.add_box(
        "rough_pad",
        pos=(0.30, 0.0, 0.010),
        size=(0.145, 0.16, 0.010),
        mass=10.0,
        rgba=(0.18, 0.16, 0.14, 1.0),
        friction=0.3,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "deck_east",
        pos=(0.8725, 0.0, 0.010),
        size=(0.4275, 0.30, 0.010),
        mass=10.0,
        rgba=(0.42, 0.32, 0.22, 1.0),
        friction=0.15,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "dock_wall",
        pos=(1.29, 0.0, 0.120),
        size=(0.030, 0.20, 0.100),
        mass=10.0,
        rgba=(0.34, 0.37, 0.39, 1.0),
        friction=0.5,
        restitution=0.9,
        fixed=True,
    )

    api.add_sphere(
        "green_ball",
        pos=(-1.30, 0.0, 1.2),
        radius=0.09,
        mass=1.0,
        rgba=(0.22, 0.78, 0.32, 1.0),
        friction=0.05,
        restitution=0.9,
        fixed=False,
    )
    api.add_box(
        "amber_slider",
        pos=(-0.10, 0.0, 0.1),
        size=(0.105, 0.08, 0.09),
        mass=1.0,
        rgba=(0.95, 0.55, 0.15, 1.0),
        friction=0.05,
        restitution=0.9,
        fixed=False,
    )
    api.add_sphere(
        "blue_ball",
        pos=(0.70, 0.0, 0.110),
        radius=0.09,
        mass=1.2,
        rgba=(0.12, 0.28, 0.92, 1.0),
        friction=0.05,
        restitution=0.9,
        fixed=False,
    )

    # Down-ramp initial speed >= 0.10 m/s (30 deg ramp: +x, -z).
    api.set_velocity(
        "green_ball",
        linear=(0.10, 0.0, -0.16),
        angular=(0.0, 0.0, 0.0),
    )

    api.set_duration(3.5)
    api.add_camera("cam0", pos=(-0.20, -3.8, 0.75), target=(0.05, 0.0, 0.22))

SCENE_MANIFEST = {
    "scene_kind": "motion_lab",
    "motion_target": "green_ball",
    "compare_pair": {"a": "blue_ball", "b": "green_ball",
                     "attribute": "friction", "levels": {"a": 2.0, "b": 0.05}},
    "families": ["motion_state_change", "friction_or_restitution_identification"],
}
</propose>
"""

_FEWSHOT_EXAMPLES = load_prompt_fewshot_examples(
    (
        "03_motion_lab__v01.py",
        "03_motion_lab__v02.py",
        "03_motion_lab__v03.py",
        "03_motion_lab__v04.py",
        "03_motion_lab__v05.py",
        "03_motion_lab__v06.py",
    ),
)


def make_template() -> SceneTemplate:
    return SceneTemplate(
        scene_kind=_SCENE_KIND,
        families=_FAMILIES,
        validate_manifest=_validate_manifest,
        required_branches=_required_branches,
        generate_questions=_generate_questions,
        manifest_schema_doc=_MANIFEST_SCHEMA_DOC,
        scene_constraints=_SCENE_CONSTRAINTS,
        authoring_playbook=_AUTHORING_PLAYBOOK,
        authoring_skeleton=_AUTHORING_SKELETON,
        golden_design_card=_GOLDEN_DESIGN_CARD,
        main_training_wiring_card=_MAIN_TRAINING_WIRING_CARD,
        fewshot_example=_FEWSHOT_EXAMPLES[0],
        fewshot_examples=_FEWSHOT_EXAMPLES,
        color_variant_bodies=_color_variant_bodies,
    )


__all__ = ["make_template"]
