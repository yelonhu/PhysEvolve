"""Collision scenes with a striker, target, and optional relation pair.

The executor constructs the feasible questions declared in SCENE_MANIFEST from
one simulation. Active families cover final pose and final relation. First-contact
and mass-comparison handlers remain available for tests but are excluded from
the Proposer menu.
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


_SCENE_KIND = "collision_arena"

_FAMILIES: tuple[str, ...] = (
    "collision_final_state",
    "collision_final_relation",
    "motion_attribution",
)
_MENU: frozenset[str] = frozenset(_FAMILIES)

# ─── locked option tables (the Solver sees these as A/B/C/D order) ─────

# Pose vocabulary for collision_final_state stems.
_POSE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("standing_upright", "Standing upright"),
    ("leaning", "Leaning"),
    ("lying_horizontally", "Lying horizontally"),
    ("floating", "Floating"),
)
_POSE_TO_IDX: Mapping[str, int] = {
    pose: idx for idx, (pose, _) in enumerate(_POSE_OPTIONS)
}

# Legacy motion-status keys kept for borderline speed checks.
_STATUS_OPTIONS: tuple[tuple[str, str], ...] = _POSE_OPTIONS
_STATUS_TO_IDX: Mapping[str, int] = _POSE_TO_IDX

_RELATIONS: tuple[tuple[str, str], ...] = (
    ("contact", "The {body_a} ends in contact with the {body_b}"),
    ("no_contact", "The {body_a} never touches the {body_b}"),
    ("pass_by", "The {body_a} passes near the {body_b} but does not touch"),
    (
        "vertical_separation",
        "The {body_a} ends well above or below the {body_b} without contact",
    ),
)
_RELATION_TO_IDX: Mapping[str, int] = {
    rel: idx for idx, (rel, _) in enumerate(_RELATIONS)
}
_PASS_BY_THRESHOLD = 0.20
_VERTICAL_SEPARATION_THRESHOLD = 0.30

_MASS_OPTION_KEYS: tuple[str, ...] = ("a", "b", "same", "cannot_determine")


# ─── small numeric helpers ────────────────────────────────────────────


def _speed(vec: Sequence[float]) -> float:
    return math.sqrt(sum(float(c) * float(c) for c in vec))


def _final_speed(view: "SimStateView", body: str) -> float:
    return _speed(view[body].final_vel)


def _delta_speed(view: "SimStateView", body: str) -> float:
    init = view[body].initial_vel
    final = view[body].final_vel
    return math.sqrt(sum((float(final[i]) - float(init[i])) ** 2 for i in range(3)))


def _euclid(p: Sequence[float], q: Sequence[float]) -> float:
    return math.sqrt(sum((float(p[i]) - float(q[i])) ** 2 for i in range(3)))


# ─── role parsing ─────────────────────────────────────────────────────


def _target(manifest: Mapping[str, object]) -> str:
    raw = manifest.get("target")
    if not isinstance(raw, str) or not raw:
        raise ValueError("manifest['target'] must be a body name")
    return raw


def _trigger(manifest: Mapping[str, object]) -> str:
    raw = manifest.get("trigger")
    if not isinstance(raw, str) or not raw:
        raise ValueError("manifest['trigger'] must be a body name")
    return raw


def _movers(manifest: Mapping[str, object]) -> list[str]:
    raw = manifest.get("movers")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(
            "manifest['movers'] must be a list of exactly 4 candidate body names"
        )
    out = []
    for m in raw:
        if not isinstance(m, str) or not m:
            raise ValueError(f"manifest['movers'] entry {m!r} must be a body name")
        out.append(m)
    return out


def _pair(manifest: Mapping[str, object], key: str) -> tuple[str, str, dict]:
    raw = manifest.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"manifest[{key!r}] must be a dict with 'a' and 'b'")
    a, b = raw.get("a"), raw.get("b")
    if not isinstance(a, str) or not a or not isinstance(b, str) or not b:
        raise ValueError(f"manifest[{key!r}] needs body-name strings 'a' and 'b'")
    if a == b:
        raise ValueError(f"manifest[{key!r}] 'a' and 'b' must differ")
    return a, b, raw


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


# ─── per-family handlers: (manifest, view) -> Optional[GeneratedVQA] ───
#
# Each handler returns None when its scene roles do not realise a clean,
# unambiguous question (the family is then silently skipped, exactly like
# throwing skips un-settled balls).


def _first_contact(
    manifest: Mapping[str, object], view: "SimStateView"
) -> Optional[GeneratedVQA]:
    target = _target(manifest)
    candidates = _movers(manifest)
    if target not in view or len(set(candidates)) != 4 or target in candidates:
        return None
    if any(b not in view for b in candidates):
        return None
    times = [view.time_of_collision(target, b) for b in candidates]
    finite = sorted(t for t in times if t is not None)
    if not finite:
        return None
    if len(finite) >= 2 and (finite[1] - finite[0]) < 2:
        return None  # ambiguous order under the 10-step aggregation window
    best_idx, best_t = -1, None
    for idx, t in enumerate(times):
        if t is None:
            continue
        if best_t is None or t < best_t:
            best_t, best_idx = t, idx
    if best_idx < 0:
        return None
    options = [prettify(f"The {b}") for b in candidates]
    question = prettify(f"Which object hits the {target} first?")
    return GeneratedVQA(
        question=question,
        options=options,
        correct_idx=best_idx,
        family="collision_first_contact",
        phrasing="forward",
        spec={"target_body": target, "candidate_bodies": candidates},
    )


def _body_phrase(name: str, colors: Mapping[str, str]) -> str:
    """Refer to a body by rgba-derived colour when available."""

    color = colors.get(name)
    if color:
        return f"{color} {name.split('_')[-1] if '_' in name else name}"
    return prettify(name.replace("_", " "))


def _classify_pose(view: "SimStateView", body: str) -> Optional[str]:
    """Map simulator final status to pose semantics."""

    if body not in view:
        return None
    b = view[body]
    status = b.final_status
    speed = _final_speed(view, body)
    if status == "ejected":
        # Airborne / out-of-workspace -> floating pose when clearly elevated.
        if float(b.final_pos[2]) > 0.35:
            return "floating"
        return None
    if status == "fallen":
        return "lying_horizontally"
    if status == "moving":
        if speed >= 0.10:
            return "leaning"
        return None
    if status == "stable":
        if speed > 0.20:
            return None
        return "standing_upright"
    return None


def _final_state(
    manifest: Mapping[str, object],
    view: "SimStateView",
    body_colors: Optional[Mapping[str, str]] = None,
) -> Optional[GeneratedVQA]:
    target = _target(manifest)
    if target not in view:
        return None
    pose = _classify_pose(view, target)
    if pose is None:
        return None
    speed = _final_speed(view, target)
    status = view[target].final_status
    if status == "moving" and speed < 0.10:
        return None
    if status == "stable" and speed > 0.20:
        return None
    colors = body_colors or {}
    color_word = colors.get(target, "")
    body_noun = "pillar" if "pillar" in target else "plate"
    if color_word:
        subject = f"{color_word} {body_noun}"
    else:
        subject = prettify(target.replace("_", " "))
    options = [opt for _, opt in _POSE_OPTIONS]
    question = prettify(
        f"Which phrase below can best describe the final pose of the {subject}?"
    )
    return GeneratedVQA(
        question=question,
        options=options,
        correct_idx=_POSE_TO_IDX[pose],
        family="collision_final_state",
        phrasing="forward",
        spec={"target_body": target, "answer_semantic": pose},
    )


def _classify_relation(view: "SimStateView", a: str, b: str) -> str:
    if view.collided(a, b):
        return "contact"
    pa, pb = view[a], view[b]
    min_dist = min(
        _euclid(pa.initial_pos, pb.initial_pos),
        _euclid(pa.final_pos, pb.final_pos),
    )
    final_dz = abs(float(pa.final_pos[2]) - float(pb.final_pos[2]))
    final_lateral = math.hypot(
        float(pa.final_pos[0]) - float(pb.final_pos[0]),
        float(pa.final_pos[1]) - float(pb.final_pos[1]),
    )
    if final_dz >= _VERTICAL_SEPARATION_THRESHOLD and final_lateral < 0.30:
        return "vertical_separation"
    if min_dist <= _PASS_BY_THRESHOLD:
        return "pass_by"
    return "no_contact"


def _final_relation(
    manifest: Mapping[str, object],
    view: "SimStateView",
    body_colors: Optional[Mapping[str, str]] = None,
) -> Optional[GeneratedVQA]:
    variants = _final_relation_variants(manifest, view, body_colors)
    return variants[0] if variants else None


def _final_relation_variants(
    manifest: Mapping[str, object],
    view: "SimStateView",
    body_colors: Optional[Mapping[str, str]] = None,
) -> list[GeneratedVQA]:
    a, b, raw = _pair(manifest, "relation_pair")
    expected = raw.get("expected")
    if not isinstance(expected, str) or expected not in _RELATION_TO_IDX:
        return []
    if a not in view or b not in view:
        return []
    if _classify_relation(view, a, b) != expected:
        return []  # declared relation not realised -> skip (no false GT)
    colors = body_colors or {}
    body_a_phrase = _body_phrase(a, colors)
    body_b_phrase = _body_phrase(b, colors)
    options = [
        prettify(opt.format(body_a=body_a_phrase, body_b=body_b_phrase))
        for _, opt in _RELATIONS
    ]
    base_spec = {
        "body_a": a,
        "body_b": b,
        "body_a_phrase": body_a_phrase,
        "body_b_phrase": body_b_phrase,
        "expected_relation": expected,
    }
    question_templates: tuple[tuple[str, str], ...] = (
        (
            "forward",
            f"What is the final interaction between the {body_a_phrase} "
            f"and the {body_b_phrase}?",
        ),
        (
            "encounter",
            f"What happens when the {body_a_phrase} encounters "
            f"the {body_b_phrase}?",
        ),
        (
            "between",
            f"What happens between the {body_a_phrase} and the {body_b_phrase}?",
        ),
        (
            "position",
            f"What is the final position of the {body_a_phrase} in relation "
            f"to the {body_b_phrase}?",
        ),
    )
    out = [
        GeneratedVQA(
            question=prettify(question),
            options=list(options),
            correct_idx=_RELATION_TO_IDX[expected],
            family="collision_final_relation",
            phrasing=phrasing,
            spec=dict(base_spec),
        )
        for phrasing, question in question_templates
    ]
    out.append(
        GeneratedVQA(
            question=prettify(
                f"Does the {body_a_phrase} collide with the {body_b_phrase}?"
            ),
            options=["Yes", "No", "Sometimes", "Uncertain"],
            correct_idx=0 if expected == "contact" else 1,
            family="collision_final_relation",
            phrasing="binary_collide",
            spec={
                **base_spec,
                "answer_semantic": "yes" if expected == "contact" else "no",
            },
        )
    )
    return out


def _attribution(
    manifest: Mapping[str, object],
    view: "SimStateView",
    body_colors: Optional[Mapping[str, str]] = None,
) -> Optional[GeneratedVQA]:
    trigger = _trigger(manifest)
    candidates = _movers(manifest)
    if trigger not in view or len(set(candidates)) != 4 or trigger in candidates:
        return None
    if any(b not in view for b in candidates):
        return None
    deltas = [_delta_speed(view, b) for b in candidates]
    collided = [view.collided(trigger, b) for b in candidates]
    eligible = sorted(
        ((deltas[i], i) for i in range(4) if collided[i] and deltas[i] > 0.30),
        reverse=True,
    )
    if not eligible:
        return None
    if len(eligible) >= 2 and (eligible[0][0] - eligible[1][0]) < 0.30:
        return None  # attribution ambiguous
    best_idx = eligible[0][1]
    colors = body_colors or {}
    options = [_body_phrase(b, colors) for b in candidates]
    options = [f"The {opt}" if not opt.lower().startswith("the ") else opt for opt in options]
    question = prettify(
        f"Which object experienced a change in motion as a result of "
        f"the {_body_phrase(trigger, colors)}?"
    )
    return GeneratedVQA(
        question=question,
        options=options,
        correct_idx=best_idx,
        family="motion_attribution",
        phrasing="attribution",
        spec={"trigger_body": trigger, "candidate_bodies": candidates},
    )


def _masses(raw: Mapping[str, object]) -> Optional[tuple[float, float]]:
    tm = raw.get("true_masses")
    if not isinstance(tm, dict):
        return None
    try:
        ma, mb = float(tm["a"]), float(tm["b"])
    except (KeyError, TypeError, ValueError):
        return None
    if ma <= 0 or mb <= 0:
        return None
    return ma, mb


def _classify_mass(ma: float, mb: float) -> str:
    if math.isclose(ma, mb, rel_tol=0.10, abs_tol=1e-9):
        return "same"
    return "a" if ma > mb else "b"


def _mass_comparison(
    manifest: Mapping[str, object], view: "SimStateView"
) -> Optional[GeneratedVQA]:
    a, b, raw = _pair(manifest, "mass_pair")
    masses = _masses(raw)
    if masses is None:
        return None
    ma, mb = masses
    if a not in view or b not in view:
        return None
    if not view.collided(a, b):
        return None
    da, db = _delta_speed(view, a), _delta_speed(view, b)
    if max(da, db) < 0.20:
        return None
    declared = _classify_mass(ma, mb)
    # Lighter body should experience the larger speed change (1D impulse approx).
    if declared == "a":
        if not (db > da * 1.10 or da < 0.30):
            return None
    elif declared == "b":
        if not (da > db * 1.10 or db < 0.30):
            return None
    else:  # same
        if max(da, db) > 1.5 * min(da, db) + 0.20:
            return None
    options = [
        prettify(f"The {a} is heavier"),
        prettify(f"The {b} is heavier"),
        prettify(f"The {a} and the {b} have roughly the same mass"),
        "It cannot be determined from the video",
    ]
    question = prettify(f"Which is heavier, the {a} or the {b}?")
    return GeneratedVQA(
        question=question,
        options=options,
        correct_idx=_MASS_OPTION_KEYS.index(declared),
        family="mass_comparison_by_dynamics",
        phrasing="comparison",
        spec={"body_a": a, "body_b": b, "true_masses": {"a": ma, "b": mb}},
    )


_FAMILY_HANDLERS: Mapping[
    str, Callable[[Mapping[str, object], "SimStateView"], Optional[GeneratedVQA]]
] = {
    "collision_first_contact": _first_contact,
    "collision_final_state": _final_state,
    "collision_final_relation": _final_relation,
    "motion_attribution": _attribution,
    "mass_comparison_by_dynamics": _mass_comparison,
}


# ─── validation ───────────────────────────────────────────────────────


# Which roles each family REQUIRES, so a declared family without its roles is a
# clear manifest error (caught statically, before simulating).
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
        if family in ("collision_first_contact", "motion_attribution"):
            key = "target" if family == "collision_first_contact" else "trigger"
            anchor = _target(manifest) if key == "target" else _trigger(manifest)
            movers = _movers(manifest)
            if anchor in movers:
                return f"{key} {anchor!r} must not also be a mover"
            if len(set(movers)) != 4:
                return "movers must be 4 distinct body names"
            for label, names in ((key, [anchor]), ("mover", movers)):
                for n in names:
                    err = _dyn(n, label)
                    if err is not None:
                        return err
        elif family == "collision_final_state":
            err = _dyn(_target(manifest), "target")
            if err is not None:
                return err
        elif family == "collision_final_relation":
            a, b, raw = _pair(manifest, "relation_pair")
            if not isinstance(raw.get("expected"), str):
                return "relation_pair needs a string 'expected' relation"
            for n in (a, b):
                err = _dyn(n, "relation body")
                if err is not None:
                    return err
        elif family == "mass_comparison_by_dynamics":
            a, b, raw = _pair(manifest, "mass_pair")
            if _masses(raw) is None:
                return "mass_pair needs 'true_masses' with positive 'a' and 'b'"
            for n in (a, b):
                err = _dyn(n, "mass body")
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
    for family in _FAMILIES:  # stable order independent of manifest ordering
        if family not in families:
            continue
        if family == "collision_final_relation":
            out.extend(
                _final_relation_variants(manifest, baseline_view, body_colors)
            )
            continue
        if family == "collision_final_state":
            vqa = _final_state(manifest, baseline_view, body_colors)
        elif family == "motion_attribution":
            vqa = _attribution(manifest, baseline_view, body_colors)
        else:
            vqa = _FAMILY_HANDLERS[family](manifest, baseline_view)
        if vqa is not None:
            out.append(vqa)
    return out


def _color_variant_bodies(manifest: Mapping[str, object]) -> list[str]:
    """Bodies referenced by rgba-derived colour words in question text."""

    bodies: list[str] = []
    for key in ("target", "trigger"):
        raw = manifest.get(key)
        if isinstance(raw, str) and raw:
            bodies.append(raw)
    movers = manifest.get("movers")
    if isinstance(movers, list):
        bodies.extend(str(m) for m in movers if isinstance(m, str))
    pair = manifest.get("relation_pair")
    if isinstance(pair, dict):
        for slot in ("a", "b"):
            name = pair.get(slot)
            if isinstance(name, str) and name:
                bodies.append(name)
    return list(dict.fromkeys(bodies))


# ─── prompt-facing docs + fewshot ─────────────────────────────────────

_MANIFEST_SCHEMA_DOC = """\
  SCENE_MANIFEST = {
      "scene_kind": "collision_arena",
      # Roles (declare only those your chosen families need):
      "target": "<free dynamic body>",          # collision_final_state
      "trigger": "<free dynamic striker>",      # motion_attribution
      "movers": [<exactly 4 free dynamic body names>],  # attribution candidates
      "relation_pair": {"a": "<body>", "b": "<body>",   # collision_final_relation
                        "expected": "contact|no_contact|pass_by|vertical_separation"},
      "families": [<subset of the families this scene supports>],
  }
"""

_SCENE_CONSTRAINTS = """\
  - Every body the verifier reads (target / movers / trigger / relation_pair /
    members) MUST be a FREE DYNAMIC body (fixed=False).
  - A valid collision scene needs at least one moving trigger or gravity-driven
    joint/spring path that produces visible motion and contact during the clip.
  - Keep trigger/intermediate/target/movers inside the workspace with margin;
    do not place bodies at x=2.0, x=2.5, x=3.0, or on strict bounds.
  - Use staggered y offsets or clear x gaps. Do not place all dynamic bodies on
    one dense same-line stack; first-submit validity should not need geometry repair.
  - attribution: provide the trigger plus EXACTLY 4 candidate "movers"; only one
    candidate should have the largest motion change after colliding with the
    trigger (gap >= 0.30 m/s).
  - final_state: the target must end UNAMBIGUOUSLY upright / moving / toppled /
    knocked-out (avoid borderline drift speeds 0.10-0.20 m/s).
  - final_relation: place the pair so the declared 'expected' relation is the one
    the simulation actually realises (direct contact, offset pass-by/no-contact,
    or stacked vertical separation).
  - You may declare SEVERAL families for one scene - each feasible family adds a
    question about the SAME video (multiple questions per scene).
"""

_AUTHORING_PLAYBOOK = """\
### Playbook: collision_arena

Naming:
  - For motion_attribution, movers should be colour-named bodies with matching
    rgba (green_peg, blue_plate, orange_ball) because option text uses body names.
  - Other role names may be descriptive, but keep the visible rgba consistent
    with any colour word in the body name.

Causal chain:
  trigger -> intermediate body -> target / relation_pair.
  The trigger can be a hinged striker or a free dynamic pusher. The intermediate
  should sit just outside contact at t = 0, then collide after motion begins.
  The trigger must actually contact the intermediate during the clip; choose the
  hinge swing arc / pusher path so it cannot miss and swing through empty space.
  The trigger path should be non-axis-trivial: use a hinge arc, offset pusher
  lane, or staged contact chain rather than a stack of same-position boxes.
  Keep every trigger/intermediate/target gap positive at t = 0, so contacts are
    caused by motion rather than initial overlap. Avoid placing all dynamic bodies
    on the same centreline unless their offsets and sizes make the contact order
    unambiguous; prefer staggered y offsets or clear x gaps.

Motion-driving rule:
  - A hinged trigger must have non-zero angular motion or an off-rest pose that
    gravity will swing; a free pusher must have non-zero linear velocity.
  - API pattern only: `api.set_velocity("trigger", linear=(...), angular=(...))`
    with at least one non-zero component. Do not leave every dynamic body at
    zero velocity in a flat, no-contact layout.

Role rules:
  - Every manifest role (target, trigger, movers, relation_pair.a/b) must be
    fixed=False.
  - movers must contain exactly 4 distinct bodies. For attribution, only one
    candidate should get the dominant delta-speed after trigger contact; keep the
    next-largest delta at least 0.30 m/s lower.
  - Idle candidates should be visibly present but outside the strike lane with
    at least one body-size plus margin away from target/intermediate at t = 0.
  - Keep every role body inside |x| < 1.5 and |y| < 1.5 with margin; repair
    should not be required to make the initial layout legal.
  - relation_pair.expected must be what the simulation actually realizes:
    contact, no_contact, pass_by, or vertical_separation.

Format-only scaffold (invent names and exact coordinates):
  - Bodies: one free dynamic trigger, one intermediate, one target/relation body,
    and four distinct free dynamic movers.
  - Dynamics: trigger moves first, one candidate receives the dominant
    delta-speed, other movers stay visible but outside the strike lane.
  - Manifest shape: "trigger", "target", "movers", optional "relation_pair",
    and only the families this one impact chain actually supports.
"""

_AUTHORING_SKELETON = """\
### Topology Skeleton: collision_arena

Plan: a visible trigger transfers motion through one intermediate body into a
queried target, while the remaining candidate movers stay outside the lane.

Build order (invent all names, coordinates, sizes, and parameters):
  1. Ground plane and optional fixed supports or hinge post.
  2. One free dynamic trigger: either a hinged striker or a free pusher.
  3. One free dynamic intermediate body placed with positive t=0 clearance.
  4. One free dynamic target / relation body reached after the intermediate.
  5. Extra free dynamic candidate movers offset from the lane for attribution.

Anti-copy: this block is a topology checklist, not code. Do NOT paste names,
coordinates, or full fragments from Playbook, Physics Modules, or this block.
"""

_GOLDEN_DESIGN_CARD = """\
## Design Target: collision_arena

- A strong scene has a readable impact theater: guided trigger, staged transfer,
  and distractors that do not receive the dominant impulse.
- Final state and final relation should be visually decisive, not borderline or
  hidden behind scenery.
- Variation axes: trigger type, target shape, distractor placement, masses,
  restitution, collision lane, final relation, and camera.
"""

_MAIN_TRAINING_WIRING_CARD = """\
## Golden Wiring Card: collision_arena

Distilled from the validated hinged-impact fewshot. This is not runnable code;
invent concrete names, coordinates, shapes, masses, and colours.

### Apparatus Wiring
  1. Add ground and optional fixed supports for a guided trigger.
  2. Create one free dynamic trigger: either a hinged striker (`api.add_hinge`
     with parent "world") or a free pusher. It must have
     non-zero linear or angular motion, or a gravity-driven off-rest pose.
  3. Place an intermediate body just outside t=0 contact in the trigger path.
     The first contact must happen after motion begins, not from overlap.
  4. Put the target/relation body downstream so the impulse has a clear final
     effect: upright/moving/fallen/ejected or a decisive relation.
  5. Add four mover candidates for attribution. One receives the dominant
     delta-speed after the trigger chain; the other candidates stay visible but
     outside the main strike lane. Use staggered y offsets or clear x gaps, not
     one dense same-line dynamic stack.
  6. Use a wide negative-y camera that frames trigger, transfer, target, and
     candidate movers together; do not zoom onto only the first contact.

### Manifest Wiring
  - `trigger`: free dynamic body that initiates contact.
  - `target`: free dynamic body for final-state questions.
  - `movers`: exactly four distinct free dynamic candidates; include the body
    with the dominant motion change.
  - `relation_pair`: two distinct free dynamic bodies plus the relation the
    simulation actually realizes.
  - `families`: declare only the families whose required roles are wired.

### Syntax Anchors
  - Hinged trigger:
      api.add_capsule("<trigger>", pos=(x, y, z), radius=r, half_height=h,
                      mass=m, rgba=(r, g, b, a), friction=f, restitution=e,
                      fixed=False, axis="z")
      api.add_hinge("<hinge>", parent="world", child="<trigger>",
                    pos=(px, py, pz), axis=(ax, ay, az))
      api.set_velocity("<trigger>", linear=(vx, vy, vz),
                       angular=(wx, wy, wz))
  - Free pusher alternative:
      api.add_box("<trigger>", pos=(x, y, z), size=(sx, sy, sz),
                  mass=m, rgba=(r, g, b, a), friction=f, restitution=e,
                  fixed=False)
      api.set_velocity("<trigger>", linear=(vx, vy, vz),
                       angular=(wx, wy, wz))
  - Camera:
      api.add_camera("cam0", pos=(cam_x, cam_y, cam_z),
                     target=(target_x, target_y, target_z))

### Validity Checks
  - Positive t=0 clearance between trigger/intermediate/target/candidates.
  - Trigger/intermediate/target/movers stay inside |x| < 1.5 and |y| < 1.5 with
    margin; avoid x=2.0/2.5/3.0 layouts.
  - Mover candidates outside the strike lane are at least one body-size plus
    margin away from target/intermediate.
  - Trigger path must visibly contact the intermediate inside the clip.
  - Attribution is invalid if two movers receive similar delta-speed; keep the
    winner visually dominant.
  - Do not leave all dynamic bodies sleeping with zero velocity in a flat layout.
  - First-submit geometry should already be non-overlapping; do not rely on
    repair to separate dense bodies.

Variation axes: trigger type, hinge axis, impact lane, target shape, distractor
placement, mass/restitution, final relation, duration, and camera.
"""

# Golden = the owner-approved hinged striker chain. It declares the active
# collision_arena families only; frozen first_contact / mass comparison handlers
# remain below but are not shown to the Proposer.
_FEWSHOT = """\
Plan: a gray cantilever (gray_anchor riser + pivot_bracket + pivot_hub)
supports a y-axis pivot; the vertical capsule swing_striker swings down and
strikes the stationary brown_hammer, which knocks the orange ball toward the
blue panel.
<propose>
def build_scene(api):
    api.add_plane(
        name="ground",
        pos=(0.0, 0.0, 0.0),
        size=(5.0, 5.0),
        rgba=(0.12, 0.12, 0.13, 1.0),
        friction=0.5,
    )

    # Cantilever pivot: thin riser on the far left + bracket + pin (swing clears riser).
    _pivot = (-1.05, 0.0, 0.56)
    api.add_box(
        "gray_anchor",
        pos=(-1.28, 0.0, 0.30),
        size=(0.03, 0.03, 0.30),
        mass=10.0,
        rgba=(0.55, 0.55, 0.58, 1.0),
        friction=0.5,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "pivot_bracket",
        pos=(-1.135, 0.0, 0.56),
        size=(0.070, 0.04, 0.025),
        mass=10.0,
        rgba=(0.50, 0.50, 0.53, 1.0),
        friction=0.5,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "pivot_hub",
        pos=_pivot,
        size=(0.03, 0.06, 0.03),
        mass=10.0,
        rgba=(0.45, 0.45, 0.48, 1.0),
        friction=0.5,
        restitution=0.0,
        fixed=True,
    )

    # Vertical capsule arm (top near pivot); world hinge at bracket tip.
    api.add_capsule(
        "swing_striker",
        pos=(-1.214, 0.0, 0.30),
        radius=0.045,
        half_height=0.20,
        mass=12.0,
        rgba=(0.40, 0.40, 0.44, 1.0),
        friction=0.30,
        restitution=0.05,
        fixed=False,
        axis="z",
    )
    api.add_hinge(
        name="hammer_pivot",
        parent="world",
        child="swing_striker",
        pos=_pivot,
        axis=(0.0, 1.0, 0.0),
    )

    # Stationary brown target in the swing arc (+x of the pivot).
    api.add_box(
        "brown_hammer",
        pos=(-0.74, 0.0, 0.10),
        size=(0.18, 0.05, 0.07),
        mass=1.0,
        rgba=(0.46, 0.24, 0.10, 1.0),
        friction=0.00,
        restitution=0.28,
        fixed=False,
    )

    # Ball sits in the brown rebound lane toward the blue panel.
    api.add_sphere(
        "orange_ball",
        pos=(-0.38, 0.0, 0.110),
        radius=0.110,
        mass=1.0,
        rgba=(0.95, 0.52, 0.12, 1.0),
        friction=0.15,
        restitution=0.40,
        fixed=False,
    )

    api.add_box(
        "blue_plate",
        pos=(0.24, 0.0, 0.28),
        size=(0.065, 0.36, 0.28),
        mass=0.35,
        rgba=(0.12, 0.28, 0.92, 1.0),
        friction=0.50,
        restitution=0.00,
        fixed=False,
    )

    api.add_box(
        "purple_bar",
        pos=(0.56, 0.0, 0.145),
        size=(0.032, 0.070, 0.145),
        mass=0.35,
        rgba=(0.58, 0.22, 0.88, 1.0),
        friction=0.55,
        restitution=0.00,
        fixed=False,
    )
    api.add_box(
        "green_bar",
        pos=(0.76, 0.0, 0.145),
        size=(0.032, 0.070, 0.145),
        mass=0.35,
        rgba=(0.22, 0.78, 0.32, 1.0),
        friction=0.55,
        restitution=0.00,
        fixed=False,
    )
    api.add_box(
        "pink_bar",
        pos=(0.56, 0.16, 0.145),
        size=(0.032, 0.032, 0.145),
        mass=0.35,
        rgba=(0.94, 0.45, 0.56, 1.0),
        friction=0.55,
        restitution=0.00,
        fixed=False,
    )

    api.set_duration(4.0)
    api.add_camera("cam0", pos=(-0.12, -3.55, 0.88), target=(0.08, 0.0, 0.26))

SCENE_MANIFEST = {
    "scene_kind": "collision_arena",
    "target": "blue_plate",
    "trigger": "swing_striker",
    "movers": ["brown_hammer", "orange_ball", "blue_plate", "purple_bar"],
    "relation_pair": {
        "a": "orange_ball",
        "b": "blue_plate",
        "expected": "contact",
    },
    "families": [
        "motion_attribution",
        "collision_final_state",
        "collision_final_relation",
    ],
}
</propose>
"""

_FEWSHOT_EXAMPLES = load_prompt_fewshot_examples(
    (
        "02_collision_arena__v01.py",
        "02_collision_arena__v02.py",
        "02_collision_arena__v03.py",
        "02_collision_arena__v04.py",
        "02_collision_arena__v05.py",
        "02_collision_arena__v06.py",
        "02_collision_arena__v07.py",
        "02_collision_arena__v08.py",
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
