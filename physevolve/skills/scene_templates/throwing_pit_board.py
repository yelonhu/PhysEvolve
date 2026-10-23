"""Scene template: ``throwing_pit_board``.

The canonical throwing playground: launched
coloured balls fall, deflected by fixed coloured "floating" walls, and drop into
one of three VISIBLE pits -- the GAPS between bottom platforms. From ONE
simulation it auto-derives many VQAs:

* ``throwing_final_destination``   - "What is the final destination of the {ball}?"
* ``throwing_current_run_judgment`` - "Will the {ball} drop into the {region} pit?"
* ``counterfactual_destination``   - "If we removed the {wall} ... which pit ...?"
* ``inverse_intervention``         - "What can we do to make {ball} reach {pit}?"

Design (video-grounded, gap-pit model; matches the golden throwing
playgrounds under ``data/golden_scenes/stage2/``):

* The pits are the VISIBLE GAPS between bottom ``platforms`` (the ball "drops
  into" a gap and rests on the ground below the platform tops). The manifest
  only names the platform bodies; the executor derives left / middle / right
  destination regions from the repaired platform edges. This keeps the question
  wording ("the left pit") bound to visible geometry without asking the
  Proposer to hand-author fragile numeric pit intervals.
* Ground truth always comes from the simulated ball trajectory: a ball counts
  as "in" a pit only if it is at rest (low final speed, not ejected), its final
  x lies inside exactly one derived gap interval, and its final z is at/below the
  platform tops (it dropped INTO the gap rather than resting on a platform).
  Colours and region names are only labels (see ``colors.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from physevolve.engine.schema import GeomType
from physevolve.skills.scene_templates import (
    BranchSpec,
    GeneratedVQA,
    SceneTemplate,
    load_prompt_fewshot_examples,
)
from physevolve.skills.scene_templates.colors import contains_color_word

if TYPE_CHECKING:
    from physevolve.engine.schema import BodyDef, SceneDef
    from physevolve.executor.repair import RepairTrace
    from physevolve.executor.state_view import SimStateView


_SCENE_KIND = "throwing_pit_board"
_REGION_KEYS: tuple[str, ...] = ("left", "right", "middle")
_REGION_LABELS: Mapping[str, str] = {
    "left": "Left {region_kind}",
    "right": "Right {region_kind}",
    "middle": "Middle {region_kind}",
}
# A ball counts as settled when its final speed drops below this (m/s).
_SETTLE_SPEED = 0.30
# A ball counts as "dropped into" a gap only if its final centre height is at
# or below the platform tops plus this slack (so it fell PAST the platforms
# into the gap rather than resting on a platform / stuck on a wall).
_PIT_Z_BAND = 0.30
# Workspace half-width in x (matches engine.schema.WORKSPACE_X_HALF).
_WORKSPACE_X_HALF = 1.5
_MAX_BALLS = 4
# Branch-dependent families are expensive (extra sims), so only the first few
# balls get counterfactual / inverse questions, and the total is capped.
_MAX_BRANCH_BALLS = 2
_MAX_QUESTIONS = 16

_FAMILIES: tuple[str, ...] = (
    "throwing_final_destination",
    "throwing_current_run_judgment",
)
_BRANCH_FAMILIES: tuple[str, ...] = (
    "counterfactual_destination",
    "inverse_intervention",
)
_ALL_FAMILIES: frozenset[str] = frozenset(_FAMILIES) | frozenset(_BRANCH_FAMILIES)


# ─── manifest parsing helpers ─────────────────────────────────────────


def _balls(manifest: Mapping[str, object]) -> list[str]:
    raw = manifest.get("balls")
    if not isinstance(raw, list) or not raw:
        raise ValueError("manifest['balls'] must be a non-empty list of body names")
    out = []
    for b in raw:
        if not isinstance(b, str) or not b:
            raise ValueError(f"manifest['balls'] entry {b!r} must be a body name")
        out.append(b)
    return out


def _region_kind(manifest: Mapping[str, object]) -> str:
    rk = manifest.get("region_kind", "pit")
    return str(rk) if rk else "pit"


def _pits(manifest: Mapping[str, object]) -> dict[str, list[float]]:
    """Return derived ``{region_key: [x_lo, x_hi]}`` from a compiled manifest.

    Authored Proposer manifests no longer need to contain ``pits``; the executor
    fills this field during ``_compile_manifest`` from the platform geometry.
    """

    raw = manifest.get("pits")
    if not isinstance(raw, dict) or set(raw) != set(_REGION_KEYS):
        raise ValueError(
            f"manifest['pits'] must map exactly keys {_REGION_KEYS} to "
            "x-intervals of the form {'x': [lo, hi]}"
        )
    out: dict[str, list[float]] = {}
    for key in _REGION_KEYS:
        spec = raw[key]
        xr = spec.get("x") if isinstance(spec, dict) else None
        if not isinstance(xr, (list, tuple)) or len(xr) != 2:
            raise ValueError(
                f"pits[{key!r}] must be {{'x': [lo, hi]}} (the gap's x-interval)"
            )
        lo, hi = float(xr[0]), float(xr[1])
        if not lo < hi:
            raise ValueError(f"pits[{key!r}]['x'] must have lo < hi")
        out[key] = [lo, hi]
    return out


def _platforms(manifest: Mapping[str, object]) -> list[str]:
    """Return the visible fixed-box platform body names that delimit the pits."""

    raw = manifest.get("platforms")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "manifest['platforms'] must be a non-empty list of visible fixed "
            "box body names (the pits are the gaps between them)"
        )
    out = []
    for p in raw:
        if not isinstance(p, str) or not p:
            raise ValueError(f"manifest['platforms'] entry {p!r} must be a name")
        out.append(p)
    return out


def _walls(manifest: Mapping[str, object]) -> list[str]:
    """Return the fixed deflector ("floating") wall body names.

    Accepts the canonical ``walls`` key, falling back to the legacy
    ``obstacles`` key so an older manifest does not hard-fail.
    """

    raw = manifest.get("walls", manifest.get("obstacles", []))
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("manifest['walls'] must be a list of body names")
    out = []
    for w in raw:
        if not isinstance(w, str) or not w:
            raise ValueError(f"manifest['walls'] entry {w!r} must be a name")
        out.append(w)
    return out


def _families(manifest: Mapping[str, object]) -> list[str]:
    raw = manifest.get("families")
    if not isinstance(raw, list) or not raw:
        raise ValueError("manifest['families'] must be a non-empty list")
    out = []
    for f in raw:
        if f not in _ALL_FAMILIES:
            raise ValueError(f"manifest['families'] has unknown family {f!r}")
        out.append(str(f))
    return out


# ─── geometry: pits are GAPS between visible platforms ─────────────────


def _bodies_by_name(scene_def: "SceneDef") -> dict[str, "BodyDef"]:
    return {b.name: b for b in scene_def.bodies}


def _compile_manifest(
    manifest: Mapping[str, object],
    scene_def: "SceneDef",
    repair_trace: Optional["RepairTrace"] = None,
) -> Mapping[str, object]:
    """Derive safe manifest-only fields from the authored board geometry."""

    compiled: dict[str, object] = dict(manifest)
    bodies = _bodies_by_name(scene_def)

    for key in ("balls", "platforms", "walls", "obstacles"):
        aligned = _align_name_list(compiled.get(key), bodies.keys())
        if aligned is not None and aligned != compiled.get(key):
            compiled[key] = aligned
            if repair_trace is not None:
                repair_trace.add(
                    "align_manifest_roles",
                    key,
                    tier="manifest",
                    after=list(aligned),
                )

    if not isinstance(compiled.get("platforms"), list) or not compiled["platforms"]:
        inferred = _infer_platforms(scene_def)
        if inferred:
            compiled["platforms"] = inferred
            if repair_trace is not None:
                repair_trace.add(
                    "infer_manifest_platforms",
                    "platforms",
                    tier="manifest",
                    after=list(inferred),
                )

    platforms = compiled.get("platforms")
    if isinstance(platforms, list) and platforms:
        derived = _derive_pits_from_platforms([str(p) for p in platforms], scene_def)
        if derived is not None:
            if derived != compiled.get("pits"):
                compiled["pits"] = derived
                if repair_trace is not None:
                    repair_trace.add(
                        "derive_throwing_pits",
                        "pits",
                        tier="manifest",
                        after=derived,
                    )
        elif "pits" in compiled:
            compiled.pop("pits", None)
            if repair_trace is not None:
                repair_trace.add(
                    "drop_authored_throwing_pits",
                    "pits",
                    tier="manifest",
                    reason="platform geometry cannot derive visible gaps",
                )
    elif "pits" in compiled:
        compiled.pop("pits", None)
        if repair_trace is not None:
            repair_trace.add(
                "drop_authored_throwing_pits",
                "pits",
                tier="manifest",
                reason="no platform roles available",
            )

    return compiled


def _align_name_list(
    raw: object,
    body_names: Sequence[str],
) -> Optional[list[str]]:
    if not isinstance(raw, list):
        return None
    body_by_norm: dict[str, list[str]] = {}
    for name in body_names:
        body_by_norm.setdefault(_norm_name(name), []).append(name)
    out: list[str] = []
    changed = False
    for item in raw:
        if not isinstance(item, str) or not item:
            out.append(item)  # type: ignore[arg-type]
            continue
        if item in body_names:
            out.append(item)
            continue
        matches = body_by_norm.get(_norm_name(item), [])
        if len(matches) == 1:
            out.append(matches[0])
            changed = True
        else:
            out.append(item)
    return out if changed else None


def _norm_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _infer_platforms(scene_def: "SceneDef") -> list[str]:
    candidates = [
        b.name
        for b in scene_def.bodies
        if b.fixed
        and b.geom_type == GeomType.BOX
        and "platform" in b.name.lower()
        and len(b.size) >= 3
    ]
    return sorted(candidates, key=lambda name: _bodies_by_name(scene_def)[name].pos[0])


def _derive_pits_from_platforms(
    platforms: Sequence[str],
    scene_def: "SceneDef",
) -> Optional[dict[str, dict[str, list[float]]]]:
    if len(platforms) < 3:
        return None
    try:
        extents = sorted(_platform_x_extents(platforms, scene_def), key=lambda x: x[0])
    except ValueError:
        return None
    intervals: list[tuple[float, float]] = []
    for left, right in ((extents[0], extents[1]), (extents[1], extents[2])):
        lo, hi = float(left[1]), float(right[0])
        if hi - lo <= 0.05:
            return None
        intervals.append((lo, hi))
    lo, hi = float(extents[-1][1]), _WORKSPACE_X_HALF
    if hi - lo <= 0.05:
        return None
    intervals.append((lo, hi))
    return {
        key: {
            "x": [
                round(max(-_WORKSPACE_X_HALF, lo), 4),
                round(min(_WORKSPACE_X_HALF, hi), 4),
            ]
        }
        for key, (lo, hi) in zip(("left", "middle", "right"), intervals, strict=True)
    }


def _derive_regions_from_platforms(
    platforms: Sequence[str],
    scene_def: "SceneDef",
) -> Optional[dict[str, dict[str, object]]]:
    """Derive left/middle/right destination regions from platform geometry."""

    derived_pits = _derive_pits_from_platforms(platforms, scene_def)
    if derived_pits is None:
        return None
    intervals = {key: list(spec["x"]) for key, spec in derived_pits.items()}
    return _derive_regions(intervals, platforms, scene_def)


def _platform_x_extents(
    platforms: Sequence[str], scene_def: "SceneDef"
) -> list[tuple[float, float]]:
    """Return each platform's ``(x_min, x_max)`` footprint (boxes only)."""

    bodies = _bodies_by_name(scene_def)
    out: list[tuple[float, float]] = []
    for name in platforms:
        body = bodies.get(name)
        if body is None:
            raise ValueError(f"platform {name!r} was not created by build_scene")
        if body.geom_type != GeomType.BOX or len(body.size) < 3:
            raise ValueError(f"platform {name!r} must be a box")
        sx = float(body.size[0])
        out.append((body.pos[0] - sx, body.pos[0] + sx))
    return out


def _platform_top(platforms: Sequence[str], scene_def: "SceneDef") -> float:
    """Return the LOWEST platform top surface z (the gap "rim" height)."""

    bodies = _bodies_by_name(scene_def)
    tops: list[float] = []
    for name in platforms:
        body = bodies[name]
        tops.append(body.pos[2] + float(body.size[2]))
    return min(tops) if tops else 0.0


def _derive_regions(
    pits: Mapping[str, list[float]],
    platforms: Sequence[str],
    scene_def: "SceneDef",
) -> dict[str, dict[str, object]]:
    """Derive each region as ``{"x": [lo, hi], "z_ceiling": float}``.

    The x-interval is the declared gap; ``z_ceiling`` (lowest platform top) is
    the height below which a ball counts as "dropped into the gap" rather than
    resting on / flying over the platforms.
    """

    z_ceiling = _platform_top(platforms, scene_def)
    return {
        key: {"x": [float(xr[0]), float(xr[1])], "z_ceiling": z_ceiling}
        for key, xr in pits.items()
    }


def _which_region(
    final_pos: Sequence[float],
    regions: Mapping[str, Mapping[str, object]],
) -> Optional[str]:
    """Gap the ball rests in (x inside interval AND at/below platform top)."""

    x, _y, z = final_pos
    matches = []
    for key, fp in regions.items():
        x_min, x_max = fp["x"]  # type: ignore[index]
        z_ceiling = float(fp["z_ceiling"])  # type: ignore[arg-type]
        if x_min <= x <= x_max and z <= z_ceiling + _PIT_Z_BAND:
            matches.append(key)
    return matches[0] if len(matches) == 1 else None


def _final_speed(view: "SimStateView", body: str) -> float:
    v = view[body].final_vel
    return (sum(c * c for c in v)) ** 0.5


def _ball_label(name: str) -> str:
    """Human label for a ball body name (``red_ball`` -> ``red ball``)."""

    return name.replace("_", " ")


def _ball_phrase(name: str, colors: Mapping[str, str]) -> str:
    """Refer to a ball by its VISIBLE colour ("the red ball").

    Falls back to the prettified body name when no colour is known.
    """

    color = colors.get(name)
    return f"{color} ball" if color else _ball_label(name)


def _wall_phrase(name: str, colors: Mapping[str, str]) -> str:
    """Refer to a wall/deflector by its visible colour ("the blue floating wall").

    Colour-names a wall/deflector ("the blue floating wall") so
    counterfactual / inverse questions name the visible obstacle.
    """

    color = colors.get(name)
    return f"{color} floating wall" if color else _ball_label(name)


# Minimum initial-x separation (m) for a ball to be unambiguously the leftmost
# or rightmost one in the video (so a spatial reference is safe).
_SPATIAL_MARGIN = 0.15


def _spatial_ref_map(
    balls: Sequence[str], view: "SimStateView"
) -> dict[str, str]:
    """Map the unambiguous leftmost / rightmost ball to a spatial phrase.

    A ball gets a spatial phrase only when its initial x is clearly separated
    (> ``_SPATIAL_MARGIN``) from the next ball, so "the leftmost launched ball"
    is unambiguous in the video. Returns ``{ball: phrase}`` (phrase WITHOUT a
    leading article; callers prepend "the ").
    """

    xs: dict[str, float] = {}
    for b in balls:
        if b in view:
            xs[b] = float(view[b].initial_pos[0])
    if len(xs) < 2:
        return {}
    ordered = sorted(xs, key=lambda b: xs[b])
    out: dict[str, str] = {}
    if xs[ordered[1]] - xs[ordered[0]] > _SPATIAL_MARGIN:
        out[ordered[0]] = "leftmost launched ball"
    if xs[ordered[-1]] - xs[ordered[-2]] > _SPATIAL_MARGIN:
        out[ordered[-1]] = "rightmost launched ball"
    return out


def _ref_phrase(
    ball: str, colors: Mapping[str, str], spatial: Mapping[str, str]
) -> str:
    """Refer to a ball by SPATIAL position when unambiguous, else by colour.

    Mixing spatial and colour references (and the colour itself varying across
    colour-permuted variants) stops the Solver from shortcutting on a fixed
    "this colour -> this pit" mapping instead of watching the video.
    """

    sp = spatial.get(ball)
    if sp:
        return sp
    return _ball_phrase(ball, colors)


def _settled_region_in(
    view: "SimStateView",
    ball: str,
    regions: Mapping[str, Mapping[str, object]],
) -> Optional[str]:
    """Region the ball rests in for ``view``, or None if moving / ejected / out.

    Hardened ground truth: the ball must be present, NOT ejected, at rest
    (final speed below the settle threshold), and resting inside exactly one
    pit footprint at pit height. Otherwise the ball has no verifiable
    destination and the caller skips it.
    """

    if ball not in view:
        return None
    body = view[ball]
    if body.final_status == "ejected":
        return None
    if _final_speed(view, ball) > _SETTLE_SPEED:
        return None
    return _which_region(body.final_pos, regions)


def _cf_key(ball: str, wall: str) -> str:
    return f"cf|{ball}|{wall}"


def _inv_key(ball: str, lever: str) -> str:
    return f"inv|{ball}|{lever}"


# ─── validation + branches ────────────────────────────────────────────


def _validate_manifest(
    manifest: Mapping[str, object], scene_def: "SceneDef"
) -> Optional[str]:
    if manifest.get("scene_kind") != _SCENE_KIND:
        return f"scene_kind must be {_SCENE_KIND!r}"
    try:
        balls = _balls(manifest)
        families = _families(manifest)
        walls = _walls(manifest)
        platforms = _platforms(manifest)
    except ValueError as exc:
        return str(exc)

    bodies = _bodies_by_name(scene_def)
    fixed_by_name = {name: bool(b.fixed) for name, b in bodies.items()}

    for ball in balls:
        if ball not in fixed_by_name:
            return f"ball {ball!r} was not created by build_scene"
        if fixed_by_name[ball]:
            return f"ball {ball!r} must be a free dynamic body (fixed=False)"
        # Anti-collapse: the ball's colour shown to the Solver is derived from
        # its rgba (so colour permutation actually changes the question). A
        # colour-baked body name (e.g. green_ball) double-encodes the colour and
        # lets the Solver/Proposer shortcut, so reject it -- use neutral names
        # like ball_a / ball_b / ball_c and set distinct rgba instead.
        if contains_color_word(ball):
            return (
                f"ball {ball!r} must NOT be colour-named; use a neutral name "
                "(e.g. ball_a) and set its rgba -- the colour word in the "
                "question comes from the rgba, not the body name"
            )

    # Platforms must exist and be fixed boxes; the executor derives left /
    # middle / right pit-style regions from the visible x-gaps between them.
    for name in platforms:
        body = bodies.get(name)
        if body is None:
            return f"platform {name!r} was not created by build_scene"
        if not body.fixed:
            return f"platform {name!r} must be fixed=True"
        if body.geom_type != GeomType.BOX:
            return f"platform {name!r} must be a box"
    if len(platforms) < 3:
        return (
            "need >= 3 platforms so the executor can derive left/middle/right "
            "visible gap regions"
        )
    if _derive_regions_from_platforms(platforms, scene_def) is None:
        return (
            "platforms must form three positive-width visible x-gaps; separate "
            "platform boxes instead of hand-authoring pit intervals"
        )

    for wall in walls:
        if wall not in fixed_by_name:
            return f"wall {wall!r} was not created by build_scene"
        if not fixed_by_name[wall]:
            return f"wall {wall!r} must be fixed=True"

    if not families:
        return "manifest declares no question families"
    if "inverse_intervention" in families and len(walls) < 2:
        return "inverse_intervention requires >= 2 walls (removable levers)"
    return None


def _required_branches(manifest: Mapping[str, object]) -> list[BranchSpec]:
    """Branch sims needed by the counterfactual / inverse families.

    Inverse branches are deliberately listed first. With the executor's fixed
    six-distinct-sim budget, two balls need exactly six a/b/both interventions;
    later counterfactual specs for walls[0:2] reuse those cached removal sets.
    counterfactual: remove {wall} + the OTHER balls
    ("remove the blue wall and other balls").
    inverse: per ball, branches for removing wall_a / wall_b / both
    (each also removing the other balls).
    """

    families = set(_families(manifest))
    balls = _balls(manifest)
    walls = _walls(manifest)
    specs: list[BranchSpec] = []
    branch_balls = balls[:_MAX_BRANCH_BALLS]

    if "inverse_intervention" in families and len(walls) >= 2:
        wall_a, wall_b = walls[0], walls[1]
        for ball in branch_balls:
            others = [b for b in balls if b != ball]
            specs.append(
                BranchSpec(key=_inv_key(ball, "a"), remove=tuple([wall_a, *others]))
            )
            specs.append(
                BranchSpec(key=_inv_key(ball, "b"), remove=tuple([wall_b, *others]))
            )
            specs.append(
                BranchSpec(
                    key=_inv_key(ball, "both"),
                    remove=tuple([wall_a, wall_b, *others]),
                )
            )

    if "counterfactual_destination" in families and walls:
        for ball in branch_balls:
            others = [b for b in balls if b != ball]
            for wall in walls:
                specs.append(
                    BranchSpec(
                        key=_cf_key(ball, wall),
                        remove=tuple([wall, *others]),
                    )
                )
    return specs


# ─── question generation ──────────────────────────────────────────────


def _generate_questions(
    manifest: Mapping[str, object],
    baseline_view: "SimStateView",
    branch_views: Mapping[str, "SimStateView"],
    body_colors: Mapping[str, str],
    scene_def: "SceneDef",
) -> Sequence[GeneratedVQA]:
    balls = _balls(manifest)[:_MAX_BALLS]
    platforms = _platforms(manifest)
    regions = _derive_regions_from_platforms(platforms, scene_def)
    if regions is None:
        return []
    region_kind = _region_kind(manifest)
    families = set(_families(manifest))
    out: list[GeneratedVQA] = []

    # Resolve each ball's settled pit once (baseline run), with hardened GT.
    settled: dict[str, Optional[str]] = {
        ball: _settled_region_in(baseline_view, ball, regions) for ball in balls
    }
    # Spatial reference map (leftmost / rightmost), mixed in alongside colour
    # references so the question text does not always carry the same colour word.
    spatial = _spatial_ref_map(balls, baseline_view)

    region_opts = [
        f"The {k} {region_kind}" if k != "middle" else f"The center {region_kind}"
        for k in ("left", "right", "middle")
    ] + ["None of the above"]

    if "throwing_final_destination" in families:
        for ball in balls:
            region = settled.get(ball)
            if region is None:
                # Escape / unsettled: still emit a destination question with
                # answer_semantic=neither so catalog layouts with neither/none
                # can bind (zero gold — GT from final_status / settle miss).
                out.append(
                    GeneratedVQA(
                        question=(
                            "What is the final destination of the "
                            f"{_ref_phrase(ball, body_colors, spatial)}?"
                        ),
                        options=[
                            f"The left {region_kind}",
                            f"The right {region_kind}",
                            "Neither pit",
                            "None of the above",
                        ],
                        correct_idx=2,
                        family="throwing_final_destination",
                        phrasing="forward",
                        spec={
                            "ball": ball,
                            "target": "neither",
                            "region_kind": region_kind,
                            "answer_semantic": "neither",
                        },
                    )
                )
                continue
            out.append(
                GeneratedVQA(
                    question=(
                        "What is the final destination of the "
                        f"{_ref_phrase(ball, body_colors, spatial)}?"
                    ),
                    options=list(region_opts),
                    correct_idx=_REGION_KEYS.index(region),
                    family="throwing_final_destination",
                    phrasing="forward",
                    spec={
                        "ball": ball,
                        "target": region,
                        "region_kind": region_kind,
                        "answer_semantic": region,
                    },
                )
            )
            out.append(
                GeneratedVQA(
                    question=(
                        "What will eventually happen to the "
                        f"{_ref_phrase(ball, body_colors, spatial)}?"
                    ),
                    options=list(region_opts),
                    correct_idx=_REGION_KEYS.index(region),
                    family="throwing_final_destination",
                    phrasing="fate",
                    spec={
                        "ball": ball,
                        "target": region,
                        "region_kind": region_kind,
                        "answer_semantic": region,
                    },
                )
            )

    if "throwing_current_run_judgment" in families:
        yn_opts = ["Yes", "No", "Can not answer", "Uncertain"]
        for i, ball in enumerate(balls):
            region = settled.get(ball)
            if region is None:
                # Unsettled ball cannot drop into any named pit -> No.
                target = _REGION_KEYS[i % len(_REGION_KEYS)]
                out.append(
                    GeneratedVQA(
                        question=(
                            f"Will the {_ref_phrase(ball, body_colors, spatial)} "
                            f"eventually drop into the {target} {region_kind}?"
                        ),
                        options=list(yn_opts),
                        correct_idx=1,
                        family="throwing_current_run_judgment",
                        phrasing="yes_no",
                        spec={
                            "ball": ball,
                            "target": target,
                            "region_kind": region_kind,
                            "answer_semantic": "no",
                        },
                    )
                )
                continue
            # Alternate the target between the ball's actual region (Yes) and a
            # different region (No) by ball index, so the Yes/No GT is balanced.
            if i % 2 == 0:
                target = region
            else:
                target = next(k for k in _REGION_KEYS if k != region)
            correct = 0 if target == region else 1  # 0=Yes, 1=No
            out.append(
                GeneratedVQA(
                    question=(
                        f"Will the {_ref_phrase(ball, body_colors, spatial)} "
                        f"eventually drop into the {target} {region_kind}?"
                    ),
                    options=list(yn_opts),
                    correct_idx=correct,
                    family="throwing_current_run_judgment",
                    phrasing="yes_no",
                    spec={
                        "ball": ball,
                        "target": target,
                        "region_kind": region_kind,
                        "answer_semantic": "yes" if correct == 0 else "no",
                    },
                )
            )
            wrong_target = (
                next(k for k in _REGION_KEYS if k != region)
                if i % 2 == 0
                else region
            )
            wrong_correct = 0 if wrong_target != region else 1  # 0=Yes, 1=No
            out.append(
                GeneratedVQA(
                    question=(
                        f"Will the {_ref_phrase(ball, body_colors, spatial)} "
                        f"finally drop into a different {region_kind} than the "
                        f"{wrong_target} {region_kind}?"
                    ),
                    options=list(yn_opts),
                    correct_idx=wrong_correct,
                    family="throwing_current_run_judgment",
                    phrasing="wrong_pit",
                    spec={
                        "ball": ball,
                        "target": wrong_target,
                        "region_kind": region_kind,
                        "answer_semantic": "yes" if wrong_correct == 0 else "no",
                    },
                )
            )

    walls = _walls(manifest)
    branch_balls = balls[:_MAX_BRANCH_BALLS]

    # Counterfactual uses a "None of the above" escape
    # in place of the destination family's "It remains uncertain".
    cf_region_opts = [
        f"The {k} {region_kind}" if k != "middle" else f"The center {region_kind}"
        for k in _REGION_KEYS
    ] + ["None of the above"]

    if "counterfactual_destination" in families and walls:
        for ball in branch_balls:
            for wall in walls:
                view = branch_views.get(_cf_key(ball, wall))
                if view is None:
                    continue  # branch failed / capped -> drop this question
                region = _settled_region_in(view, ball, regions)
                if region is None:
                    continue
                if region == settled.get(ball):
                    continue
                out.append(
                    GeneratedVQA(
                        question=(
                            f"If we removed the {_wall_phrase(wall, body_colors)} "
                            f"and other balls, which {region_kind} would the "
                            f"{_ref_phrase(ball, body_colors, spatial)} drop into?"
                        ),
                        options=list(cf_region_opts),
                        correct_idx=_REGION_KEYS.index(region),
                        family="counterfactual_destination",
                        phrasing="counterfactual",
                        spec={
                            "ball": ball,
                            "removed_wall": wall,
                            "target": region,
                            "region_kind": region_kind,
                            "answer_semantic": region,
                        },
                    )
                )

    if "inverse_intervention" in families and len(walls) >= 2:
        wall_a, wall_b = walls[0], walls[1]
        for ball in branch_balls:
            vqa = _build_inverse_question(
                ball, wall_a, wall_b, region_kind, regions,
                baseline_view, branch_views, body_colors, spatial,
            )
            if vqa is not None:
                out.append(vqa)

    return out[:_MAX_QUESTIONS]


def _build_inverse_question(
    ball: str,
    wall_a: str,
    wall_b: str,
    region_kind: str,
    regions: Mapping[str, Mapping[str, object]],
    baseline_view: "SimStateView",
    branch_views: Mapping[str, "SimStateView"],
    body_colors: Mapping[str, str],
    spatial: Mapping[str, str],
) -> Optional[GeneratedVQA]:
    """Build a verifier-grounded inverse-intervention question, or None.

    Computes the ball's settled region under each intervention (baseline +
    remove_a / remove_b / remove_both, each from a real branch sim) and asks
    "what can we do to make the {ball} reach the {target} pit?". The target is
    the first region (!= baseline) achieved by EXACTLY ONE intervention, so the
    answer is unambiguous; if none qualifies, falls back to a region no
    intervention reaches (answer = "None of the above"). Returns None when even
    that is ambiguous or branches are missing.
    """

    va = branch_views.get(_inv_key(ball, "a"))
    vb = branch_views.get(_inv_key(ball, "b"))
    vboth = branch_views.get(_inv_key(ball, "both"))
    if va is None or vb is None or vboth is None:
        return None  # need all three branches to ground the options

    baseline_region = _settled_region_in(baseline_view, ball, regions)
    # intervention -> resulting region (None if not settled in that branch)
    outcomes = {
        "a": _settled_region_in(va, ball, regions),
        "b": _settled_region_in(vb, ball, regions),
        "both": _settled_region_in(vboth, ball, regions),
    }
    # Option order A/B/C/D (canonical fallback; catalog layout applied at render).
    option_levers = ("a", "b", "both", "none")
    wall_a_phrase = _wall_phrase(wall_a, body_colors)
    wall_b_phrase = _wall_phrase(wall_b, body_colors)
    options = [
        f"Remove the {wall_a_phrase} and other balls",
        f"Remove the {wall_b_phrase} and other balls",
        "Remove both floating walls and other balls",
        "None of the above",
    ]
    lever_semantics = {
        "a": "remove_a",
        "b": "remove_b",
        "both": "remove_both",
        "none": "none",
    }

    for target in _REGION_KEYS:
        if target == baseline_region:
            continue
        achievers = [k for k in ("a", "b", "both") if outcomes[k] == target]
        if len(achievers) == 1:
            correct_idx = option_levers.index(achievers[0])
            return GeneratedVQA(
                question=(
                    "What can we do to make the "
                    f"{_ref_phrase(ball, body_colors, spatial)} "
                    f"drop into the {target} {region_kind}?"
                ),
                options=options,
                correct_idx=correct_idx,
                family="inverse_intervention",
                phrasing="inverse",
                spec={
                    "ball": ball,
                    "target": target,
                    "region_kind": region_kind,
                    "answer_semantic": lever_semantics[achievers[0]],
                    "wall_a_phrase": wall_a_phrase,
                    "wall_b_phrase": wall_b_phrase,
                },
            )
    # Fallback: a region no intervention (and not baseline) reaches -> "None".
    for target in _REGION_KEYS:
        if target == baseline_region:
            continue
        if all(outcomes[k] != target for k in ("a", "b", "both")):
            return GeneratedVQA(
                question=(
                    "What can we do to make the "
                    f"{_ref_phrase(ball, body_colors, spatial)} "
                    f"drop into the {target} {region_kind}?"
                ),
                options=options,
                correct_idx=option_levers.index("none"),
                family="inverse_intervention",
                phrasing="inverse",
                spec={
                    "ball": ball,
                    "target": target,
                    "region_kind": region_kind,
                    "answer_semantic": "none",
                    "wall_a_phrase": wall_a_phrase,
                    "wall_b_phrase": wall_b_phrase,
                },
            )
    return None


# ─── prompt-facing docs + fewshot ─────────────────────────────────────

_MANIFEST_SCHEMA_DOC = """\
  SCENE_MANIFEST = {
      "scene_kind": "throwing_pit_board",
      "balls": ["<neutral_ball_1>", "<neutral_ball_2>", ...],  # NEUTRAL names (NOT colour
                                                     #   names); each free dynamic,
                                                     #   launched, drops into a pit
      "platforms": [<visible fixed box names>],      # >=3 fixed boxes; executor derives
                                                     #   left/middle/right gap regions
                                                     #   from their repaired x-edges
      "walls": [<fixed floating-wall body names>],   # >=2 needed for inverse_intervention;
                                                     #   first two are levers A/B
      "region_kind": "pit",                          # label noun used in the question
      "families": [                                  # any subset of:
          "throwing_final_destination",              #   forward (baseline sim)
          "throwing_current_run_judgment",           #   yes/no  (baseline sim)
          "counterfactual_destination",              #   "if we removed X" (branch sim)
          "inverse_intervention",                    #   "what makes it reach Y" (branch sim)
      ],
  }
"""

_SCENE_CONSTRAINTS = """\
  - Build fixed bottom platforms (boxes) and use the x-gaps between them as
    visible pits. List platform body names in "platforms"; do not write a
    separate "pits" interval map.
  - Launch several free dynamic balls from above. Ball names must be neutral:
    valid examples are ball_0 / ball_1 / ball_2; invalid examples are ball_red,
    ball_blue, green_ball, and red_ball. The question colour comes from rgba.
  - Keep the whole board centered inside |x| < 1.5 with margin. Never put
    platform, deflector, or ball centres at x >= 1.5 or outside the strict
    workspace; avoid all-positive platform chains such as 0.5, 1.5, 2.5.
  - Use a golden-style centered platform band: left platform at negative x,
    middle near zero, right positive but still well inside the workspace. Small
    platform half_x values around 0.25-0.45 leave visible gaps; do not copy the
    exact golden coordinates, but keep this centered spacing shape.
  - A valid scene needs >= 3 fixed platforms with positive x-gaps, >= 1 free
    dynamic ball, negative-z gravity, duration, and a camera. Platform-only /
    wall-only scenes are trivial.
  - Throwing is z-axis falling: balls start above platform tops with z clearance
    and visibly descend in negative z. Use y only for lane/depth spacing, not as
    the fall axis.
  - The executor derives left/middle/right destination regions from repaired
    platform x-edges; your job is to create separated platform geometry.
  - Add at least two fixed floating walls when declaring counterfactual or
    inverse families; the first two names are removable levers A/B and should
    be the strongest trajectory-changing deflectors.
  - Set clear, distinct rgba values for balls and walls so colour words match the
    rendered video.
  - Use a wide negative-y board camera. Calibration: cam0 pos approx
    (0, -3.5, 1.0), target approx (0, 0, 0.7); vary the numbers, but keep the
    whole board, deflectors, and final gaps visible.
"""

_AUTHORING_PLAYBOOK = """\
### Playbook: throwing_pit_board

Use z-up coordinates. The board lies in x/y; balls fall in negative z.

Naming:
  - platforms and walls may have descriptive names, but every name must be
    lowercase and unique.

Gap geometry recipe:
  1. For a platform box with pos=(px, 0.0, pz), size=(sx, sy, sz):
       x-span = [px - sx, px + sx]
       top_z  = pz + sz
  2. Sort platforms by x and leave positive open x-gaps between their edges.
  3. Do NOT include a separate pit interval map; the executor derives left/middle/right regions from platform spans after repair.
  4. Do not invent off-screen regions; make the visible platform gaps readable.

Settling recipe:
  - Spawn each ball above the platform tops: z > platform_top + radius + 0.05.
  - Give each ball a different modest x/z velocity; do not copy one velocity
    tuple across all balls. Keep launch speed below the hard-chaos regime.
  - Use negative-z gravity chosen for the board height and duration; tune it so
    balls visibly contact deflectors and settle within the clip.
  - The executor only asks questions for balls that finish at rest in exactly
    one executor-derived gap below the platform tops.
  - Counterfactual / inverse questions need >= 2 wall names in manifest["walls"].

Branch-family obligations:
  - Add at least one rotated / slanted deflector when declaring branch families;
    all-axis-aligned walls usually make shallow interventions.
  - Put the two actionable inverse levers first in manifest["walls"]. Removing
    lever A, lever B, and both must produce verifier-distinguishable outcomes.
  - Removing one wall should change at least one ball's settled pit in a branch
    sim; wall/platform geometry, not raw speed, should determine the final pit.
  - Balls should start in separated x lanes with non-identical velocities and
    visible colours.

Format-only scaffold (invent names and exact coordinates):
  - Bodies: 3 fixed platform boxes, 2 fixed floating walls, 2-3 neutral balls.
  - Dynamics: each ball gets modest x/z velocity, negative-z gravity, duration,
    and a wide negative-y camera framing platforms, walls, and the final pits.
  - Manifest shape: "balls", "platforms", "walls", "region_kind": "pit", and
    the realised "families"; no authored "pits" interval map.
"""

_AUTHORING_SKELETON = """\
### Topology Skeleton: throwing_pit_board

Plan: neutral-named balls fall through fixed deflectors and settle in visible
x-gaps between fixed bottom platforms.

Build order (invent all names, coordinates, sizes, and parameters):
  1. Ground plus two or more fixed platform boxes near the bottom of the view.
  2. Choose open x-gaps between sorted platform edges; executor derives the
     left/middle/right regions from these platforms.
  3. Add at least two fixed floating walls if branch families are declared.
     List the two strongest intervention levers first in manifest["walls"].
  4. Add several free dynamic balls above the platforms with neutral names and
     distinct rgba values.
  5. Use non-identical modest velocities and negative-z gravity so
     wall/platform geometry, not raw speed, determines the final pit.

Anti-copy: this block is a topology checklist, not code. Do NOT paste the old
three-platform/two-wall coordinates or the skeleton body names.
"""

_GOLDEN_DESIGN_CARD = """\
## Design Target: throwing_pit_board

- A strong scene is a real board: visible gap pits plus deflectors create several
  possible trajectories, not a toy platform stack.
- Interventions should matter: branch simulations change a ball's settled pit.
- Variation axes: platform gaps, wall angles, launch lanes, restitution/friction,
  gravity strength, camera framing, and declared families.
"""

_MAIN_TRAINING_WIRING_CARD = """\
## Golden Wiring Card: throwing_pit_board

Distilled from the validated board fewshot. This is not runnable code and does
not give coordinates; use it as the single construction pattern.

### Apparatus Wiring
  1. Add ground, then >=3 fixed low platform boxes spread left-to-right but keep
     the board centered inside |x| < 1.5. Do not place platform/deflector/ball
     centres at x >= 1.5, and avoid all-positive chains like 0.5, 1.5, 2.5.
     A robust shape resembles the validated board spacing: left at negative x,
     middle near zero, right around the positive side but not near the boundary,
     with small platform half_x values around 0.25-0.45. Vary the numbers; do
     not paste the exact golden coordinate triplet.
  2. Leave positive visible x-gaps between sorted platform edges. Do not include a separate pit interval map; the executor derives left/middle/right regions from repaired platform geometry.
  3. Add >=2 fixed floating deflectors if declaring branch families. At least
     one deflector should be rotated/slanted with `api.set_rotation(...)` so
     removing a wall can change a settled pit. The first two manifest wall names
     are inverse levers A/B; removing A, B, and both must have distinct effects.
  4. Add 2-3 free dynamic neutral-named balls above the board. Use names like
     `ball_0`, `ball_1`, `ball_2`; never `ball_red`, `ball_blue`,
     `green_ball`, or `red_ball`. Give each ball a distinct rgba, separated
     launch lane, and non-identical modest x/z velocity. The visible throw
     should descend along negative z; y is only lane/depth spacing.
  5. Set negative-z gravity, duration, and a camera that shows deflectors,
     platform gaps, and final settled pits. Camera calibration: cam0 pos approx
     (0, -3.5, 1.0), target approx (0, 0, 0.7). Vary the numbers, but keep wide
     negative-y framing.

### Manifest Wiring
  - `balls`: only free dynamic neutral ball names.
  - `platforms`: fixed boxes whose x-edges define visible destination gaps.
  - `walls`: fixed deflectors used by branch sims; the first two are inverse
    levers A/B and must be the actionable pair.
  - `families`: declare only families the video supports.

### Syntax Anchors
  - Deflector:
      api.add_box("<deflector>", pos=(x, y, z), size=(sx, sy, sz),
                  mass=m, rgba=(r, g, b, a), friction=f, restitution=e,
                  fixed=True)
      api.set_rotation("<deflector>", quaternion=(qw, qx, qy, qz))
  - Neutral ball launch:
      api.add_sphere("<neutral_ball>", pos=(x, y, z), radius=r, mass=m,
                     rgba=(r, g, b, a), friction=f, restitution=e,
                     fixed=False)
      api.set_velocity("<neutral_ball>", linear=(vx, vy, vz),
                       angular=(wx, wy, wz))
  - Camera:
      api.add_camera("cam0", pos=(cam_x, cam_y, cam_z),
                     target=(target_x, target_y, target_z))

### Validity Checks
  - Every body centre stays inside |x| < 1.5 with margin; platform sizes are
    half-extents.
  - Platform centers are not all positive; use a centered left / middle / right
    board so repair never clamps the right platform into the boundary.
  - Platform boxes have positive visible x-gaps; no separate pit interval
    arithmetic belongs in the manifest.
  - Ball spawn height clears platform tops by radius + gap, and balls do not
    overlap each other or deflectors at t=0.
  - Counterfactual/inverse scenes need a wall removal that can change at least
    one ball's settled pit.

Variation axes: platform gaps, deflector count/angle, launch lanes, rgba,
restitution/friction, gravity strength, duration, and camera.
"""

# Golden = ``data/golden_scenes/stage2/01_throwing_pit_board__v*.py``
# plus SCENE_MANIFEST. The three pits are the visible gaps: left = between
# left_platform and middle_platform; middle = between middle and right; right =
# off the right edge of right_platform.
_FEWSHOT = """\
Plan: three balls fall through a board of slanted coloured walls and drop into
the gaps between three bottom platforms; the same scene supports a destination /
yes-no / counterfactual / inverse question per ball.
<propose>
def build_scene(api):
    api.add_plane(name="ground", pos=(0.0, 0.0, 0.0), size=(5.0, 5.0),
                  rgba=(0.11, 0.11, 0.11, 1.0), friction=0.5)
    # Bottom platforms; the gaps between them are the pits.
    api.add_box("left_platform", pos=(-0.92, 0.0, 0.105), size=(0.34, 0.13, 0.105),
                mass=10.0, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.5,
                restitution=0.0, fixed=True)
    api.add_box("middle_platform", pos=(0.10, 0.0, 0.105), size=(0.40, 0.13, 0.105),
                mass=10.0, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.5,
                restitution=0.0, fixed=True)
    api.add_box("right_platform", pos=(1.00, 0.0, 0.105), size=(0.33, 0.13, 0.105),
                mass=10.0, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.5,
                restitution=0.0, fixed=True)
    # Slanted / vertical floating walls (deflectors; removable for cf / inverse).
    api.add_box("black_left_wall", pos=(-1.39, 0.0, 0.62), size=(0.50, 0.055, 0.055),
                mass=10.0, rgba=(0.01, 0.015, 0.018, 1.0), friction=0.5,
                restitution=0.5, fixed=True)
    api.set_rotation("black_left_wall", quaternion=(0.9239, 0.0, 0.3827, 0.0))
    api.add_box("orange_wall", pos=(-0.45, 0.0, 0.75), size=(0.035, 0.060, 0.315),
                mass=10.0, rgba=(0.95, 0.72, 0.10, 1.0), friction=0.5,
                restitution=0.5, fixed=True)
    api.add_box("blue_wall", pos=(0.20, 0.0, 0.70), size=(0.39, 0.055, 0.055),
                mass=10.0, rgba=(0.02, 0.20, 0.90, 1.0), friction=0.5,
                restitution=0.5, fixed=True)
    api.set_rotation("blue_wall", quaternion=(0.9455, 0.0, -0.3256, 0.0))
    api.add_box("pink_right_wall", pos=(1.29, 0.0, 0.64), size=(0.55, 0.060, 0.055),
                mass=10.0, rgba=(0.94, 0.45, 0.56, 1.0), friction=0.5,
                restitution=0.5, fixed=True)
    api.set_rotation("pink_right_wall", quaternion=(0.9304, 0.0, -0.3665, 0.0))
    # Three launched balls. NEUTRAL names (ball_a/b/c) + distinct rgba: the
    # colour word in the question ("the green ball") is derived from the rgba,
    # never from the body name, so colour-permuted variants stay correct.
    api.add_sphere("ball_a", pos=(-1.34, 0.0, 1.30), radius=0.085, mass=1.0,
                   rgba=(0.2, 0.75, 0.25, 1.0), friction=0.2, restitution=0.7,
                   fixed=False)
    api.add_sphere("ball_b", pos=(-0.05, 0.0, 1.48), radius=0.085, mass=1.0,
                   rgba=(0.95, 0.55, 0.15, 1.0), friction=0.2, restitution=0.9,
                   fixed=False)
    api.add_sphere("ball_c", pos=(1.04, 0.0, 1.25), radius=0.085, mass=1.0,
                   rgba=(0.2, 0.3, 0.9, 1.0), friction=0.2, restitution=0.9,
                   fixed=False)
    api.set_velocity("ball_a", linear=(0.26, 0.0, -0.18), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_b", linear=(0.10, 0.0, -0.24), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_c", linear=(-0.06, 0.0, -0.40), angular=(0.0, 0.0, 0.0))
    api.set_gravity((0.0, 0.0, -1.8))
    api.set_duration(4.9)
    api.add_camera("cam0", pos=(0.0, -3.70, 1.05), target=(0.0, 0.0, 0.72))

SCENE_MANIFEST = {
    "scene_kind": "throwing_pit_board",
    "balls": ["ball_a", "ball_b", "ball_c"],
    "platforms": ["left_platform", "middle_platform", "right_platform"],
    "walls": ["black_left_wall", "orange_wall", "blue_wall", "pink_right_wall"],
    "region_kind": "pit",
    "families": ["throwing_final_destination", "throwing_current_run_judgment",
                 "counterfactual_destination", "inverse_intervention"],
}
</propose>
"""

_FEWSHOT_EXAMPLES = load_prompt_fewshot_examples(
    (
        "01_throwing_pit_board__v01.py",
        "01_throwing_pit_board__v02.py",
        "01_throwing_pit_board__v03.py",
        "01_throwing_pit_board__v04.py",
        "01_throwing_pit_board__v05.py",
        "01_throwing_pit_board__v06.py",
        "01_throwing_pit_board__v07.py",
        "01_throwing_pit_board__v08.py",
        "01_throwing_pit_board__v09.py",
        "01_throwing_pit_board__v10.py",
    ),
    drop_manifest_keys=("pits",),
)


def _color_variant_bodies(manifest: Mapping[str, object]) -> list[str]:
    """Balls are referred to by their rgba-derived colour, so recolouring them
    mints valid colour variants (the GT is by body identity, unchanged)."""

    raw = manifest.get("balls")
    if not isinstance(raw, list):
        return []
    return [str(b) for b in raw if isinstance(b, str)]


def make_template() -> SceneTemplate:
    return SceneTemplate(
        scene_kind=_SCENE_KIND,
        families=_FAMILIES + _BRANCH_FAMILIES,
        validate_manifest=_validate_manifest,
        required_branches=_required_branches,
        generate_questions=_generate_questions,
        compile_manifest=_compile_manifest,
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
