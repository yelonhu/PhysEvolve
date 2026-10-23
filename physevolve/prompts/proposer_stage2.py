"""Stage-2 Proposer prompts for scene programs and manifests.

The Proposer supplies ``build_scene(api)`` and a ``SCENE_MANIFEST`` literal.
The executor derives feasible questions from the declared families of the
throwing, collision, or motion template.

``build_proposer_scene_template_prompt`` combines the template constraints,
manifest schema, demonstration program, and optional question-family guidance.
"""

from __future__ import annotations

from typing import Optional, Sequence

from physevolve.skills.question_families import QuestionFamily, lookup_family


_ROLE_AND_RULES: str = """\
# Role

You are a Scene Proposer. Design ONE executable physics scene plus a
`SCENE_MANIFEST`. A separate model will watch the rendered video and answer
A/B/C/D multiple-choice questions grounded in the real simulation.

Your final product is:

- Python code: one `def build_scene(api):` function containing PhysDSL calls.
- Manifest data: one module-level `SCENE_MANIFEST = {...}` literal naming the
  bodies, regions, and families that are truly supported by the scene.

Use `api` only inside `build_scene`.

Do not write question text, options, or the correct answer. The executor creates
those from your manifest and from simulated trajectories.
Never define `get_vqa` or `VQA_SPEC`.
"""


_PHYSDSL_API_REFERENCE: str = """\

## PhysDSL API Reference

Every `api.*` call mutates the scene builder and returns `None`; do not branch
on return values. Use these exact parameter names.

- `api.add_box(name, pos, size, mass, rgba, friction, restitution, fixed)`:
  rectangular body. `pos` is the world-frame center; `size=(sx, sy, sz)` is
  half-extents, not full width.
- `api.add_sphere(name, pos, radius, mass, rgba, friction, restitution, fixed)`:
  spherical dynamic or fixed body. Use `rgba=(r, g, b, a)`, not `color=...`.
- `api.add_cylinder(name, pos, radius, half_height, mass, rgba, friction,
  restitution, fixed, axis="z")`: cylinder aligned to local/world `"x"`, `"y"`,
  or `"z"`.
- `api.add_capsule(name, pos, radius, half_height, mass, rgba, friction,
  restitution, fixed, axis="z")`: rounded rod; useful for strikers and batons.
- `api.add_plane(name="ground", pos=(0.0, 0.0, 0.0), size=(5.0, 5.0),
  rgba=(...), friction=...)`: massless fixed plane. It accepts no `mass` and no
  `restitution`.
- `api.set_velocity(body_name, linear=(vx, vy, vz), angular=(wx, wy, wz))`:
  initial velocity for an existing non-fixed body.
- `api.set_rotation(body_name, quaternion=(qw, qx, qy, qz))`: initial
  orientation quaternion. The executor normalizes it.
- `api.add_hinge(name, parent, child, pos, axis=(ax, ay, az), limit=None)`:
  1-DOF rotational joint. `child` must name an existing body; `parent` is
  `"world"` or an existing non-plane body.
- `api.add_slide(name, parent, child, pos, axis=(ax, ay, az), limit=None)`:
  1-DOF prismatic joint. Parent/child rules match `add_hinge`.
- `api.add_spring(name, body_a, body_b, stiffness, damping,
  anchor_a=(0, 0, 0), anchor_b=(0, 0, 0), rest_length=None)`: damped spring
  between existing non-plane bodies; at least one endpoint must be dynamic.
- `api.set_gravity(vector)`: world gravity vector, usually negative z.
- `api.set_duration(seconds)`: total simulation time.
- `api.add_camera(name, pos, target)`: named camera only; no `fov` or
  `duration` keyword.

Invalid aliases that cause runtime failure: `color=`, `vel=`, `rot=`,
`angle=`, camera `fov=`, camera `duration=`, `api.set_friction(...)`, and
missing `child` or `pos` in `api.add_hinge(...)`.
"""


_SCENE_API_REFERENCES: dict[str, str] = {
    "throwing_pit_board": """\
## Scene API Focus

- `api.add_box("<platform>", pos=(...), size=(...), mass=..., rgba=(...),
  friction=..., restitution=..., fixed=True)`: fixed platforms and deflectors.
- `api.set_rotation("<deflector>", quaternion=(qw, qx, qy, qz))`: rotate a
  wall so branch removal can change a trajectory.
- `api.add_sphere("<neutral_ball>", pos=(...), radius=..., mass=...,
  rgba=(...), friction=..., restitution=..., fixed=False)`: free launched ball.
- `api.set_velocity("<neutral_ball>", linear=(vx, vy, vz),
  angular=(wx, wy, wz))`: modest, non-identical x/z launch velocities.
- `api.set_gravity(...)`, `api.set_duration(...)`, `api.add_camera(...)`: tune
  settling time and make platform gaps/deflectors visible.
""",
    "collision_arena": """\
## Scene API Focus

- `api.add_box(...)`, `api.add_sphere(...)`, `api.add_capsule(...)`: dynamic
  trigger/intermediate/target/movers plus optional fixed supports.
- `api.add_hinge("<hinge>", parent="world", child="<trigger>",
  pos=(px, py, pz), axis=(ax, ay, az))`: world-parent striker pivot.
- `api.set_velocity("<trigger>", linear=(vx, vy, vz),
  angular=(wx, wy, wz))`: give the trigger non-zero motion.
- `api.add_camera("cam0", pos=(...), target=(...))`: show trigger contact,
  transfer, and final target/relation state.
""",
    "motion_lab": """\
## Scene API Focus

- `api.add_box(...)`: fixed ramps, decks, pads, lanes, or stop walls.
- `api.set_rotation("<ramp>", quaternion=(qw, qx, qy, qz))`: tilt ramps when
  gravity should create speed.
- `api.add_sphere(...)`, `api.add_box(...)`: free motion_target and compare_pair
  bodies; matched bodies need separated starts.
- `api.set_velocity("<body>", linear=(vx, vy, vz), angular=(wx, wy, wz))`:
  create visible speed change without encoding comparison answers by launch bias.
- `api.add_camera("cam0", pos=(...), target=(...))`: capture path, slowing,
  stop, or rebound apex.
""",
}


_CODEGEN_PLAYBOOK = """\
## Codegen Rules

### Coordinate Frame
  - Right-handed world; z is UP. The ground plane sits at z = 0.
  - Layout on the table uses x (left/right) and y (depth); vertical motion uses z.
  - Gravity pulls down: use api.set_gravity((0.0, 0.0, -g)) with g > 0.
    Tune g with duration so contacts and settling happen inside the clip.
  - Typical camera: use a wide negative-y shot toward the causal region, using
    symbolic values such as pos=(cam_x, cam_y, cam_z),
    target=(target_x, target_y, target_z). Avoid zoomed-in top/side shots that
    hide the global layout or final state.

### Executor Hard Limits (DSL Validation)
  - Body centres must satisfy |x| < 1.5, |y| < 1.5, and 0 < z < 2.5.
    The bounds are strict: x=1.5, x=-1.5, y=1.5, y=-1.5, z=0, and z=2.5
    are rejected. Keep at least 0.1 m margin from workspace edges.
  - The only z=0 exception is api.add_plane("ground", pos=(0.0, 0.0, 0.0), ...).
    All boxes, spheres, cylinders, capsules, fixed scenery, and dynamic bodies
    need centre z > 0.
  - Box size is half-size per axis, not full width. Every size component must
    be in [0.01, 1.0]. Sphere radius, cylinder/capsule radius, and
    cylinder/capsule half_height must also be in [0.01, 1.0].
  - All non-plane primitives need mass in [0.01, 100.0], even when fixed=True.
    Use a positive mass such as 10.0 for fixed platforms and walls. Only
    add_plane has no mass argument.
  - Manifest region labels may describe areas near workspace edges, but every
    actual body centre coordinate must still obey the strict limits above.

### Python Structure
  - Reply with at most one short Plan line, then exactly one <propose>...</propose> block.
  - Inside <propose>: `def build_scene(api):` with only api.* calls, then a module-level
    `SCENE_MANIFEST = {...}` at the SAME indent as `def` (never indented inside the function).
  - Write all `api.*` calls first; write `SCENE_MANIFEST` last, outside
    `build_scene`. Put final numeric manifest values directly in the literal.
  - No imports, no classes, no markdown fences. Use tuple literals for vectors:
    pos=(x, y, z), linear=(vx, vy, vz) — not Python lists.
  - Body names: lowercase letters, digits, underscore; unique per scene.
  - Keep comments short: at most a few section labels. Do not put long
    derivations, retries, or geometry arithmetic inside `build_scene`.
  - Prompt literal numbers are calibration hints, not defaults. Do not assign
    identical positions, velocities, colours, friction, or restitution to repeated
    bodies unless equality is the tested invariant.

### Spawn Layering (build bottom to top)
  1. Optional api.add_plane("ground", pos=(0.0, 0.0, 0.0), ...) at z = 0.
  2. Fixed scenery (platforms, walls, ramps) with fixed=True; solid lowest point at z >= 0.
  3. Dynamic bodies (fixed=False) spawned ABOVE with a visible gap at t = 0:
     sphere centre z >= radius + 0.05; box centre z >= size_z + 0.05.
  4. api.set_velocity / api.set_rotation only on existing dynamic bodies.

### Non-Overlap At t = 0
  - Two spheres: centre distance > r1 + r2 + 0.05 m.
  - Box vs box: separate centres by more than sum of half-sizes on the closest axis + 0.05 m.
  - Sphere vs box: centre distance > r + max(box half_sizes) + 0.05 m on the closest axis.

Verifier-readable bodies must be free dynamic (fixed=False). Fixed walls/planes are never
read for trajectory questions.

add_hinge parent must be "world" or a body from add_box/add_sphere/add_cylinder/add_capsule.
Never hinge to an add_plane name (planes are not compiled bodies).

add_slide follows the same parent/child rule as add_hinge. add_spring endpoints
must be existing non-plane bodies.

### Quality Criteria
  - Make a visible causal chain: an initial condition or hidden parameter should
    change contact timing, final pose, final region, speed bucket, or attribution.
  - Prefer geometry, mass, friction, restitution, joints, springs, and timing over
    huge launch speeds. Difficulty comes from physical structure, not chaos.
  - Keep the evidence readable on video: contacts should not be hidden behind
    scenery, and final states should not be borderline.
  - Give distractors a purpose: near misses, delayed contacts, or alternative
    movers are useful only when the executor can still determine a unique GT.
  - Declare only manifest families that the scene genuinely supports. A scene may
    serve several families, but each one needs clear simulated evidence.

### Bad Outputs (zero reward)
  - Multiple drafts, chain-of-thought, or Python outside <propose>.
  - Missing </propose>, missing `SCENE_MANIFEST`, or `SCENE_MANIFEST` nested inside build_scene.
  - No free dynamic body, only fixed platforms/walls, or platform-only scenes.
  - All dynamic bodies have zero initial motion and no gravity/joint/spring path
    that can create visible motion or contact.
  - Using y as "height", positive-y gravity, or violating Executor hard limits above.
  - Using banned Python names: open, eval, exec, compile, __import__, globals,
    locals, vars, getattr, setattr, delattr, hasattr, breakpoint, input.
"""


_FEWSHOT_GUIDANCE = """\
## Validated Golden Example (cold-start only)

The example at the end is a complete code + manifest + executor-grounding
pattern:

- `build_scene` shows an executable apparatus layout.
- `SCENE_MANIFEST` shows role wiring for the scene template.
- The executor derives GT from the simulated trajectory or branch simulation.

Reuse the apparatus pattern and manifest role wiring, but change at least 3 of:
rgba, positions, sizes, masses, friction, restitution, camera, duration, and
moderate velocities. Do NOT submit a verbatim AST copy of the golden program.
"""


def _format_reference_block(reference_examples: Sequence[str]) -> str:
    """Render the "recent successful scenes" block, or an empty string.

    Mirrors the Stage-1 reference block (same "vary, do NOT copy" framing).
    The summaries are deterministic, coordinate-free, name-free projections
    of recent OK scenes, so they teach "what good looks like" without giving
    the Proposer a verbatim program to copy.
    """

    cleaned = tuple(str(s).strip() for s in reference_examples if str(s).strip())
    if not cleaned:
        return ""
    lines = "\n".join(f"  - ref{i + 1}: {summary}" for i, summary in enumerate(cleaned))
    return (
        "\n## Recent Successful Scenes (vary them, do NOT copy)\n\n"
        f"{lines}\n"
        "  These are coarse summaries of recent valid scenes for this "
        "scene_kind.\n"
        "  Treat them as anti-patterns: change the body layout, counts, "
        "colours,\n"
        "  masses/friction/restitution, and causal structure. Do not reproduce "
        "any\n"
        "  single set of bodies and coordinates verbatim.\n"
    )


def _format_reference_block_during_fewshot(
    reference_examples: Sequence[str],
) -> str:
    """Render parameter-band nudges that coexist with an executable fewshot.

    Unlike :func:`_format_reference_block` (main-training, no golden code), this
    block assumes the golden program is already shown and only pushes the
    Proposer to edit numeric / topology dimensions — never a second program.
    """

    cleaned = tuple(str(s).strip() for s in reference_examples if str(s).strip())
    if not cleaned:
        return ""
    lines = "\n".join(f"  - ref{i + 1}: {summary}" for i, summary in enumerate(cleaned))
    return (
        "\n## Recent OK Parameter Bands (vary the golden — do NOT copy)\n\n"
        "The golden example above teaches FORMAT and manifest wiring only.\n"
        "These summaries describe recent *successful* parameter / topology\n"
        "choices for this scene_kind (coordinate-free, name-free):\n\n"
        f"{lines}\n"
        "Required edits vs the shown golden:\n"
        "  - Change at least 3 of: rgba, positions, sizes, masses, friction,\n"
        "    restitution, camera, duration, and moderate velocities.\n"
        "  - Keep the golden's apparatus ROLES / SCENE_MANIFEST wiring, not\n"
        "    its numeric literals or body names.\n"
        "  - Do NOT paste the golden program verbatim; do NOT reconstruct\n"
        "    these ref summaries as executable code.\n"
    )


def _format_family_card(family: QuestionFamily) -> str:
    """Render the positive task card for one question family."""

    example = (
        family.stem_exemplars[0]
        if family.stem_exemplars
        else "(executor-generated family question)"
    )
    other_examples = "\n".join(
        f'  - "{q}"' for q in family.stem_exemplars[1:]
    )
    recommended = ", ".join(family.recommended_phrasings) or "(any supported)"
    lines = [
        f"\n## Expected Grounding: `{family.name}`",
        "",
        f'- Example generated question: "{example}"',
        f"- Options shown to Solver: {family.option_style}",
        f"- GT source: {family.evidence_rule}",
        f"- Preferred surface forms: {recommended}.",
        "- Scene obligation: build video evidence that makes the GT unique to "
        "the executor and visually recoverable by the Solver.",
    ]
    if other_examples:
        lines.append("")
        lines.append("Other compatible generated-question shapes:")
        lines.append(other_examples)
    return "\n".join(lines) + "\n"


def _format_required_family_block(family: Optional[QuestionFamily]) -> str:
    """Render the hard obligation for the sampled family."""

    if family is None:
        return ""
    return (
        f"\n## Required Family: `{family.name}`\n\n"
        f"- `SCENE_MANIFEST['families']` must include `{family.name}`.\n"
        "- The generated simulation must actually produce at least one executor "
        "VQA for this family; another family cannot be used as fallback.\n"
    )


def _format_scene_api_reference(scene_kind: str) -> str:
    """Return the scene-specific PhysDSL surface needed for this template."""

    block = _SCENE_API_REFERENCES.get(scene_kind, "").strip()
    if not block:
        return ""
    return "\n" + block + "\n"


def _format_scene_constraints(constraints: str) -> str:
    """Render compact per-template hard constraints."""

    lines = constraints.strip().splitlines()
    if not lines:
        return ""
    # Template docs are stored indented for readability in Python. Remove one
    # prompt-doc indent level so bullets render consistently.
    cleaned = "\n".join(line[2:] if line.startswith("  ") else line for line in lines)
    return "\n## Scene Hard Constraints\n\n" + cleaned.rstrip() + "\n"


_PREFLIGHT_CHECKS: dict[str, tuple[str, ...]] = {
    "throwing_pit_board": (
        "Ball body names are neutral (`ball_0`, `ball_1`, ...), never "
        "`ball_red`, `ball_blue`, `green_ball`, or `red_ball`.",
        "All body centres obey strict workspace limits: |x| < 1.5, |y| < 1.5, "
        "0 < z < 2.5. Do not place a board at x=0.5,1.5,2.5.",
        "Use a centered left/middle/right platform board: left at negative x, "
        "middle near zero, right positive but inside margin; avoid all-positive "
        "chains that repair clamps into the boundary.",
        "Throwing is z-axis falling: balls spawn high in z, descend under "
        "negative-z gravity, and y is only depth/lane spacing.",
        "Camera is a wide negative-y board shot; do not zoom in from positive y "
        "or top-down so the platform gaps and deflectors stay visible.",
        "No ball/deflector/platform overlaps at t=0; launched balls start above "
        "platform tops with radius clearance.",
        "The required family can produce at least one VQA: balls settle in "
        "executor-derived gaps, and branch families have removable walls that "
        "can change a route.",
    ),
    "collision_arena": (
        "All trigger/intermediate/target/movers stay inside |x|,|y| < 1.5 with "
        "margin; no x=2.0/2.5/3.0 body chains.",
        "Dynamic bodies are staggered or separated by clear gaps; never stack "
        "trigger/intermediate/target/movers on one dense line.",
        "Mover candidates are outside the main strike lane and not initially "
        "touching the target or intermediate.",
        "The trigger has non-zero linear or angular motion and visibly contacts "
        "the intermediate inside the clip.",
        "Camera is a wide negative-y shot that shows trigger, transfer, target, "
        "and candidate movers; avoid zooming onto one contact.",
        "The required family can produce at least one VQA with no geometry repair "
        "needed for first-submit validity.",
    ),
    "motion_lab": (
        "Every `size` component is a half-extent <= 1.0; do not use "
        "`size=(2.0, ...)` or tall block ramps.",
        "Surfaces are thin plates/decks/ramps with visible clearance; balls do "
        "not start embedded in a ramp/deck.",
        "For friction comparison, matched bodies have nonzero identical initial "
        "velocity or a shared downhill gravity path, plus a visible final-speed "
        "or travel-distance contrast.",
        "For speed-profile and pairwise-speed questions, design visible "
        "acceleration/deceleration, near-constant motion, or a same/consistent/"
        "crossover speed relation; do not rely on vague 'moving vs stopped' text.",
        "Compare bodies start separated from each other and from surfaces; no "
        "same-position or same-lane overlap.",
        "Camera is a wide negative-y shot that shows the full motion path, not a "
        "zoomed-in view of one body.",
        "The required family can produce at least one VQA: motion_target starts "
        "with speed >= 0.10 m/s, or compare_pair shows a visible contrast.",
    ),
}


def _format_preflight_checks(scene_kind: str) -> str:
    """Render a final executor-facing checklist before output."""

    checks = _PREFLIGHT_CHECKS.get(scene_kind, ())
    if not checks:
        return ""
    lines = "\n".join(f"- {check}" for check in checks)
    return "\n## Preflight Checks Before <propose>\n\n" + lines + "\n"


def _format_wiring_card(scene_kind: str, card: str) -> str:
    """Render the main-training fewshot distillation card."""

    cleaned = card.strip()
    if not cleaned:
        raise ValueError(f"scene_kind {scene_kind!r} has no main_training_wiring_card")
    return "\n" + cleaned + "\n"


def build_proposer_scene_template_prompt(
    scene_kind: str,
    *,
    question_family: Optional[str] = None,
    include_fewshot: bool = True,
    fewshot_index: int = 0,
    reference_examples: Sequence[str] = (),
) -> str:
    """Return the unified scene-manifest Proposer prompt for ``scene_kind``.

    Args:
        scene_kind: A registered scene-template key (e.g.
            ``"throwing_pit_board"``, ``"collision_arena"`` or
            ``"motion_lab"``).
        question_family: Optional :mod:`physevolve.skills.question_families`
            key. When given, the family must be served by ``scene_kind`` and
            its task card is injected above the scene section.
        include_fewshot: When True (cold start) the executable golden fewshot
            and its reuse guidance are appended, teaching format / manifest /
            executor success. When False (main training) the executable
            fewshot is dropped entirely so the policy cannot collapse onto a
            verbatim copy; the non-code contract scaffolding
            (``scene_constraints`` + ``manifest_schema_doc`` + family card) is
            always kept.
        fewshot_index: Deterministic cold-start variant selector. Ignored when
            ``include_fewshot=False``.
        reference_examples: Coordinate-free one-line summaries of recent OK
            scenes (from the Stage-2 ReferenceBuffer). When
            ``include_fewshot=False`` they form the main-training "vary, do
            NOT copy" block. When ``include_fewshot=True`` they append after
            the golden as parameter-band nudges (recipe hybrid; empty is a no-op).

    Raises:
        ValueError: If ``scene_kind`` is not registered or ``question_family``
            does not bind to it.
    """

    from physevolve.skills.scene_templates import (
        get_scene_template_registry,
        select_fewshot_example,
    )

    template = get_scene_template_registry().lookup(scene_kind)

    family: Optional[QuestionFamily] = None
    family_card = ""
    if question_family is not None:
        family = lookup_family(question_family)
        if family.scene_kind != scene_kind:
            raise ValueError(
                f"question_family {question_family!r} has scene_kind "
                f"{family.scene_kind!r}, not scene_kind {scene_kind!r}"
            )
        if question_family not in template.families:
            raise ValueError(
                f"question_family {question_family!r} is not served by "
                f"scene_kind {scene_kind!r} (serves {template.families})"
            )
        family_card = _format_family_card(family)

    families = ", ".join(template.families)
    if include_fewshot:
        selected_fewshot = select_fewshot_example(template, fewshot_index)
        exemplar_block = (
            _FEWSHOT_GUIDANCE
            + f"{selected_fewshot.rstrip()}\n"
            + _format_reference_block_during_fewshot(reference_examples)
        )
    else:
        # Main-training mode: NO executable golden fewshot. Use one integrated
        # wiring card distilled from the golden, not split prose/idiom layers.
        exemplar_block = (
            _format_wiring_card(scene_kind, template.main_training_wiring_card)
            + _format_reference_block(reference_examples)
        )
    return (
        _ROLE_AND_RULES
        + _PHYSDSL_API_REFERENCE
        + _format_scene_api_reference(scene_kind)
        + _CODEGEN_PLAYBOOK
        + family_card
        + _format_required_family_block(family)
        + f"\n## Scene Template: `{scene_kind}`\n\n"
        + "This reusable scene can auto-generate questions for any of these "
        "families:\n"
        + f"  {families}\n"
        + "Declare in SCENE_MANIFEST['families'] the SUBSET your scene actually "
        "supports (at least the one above). The SAME scene may serve several "
        "families at once - the executor emits every feasible one.\n"
        + _format_scene_constraints(template.scene_constraints)
        + f"\n### SCENE_MANIFEST Schema\n{template.manifest_schema_doc.rstrip()}\n"
        + exemplar_block
        + _format_preflight_checks(scene_kind)
        + "\n## Output Format\n"
        + "Optionally one short plain-text line naming the principle you test, "
        "then one <propose> block with `def build_scene(api):` and a literal "
        "`SCENE_MANIFEST = {...}`. Never write question text, options, answers, "
        "VQA_SPEC, or get_vqa.\n"
    )


# Generic role-and-rules block. Existing
# smoke tests import ``PROPOSER_STAGE2_TEMPLATE``; the RL training path always
# calls ``build_proposer_scene_template_prompt`` with an explicit scene_kind.
PROPOSER_STAGE2_TEMPLATE: str = (
    _ROLE_AND_RULES + _PHYSDSL_API_REFERENCE + _CODEGEN_PLAYBOOK
)


__all__ = [
    "PROPOSER_STAGE2_TEMPLATE",
    "build_proposer_scene_template_prompt",
]
