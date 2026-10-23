"""Stage-1 Proposer prompts for executable PhysDSL scenes.

The builder selects a physics skill and derives its difficulty from rolling
Solver accuracy. Prompts contain scene guidance and optional reference summaries.
The response must include one ``<propose>`` block with ``build_scene(api)``;
leading prose and optional reasoning tags are accepted by the parser.

Stage-2 scene-manifest prompts live in :mod:`physevolve.prompts.proposer_stage2`.
DSL bounds and parameter levels are enforced by :mod:`physevolve.engine.physdsl`.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

from physevolve.engine.schema import FRICTION_LEVELS, MASS_LEVELS

# ─── Physics-skill vocabulary ────────────────────────────────

PHYSICS_SKILL_TARGETS: tuple[str, ...] = (
    "dynamics_collision",
    "dynamics_chain_collision",
    "dynamics_throwing",
    "dynamics_manipulation",
    "dynamics_motion_state",
    "property_mass_contrast",
    "property_parameter_identification",
    "property_bounce_restitution",
)
"""Closed physics-skill vocabulary used by the Stage-1 curriculum."""

# ─── Adaptive-difficulty vocabulary ──────────────

DIFFICULTY_LEVELS: tuple[str, ...] = ("easy", "medium", "hard")
"""Closed difficulty vocabulary, orthogonal to :data:`PHYSICS_SKILL_TARGETS`.

The builder samples a skill with ``adaptive_pick`` and derives its difficulty
from rolling Solver accuracy with ``derive_difficulty``. The prompt then renders
the corresponding skill guidance and difficulty band.
"""


# Compatibility alias.
MECHANISM_TARGETS: tuple[str, ...] = PHYSICS_SKILL_TARGETS
"""Deprecated alias for :data:`PHYSICS_SKILL_TARGETS`.

Existing callers referenced ``MECHANISM_TARGETS``; new code should use
``PHYSICS_SKILL_TARGETS``. The two names refer to the same tuple object
for compatibility.
"""

# ─── Constants appearing inside the template text ────────────────────

_MASS_LIST = ", ".join(str(m) for m in MASS_LEVELS)
_FRICTION_LIST = ", ".join(str(f) for f in FRICTION_LEVELS)

_NO_COPY_CLAUSE = (
    "You MUST NOT reuse the example's body names, positions, sizes, masses, "
    "friction, restitution, or velocities verbatim. Vary at least 3 of these "
    "dimensions. Do not make a scene harder only by increasing speed. "
    "The example is a structural hint, not a template to copy."
)


# ─── Difficulty bands ────────────────────────────────────────────────

_DIFFICULTY_HINTS: dict[str, str] = {
    "easy": """\
  - Target difficulty: EASY.
  - Use 1 to 2 dynamic bodies. Single primary interaction event
    (one collision, one bounce, or one launched body).
  - Keep every initial speed below 1.5 m/s.
  - Total simulation duration <= 1.5 seconds (`api.set_duration`).
  - Keep mass/friction differences within one level on the lists above;
    no extreme contrasts.
""",
    "medium": """\
  - Target difficulty: MEDIUM.
  - Use 2 to 4 dynamic bodies and at least one structural difficulty source:
    delayed second contact, near-miss distractor, hidden mass/friction contrast,
    bounce-then-settle, or a short chain reaction.
  - Prefer moderate speeds (1.0-2.4 m/s). Increasing speed alone is not
    a valid difficulty increase.
  - Total simulation duration 1.8 to 2.4 seconds.
""",
    "hard": """\
  - Target difficulty: HARD.
  - Use 4 to 8 dynamic bodies. Combine at least TWO non-speed sources of
    difficulty: causal chain, distractor body, near miss, hidden parameter
    contrast, delayed second event, bounce/settle ambiguity, or counterfactual
    obstacle.
  - Keep most initial speeds <= 2.6 m/s (throwing may use one vertical
    component up to 3.0 m/s). Do NOT solve HARD by using 3.5-4.0 m/s impacts.
  - Total simulation duration 2.5 to 3.0 seconds.
  - The outcome must remain visually disentangled and executor-verifiable.
""",
}

assert set(_DIFFICULTY_HINTS) == set(DIFFICULTY_LEVELS), (
    "every DIFFICULTY_LEVELS entry must have a paired hint block"
)


def _format_reference_block(reference_examples: Sequence[str]) -> str:
    """Render the "recent reference scenes" block, or an empty string.

    Empty references produce an empty block.
    """

    cleaned = tuple(str(s).strip() for s in reference_examples if str(s).strip())
    if not cleaned:
        return ""
    lines = "\n".join(
        f"  - ref{i + 1}: {summary}" for i, summary in enumerate(cleaned)
    )
    return (
        "\nRecent reference scenes (vary them, do NOT copy):\n"
        f"{lines}\n"
        "  These are short summaries of recent scenes from this skill bucket.\n"
        "  Treat them as anti-patterns: change the body-count band, causal\n"
        "  structure, hidden parameters, or distractor layout. Do not merely\n"
        "  reuse the same template with larger velocities.\n"
    )

# ─── Base template shared by every variant ───────────────────────────

_BASE_TEMPLATE: str = f"""\
You are the Proposer in PhysEvolve. Your job is to write ONE executable PhysDSL
scene for a physics simulator. Another AI will watch the rendered video and predict
what happens.

Your output will be executed automatically. Valid executable code is more important
than creative explanation.

API available on the `api` object:
  api.add_box(name, pos, size, mass, rgba, friction, restitution, fixed)
  api.add_sphere(name, pos, radius, mass, rgba, friction, restitution, fixed)
  api.add_cylinder(name, pos, radius, half_height, mass, rgba, friction, restitution, fixed)
  api.add_capsule(name, pos, radius, half_height, mass, rgba, friction, restitution, fixed)
  api.add_plane(name, pos, size, rgba, friction)
  api.set_velocity(body_name, linear, angular)
  api.set_rotation(body_name, quaternion)
  api.add_hinge(name, parent, child, pos, axis, limit)
  api.add_slide(name, parent, child, pos, axis, limit)
  api.add_spring(name, body_a, body_b, stiffness, damping, anchor_a=..., anchor_b=..., rest_length=None)
  api.set_gravity(vector)
  api.set_duration(seconds)
  api.add_camera(name, pos, target)

Hard constraints:
  - Output exactly one <propose> block.
  - The <propose> block must contain exactly one top-level `def build_scene(api):`.
  - Do not use Markdown fences. Do not explain after </propose>.
  - No imports. No math module. Use only literals, tuples, lists, loops, and basic arithmetic.
  - Body names must be lowercase letters/numbers/underscore and unique.
  - Workspace: |x| < 1.5, |y| < 1.5, 0 < z < 2.5.
  - Avoid overlaps at t=0. Keep centers separated by more than the sum of radii/half-sizes.
  - Use 1 to 20 dynamic bodies. Fixed platforms and the default ground do not count.
  - A default ground plane and default camera are added automatically. Do not call api.add_plane()
    unless you intentionally need a custom ground.
  - Mass must be one of: {_MASS_LIST} kg.
  - Friction must be one of: {_FRICTION_LIST}.
  - Restitution must be one of: 0.0, 0.5, 0.9.
  - Object sizes must be between 0.01 and 1.0 meters.
  - For add_hinge / add_slide, parent="world" is allowed.
  - For add_spring: stiffness in [0.1, 10000], damping in [0.0, 1000],
    anchor_a / anchor_b are body-local offsets in [-2, 2] per axis.

Response style:
  - You may write a short plain-text construction plan (one or two sentences)
    BEFORE the <propose> block.
  - Do not brainstorm alternatives. Do not self-correct repeatedly.
  - Do not put Python code outside <propose>. Code belongs only inside
    <propose>...</propose>.
"""

# ─── Per-skill guidance + example skeletons ──────────────────────────


def _section(
    focus: str,
    example: str,
    *,
    difficulty_block: str = "",
    reference_block: str = "",
) -> str:
    """Format the skill-conditioned tail that every variant ends with.

    Layout (top-to-bottom inside the rendered prompt):

    1. ``Skill focus`` block (physics-skill guidance).
    2. ``Difficulty band`` block (optional, only when ``difficulty_block``
       is non-empty). Controls body count / velocity magnitude / event
       complexity so the adaptive sampler has a knob to escalate
       difficulty independently of the skill axis.
    3. ``Recent reference scenes`` block (optional, only when the
       reference buffer has produced at least one entry for this skill).
    4. ``Anti-copy rule`` (still applies to the example skeleton).
    5. ``Output format`` + example skeleton.
    6. ``Bad outputs to avoid`` checklist.
    """

    difficulty_section = (
        f"\nDifficulty band:\n{difficulty_block.rstrip()}\n"
        if difficulty_block
        else ""
    )

    return (
        f"\nSkill focus:\n{focus.rstrip()}\n"
        f"{difficulty_section}"
        f"{reference_block}"
        f"\nAnti-copy rule:\n  - {_NO_COPY_CLAUSE}\n"
        "\nOutput format:\n"
        "\nOptionally write a one-sentence plan, then immediately the "
        "<propose> block:\n"
        "<propose>\ndef build_scene(api):\n    # executable PhysDSL code only\n</propose>\n"
        f"\nExample skeleton (vary it, do not copy):\n{example.rstrip()}\n"
        "\nBad outputs to avoid:\n"
        "  - Long repeated reasoning.\n"
        "  - Multiple scene options.\n"
        "  - Python code outside the <propose> block.\n"
        "  - Missing <propose> or missing </propose>.\n"
        "  - Calling unknown methods such as api.add_ramp or api.add_domino_chain.\n"
        "  - Placing objects at z=0 when they have positive radius/height.\n"
        "  - Copying the example skeleton's body names, positions, masses, or velocities verbatim.\n"
    )


_GENERIC_FOCUS = """\
  - Make physical dynamics matter.
  - Pick any of: collision, sliding, bouncing, projectile motion, contact
    manipulation, or clear mass/friction/restitution contrast.
  - Prefer structural difficulty: hidden parameters, distractors, delayed
    events, and causal chains. Avoid speed-only scenes.
  - Avoid fragile scenes with deep interpenetration, exact balancing, or many
    objects moving past each other simultaneously.
"""

_GENERIC_EXAMPLE = """\
Plan: a heavy block sits on the ground while a light sphere flies into it
from the left with a moderate horizontal velocity, transferring momentum.
<propose>
def build_scene(api):
    api.add_sphere("striker", pos=(-0.7, 0.0, 0.18), radius=0.14, mass=0.5,
                   rgba=(0.95, 0.6, 0.2, 1.0), friction=0.5, restitution=0.5, fixed=False)
    api.add_box("target_block", pos=(0.25, 0.0, 0.18), size=(0.18, 0.18, 0.18), mass=5.0,
                rgba=(0.2, 0.6, 0.9, 1.0), friction=0.5, restitution=0.0, fixed=False)
    api.set_velocity("striker", linear=(1.8, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
    api.set_duration(2.0)
</propose>
"""


_SKILL_SECTIONS: dict[str, tuple[str, str]] = {
    "dynamics_collision": (
        """\
  - The dominant event is a single clear collision that transfers momentum
    between two free dynamic bodies. Targets dynamics/collision.
  - At least one body has nonzero initial velocity; both bodies are free
    (fixed=False) so the post-collision motion is observable.
  - For harder cases, add a distractor or delayed downstream contact; do not
    just raise the incoming speed.
  - Do not pin the target to the ground; avoid spring/hinge/slide constraints
    for this skill.
""",
        """\
Plan: a puck slides toward a stationary cube; the collision sends the cube
forward and slows the puck. Both stay on the ground.
<propose>
def build_scene(api):
    api.add_cylinder("puck", pos=(-0.6, 0.05, 0.05), radius=0.10, half_height=0.04,
                     mass=1.0, rgba=(0.9, 0.2, 0.2, 1.0), friction=0.2, restitution=0.5,
                     fixed=False, axis="z")
    api.add_box("cube", pos=(0.35, 0.05, 0.10), size=(0.10, 0.10, 0.10), mass=0.5,
                rgba=(0.2, 0.8, 0.3, 1.0), friction=0.2, restitution=0.5, fixed=False)
    api.set_velocity("puck", linear=(1.8, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
    api.set_duration(2.0)
</propose>
""",
    ),
    "dynamics_chain_collision": (
        """\
  - The dominant event is a sequential chain of collisions between 3 to 5
    dynamic bodies arranged in a line. Targets dynamics/collision
    multi-body reasoning.
  - The first body in the chain has the initial velocity; the others are at
    rest. Keep spacing larger than the sum of their radii/half-sizes.
  - For harder cases, vary mass or restitution along the chain so the final
    body motion is not determined by speed alone.
  - Do not add springs/hinges; the cascade comes from collision propagation.
""",
        """\
Plan: four spheres are lined up along the x-axis; the leftmost one is
launched into them and the impact propagates down the line.
<propose>
def build_scene(api):
    api.add_sphere("ball_1", pos=(-0.9, 0.0, 0.10), radius=0.08, mass=1.0,
                   rgba=(1.0, 0.2, 0.2, 1.0), friction=0.2, restitution=0.9, fixed=False)
    api.add_sphere("ball_2", pos=(-0.4, 0.0, 0.10), radius=0.08, mass=1.0,
                   rgba=(0.2, 1.0, 0.2, 1.0), friction=0.2, restitution=0.9, fixed=False)
    api.add_sphere("ball_3", pos=(0.1, 0.0, 0.10), radius=0.08, mass=1.0,
                   rgba=(0.2, 0.2, 1.0, 1.0), friction=0.2, restitution=0.9, fixed=False)
    api.add_sphere("ball_4", pos=(0.6, 0.0, 0.10), radius=0.08, mass=1.0,
                   rgba=(0.9, 0.9, 0.2, 1.0), friction=0.2, restitution=0.9, fixed=False)
    api.set_velocity("ball_1", linear=(2.2, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
    api.set_duration(2.0)
</propose>
""",
    ),
    "dynamics_throwing": (
        """\
  - The dominant motion is a projectile being thrown: a body is launched with
    a combined horizontal AND vertical initial velocity, then follows a
    ballistic arc under gravity until it lands or hits another body. Targets
    dynamics/throwing.
  - Use exactly one launched dynamic body with non-zero ``linear[0]`` (or
    ``linear[1]``) AND non-zero ``linear[2]`` (>= 1.0 m/s upward). The
    horizontal component should be in [1.0, 4.0] m/s and the vertical
    component in [1.0, 4.0] m/s so the arc is visible inside the workspace.
  - Place the launch point so the apex stays under z < 2.4 and the body
    lands inside the workspace. The target can be the ground or a fixed box.
  - For harder cases, use an obstacle, wall, or landing region boundary so
    destination depends on trajectory geometry, not only launch speed.
  - Do not use springs, hinges, or slide joints.
""",
        """\
Plan: a small sphere is thrown forward and upward from the left; it follows
a parabolic arc under gravity and lands inside the workspace.
<propose>
def build_scene(api):
    api.add_sphere("projectile", pos=(-1.0, 0.0, 0.30), radius=0.08, mass=0.5,
                   rgba=(0.95, 0.55, 0.15, 1.0), friction=0.5, restitution=0.5, fixed=False)
    api.set_velocity("projectile", linear=(2.5, 0.0, 2.5), angular=(0.0, 0.0, 0.0))
    api.add_box("landing_pad", pos=(0.7, 0.0, 0.05), size=(0.30, 0.30, 0.05), mass=10.0,
                rgba=(0.4, 0.4, 0.4, 1.0), friction=1.0, restitution=0.0, fixed=True)
    api.set_duration(2.0)
</propose>
""",
    ),
    "dynamics_manipulation": (
        """\
  - The dominant event is contact-driven manipulation: one moving body
    pushes one or more other bodies via sustained pushing contact (not a
    single billiard-style elastic collision). Targets
    dynamics/manipulation.
  - Use one "pusher" body with low restitution (0.0) and 2 to 4 "pushed"
    bodies arranged so the pusher contacts them gradually. The pushed
    bodies should be free (fixed=False) and visibly displaced by the
    pusher.
  - For harder cases, use staggered contact points or different pushed-body
    masses; do not make the plough faster as the main difficulty knob.
  - Keep friction in [0.2, 1.0] so contact is firm but motion is still
    observable. Do not use springs or hinges.
""",
        """\
Plan: a flat plough-shaped box drives into a small cluster of cubes from the
side; contact stays engaged and the cubes are pushed several centimetres
forward.
<propose>
def build_scene(api):
    api.add_box("plough", pos=(-0.9, 0.0, 0.12), size=(0.20, 0.30, 0.12), mass=5.0,
                rgba=(0.3, 0.3, 0.3, 1.0), friction=0.5, restitution=0.0, fixed=False)
    api.add_box("cube_a", pos=(-0.20, -0.10, 0.08), size=(0.08, 0.08, 0.08), mass=0.5,
                rgba=(0.9, 0.2, 0.2, 1.0), friction=0.5, restitution=0.0, fixed=False)
    api.add_box("cube_b", pos=(-0.20,  0.10, 0.08), size=(0.08, 0.08, 0.08), mass=0.5,
                rgba=(0.2, 0.9, 0.2, 1.0), friction=0.5, restitution=0.0, fixed=False)
    api.add_box("cube_c", pos=( 0.05,  0.00, 0.08), size=(0.08, 0.08, 0.08), mass=0.5,
                rgba=(0.2, 0.2, 0.9, 1.0), friction=0.5, restitution=0.0, fixed=False)
    api.set_velocity("plough", linear=(2.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
    api.set_duration(2.5)
</propose>
""",
    ),
    "dynamics_motion_state": (
        """\
  - The scene must let an observer judge whether each dynamic body ends up
    MOVING or STATIC at the final frame. Targets dynamics/motion
    (the "is this body still moving?" question).
  - Use 2 or 3 dynamic bodies whose friction levels are deliberately
    different (e.g. one at 0.05, one at 1.0 or 2.0) and give each a similar
    horizontal launch velocity (~2.0 m/s). One body should clearly stop
    before the simulation ends; another should clearly still be moving.
  - Difficulty should come from matched initial speeds plus hidden friction
    or collision effects, not from one body being obviously faster.
  - Keep geometry simple. Do not use springs, hinges, or bouncing.
""",
        """\
Plan: two pucks are launched with the same speed but very different friction
levels. The high-friction puck halts; the low-friction puck still slides at
the end of the clip.
<propose>
def build_scene(api):
    api.add_cylinder("smooth_puck", pos=(-1.0, -0.30, 0.05), radius=0.10, half_height=0.04,
                     mass=1.0, rgba=(0.3, 0.7, 0.9, 1.0), friction=0.05, restitution=0.0,
                     fixed=False, axis="z")
    api.add_cylinder("grippy_puck", pos=(-1.0,  0.30, 0.05), radius=0.10, half_height=0.04,
                     mass=1.0, rgba=(0.9, 0.4, 0.2, 1.0), friction=2.0, restitution=0.0,
                     fixed=False, axis="z")
    api.set_velocity("smooth_puck", linear=(2.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
    api.set_velocity("grippy_puck", linear=(2.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
    api.set_duration(2.5)
</propose>
""",
    ),
    "property_mass_contrast": (
        """\
  - The scene must contain at least two free dynamic bodies whose mass
    differs by at least one full level on MASS_LEVELS (e.g. 0.1 vs 5.0).
    Targets property/mass.
  - Drive an interaction that makes the contrast observable: collision,
    pushing, or a shared support. Do not pin either body.
  - Keep launch speeds moderate and similar-looking geometry; make the mass
    difference the dominant explanatory variable.
""",
        """\
Plan: a very light sphere collides with a much heavier cube. After the
impact the light sphere rebounds while the heavy cube barely moves.
<propose>
def build_scene(api):
    api.add_sphere("featherweight", pos=(-0.7, 0.05, 0.10), radius=0.10, mass=0.1,
                   rgba=(0.9, 0.7, 0.7, 1.0), friction=0.2, restitution=0.9, fixed=False)
    api.add_box("heavyweight", pos=(0.20, 0.05, 0.15), size=(0.15, 0.15, 0.15), mass=10.0,
                rgba=(0.2, 0.2, 0.2, 1.0), friction=0.5, restitution=0.5, fixed=False)
    api.set_velocity("featherweight", linear=(2.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
    api.set_duration(2.0)
</propose>
""",
    ),
    "property_parameter_identification": (
        """\
  - The scene must let an observer infer hidden parameters (mass and/or
    friction) from motion alone. Targets property/mass and
    property/attribute reasoning.
  - Use at least two free dynamic bodies whose mass OR friction levels are
    different but whose geometry is identical, so motion is the only cue.
  - Pick an interaction (collision, sliding) that produces clearly distinct
    trajectories for the two bodies under matched initial speeds.
""",
        """\
Plan: two geometrically identical cubes start side by side with the same
launch velocity but different friction levels. They travel different
distances.
<propose>
def build_scene(api):
    api.add_box("twin_low_fric", pos=(-0.8, -0.30, 0.10), size=(0.10, 0.10, 0.10), mass=1.0,
                rgba=(0.9, 0.5, 0.2, 1.0), friction=0.05, restitution=0.0, fixed=False)
    api.add_box("twin_high_fric", pos=(-0.8, 0.30, 0.10), size=(0.10, 0.10, 0.10), mass=1.0,
                rgba=(0.2, 0.5, 0.9, 1.0), friction=2.0, restitution=0.0, fixed=False)
    api.set_velocity("twin_low_fric", linear=(1.8, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
    api.set_velocity("twin_high_fric", linear=(1.8, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
    api.set_duration(2.5)
</propose>
""",
    ),
    "property_bounce_restitution": (
        """\
  - The dominant motion is a body falling under gravity and bouncing one or
    more times on a fixed surface. Targets property/attribute
    (restitution / bounciness).
  - Use restitution = 0.9 on at least one of the participating bodies (sphere
    or plane). The fixed surface may be a fixed box or the default ground.
  - Harder variants should compare bounce height, settling time, or a second
    contact after bounce. Avoid horizontal speed that overshadows restitution.
""",
        """\
Plan: a high-restitution sphere is released above a heavy fixed plate; it
bounces several times under gravity.
<propose>
def build_scene(api):
    api.add_box("plate", pos=(0.0, 0.0, 0.05), size=(0.5, 0.5, 0.05), mass=10.0,
                rgba=(0.4, 0.4, 0.4, 1.0), friction=0.5, restitution=0.9, fixed=True)
    api.add_sphere("bouncer", pos=(0.0, 0.0, 1.6), radius=0.10, mass=0.5,
                   rgba=(1.0, 0.9, 0.1, 1.0), friction=0.2, restitution=0.9, fixed=False)
    api.set_duration(2.0)
</propose>
""",
    ),
}

assert set(_SKILL_SECTIONS) == set(PHYSICS_SKILL_TARGETS), (
    "every PHYSICS_SKILL_TARGETS entry must have a paired section"
)


def _render_prompt(
    skill_target: Optional[str],
    difficulty_level: Optional[str] = None,
    reference_examples: Sequence[str] = (),
) -> str:
    if skill_target is None:
        focus, example = _GENERIC_FOCUS, _GENERIC_EXAMPLE
    else:
        if skill_target not in _SKILL_SECTIONS:
            raise ValueError(
                f"unknown skill_target {skill_target!r}; expected one "
                f"of {PHYSICS_SKILL_TARGETS}"
            )
        focus, example = _SKILL_SECTIONS[skill_target]

    difficulty_block = ""
    if difficulty_level is not None:
        if difficulty_level not in _DIFFICULTY_HINTS:
            raise ValueError(
                f"unknown difficulty_level {difficulty_level!r}; expected "
                f"one of {DIFFICULTY_LEVELS}"
            )
        difficulty_block = _DIFFICULTY_HINTS[difficulty_level]

    reference_block = _format_reference_block(reference_examples)

    return _BASE_TEMPLATE + _section(
        focus,
        example,
        difficulty_block=difficulty_block,
        reference_block=reference_block,
    )


# Generic prompt for callers without skill conditioning.
# Smoke tests and standalone callers import
# ``PROPOSER_STAGE1_TEMPLATE``; the training path now goes through the
# skill-aware builder.
PROPOSER_STAGE1_TEMPLATE: str = _render_prompt(None)


def build_proposer_stage1_prompt(
    skill_target: Optional[str] = None,
    *,
    difficulty_level: Optional[str] = None,
    reference_examples: Sequence[str] = (),
    mechanism_target: Optional[str] = None,
) -> str:
    """Return the Stage-1 Proposer prompt for an optional physics skill.

    Args:
        skill_target: One of :data:`PHYSICS_SKILL_TARGETS`, or ``None`` to
            request the generic prompt (used by standalone smoke
            paths and prompt-only diagnostics). RL training should always
            pass an explicit target so each GRPO group sees a distinct
            anchor.
        difficulty_level: One of :data:`DIFFICULTY_LEVELS`, or ``None`` to
            omit the difficulty band. Training passes an explicit level.
        reference_examples: Optional sequence of short one-line summaries
            of recently-validated OK scenes for this skill. When
            non-empty, a "Recent reference scenes (vary, do NOT copy)"
            block is appended after the difficulty band. Each entry is
            a plain string; the builder does **not** validate semantic
            content beyond non-emptiness so callers stay decoupled from
            :class:`ReferenceEntry`.
        mechanism_target: Deprecated alias for ``skill_target``. Emits a
            :class:`DeprecationWarning` and is forwarded as-is.

    Returns:
        A self-contained prompt string ending with the
        ``<propose>...</propose>`` scaffold (no ``<think>`` block under
        Qwen3-VL Instruct).

    Raises:
        ValueError: If both ``skill_target`` and ``mechanism_target`` are
            provided (ambiguous), if the resolved skill is not in
            :data:`PHYSICS_SKILL_TARGETS`, or if ``difficulty_level``
            is not in :data:`DIFFICULTY_LEVELS`.
    """

    if mechanism_target is not None:
        if skill_target is not None and skill_target != mechanism_target:
            raise ValueError(
                "pass either skill_target= or mechanism_target=, not both"
            )
        warnings.warn(
            "build_proposer_stage1_prompt(mechanism_target=...) is "
            "deprecated; use skill_target= instead "
            "(see PHYSICS_SKILL_TARGETS).",
            DeprecationWarning,
            stacklevel=2,
        )
        skill_target = mechanism_target
    return _render_prompt(
        skill_target,
        difficulty_level=difficulty_level,
        reference_examples=reference_examples,
    )


__all__ = [
    "DIFFICULTY_LEVELS",
    "MECHANISM_TARGETS",
    "PHYSICS_SKILL_TARGETS",
    "PROPOSER_STAGE1_TEMPLATE",
    "build_proposer_stage1_prompt",
]
