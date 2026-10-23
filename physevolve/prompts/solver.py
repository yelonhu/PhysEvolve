"""Solver prompts for Stage-1 Forward, Inverse, and Parameter tasks.

The serving layer supplies video frames. These templates request a structured
``<answer>`` block; optional reasoning prose may precede it.
"""

from __future__ import annotations

from typing import Iterable

from physevolve.engine.schema import (
    FRICTION_LEVELS,
    MASS_LEVELS,
    BodyDef,
    BodyTrajectory,
    SceneDef,
    world_to_grid,
)


def _mass_level(mass: float) -> int:
    """Return the 1-indexed level for a mass snapped to ``MASS_LEVELS``."""

    for i, m in enumerate(MASS_LEVELS):
        if abs(mass - m) < 1e-6:
            return i + 1
    raise ValueError(f"mass {mass} not in MASS_LEVELS")


def _friction_level(friction: float) -> int:
    for i, f in enumerate(FRICTION_LEVELS):
        if abs(friction - f) < 1e-6:
            return i + 1
    raise ValueError(f"friction {friction} not in FRICTION_LEVELS")


def _dyn_bodies(scene_def: SceneDef) -> list[BodyDef]:
    """Non-fixed, non-plane bodies in Proposer-authored order."""

    return [
        b
        for b in scene_def.bodies
        if not b.fixed and b.geom_type.value != "plane"
    ]


def _format_params_block(scene_def: SceneDef) -> str:
    """Lines like ``- ball: mass_level=3 (1.0 kg), friction_level=2 (0.2)``."""

    lines: list[str] = []
    for b in _dyn_bodies(scene_def):
        ml = _mass_level(b.mass)
        fl = _friction_level(b.friction_slide)
        lines.append(
            f"- {b.name}: mass_level={ml} ({b.mass} kg), "
            f"friction_level={fl} ({b.friction_slide})"
        )
    return "\n".join(lines) if lines else "- (no dynamic bodies)"


def _format_initial_block(scene_def: SceneDef) -> str:
    """Initial position (meters) and linear velocity (m/s) per body."""

    lines: list[str] = []
    for b in _dyn_bodies(scene_def):
        lines.append(
            f"- {b.name}: pos={_fmt_vec(b.pos)} m, "
            f"initial_velocity={_fmt_vec(b.initial_lin_vel)} m/s"
        )
    return "\n".join(lines) if lines else "- (no dynamic bodies)"


def _format_final_block(
    scene_def: SceneDef, trajectories: dict[str, BodyTrajectory]
) -> str:
    """Final grid position + status per body (no raw continuous pos)."""

    lines: list[str] = []
    for b in _dyn_bodies(scene_def):
        traj = trajectories[b.name]
        pT = traj.positions[-1]
        grid = world_to_grid((float(pT[0]), float(pT[1]), float(pT[2])))
        status = traj.final_status.value
        lines.append(
            f"- {b.name}: final_grid={list(grid)}, status={status!r}"
        )
    return "\n".join(lines) if lines else "- (no dynamic bodies)"


def _fmt_vec(v: Iterable[float]) -> str:
    """Short, locale-independent vector format for prompt text."""

    return "(" + ", ".join(f"{float(x):.3f}" for x in v) + ")"


_SOLVER_GLOBAL_RULES: str = """\
Rules:
- You are the Solver. Reason from the video and text only.
- Do not call tools, write code, mention MuJoCo, or ask for more data.
- Use only body names that appear in the prompt. Do not invent body names.
- Output exactly one <answer> block. You MAY write a short plain-text
  rationale (one or two sentences, max ~80 words) BEFORE the <answer> block.
  Do not brainstorm alternatives. Do not repeat yourself. Do not include JSON
  outside the <answer> block.
- The <answer> block must contain valid JSON only: no markdown, no comments,
  no trailing explanation, no Python code.
- Include every dynamic body exactly once in the JSON answer.
- If uncertain, make the physically most plausible estimate and still output
  valid JSON.
"""


FORWARD_TEMPLATE: str = """\
Task: Predict the final state of the scene after the simulation ends.

You are shown the first half of the simulation video. You are also given all
initial conditions and all physical parameters. Use early motion, gravity,
friction, restitution, collisions, and momentum transfer to predict where each
dynamic body ends.

Scene parameters:
{params_block}

Initial conditions:
{initial_block}

Output schema for each dynamic body:
  "grid": [gx, gy, gz]
      - each value is an integer in [0, 15]
      - grid mapping:
        gx = round((x + 1.5) / 0.1875)
        gy = round((y + 1.5) / 0.1875)
        gz = round(z / 0.1875)
      - outputs are clipped to [0, 15]
  "status": one of "stable", "moving", "fallen", "ejected"

{global_rules}

Reply with an optional one-sentence rationale, then exactly one <answer>
block in this structure:

<answer>
{{"actual_body_name": {{"grid": [8, 8, 2], "status": "stable"}}}}
</answer>

Example reply style only:
The moving ball should hit the heavier block, lose speed, and remain near the
center. The block moves slightly and both bodies settle on the ground.
<answer>
{{"ball_a": {{"grid": [7, 8, 1], "status": "stable"}}, "block_b": {{"grid": [9, 8, 1], "status": "stable"}}}}
</answer>
""".replace("{global_rules}", _SOLVER_GLOBAL_RULES)


INVERSE_TEMPLATE: str = """\
Task: Infer the initial linear velocity of each dynamic body.

You are shown the second half of the simulation video. You are also given all
physical parameters and the final state. Work backward from the final motion,
collisions, resting positions, and visible late trajectory to estimate the
initial velocity bin for each body.

Scene parameters:
{params_block}

Final state:
{final_block}

Velocity bins:
  index 0 -> -6 m/s
  index 1 -> -3 m/s
  index 2 -> -1 m/s
  index 3 ->  0 m/s
  index 4 ->  1 m/s
  index 5 ->  3 m/s
  index 6 ->  6 m/s

Output schema for each dynamic body:
  "initial_vel_grid": [vx_bin, vy_bin, vz_bin]
      - each bin is an integer in [0, 6]

{global_rules}

Reply with an optional one-sentence rationale, then exactly one <answer>
block in this structure:

<answer>
{{"actual_body_name": {{"initial_vel_grid": [3, 3, 3]}}}}
</answer>

Example reply style only:
The ball approaches from the left before impact, so its initial x velocity
was positive. The block was likely at rest, with no vertical launch.
<answer>
{{"ball_a": {{"initial_vel_grid": [5, 3, 3]}}, "block_b": {{"initial_vel_grid": [3, 3, 3]}}}}
</answer>
""".replace("{global_rules}", _SOLVER_GLOBAL_RULES)


_PARAMETER_TEMPLATE_RAW: str = """\
Task: Infer the hidden physical parameters of each dynamic body.

You are shown the full simulation video. You are given the initial and final
states, but the masses and frictions are hidden. Use acceleration, deceleration,
collision response, sliding distance, bouncing, and how much each body resists
motion to infer the parameter levels.

Initial conditions:
{initial_block}

Final state:
{final_block}

Mass levels:
  1 -> {mass_1} kg
  2 -> {mass_2} kg
  3 -> {mass_3} kg
  4 -> {mass_4} kg
  5 -> {mass_5} kg

Friction levels:
  1 -> {friction_1}
  2 -> {friction_2}
  3 -> {friction_3}
  4 -> {friction_4}
  5 -> {friction_5}

Output schema for each dynamic body:
  "mass_level": integer in [1, 5]
  "friction_level": integer in [1, 5]

{global_rules}

Reply with an optional one-sentence rationale, then exactly one <answer>
block in this structure:

<answer>
{{{{"actual_body_name": {{{{"mass_level": 3, "friction_level": 2}}}}}}}}
</answer>

Example reply style only:
The red ball changes speed more after contact, so it is lighter. The blue
block slides only a short distance, so its friction is higher.
<answer>
{{{{"ball_a": {{{{"mass_level": 2, "friction_level": 2}}}}, "block_b": {{{{"mass_level": 4, "friction_level": 4}}}}}}}}
</answer>
"""

PARAMETER_TEMPLATE = _PARAMETER_TEMPLATE_RAW.format(
    initial_block="{initial_block}",
    final_block="{final_block}",
    mass_1=MASS_LEVELS[0],
    mass_2=MASS_LEVELS[1],
    mass_3=MASS_LEVELS[2],
    mass_4=MASS_LEVELS[3],
    mass_5=MASS_LEVELS[4],
    friction_1=FRICTION_LEVELS[0],
    friction_2=FRICTION_LEVELS[1],
    friction_3=FRICTION_LEVELS[2],
    friction_4=FRICTION_LEVELS[3],
    friction_5=FRICTION_LEVELS[4],
    global_rules=_SOLVER_GLOBAL_RULES,
)


# Multiple-choice short-answer instruction used by the training Solver.
MC_SHORT_ANSWER_INSTRUCTION: str = (
    "Answer with the option's letter from the given choices directly. "
    "You can only answer one letter from A, B, C, or D."
)


_VQA_TRAINING_RULES: str = (
    """\
Rules:
- Watch the full video and answer from the video and question only.
- """
    + MC_SHORT_ANSWER_INSTRUCTION
    + """ Put that single letter inside <answer>...</answer>.
- Answers without the <answer> block are invalid. Do not write JSON, markdown
  fences, code, or extra explanation after </answer>.
"""
)


VQA_TRAINING_TEMPLATE: str = """\
{question}

A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

%(rules)s

Example reply style only:
<answer>
A
</answer>
""" % {"rules": _VQA_TRAINING_RULES}


# Backward-compatible name used by the Stage-2 training observation builder.
VQA_TEMPLATE: str = VQA_TRAINING_TEMPLATE


__all__ = [
    "FORWARD_TEMPLATE",
    "INVERSE_TEMPLATE",
    "PARAMETER_TEMPLATE",
    "MC_SHORT_ANSWER_INSTRUCTION",
    "VQA_TEMPLATE",
    "VQA_TRAINING_TEMPLATE",
    "_format_params_block",
    "_format_initial_block",
    "_format_final_block",
    "_mass_level",
    "_friction_level",
    "_dyn_bodies",
]
