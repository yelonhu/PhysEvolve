"""Colour naming + colour-permutation for scene templates.

Questions refer to bodies by their *visible colour* ("the red ball", "the blue
floating wall"), never by an internal name. Two utilities support that:

* :func:`rgba_to_color_name` maps a body's ``rgba`` to the nearest colour word,
  so questions/options refer to what the Solver actually sees - even if the
  Proposer named the body ``ball_1``. This is the colour-alignment guarantee:
  the GT is computed from the body's trajectory (by identity), and the colour
  word is derived from that same body's ``rgba``, so they can never disagree.

* :func:`color_permuted_scene` produces colour-swapped *variants* of one golden
  scene. GROUND-TRUTH GENERATION IDEA: colours do not affect physics, so a
  colour permutation leaves every body's trajectory (and thus every answer, by
  body identity) unchanged - only the rgba (hence the colour word in the
  question) changes. Re-simulating a colour-permuted scene yields a visually
  distinct video whose questions are auto-recomputed correctly. This is how one
  golden scene becomes many colour-variant questions.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from physevolve.engine.schema import SceneDef


# Named palette used both to name observed colours and to recolour variants.
# RGB in [0, 1]; alpha is ignored for naming.
_PALETTE: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("red", (0.9, 0.15, 0.15)),
    ("green", (0.2, 0.75, 0.25)),
    ("blue", (0.2, 0.3, 0.9)),
    ("yellow", (0.95, 0.9, 0.2)),
    ("orange", (0.95, 0.55, 0.15)),
    ("purple", (0.6, 0.2, 0.8)),
    ("cyan", (0.2, 0.8, 0.85)),
    ("white", (0.95, 0.95, 0.95)),
    ("gray", (0.5, 0.5, 0.5)),
    ("brown", (0.5, 0.32, 0.18)),
    ("black", (0.08, 0.08, 0.08)),
    ("pink", (0.95, 0.55, 0.75)),
)

_PALETTE_BY_NAME: Mapping[str, tuple[float, float, float]] = dict(_PALETTE)

# Public set of palette colour words. Used to FORBID colour-named bodies in
# scene templates (the colour shown to the Solver must be derived from rgba, not
# baked into the body name, so colour permutation actually changes the question).
PALETTE_COLOR_NAMES: frozenset[str] = frozenset(name for name, _ in _PALETTE)


def contains_color_word(name: str) -> bool:
    """Return True if any ``_``-delimited token of ``name`` is a palette colour.

    e.g. ``green_ball`` -> True, ``ball_a`` -> False. Used by scene-template
    manifest validation to reject colour-baked body names.
    """

    return any(tok in PALETTE_COLOR_NAMES for tok in str(name).lower().split("_"))


def rgba_to_color_name(rgba: Sequence[float]) -> str:
    """Return the nearest palette colour name for an ``rgba`` tuple."""

    r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
    best_name = "gray"
    best_dist = float("inf")
    for name, (pr, pg, pb) in _PALETTE:
        dist = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def body_color_map(scene_def: "SceneDef") -> dict[str, str]:
    """Return ``{body_name: nearest_colour_name}`` for every body."""

    return {b.name: rgba_to_color_name(b.rgba) for b in scene_def.bodies}


def color_permuted_scene(
    scene_def: "SceneDef",
    body_names: Sequence[str],
    color_names: Sequence[str],
) -> tuple["SceneDef", dict[str, str]]:
    """Recolour ``body_names`` to ``color_names`` (1:1) -> (new scene, color map).

    Physics is unaffected (rgba is purely visual), so re-simulating the
    returned scene gives identical trajectories with a different appearance.
    The returned color map is ``{body_name: new_colour_name}`` for the
    recoloured bodies; other bodies keep their original colours.

    Raises:
        ValueError: If lengths mismatch or a colour name is unknown.
    """

    if len(body_names) != len(color_names):
        raise ValueError("body_names and color_names must have equal length")
    assignment: dict[str, str] = {}
    for name, color in zip(body_names, color_names):
        if color not in _PALETTE_BY_NAME:
            raise ValueError(f"unknown colour {color!r}; palette={sorted(_PALETTE_BY_NAME)}")
        assignment[name] = color

    new_bodies = []
    for b in scene_def.bodies:
        if b.name in assignment:
            rgb = _PALETTE_BY_NAME[assignment[b.name]]
            alpha = float(b.rgba[3]) if len(b.rgba) > 3 else 1.0
            new_bodies.append(
                dataclasses.replace(b, rgba=(rgb[0], rgb[1], rgb[2], alpha))
            )
        else:
            new_bodies.append(b)
    new_scene = dataclasses.replace(scene_def, bodies=tuple(new_bodies))
    return new_scene, dict(assignment)


__all__ = [
    "PALETTE_COLOR_NAMES",
    "body_color_map",
    "color_permuted_scene",
    "contains_color_word",
    "rgba_to_color_name",
]
