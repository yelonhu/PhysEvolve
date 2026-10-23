"""Tests for :mod:`physevolve.tests.helpers.structural_checks`.

At least one positive and one negative case per entry of the
:data:`CONSTRAINTS` grid, plus dedicated coverage for
:func:`count_api_calls` and :func:`classify_hinge_parent`.
"""

from __future__ import annotations

import pytest

from physevolve.tests.helpers.structural_checks import (
    CONSTRAINTS,
    check_constraint,
    classify_hinge_parent,
    count_api_calls,
    extract_scene_stats,
)


# ─── count_api_calls ─────────────────────────────────────────────────


def test_count_api_calls_basic() -> None:
    code = """
def build_scene(api):
    api.add_plane('ground', pos=(0, 0, 0), size=(5, 5))
    api.add_sphere('a', pos=(0, 0, 1), radius=0.1, mass=1.0)
    api.add_sphere('b', pos=(0, 0, 2), radius=0.1, mass=1.0)
    api.set_velocity('a', linear=(0.5, 0, 0))
"""
    counts = count_api_calls(code)
    assert counts == {
        "add_plane": 1,
        "add_sphere": 2,
        "set_velocity": 1,
    }


def test_count_api_calls_accepts_indented_body() -> None:
    # Body without the ``def build_scene(api):`` header — the wrap
    # fallback in parse_build_scene should recover it.
    code = "    api.add_plane('g', pos=(0, 0, 0))\n    api.add_sphere('a', pos=(0, 0, 1), radius=0.1, mass=1.0)\n"
    counts = count_api_calls(code)
    assert counts == {"add_plane": 1, "add_sphere": 1}


def test_count_api_calls_unparsable_returns_none() -> None:
    assert count_api_calls("def build_scene(api) [!!! syntax error") is None


def test_count_api_calls_ignores_non_api_calls() -> None:
    code = """
def build_scene(api):
    other.add_box('x')
    api.add_box('a', pos=(0, 0, 1), size=(0.1, 0.1, 0.1), mass=1.0)
"""
    counts = count_api_calls(code)
    assert counts == {"add_box": 1}


# ─── classify_hinge_parent ───────────────────────────────────────────


def test_classify_hinge_parent_world() -> None:
    code = """
def build_scene(api):
    api.add_sphere('bob', pos=(0, 0, 1), radius=0.1, mass=1.0)
    api.add_hinge('h', parent='world', child='bob', pos=(0,0,1.5), axis=(0,1,0))
"""
    assert classify_hinge_parent(code, "h") == "world"


def test_classify_hinge_parent_body() -> None:
    code = """
def build_scene(api):
    api.add_box('anchor', pos=(0,0,1.5), size=(0.05,0.05,0.05), mass=1.0, fixed=True)
    api.add_sphere('bob', pos=(0.1,0,1.2), radius=0.1, mass=1.0)
    api.add_hinge('pivot', parent='anchor', child='bob', pos=(0,0,0), axis=(0,1,0))
"""
    assert classify_hinge_parent(code, "pivot") == "body"


def test_classify_hinge_parent_missing() -> None:
    code = """
def build_scene(api):
    api.add_sphere('bob', pos=(0, 0, 1), radius=0.1, mass=1.0)
"""
    assert classify_hinge_parent(code, "missing") is None


# ─── Sample scene helpers ────────────────────────────────────────────


def _wrap(body: str) -> str:
    """Prepend ``def build_scene(api):`` and a 4-space indent."""

    indented = "\n".join("    " + line if line.strip() else line
                          for line in body.strip("\n").splitlines())
    return f"def build_scene(api):\n{indented}\n"


# Good samples per category (mirror the hand-written few-shots, but
# simplified to the minimum needed to satisfy structural_checks).

_GOOD: dict[str, str] = {
    "single_body_freefall": _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_sphere('a', pos=(0,0,1.5), radius=0.1, mass=1.0)
"""),
    "single_body_projectile": _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_sphere('a', pos=(-0.5,0,0.5), radius=0.1, mass=1.0)
api.set_velocity('a', linear=(2.0, 0.0, 1.0))
"""),
    "two_body_collision": _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_sphere('a', pos=(-0.6,0,0.12), radius=0.1, mass=1.0)
api.add_sphere('b', pos=(0.6,0,0.12), radius=0.1, mass=1.0)
api.set_velocity('a', linear=(3.0, 0.0, 0.0))
"""),
    "stacking_free": _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_box('base', pos=(0,0,0.1), size=(0.15,0.15,0.1), mass=5.0)
api.add_box('top',  pos=(0,0,0.3), size=(0.1,0.1,0.08), mass=0.5)
"""),
    "simple_pendulum_explicit": _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_box('anchor', pos=(0,0,1.8), size=(0.05,0.05,0.05), mass=1.0, fixed=True)
api.add_sphere('bob', pos=(0.3,0,1.5), radius=0.08, mass=1.0)
api.add_hinge('pivot', parent='anchor', child='bob', pos=(0,0,0), axis=(0,1,0))
"""),
    "simple_pendulum_worldfixed": _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_sphere('bob', pos=(0.4,0,1.5), radius=0.08, mass=1.0)
api.add_hinge('pivot', parent='world', child='bob', pos=(0,0,1.8), axis=(0,1,0))
"""),
    "double_pendulum": _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_box('anchor', pos=(0,0,2.0), size=(0.05,0.05,0.05), mass=1.0, fixed=True)
api.add_sphere('bob1', pos=(0.3,0,1.7), radius=0.07, mass=1.0)
api.add_sphere('bob2', pos=(0.55,0,1.45), radius=0.06, mass=0.5)
api.add_hinge('h1', parent='anchor', child='bob1', pos=(0,0,0), axis=(0,1,0))
api.add_hinge('h2', parent='bob1', child='bob2', pos=(0,0,0), axis=(0,1,0))
"""),
    "slider_mechanism": _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_box('cart', pos=(0,0,0.3), size=(0.12,0.08,0.06), mass=1.0)
api.add_sphere('obst', pos=(0.6,0,0.08), radius=0.08, mass=5.0)
api.add_slide('rail', parent='world', child='cart', pos=(0,0,0.3), axis=(1,0,0))
"""),
    "pendulum_strikes_free": _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_box('anchor', pos=(0,0,1.8), size=(0.05,0.05,0.05), mass=1.0, fixed=True)
api.add_sphere('bob', pos=(0.4,0,1.5), radius=0.08, mass=1.0)
api.add_sphere('free', pos=(0.7,0,0.1), radius=0.08, mass=1.0)
api.add_hinge('pivot', parent='anchor', child='bob', pos=(0,0,0), axis=(0,1,0))
"""),
    "spring_two_body": _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_sphere('a', pos=(-0.3,0,0.12), radius=0.1, mass=1.0)
api.add_sphere('b', pos=(0.3,0,0.12), radius=0.1, mass=1.0)
api.add_spring('s', body_a='a', body_b='b', stiffness=80.0, damping=0.5)
"""),
    "spring_chain": _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_sphere('a', pos=(-0.5,0,0.12), radius=0.08, mass=1.0)
api.add_sphere('b', pos=( 0.0,0,0.12), radius=0.08, mass=1.0)
api.add_sphere('c', pos=( 0.5,0,0.12), radius=0.08, mass=1.0)
api.add_spring('s1', body_a='a', body_b='b', stiffness=80.0, damping=0.5)
api.add_spring('s2', body_a='b', body_b='c', stiffness=80.0, damping=0.5)
"""),
    "mixed_constraint": _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_box('anchor', pos=(0,0,1.8), size=(0.05,0.05,0.05), mass=1.0, fixed=True)
api.add_sphere('bob', pos=(0.3,0,1.5), radius=0.08, mass=1.0)
api.add_sphere('free', pos=(0.6,0,0.1), radius=0.08, mass=1.0)
api.add_hinge('pivot', parent='anchor', child='bob', pos=(0,0,0), axis=(0,1,0))
api.add_spring('s', body_a='bob', body_b='free', stiffness=40.0, damping=0.3)
"""),
}


# Per-category negative samples — each one violates exactly one rule.

_BAD: dict[str, tuple[str, str]] = {
    # (code, expected substring in error message)
    "single_body_freefall": (
        _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_sphere('a', pos=(0,0,1.5), radius=0.1, mass=1.0)
api.add_sphere('b', pos=(0,0,0.5), radius=0.1, mass=1.0)
"""),
        "expected ≤1 dynamic bodies",
    ),
    "single_body_projectile": (
        _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_sphere('a', pos=(0,0,1.5), radius=0.1, mass=1.0)
api.set_velocity('a', linear=(0.0, 0.0, -2.0))
"""),
        "non-zero horizontal",
    ),
    "two_body_collision": (
        _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_sphere('a', pos=(-0.6,0,0.12), radius=0.1, mass=1.0)
api.add_sphere('b', pos=(0.6,0,0.12), radius=0.1, mass=1.0)
"""),
        "aimed at the other body",
    ),
    "stacking_free": (
        _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_sphere('a', pos=(-0.5,0,0.1), radius=0.1, mass=1.0)
api.add_sphere('b', pos=(0.5,0,0.1), radius=0.1, mass=1.0)
"""),
        "no pair of dynamic bodies",
    ),
    "simple_pendulum_explicit": (
        _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_sphere('bob', pos=(0.4,0,1.5), radius=0.08, mass=1.0)
api.add_hinge('pivot', parent='world', child='bob', pos=(0,0,1.8), axis=(0,1,0))
"""),
        "hinge parent must be 'body'",
    ),
    "simple_pendulum_worldfixed": (
        _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_box('anchor', pos=(0,0,1.8), size=(0.05,0.05,0.05), mass=1.0, fixed=True)
api.add_sphere('bob', pos=(0.3,0,1.5), radius=0.08, mass=1.0)
api.add_hinge('pivot', parent='anchor', child='bob', pos=(0,0,0), axis=(0,1,0))
"""),
        "hinge parent must be 'world'",
    ),
    "double_pendulum": (
        # Second hinge's parent does not equal first hinge's child.
        _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_box('anchor', pos=(0,0,2.0), size=(0.05,0.05,0.05), mass=1.0, fixed=True)
api.add_sphere('bob1', pos=(0.3,0,1.7), radius=0.07, mass=1.0)
api.add_sphere('bob2', pos=(0.55,0,1.45), radius=0.06, mass=0.5)
api.add_hinge('h1', parent='anchor', child='bob1', pos=(0,0,0), axis=(0,1,0))
api.add_hinge('h2', parent='anchor', child='bob2', pos=(0,0,0), axis=(0,1,0))
"""),
        "second hinge parent must equal first hinge's child",
    ),
    "slider_mechanism": (
        _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_box('cart', pos=(0,0,0.3), size=(0.12,0.08,0.06), mass=1.0)
"""),
        "expected ≥1 slides",
    ),
    "pendulum_strikes_free": (
        # Two dynamic bodies, but BOTH are on the hinge graph — no
        # free third body for the pendulum to strike.
        _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_box('anchor', pos=(0,0,2.0), size=(0.05,0.05,0.05), mass=1.0, fixed=True)
api.add_sphere('bob1', pos=(0.3,0,1.7), radius=0.07, mass=1.0)
api.add_sphere('bob2', pos=(0.55,0,1.45), radius=0.06, mass=0.5)
api.add_hinge('h1', parent='anchor', child='bob1', pos=(0,0,0), axis=(0,1,0))
api.add_hinge('h2', parent='bob1', child='bob2', pos=(0,0,0), axis=(0,1,0))
"""),
        "no dynamic body is free",
    ),
    "spring_two_body": (
        _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_sphere('a', pos=(-0.3,0,0.12), radius=0.1, mass=1.0)
api.add_sphere('b', pos=(0.3,0,0.12), radius=0.1, mass=1.0)
"""),
        "expected ≥1 springs",
    ),
    "spring_chain": (
        # 3 dynamic bodies but only 1 spring — hits the "≥2 springs"
        # gate specific to spring_chain_connected.
        _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_sphere('a', pos=(-0.3,0,0.12), radius=0.08, mass=1.0)
api.add_sphere('b', pos=( 0.0,0,0.12), radius=0.08, mass=1.0)
api.add_sphere('c', pos=( 0.3,0,0.12), radius=0.08, mass=1.0)
api.add_spring('s', body_a='a', body_b='b', stiffness=80.0, damping=0.5)
"""),
        "≥2 springs",
    ),
    "mixed_constraint": (
        # Has a spring but no joint — caught by the numeric
        # min_joints=1 check (hinge+slide total).
        _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_sphere('a', pos=(-0.3,0,0.12), radius=0.1, mass=1.0)
api.add_sphere('b', pos=(0.3,0,0.12), radius=0.1, mass=1.0)
api.add_spring('s', body_a='a', body_b='b', stiffness=80.0, damping=0.5)
"""),
        "≥1 joints",
    ),
}


@pytest.mark.parametrize("category", sorted(CONSTRAINTS.keys()))
def test_constraint_grid_positive(category: str) -> None:
    """Every category has a canonical good sample that passes."""

    assert category in _GOOD, f"missing positive sample for {category}"
    ok, msg = check_constraint(_GOOD[category], CONSTRAINTS[category])
    assert ok, f"[{category}] expected pass, got: {msg}"


@pytest.mark.parametrize("category", sorted(CONSTRAINTS.keys()))
def test_constraint_grid_negative(category: str) -> None:
    """Every category has a hand-crafted violation with a recognisable message."""

    assert category in _BAD, f"missing negative sample for {category}"
    code, marker = _BAD[category]
    ok, msg = check_constraint(code, CONSTRAINTS[category])
    assert not ok, f"[{category}] expected failure, but got pass"
    assert marker in msg, (
        f"[{category}] expected substring {marker!r} in error msg, got {msg!r}"
    )


# ─── extract_scene_stats shapes ──────────────────────────────────────


def test_extract_scene_stats_counts_match() -> None:
    code = _GOOD["double_pendulum"]
    stats = extract_scene_stats(code)
    assert stats is not None
    # 3 registered bodies: 1 fixed anchor + 2 dynamic bobs.
    assert len(stats.bodies) == 3
    assert len(stats.dynamic_bodies) == 2
    assert len(stats.fixed_bodies) == 1
    # 2 hinges, 0 slides, 0 springs.
    assert len(stats.hinges) == 2
    assert len(stats.slides) == 0
    assert len(stats.springs) == 0
    # The first hinge chains to the second.
    assert stats.hinges[1].parent == stats.hinges[0].child


def test_extract_scene_stats_handles_unary_minus() -> None:
    code = _wrap("""
api.add_plane('g', pos=(0,0,0), size=(5,5))
api.add_sphere('a', pos=(-0.5, -0.2, 0.3), radius=0.1, mass=1.0)
api.set_velocity('a', linear=(-1.0, 0.0, 0.0))
""")
    stats = extract_scene_stats(code)
    assert stats is not None
    a = stats.bodies[0]
    assert a.pos is not None and a.pos[0] == pytest.approx(-0.5)
    assert stats.velocities[0].linear[0] == pytest.approx(-1.0)


def test_extract_scene_stats_tolerates_garbage_body() -> None:
    # Non-literal body name — the call should be skipped quietly.
    code = """
def build_scene(api):
    api.add_plane('g', pos=(0,0,0), size=(5,5))
    name_expr = 'a'
    api.add_sphere(name_expr, pos=(0,0,1.5), radius=0.1, mass=1.0)
"""
    stats = extract_scene_stats(code)
    assert stats is not None
    assert len(stats.bodies) == 0  # Could not recover name → skipped.
    assert stats.api_call_counts.get("add_sphere") == 1
