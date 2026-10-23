"""Branch simulation: remove bodies, then re-simulate the scene.

Counterfactual ("if we removed the blue wall, which pit would the green ball
drop into?") and inverse-intervention ("what can we do to make the ball reach
the right pit?") questions are about a *modified* scene. They can only
be verifier-grounded by actually re-simulating the scene with the intervention
applied and reading the outcome from the real branch run -- never by asking the
Proposer to self-declare outcomes.

This module clones a :class:`~physevolve.engine.schema.SceneDef` with a set of
bodies removed (mirroring the clone pattern in
:mod:`physevolve.verification.dense_scorer`) and re-runs the physics via the
executor's guarded simulate. Branches are capped and deduplicated to bound the
extra simulation cost; a branch that fails to simulate is skipped (the question
that depended on it is simply not generated).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Sequence

from physevolve.engine.schema import SceneDef
from physevolve.executor.state_view import SimStateView, sim_state_public_view

if TYPE_CHECKING:
    from physevolve.skills.scene_templates import BranchSpec


def clone_scene_without(scene_def: SceneDef, remove_names: set[str]) -> SceneDef:
    """Return a copy of ``scene_def`` with ``remove_names`` bodies deleted.

    Joints / springs that reference a removed body are also dropped so the
    MJCF compiler does not choke on a dangling reference.
    """

    kept_bodies = tuple(b for b in scene_def.bodies if b.name not in remove_names)
    kept_joints = tuple(
        j
        for j in scene_def.joints
        if j.parent not in remove_names and j.child not in remove_names
    )
    kept_springs = tuple(
        s
        for s in scene_def.springs
        if s.body_a not in remove_names and s.body_b not in remove_names
    )
    return dataclasses.replace(
        scene_def,
        bodies=kept_bodies,
        joints=kept_joints,
        springs=kept_springs,
    )


def simulate_with_removed(
    scene_def: SceneDef,
    remove_names: Sequence[str],
    *,
    seed: int,
    sim_timeout_s: float,
) -> SimStateView:
    """Clone the scene without ``remove_names`` bodies and re-simulate.

    Returns a read-only :class:`SimStateView` of the branch run. Raises on any
    simulation failure (the caller decides whether to skip the branch).
    """

    branch_def = clone_scene_without(scene_def, set(remove_names))
    # Lazy import to avoid a module-load cycle (pipeline imports this module
    # lazily too).
    from physevolve.executor.pipeline import _simulate_guarded

    sim_state = _simulate_guarded(branch_def, seed, sim_timeout_s)
    return sim_state_public_view(sim_state)


def run_branch_sims(
    scene_def: SceneDef,
    branches: Sequence["BranchSpec"],
    *,
    seed: int,
    sim_timeout_s: float,
    max_branches: int = 6,
) -> dict[str, SimStateView]:
    """Run up to ``max_branches`` branch sims; return ``{branch_key: view}``.

    Identical removal sets are simulated once and shared. Branches that fail
    to simulate are omitted from the result (the dependent question is then
    not generated).
    """

    out: dict[str, SimStateView] = {}
    by_remove: dict[frozenset, SimStateView] = {}
    distinct = 0
    for spec in branches:
        remove_set = frozenset(spec.remove)
        cached = by_remove.get(remove_set)
        if cached is not None:
            out[spec.key] = cached
            continue
        if distinct >= max_branches:
            continue
        try:
            view = simulate_with_removed(
                scene_def, spec.remove, seed=seed, sim_timeout_s=sim_timeout_s
            )
        except Exception:  # noqa: BLE001 — a failed branch just drops its question.
            continue
        by_remove[remove_set] = view
        out[spec.key] = view
        distinct += 1
    return out


__all__ = [
    "clone_scene_without",
    "run_branch_sims",
    "simulate_with_removed",
]
