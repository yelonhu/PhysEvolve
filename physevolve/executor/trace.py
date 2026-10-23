"""Diagnostic ground-truth summaries for the Stage-2 rollout probe.

This module exists purely for *inspection* (the rollout probe + offline
debugging). It is NOT on the training hot path: nothing here influences
rewards, option ordering, or feasibility. It turns a :class:`SimState`
(and the executor's scene-template trace) into a compact, JSON-safe dict
so a human can read exactly what the simulator produced and how each VQA
option was derived from it.

Keep this isolated: if you need a new ground-truth field for debugging,
add it here rather than threading it through the executor's hot path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from physevolve.executor.state_view import sim_state_public_view

if TYPE_CHECKING:
    from physevolve.engine.schema import SimState


def _round_vec(vec: Any, ndigits: int = 4) -> list[float]:
    """Round an iterable of floats into a JSON-safe list."""

    return [round(float(x), ndigits) for x in vec]


def summarize_sim_state(sim_state: "SimState") -> dict:
    """Summarize the raw simulator ground-truth a verifier sees.

    Returns a JSON-safe dict with one entry per *dynamic* body (the only
    bodies a Stage-2 skill card / scene template can read via
    :class:`SimStateView`) plus the aggregated collision events. The
    per-body fields mirror :class:`SimStateView` exactly, so the probe
    report shows the same numbers the verifier reasoned over.

    Args:
        sim_state: The baseline simulation output.

    Returns:
        ``{"scene_id", "bodies": {name: {...}}, "collisions": [...]}``.
    """

    view = sim_state_public_view(sim_state)
    bodies: dict[str, dict] = {}
    for name in view.body_names:
        body = view[name]
        bodies[name] = {
            "initial_pos": _round_vec(body.initial_pos),
            "final_pos": _round_vec(body.final_pos),
            "initial_vel": _round_vec(body.initial_vel),
            "final_vel": _round_vec(body.final_vel),
            "final_status": str(body.final_status),
            "max_speed": round(float(body.max_speed), 4),
            "max_height": round(float(body.max_height), 4),
            "rebound_height": round(float(body.rebound_height), 4),
        }
    collisions = [
        {
            "t_step": int(ev.t_step),
            "body_a": ev.body_a,
            "body_b": ev.body_b,
            "contact_force_norm": round(float(ev.contact_force_norm), 3),
        }
        for ev in sim_state.collisions
    ]
    return {
        "scene_id": sim_state.scene_id,
        "bodies": bodies,
        "collisions": collisions,
    }


__all__ = ["summarize_sim_state"]
