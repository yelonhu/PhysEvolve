"""Task router — dispatches a :class:`SimState` to one of 3 task modes.

Provides:

* :class:`TaskMode` and :class:`SolverObservation`.
* :func:`select_task_mode` — weighted random draw over ``TaskMode``.
* :func:`assemble_observation` — dispatches to
  ``tasks.{forward,inverse,parameter}.build_observation`` and wraps
  the return into a :class:`SolverObservation`.
* :func:`parse_solver_output` — extracts ``<answer>...</answer>``,
  JSON-decodes, schema-validates per mode. Returns ``None`` on any
  failure so the scorer is never invoked on malformed data.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

from physevolve.engine.schema import SimState
from physevolve.inference.parsing import extract_answer_block
from physevolve.tasks.config import load_task_weights

# ─── Enums and dataclasses ────────────────────────────────────────────


class TaskMode(str, Enum):
    """Stage-1 dense modes plus the Stage-2 VQA mode.

    ``FORWARD`` / ``INVERSE`` / ``PARAMETER`` are dense Stage-1 self-play
    tasks. ``VQA`` is the Stage-2 multiple-choice task built from a
    Proposer-supplied ``get_vqa(sim_state)`` payload; the Solver picks one
    of A/B/C/D rather than emitting a JSON object.
    """

    FORWARD = "forward"
    INVERSE = "inverse"
    PARAMETER = "parameter"
    VQA = "vqa"


@dataclass(frozen=True, slots=True)
class SolverObservation:
    """What the Solver sees for a single scene in a single task mode.

    ``ground_truth`` is kept alongside the observation for scorer use on
    the server side. It MUST NOT be forwarded to the Solver.
    """

    scene_id: str
    mode: TaskMode
    video_frames: np.ndarray
    text_prompt: str
    ground_truth: dict


# ─── Mode selection ───────────────────────────────────────────────────


def _load_task_weights() -> dict[TaskMode, float]:
    """Read task weights via :func:`tasks.config.load_task_weights`.

    Normalises to sum 1.0 so ``random.choices`` is robust to minor
    rounding in hand-edited YAML.
    """

    raw = load_task_weights()
    weights = {
        TaskMode.FORWARD: float(raw.get("forward", 0.40)),
        TaskMode.INVERSE: float(raw.get("inverse", 0.30)),
        TaskMode.PARAMETER: float(raw.get("parameter", 0.30)),
    }
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("task_weights must sum to a positive number")
    return {mode: w / total for mode, w in weights.items()}


def select_task_mode(
    sim_state: SimState, rng: random.Random
) -> TaskMode:
    """Sample a :class:`TaskMode` from the configured weights.

    ``sim_state`` is accepted for API compatibility —
    future curricula may route based on scene properties (e.g. force
    ``PARAMETER`` on scenes with hidden mass ambiguity).
    """

    _ = sim_state  # intentionally unused in v0.1
    weights = _load_task_weights()
    modes = list(weights.keys())
    probs = [weights[m] for m in modes]
    return rng.choices(modes, weights=probs, k=1)[0]


# ─── Assembly ─────────────────────────────────────────────────────────


def assemble_observation(
    sim_state: SimState,
    mode: TaskMode,
    *,
    vqa_payload: Optional[tuple[str, list[str], int]] = None,
) -> SolverObservation:
    """Build the complete :class:`SolverObservation` for ``mode``.

    Args:
        sim_state: Engine output that supplies video frames, scene def,
            and trajectories.
        mode: Task mode. ``VQA`` requires ``vqa_payload``.
        vqa_payload: ``(question, options, correct_idx)`` triple from the
            Stage-2 Proposer's ``get_vqa(sim_state)``. Ignored for the
            dense modes; required for :attr:`TaskMode.VQA`.

    Raises:
        ValueError: If ``mode == VQA`` and ``vqa_payload`` is ``None``.
    """

    # Local imports to keep the router free of the per-mode cyclic
    # dependency that would exist if the per-mode modules imported
    # :class:`TaskMode` from here.
    from physevolve.tasks import forward as _forward
    from physevolve.tasks import inverse as _inverse
    from physevolve.tasks import parameter as _parameter
    from physevolve.tasks import vqa as _vqa

    if mode is TaskMode.FORWARD:
        video, text, gt = _forward.build_observation(sim_state)
    elif mode is TaskMode.INVERSE:
        video, text, gt = _inverse.build_observation(sim_state)
    elif mode is TaskMode.PARAMETER:
        video, text, gt = _parameter.build_observation(sim_state)
    elif mode is TaskMode.VQA:
        if vqa_payload is None:
            raise ValueError(
                "TaskMode.VQA requires a vqa_payload from the Stage-2 "
                "Proposer's get_vqa(sim_state)"
            )
        video, text, gt = _vqa.build_observation(sim_state, vqa_payload)
    else:
        raise ValueError(f"unknown task mode {mode!r}")

    return SolverObservation(
        scene_id=sim_state.scene_id,
        mode=mode,
        video_frames=video,
        text_prompt=text,
        ground_truth=gt,
    )


# ─── Parsing ──────────────────────────────────────────────────────────


_VALID_STATUSES: frozenset[str] = frozenset(
    {"stable", "moving", "fallen", "ejected"}
)


def _validate_forward(payload: Any) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    out: dict = {}
    for name, entry in payload.items():
        if not isinstance(name, str):
            return None
        if not isinstance(entry, dict):
            return None
        grid = entry.get("grid")
        status = entry.get("status")
        if not isinstance(grid, list) or len(grid) != 3:
            return None
        if not all(isinstance(g, int) and not isinstance(g, bool) for g in grid):
            return None
        if not all(0 <= g <= 15 for g in grid):
            return None
        if not isinstance(status, str) or status.lower() not in _VALID_STATUSES:
            return None
        out[name] = {"grid": list(grid), "status": status.lower()}
    return out


def _validate_inverse(payload: Any) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    out: dict = {}
    for name, entry in payload.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            return None
        vg = entry.get("initial_vel_grid")
        if not isinstance(vg, list) or len(vg) != 3:
            return None
        if not all(
            isinstance(v, int) and not isinstance(v, bool) for v in vg
        ):
            return None
        if not all(0 <= v <= 6 for v in vg):
            return None
        out[name] = {"initial_vel_grid": list(vg)}
    return out


def _validate_parameter(payload: Any) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    out: dict = {}
    for name, entry in payload.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            return None
        ml = entry.get("mass_level")
        fl = entry.get("friction_level")
        if not (isinstance(ml, int) and not isinstance(ml, bool)):
            return None
        if not (isinstance(fl, int) and not isinstance(fl, bool)):
            return None
        if not (1 <= ml <= 5 and 1 <= fl <= 5):
            return None
        out[name] = {"mass_level": ml, "friction_level": fl}
    return out


_VALIDATORS = {
    TaskMode.FORWARD: _validate_forward,
    TaskMode.INVERSE: _validate_inverse,
    TaskMode.PARAMETER: _validate_parameter,
}

_VQA_LETTER_TO_IDX: dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3}


def _parse_vqa_answer(raw_text: str) -> Optional[dict]:
    """Extract a single A/B/C/D letter from the Stage-2 VQA Solver answer.

    Accepts ``<answer>A</answer>`` and minor variants: lowercase, surrounding
    whitespace, or a single-letter line with a trailing period
    (``A.``). Anything else (multi-letter, JSON, full word, no answer
    block) returns ``None``.
    """

    block = extract_answer_block(raw_text)
    if block is None:
        return None
    stripped = block.strip()
    # Tolerate a trailing period (``A.``) or whitespace; nothing else.
    if stripped.endswith("."):
        stripped = stripped[:-1].strip()
    if len(stripped) != 1:
        return None
    letter = stripped.upper()
    if letter not in _VQA_LETTER_TO_IDX:
        return None
    return {"letter": letter, "predicted_idx": _VQA_LETTER_TO_IDX[letter]}


def parse_solver_output(
    raw_text: str, mode: TaskMode
) -> Optional[dict]:
    """Extract + JSON-decode + schema-validate the Solver's answer.

    Returns the validated dict on success, or ``None`` on any failure
    (missing ``<answer>`` tag, invalid JSON, schema violation). The
    scorer must treat a ``None`` return as "reward = 0".

    Schema per mode:

    * ``FORWARD``: ``{name: {"grid": [int*3 in 0..15],
      "status": str in {"stable","moving","fallen","ejected"}}}``
    * ``INVERSE``: ``{name: {"initial_vel_grid": [int*3 in 0..6]}}``
    * ``PARAMETER``: ``{name: {"mass_level": int in 1..5,
      "friction_level": int in 1..5}}``
    * ``VQA`` (docs/architecture.md): returns
      ``{"letter": "A"|"B"|"C"|"D", "predicted_idx": int in 0..3}``.
      The Solver picks one of four options; no JSON inside ``<answer>``.
    """

    if mode is TaskMode.VQA:
        return _parse_vqa_answer(raw_text)

    block = extract_answer_block(raw_text)
    if block is None:
        return None

    try:
        payload = json.loads(block)
    except (ValueError, TypeError):
        return None

    validator = _VALIDATORS[mode]
    return validator(payload)


__all__ = [
    "TaskMode",
    "SolverObservation",
    "select_task_mode",
    "assemble_observation",
    "parse_solver_output",
]
