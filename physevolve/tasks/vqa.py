"""Stage-2 VQA observation builder (docs/architecture.md).

The Stage-2 Proposer emits both ``build_scene(api)`` and ``get_vqa(sim_state)``.
The executor calls ``get_vqa(SimStateView)`` and stores the result as
``ExecutorResult.vqa_payload = (question, options, correct_idx)`` (see
[`physevolve/executor/pipeline.py`](../executor/pipeline.py)).

This module takes that payload plus the canonical :class:`SimState`
(for the video frames) and assembles a :class:`SolverObservation`-shape
triple ``(video, text_prompt, ground_truth)`` that mirrors the Stage-1
modes. Training renders the text prompt from
:data:`physevolve.prompts.solver.VQA_TRAINING_TEMPLATE`.

The Solver picks one of ``A``/``B``/``C``/``D`` inside ``<answer>...</answer>``.
Reward is exact match on ``correct_idx`` (see
:func:`physevolve.verification.vqa_scorer.score_vqa`).
"""

from __future__ import annotations

import numpy as np

from physevolve.engine.schema import SimState
from physevolve.prompts.solver import VQA_TRAINING_TEMPLATE

_VQA_PAYLOAD_OPTIONS_LEN = 4
_VQA_PAYLOAD_LETTERS: tuple[str, ...] = ("A", "B", "C", "D")


def build_observation(
    sim_state: SimState,
    vqa_payload: tuple[str, list[str], int],
) -> tuple[np.ndarray, str, dict]:
    """Assemble the Stage-2 VQA observation triple.

    Args:
        sim_state: Full engine output. ``video_frames`` is forwarded
            in its entirety so the Solver sees the whole scene — VQA
            questions can ask about any frame.
        vqa_payload: ``(question, options, correct_idx)`` as returned
            by the Stage-2 Proposer's ``get_vqa(sim_state_view)``,
            already schema-validated by the executor pipeline.

    Returns:
        ``(video_slice, text_prompt, ground_truth)`` where
        ``video_slice`` is ``(n_cam, n_frames, H, W, 3)`` uint8,
        ``text_prompt`` is the formatted Solver prompt, and
        ``ground_truth`` is::

            {
                "correct_idx": int,        # 0..3
                "correct_letter": "A"|"B"|"C"|"D",
                "options": list[str],      # original 4 options
                "question": str,
            }

        The scorer only consumes ``correct_idx``; the surrounding fields
        are kept so audit logs can reconstruct the exact question the
        Solver saw.

    Raises:
        ValueError: If ``vqa_payload`` does not have the required shape
            ``(str, list[str] of length 4, int in [0, 3])``. The pipeline
            already validates this in
            ``physevolve/executor/pipeline.py::_validate_vqa_payload``,
            so this is a defence-in-depth check.
    """

    question, options, correct_idx = vqa_payload
    if not isinstance(question, str) or not question.strip():
        raise ValueError("vqa_payload question must be a non-empty string")
    if not isinstance(options, (list, tuple)) or len(options) != _VQA_PAYLOAD_OPTIONS_LEN:
        raise ValueError(
            "vqa_payload options must be a length-4 sequence; "
            f"got {type(options).__name__} of length "
            f"{len(options) if hasattr(options, '__len__') else 'unknown'}"
        )
    if not all(isinstance(o, str) and o.strip() for o in options):
        raise ValueError("vqa_payload options must all be non-empty strings")
    if not isinstance(correct_idx, int) or isinstance(correct_idx, bool):
        raise ValueError("vqa_payload correct_idx must be a plain int")
    if not (0 <= correct_idx < _VQA_PAYLOAD_OPTIONS_LEN):
        raise ValueError(
            f"vqa_payload correct_idx must be in [0, {_VQA_PAYLOAD_OPTIONS_LEN - 1}]; "
            f"got {correct_idx}"
        )

    # Stage-2 Solver sees the whole video so it can reason about both
    # the setup and the outcome before answering an MC question.
    video_slice = sim_state.video_frames

    text_prompt = VQA_TRAINING_TEMPLATE.format(
        question=question,
        opt_a=options[0],
        opt_b=options[1],
        opt_c=options[2],
        opt_d=options[3],
    )

    ground_truth: dict = {
        "correct_idx": int(correct_idx),
        "correct_letter": _VQA_PAYLOAD_LETTERS[correct_idx],
        "options": list(options),
        "question": question,
    }
    return video_slice, text_prompt, ground_truth


__all__ = ["build_observation"]
