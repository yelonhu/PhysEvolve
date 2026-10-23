"""Stage-2 VQA exact-match scorer (docs/architecture.md).

The Solver picks one of A/B/C/D inside ``<answer>...</answer>``. Reward is
1.0 if the parsed letter index equals ``correct_idx``, else 0.0. Malformed
output (no ``<answer>`` tag, multi-letter, non-letter) short-circuits to 0
in the caller (``tasks.router.parse_solver_output`` returns ``None``).

Aggregation convention: a VQA scene is a single (Q, options) pair, so this
scorer returns a single scalar — no per-body averaging.
"""

from __future__ import annotations

from typing import Mapping, Optional


def score_vqa(
    parsed: Optional[Mapping[str, object]],
    ground_truth: Mapping[str, object],
) -> float:
    """Return ``1.0`` for exact A/B/C/D match, else ``0.0``.

    Args:
        parsed: Output of
            :func:`physevolve.tasks.router.parse_solver_output` for
            :attr:`TaskMode.VQA`. ``None`` (parse failed) maps to 0.
            Otherwise expected to contain
            ``{"predicted_idx": int in 0..3, "letter": "A"|"B"|"C"|"D"}``.
        ground_truth: Dict from
            :func:`physevolve.tasks.vqa.build_observation`; must
            contain ``"correct_idx": int in 0..3``.

    Returns:
        Float reward in ``{0.0, 1.0}``.

    Raises:
        KeyError: If ``ground_truth`` does not contain ``correct_idx`` —
            this is a programmer error (the assembler always populates
            it), not a Solver failure.
    """

    correct_idx = int(ground_truth["correct_idx"])
    if not (0 <= correct_idx < 4):
        raise ValueError(
            f"ground_truth.correct_idx must be in [0, 3]; got {correct_idx}"
        )
    if parsed is None:
        return 0.0
    raw_pred = parsed.get("predicted_idx")
    try:
        predicted_idx = int(raw_pred)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return 1.0 if predicted_idx == correct_idx else 0.0


__all__ = ["score_vqa"]
