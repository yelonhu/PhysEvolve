"""Verification layer — reward computation.

Public surface:

* :func:`score_forward`, :func:`score_inverse`, :func:`score_parameter`
  — the Stage-1 dense reward functions.
* :func:`compute_proposer_reward` — failure penalty + learnability.

The scorers are pure numeric functions; parsing Solver text lives in
``tasks.router.parse_solver_output`` so malformed JSON returns ``None``
(→ reward 0) before a scorer is ever invoked.
"""

from physevolve.verification.dense_scorer import (
    score_forward,
    score_inverse,
    score_parameter,
)
from physevolve.verification.naive_diagnostics import (
    COLLAPSE_MID_RUNG,
    COLLAPSE_VELOCITY_EXTRAPOLATION,
    COLLAPSE_ZERO_VELOCITY,
    classify_collapse_forward,
    classify_collapse_inverse,
    classify_collapse_parameter,
    exact_match_forward,
    exact_match_inverse,
    exact_match_parameter,
    naive_diagnostics,
)
from physevolve.verification.proposer_reward import (
    CopyControl,
    IntrinsicBreakdown,
    IntrinsicInputs,
    apply_copy_control,
    compute_proposer_reward,
    compute_proposer_reward_with_intrinsic,
    compute_scene_signature,
    intrinsic_reward_terms,
)
from physevolve.verification.vqa_scorer import score_vqa

__all__ = [
    "COLLAPSE_MID_RUNG",
    "COLLAPSE_VELOCITY_EXTRAPOLATION",
    "COLLAPSE_ZERO_VELOCITY",
    "CopyControl",
    "IntrinsicBreakdown",
    "IntrinsicInputs",
    "apply_copy_control",
    "classify_collapse_forward",
    "classify_collapse_inverse",
    "classify_collapse_parameter",
    "compute_proposer_reward",
    "compute_proposer_reward_with_intrinsic",
    "compute_scene_signature",
    "exact_match_forward",
    "exact_match_inverse",
    "exact_match_parameter",
    "intrinsic_reward_terms",
    "naive_diagnostics",
    "score_forward",
    "score_inverse",
    "score_parameter",
    "score_vqa",
]
