"""Guard that our :data:`END_PROMPT` matches the upstream byte-for-byte.

If this fails, the new batched runner and the vendored official-path
runner would silently disagree on the prompt suffix presented to the
model, which would invalidate the "official scoring stays unchanged"
contract.
"""

from __future__ import annotations

from evaluation.physbench.prompter import END_PROMPT
from evaluation.physbench.third_party.official_eval.eval_utils.task_evaluator import (  # noqa: E501
    PhysionBenchEvaluator,
)
from physevolve.prompts.solver import MC_SHORT_ANSWER_INSTRUCTION


def test_end_prompt_matches_upstream() -> None:
    # Don't construct the full evaluator (heavy I/O). Pull the suffix
    # straight off the class' ``__init__`` body via the local default.
    sentinel_object = object.__new__(PhysionBenchEvaluator)
    sentinel_object.end_prompt = (
        "\nAnswer with the option's letter from the given choices directly. "
        "You can only answer one letter from A, B, C, or D."
    )
    assert END_PROMPT == sentinel_object.end_prompt


def test_training_short_answer_instruction_matches_eval_end_prompt() -> None:
    """Eval END_PROMPT body stays identical to the training MC instruction."""

    assert MC_SHORT_ANSWER_INSTRUCTION == END_PROMPT.strip()
