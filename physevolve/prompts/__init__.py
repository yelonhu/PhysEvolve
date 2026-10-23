"""Prompt templates shared by training and inference callers.
"""

from physevolve.prompts.proposer_stage1 import (
    DIFFICULTY_LEVELS,
    MECHANISM_TARGETS,
    PHYSICS_SKILL_TARGETS,
    PROPOSER_STAGE1_TEMPLATE,
    build_proposer_stage1_prompt,
)
from physevolve.prompts.proposer_stage2 import (
    PROPOSER_STAGE2_TEMPLATE,
    build_proposer_scene_template_prompt,
)
from physevolve.prompts.solver import (
    FORWARD_TEMPLATE,
    INVERSE_TEMPLATE,
    PARAMETER_TEMPLATE,
    MC_SHORT_ANSWER_INSTRUCTION,
    VQA_TEMPLATE,
    VQA_TRAINING_TEMPLATE,
)

__all__ = [
    "DIFFICULTY_LEVELS",
    "MECHANISM_TARGETS",
    "PHYSICS_SKILL_TARGETS",
    "PROPOSER_STAGE1_TEMPLATE",
    "PROPOSER_STAGE2_TEMPLATE",
    "FORWARD_TEMPLATE",
    "INVERSE_TEMPLATE",
    "PARAMETER_TEMPLATE",
    "MC_SHORT_ANSWER_INSTRUCTION",
    "VQA_TEMPLATE",
    "VQA_TRAINING_TEMPLATE",
    "build_proposer_stage1_prompt",
    "build_proposer_scene_template_prompt",
]
