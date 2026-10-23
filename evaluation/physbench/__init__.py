"""PhysBench evaluation: official scorer path.

Canonical eval is the vendored official evaluator driven by
``scripts/eval/run_physbench_official_eval.py``. This package exposes
dataset loading, prompting, visual loading, and the local letter scorer
used to join predictions to the vendored public answer files.
"""

from evaluation.physbench.data_loader import (
    PhysBenchItem,
    load_items,
    resolve_visual_paths,
    sample_items,
)
from evaluation.physbench.prompter import (
    END_PROMPT,
    build_content,
    count_tags,
)
from evaluation.physbench.scorer import (
    ABILITY_TYPE_ORDER,
    CategoryStat,
    ScoreReport,
    SUB_TYPE_ORDER,
    TASK_TYPE_ORDER,
    format_markdown,
    format_report,
    print_report,
    score,
)
from evaluation.physbench.visual_loader import load_image, load_video

__all__ = [
    "ABILITY_TYPE_ORDER",
    "CategoryStat",
    "END_PROMPT",
    "PhysBenchItem",
    "SUB_TYPE_ORDER",
    "ScoreReport",
    "TASK_TYPE_ORDER",
    "build_content",
    "count_tags",
    "format_markdown",
    "format_report",
    "load_image",
    "load_items",
    "load_video",
    "print_report",
    "resolve_visual_paths",
    "sample_items",
    "score",
]
