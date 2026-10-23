"""Inference-side utilities — parsing + model backends.

This package is intentionally torch-free at the top level. The HF model
backend imports torch lazily so modules like ``parsing.py`` can be used
in worker processes that don't need PyTorch.

Public surface:

* :func:`extract_answer_block` / :func:`extract_propose_block` — regex
  pullers for the locked ``<answer>`` / ``<propose>`` tags.
* :func:`extract_json_from_answer` — tolerant JSON parser (json5
  fallback for trailing-comma permissiveness).
* :func:`extract_mc_letter` — Stage-2 multiple-choice letter extractor.
* :class:`HFBackend` — HuggingFace ``AutoModelForImageTextToText`` wrapper
  for evaluation inference. Imported lazily.
"""

from physevolve.inference.parsing import (
    extract_answer_block,
    extract_json_from_answer,
    extract_mc_letter,
    extract_propose_block,
)

__all__ = [
    "extract_answer_block",
    "extract_json_from_answer",
    "extract_mc_letter",
    "extract_propose_block",
]
