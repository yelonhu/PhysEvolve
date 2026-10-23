 # Official PhysBench Eval Vendor

This directory vendors the PhysBench evaluation semantics used by the
official repository:

https://github.com/physical-superintelligence-lab/PhysBench/tree/main/eval

The vendored modules are intentionally small and focused on the parts needed
for PhysEvolve benchmark runs:

- `eval_utils/task_evaluator.py`: official dataset split, mode filtering,
  end prompt, result schema, resume behavior, and visual path mapping.
- `eval_utils/caculate_core.py`: official local scoring semantics for val/test
  answer files.
- `print_test_score.py`: official print-style entry point over the scorer.

The upstream project is licensed under Apache License 2.0. `LICENSE` in this
directory is copied from the upstream repository. Local changes are limited to
formatting, type hints, dependency trimming, and optional hooks for audit
metadata; the prompt, split/mode filtering, result JSON schema, and scoring
rule are kept aligned with upstream.
