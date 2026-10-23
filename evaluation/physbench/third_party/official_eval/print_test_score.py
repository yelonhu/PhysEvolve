"""Vendored PhysBench print-score entry point.

Source:
https://github.com/physical-superintelligence-lab/PhysBench/blob/main/eval/print_test_score.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .eval_utils.caculate_core import calculate_accuracy, print_accuracies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--name", type=str, default="Unknown")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    accuracies = calculate_accuracy(args.answer_file, args.test_file)
    print_accuracies(accuracies, name=args.name)
    print("Completed evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
