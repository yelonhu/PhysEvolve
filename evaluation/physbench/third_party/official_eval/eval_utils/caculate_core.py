"""Vendored PhysBench scorer.

Source:
https://github.com/physical-superintelligence-lab/PhysBench/blob/main/eval/eval_utils/caculate_core.py

Local changes:
- formatted with standard indentation and type hints;
- added ``format_markdown`` for writing stable report files;
- kept official ``answer_true`` semantics exactly.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


task_type_order = ["property", "relationships", "scene", "dynamics"]
ability_type_order = [
    "identify",
    "comparison",
    "static",
    "dynamic",
    "perception",
    "prediction",
    "judgment",
    "reasoning",
]

sub_task_order = [
    "number",
    "mass",
    "color",
    "attribute",
    "size",
    "location",
    "depth",
    "distance",
    "motion",
    "temperature",
    "viewpoint",
    "air",
    "light",
    "collision",
    "throwing",
    "manipulation",
    "fluid",
    "chemistry",
    "others",
]


def answer_true(item: dict[str, Any], gt_item: dict[str, Any]) -> bool:
    return item["answer"] == gt_item["answer"] or item["answer"].startswith(
        gt_item["answer"]
    )


def calculate_accuracy(
    val_annotation_file: str | Path,
    user_submission_file: str | Path,
) -> dict[str, Any]:
    with open(user_submission_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    with open(val_annotation_file, "r", encoding="utf-8") as file:
        gt_data = json.load(file)
    total_questions = len(data)
    correct_answers = 0

    task_type_counts: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": -1, "total": -1}
    )
    sub_type_counts: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": -1, "total": -1}
    )
    ability_type_counts: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": -1, "total": -1}
    )

    for item in data:
        if item["answer"] is None:
            continue

        gt_item = next((gt for gt in gt_data if gt["idx"] == item["idx"]), None)

        if gt_item is None:
            print(f'Unknown idx : {item["idx"]}')
            continue

        if answer_true(item, gt_item):
            correct_answers += 1
            task_type_counts[gt_item["task_type"]]["correct"] = (
                max(task_type_counts[gt_item["task_type"]]["correct"], 0) + 1
            )
            sub_type_counts[gt_item["sub_type"]]["correct"] = (
                max(sub_type_counts[gt_item["sub_type"]]["correct"], 0) + 1
            )
            ability_type_counts[gt_item["ability_type"]]["correct"] = (
                max(ability_type_counts[gt_item["ability_type"]]["correct"], 0) + 1
            )

        task_type_counts[gt_item["task_type"]]["total"] = (
            max(task_type_counts[gt_item["task_type"]]["total"], 0) + 1
        )
        sub_type_counts[gt_item["sub_type"]]["total"] = (
            max(sub_type_counts[gt_item["sub_type"]]["total"], 0) + 1
        )
        ability_type_counts[gt_item["ability_type"]]["total"] = (
            max(ability_type_counts[gt_item["ability_type"]]["total"], 0) + 1
        )

    overall_accuracy = (
        correct_answers / total_questions * 100 if total_questions > 0 else 0
    )

    def calculate_specific_accuracy(
        counts: defaultdict[str, dict[str, int]],
    ) -> dict[str, dict[str, float | int]]:
        return {
            key: {
                "accuracy": (
                    max(value["correct"] / value["total"] * 100, 0)
                    if value["total"] > 0
                    else -1
                ),
                "correct": value["correct"],
                "total": value["total"],
            }
            for key, value in counts.items()
        }

    task_type_accuracy = calculate_specific_accuracy(task_type_counts)
    sub_type_accuracy = calculate_specific_accuracy(sub_type_counts)
    ability_type_accuracy = calculate_specific_accuracy(ability_type_counts)

    return {
        "overall_accuracy": overall_accuracy,
        "overall_correct": correct_answers,
        "overall_total": total_questions,
        "task_type_accuracy": task_type_accuracy,
        "sub_type_accuracy": sub_type_accuracy,
        "ability_type_accuracy": ability_type_accuracy,
    }


def calculate_weighted_avg(accuracy_data: dict[str, dict[str, Any]]) -> float:
    total_correct = sum(
        data["correct"] for data in accuracy_data.values() if data["total"] > 0
    )
    total_questions = sum(
        data["total"] for data in accuracy_data.values() if data["total"] > 0
    )
    if total_questions > 0:
        return total_correct / total_questions * 100
    return -1


def format_markdown(accuracies: dict[str, Any], name: str = "Unknown") -> str:
    lines = [
        f"# PhysBench Official Eval — {name}",
        "",
        (
            f"Overall Accuracy: **{accuracies['overall_accuracy']:.2f}%** "
            f"({accuracies['overall_correct']} correct out of "
            f"{accuracies['overall_total']})"
        ),
        "",
    ]

    def generate_markdown_table(
        title: str,
        accuracy_data: dict[str, dict[str, Any]],
        order: list[str],
    ) -> str:
        ordered = {
            key: accuracy_data.get(key, {"accuracy": -1, "correct": -1, "total": -1})
            for key in order
        }
        headers = "| model | " + " | ".join(ordered.keys()) + " | avg |"
        separator = "| --- | " + " | ".join(["---"] * len(ordered)) + " | --- |"
        values = (
            f"| {name} | "
            + " | ".join(f"{data['accuracy']:.2f}" for data in ordered.values())
            + f" | {calculate_weighted_avg(ordered):.2f} |"
        )
        return f"**{title}**\n\n{headers}\n{separator}\n{values}\n"

    lines.append(
        generate_markdown_table(
            "Task Type Accuracy",
            accuracies["task_type_accuracy"],
            task_type_order,
        )
    )
    lines.append(
        generate_markdown_table(
            "Sub Type Accuracy",
            accuracies["sub_type_accuracy"],
            sub_task_order,
        )
    )
    lines.append(
        generate_markdown_table(
            "Ability Type Accuracy",
            accuracies["ability_type_accuracy"],
            ability_type_order,
        )
    )
    return "\n".join(lines)


def print_accuracies(accuracies: dict[str, Any], name: str = "Unknown") -> None:
    print(
        f"Overall Accuracy: {accuracies['overall_accuracy']:.2f}% "
        f"({accuracies['overall_correct']} correct out of "
        f"{accuracies['overall_total']})"
    )
    print("\nTask Type Accuracies:")
    for task_type, data in accuracies["task_type_accuracy"].items():
        print(
            f" {task_type}: {data['accuracy']:.2f}% "
            f"({data['correct']} correct out of {data['total']})"
        )
    print("\nSub Type Accuracies:")
    for sub_type in sub_task_order:
        data = accuracies["sub_type_accuracy"].get(
            sub_type, {"accuracy": -1, "correct": -1, "total": -1}
        )
        print(
            f" {sub_type}: {data['accuracy']:.2f}% "
            f"({data['correct']} correct out of {data['total']})"
        )
    print("\nAbility Type Accuracies:")
    for ability_type, data in accuracies["ability_type_accuracy"].items():
        print(
            f" {ability_type}: {data['accuracy']:.2f}% "
            f"({data['correct']} correct out of {data['total']})"
        )
    print("\nMarkdown Tables:\n")
    print(format_markdown(accuracies, name=name))


__all__ = [
    "ability_type_order",
    "answer_true",
    "calculate_accuracy",
    "calculate_weighted_avg",
    "format_markdown",
    "print_accuracies",
    "sub_task_order",
    "task_type_order",
]
