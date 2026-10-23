"""Vendored PhysBench task evaluator.

Source:
https://github.com/physical-superintelligence-lab/PhysBench/blob/main/eval/eval_utils/task_evaluator.py

Local changes:
- formatted with normal Python indentation and type hints;
- trimmed model-specific branches that are not needed for the local Qwen3-VL
  adapter, while preserving the official general-model prompt/visual/result
  path;
- added optional ``set_current_item`` model hook so local adapters can write
  audit metadata without changing the public ``qa(image, prompt, mode)``
  interface.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm


test_frame = 8  # 1, 2, 4, 8(default), 16, 32 in upstream.

task_split: dict[str, str] = {
    # Local Qwen adapters are general interleaved models.
    "Qwen3-VL-8B-Instruct": "general",
    "Qwen3-VL-8B-Thinking": "general",
    "qwen3-vl-local": "general",
    # Upstream general-model keys retained for parity/reference.
    "llava-interleave-qwen-7b-hf": "general",
    "llava-interleave-qwen-7b-dpo-hf": "general",
    "vila-1.5-3b-s2": "general",
    "vila-1.5-3b": "general",
    "vila-1.5-8b": "general",
    "vila-1.5-13b": "general",
    "Phi-3-vision-128k-instruct": "general",
    "Phi-3.5V": "general",
    "mPLUG-Owl3-1B-241014": "general",
    "mPLUG-Owl3-2B-241014": "general",
    "mPLUG-Owl3-7B-241101": "general",
    "LLaVA-NeXT-Video-7B-DPO-hf": "general",
    "LLaVA-NeXT-Video-7B-hf": "general",
    "Mantis-8B-Idefics2": "general",
    "Mantis-llava-7b": "general",
    "Mantis-8B-siglip-llama3": "general",
    "Mantis-8B-clip-llama3": "general",
    "InternVL2-1B": "general",
    "InternVL2-2B": "general",
    "InternVL2-4B": "general",
    "InternVL2-8B": "general",
    "InternVL2-26B": "general",
    "InternVL2-40B": "general",
    "InternVL2-76B": "general",
    "InternVL2_5-1B": "general",
    "InternVL2_5-2B": "general",
    "InternVL2_5-4B": "general",
    "InternVL2_5-8B": "general",
    "InternVL2_5-26B": "general",
    "InternVL2_5-38B": "general",
    "InternVL2_5-78B": "general",
    "gpt4v": "general",
    "gpt4o": "general",
    "o1": "general",
    "gpt4o-mini": "general",
    "gemini-1.5-flash": "general",
    "gemini-1.5-pro": "general",
}


class PhysionBenchEvaluator:
    """Official PhysBench evaluator interface for model ``qa`` adapters."""

    def __init__(
        self,
        model: Any,
        mode: str,
        dataset_path: str,
        model_name: str,
        sample_ratio: float | None = None,
        resume: bool = True,
        split: str = "test",
    ) -> None:
        super().__init__()
        self.model = model
        self.mode = mode
        self.model_name = model_name
        self.seed = 2024  # upstream fixed seed
        self.resume = resume

        assert mode in ["image-only", "image&video", "general"], (
            f"not supporting {mode}"
        )

        # Official end prompt, copied verbatim from upstream.
        self.end_prompt = (
            "\nAnswer with the option's letter from the given choices directly. "
            "You can only answer one letter from A, B, C, or D."
        )
        self.video2image_prompt = (
            "This is a series of images sampled at equal intervals from the "
            "beginning to the end of a video, based on the series of images, "
            "output the best option for the question.\n"
        )

        self.sample_ratio = sample_ratio
        self.split = split
        self._load_dataset(dataset_path)

    def _load_dataset(self, dataset_path: str, result_path: str = "results") -> None:
        self.dataset_path = dataset_path
        os.makedirs(os.path.join(self.dataset_path, result_path), exist_ok=True)
        if self.sample_ratio is None:
            self.result_file = os.path.join(
                self.dataset_path, result_path, self.model_name + ".json"
            )
        else:
            self.result_file = os.path.join(
                self.dataset_path,
                result_path,
                self.model_name + f"_{self.sample_ratio}" + ".json",
            )

        if test_frame != 8:
            self.result_file = self.result_file.replace(
                self.model_name, f"{self.model_name}_{test_frame}"
            )
            self.mode = "video-only"

        with open(
            os.path.join(self.dataset_path, "test.json"),
            "r",
            encoding="utf-8",
        ) as file:
            dataset = json.load(file)

        if self.split == "val":
            dataset = [item for item in dataset if item["split"] == "val"]
            self.result_file = self.result_file.replace(".json", "_val.json")

        if self.mode == "image-only":
            self.dataset = [item for item in dataset if item["mode"] != "general"]
        elif self.mode == "image&video":
            self.dataset = [item for item in dataset if item["mode"] != "general"]
        elif self.mode == "video-only":
            self.dataset = [item for item in dataset if item["mode"] == "image&video"]
        else:
            self.dataset = dataset

        self.model_answers: list[dict[str, Any]] = []

        if self.sample_ratio is not None:
            assert self.resume is False
            random.seed(self.seed)
            dataset_size = len(self.dataset)
            sample_size = int(self.sample_ratio * dataset_size)
            self.dataset = random.sample(self.dataset, sample_size)
        elif self.resume and os.path.exists(self.result_file):
            with open(self.result_file, "r", encoding="utf-8") as file:
                model_answers = json.load(file)
            existing_items = {
                item["idx"] for item in model_answers if item["answer"] is not None
            }
            not_match = 0
            for answer in model_answers:
                if answer["answer"] is None or answer["answer"] == "":
                    continue
                matching_item = next(
                    (item for item in self.dataset if item["idx"] == answer["idx"]),
                    None,
                )
                if matching_item is not None:
                    self.model_answers.append(answer)
                else:
                    not_match += 1
            if not_match != 0:
                print("[Info] Unmatched: ", not_match)
            self.dataset = [
                item for item in self.dataset if item["idx"] not in existing_items
            ]
            print(f"[Info] Still have {len(self.dataset)} to test")

    def _process_visual_path(self, file_name: str) -> str:
        if file_name.endswith(".mp4"):
            return str(Path(self.dataset_path) / "video" / file_name)
        if (
            file_name.endswith(".jpg")
            or file_name.endswith(".JPG")
            or file_name.endswith(".png")
        ):
            return str(Path(self.dataset_path) / "image" / file_name)
        raise NotImplementedError

    def test(self) -> None:
        """Run the official general-model path and persist JSON after each item."""

        for item in tqdm(self.dataset):
            prompt = item["question"] + self.end_prompt
            visuals = [self._process_visual_path(f) for f in item["file_name"]]
            hook = getattr(self.model, "set_current_item", None)
            if callable(hook):
                hook(item=item, prompt=prompt, visuals=visuals)
            answer = self.model.qa(image=visuals, prompt=prompt, mode=item["mode"])
            self.model_answers.append(
                {
                    "idx": item["idx"],
                    "answer": answer,
                }
            )

            os.makedirs(os.path.dirname(self.result_file), exist_ok=True)
            with open(self.result_file, "w", encoding="utf-8") as file:
                json.dump(self.model_answers, file, ensure_ascii=False, indent=4)


__all__ = ["PhysionBenchEvaluator", "task_split", "test_frame"]
