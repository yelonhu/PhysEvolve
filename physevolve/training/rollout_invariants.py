"""Shape and grouping invariants for PhysEvolve veRL rollouts."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch


ROLLOUT_NON_TENSOR_KEYS: tuple[str, ...] = (
    "raw_prompt",
    "agent_name",
    "index",
    "uid",
    "group_id",
    "role",
    "problem_type",
    "scene_id",
    "data_source",
    "reward_model",
    "stop_sequences",
    "max_tokens",
    "nz_video_bundle",
)


def validate_update_batch(batch: Any, scores: Any, *, group_size: int) -> None:
    """Fail fast before PPO if PhysEvolve rollout rows cannot support GRPO."""

    if not hasattr(batch, "batch") or not hasattr(batch, "non_tensor_batch"):
        raise TypeError("rollout_batch must be a veRL DataProto-like object")
    if "responses" not in batch.batch.keys():
        raise ValueError("rollout_batch.batch must contain responses")
    if "uid" not in batch.non_tensor_batch:
        raise ValueError("rollout_batch.non_tensor_batch must contain uid for GRPO")

    if not isinstance(scores, torch.Tensor):
        raise TypeError("token_level_scores must be a torch.Tensor")
    batch_size = int(batch.batch.batch_size[0])
    response_length = int(batch.batch["responses"].shape[1])
    if tuple(scores.shape) != (batch_size, response_length):
        raise ValueError(
            "token_level_scores shape must match rollout responses: "
            f"got {tuple(scores.shape)}, expected {(batch_size, response_length)}"
        )

    uid = batch.non_tensor_batch["uid"]
    if not isinstance(uid, np.ndarray) or uid.shape[0] != batch_size:
        raise ValueError("uid must be a numpy array aligned to rollout batch dimension")
    if any(x is None or str(x) == "" for x in uid.tolist()):
        raise ValueError("uid values must be non-empty")

    group_size = int(group_size)
    if group_size <= 1:
        return
    counts = Counter(str(x) for x in uid.tolist())
    bad = {key: count for key, count in counts.items() if count != group_size}
    if bad:
        preview = dict(list(bad.items())[:5])
        raise ValueError(
            "each GRPO uid group must have exactly group_size rows; "
            f"group_size={group_size}, bad_groups={preview}"
        )


def validate_multimodal_token_alignment(batch: Any, processor: Any) -> dict[str, int]:
    """Validate Qwen-VL video placeholder tokens before FSDP forward.

    veRL stores processor outputs in ``non_tensor_batch["multi_modal_inputs"]``
    while the model sees prompt/response ids in ``batch["input_ids"]``. For
    Qwen-VL those two views must agree: one video placeholder token per
    merged visual feature token.
    """

    if processor is None or not hasattr(batch, "batch") or not hasattr(batch, "non_tensor_batch"):
        return {}
    if "input_ids" not in batch.batch.keys():
        return {}
    multi_modal_inputs = batch.non_tensor_batch.get("multi_modal_inputs")
    if multi_modal_inputs is None:
        return {}

    video_token_id = getattr(processor, "video_token_id", None)
    if video_token_id is None:
        return {}

    expected = 0
    for item in _iter_non_tensor_items(multi_modal_inputs):
        if item is None:
            continue
        if hasattr(item, "data"):
            item = item.data
        if not isinstance(item, dict):
            continue
        expected += _video_feature_tokens(
            item.get("video_grid_thw"),
            merge_size=_video_merge_size(processor),
        )

    observed = int((batch.batch["input_ids"] == int(video_token_id)).sum().item())
    if expected and observed != expected:
        batch_size = int(batch.batch.batch_size[0])
        raise ValueError(
            "PhysEvolve Qwen-VL video token alignment failed before old_logprob: "
            f"video_tokens={observed}, video_feature_tokens={expected}, "
            f"batch_size={batch_size}. This usually means compact vLLM video "
            "tokens leaked into the FSDP logprob/update path."
        )
    return {
        "video_tokens": observed,
        "video_feature_tokens": expected,
    }


def response_lengths_from_mask(batch: Any) -> list[int]:
    """Return valid response lengths from a rollout DataProto."""

    if "response_mask" not in batch.batch.keys():
        responses = batch.batch["responses"]
        return [int(responses.shape[1])] * int(responses.shape[0])
    mask = batch.batch["response_mask"]
    return [max(1, int(x)) for x in mask.sum(dim=-1).tolist()]


def _iter_non_tensor_items(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _video_feature_tokens(video_grid_thw: Any, *, merge_size: int) -> int:
    if video_grid_thw is None:
        return 0
    grid = video_grid_thw
    if hasattr(grid, "detach"):
        grid = grid.detach().cpu()
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    if isinstance(grid, tuple):
        grid = list(grid)
    if not isinstance(grid, list):
        raise TypeError(f"video_grid_thw must be list-like; got {type(video_grid_thw).__name__}")
    if grid and not isinstance(grid[0], (list, tuple)):
        grid = [grid]
    denom = max(1, int(merge_size) ** 2)
    total = 0
    for row in grid:
        if len(row) != 3:
            raise ValueError(f"video_grid_thw rows must have length 3; got {row!r}")
        t, h, w = (int(x) for x in row)
        total += (t * h * w) // denom
    return int(total)


def _video_merge_size(processor: Any) -> int:
    video_processor = getattr(processor, "video_processor", None)
    return int(getattr(video_processor, "merge_size", 2) or 2)


__all__ = [
    "ROLLOUT_NON_TENSOR_KEYS",
    "response_lengths_from_mask",
    "validate_multimodal_token_alignment",
    "validate_update_batch",
]

