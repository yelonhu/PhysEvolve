"""Compatibility fixes for the current veRL worker environment.

This module is imported through ``actor_rollout_ref.model.external_lib`` so the
fixes run inside Ray worker processes before veRL builds and monkey-patches the
HF model. Keep this module small and tied to explicit upstream gaps.
"""

from __future__ import annotations

from typing import Any


VALUE_HEAD_SYMBOL = "AutoModelForCausalLMWithValueHead"
QWEN3VL_INPUT_EMBEDS_PATCH = "qwen3vl_input_embeds"
QWEN3VL_FAST_POS_PATCH = "qwen3vl_fast_pos_embed"
_PATCHED_TARGETS: dict[str, Any] = {}


def ensure_trl_value_head_compat() -> dict[str, Any]:
    """Provide the legacy TRL value-head symbol expected by current veRL.

    Newer TRL releases no longer export ``AutoModelForCausalLMWithValueHead``,
    while this veRL checkout only patches ``state_dict`` on that class. A tiny
    stub is enough for that monkey patch path and avoids changing the conda
    environment for the training smoke.
    """

    try:
        import trl  # type: ignore
    except Exception as exc:  # pragma: no cover - environment specific.
        return {
            "trl_available": False,
            "has_value_head": False,
            "patched": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if hasattr(trl, VALUE_HEAD_SYMBOL):
        return {
            "trl_available": True,
            "has_value_head": True,
            "patched": False,
            "error": None,
        }

    class AutoModelForCausalLMWithValueHead:  # noqa: D401 - compatibility stub.
        """Compatibility stub for veRL's value-head monkey patch."""

    setattr(trl, VALUE_HEAD_SYMBOL, AutoModelForCausalLMWithValueHead)
    return {
        "trl_available": True,
        "has_value_head": True,
        "patched": True,
        "error": None,
    }


def trl_value_head_compat_status() -> dict[str, Any]:
    """Return the import-time TRL compatibility status."""

    return dict(_IMPORT_STATUS)


def apply_qwen3vl_fsdp_patches() -> dict[str, Any]:
    """Apply the two Qwen3-VL fixes needed by current veRL FSDP workers.

    Upstream gaps this isolates:

    1. ``verl/workers/engine/fsdp/transformer_impl.py:885`` calls
       ``extract_multi_modal_inputs`` after ``micro_batch.to(device)``, so
       multimodal tensors from the non-tensor path can remain on CPU.
       We wrap veRL's Qwen3-VL ``_get_input_embeds`` boundary and move visual
       tensors to ``input_ids.device``.
    2. ``transformers.models.qwen3_vl.modeling_qwen3_vl`` in the current
       environment derives interpolation index devices from
       ``self.pos_embed.weight.device`` before FSDP CPUOffload has gathered
       that embedding shard.  We replace only that device assumption: the
       interpolation tensors follow the incoming ``grid_thw.device``.

    Delete this function once veRL moves multimodal tensors in
    ``prepare_model_inputs`` and Transformers derives Qwen3-VL interpolation
    indices from input tensors instead of a possibly offloaded parameter.
    """

    input_status = _patch_qwen3vl_input_embeds()
    fast_pos_status = _patch_qwen3vl_fast_pos_embed()
    errors = [
        str(status["error"])
        for status in (input_status, fast_pos_status)
        if status.get("error")
    ]
    return {
        "available": bool(input_status.get("available") or fast_pos_status.get("available")),
        "patched": bool(input_status.get("patched") or fast_pos_status.get("patched")),
        "input_embeds": input_status,
        "fast_pos": fast_pos_status,
        "error": "; ".join(errors) or None,
    }


def qwen3vl_fsdp_patch_status() -> dict[str, Any]:
    """Return the import-time Qwen3-VL FSDP patch status."""

    return dict(_QWEN3VL_IMPORT_STATUS)


def _patch_qwen3vl_input_embeds() -> dict[str, Any]:
    try:
        from verl.models.transformers import qwen3_vl  # type: ignore
    except Exception as exc:  # pragma: no cover - optional veRL import.
        return {
            "available": False,
            "patched": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    original = getattr(qwen3_vl, "_get_input_embeds", None)
    if original is None:
        return {
            "available": True,
            "patched": False,
            "error": "_get_input_embeds not found",
        }
    if original is _PATCHED_TARGETS.get(QWEN3VL_INPUT_EMBEDS_PATCH):
        return {
            "available": True,
            "patched": False,
            "error": None,
        }

    def _wrapped_get_input_embeds(
        model: Any,
        input_ids: Any,
        attention_mask: Any = None,
        pixel_values: Any = None,
        pixel_values_videos: Any = None,
        image_grid_thw: Any = None,
        video_grid_thw: Any = None,
    ):
        device = getattr(input_ids, "device", None)
        if device is not None:
            if hasattr(pixel_values, "to"):
                pixel_values = pixel_values.to(device, non_blocking=True)
            if hasattr(pixel_values_videos, "to"):
                pixel_values_videos = pixel_values_videos.to(device, non_blocking=True)
            if hasattr(image_grid_thw, "to"):
                image_grid_thw = image_grid_thw.to(device, non_blocking=True)
            if hasattr(video_grid_thw, "to"):
                video_grid_thw = video_grid_thw.to(device, non_blocking=True)
        return original(
            model,
            input_ids,
            attention_mask,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
        )

    qwen3_vl._get_input_embeds = _wrapped_get_input_embeds
    _PATCHED_TARGETS[QWEN3VL_INPUT_EMBEDS_PATCH] = _wrapped_get_input_embeds
    return {
        "available": True,
        "patched": True,
        "error": None,
    }


def _patch_qwen3vl_fast_pos_embed() -> dict[str, Any]:
    try:
        import torch
        from transformers.models.qwen3_vl import modeling_qwen3_vl  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency import.
        return {"available": False, "patched": False, "error": f"{type(exc).__name__}: {exc}"}

    vision_cls = getattr(modeling_qwen3_vl, "Qwen3VLVisionModel", None)
    if vision_cls is None:
        return {"available": True, "patched": False, "error": "Qwen3VLVisionModel not found"}
    current = getattr(vision_cls, "fast_pos_embed_interpolate", None)
    if current is None:
        return {"available": True, "patched": False, "error": "fast_pos_embed_interpolate not found"}
    if current is _PATCHED_TARGETS.get(QWEN3VL_FAST_POS_PATCH):
        return {"available": True, "patched": False, "error": None}

    def _fast_pos_embed_interpolate(self: Any, grid_thw: Any):  # noqa: ANN001
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = grid_thw.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for _t, h, w in zip(grid_ts, grid_hs, grid_ws, strict=False):
            h_int = int(h.item())
            w_int = int(w.item())
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h_int, device=device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w_int, device=device)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        split_sizes = [int(h.item()) * int(w.item()) for h, w in zip(grid_hs, grid_ws, strict=False)]
        patch_pos_embeds = patch_pos_embeds.split(split_sizes)

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=False):
            t_int = int(t.item())
            h_int = int(h.item())
            w_int = int(w.item())
            pos_embed = pos_embed.repeat(t_int, 1)
            pos_embed = (
                pos_embed.view(t_int, h_int // merge_size, merge_size, w_int // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    vision_cls.fast_pos_embed_interpolate = _fast_pos_embed_interpolate
    _PATCHED_TARGETS[QWEN3VL_FAST_POS_PATCH] = _fast_pos_embed_interpolate
    return {"available": True, "patched": True, "error": None}


_IMPORT_STATUS = ensure_trl_value_head_compat()
_QWEN3VL_IMPORT_STATUS = apply_qwen3vl_fsdp_patches()


__all__ = [
    "VALUE_HEAD_SYMBOL",
    "apply_qwen3vl_fsdp_patches",
    "ensure_trl_value_head_compat",
    "qwen3vl_fsdp_patch_status",
    "trl_value_head_compat_status",
]
