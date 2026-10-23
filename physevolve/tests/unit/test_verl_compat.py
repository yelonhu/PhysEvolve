"""Tests for PhysEvolve-native veRL compatibility helpers."""

from __future__ import annotations

import sys
import types

import torch

from physevolve.training.verl_compat import (
    VALUE_HEAD_SYMBOL,
    apply_qwen3vl_fsdp_patches,
    ensure_trl_value_head_compat,
)


def test_trl_value_head_compat_installs_missing_symbol(monkeypatch) -> None:
    fake_trl = types.ModuleType("trl")
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    status = ensure_trl_value_head_compat()

    assert status["trl_available"] is True
    assert status["has_value_head"] is True
    assert status["patched"] is True
    assert hasattr(fake_trl, VALUE_HEAD_SYMBOL)


def test_trl_value_head_compat_is_noop_when_symbol_exists(monkeypatch) -> None:
    fake_trl = types.ModuleType("trl")

    class ExistingValueHead:
        """Fake existing TRL value-head class."""

    setattr(fake_trl, VALUE_HEAD_SYMBOL, ExistingValueHead)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)

    status = ensure_trl_value_head_compat()

    assert status["trl_available"] is True
    assert status["has_value_head"] is True
    assert status["patched"] is False
    assert getattr(fake_trl, VALUE_HEAD_SYMBOL) is ExistingValueHead


def test_qwen3vl_fsdp_patches_move_visual_tensors(monkeypatch) -> None:
    fake_qwen3_vl = types.ModuleType("qwen3_vl")
    seen = {}

    def _get_input_embeds(
        model,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
    ):
        seen["pixel_values_videos_device"] = pixel_values_videos.device
        seen["video_grid_thw_device"] = video_grid_thw.device
        return {"ok": True}

    fake_qwen3_vl._get_input_embeds = _get_input_embeds
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.qwen3_vl = fake_qwen3_vl
    fake_models = types.ModuleType("models")
    fake_models.transformers = fake_transformers
    fake_verl = types.ModuleType("verl")
    fake_verl.models = fake_models

    monkeypatch.setitem(sys.modules, "verl", fake_verl)
    monkeypatch.setitem(sys.modules, "verl.models", fake_models)
    monkeypatch.setitem(sys.modules, "verl.models.transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "verl.models.transformers.qwen3_vl", fake_qwen3_vl)

    status = apply_qwen3vl_fsdp_patches()
    assert status["input_embeds"]["patched"] is True

    input_ids = torch.ones((1, 2), dtype=torch.long, device="meta")
    fake_qwen3_vl._get_input_embeds(
        object(),
        input_ids,
        pixel_values_videos=torch.ones((1, 2), dtype=torch.float32),
        video_grid_thw=torch.ones((1, 3), dtype=torch.long),
    )

    assert seen["pixel_values_videos_device"] == input_ids.device
    assert seen["video_grid_thw_device"] == input_ids.device


def test_qwen3vl_fast_pos_patch_uses_grid_device_not_weight_device(monkeypatch) -> None:
    fake_modeling = types.ModuleType("modeling_qwen3_vl")

    class FakePosEmbed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((16, 2), device="meta"))
            self.seen_index_device = None

        def forward(self, index: torch.Tensor) -> torch.Tensor:
            self.seen_index_device = index.device
            return torch.ones((*index.shape, 2), device=index.device)

    class FakeVisionModel:
        def __init__(self) -> None:
            self.pos_embed = FakePosEmbed()
            self.num_grid_per_side = 4
            self.config = types.SimpleNamespace(spatial_merge_size=1)

        def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
            raise AssertionError("unpatched fast_pos_embed_interpolate should not run")

    fake_modeling.Qwen3VLVisionModel = FakeVisionModel
    fake_qwen3_vl_pkg = types.ModuleType("qwen3_vl")
    fake_qwen3_vl_pkg.modeling_qwen3_vl = fake_modeling
    fake_models = types.ModuleType("models")
    fake_models.qwen3_vl = fake_qwen3_vl_pkg
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.models = fake_models

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.models", fake_models)
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_vl", fake_qwen3_vl_pkg)
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
        fake_modeling,
    )

    status = apply_qwen3vl_fsdp_patches()
    assert status["fast_pos"]["patched"] is True

    model = FakeVisionModel()
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
    pos_embeds = model.fast_pos_embed_interpolate(grid_thw)

    assert model.pos_embed.seen_index_device == grid_thw.device
    assert pos_embeds.device == grid_thw.device
