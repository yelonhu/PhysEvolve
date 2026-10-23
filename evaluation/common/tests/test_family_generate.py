"""CPU-only tests for per-family eval generate (no weight load, no GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evaluation.common import family_generate as fg
from evaluation.common.common import (
    GenerationSettings,
    generate_reference_answer,
)
from physevolve.inference.hf_backend import GenerationResult, HFBackend


def _rgb(n_frames: int = 5, size: int = 8) -> np.ndarray:
    return np.full((n_frames, size, size, 3), 40, dtype=np.uint8)


def test_detect_eval_family_from_directory_names() -> None:
    """Path names must pick InternVL / Llama / Pixtral without opening weights."""

    root = Path("models")
    assert fg.detect_eval_family(root / "InternVL2_5-8B") == "internvl"
    assert fg.detect_eval_family(root / "Llama-3.2-11B-Vision-Instruct") == "llama32_vision"
    assert fg.detect_eval_family(root / "Pixtral-12B-2409") == "pixtral"
    assert fg.detect_eval_family(root / "Qwen3-VL-8B-Instruct") == "qwen"


def test_resolve_eval_family_honors_explicit_override() -> None:
    """--model-family wins even when the path name looks like Qwen."""

    qwen = Path("Qwen/Qwen3-VL-8B-Instruct")
    assert fg.resolve_eval_family(qwen, "internvl") == "internvl"
    assert fg.resolve_eval_family(qwen, None) == "qwen"
    assert fg.resolve_eval_family(qwen, "auto") == "qwen"
    with pytest.raises(ValueError, match="unsupported --model-family"):
        fg.resolve_eval_family(qwen, "llava")


def test_assert_local_weights_ready_refuses_empty_dir(tmp_path: Path) -> None:
    """Incomplete dumps must fail locally instead of fetching Hugging Face."""

    empty = tmp_path / "InternVL2_5-8B"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no Hugging Face fetch"):
        fg.assert_local_weights_ready(empty, "internvl")
    cfg_only = tmp_path / "Llama-3.2-11B-Vision-Instruct"
    cfg_only.mkdir()
    (cfg_only / "config.json").write_text(
        '{"model_type": "mllama"}', encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="local weights not ready"):
        fg.assert_local_weights_ready(cfg_only, "llama32_vision")
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError, match="not found"):
        fg.assert_local_weights_ready(missing, "internvl")


def test_video_frames_accept_bare_ndarray_and_presampled_tuple() -> None:
    """Family packing must accept both PhysBench ndarray and ContPhy tuples."""

    frames = _rgb(8)
    bare = fg.video_frames_from_entry(frames)
    wrapped = fg.video_frames_from_entry((frames, {"fps": 2.0}))
    assert len(bare) == 8
    assert len(wrapped) == 8
    assert wrapped[0].shape == (8, 8, 3)


def test_internvl_video_question_uses_frame_prefix() -> None:
    """InternVL video slots must be Frame1: <image>, not InternVL's 32-seg sampler."""

    frames = _rgb(5)
    content = [
        {"type": "video", "video": frames},
        {"type": "text", "text": "Which outcome occurs?"},
    ]
    question = fg.internvl_question_from_content(content)
    assert question.startswith("Frame1: <image>\n")
    assert "Frame5: <image>\n" in question
    assert "Frame6:" not in question
    assert question.endswith("Which outcome occurs?")
    assert question.count("<image>") == 5
    assert fg.count_visual_slots(content) == 5

    source = Path(fg.__file__).read_text(encoding="utf-8")
    assert "from decord" not in source
    assert "import decord" not in source
    assert "VideoReader(" not in source


def test_internvl_question_keeps_official_tag_order() -> None:
    """PhysBench mixed <video>/<image> order must survive InternVL rewriting."""

    content = [
        {"type": "text", "text": "Look "},
        {"type": "video", "video": _rgb(2)},
        {"type": "text", "text": " then "},
        {"type": "image", "image": _rgb(1)[0]},
        {"type": "text", "text": " choose."},
    ]
    question = fg.internvl_question_from_content(content)
    assert question == (
        "Look Frame1: <image>\nFrame2: <image>\n then Image-1: <image>\n choose."
    )


def test_internvl_dynamic_preprocess_tile_count() -> None:
    """Tiny square PIL is one 448 tile; a wide still uses more tiles + thumbnail."""

    square = Image.new("RGB", (16, 16), color=(10, 20, 30))
    square_tiles = fg.internvl_dynamic_preprocess(
        square, max_num=1, image_size=448, use_thumbnail=True
    )
    assert len(square_tiles) == 1
    assert square_tiles[0].size == (448, 448)

    wide = Image.new("RGB", (800, 200), color=(40, 50, 60))
    wide_tiles = fg.internvl_dynamic_preprocess(
        wide, max_num=12, image_size=448, use_thumbnail=True
    )
    assert len(wide_tiles) > 1
    assert all(tile.size == (448, 448) for tile in wide_tiles)


def test_llama_and_pixtral_flatten_video_to_n_images() -> None:
    """Llama/Pixtral must expand N frames into N image slots, not a Qwen video tensor."""

    content = [
        {"type": "video", "video": (_rgb(3), {"fps": 2.0})},
        {"type": "text", "text": "prompt"},
    ]
    llama_parts, llama_images = fg.llama_style_chat_content(content)
    assert [part["type"] for part in llama_parts] == ["image", "image", "image", "text"]
    assert len(llama_images) == 3

    pixtral_parts = fg.pixtral_chat_content(content)
    urls = [
        part["image_url"]["url"]
        for part in pixtral_parts
        if part["type"] == "image_url"
    ]
    assert len(urls) == 3
    assert all(url.startswith("data:image/jpeg;base64,") for url in urls)
    with pytest.raises(ValueError, match="at most 12"):
        fg.pixtral_chat_content(
            [{"type": "video", "video": _rgb(13)}, {"type": "text", "text": "x"}]
        )


def test_open_eval_backend_selects_family_without_loading() -> None:
    """Constructing a family backend must not load weights or start vLLM."""

    internvl = fg.open_eval_backend(
        "models/InternVL2_5-8B"
    )
    llama = fg.open_eval_backend(
        "models/Llama-3.2-11B-Vision-Instruct"
    )
    pixtral = fg.open_eval_backend(
        "models/Pixtral-12B-2409"
    )
    qwen = fg.open_eval_backend(
        "Qwen/Qwen3-VL-8B-Instruct"
    )
    assert isinstance(internvl, fg.InternVLFamilyBackend)
    assert internvl._model is None
    assert isinstance(llama, fg.LlamaVisionFamilyBackend)
    assert llama._model is None
    assert isinstance(pixtral, fg.PixtralFamilyBackend)
    assert pixtral._model is None
    assert isinstance(qwen, HFBackend)
    assert qwen._model is None
    assert "vllm" not in vars(fg)
    assert "LLM" not in vars(fg)


def test_dummy_family_backend_through_generate_reference_answer() -> None:
    """Shared generate_reference_answer must accept a family-shaped dummy backend."""

    class DummyFamilyBackend:
        def generate_multimodal_with_metadata(
            self,
            content: list[dict] | None = None,
            **kwargs: object,
        ) -> GenerationResult:
            assert content is not None
            assert kwargs.get("stop_sequences") == ["</answer>", "<|im_end|>"]
            return GenerationResult(
                text="<answer>B</answer>",
                metadata={"eval_family": "internvl", "fixture": True},
                generated_text="<answer>B</answer>",
                prompt_text="Frame1: <image>\n",
                prompt_tail="Frame1: <image>\n",
            )

    result = generate_reference_answer(
        DummyFamilyBackend(),
        [{"type": "video", "video": _rgb(2)}, {"type": "text", "text": "q"}],
        GenerationSettings(max_new_tokens=512),
    )
    assert result.text == "<answer>B</answer>"
    assert result.metadata["eval_family"] == "internvl"


def test_ensure_hf_generate_adds_generate_when_missing() -> None:
    """transformers>=4.50 remote-code LLMs need GenerationMixin attached."""

    class DummyRemoteLM:
        pass

    dummy = DummyRemoteLM()
    assert not callable(getattr(dummy, "generate", None))
    fg.ensure_hf_generate(dummy)
    assert callable(dummy.generate)
    fg.ensure_hf_generate(dummy)
    assert callable(dummy.generate)


def test_ensure_hf_generate_leaves_existing_generate() -> None:
    """Do not replace a custom generate (InternVLChatModel keeps ViT inject)."""

    class DummyWithGenerate:
        def generate(self, *args: object, **kwargs: object) -> str:
            return "kept"

    dummy = DummyWithGenerate()
    fg.ensure_hf_generate(dummy)
    assert dummy.generate() == "kept"


def test_ensure_hf_generate_fills_generation_config_from_model_config() -> None:
    """InternLM2 constructed without GenerationMixin leaves generation_config None."""

    from transformers import PretrainedConfig

    class DummyLMWithConfig:
        def __init__(self) -> None:
            self.config = PretrainedConfig()

    dummy = DummyLMWithConfig()
    assert getattr(dummy, "generation_config", None) is None
    fg.ensure_hf_generate(dummy)
    assert dummy.generation_config is not None
    assert dummy.generation_config._from_model_config is True
    assert callable(dummy.generate)


def test_legacy_past_key_values_empty_cache_is_none() -> None:
    """Empty HF Cache must become None so InternVL keeps first-step inputs_embeds."""

    assert fg._legacy_past_key_values(None) is None

    class EmptyCache:
        def get_seq_length(self) -> int:
            return 0

    assert fg._legacy_past_key_values(EmptyCache()) is None


def test_legacy_past_key_values_legacy_tuple_passthrough() -> None:
    """Non-empty cache converts to the InternLM2 tuple layout."""

    class FakeKey:
        shape = (1, 1, 4)

    legacy = ((FakeKey(), FakeKey()),)

    class FilledCache:
        def get_seq_length(self) -> int:
            return 4

        def to_legacy_cache(self) -> tuple:
            return legacy

    assert fg._legacy_past_key_values(FilledCache()) is legacy


def test_patch_internlm2_prepare_inputs_sees_none_for_empty_cache() -> None:
    """Wrapper must pass None into the original InternLM2 prepare_inputs."""

    class DummyLM:
        def __init__(self) -> None:
            self.seen: object = "unset"

        def prepare_inputs_for_generation(
            self,
            input_ids: object,
            past_key_values: object = None,
            **kwargs: object,
        ) -> dict[str, object]:
            self.seen = past_key_values
            return {"past_key_values": past_key_values}

    class EmptyCache:
        def get_seq_length(self) -> int:
            return 0

    dummy = DummyLM()
    fg._patch_internlm2_prepare_inputs(dummy)
    dummy.prepare_inputs_for_generation("ids", past_key_values=EmptyCache())
    assert dummy.seen is None
    first = dummy.prepare_inputs_for_generation
    fg._patch_internlm2_prepare_inputs(dummy)
    assert dummy.prepare_inputs_for_generation is first
