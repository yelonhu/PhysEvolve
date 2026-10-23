"""HFBackend tests.

Validation and processor-only tokenization always run. Full generation
loads the local Qwen3-VL-8B-Instruct weights and is skipped by default::

    PHYSEVOLVE_FULL_SMOKE=1 conda run -n physevolve \
        pytest physevolve/tests/unit/test_hf_backend.py -v
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from physevolve.inference.hf_backend import (
    DEFAULT_MODEL,
    HFBackend,
    _restore_assistant_prefill,
    prepare_hf_multimodal_inputs,
    presampled_video_metadata,
    split_presampled_video,
)
from physevolve.integrations.video_bridge import VideoBundle


_OPT_IN = os.environ.get("PHYSEVOLVE_FULL_SMOKE") == "1"
_PROCESSOR_DIR = Path(DEFAULT_MODEL)
_HAS_LOCAL_PROCESSOR = (
    (_PROCESSOR_DIR / "preprocessor_config.json").is_file()
    and (_PROCESSOR_DIR / "video_preprocessor_config.json").is_file()
    and (_PROCESSOR_DIR / "tokenizer.json").is_file()
)


# ─── Validation-only tests (always run, no model load) ───────────────


def test_video_validation_rejects_non_ndarray() -> None:
    with pytest.raises(TypeError):
        HFBackend._validate_video([[1, 2, 3]])  # type: ignore[arg-type]


def test_video_validation_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        HFBackend._validate_video(np.zeros((3, 4, 5), dtype=np.uint8))


def test_video_validation_rejects_wrong_dtype() -> None:
    arr = np.zeros((4, 32, 32, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        HFBackend._validate_video(arr)


def test_video_validation_happy_path() -> None:
    arr = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    # Must not raise.
    HFBackend._validate_video(arr)


def test_video_validation_accepts_tuple_and_bundle() -> None:
    arr = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    HFBackend._validate_video((arr, {"fps": 2.0, "do_sample_frames": False}))
    HFBackend._validate_video(VideoBundle(frames=arr, metadata={"fps": 2.0}))


def test_presampled_metadata_omits_do_sample_frames() -> None:
    frames = np.zeros((10, 8, 8, 3), dtype=np.uint8)
    metadata = presampled_video_metadata(frames)
    assert "do_sample_frames" not in metadata
    assert metadata["total_num_frames"] == 10
    assert metadata["fps"] == 2.0
    assert metadata["frames_indices"] == list(range(10))
    assert metadata["width"] == 8
    assert metadata["height"] == 8


def test_split_presampled_video_keeps_fps_and_drops_sample_flag() -> None:
    frames = np.full((4, 8, 8, 3), 40, dtype=np.uint8)
    bundle = VideoBundle(
        frames=frames,
        metadata={
            "do_sample_frames": True,
            "fps": 8.0,
            "total_num_frames": 4,
            "width": 8,
            "height": 8,
            "duration": 0.5,
            "frames_indices": [0, 1, 2, 3],
        },
    )
    out_frames, metadata = split_presampled_video(bundle)
    assert out_frames is frames
    assert "do_sample_frames" not in metadata
    assert metadata["fps"] == 8.0
    _, tuple_meta = split_presampled_video(
        (frames, {"fps": 2.0, "do_sample_frames": False})
    )
    assert "do_sample_frames" not in tuple_meta
    assert tuple_meta["fps"] == 2.0


def test_prepare_passes_do_sample_frames_false_to_processor() -> None:
    frames = np.full((6, 8, 8, 3), 40, dtype=np.uint8)
    processor = _RecordingProcessor()
    content = [
        {
            "type": "video",
            "video": (frames, {"fps": 2.0, "do_sample_frames": True}),
        },
        {"type": "text", "text": "Finish by Yes or No"},
    ]
    prepare_hf_multimodal_inputs(processor, content)
    assert processor.template_calls[0]["tokenize"] is False
    roles = [msg["role"] for msg in processor.template_calls[0]["messages"]]
    assert roles == ["user"]
    call = processor.processor_calls[0]
    assert call["do_sample_frames"] is False
    assert len(call["video_metadata"]) == 1
    assert "do_sample_frames" not in call["video_metadata"][0]
    assert call["videos"][0].shape == (6, 8, 8, 3)


def test_prepare_messages_keeps_system_and_fps() -> None:
    frames = np.full((6, 8, 8, 3), 40, dtype=np.uint8)
    processor = _RecordingProcessor()
    prompt1 = "Finish your answer by Yes or No"
    messages = [
        {"role": "system", "content": prompt1},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "These are the frames from the video."},
                {
                    "type": "video",
                    "video": (frames, {"fps": 1.0, "do_sample_frames": False}),
                },
            ],
        },
    ]
    prepare_hf_multimodal_inputs(processor, messages=messages)
    template_messages = processor.template_calls[0]["messages"]
    assert template_messages[0]["role"] == "system"
    assert template_messages[0]["content"] == prompt1
    user_body = template_messages[1]["content"]
    assert user_body[0]["type"] == "text"
    assert prompt1 not in user_body[0]["text"]
    call = processor.processor_calls[0]
    assert call["do_sample_frames"] is False
    assert call["video_metadata"][0]["fps"] == 1.0
    assert "do_sample_frames" not in call["video_metadata"][0]


def test_generate_requires_load() -> None:
    b = HFBackend()  # no load()
    with pytest.raises(RuntimeError):
        b.generate("hi")


def test_default_model_is_instruct_snapshot() -> None:
    assert DEFAULT_MODEL.endswith("Qwen3-VL-8B-Instruct")


def test_decode_generated_tokens_preserves_structural_special_tokens() -> None:
    processor = _FakeProcessor("<think>reason</think><propose>code</propose>")

    text = HFBackend._decode_generated_tokens(processor, token_ids=[1, 2, 3])

    assert text.startswith("<think>")
    assert processor.skip_special_tokens is False
    assert processor.clean_up_tokenization_spaces is False


def test_restore_assistant_prefill_adds_trimmed_opening_think() -> None:
    assistant_text, metadata = _restore_assistant_prefill(
        prompt_text="<|im_start|>assistant\n<think>",
        generated_text="reasoning</think><propose>code</propose>",
    )

    assert assistant_text.startswith("<think>")
    assert "reasoning</think>" in assistant_text
    assert metadata["assistant_prefill_restored"] is True


def test_restore_assistant_prefill_does_not_duplicate_existing_think() -> None:
    assistant_text, metadata = _restore_assistant_prefill(
        prompt_text="<|im_start|>assistant\n<think>",
        generated_text="<think>reasoning</think>",
    )

    assert assistant_text == "<think>reasoning</think>"
    assert metadata["assistant_prefill_restored"] is False


# ─── Processor-only (no model weights) ───────────────────────────────


@pytest.mark.skipif(
    not _HAS_LOCAL_PROCESSOR,
    reason="local Qwen3-VL-8B-Instruct processor files are missing",
)
def test_presampled_10x512_keeps_all_frames_and_token_count() -> None:
    pytest.importorskip("torch")
    from transformers import AutoProcessor

    frames = np.full((10, 512, 512, 3), 80, dtype=np.uint8)
    frames[:, :256, :, 0] = 180
    assert float(frames.mean()) > 1.0

    processor = AutoProcessor.from_pretrained(
        DEFAULT_MODEL,
        trust_remote_code=True,
        local_files_only=True,
    )
    content = [
        {"type": "video", "video": frames},
        {
            "type": "text",
            "text": (
                "These are the frames from the video. The video you're seeing "
                "was generated by a simulator. Finish your answer by Yes or No"
            ),
        },
    ]
    captured: list[str] = []
    handler = _ListLogHandler(captured)
    tf_logger = logging.getLogger("transformers")
    prev_level = tf_logger.level
    tf_logger.addHandler(handler)
    tf_logger.setLevel(logging.DEBUG)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            inputs, _prompt_text = prepare_hf_multimodal_inputs(processor, content)
    finally:
        tf_logger.removeHandler(handler)
        tf_logger.setLevel(prev_level)

    messages = [str(w.message) for w in caught] + captured
    assert not any("fps=24" in msg for msg in messages)
    n_tokens = int(inputs["input_ids"].shape[-1])
    assert 1300 <= n_tokens <= 1550, f"unexpected prompt_token_count={n_tokens}"
    grid = inputs["video_grid_thw"]
    temporal = int(grid[0][0]) if getattr(grid, "ndim", 1) > 1 else int(grid[0])
    assert temporal == 5




# ─── Full-load smoke (opt-in) ────────────────────────────────────────


@pytest.mark.skipif(
    not _OPT_IN,
    reason="PHYSEVOLVE_FULL_SMOKE=1 required (downloads ~16 GB)",
)
def test_text_only_generation() -> None:
    pytest.importorskip("torch")
    with HFBackend() as backend:
        text = backend.generate(
            "What is 2+2? Reply with a single digit.",
            max_new_tokens=16,
            temperature=0.0,
        )
    assert isinstance(text, str) and len(text.strip()) > 0


@pytest.mark.skipif(
    not _OPT_IN,
    reason="PHYSEVOLVE_FULL_SMOKE=1 required (downloads ~16 GB)",
)
def test_video_plus_text_generation() -> None:
    pytest.importorskip("torch")
    # Synthetic 4-frame 64x64 video — cheap and stays under Qwen3-VL
    # patcher's min-resolution limit.
    video = np.full((4, 64, 64, 3), 127, dtype=np.uint8)
    with HFBackend() as backend:
        text = backend.generate(
            "Describe the colour palette of this video in one word.",
            video=video,
            max_new_tokens=32,
            temperature=0.0,
        )
    assert isinstance(text, str) and len(text.strip()) > 0


class _FakeIds:
    def __init__(self, n: int) -> None:
        self.shape = (1, n)
        self._n = n

    def __getitem__(self, item: Any) -> list[int]:
        _ = item
        return list(range(self._n))


class _RecordingProcessor:
    def __init__(self) -> None:
        self.template_calls: list[dict[str, Any]] = []
        self.processor_calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
        self.template_calls.append({"messages": messages, **kwargs})
        if kwargs.get("tokenize"):
            ids = _FakeIds(3)
            return {"input_ids": ids, "attention_mask": ids}
        return "PROMPT"

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.processor_calls.append(kwargs)
        return {"input_ids": _FakeIds(12)}

    def batch_decode(
        self,
        token_ids: Any,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]:
        _ = token_ids, skip_special_tokens, clean_up_tokenization_spaces
        return ["decoded"]


def test_legacy_paper_video_packing_uses_processor_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHYSEVOLVE_LEGACY_VIDEO_PACKING", "1")
    frames = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    processor = _RecordingProcessor()
    prepare_hf_multimodal_inputs(
        processor,
        [{"type": "video", "video": frames}, {"type": "text", "text": "What happens?"}],
    )
    call = processor.template_calls[0]
    assert call["tokenize"] is True
    assert call["messages"][0]["content"][0]["video"] is frames
    assert processor.processor_calls == []


class _ListLogHandler(logging.Handler):
    def __init__(self, bucket: list[str]) -> None:
        super().__init__()
        self.bucket = bucket

    def emit(self, record: logging.LogRecord) -> None:
        self.bucket.append(record.getMessage())


class _FakeProcessor:
    def __init__(self, decoded: str) -> None:
        self.decoded = decoded
        self.skip_special_tokens = None
        self.clean_up_tokenization_spaces = None

    def batch_decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]:
        _ = token_ids
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        return [self.decoded]
