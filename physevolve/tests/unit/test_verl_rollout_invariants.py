"""Unit tests for PhysEvolve veRL rollout invariants."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from physevolve.integrations.video_bridge import VideoBundle
from physevolve.training.rollout_invariants import (
    ROLLOUT_NON_TENSOR_KEYS,
    validate_multimodal_token_alignment,
    validate_update_batch,
)
from physevolve.training.online_step_builder import GenerationSpec, _concat_rollouts
from physevolve.training.verl_agent_loop import (
    _sampling_params_for_row,
    _videos_from_bundle,
    _vllm_videos_from_bundle,
    PhysEvolveSingleTurnAgentLoop,
)
from physevolve.training.physevolve_ray_trainer import _PhysEvolveTrainerRolloutBackend


def _make_single_spec(
    *,
    prompt: str = "hello",
    video: object | None = None,
    stop_sequences: tuple[str, ...] = (),
    max_tokens: int = 8,
    rollout_metadata: dict | None = None,
) -> GenerationSpec:
    return GenerationSpec(
        prompt=prompt,
        video=video,
        max_new_tokens=int(max_tokens),
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        stop_sequences=tuple(stop_sequences),
        rollout_metadata=dict(rollout_metadata or {}),
    )


def test_rollout_dataproto_has_stable_non_tensor_schema() -> None:
    pytest.importorskip("verl")
    backend = _PhysEvolveTrainerRolloutBackend(_fake_trainer())

    data = backend._make_dataproto_batch(  # noqa: SLF001
        [
            _make_single_spec(
                stop_sequences=("</answer>",),
                max_tokens=32,
                rollout_metadata={
                    "uid": "group-a",
                    "group_id": "group-a",
                    "index": 7,
                    "role": "solver",
                    "problem_type": "pred_forward",
                    "scene_id": "scene-a",
                },
            )
        ]
    )

    assert tuple(data.non_tensor_batch.keys()) == ROLLOUT_NON_TENSOR_KEYS
    assert data.non_tensor_batch["uid"].tolist() == ["group-a"]
    assert data.non_tensor_batch["index"].tolist() == [7]
    assert data.non_tensor_batch["data_source"].tolist() == ["physevolve"]
    assert data.non_tensor_batch["reward_model"].tolist() == [
        {"style": "rule", "ground_truth": ""}
    ]
    assert data.non_tensor_batch["nz_video_bundle"].tolist() == [None]


def test_rollout_generation_pads_to_agent_worker_count() -> None:
    pytest.importorskip("verl")
    trainer = _fake_trainer(agent_workers=4)
    backend = _PhysEvolveTrainerRolloutBackend(trainer)
    data = backend._make_dataproto_batch(  # noqa: SLF001
        [_make_single_spec(rollout_metadata={"uid": "u", "group_id": "u"})]
    )

    output = backend._generate_padded(data)  # noqa: SLF001

    assert trainer.async_rollout_manager.seen_batch_size == 4
    assert len(output) == 1


def test_rollout_padding_unpads_by_worker_padding_not_group_size() -> None:
    pytest.importorskip("verl")
    trainer = _fake_trainer(agent_workers=8, rollout_n=4)
    backend = _PhysEvolveTrainerRolloutBackend(trainer)
    data = backend._make_dataproto_batch(  # noqa: SLF001
        [_make_single_spec(rollout_metadata={"uid": "u", "group_id": "u"})]
    )

    output = backend._generate_padded(data)  # noqa: SLF001

    assert trainer.async_rollout_manager.seen_batch_size == 8
    assert len(output) == 1


def test_rollout_generation_restores_physevolve_metadata() -> None:
    pytest.importorskip("verl")
    trainer = _fake_trainer(agent_workers=2, strip_non_tensor=True)
    backend = _PhysEvolveTrainerRolloutBackend(trainer)
    data = backend._make_dataproto_batch(  # noqa: SLF001
        [
            _make_single_spec(
                rollout_metadata={"uid": "u", "group_id": "u", "role": "proposer"},
            )
        ]
    )

    output = backend._generate_padded(data)  # noqa: SLF001

    assert output.non_tensor_batch["uid"].tolist() == ["u"]
    assert output.non_tensor_batch["role"].tolist() == ["proposer"]


def test_rollout_dataproto_batch_aggregates_rows() -> None:
    """Multi-row batch keeps the same schema and packs all per-row fields."""

    pytest.importorskip("verl")
    backend = _PhysEvolveTrainerRolloutBackend(_fake_trainer())

    specs = [
        _make_single_spec(
            prompt="p0",
            stop_sequences=("</propose>",),
            max_tokens=16,
            rollout_metadata={"uid": "a", "group_id": "a", "index": 0, "role": "proposer"},
        ),
        _make_single_spec(
            prompt="p1",
            stop_sequences=("</answer>",),
            max_tokens=32,
            rollout_metadata={
                "uid": "b",
                "group_id": "b",
                "index": 1,
                "role": "solver",
                "scene_id": "scene-1",
            },
        ),
    ]

    data = backend._make_dataproto_batch(specs)  # noqa: SLF001

    assert tuple(data.non_tensor_batch.keys()) == ROLLOUT_NON_TENSOR_KEYS
    assert data.batch.batch_size[0] == 2
    assert data.non_tensor_batch["uid"].tolist() == ["a", "b"]
    assert data.non_tensor_batch["role"].tolist() == ["proposer", "solver"]
    assert data.non_tensor_batch["raw_prompt"].tolist() == ["p0", "p1"]
    assert data.non_tensor_batch["max_tokens"].tolist() == [16, 32]
    assert data.non_tensor_batch["stop_sequences"].tolist() == [
        ("</propose>",),
        ("</answer>",),
    ]


def test_sampling_params_include_stop_and_max_tokens() -> None:
    params = _sampling_params_for_row(
        {"temperature": 0.6},
        max_tokens=np.int64(17),
        stop_sequences=np.array(["</answer>"], dtype=object),
    )

    assert params["max_tokens"] == 17
    assert params["stop"] == ["</answer>"]
    assert params["include_stop_str_in_output"] is True


def test_agent_loop_filters_vllm_only_video_metadata_for_hf_processor() -> None:
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    bundle = VideoBundle(
        frames=frames,
        metadata={
            "total_num_frames": 2,
            "fps": 1.0,
            "width": 8,
            "height": 8,
            "duration": 2.0,
            "frames_indices": [0, 1],
            "do_sample_frames": False,
        },
    )

    videos = _videos_from_bundle(bundle)

    assert videos is not None
    assert videos[0][0] is frames
    assert "do_sample_frames" not in videos[0][1]
    assert videos[0][1]["frames_indices"] == [0, 1]

    vllm_videos = _vllm_videos_from_bundle(bundle)
    assert vllm_videos is not None
    assert vllm_videos[0][1]["do_sample_frames"] is False


def test_agent_loop_uses_expanded_training_prompt_for_video() -> None:
    pytest.importorskip("verl")
    import asyncio

    loop = PhysEvolveSingleTurnAgentLoop.__new__(PhysEvolveSingleTurnAgentLoop)
    loop.loop = asyncio.new_event_loop()
    loop.processor = _FakeProcessor(video_token_id=42, expanded_video_tokens=4)
    loop.apply_chat_template_kwargs = {}

    async def _compact_prompt(messages, *, videos):  # noqa: ANN001
        return [1, 42, 2]

    loop._apply_generation_chat_template = _compact_prompt  # noqa: SLF001
    try:
        prompt_ids, diagnostics = loop.loop.run_until_complete(
            loop._apply_training_chat_template(  # noqa: SLF001
                [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "solve"}]}],
                videos=[(np.zeros((1, 8, 8, 3), dtype=np.uint8), {"fps": 1.0})],
            )
        )
    finally:
        loop.loop.close()

    assert prompt_ids == [1, 42, 42, 42, 42, 2]
    assert diagnostics["physevolve/generation_video_tokens"] == 1
    assert diagnostics["physevolve/training_video_tokens"] == 4
    assert diagnostics["physevolve/video_feature_tokens"] == 4


def test_multimodal_alignment_rejects_compact_video_tokens() -> None:
    pytest.importorskip("verl")
    from tensordict import TensorDict
    from verl import DataProto

    batch = TensorDict(
        {"input_ids": torch.tensor([[1, 42, 2], [1, 42, 2]], dtype=torch.long)},
        batch_size=[2],
    )
    data = DataProto(
        batch=batch,
        non_tensor_batch={
            "multi_modal_inputs": np.array(
                [
                    {"video_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long)},
                    {"video_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long)},
                ],
                dtype=object,
            )
        },
    )

    with pytest.raises(ValueError, match="compact vLLM video tokens leaked"):
        validate_multimodal_token_alignment(data, _FakeProcessor(video_token_id=42, expanded_video_tokens=4))


def test_update_batch_requires_exact_grpo_uid_groups() -> None:
    pytest.importorskip("verl")
    from tensordict import TensorDict
    from verl import DataProto

    batch = TensorDict(
        {
            "responses": torch.ones((3, 4), dtype=torch.long),
            "attention_mask": torch.ones((3, 8), dtype=torch.long),
        },
        batch_size=[3],
    )
    data = DataProto(
        batch=batch,
        non_tensor_batch={"uid": np.array(["a", "a", "b"], dtype=object)},
    )
    scores = torch.zeros((3, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="exactly group_size"):
        validate_update_batch(data, scores, group_size=2)


def test_update_batch_accepts_grouped_scores() -> None:
    pytest.importorskip("verl")
    from tensordict import TensorDict
    from verl import DataProto

    batch = TensorDict(
        {
            "responses": torch.ones((4, 4), dtype=torch.long),
            "attention_mask": torch.ones((4, 8), dtype=torch.long),
        },
        batch_size=[4],
    )
    data = DataProto(
        batch=batch,
        non_tensor_batch={"uid": np.array(["a", "a", "b", "b"], dtype=object)},
    )
    validate_update_batch(data, torch.zeros((4, 4), dtype=torch.float32), group_size=2)


def test_trainer_rollout_backend_rejects_video_without_metadata() -> None:
    pytest.importorskip("verl")
    backend = _PhysEvolveTrainerRolloutBackend(_fake_trainer())
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="missing required keys"):
        backend._make_dataproto_batch(  # noqa: SLF001
            [
                _make_single_spec(
                    video=VideoBundle(frames=frames, metadata={}),
                    rollout_metadata={"uid": "u"},
                )
            ]
        )


def test_concat_rollouts_drops_per_call_timing_metadata() -> None:
    pytest.importorskip("verl")
    from tensordict import TensorDict
    from verl import DataProto

    rollouts = []
    for idx in range(2):
        rollouts.append(
            DataProto(
                batch=TensorDict({"responses": torch.ones((1, 2), dtype=torch.long)}, batch_size=[1]),
                non_tensor_batch={"uid": np.array([f"u{idx}"], dtype=object)},
                meta_info={"timing": {"elapsed": idx}},
            )
        )

    merged = _concat_rollouts(rollouts)

    assert len(merged) == 2
    assert "timing" not in merged.meta_info


class _FakeRolloutManager:
    def __init__(self, *, strip_non_tensor: bool = False) -> None:
        self.strip_non_tensor = strip_non_tensor
        self.seen_batch_size: int | None = None

    def generate_sequences(self, data):
        self.seen_batch_size = len(data)
        if self.strip_non_tensor:
            from verl import DataProto

            return DataProto(batch=data.batch, non_tensor_batch={}, meta_info=data.meta_info)
        return data


class _FakeVideoProcessor:
    merge_size = 2


class _FakeProcessor:
    video_processor = _FakeVideoProcessor()

    def __init__(self, *, video_token_id: int, expanded_video_tokens: int) -> None:
        self.video_token_id = video_token_id
        self.expanded_video_tokens = expanded_video_tokens

    def apply_chat_template(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return "<|video_pad|> solve"

    def __call__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        ids = [1] + [self.video_token_id] * self.expanded_video_tokens + [2]
        grid_hw = 2 * self.video_processor.merge_size
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "video_grid_thw": torch.tensor([[1, grid_hw, grid_hw]], dtype=torch.long),
        }


def _fake_trainer(
    agent_workers: int = 1,
    rollout_n: int = 1,
    *,
    strip_non_tensor: bool = False,
):
    return SimpleNamespace(
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(
                    n=rollout_n,
                    agent=SimpleNamespace(num_workers=agent_workers),
                )
            )
        ),
        async_rollout_manager=_FakeRolloutManager(strip_non_tensor=strip_non_tensor),
        tokenizer=SimpleNamespace(
            eos_token_id=1,
            pad_token_id=0,
            decode=lambda ids, **kwargs: "decoded",
        ),
    )
