"""veRL agent-loop bridge for PhysEvolve Solver video requests."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import numpy as np

from physevolve.integrations.video_bridge import VLLM_VIDEO_METADATA_KEYS, VideoBundle


try:
    from verl.experimental.agent_loop.agent_loop import (  # type: ignore
        AgentLoopBase,
        AgentLoopOutput,
        register,
    )
    from verl.utils.chat_template import apply_chat_template  # type: ignore
    from verl.utils.profiler import simple_timer  # type: ignore
    from verl.utils.tokenizer import normalize_token_ids  # type: ignore

    _VERL_AGENT_LOOP_AVAILABLE = True
except Exception:  # pragma: no cover - import-only fallback.
    AgentLoopBase = object  # type: ignore[misc, assignment]
    AgentLoopOutput = object  # type: ignore[misc, assignment]
    _VERL_AGENT_LOOP_AVAILABLE = False

    def register(name: str):  # type: ignore[no-untyped-def]
        def _decorator(cls):  # type: ignore[no-untyped-def]
            return cls

        return _decorator

    class simple_timer:  # type: ignore[no-redef]
        def __init__(self, name: str, metrics: dict[str, Any]) -> None:
            self.name = name
            self.metrics = metrics

        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc_info: object) -> None:
            self.metrics.setdefault(self.name, 0.0)

    apply_chat_template = None  # type: ignore[assignment]
    normalize_token_ids = None  # type: ignore[assignment]


@register("physevolve_single_turn")
class PhysEvolveSingleTurnAgentLoop(AgentLoopBase):  # type: ignore[misc, valid-type]
    """Single-turn veRL agent loop that forwards PhysEvolve video bundles."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _VERL_AGENT_LOOP_AVAILABLE:
            raise RuntimeError("PhysEvolveSingleTurnAgentLoop requires veRL to be importable")
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> Any:
        """Generate one response with optional `nz_video_bundle` multimodal data."""

        raw_prompt = kwargs["raw_prompt"]
        bundle = kwargs.get("nz_video_bundle")
        template_videos = _videos_from_bundle(bundle)
        generation_videos = _vllm_videos_from_bundle(bundle)
        messages = _messages_from_raw_prompt(raw_prompt, has_video=bool(template_videos))

        generation_prompt_ids = await self._apply_generation_chat_template(
            messages,
            videos=template_videos,
        )
        training_prompt_ids, prompt_diagnostics = await self._apply_training_chat_template(
            messages,
            videos=template_videos,
        )

        metrics: dict[str, Any] = {}
        sampling_params = _sampling_params_for_row(
            sampling_params,
            max_tokens=kwargs.get("max_tokens"),
            stop_sequences=kwargs.get("stop_sequences"),
        )
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=generation_prompt_ids,
                sampling_params=sampling_params,
                video_data=generation_videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = (
                output.num_preempted if output.num_preempted is not None else -1
            )
        metrics.update(prompt_diagnostics)

        response_mask = [1] * len(output.token_ids)
        routed_experts = output.routed_experts
        if routed_experts is not None and len(training_prompt_ids) != len(generation_prompt_ids):
            routed_experts = None

        agent_output = AgentLoopOutput(
            prompt_ids=training_prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length]
            if output.log_probs
            else None,
            routed_experts=(
                routed_experts[: len(generation_prompt_ids) + self.response_length]
                if routed_experts is not None
                else None
            ),
            # veRL postprocess replays multi_modal_data through the HF
            # processor, so keep vLLM-only metadata out of this copy.
            multi_modal_data={"videos": template_videos} if template_videos else {},
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
        agent_output.extra_fields.update(
            {
                "turn_scores": [],
                "tool_rewards": [],
                "physevolve_prompt_diagnostics": prompt_diagnostics,
            }
        )
        return agent_output

    async def _apply_generation_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        videos: list[tuple[Any, dict[str, Any]]] | None,
    ) -> list[int]:
        """Tokenize Qwen-VL video prompts in the shape vLLM expects.

        veRL's default helper calls the HF processor with ``videos=...`` and
        returns already-expanded video placeholder tokens. vLLM's
        ``TokensPrompt`` path expects the compact chat-template token stream
        instead, then expands placeholders from ``multi_modal_data`` inside
        vLLM. Using the expanded token stream causes Qwen-VL M-RoPE to see
        more video placeholders than there are ``video_grid_thw`` entries.
        """

        if not videos:
            return await self.apply_chat_template(messages, videos=videos)
        if apply_chat_template is None or normalize_token_ids is None:
            return await self.apply_chat_template(messages, videos=videos)

        raw_prompt = await self.loop.run_in_executor(
            None,
            lambda: apply_chat_template(
                self.processor,
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            ),
        )
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(raw_prompt, add_special_tokens=False),
        )
        return normalize_token_ids(prompt_ids)

    async def _apply_training_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        videos: list[tuple[Any, dict[str, Any]]] | None,
    ) -> tuple[list[int], dict[str, Any]]:
        """Return processor-expanded prompt ids for FSDP logprob/update.

        vLLM receives compact prompt ids through
        :meth:`_apply_generation_chat_template`, but veRL's actor forward sees
        the returned ``prompt_ids`` together with ``multi_modal_inputs``. For
        Qwen-VL those ids must contain one video placeholder token per visual
        feature token; otherwise Transformers raises a video token/feature
        mismatch during old-logprob recomputation.
        """

        if not videos:
            prompt_ids = await self._apply_generation_chat_template(messages, videos=videos)
            return prompt_ids, {
                "physevolve/generation_prompt_len": len(prompt_ids),
                "physevolve/training_prompt_len": len(prompt_ids),
                "physevolve/generation_video_tokens": 0,
                "physevolve/training_video_tokens": 0,
                "physevolve/video_feature_tokens": 0,
            }
        if apply_chat_template is None or normalize_token_ids is None:
            prompt_ids = await self.apply_chat_template(messages, videos=videos)
            video_token_count = _count_token(prompt_ids, _processor_attr(self.processor, "video_token_id"))
            return prompt_ids, {
                "physevolve/generation_prompt_len": len(prompt_ids),
                "physevolve/training_prompt_len": len(prompt_ids),
                "physevolve/generation_video_tokens": video_token_count,
                "physevolve/training_video_tokens": video_token_count,
                "physevolve/video_feature_tokens": video_token_count,
            }

        raw_prompt = await self.loop.run_in_executor(
            None,
            lambda: apply_chat_template(
                self.processor,
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            ),
        )
        video_arrays, video_metadata = _split_videos(videos)
        model_inputs = await self.loop.run_in_executor(
            None,
            lambda: self.processor(
                text=[raw_prompt],
                videos=video_arrays,
                video_metadata=video_metadata,
                return_tensors="pt",
                do_sample_frames=False,
            ),
        )
        prompt_ids = normalize_token_ids(model_inputs["input_ids"])
        video_grid_thw = model_inputs.get("video_grid_thw")
        feature_tokens = _feature_tokens_from_grid(
            video_grid_thw,
            merge_size=_video_merge_size(self.processor),
        )
        video_token_id = _processor_attr(self.processor, "video_token_id")
        training_video_tokens = _count_token(prompt_ids, video_token_id)
        generation_prompt_ids = await self._apply_generation_chat_template(messages, videos=videos)
        generation_video_tokens = _count_token(generation_prompt_ids, video_token_id)
        diagnostics = {
            "physevolve/generation_prompt_len": len(generation_prompt_ids),
            "physevolve/training_prompt_len": len(prompt_ids),
            "physevolve/generation_video_tokens": generation_video_tokens,
            "physevolve/training_video_tokens": training_video_tokens,
            "physevolve/video_feature_tokens": feature_tokens,
        }
        if feature_tokens and training_video_tokens != feature_tokens:
            raise ValueError(
                "PhysEvolve Qwen-VL training prompt/video mismatch before PPO: "
                f"video_tokens={training_video_tokens}, feature_tokens={feature_tokens}, "
                f"generation_video_tokens={generation_video_tokens}, "
                f"training_prompt_len={len(prompt_ids)}"
            )
        return prompt_ids, diagnostics


def _messages_from_raw_prompt(raw_prompt: Any, *, has_video: bool = False) -> list[dict[str, Any]]:
    if isinstance(raw_prompt, str):
        if has_video:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": raw_prompt},
                    ],
                }
            ]
        return [{"role": "user", "content": raw_prompt}]
    if isinstance(raw_prompt, list):
        return raw_prompt
    raise TypeError(f"raw_prompt must be str or messages list; got {type(raw_prompt).__name__}")


def _videos_from_bundle(bundle: Any) -> list[tuple[Any, dict[str, Any]]] | None:
    """Return HF processor-compatible videos for prompt tokenization."""

    if bundle is None:
        return None
    if isinstance(bundle, VideoBundle):
        _validate_video_metadata(bundle.metadata)
        return [_processor_video_tuple(bundle.frames, bundle.metadata)]
    if isinstance(bundle, tuple) and len(bundle) == 2:
        frames, metadata = bundle
        _validate_video_metadata(metadata)
        return [_processor_video_tuple(frames, metadata)]
    raise TypeError(
        "nz_video_bundle must be VideoBundle or (frames, metadata); "
        f"got {type(bundle).__name__}"
    )


def _vllm_videos_from_bundle(bundle: Any) -> list[tuple[Any, dict[str, Any]]] | None:
    """Return vLLM-compatible videos for multimodal generation."""

    if bundle is None:
        return None
    if isinstance(bundle, VideoBundle):
        _validate_video_metadata(bundle.metadata)
        return [bundle.as_tuple()]
    if isinstance(bundle, tuple) and len(bundle) == 2:
        frames, metadata = bundle
        _validate_video_metadata(metadata)
        return [(frames, dict(metadata))]
    raise TypeError(
        "nz_video_bundle must be VideoBundle or (frames, metadata); "
        f"got {type(bundle).__name__}"
    )


def _processor_video_tuple(frames: Any, metadata: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Return a Qwen-VL processor-compatible video tuple.

    PhysEvolve stores the stricter vLLM metadata contract internally, including
    ``do_sample_frames``.  In vLLM 0.11's Qwen-VL path the Hugging Face
    processor first converts metadata to ``transformers.VideoMetadata``, whose
    constructor does not accept that vLLM-only flag.  The delivered frames are
    already pre-sliced by PhysEvolve, so dropping the flag preserves behavior while
    allowing the processor to build prompt multimodal placeholders.
    """

    processor_metadata = dict(metadata)
    processor_metadata.pop("do_sample_frames", None)
    return frames, processor_metadata


def _split_videos(
    videos: list[tuple[Any, dict[str, Any]]],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Split processor-compatible video tuples into HF processor kwargs."""

    if not videos:
        return [], []
    video_arrays, video_metadata = zip(*videos, strict=False)
    return list(video_arrays), [dict(metadata) for metadata in video_metadata]


def _feature_tokens_from_grid(video_grid_thw: Any, *, merge_size: int) -> int:
    """Return Qwen-VL placeholder tokens implied by `video_grid_thw`."""

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
    for item in grid:
        if len(item) != 3:
            raise ValueError(f"video_grid_thw rows must have length 3; got {item!r}")
        t, h, w = (int(x) for x in item)
        total += (t * h * w) // denom
    return int(total)


def _video_merge_size(processor: Any) -> int:
    video_processor = getattr(processor, "video_processor", None)
    return int(getattr(video_processor, "merge_size", 2) or 2)


def _processor_attr(processor: Any, name: str) -> int | None:
    value = getattr(processor, name, None)
    if value is None:
        return None
    return int(value)


def _count_token(token_ids: list[int], token_id: int | None) -> int:
    if token_id is None:
        return 0
    return sum(1 for token in token_ids if int(token) == int(token_id))


def _sampling_params_for_row(
    sampling_params: dict[str, Any],
    *,
    max_tokens: Any,
    stop_sequences: Any,
) -> dict[str, Any]:
    out = dict(sampling_params)
    try:
        max_tokens_int = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens_int = 0
    if max_tokens_int > 0:
        out["max_tokens"] = max_tokens_int

    stops = _as_stop_list(stop_sequences)
    if stops:
        out["stop"] = stops
        # Keep the closing tag in the decoded response so PhysEvolve parsers see
        # the same structural marker that requested the stop.
        out["include_stop_str_in_output"] = True
    return out


def _as_stop_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):  # type: ignore[name-defined]
        value = value.tolist()
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return []


def _validate_video_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise TypeError(f"video metadata must be a dict; got {type(metadata).__name__}")
    missing = [key for key in VLLM_VIDEO_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"video metadata missing required keys: {missing}")


__all__ = [
    "PhysEvolveSingleTurnAgentLoop",
    "_messages_from_raw_prompt",
    "_videos_from_bundle",
]
