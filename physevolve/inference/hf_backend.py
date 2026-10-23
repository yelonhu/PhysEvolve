"""HuggingFace inference backend for released checkpoint evaluation.

Training rollouts use veRL/vLLM. This module lazily loads Torch and
Transformers so CPU-only protocol checks can import it. Video frames are
presampled by default; the paper's CausalPhys video packing is documented in
``docs/evaluation.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from physevolve.integrations.video_bridge import VideoBundle

if TYPE_CHECKING:  # pragma: no cover — type hints only.
    import torch
    from transformers import PreTrainedModel  # noqa: F401
    from transformers import ProcessorMixin  # noqa: F401


# ─── Model path resolution ───────────────────────────────────────────


DEFAULT_MODEL: str = "Qwen/Qwen3-VL-8B-Instruct"
_PRESAMPLED_VIDEO_FPS = 2.0
_HF_VIDEO_METADATA_KEYS = (
    "total_num_frames",
    "fps",
    "width",
    "height",
    "duration",
    "frames_indices",
)


def _default_device_map() -> str:
    """Prefer a single CUDA device to avoid multi-GPU placement surprises."""

    return "auto"


# ─── HFBackend ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Decoded assistant response plus lightweight diagnostic metadata.

    ``text`` is the parser-facing response: generated tokens with any
    chat-template assistant prefill (e.g. Qwen Thinking's ``<think>``)
    restored if it was trimmed with the prompt.
    ``generated_text`` is the raw decoded token range after ``input_len``,
    without prefill restoration.
    """

    text: str
    metadata: dict[str, Any]
    generated_text: str = ""
    prompt_text: str = ""
    prompt_tail: str = ""


class HFBackend:
    """Thin wrapper around ``AutoModelForImageTextToText`` for evaluation.

    Attributes:
        model_name_or_path: HuggingFace hub tag or local directory.
        torch_dtype: Parameter dtype. Defaults to ``torch.bfloat16``.
        device_map: ``accelerate`` device-map strategy. ``"auto"`` fans
            the model across visible devices; a specific ``"cuda:0"``
            is also accepted for single-GPU probing.
        trust_remote_code: Forwarded to ``from_pretrained``. Some Qwen-VL
            tags require this.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        *,
        torch_dtype: Any = None,
        device_map: str | dict | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.torch_dtype = torch_dtype
        self.device_map = device_map or _default_device_map()
        self.trust_remote_code = trust_remote_code
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    # ── Context-manager semantics ─────────────────────────────────────

    def __enter__(self) -> "HFBackend":
        self.load()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ── Load / unload ────────────────────────────────────────────────

    def load(self) -> None:
        """Download (if needed) and load model + processor into memory."""

        import torch  # local import — keeps module torch-free at top level.
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        dtype = self.torch_dtype if self.torch_dtype is not None else torch.bfloat16
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name_or_path,
            dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        self._model.eval()

    def close(self) -> None:
        """Release model + processor + cached CUDA memory."""

        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if self._torch is not None:
            try:
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 — best-effort.
                pass

    # ── Inference ────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        video: Optional[np.ndarray | VideoBundle | tuple[np.ndarray, dict[str, Any]]] = None,
        *,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        rollout_metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Run a single forward pass and return the decoded completion.

        Args:
            prompt: Text prompt as it should be shown to the Solver/Proposer.
                Callers are responsible for adding any chat-template
                wrapping (most Qwen3-VL prompts should already be in
                chat-template form, so the processor's
                ``apply_chat_template`` is called internally when needed).
            video: Optional ``(n_frames, H, W, 3) uint8`` numpy array.
                ``None`` sends a text-only prompt.
            max_new_tokens: Generation cap.
            temperature: Softmax temperature. ``0`` triggers greedy.
            top_p: Nucleus sampling top-p.
            top_k: Optional top-k sampling cutoff.

        Returns:
            The decoded generation string (prompt tokens stripped).

        Raises:
            RuntimeError: if :meth:`load` has not been called yet.
        """

        return self.generate_with_metadata(
            prompt,
            video=video,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            rollout_metadata=rollout_metadata,
        ).text

    def generate_with_metadata(
        self,
        prompt: str,
        video: Optional[np.ndarray | VideoBundle | tuple[np.ndarray, dict[str, Any]]] = None,
        *,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        rollout_metadata: Optional[dict[str, Any]] = None,
    ) -> GenerationResult:
        """Run generation and return decoded text plus diagnostics."""

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if video is not None:
            self._validate_video(video)
            content.insert(0, {"type": "video", "video": video})

        return self.generate_multimodal_with_metadata(
            content,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
        )

    def generate_multimodal(
        self,
        content: list[dict[str, Any]],
        *,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> str:
        """Run generation over a pre-built chat-template content list.

        The content list must be an ordered sequence of entries whose
        ``type`` field is one of ``"text"``, ``"image"``, or ``"video"``,
        matching the HuggingFace multimodal chat-template schema used by
        Qwen3-VL. Text entries carry a ``"text"`` payload; image entries a
        ``"image"`` payload (``np.ndarray`` ``(H, W, 3) uint8`` or a
        :class:`PIL.Image.Image`); video entries a ``"video"`` payload
        (``np.ndarray`` ``(n_frames, H, W, 3) uint8``, optional
        ``(frames, metadata)`` tuple, or :class:`VideoBundle`). Pre-sampled
        frames are not resampled by the Qwen3-VL processor.

        Args:
            content: Pre-assembled content list. Callers are responsible
                for ordering it to match any ``<video>``/``<image>``
                placeholders in the text.
            max_new_tokens: Generation cap. Defaults to ``4096`` for
                Thinking-model probing. Benchmark eval callers can pass
                a smaller value when the task demands it.
            temperature: Sampling temperature. ``0`` triggers greedy
                decoding — the default, matching benchmark eval convention.
            top_p: Nucleus cutoff (only applied when sampling).
            top_k: Optional top-k cutoff (only applied when sampling).

        Returns:
            Decoded text with prompt tokens stripped.

        Raises:
            RuntimeError: if :meth:`load` has not been called yet.
            ValueError: if a ``video`` or ``image`` entry fails validation.
        """

        return self.generate_multimodal_with_metadata(
            content,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
        ).text

    def generate_multimodal_batch(
        self,
        contents: list[list[dict[str, Any]]],
        *,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        assistant_prefill: Optional[str] = None,
    ) -> list[GenerationResult]:
        """Batched multimodal generation over a list of content lists.

        Each element of ``contents`` is the same kind of pre-built
        content list accepted by :meth:`generate_multimodal_with_metadata`.
        Batching uses left-padding on the tokenizer so causal generation
        works correctly across mixed prompt lengths. Per-item stop
        sequences are honoured via ``_truncate_at_stop`` post-hoc; the
        token-level stopping criterion only fires when *every* row in
        the batch has reached its stop, so a small post-truncation step
        recovers the correct per-row text.

        ``assistant_prefill`` (e.g. ``"<answer>"``) is appended to the
        chat template after ``add_generation_prompt`` so the model is
        forced to continue from inside that tag. Used by the
        ``answer_only`` PhysBench mode.

        Returns one :class:`GenerationResult` per input content list,
        with metadata populated identically to the single-item path.
        """

        if self._model is None or self._processor is None:
            raise RuntimeError("HFBackend.load() must be called first")
        if not contents:
            return []

        torch = self._torch
        processor = self._processor
        tokenizer = getattr(processor, "tokenizer", processor)

        # Per-row prompt assembly. We use ``apply_chat_template`` once per
        # row so the multimodal alignment (image/video tokens vs visual
        # features) matches the single-item path byte-for-byte; only the
        # text concatenation across the batch is new.
        per_row_inputs: list[dict[str, Any]] = []
        per_row_prompt_text: list[str] = []
        for content in contents:
            inputs, prompt_text = prepare_hf_multimodal_inputs(processor, content)
            if assistant_prefill:
                # Append the prefill tokens after the generation prompt so
                # the model is forced to continue from inside the supplied
                # opening tag. Encode without specials so we keep the chat
                # template structure intact.
                extra_ids = tokenizer.encode(
                    assistant_prefill, add_special_tokens=False,
                )
                if extra_ids:
                    extra_tensor = torch.tensor(
                        [extra_ids],
                        dtype=inputs["input_ids"].dtype,
                        device=inputs["input_ids"].device,
                    )
                    inputs["input_ids"] = torch.cat(
                        [inputs["input_ids"], extra_tensor], dim=1,
                    )
                    if "attention_mask" in inputs:
                        attn_extra = torch.ones_like(extra_tensor)
                        inputs["attention_mask"] = torch.cat(
                            [inputs["attention_mask"], attn_extra], dim=1,
                        )
                    prompt_text = prompt_text + assistant_prefill
            per_row_inputs.append(inputs)
            per_row_prompt_text.append(prompt_text)

        # Left-pad inputs to a common length and concatenate. Use the
        # pad-token id from the tokenizer when available; fall back to
        # the EOS id so attention masks behave correctly.
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError(
                "tokenizer must define pad_token_id or eos_token_id "
                "for batched generation"
            )

        max_len = max(int(r["input_ids"].shape[-1]) for r in per_row_inputs)
        batch: dict[str, Any] = {}
        input_ids_rows: list[Any] = []
        attn_rows: list[Any] = []
        for row in per_row_inputs:
            ids = row["input_ids"]
            cur_len = int(ids.shape[-1])
            pad = max_len - cur_len
            if pad > 0:
                pad_tensor = torch.full(
                    (1, pad), pad_id, dtype=ids.dtype, device=ids.device,
                )
                ids = torch.cat([pad_tensor, ids], dim=1)
            input_ids_rows.append(ids)

            attn = row.get("attention_mask")
            if attn is None:
                attn = torch.ones_like(row["input_ids"])
            if pad > 0:
                attn_pad = torch.zeros(
                    (1, pad), dtype=attn.dtype, device=attn.device,
                )
                attn = torch.cat([attn_pad, attn], dim=1)
            attn_rows.append(attn)
        batch["input_ids"] = torch.cat(input_ids_rows, dim=0)
        batch["attention_mask"] = torch.cat(attn_rows, dim=0)

        # Stack multimodal tensor inputs across rows by concatenation
        # (Qwen3-VL's processor packs each row's pixel features without
        # a leading batch dim — the row->item mapping comes from
        # ``image_grid_thw`` / ``video_grid_thw``). We pass through any
        # tensor field the per-row dicts contain so processor variants
        # stay compatible.
        passthrough_keys: set[str] = set()
        for row in per_row_inputs:
            passthrough_keys.update(k for k in row.keys())
        passthrough_keys -= {"input_ids", "attention_mask"}
        for key in sorted(passthrough_keys):
            tensors = []
            lists: list[Any] = []
            for row in per_row_inputs:
                if key not in row:
                    continue
                v = row[key]
                if hasattr(v, "shape"):
                    tensors.append(v)
                elif isinstance(v, list):
                    lists.append(v)
            if tensors:
                batch[key] = torch.cat(tensors, dim=0)
            elif lists:
                batch[key] = [item for sub in lists for item in sub]

        # Move to model device.
        batch = {
            k: (v.to(self._model.device) if hasattr(v, "to") else v)
            for k, v in batch.items()
        }

        do_sample = temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": int(pad_id),
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)
            if top_k is not None:
                gen_kwargs["top_k"] = int(top_k)

        # Batched all-rows-stopped criterion so generation can early-exit
        # when every prompt has produced its stop string. Per-row
        # truncation still happens below.
        stopping_criteria = _build_batched_stopping_criteria(
            processor,
            stop_sequences or [],
            prompt_len=int(batch["input_ids"].shape[-1]),
        )
        if stopping_criteria is not None:
            gen_kwargs["stopping_criteria"] = stopping_criteria

        with torch.inference_mode():
            out = self._model.generate(**batch, **gen_kwargs)

        input_len = int(batch["input_ids"].shape[-1])
        results: list[GenerationResult] = []
        for i, prompt_text in enumerate(per_row_prompt_text):
            # Recover the per-row prompt length (pre-padding) so we can
            # report the generated_token_count.
            raw_ids = per_row_inputs[i]["input_ids"]
            row_prompt_len = int(raw_ids.shape[-1])
            trimmed = out[i, input_len:]
            # Strip any trailing pads / EOS for the decode pass — keeps
            # the reported token count tight against what the model
            # actually produced for this row.
            trimmed_list = trimmed.tolist()
            trimmed_clean = _strip_trailing_pad(trimmed_list, int(pad_id))
            row_trimmed = trimmed[: len(trimmed_clean)] if trimmed_clean else trimmed[:0]
            decoded_generated = self._decode_generated_tokens(
                processor, row_trimmed,
            )
            generated_text, stop_meta = _truncate_at_stop(
                decoded_generated, stop_sequences or [],
            )
            assistant_text, assistant_meta = _restore_assistant_prefill(
                prompt_text=prompt_text,
                generated_text=generated_text,
            )
            metadata = {
                "generated_token_count": int(row_trimmed.shape[-1]),
                "prompt_token_count": int(row_prompt_len),
                "max_new_tokens": int(max_new_tokens),
                "hit_max_new_tokens": int(row_trimmed.shape[-1]) >= int(max_new_tokens),
                "decoded_generated_char_count": len(decoded_generated),
                "generated_char_count": len(generated_text),
                "assistant_char_count": len(assistant_text),
                "prompt_char_count": len(prompt_text),
                "generation_time_stop_enabled": stopping_criteria is not None,
                **stop_meta,
                **assistant_meta,
            }
            results.append(
                GenerationResult(
                    text=assistant_text,
                    metadata=metadata,
                    generated_text=generated_text,
                    prompt_text=prompt_text,
                    prompt_tail=_tail(prompt_text),
                )
            )
        return results

    def generate_multimodal_with_metadata(
        self,
        content: list[dict[str, Any]] | None = None,
        *,
        messages: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Run multimodal generation and retain structural special tokens.

        The default path is a single ``role=user`` turn built from ``content``
        (PhysBench / CausalPhys). Pass ``messages`` for an official-style
        system+user transcript, or ``system=`` to prepend a system turn to
        ``content``. ``messages`` and ``system`` must not be combined.
        """

        if self._model is None or self._processor is None:
            raise RuntimeError("HFBackend.load() must be called first")

        torch = self._torch
        inputs, prompt_text = prepare_hf_multimodal_inputs(
            self._processor,
            content,
            messages=messages,
            system=system,
        )
        inputs = {
            k: (v.to(self._model.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        do_sample = temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)
            if top_k is not None:
                gen_kwargs["top_k"] = int(top_k)
        stopping_criteria = _build_stopping_criteria(
            self._processor,
            stop_sequences or [],
            prompt_len=int(inputs["input_ids"].shape[-1]),
        )
        if stopping_criteria is not None:
            gen_kwargs["stopping_criteria"] = stopping_criteria

        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        trimmed = out[0, input_len:]
        decoded_generated = self._decode_generated_tokens(self._processor, trimmed)
        generated_text, stop_meta = _truncate_at_stop(
            decoded_generated,
            stop_sequences or [],
        )
        assistant_text, assistant_meta = _restore_assistant_prefill(
            prompt_text=prompt_text,
            generated_text=generated_text,
        )
        prompt_tail = _tail(prompt_text)
        metadata = {
            "generated_token_count": int(trimmed.shape[-1]),
            "prompt_token_count": int(input_len),
            "max_new_tokens": int(max_new_tokens),
            "hit_max_new_tokens": int(trimmed.shape[-1]) >= int(max_new_tokens),
            "decoded_generated_char_count": len(decoded_generated),
            "generated_char_count": len(generated_text),
            "assistant_char_count": len(assistant_text),
            "prompt_char_count": len(prompt_text),
            "generation_time_stop_enabled": stopping_criteria is not None,
            **stop_meta,
            **assistant_meta,
        }
        return GenerationResult(
            text=assistant_text,
            metadata=metadata,
            generated_text=generated_text,
            prompt_text=prompt_text,
            prompt_tail=prompt_tail,
        )

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _validate_video(video: Any) -> None:
        """Accept a pre-sampled ndarray, ``(frames, metadata)``, or VideoBundle."""

        split_presampled_video(video)

    @staticmethod
    def _validate_image(image: Any) -> None:
        """Accept an ``(H, W, 3) uint8`` ndarray OR a PIL Image.

        PIL images are passed through untouched — the Qwen3-VL processor
        accepts either form. We only guard against obviously-wrong shapes
        so mistakes surface early instead of deep inside ``apply_chat_template``.
        """

        # Lazy PIL import keeps the dependency optional for callers that
        # only ever pass numpy arrays.
        try:
            from PIL.Image import Image as PILImage
        except ImportError:  # pragma: no cover
            PILImage = None  # type: ignore

        if PILImage is not None and isinstance(image, PILImage):
            return
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"image ndarray must be shaped (H, W, 3); got {image.shape}"
                )
            if image.dtype != np.uint8:
                raise ValueError(
                    f"image ndarray must be dtype uint8; got {image.dtype}"
                )
            return
        raise TypeError(
            f"image must be PIL.Image or np.ndarray (H, W, 3) uint8; got {type(image)!r}"
        )

    @staticmethod
    def _decode_generated_tokens(processor: Any, token_ids: Any) -> str:
        """Decode generated tokens while preserving structural tags.

        Qwen3-VL Thinking may represent ``<think>`` / ``</think>`` as
        special tokens. Skipping special tokens would erase the wrappers
        that downstream parsers and rewards depend on.
        """

        return processor.batch_decode(
            [token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]


__all__ = [
    "HFBackend",
    "GenerationResult",
    "DEFAULT_MODEL",
    "presampled_video_metadata",
    "split_presampled_video",
    "prepare_hf_multimodal_inputs",
]


def _build_stopping_criteria(
    processor: Any,
    stop_sequences: list[str],
    *,
    prompt_len: int,
) -> Any | None:
    """Build token-level stopping criteria for exact stop strings.

    ``_truncate_at_stop`` below is still kept as a post-generation safety net,
    but this criteria prevents Thinking-model benchmark runs from wasting time
    after the closing ``</answer>`` tag has already been generated.
    """

    encoded: list[list[int]] = []
    tokenizer = getattr(processor, "tokenizer", processor)
    for stop in stop_sequences:
        if not stop:
            continue
        try:
            ids = tokenizer.encode(stop, add_special_tokens=False)
        except AttributeError:
            ids = tokenizer(stop, add_special_tokens=False)["input_ids"]
        if ids:
            encoded.append([int(x) for x in ids])
    if not encoded:
        return None

    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StopOnTokenSequences(StoppingCriteria):
        def __init__(self, stop_token_ids: list[list[int]], input_len: int) -> None:
            self.stop_token_ids = stop_token_ids
            self.input_len = input_len

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            generated = input_ids[0, self.input_len :].tolist()
            for seq in self.stop_token_ids:
                if len(generated) >= len(seq) and generated[-len(seq) :] == seq:
                    return True
            return False

    return StoppingCriteriaList([_StopOnTokenSequences(encoded, prompt_len)])


def _build_batched_stopping_criteria(
    processor: Any,
    stop_sequences: list[str],
    *,
    prompt_len: int,
) -> Any | None:
    """Batched all-rows-stopped criterion for :meth:`generate_multimodal_batch`.

    The criterion only fires when *every* row has produced one of the
    stop sequences in its trailing tokens. Per-row truncation still
    happens post-generation in :func:`_truncate_at_stop`. We accept a
    small amount of wasted compute on early-finishing rows in exchange
    for not silently dropping mid-generation rows.
    """

    encoded: list[list[int]] = []
    tokenizer = getattr(processor, "tokenizer", processor)
    for stop in stop_sequences:
        if not stop:
            continue
        try:
            ids = tokenizer.encode(stop, add_special_tokens=False)
        except AttributeError:
            ids = tokenizer(stop, add_special_tokens=False)["input_ids"]
        if ids:
            encoded.append([int(x) for x in ids])
    if not encoded:
        return None

    from transformers import StoppingCriteria, StoppingCriteriaList

    class _AllRowsStopped(StoppingCriteria):
        def __init__(self, stop_token_ids: list[list[int]], input_len: int) -> None:
            self.stop_token_ids = stop_token_ids
            self.input_len = input_len

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            generated = input_ids[:, self.input_len :]
            n_rows = int(generated.shape[0])
            for row_idx in range(n_rows):
                row = generated[row_idx].tolist()
                if not _row_has_stop(row, self.stop_token_ids):
                    return False
            return True

    return StoppingCriteriaList([_AllRowsStopped(encoded, prompt_len)])


def _row_has_stop(row: list[int], stop_token_ids: list[list[int]]) -> bool:
    for seq in stop_token_ids:
        if len(row) >= len(seq) and row[-len(seq) :] == seq:
            return True
    return False


def _strip_trailing_pad(token_ids: list[int], pad_id: int) -> list[int]:
    """Drop the trailing run of pad tokens (HF generate writes pads when
    individual rows finish early in a batch)."""

    end = len(token_ids)
    while end > 0 and int(token_ids[end - 1]) == int(pad_id):
        end -= 1
    return token_ids[:end]


def _truncate_at_stop(text: str, stop_sequences: list[str]) -> tuple[str, dict[str, Any]]:
    """Return text truncated after the earliest stop sequence plus metadata."""

    if not stop_sequences:
        return text, {
            "hit_stop_sequence": False,
            "stop_sequence": None,
            "stop_index": None,
        }
    best: int | None = None
    best_stop = ""
    for stop in stop_sequences:
        if not stop:
            continue
        idx = text.find(stop)
        if idx >= 0 and (best is None or idx < best):
            best = idx
            best_stop = stop
    if best is None:
        return text, {
            "hit_stop_sequence": False,
            "stop_sequence": None,
            "stop_index": None,
        }
    return text[: best + len(best_stop)], {
        "hit_stop_sequence": True,
        "stop_sequence": best_stop,
        "stop_index": best,
    }


def presampled_video_metadata(
    video: np.ndarray,
    *,
    fps: float = _PRESAMPLED_VIDEO_FPS,
) -> dict[str, Any]:
    """Build Qwen3-VL ``video_metadata`` for a pre-sampled clip.

    ``do_sample_frames`` is intentionally omitted: ``transformers.VideoMetadata``
    does not accept that field. Pass ``do_sample_frames=False`` as a
    processor keyword instead (same contract as training).
    """

    _validate_video_array(video)
    n_frames = int(video.shape[0])
    height = int(video.shape[1])
    width = int(video.shape[2])
    fps_value = float(fps)
    return {
        "total_num_frames": n_frames,
        "fps": fps_value,
        "width": width,
        "height": height,
        "duration": float(n_frames) / fps_value,
        "frames_indices": list(range(n_frames)),
    }


def split_presampled_video(
    video: np.ndarray | VideoBundle | tuple[np.ndarray, dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(frames, hf_metadata)`` without stripping caller metadata.

    Extra keys (including ``do_sample_frames``) are dropped from the
    dict that is later passed as ``video_metadata``.
    """

    extra: dict[str, Any] = {}
    if isinstance(video, VideoBundle):
        frames = video.frames
        extra = dict(video.metadata)
    elif isinstance(video, tuple) and len(video) == 2 and isinstance(video[1], dict):
        frames, extra = video[0], dict(video[1])
    elif isinstance(video, np.ndarray):
        frames = video
    else:
        raise TypeError(
            "video must be np.ndarray (n_frames, H, W, 3), "
            f"VideoBundle, or (frames, metadata); got {type(video)!r}"
        )
    _validate_video_array(frames)
    extra.pop("do_sample_frames", None)
    metadata = presampled_video_metadata(frames)
    for key in _HF_VIDEO_METADATA_KEYS:
        if key in extra and extra[key] is not None:
            metadata[key] = extra[key]
    return frames, metadata


def _normalize_chat_messages(
    content: list[dict[str, Any]] | None,
    *,
    messages: list[dict[str, Any]] | None,
    system: str | None,
) -> list[dict[str, Any]]:
    """Return a chat transcript. Default is a single user turn."""

    if messages is not None and content is not None:
        raise ValueError("pass content or messages, not both")
    if messages is not None:
        if system is not None:
            raise ValueError("system= is for the content path; put system in messages")
        if not messages:
            raise ValueError("messages must be non-empty")
        return messages
    if content is None:
        raise ValueError("content or messages is required")
    transcript: list[dict[str, Any]] = []
    if system is not None:
        transcript.append({"role": "system", "content": system})
    transcript.append({"role": "user", "content": content})
    return transcript


def _prepare_message_body(
    body: Any,
) -> tuple[Any, list[np.ndarray], list[dict[str, Any]], list[Any]]:
    """Rewrite one message body for the chat template and collect pixels."""

    if isinstance(body, str):
        return body, [], [], []
    if not isinstance(body, list):
        raise TypeError(
            "message content must be a string or a list of typed entries; "
            f"got {type(body)!r}"
        )
    template_content: list[dict[str, Any]] = []
    videos: list[np.ndarray] = []
    video_metadata: list[dict[str, Any]] = []
    images: list[Any] = []
    for entry in body:
        kind = entry.get("type")
        if kind == "video":
            frames, metadata = split_presampled_video(entry["video"])
            # Match training: the chat template only needs a video slot.
            # Pixels go through processor(videos=..., video_metadata=...).
            template_content.append({"type": "video"})
            videos.append(frames)
            video_metadata.append(metadata)
        elif kind == "image":
            HFBackend._validate_image(entry["image"])
            template_content.append(entry)
            images.append(entry["image"])
        elif kind == "text":
            template_content.append(entry)
        else:
            raise ValueError(f"unsupported content entry type: {kind!r}")
    return template_content, videos, video_metadata, images


def prepare_hf_multimodal_inputs(
    processor: Any,
    content: list[dict[str, Any]] | None = None,
    *,
    messages: list[dict[str, Any]] | None = None,
    system: str | None = None,
) -> tuple[Any, str]:
    """Tokenize chat content, keeping pre-sampled video frames as-is.

    Video-bearing turns use ``tokenize=False`` then
    ``processor(..., videos=..., video_metadata=..., do_sample_frames=False)``.
    Text/image-only turns keep the existing ``apply_chat_template`` tokenize
    path so image eval is unchanged. Omitting ``messages`` keeps the
    PhysBench user-only transcript.
    """

    transcript = _normalize_chat_messages(
        content, messages=messages, system=system,
    )
    template_messages: list[dict[str, Any]] = []
    videos: list[np.ndarray] = []
    video_metadata: list[dict[str, Any]] = []
    images: list[Any] = []
    for message in transcript:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported chat role: {role!r}")
        body, body_videos, body_meta, body_images = _prepare_message_body(
            message.get("content"),
        )
        template_messages.append({"role": role, "content": body})
        videos.extend(body_videos)
        video_metadata.extend(body_meta)
        images.extend(body_images)

    if videos and os.environ.get("PHYSEVOLVE_LEGACY_VIDEO_PACKING") == "1":
        # Preserve processor-side sampling for an explicitly selected protocol.
        legacy_messages: list[dict[str, Any]] = []
        for message in transcript:
            body = message["content"]
            if isinstance(body, list):
                body = [
                    {"type": "video", "video": split_presampled_video(entry["video"])[0]}
                    if entry.get("type") == "video" else entry
                    for entry in body
                ]
            legacy_messages.append({"role": message["role"], "content": body})
        inputs = processor.apply_chat_template(
            legacy_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    elif videos:
        prompt = processor.apply_chat_template(
            template_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        kwargs: dict[str, Any] = {
            "text": [prompt],
            "videos": videos,
            "video_metadata": video_metadata,
            "do_sample_frames": False,
            "return_tensors": "pt",
        }
        if images:
            kwargs["images"] = images
        inputs = processor(**kwargs)
    else:
        inputs = processor.apply_chat_template(
            template_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    if hasattr(inputs, "items"):
        inputs = dict(inputs)
    prompt_text = HFBackend._decode_generated_tokens(
        processor,
        inputs["input_ids"][0],
    )
    return inputs, prompt_text


def _validate_video_array(video: np.ndarray) -> None:
    if not isinstance(video, np.ndarray):
        raise TypeError(
            f"video must be np.ndarray (n_frames, H, W, 3); got {type(video)!r}"
        )
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(
            f"video must be shaped (n_frames, H, W, 3); got {video.shape}"
        )
    if video.dtype != np.uint8:
        raise ValueError(
            f"video must be dtype uint8; got {video.dtype}"
        )


def _restore_assistant_prefill(
    *,
    prompt_text: str,
    generated_text: str,
) -> tuple[str, dict[str, Any]]:
    """Restore an assistant ``<think>`` prefill removed with the prompt.

    Returns the response and diagnostic flags. Prompts without this prefill,
    including the Qwen3-VL Instruct template, leave the response unchanged.
    """

    prompt_ends_with_think = prompt_text.rstrip().endswith("<think>")
    generated_starts_with_think = generated_text.lstrip().lower().startswith("<think>")
    restored = prompt_ends_with_think and not generated_starts_with_think
    if restored:
        return "<think>\n" + generated_text, {
            "prompt_ends_with_think": True,
            "generated_starts_with_think": False,
            "assistant_prefill_restored": True,
        }
    return generated_text, {
        "prompt_ends_with_think": prompt_ends_with_think,
        "generated_starts_with_think": generated_starts_with_think,
        "assistant_prefill_restored": False,
    }


def _tail(text: str, *, max_chars: int = 512) -> str:
    """Return the diagnostic tail of a prompt/template string."""

    if len(text) <= max_chars:
        return text
    return text[-max_chars:]
