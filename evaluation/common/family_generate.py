"""Per-family VLM generate paths for non-Qwen reference evaluation.

This is the single upstream-workaround module for InternVL2.5, Llama-3.2-Vision,
and Pixtral-12B. Dataset iteration, prompts, scorers, and shard→merge stay in
the existing PhysBench / CausalPhys / ContPhy launchers. Qwen3-VL continues to
use :class:`~physevolve.inference.hf_backend.HFBackend` unchanged.

Delete this file when all three families ship an official AutoProcessor video
contract that can consume the shared ``[{type, image|video|text}, ...]``
content list without custom chat or pixel packing.

Sources (local copies; no Hugging Face fetch):

- InternVL2.5-8B README Quick Start ``dynamic_preprocess`` / 448 px /
  video ``max_num=1``:
  ``models/InternVL2_5-8B/README.md``
- Llama-3.2-11B-Vision-Instruct HuggingFace Mllama chat template
  (``{"type": "image"}`` placeholders + ``AutoProcessor``)
- Pixtral-12B-2409 Mistral-native dump (``params.json`` + ``tekken.json``)
  via the already-installed vLLM ``tokenizer_mode="mistral"`` path. Do not
  install ``mistral_inference`` and do not convert the dump to HF weights.

InternVL ``InternLM2ForCausalLM`` on transformers>=4.50: ``ensure_hf_generate``
attaches ``GenerationMixin`` to ``language_model`` only (not
``InternVLChatModel``, which already has a custom ``generate`` that injects
ViT embeddings) and fills ``generation_config`` via
``GenerationConfig.from_model_config`` when it is still ``None``. Empty
``DynamicCache`` is converted to ``None`` in
``_patch_internlm2_prepare_inputs`` so first-step ``inputs_embeds`` (ViT)
is preserved. Delete those helpers when InternVL remote code inherits
``GenerationMixin`` and ``prepare_inputs_for_generation`` handles HF
``Cache`` (no ``[0][0]`` index on an empty cache).
"""

from __future__ import annotations

import base64
import io
import json
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from physevolve.inference.hf_backend import GenerationResult, HFBackend


EvalFamily = Literal["qwen", "internvl", "llama32_vision", "pixtral"]
EVAL_FAMILIES: tuple[EvalFamily, ...] = (
    "qwen",
    "internvl",
    "llama32_vision",
    "pixtral",
)

INTERN_VL_IMAGE_SIZE = 448
INTERN_VL_VIDEO_MAX_NUM = 1
INTERN_VL_IMAGE_MAX_NUM = 12
INTERN_VL_MEAN = (0.485, 0.456, 0.406)
INTERN_VL_STD = (0.229, 0.224, 0.225)
PIXTRAL_MAX_IMAGES = 12

_WEIGHT_GLOBS = ("*.safetensors", "*.bin", "consolidated*.safetensors")


# ─── Family detection ────────────────────────────────────────────────


def detect_eval_family(model_path: str | Path) -> EvalFamily:
    """Infer the eval family from a local directory name and config files."""

    root = Path(model_path)
    blob = _normalize_name(root.name)
    model_type, architectures = _peek_hf_config(root)

    if (
        "internvl" in blob
        or model_type == "internvl_chat"
        or any("internvl" in item for item in architectures)
    ):
        return "internvl"
    if "pixtral" in blob or _looks_like_mistral_pixtral(root, blob):
        return "pixtral"
    if (
        "llama_3_2" in blob
        or "llama32" in blob
        or "mllama" in blob
        or model_type == "mllama"
        or any("mllama" in item for item in architectures)
        or ("llama" in blob and "vision" in blob)
    ):
        return "llama32_vision"
    return "qwen"


def resolve_eval_family(
    model_path: str | Path,
    family: str | None = None,
) -> EvalFamily:
    """Return ``family`` if given, otherwise detect it from ``model_path``."""

    if family is None or family == "auto":
        return detect_eval_family(model_path)
    if family not in EVAL_FAMILIES:
        raise ValueError(
            f"unsupported --model-family {family!r}; expected one of "
            f"{list(EVAL_FAMILIES)}"
        )
    return family  # type: ignore[return-value]


def _normalize_name(name: str) -> str:
    return name.casefold().replace(".", "_").replace("-", "_").replace(" ", "_")


def _peek_hf_config(root: Path) -> tuple[str, list[str]]:
    config_path = root / "config.json"
    if not config_path.is_file():
        return "", []
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "", []
    if not isinstance(payload, dict):
        return "", []
    model_type = str(payload.get("model_type") or "").casefold()
    raw_arch = payload.get("architectures") or []
    architectures = [
        str(item).casefold() for item in raw_arch if item is not None
    ]
    return model_type, architectures


def _looks_like_mistral_pixtral(root: Path, blob: str) -> bool:
    if "pixtral" in blob:
        return True
    has_mistral_files = (root / "params.json").is_file() and (
        root / "tekken.json"
    ).is_file()
    return has_mistral_files and not (root / "config.json").is_file()


def assert_local_weights_ready(model_path: str | Path, family: EvalFamily) -> None:
    """Refuse Hugging Face fetches when the local dump is still incomplete."""

    root = Path(model_path)
    if not root.is_dir():
        raise FileNotFoundError(
            f"model directory not found: {root}; expected a finished dump "
            "under models/"
        )
    if family == "pixtral":
        missing = [
            name
            for name in ("params.json", "tekken.json")
            if not (root / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Pixtral dump at {root} is missing {missing}; "
                "owner download is not finished (no Hugging Face fetch)"
            )
    elif family != "qwen":
        if not (root / "config.json").is_file():
            raise FileNotFoundError(
                f"{family} dump at {root} has no config.json; "
                "owner download is not finished (no Hugging Face fetch)"
            )
    if family == "qwen":
        return
    weight_files = [
        path
        for pattern in _WEIGHT_GLOBS
        for path in root.glob(pattern)
        if path.is_file()
    ]
    if not weight_files:
        raise FileNotFoundError(
            f"local weights not ready at {root}; wait for the owner download "
            "under models/ "
            "(this eval path never downloads from Hugging Face)"
        )


def ensure_hf_generate(module: Any) -> None:
    """Attach ``GenerationMixin`` when a remote-code LLM lost ``.generate``.

    InternVL2.5 ``InternLM2ForCausalLM`` implements
    ``prepare_inputs_for_generation`` but does not subclass
    ``GenerationMixin``. transformers >= 4.50 removed ``generate`` from
    ``PreTrainedModel``, so ``InternVLChatModel.chat`` crashes on the first
    item. After the mixin, ``language_model.generation_config`` can still be
    ``None`` because the LM was constructed without ``GenerationMixin`` in
    its MRO; fill it from ``module.config`` only. Packing, prompts, and
    sampling kwargs stay unchanged.

    Delete when InternVL remote code inherits ``GenerationMixin``, or when
    ``language_model.generate`` and ``language_model.generation_config``
    exist after ``from_pretrained``.
    """

    if not callable(getattr(module, "generate", None)):
        from transformers import GenerationMixin

        cls = module.__class__
        if GenerationMixin not in cls.__mro__:
            module.__class__ = type(cls.__name__, (cls, GenerationMixin), {})
        if not callable(getattr(module, "generate", None)):
            raise AttributeError(
                f"{cls.__name__} still has no generate after attaching "
                "GenerationMixin"
            )

    if getattr(module, "generation_config", None) is not None:
        return
    config = getattr(module, "config", None)
    if config is None:
        return
    from transformers import GenerationConfig

    module.generation_config = GenerationConfig.from_model_config(config)


_INTERLM2_PREPARE_PATCHED = "_physevolve_internlm2_prepare_patched"


def _legacy_past_key_values(past: Any) -> Any:
    """Map empty HF ``Cache`` to ``None`` so InternLM2 keeps first-step embeds.

    transformers>=4.50 passes an empty ``DynamicCache`` into
    ``prepare_inputs_for_generation``. InternVL remote InternLM2 then does
    ``past_key_values[0][0].shape[2]`` and crashes. Treating length-0 cache
    as ``None`` preserves InternVL's ``inputs_embeds`` path (ViT already
    injected). Non-empty cache is converted to the legacy tuple InternLM2
    already indexes.
    """

    if past is None:
        return None
    get_len = getattr(past, "get_seq_length", None)
    if callable(get_len):
        try:
            length = get_len()
        except TypeError:
            length = get_len(0)
        if int(length) == 0:
            return None
    to_legacy = getattr(past, "to_legacy_cache", None)
    if callable(to_legacy):
        past = to_legacy()
    try:
        key = past[0][0]
    except (IndexError, KeyError, TypeError):
        return None
    if key is None:
        return None
    return past


def _patch_internlm2_prepare_inputs(module: Any) -> None:
    """Wrap InternLM2 ``prepare_inputs_for_generation``; do not rewrite slicing."""

    if getattr(module, _INTERLM2_PREPARE_PATCHED, False):
        return
    bound = getattr(module, "prepare_inputs_for_generation", None)
    if not callable(bound):
        raise AttributeError(
            "InternLM2 language_model has no prepare_inputs_for_generation"
        )
    unbound = getattr(bound, "__func__", bound)

    def wrapped(
        self: Any,
        input_ids: Any,
        past_key_values: Any = None,
        attention_mask: Any = None,
        inputs_embeds: Any = None,
        **kwargs: Any,
    ) -> Any:
        past_key_values = _legacy_past_key_values(past_key_values)
        return unbound(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    module.prepare_inputs_for_generation = types.MethodType(wrapped, module)
    setattr(module, _INTERLM2_PREPARE_PATCHED, True)


# ─── Shared content flattening ───────────────────────────────────────


def video_frames_from_entry(video: Any) -> tuple[np.ndarray, ...]:
    """Return RGB frames from a bare ndarray or ``(frames, metadata)`` tuple."""

    frames = video[0] if _is_presampled_tuple(video) else video
    if not isinstance(frames, np.ndarray):
        raise TypeError(
            "video frames must be np.ndarray (n_frames, H, W, 3); "
            f"got {type(frames)!r}"
        )
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"video must be shaped (n_frames, H, W, 3); got {frames.shape}"
        )
    return tuple(frames[index] for index in range(int(frames.shape[0])))


def _is_presampled_tuple(video: Any) -> bool:
    return (
        isinstance(video, tuple)
        and len(video) == 2
        and isinstance(video[1], dict)
    )


def ndarray_to_pil(image: Any) -> Any:
    """Convert an RGB ndarray or pass through a PIL image."""

    from PIL.Image import Image as PILImage
    from PIL import Image

    if isinstance(image, PILImage):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"image ndarray must be shaped (H, W, 3); got {array.shape}"
        )
    if array.dtype != np.uint8:
        array = array.astype(np.uint8, copy=False)
    return Image.fromarray(array, mode="RGB")


@dataclass(frozen=True, slots=True)
class InternVLVisual:
    """One InternVL ``<image>`` slot and its tile budget."""

    image: Any
    max_num: int
    source: str


def internvl_visual_plan(content: Sequence[dict[str, Any]]) -> list[InternVLVisual]:
    """Walk shared content and emit InternVL pixel slots in prompt order."""

    plan: list[InternVLVisual] = []
    for entry in content:
        kind = entry.get("type")
        if kind == "text":
            continue
        if kind == "video":
            for frame in video_frames_from_entry(entry["video"]):
                plan.append(
                    InternVLVisual(
                        image=ndarray_to_pil(frame),
                        max_num=INTERN_VL_VIDEO_MAX_NUM,
                        source="video",
                    )
                )
            continue
        if kind == "image":
            plan.append(
                InternVLVisual(
                    image=ndarray_to_pil(entry["image"]),
                    max_num=INTERN_VL_IMAGE_MAX_NUM,
                    source="image",
                )
            )
            continue
        raise ValueError(f"unsupported content entry type: {kind!r}")
    return plan


def internvl_question_from_content(content: Sequence[dict[str, Any]]) -> str:
    """Rebuild the official text order with InternVL ``<image>`` placeholders.

    Video slots become ``Frame{{i}}: <image>\\n`` (448 px, ``max_num=1``).
    Still images become ``<image>\\n`` when they are the only visual, otherwise
    ``Image-{{i}}: <image>\\n``. Frame indices come from our loaders, not
    InternVL's ``VideoReader`` / ``num_segments=32`` sampler.
    """

    n_images = sum(1 for entry in content if entry.get("type") == "image")
    n_videos = sum(1 for entry in content if entry.get("type") == "video")
    parts: list[str] = []
    frame_index = 0
    image_index = 0
    for entry in content:
        kind = entry.get("type")
        if kind == "text":
            parts.append(str(entry.get("text") or ""))
            continue
        if kind == "video":
            for _frame in video_frames_from_entry(entry["video"]):
                frame_index += 1
                parts.append(f"Frame{frame_index}: <image>\n")
            continue
        if kind == "image":
            image_index += 1
            if n_videos == 0 and n_images == 1:
                parts.append("<image>\n")
            else:
                parts.append(f"Image-{image_index}: <image>\n")
            continue
        raise ValueError(f"unsupported content entry type: {kind!r}")
    question = "".join(parts)
    n_placeholders = question.count("<image>")
    n_visuals = count_visual_slots(content)
    if n_placeholders != n_visuals:
        raise ValueError(
            "InternVL <image> count "
            f"{n_placeholders} != visual slots {n_visuals}"
        )
    return question


def count_visual_slots(content: Sequence[dict[str, Any]]) -> int:
    """Count video frames plus still images in shared eval content."""

    total = 0
    for entry in content:
        kind = entry.get("type")
        if kind == "video":
            total += len(video_frames_from_entry(entry["video"]))
        elif kind == "image":
            total += 1
    return total


def llama_style_chat_content(
    content: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Return interleaved chat parts and PIL images for Llama / similar HFs."""

    message_content: list[dict[str, Any]] = []
    images: list[Any] = []
    for entry in content:
        kind = entry.get("type")
        if kind == "text":
            text = str(entry.get("text") or "")
            if text:
                message_content.append({"type": "text", "text": text})
            continue
        if kind == "video":
            for frame in video_frames_from_entry(entry["video"]):
                images.append(ndarray_to_pil(frame))
                message_content.append({"type": "image"})
            continue
        if kind == "image":
            images.append(ndarray_to_pil(entry["image"]))
            message_content.append({"type": "image"})
            continue
        raise ValueError(f"unsupported content entry type: {kind!r}")
    return message_content, images


def pixtral_chat_content(content: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return vLLM/Mistral chat parts with JPEG data-URI images."""

    message_content: list[dict[str, Any]] = []
    n_images = 0
    for entry in content:
        kind = entry.get("type")
        if kind == "text":
            text = str(entry.get("text") or "")
            if text:
                message_content.append({"type": "text", "text": text})
            continue
        if kind == "video":
            frames = video_frames_from_entry(entry["video"])
            n_images += len(frames)
            for frame in frames:
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_data_uri(ndarray_to_pil(frame))},
                    }
                )
            continue
        if kind == "image":
            n_images += 1
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": pil_to_data_uri(ndarray_to_pil(entry["image"]))
                    },
                }
            )
            continue
        raise ValueError(f"unsupported content entry type: {kind!r}")
    if n_images > PIXTRAL_MAX_IMAGES:
        raise ValueError(
            f"Pixtral eval allows at most {PIXTRAL_MAX_IMAGES} images; "
            f"got {n_images}"
        )
    return message_content


def pil_to_data_uri(image: Any) -> str:
    """Encode a PIL image as a JPEG data URI for vLLM image_url content."""

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# ─── InternVL2.5 preprocess (README Quick Start, typed) ──────────────


def internvl_find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[tuple[int, int]] | set[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Pick the tile grid closest to ``aspect_ratio`` (InternVL README)."""

    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def internvl_dynamic_preprocess(
    image: Any,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = INTERN_VL_IMAGE_SIZE,
    use_thumbnail: bool = False,
) -> list[Any]:
    """Split one PIL image into 448×448 tiles (InternVL README)."""

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda item: item[0] * item[1],
    )
    target_aspect_ratio = internvl_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for index in range(blocks):
        box = (
            (index % (target_width // image_size)) * image_size,
            (index // (target_width // image_size)) * image_size,
            ((index % (target_width // image_size)) + 1) * image_size,
            ((index // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if len(processed_images) != blocks:
        raise RuntimeError(
            f"InternVL tile count {len(processed_images)} != blocks {blocks}"
        )
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def internvl_build_transform(input_size: int = INTERN_VL_IMAGE_SIZE) -> Any:
    """ImageNet-normalized 448 resize used by InternVL ``load_image``."""

    import torchvision.transforms as transforms
    from torchvision.transforms.functional import InterpolationMode

    return transforms.Compose(
        [
            transforms.Lambda(
                lambda img: img.convert("RGB") if img.mode != "RGB" else img
            ),
            transforms.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=INTERN_VL_MEAN, std=INTERN_VL_STD),
        ]
    )


# ─── Backends ────────────────────────────────────────────────────────


def _single_visible_gpu_device_map() -> str | None:
    """Match PhysBench HF eval: one visible GPU → ``cuda:0``."""

    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    visible = [token for token in raw.split(",") if token.strip()]
    if len(visible) == 1:
        return "cuda:0"
    return None


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def _truncate_generated(
    text: str, stop_sequences: Sequence[str] | None
) -> tuple[str, dict[str, Any]]:
    """Cut the completion at the earliest stop string."""

    if not stop_sequences:
        return text, {"hit_stop_sequence": False, "stop_sequence": None}
    best: int | None = None
    best_stop = ""
    for stop in stop_sequences:
        if not stop:
            continue
        index = text.find(stop)
        if index >= 0 and (best is None or index < best):
            best = index
            best_stop = stop
    if best is None:
        return text, {"hit_stop_sequence": False, "stop_sequence": None}
    return text[: best + len(best_stop)], {
        "hit_stop_sequence": True,
        "stop_sequence": best_stop,
        "stop_index": best,
    }


def _model_device(model: Any) -> Any:
    return next(model.parameters()).device


class _FamilyBackendBase:
    """Unloaded context-manager backend with the HFBackend generate surface."""

    family: EvalFamily
    name: str

    def __init__(
        self,
        model_path: str | Path,
        *,
        device_map: str | dict | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.device_map = device_map
        if self.device_map is None:
            self.device_map = _single_visible_gpu_device_map() or "auto"
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None

    def __enter__(self) -> "_FamilyBackendBase":
        self.load()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def load(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._tokenizer = None
        if self._torch is not None:
            try:
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 — best-effort
                pass

    def generate_multimodal_with_metadata(
        self,
        content: list[dict[str, Any]] | None = None,
        *,
        messages: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> GenerationResult:
        if messages is not None or system is not None:
            raise ValueError(
                f"{self.family} eval backend accepts content= only; "
                "do not pass messages= or system="
            )
        if content is None:
            raise ValueError("content is required")
        if self._model is None:
            raise RuntimeError(f"{type(self).__name__}.load() must be called first")
        return self._generate(
            content,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences or [],
        )

    def _generate(
        self,
        content: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None,
        stop_sequences: list[str],
    ) -> GenerationResult:
        raise NotImplementedError

    def _result(
        self,
        *,
        text: str,
        prompt_text: str,
        prompt_token_count: int | None,
        generated_token_count: int | None,
        max_new_tokens: int,
        stop_sequences: list[str],
    ) -> GenerationResult:
        truncated, stop_meta = _truncate_generated(text, stop_sequences)
        metadata: dict[str, Any] = {
            "eval_family": self.family,
            "max_new_tokens": int(max_new_tokens),
            "generated_char_count": len(truncated),
            "prompt_char_count": len(prompt_text),
            **stop_meta,
        }
        if prompt_token_count is not None:
            metadata["prompt_token_count"] = int(prompt_token_count)
        if generated_token_count is not None:
            metadata["generated_token_count"] = int(generated_token_count)
            metadata["hit_max_new_tokens"] = int(generated_token_count) >= int(
                max_new_tokens
            )
        return GenerationResult(
            text=truncated,
            metadata=metadata,
            generated_text=truncated,
            prompt_text=prompt_text,
            prompt_tail=prompt_text[-512:],
        )


class InternVLFamilyBackend(_FamilyBackendBase):
    """InternVL2.5 ``model.chat`` path over already-sampled frames."""

    family: EvalFamily = "internvl"
    name = "internvl"

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        assert_local_weights_ready(self.model_path, "internvl")
        self._torch = torch
        dtype = torch.bfloat16
        use_flash = _flash_attn_available()
        self._model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash,
            trust_remote_code=True,
            local_files_only=True,
            device_map=self.device_map,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        self._model.eval()
        language_model = getattr(self._model, "language_model", None)
        if language_model is None:
            raise AttributeError("InternVL chat model has no language_model")
        ensure_hf_generate(language_model)
        _patch_internlm2_prepare_inputs(language_model)

    def _generate(
        self,
        content: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None,
        stop_sequences: list[str],
    ) -> GenerationResult:
        question = internvl_question_from_content(content)
        plan = internvl_visual_plan(content)
        torch = self._torch
        pixel_values = None
        num_patches_list: list[int] = []
        if plan:
            transform = internvl_build_transform(INTERN_VL_IMAGE_SIZE)
            tiles: list[Any] = []
            for visual in plan:
                processed = internvl_dynamic_preprocess(
                    visual.image,
                    min_num=1,
                    max_num=visual.max_num,
                    image_size=INTERN_VL_IMAGE_SIZE,
                    use_thumbnail=True,
                )
                stacked = torch.stack([transform(tile) for tile in processed])
                tiles.append(stacked)
                num_patches_list.append(int(stacked.shape[0]))
            pixel_values = torch.cat(tiles, dim=0)
            device = _model_device(self._model)
            pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
        generation_config: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_config["temperature"] = float(temperature)
            generation_config["top_p"] = float(top_p)
            if top_k is not None:
                generation_config["top_k"] = int(top_k)
        response = self._model.chat(
            self._tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list or None,
        )
        prompt_tokens = None
        generated_tokens = None
        if self._tokenizer is not None:
            prompt_tokens = len(self._tokenizer.encode(question))
            generated_tokens = len(self._tokenizer.encode(str(response)))
        return self._result(
            text=str(response),
            prompt_text=question,
            prompt_token_count=prompt_tokens,
            generated_token_count=generated_tokens,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
        )


class LlamaVisionFamilyBackend(_FamilyBackendBase):
    """Llama-3.2-Vision Mllama path: video becomes an image sequence."""

    family: EvalFamily = "llama32_vision"
    name = "llama32_vision"

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        assert_local_weights_ready(self.model_path, "llama32_vision")
        self._torch = torch
        self._model = MllamaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            local_files_only=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self._model.eval()

    def _generate(
        self,
        content: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None,
        stop_sequences: list[str],
    ) -> GenerationResult:
        message_content, images = llama_style_chat_content(content)
        messages = [{"role": "user", "content": message_content}]
        prompt_text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        processor_kwargs: dict[str, Any] = {
            "text": prompt_text,
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
        if images:
            processor_kwargs["images"] = images
        inputs = self._processor(**processor_kwargs)
        device = _model_device(self._model)
        inputs = {
            key: (value.to(device) if hasattr(value, "to") else value)
            for key, value in dict(inputs).items()
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
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
            gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
        torch = self._torch
        with torch.inference_mode():
            output = self._model.generate(**inputs, **gen_kwargs)
        input_len = int(inputs["input_ids"].shape[-1])
        generated_ids = output[0, input_len:]
        decoded = self._processor.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return self._result(
            text=decoded,
            prompt_text=prompt_text,
            prompt_token_count=input_len,
            generated_token_count=int(generated_ids.shape[-1]),
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
        )


class PixtralFamilyBackend(_FamilyBackendBase):
    """Pixtral-12B Mistral dump via vLLM ``tokenizer_mode=mistral``."""

    family: EvalFamily = "pixtral"
    name = "pixtral"

    def load(self) -> None:
        from vllm import LLM

        assert_local_weights_ready(self.model_path, "pixtral")
        self._model = LLM(
            model=self.model_path,
            tokenizer_mode="mistral",
            dtype="bfloat16",
            limit_mm_per_prompt={"image": PIXTRAL_MAX_IMAGES},
            trust_remote_code=True,
        )

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._tokenizer = None

    def _generate(
        self,
        content: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int | None,
        stop_sequences: list[str],
    ) -> GenerationResult:
        from vllm import SamplingParams

        message_content = pixtral_chat_content(content)
        messages = [{"role": "user", "content": message_content}]
        sampling_kwargs: dict[str, Any] = {
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature) if temperature > 0 else 0.0,
            "top_p": float(top_p),
        }
        if top_k is not None:
            sampling_kwargs["top_k"] = int(top_k)
        if stop_sequences:
            sampling_kwargs["stop"] = [
                stop for stop in stop_sequences if stop and stop != "<|im_end|>"
            ]
        outputs = self._model.chat(
            messages,
            sampling_params=SamplingParams(**sampling_kwargs),
        )
        completion = outputs[0].outputs[0]
        text = getattr(completion, "text", "") or ""
        prompt_text = json.dumps(message_content, ensure_ascii=False)
        prompt_tokens = getattr(outputs[0], "prompt_token_ids", None)
        generated_tokens = getattr(completion, "token_ids", None)
        return self._result(
            text=text,
            prompt_text=prompt_text,
            prompt_token_count=(
                len(prompt_tokens) if prompt_tokens is not None else None
            ),
            generated_token_count=(
                len(generated_tokens) if generated_tokens is not None else None
            ),
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
        )


def open_eval_backend(
    model_path: str | Path,
    *,
    family: str | None = None,
    device_map: str | dict | None = None,
) -> Any:
    """Return an unloaded backend with ``generate_multimodal_with_metadata``.

    ``family="qwen"`` (the default detection for Qwen-VL paths) returns the
    existing :class:`HFBackend` so PhysBench / CausalPhys / ContPhy Qwen dumps
    keep their packing. Other families never enter Qwen ``video_metadata`` /
    ``do_sample_frames`` packing.
    """

    resolved = resolve_eval_family(model_path, family)
    if resolved == "qwen":
        if device_map is None:
            return HFBackend(str(model_path))
        return HFBackend(str(model_path), device_map=device_map)
    if resolved == "internvl":
        return InternVLFamilyBackend(model_path, device_map=device_map)
    if resolved == "llama32_vision":
        return LlamaVisionFamilyBackend(model_path, device_map=device_map)
    if resolved == "pixtral":
        return PixtralFamilyBackend(model_path, device_map=device_map)
    raise ValueError(f"unsupported eval family {resolved!r}")


__all__ = [
    "EVAL_FAMILIES",
    "INTERN_VL_IMAGE_MAX_NUM",
    "INTERN_VL_IMAGE_SIZE",
    "INTERN_VL_VIDEO_MAX_NUM",
    "InternVLFamilyBackend",
    "InternVLVisual",
    "LlamaVisionFamilyBackend",
    "PIXTRAL_MAX_IMAGES",
    "PixtralFamilyBackend",
    "assert_local_weights_ready",
    "count_visual_slots",
    "detect_eval_family",
    "ensure_hf_generate",
    "internvl_build_transform",
    "internvl_dynamic_preprocess",
    "internvl_find_closest_aspect_ratio",
    "internvl_question_from_content",
    "internvl_visual_plan",
    "llama_style_chat_content",
    "ndarray_to_pil",
    "open_eval_backend",
    "pil_to_data_uri",
    "pixtral_chat_content",
    "resolve_eval_family",
    "video_frames_from_entry",
]
