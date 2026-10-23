"""Qwen3-VL adapter for the vendored official PhysBench evaluator."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

from evaluation.common.family_generate import (
    open_eval_backend,
    resolve_eval_family,
)

from .visual_loader import load_image, load_video


_TAG_RE = re.compile(r"<(video|image)>", re.IGNORECASE)
_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_LEADING_MC_RE = re.compile(r"^\s*\(?\s*([ABCD])\b", re.IGNORECASE)
# Instruct often returns the
# letter inside prose, e.g. "The answer is **B**." or "Looking at the
# video, C is correct.". The leading-letter parser misses those shapes;
# this is the substring fallback used after every leading attempt has
# returned None. Mirrors physevolve.inference.parsing.extract_mc_letter
# so eval / training stay aligned.
_EMBEDDED_MC_RE = re.compile(r"\b([ABCD])\b")


class QwenPhysBenchAdapter:
    """Expose the official ``qa(image, prompt, mode)`` interface.

    The official PhysBench evaluator owns dataset iteration, prompt suffixing,
    visual path resolution, result JSON writing, and scoring. This adapter
    converts those official inputs into the shared content list, runs local
    inference through :func:`~evaluation.common.family_generate.open_eval_backend`,
    and returns a scorer-compatible answer string. Qwen-VL still uses
    :class:`~physevolve.inference.hf_backend.HFBackend`; InternVL / Llama /
    Pixtral use per-family generate paths.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_video_frames: int = 8,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        audit_file: str | Path | None = None,
        model_family: str | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.n_video_frames = n_video_frames
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.audit_file = Path(audit_file) if audit_file is not None else None
        self.model_family = resolve_eval_family(self.model_path, model_family)
        self.backend = open_eval_backend(
            self.model_path, family=self.model_family
        )
        self._current_item: dict[str, Any] | None = None
        self._current_prompt: str | None = None
        self._current_visuals: list[str] | None = None

    def __enter__(self) -> "QwenPhysBenchAdapter":
        self.backend.load()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.backend.close()

    def set_current_item(
        self,
        *,
        item: dict[str, Any],
        prompt: str,
        visuals: Sequence[str],
    ) -> None:
        """Receive optional per-item context from the vendored evaluator."""

        self._current_item = dict(item)
        self._current_prompt = prompt
        self._current_visuals = list(visuals)

    def qa(
        self,
        image: str | Sequence[str],
        prompt: str,
        mode: str | None = None,
    ) -> str | None:
        """Run one official PhysBench item and return an answer string."""

        visuals = [image] if isinstance(image, str) else list(image)
        content = _build_qwen_content(
            prompt=prompt,
            visual_paths=visuals,
            n_video_frames=self.n_video_frames,
        )
        t0 = time.perf_counter()
        result = self.backend.generate_multimodal_with_metadata(
            content,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop_sequences=["</answer>", "<|im_end|>"],
        )
        elapsed_s = time.perf_counter() - t0
        answer = parse_physbench_mc_answer(result.text)
        if self.audit_file is not None:
            self._append_audit(
                mode=mode,
                answer=answer,
                raw_text=result.text,
                elapsed_s=elapsed_s,
                metadata=result.metadata,
                prompt_tail=result.prompt_tail,
                visuals=visuals,
            )
        return answer if answer is not None else result.text.strip()

    def _append_audit(
        self,
        *,
        mode: str | None,
        answer: str | None,
        raw_text: str,
        elapsed_s: float,
        metadata: dict[str, Any],
        prompt_tail: str,
        visuals: Sequence[str],
    ) -> None:
        assert self.audit_file is not None
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "idx": None if self._current_item is None else self._current_item.get("idx"),
            "mode": mode,
            "answer": answer,
            "raw_text": raw_text,
            "elapsed_s": elapsed_s,
            "metadata": metadata,
            "prompt_tail": prompt_tail,
            "visuals": list(visuals),
        }
        with self.audit_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_physbench_mc_answer(raw_text: str) -> str | None:
    """Strict PhysBench MC parser for adapter outputs.

    Supported output shapes (in priority order):

    1. PhysEvolve-trained / prompted format: ``<answer>A</answer>``.
    2. Native Qwen3-VL-Thinking format:
       ``<think>...</think>\n\nA<|im_end|>``.
    3. Instruct terse format: a bare leading ``A`` / ``B`` / ``C`` / ``D``
       (e.g. ``"A<|im_end|>"`` or ``"(B)"``).
    4. Qwen3-VL Instruct prose fallback: if every leading-style attempt above
       returns None, search for the first stand-alone ``A``-``D`` token in
       the answer body via :data:`_EMBEDDED_MC_RE`. This recovers shapes
       like ``"The answer is **B**.<|im_end|>"`` that the Instruct base
       commonly emits without an ``<answer>`` wrapper. The fallback only
       triggers when leading parsing has failed, so existing positive
       cases keep their current verdict.

    We still never search inside ``<think>``. This avoids treating prose
    such as "Option A ..." inside the reasoning trace as the final answer.
    """

    block = _extract_answer_block(raw_text)
    if block is not None:
        leading = _leading_mc_letter(block)
        if leading is not None:
            return leading
        # Instruct fallback: <answer> body without a leading letter.
        return _embedded_mc_letter(block)
    post_think = _text_after_last_think(raw_text)
    if post_think is not None:
        tail = _strip_qwen_special_tail(post_think)
        leading = _leading_mc_letter(tail)
        if leading is not None:
            return leading
        return _embedded_mc_letter(tail)
    leading = _leading_mc_letter(raw_text)
    if leading is not None:
        return leading
    # Instruct fallback: plain Instruct prose, no <answer>, no </think>.
    return _embedded_mc_letter(_strip_qwen_special_tail(raw_text))


def _extract_answer_block(text: str) -> str | None:
    match = _ANSWER_BLOCK_RE.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def _text_after_last_think(text: str) -> str | None:
    matches = list(_THINK_CLOSE_RE.finditer(text))
    if not matches:
        return None
    return text[matches[-1].end() :].strip()


def _strip_qwen_special_tail(text: str) -> str:
    return text.replace("<|im_end|>", "").strip()


def _leading_mc_letter(text: str) -> str | None:
    match = _LEADING_MC_RE.match(text.strip())
    if match is None:
        return None
    return match.group(1).upper()


def _embedded_mc_letter(text: str) -> str | None:
    """Return the first stand-alone A/B/C/D token in ``text``.

    Used by :func:`parse_physbench_mc_answer` as an Instruct fallback after
    every leading-letter attempt has failed. The regex is identical to
    :data:`physevolve.inference.parsing._MC_LETTER_RE` so PhysBench eval
    and PhysEvolve training share one MC-extraction contract.
    """

    if not text:
        return None
    match = _EMBEDDED_MC_RE.search(text)
    if match is None:
        return None
    return match.group(1).upper()


def _build_qwen_content(
    *,
    prompt: str,
    visual_paths: Sequence[str | Path],
    n_video_frames: int,
) -> list[dict[str, Any]]:
    """Convert official prompt/path inputs to Qwen3-VL content entries."""

    paths = [Path(p) for p in visual_paths]
    tag_spans = [
        (m.start(), m.end(), m.group(1).lower()) for m in _TAG_RE.finditer(prompt)
    ]
    if len(tag_spans) != len(paths):
        raise ValueError(
            f"prompt has {len(tag_spans)} visual tags but {len(paths)} paths"
        )

    content: list[dict[str, Any]] = []
    cursor = 0
    for (start, end, kind), path in zip(tag_spans, paths):
        if start > cursor:
            chunk = prompt[cursor:start]
            if chunk:
                content.append({"type": "text", "text": chunk})
        ext = path.suffix.lower()
        if kind == "video":
            if ext != ".mp4":
                raise ValueError(f"<video> tag maps to non-video path: {path}")
            content.append(
                {
                    "type": "video",
                    "video": load_video(path, n_frames=n_video_frames),
                }
            )
        else:
            if ext not in {".jpg", ".jpeg", ".png"}:
                raise ValueError(f"<image> tag maps to non-image path: {path}")
            content.append({"type": "image", "image": load_image(path)})
        cursor = end

    tail = prompt[cursor:]
    if tail:
        content.append({"type": "text", "text": tail})
    return content


__all__ = [
    "QwenPhysBenchAdapter",
    "parse_physbench_mc_answer",
]
