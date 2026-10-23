"""Turn a :class:`PhysBenchItem` into an :class:`HFBackend`-ready content list.

PhysBench questions embed ``<video>`` and ``<image>`` tags inline. The
tags map positionally onto ``item.file_names``: the first ``<video>`` or
``<image>`` consumes the first filename, the second consumes the second,
and so on. To match Qwen3-VL's chat-template conventions we must emit a
content list whose entries are ordered identically to those tags.

Example:
    question = "Following the content of the <video>, which of A. <image>"
               " B. <image> C. <image> D. <image> happens first?"
    file_names = [v.mp4, a.png, b.png, c.png, d.png]

    → content = [
        {"type": "text",  "text": "Following the content of the "},
        {"type": "video", "video": <array>},
        {"type": "text",  "text": ", which of A. "},
        {"type": "image", "image": <array>},
        {"type": "text",  "text": " B. "},
        {"type": "image", "image": <array>},
        ...
        {"type": "text",  "text": " happens first?<END_PROMPT>"},
      ]

The final text chunk has the benchmark's canonical end-prompt appended
so the model knows to answer with a single A/B/C/D letter.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data_loader import PhysBenchItem
from .visual_loader import load_image, load_video


# ─── Constants ───────────────────────────────────────────────────────


END_PROMPT: str = (
    "\nAnswer with the option's letter from the given choices directly. "
    "You can only answer one letter from A, B, C, or D."
)
"""Canonical PhysBench end-prompt. Transcribed from upstream
``eval/eval_utils/task_evaluator.py``; the string is part of the
benchmark specification rather than evaluation code and is reproduced
verbatim so the model sees the exact instruction used in the leaderboard.
"""


ANSWER_ONLY_SUFFIX: str = (
    "\nRespond with only the answer letter inside <answer>...</answer> tags. "
    "Do not reason. Output format: <answer>X</answer> where X is A, B, C, or D."
)
"""Suffix appended in ``answer_only`` mode to discourage a ``<think>``
chain. The runner additionally supplies an ``<answer>`` assistant
prefill so the model continues from inside the answer tag."""

ANSWER_ONLY_PREFILL: str = "<answer>"
"""Assistant-turn prefill used by ``answer_only`` mode."""

_TAG_RE = re.compile(r"<(video|image)>", re.IGNORECASE)


# ─── Content assembly ────────────────────────────────────────────────


def build_content(
    item: PhysBenchItem,
    resolved_paths: Sequence[Path],
    *,
    n_video_frames: int = 8,
    end_prompt: str = END_PROMPT,
    answer_only: bool = False,
) -> list[dict[str, Any]]:
    """Produce a chat-template content list for ``item``.

    Args:
        item: Loaded PhysBench question.
        resolved_paths: Absolute paths to the files named in
            ``item.file_names``, in the same order. Use
            :func:`data_loader.resolve_visual_paths` to produce them.
        n_video_frames: Frames to sample per video. Defaults to ``8``,
            the upstream benchmark default.
        end_prompt: The suffix appended to the final text chunk. The
            default matches the upstream MC-answer instruction and is
            what yields leaderboard-comparable numbers.

    Returns:
        A content list suitable for
        :meth:`HFBackend.generate_multimodal`.

    Raises:
        ValueError: if the number of inline tags does not match the number
            of resolved paths, or if a tag's kind disagrees with the
            extension of the corresponding file.
    """

    paths = list(resolved_paths)
    tag_spans = [(m.start(), m.end(), m.group(1).lower())
                 for m in _TAG_RE.finditer(item.question)]

    if len(tag_spans) != len(paths):
        raise ValueError(
            f"item idx={item.idx}: question has {len(tag_spans)} visual "
            f"tags but {len(paths)} file paths were provided"
        )

    content: list[dict[str, Any]] = []
    cursor = 0
    for (start, end, kind), path in zip(tag_spans, paths):
        # Emit the text leading up to this tag, if any.
        if start > cursor:
            chunk = item.question[cursor:start]
            if chunk:
                content.append({"type": "text", "text": chunk})

        # Emit the visual chunk; verify kind matches the file extension.
        ext = path.suffix.lower()
        if kind == "video":
            if ext != ".mp4":
                raise ValueError(
                    f"item idx={item.idx}: <video> tag maps to non-.mp4 "
                    f"file {path.name}"
                )
            video = load_video(path, n_frames=n_video_frames)
            content.append({"type": "video", "video": video})
        else:  # "image"
            if ext not in {".jpg", ".jpeg", ".png"}:
                raise ValueError(
                    f"item idx={item.idx}: <image> tag maps to non-image "
                    f"file {path.name}"
                )
            image = load_image(path)
            content.append({"type": "image", "image": image})
        cursor = end

    # Tail text after the last tag (plus the end prompt). When the
    # caller selects ``answer_only`` mode an additional suffix is
    # appended that asks the model to skip the ``<think>`` chain; the
    # actual ``<answer>`` prefill is plumbed in via the backend so the
    # assistant turn starts with ``<answer>`` already opened.
    tail = item.question[cursor:]
    tail_text = tail + end_prompt
    if answer_only:
        tail_text = tail_text + ANSWER_ONLY_SUFFIX
    if tail_text:
        content.append({"type": "text", "text": tail_text})

    # Guarantee at least one text chunk so the model sees the end prompt
    # even on the (edge-case) question that ends on a visual tag with no
    # trailing characters.
    if not any(c.get("type") == "text" for c in content):
        content.append({"type": "text", "text": end_prompt})

    return content


# ─── Introspection helpers (used by tests + runner logging) ─────────


def count_tags(item: PhysBenchItem) -> tuple[int, int]:
    """Return ``(n_video_tags, n_image_tags)`` in ``item.question``."""

    n_video = 0
    n_image = 0
    for m in _TAG_RE.finditer(item.question):
        if m.group(1).lower() == "video":
            n_video += 1
        else:
            n_image += 1
    return n_video, n_image


__all__ = [
    "ANSWER_ONLY_PREFILL",
    "ANSWER_ONLY_SUFFIX",
    "END_PROMPT",
    "build_content",
    "count_tags",
]
