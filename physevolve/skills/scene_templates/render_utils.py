"""Shared color naming, option shuffling, and text formatting for scene templates.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping


_VQA_LETTERS: tuple[str, ...] = ("A", "B", "C", "D")


def option_letter_for_idx(idx: int) -> str:
    """Return ``"A" / "B" / "C" / "D"`` for an option index in ``[0, 3]``."""

    if not isinstance(idx, int) or idx < 0 or idx >= 4:
        raise ValueError(f"option index {idx!r} must be in [0, 3]")
    return _VQA_LETTERS[idx]


def prettify(text: str) -> str:
    """Replace body-name underscores with spaces for human-readable VQA text.

    No template uses underscores in legitimate display text, so a blanket
    ``_`` -> ` ` substitution safely turns ``red_ball`` / ``gray_target`` into
    ``red ball`` / ``gray target`` without changing semantics. The ground-truth
    extractors always run on the real (underscored) names, so this is purely
    cosmetic and never affects ``correct_idx``.
    """

    return text.replace("_", " ")


def shuffle_options_deterministic(
    options: list[str],
    correct_idx: int,
    *,
    seed_key: tuple[str, str, Mapping[str, object]],
    salt: str = "",
) -> tuple[list[str], int]:
    """Permute ``options`` by a content-seeded RNG and remap ``correct_idx``.

    The seed is stable across processes (``hashlib.md5`` over a canonical
    string), unlike the salted builtin ``hash`` of ``str``. ``salt`` (the
    executor passes ``scene_id``) mixes in per-instance entropy so repeated
    rollouts of the same question still spread across letters.

    Args:
        options: The 4 option strings in their canonical order.
        correct_idx: Index of the correct option before shuffling.
        seed_key: ``(key, phrasing, params)`` content that identifies the
            question; ``key`` is typically ``"{scene_kind}:{family}"``.
        salt: Extra entropy (the executor passes ``scene_id``).

    Returns:
        ``(shuffled_options, new_correct_idx)``.
    """

    key, phrasing, params = seed_key
    sig = f"{salt}|{key}|{phrasing}|{canonical_params_signature(params)}"
    seed = int(hashlib.md5(sig.encode("utf-8")).hexdigest(), 16) % (2**32)
    order = list(range(len(options)))
    random.Random(seed).shuffle(order)
    shuffled = [options[i] for i in order]
    new_correct = order.index(int(correct_idx))
    return shuffled, new_correct


def canonical_params_signature(params: Mapping[str, object]) -> str:
    """Return a stable string summarising ``params`` for novelty hashing.

    Sorted-key, ``repr``-based serialisation. Not cryptographic — only needs
    to be deterministic across runs so the novelty buffer treats "same scene,
    same params" as a duplicate.
    """

    return _canonicalise_repr(params)


def _canonicalise_repr(value: Any) -> str:
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: kv[0])
        body = ",".join(f"{k!r}:{_canonicalise_repr(v)}" for k, v in items)
        return "{" + body + "}"
    if isinstance(value, list):
        return "[" + ",".join(_canonicalise_repr(v) for v in value) + "]"
    return repr(value)


__all__ = [
    "canonical_params_signature",
    "option_letter_for_idx",
    "prettify",
    "shuffle_options_deterministic",
]
