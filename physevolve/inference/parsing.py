"""Tolerant text extractors for Solver / Proposer outputs.

All four functions return ``None`` on malformed input rather than
raising. The intended call site (reward computation during RL rollouts)
receives many malformed strings per step, and raising there would force
every caller to guard individually.

Tag conventions:

* Solver output: ``<think>...</think><answer>...</answer>``.
* Proposer output: ``<think>...</think><propose>...</propose>``.

Robustness rules:

* Case-insensitive tag matching.
* DOTALL: body may span lines.
* First occurrence wins when multiple tags appear (the Proposer / Solver
  sometimes emits a second ``<answer>`` block inside a ``<think>``
  discussion; we trust the spec and take the first well-formed one).
* ``extract_json_from_answer`` first tries ``json.loads``; on failure it
  falls back to ``json5.loads`` (tolerant of trailing commas, single
  quotes, comments).
* vLLM may omit the stop string from generated text. If generation stops
  on ``</propose>``, the Proposer extractor accepts content through EOF.
  Solver answers are stricter: EOF fallback is accepted only for bodies
  that look like Stage-1 JSON answers or Stage-2 A/B/C/D choices.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

try:  # pragma: no cover — trivial import path.
    import json5  # type: ignore
except ImportError:  # pragma: no cover
    json5 = None  # type: ignore


# ─── Tag patterns ────────────────────────────────────────────────────


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_ANSWER_OPEN_RE = re.compile(r"<answer>", re.IGNORECASE)
_PROPOSE_RE = re.compile(r"<propose>(.*?)</propose>", re.IGNORECASE | re.DOTALL)
_PROPOSE_OPEN_RE = re.compile(r"<propose>", re.IGNORECASE)


def extract_answer_block(text: str) -> Optional[str]:
    """Return the inner text of the first ``<answer>...</answer>`` block.

    Returns ``None`` if no well-formed block is found. Whitespace around
    the captured content is stripped.
    """

    if not isinstance(text, str):
        return None
    return _extract_tag_block(
        text,
        closed_pattern=_ANSWER_RE,
        open_pattern=_ANSWER_OPEN_RE,
        eof_validator=_looks_like_answer_body,
    )


def extract_propose_block(text: str) -> Optional[str]:
    """Return the inner text of the first ``<propose>...</propose>`` block."""

    return _extract_tag_block(
        text,
        closed_pattern=_PROPOSE_RE,
        open_pattern=_PROPOSE_OPEN_RE,
        eof_validator=None,
    )


def _extract_tag_block(
    text: str,
    *,
    closed_pattern: re.Pattern[str],
    open_pattern: re.Pattern[str],
    eof_validator: Callable[[str], bool] | None = None,
) -> Optional[str]:
    """Extract a tagged block, accepting EOF when vLLM swallowed the stop.

    The well-formed ``<tag>...</tag>`` form wins. If the closing tag is
    absent but an opening tag exists, return everything after the first
    opening tag only if the optional EOF validator accepts it. Passing no
    validator preserves the raw EOF fallback used for Proposer code.
    """

    if not isinstance(text, str):
        return None
    closed = closed_pattern.search(text)
    if closed is not None:
        return closed.group(1).strip()
    opened = open_pattern.search(text)
    if opened is None:
        return None
    body = text[opened.end():].strip()
    if eof_validator is not None and not eof_validator(body):
        return None
    return body


def _looks_like_answer_body(body: str) -> bool:
    """Return whether an unclosed answer body is plausible model output."""

    stripped = body.strip()
    if not stripped:
        return False
    if _MC_LETTER_RE.fullmatch(stripped):
        return True
    return stripped.startswith("{") and _balanced_braces(stripped)


def _balanced_braces(text: str) -> bool:
    """Best-effort JSON object boundary check for swallowed stop tags."""

    depth = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "\"":
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not in_string


# ─── JSON extraction ─────────────────────────────────────────────────


def extract_json_from_answer(text: str) -> Optional[dict]:
    """Pull ``<answer>`` and parse its contents as JSON.

    Tries ``json.loads`` first; falls back to ``json5.loads`` for
    tolerance of trailing commas / single quotes / comments when the
    ``json5`` package is importable.

    Only ``dict`` values are returned — lists, strings, and scalars are
    rejected with ``None`` (the Stage-1 Solver prediction schemas are
    all top-level objects keyed by body name).
    """

    body = extract_answer_block(text)
    if body is None:
        return None

    # 1. Strict JSON first.
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        payload = None

    # 2. json5 fallback for trailing commas etc.
    if payload is None and json5 is not None:
        try:
            payload = json5.loads(body)
        except (ValueError, Exception):  # noqa: BLE001
            payload = None

    if not isinstance(payload, dict):
        return None
    return payload


# ─── Multiple-choice letter extraction (Stage 2) ─────────────────────


_MC_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def extract_mc_letter(text: str) -> Optional[str]:
    """Extract a single A/B/C/D letter from ``<answer>``.

    Behaviour:

    * First the ``<answer>`` block is pulled. If absent, returns ``None``.
    * The block is searched for the first stand-alone ``A``/``B``/``C``/``D``
      letter (case-insensitive, word-boundary-delimited).
    * Returns the uppercase letter or ``None``.

    The word-boundary match avoids false positives on words that happen
    to contain A/B/C/D adjacent to other alphanumerics.
    """

    body = extract_answer_block(text)
    if body is None:
        return None
    m = _MC_LETTER_RE.search(body)
    if m is None:
        return None
    return m.group(1).upper()


__all__ = [
    "extract_answer_block",
    "extract_propose_block",
    "extract_json_from_answer",
    "extract_mc_letter",
]
