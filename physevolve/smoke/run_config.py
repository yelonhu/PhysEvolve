"""Configuration objects and run-id helpers for smoke runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence


DEFAULT_PROMPT_VERSION = "proposer_stage1_v1"


@dataclass(frozen=True, slots=True)
class SmokeGenerationConfig:
    """Generation parameters shared by smoke scripts."""

    max_new_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    stop_sequences: tuple[str, ...] = ("</propose>",)


def build_run_id(
    *,
    gate: str,
    model: str,
    prompt_version: str,
    n_samples: int,
    now: Optional[datetime] = None,
) -> str:
    """Build a smoke run id in the canonical timestamped format."""

    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M")
    return (
        f"{stamp}_{_slug(gate)}_{model_short_name(model)}_"
        f"{_slug(prompt_version)}_n{int(n_samples)}"
    )


def model_short_name(model: str) -> str:
    """Return a compact filesystem-safe model label."""

    name = Path(str(model).rstrip("/")).name or str(model)
    lower = name.lower()
    replacements: Sequence[tuple[str, str]] = (
        ("qwen3-vl-8b-thinking", "qwen3vl8bthinking"),
        ("qwen3_vl_8b_thinking", "qwen3vl8bthinking"),
        ("qwen3-vl-8b-instruct", "qwen3vl8binstruct"),
        ("qwen3_vl_8b_instruct", "qwen3vl8binstruct"),
    )
    for old, new in replacements:
        lower = lower.replace(old, new)
    return _slug(lower)


def _slug(value: str) -> str:
    chars: list[str] = []
    prev_sep = False
    for ch in value.lower():
        if ch.isalnum():
            chars.append(ch)
            prev_sep = False
        elif not prev_sep:
            chars.append("_")
            prev_sep = True
    return "".join(chars).strip("_") or "unknown"


__all__ = [
    "DEFAULT_PROMPT_VERSION",
    "SmokeGenerationConfig",
    "build_run_id",
    "model_short_name",
]
