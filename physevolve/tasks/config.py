"""Task-mode weights loaded from ``configs/executor.yaml``.

Module defaults are used when the configuration file is unavailable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

import yaml

# configs/ lives at the repo root; this file is at
# ``physevolve/tasks/config.py`` → three parents → repo root.
_CONFIG_PATH: Path = Path(__file__).resolve().parents[2] / "configs" / "executor.yaml"


_DEFAULT_TASK_WEIGHTS: dict[str, float] = {
    "forward": 0.40,
    "inverse": 0.30,
    "parameter": 0.30,
}


@lru_cache(maxsize=1)
def load_task_weights() -> Mapping[str, float]:
    """Return ``{mode_name: weight}`` from YAML; falls back to defaults."""

    if not _CONFIG_PATH.exists():
        return dict(_DEFAULT_TASK_WEIGHTS)
    with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        return dict(_DEFAULT_TASK_WEIGHTS)
    raw = loaded.get("task_weights")
    if not isinstance(raw, dict):
        return dict(_DEFAULT_TASK_WEIGHTS)
    return {
        "forward": float(raw.get("forward", _DEFAULT_TASK_WEIGHTS["forward"])),
        "inverse": float(raw.get("inverse", _DEFAULT_TASK_WEIGHTS["inverse"])),
        "parameter": float(
            raw.get("parameter", _DEFAULT_TASK_WEIGHTS["parameter"])
        ),
    }


__all__ = ["load_task_weights"]
