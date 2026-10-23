"""Lazy loader for ``configs/executor.yaml``.

The pipeline imports :func:`load_executor_config` on demand and treats
every field as a *default* that an explicit keyword argument can
override. The file is parsed once per process and cached.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

# configs/ lives at the repo root. This file is at
# ``physevolve/executor/config.py`` → three parents → repo root.
_CONFIG_PATH: Path = Path(__file__).resolve().parents[2] / "configs" / "executor.yaml"


_FALLBACK_CONFIG: dict[str, Any] = {
    "timeouts": {
        "sandbox_timeout_s": 5.0,
        "sim_timeout_s": 30.0,
        "render_timeout_s": 15.0,
    },
    "simulation": {
        "timestep": 0.002,
        "default_duration": 8.0,
        "n_frames": 32,
        "default_gravity": [0.0, 0.0, -9.81],
    },
    "rendering": {
        "height": 480,
        "width": 480,
        "n_cameras": 1,
        "default_camera_pos": [1.8, -1.8, 1.8],
        "default_camera_target": [0.0, 0.0, 0.5],
    },
    "task_weights": {
        "forward": 0.40,
        "inverse": 0.30,
        "parameter": 0.30,
    },
    "triviality": {
        "max_displacement_m": 0.05,
        "require_zero_collisions": True,
    },
}


@lru_cache(maxsize=1)
def load_executor_config() -> Mapping[str, Any]:
    """Return the parsed ``executor.yaml`` (cached).

    Falls back to the hard-coded :data:`_FALLBACK_CONFIG` if the file is
    missing, such as an installation without the repository's ``configs/``
    directory. Keep the fallback values synchronized with the YAML file.
    """

    if not _CONFIG_PATH.exists():
        return _FALLBACK_CONFIG
    with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        return _FALLBACK_CONFIG
    return loaded


__all__ = ["load_executor_config"]
