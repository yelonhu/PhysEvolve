"""Boundary modules between PhysEvolve and external runtimes.

The active training path imports the video bridge symbols below; no
other runtime adapter is needed once the PhysEvolve-native veRL trainer in
``physevolve/training/`` is used. Imports are lazy so probing-only
environments (no ``verl``/``vllm``) can still load this package.
"""

from __future__ import annotations

import importlib
from typing import Any


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "VideoBundle": (
        "physevolve.integrations.video_bridge",
        "VideoBundle",
    ),
    "sim_state_to_vllm_video": (
        "physevolve.integrations.video_bridge",
        "sim_state_to_vllm_video",
    ),
    "video_bundle_from_observation": (
        "physevolve.integrations.video_bridge",
        "video_bundle_from_observation",
    ),
    "VLLM_VIDEO_METADATA_KEYS": (
        "physevolve.integrations.video_bridge",
        "VLLM_VIDEO_METADATA_KEYS",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial introspection.
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))


__all__ = list(_LAZY_EXPORTS.keys())
