"""Fixture: fixed platform + dynamic box resting above it."""

from __future__ import annotations

from physevolve.engine.physdsl import PhysDSLApi
from physevolve.engine.schema import SceneDef


def build_scene() -> SceneDef:
    api = PhysDSLApi()
    api.add_box(
        name="platform",
        pos=(0.0, 0.0, 0.3),
        size=(0.3, 0.3, 0.02),
        mass=1.0,
        rgba=(0.5, 0.5, 0.8, 1.0),
        fixed=True,
    )
    api.add_box(
        name="cargo",
        pos=(0.0, 0.0, 0.5),
        size=(0.1, 0.1, 0.1),
        mass=0.5,
        rgba=(0.8, 0.6, 0.2, 1.0),
    )
    return api.freeze()
