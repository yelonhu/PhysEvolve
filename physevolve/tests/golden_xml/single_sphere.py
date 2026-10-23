"""Fixture: one dynamic sphere + auto-added ground plane + default camera."""

from __future__ import annotations

from physevolve.engine.physdsl import PhysDSLApi
from physevolve.engine.schema import SceneDef


def build_scene() -> SceneDef:
    api = PhysDSLApi()
    api.add_sphere(
        name="ball",
        pos=(0.0, 0.0, 1.0),
        radius=0.1,
        mass=1.0,
        rgba=(0.8, 0.2, 0.2, 1.0),
    )
    return api.freeze()
