"""Fixture: fixed anchor + dynamic bob linked by a hinge; non-default gravity."""

from __future__ import annotations

from physevolve.engine.physdsl import PhysDSLApi
from physevolve.engine.schema import SceneDef


def build_scene() -> SceneDef:
    api = PhysDSLApi()
    api.set_gravity((0.0, 0.0, -3.7))
    api.add_box(
        name="anchor",
        pos=(0.0, 0.0, 1.5),
        size=(0.05, 0.05, 0.05),
        mass=1.0,
        fixed=True,
    )
    api.add_sphere(
        name="bob",
        pos=(0.3, 0.0, 1.5),
        radius=0.08,
        mass=0.5,
    )
    api.add_hinge(
        name="pivot",
        parent="anchor",
        child="bob",
        pos=(0.0, 0.0, 1.5),
        axis=(0.0, 1.0, 0.0),
        limit=(-1.5, 1.5),
    )
    return api.freeze()
