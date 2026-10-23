"""Fixture: dynamic bob hinged directly to the worldbody via parent="world".

Exercises the ``parent="world"`` shortcut in
:func:`physevolve.engine.physdsl.PhysDSLApi.add_hinge`. The bob must
remain at the ``<worldbody>`` level (not nested inside another body)
and emit its hinge joint inline instead of the default ``<freejoint/>``.
"""

from __future__ import annotations

from physevolve.engine.physdsl import PhysDSLApi
from physevolve.engine.schema import SceneDef


def build_scene() -> SceneDef:
    api = PhysDSLApi()
    api.set_gravity((0.0, 0.0, -9.81))
    api.add_sphere(
        name="bob",
        pos=(0.3, 0.0, 1.5),
        radius=0.08,
        mass=0.5,
    )
    api.add_hinge(
        name="pivot",
        parent="world",
        child="bob",
        pos=(0.0, 0.0, 1.5),
        axis=(0.0, 1.0, 0.0),
        limit=(-1.5, 1.5),
    )
    return api.freeze()
