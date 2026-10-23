"""Fixture: two dynamic boxes connected by a damped spring (spatial tendon).

Exercises spring compilation: ``api.add_spring`` must land
as a ``<tendon><spatial/></tendon>`` sibling block after ``<worldbody>``,
with a ``<site>`` anchor inside each connected body.
"""

from __future__ import annotations

from physevolve.engine.physdsl import PhysDSLApi
from physevolve.engine.schema import SceneDef


def build_scene() -> SceneDef:
    api = PhysDSLApi()
    api.add_box(
        name="box_a",
        pos=(-0.3, 0.0, 0.5),
        size=(0.1, 0.1, 0.1),
        mass=1.0,
    )
    api.add_box(
        name="box_b",
        pos=(0.3, 0.0, 0.5),
        size=(0.1, 0.1, 0.1),
        mass=1.0,
    )
    api.add_spring(
        name="tether",
        body_a="box_a",
        body_b="box_b",
        stiffness=50.0,
        damping=1.5,
        anchor_a=(0.1, 0.0, 0.0),
        anchor_b=(-0.1, 0.0, 0.0),
        rest_length=0.4,
    )
    return api.freeze()
