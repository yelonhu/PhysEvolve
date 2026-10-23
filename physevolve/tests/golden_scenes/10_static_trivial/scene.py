"""Golden scene 10 — static scene with no dynamic bodies.

A single fixed block. No dynamic bodies, no initial velocities, no
collisions. The pipeline must classify this as
:attr:`FailureReason.TRIVIAL` — that is the whole point of the
the static-scene criterion.
"""


def build_scene(api):
    api.add_box(
        "wall",
        pos=(0.0, 0.0, 0.1),
        size=(0.1, 0.1, 0.1),
        mass=1.0,
        fixed=True,
    )
    api.set_duration(0.5)
