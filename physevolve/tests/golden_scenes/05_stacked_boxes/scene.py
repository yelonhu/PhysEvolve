"""Golden scene 05 — two stacked boxes settle stably.

Two boxes are placed centred on the z-axis: a large base box and a
smaller upper box. After a short duration under default gravity both
bodies settle in place (no collisions beyond the initial seating).
Tests the ``STABLE`` classifier path and contact-stacking stability.
"""


def build_scene(api):
    api.add_box(
        "lower",
        pos=(0.0, 0.0, 0.1),
        size=(0.15, 0.15, 0.1),
        mass=5.0,
        friction=1.0,
    )
    api.add_box(
        "upper",
        pos=(0.0, 0.0, 0.32),
        size=(0.1, 0.1, 0.1),
        mass=1.0,
        friction=1.0,
    )
    api.set_duration(0.8)
