"""Golden scene 09 — block slides off a ledge (triviality edge case).

A block's centre of mass is placed past the edge of a fixed platform;
torque from gravity tips it off. The block lands on the ground and
settles. This scene MUST NOT be classified as TRIVIAL — displacement
is well above the 5 cm threshold, even though the final status is
``STABLE`` (low final velocity on the ground).
"""


def build_scene(api):
    api.add_box(
        "platform",
        pos=(0.0, 0.0, 0.2),
        size=(0.3, 0.3, 0.02),
        mass=1.0,
        fixed=True,
    )
    api.add_box(
        "block",
        pos=(0.35, 0.0, 0.27),
        size=(0.08, 0.08, 0.05),
        mass=1.0,
        friction=0.5,
    )
    api.set_duration(1.2)
