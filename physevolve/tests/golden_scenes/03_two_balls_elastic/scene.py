"""Golden scene 03 — two balls head-on collision (momentum).

Ball A is at rest; ball B moves toward A at 1.5 m/s. The collision
transfers momentum from B to A; both end up further along +x after
rolling on the ground. Serves as the momentum-conservation smoke
test — at least one collision event must be recorded, and the trailing
ball (B) must NOT end up ahead of the leading ball (A).
"""


def build_scene(api):
    api.add_sphere(
        "a",
        pos=(0.3, 0.0, 0.15),
        radius=0.05,
        mass=1.0,
        friction=0.05,
    )
    api.add_sphere(
        "b",
        pos=(-0.3, 0.0, 0.15),
        radius=0.05,
        mass=1.0,
        friction=0.05,
    )
    api.set_velocity("b", linear=(1.5, 0.0, 0.0))
    api.set_duration(1.0)
