"""Golden scene 07 — three dominoes knocked over by momentum.

Three tall thin boxes spaced 0.25 m apart along +x. The first domino
starts with +x linear velocity, tips over, and (via friction + contact)
knocks the second and third in turn. Exercises multi-body collision
cascading and the collision-aggregation window.
"""


def build_scene(api):
    api.add_box(
        "d1",
        pos=(0.0, 0.0, 0.15),
        size=(0.03, 0.08, 0.15),
        mass=0.5,
        friction=1.0,
    )
    api.add_box(
        "d2",
        pos=(0.25, 0.0, 0.15),
        size=(0.03, 0.08, 0.15),
        mass=0.5,
        friction=1.0,
    )
    api.add_box(
        "d3",
        pos=(0.5, 0.0, 0.15),
        size=(0.03, 0.08, 0.15),
        mass=0.5,
        friction=1.0,
    )
    api.set_velocity("d1", linear=(1.5, 0.0, 0.0))
    api.set_duration(2.0)
