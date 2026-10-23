"""Golden scene 04 — two balls inelastic-ish collision (energy dissipation).

Same geometry as scene 03 but with high friction (0.5 vs 0.05) so
friction dissipates kinetic energy between the collision and the end
of the simulation. Compared with scene 03, both balls travel a
smaller distance.
"""


def build_scene(api):
    api.add_sphere(
        "a",
        pos=(0.3, 0.0, 0.15),
        radius=0.05,
        mass=1.0,
        friction=0.5,
    )
    api.add_sphere(
        "b",
        pos=(-0.3, 0.0, 0.15),
        radius=0.05,
        mass=1.0,
        friction=0.5,
    )
    api.set_velocity("b", linear=(1.5, 0.0, 0.0))
    api.set_duration(1.0)
