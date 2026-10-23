"""Golden scene 08 — projectile motion.

A sphere is launched from ``(-0.5, 0, 0.3)`` with linear velocity
``(1.5, 0, 2.0)`` m/s. Under default gravity it follows a ballistic
arc, bounces on the ground, and rolls to a stop near the +x edge of
the workspace. Exercises the initial-velocity seeding and the final-
state (FALLEN) classifier.
"""


def build_scene(api):
    api.add_sphere(
        "ball",
        pos=(-0.5, 0.0, 0.3),
        radius=0.05,
        mass=0.5,
        friction=1.0,
    )
    api.set_velocity("ball", linear=(1.5, 0.0, 2.0))
    api.set_duration(1.5)
