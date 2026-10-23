"""Golden scene 02 — ball rolling on inclined plane (friction test).

A sphere placed at the high end of a 20 deg inclined ramp rolls down
under gravity, leaves the ramp, and rolls along the ground. The
friction coefficient determines how far it travels before coming to
rest, so this scene exercises the friction pipeline end-to-end.
"""


def build_scene(api):
    # 20 deg tilt around the y-axis, encoded as a quaternion.
    # sin(10 deg) ~= 0.17365, cos(10 deg) ~= 0.98481.
    api.add_box(
        "ramp",
        pos=(-0.5, 0.0, 0.3),
        size=(0.25, 0.2, 0.02),
        mass=1.0,
        fixed=True,
    )
    api.set_rotation("ramp", (0.98481, 0.0, 0.17365, 0.0))

    api.add_sphere(
        "ball",
        pos=(-0.55, 0.0, 0.4),
        radius=0.04,
        mass=0.5,
        friction=2.0,
    )
    api.set_duration(1.2)
