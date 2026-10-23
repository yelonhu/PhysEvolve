"""Golden scene: motion_lab

Source: physevolve/skills/scene_templates/motion_lab.py
Families: motion_state_change, friction_or_restitution_identification
Plan: a low-friction green ball rolls down a fixed ramp under gravity, strikes
an amber slider at the dock entrance, and transfers motion to a high-friction
blue ball that only nudges forward near the stop wall. The same connected
apparatus supports both the green ball motion-change question and the
blue-vs-green friction comparison.
"""


def build_scene(api):
    api.add_plane(
        name="ground",
        pos=(0.0, 0.0, 0.0),
        size=(5.0, 5.0),
        rgba=(0.65, 0.65, 0.65, 1.0),
        friction=0.5,
    )

    # Flush contact surfaces: the wood deck is split around the rough pad so
    # the high-friction patch is embedded instead of raised above the table.
    api.add_box(
        "deck_west",
        pos=(-0.2725, 0.0, 0.010),
        size=(0.4275, 0.30, 0.010),
        mass=10.0,
        rgba=(0.42, 0.32, 0.22, 1.0),
        friction=0.3,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "ramp_surface",
        pos=(-0.95, 0.0, 0.215),
        size=(0.72, 0.22, 0.025),
        mass=10.0,
        rgba=(0.48, 0.36, 0.24, 1.0),
        friction=0.15,
        restitution=0.0,
        fixed=True,
    )
    api.set_rotation("ramp_surface", quaternion=(0.96593, 0, 0.25882, 0))

    api.add_box(
        "rough_pad",
        pos=(0.30, 0.0, 0.010),
        size=(0.145, 0.16, 0.010),
        mass=10.0,
        rgba=(0.18, 0.16, 0.14, 1.0),
        friction=0.3,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "deck_east",
        pos=(0.8725, 0.0, 0.010),
        size=(0.4275, 0.30, 0.010),
        mass=10.0,
        rgba=(0.42, 0.32, 0.22, 1.0),
        friction=0.15,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "dock_wall",
        pos=(1.29, 0.0, 0.120),
        size=(0.030, 0.20, 0.100),
        mass=10.0,
        rgba=(0.34, 0.37, 0.39, 1.0),
        friction=0.5,
        restitution=0.9,
        fixed=True,
    )

    api.add_sphere(
        "green_ball",
        pos=(-1.30, 0.0, 1.2),
        radius=0.09,
        mass=1.0,
        rgba=(0.22, 0.78, 0.32, 1.0),
        friction=0.05,
        restitution=0.9,
        fixed=False,
    )
    api.add_box(
        "amber_slider",
        pos=(-0.10, 0.0, 0.1),
        size=(0.105, 0.08, 0.09),
        mass=1.0,
        rgba=(0.95, 0.55, 0.15, 1.0),
        friction=0.05,
        restitution=0.9,
        fixed=False,
    )
    api.add_sphere(
        "blue_ball",
        pos=(0.70, 0.0, 0.110),
        radius=0.09,
        mass=1.2,
        rgba=(0.12, 0.28, 0.92, 1.0),
        friction=0.05,
        restitution=0.9,
        fixed=False,
    )

    # Down-ramp initial speed >= 0.10 m/s (30 deg ramp: +x, -z).
    api.set_velocity(
        "green_ball",
        linear=(0.10, 0.0, -0.16),
        angular=(0.0, 0.0, 0.0),
    )

    api.set_duration(3.5)
    api.add_camera("cam0", pos=(-0.20, -3.8, 0.75), target=(0.05, 0.0, 0.22))

SCENE_MANIFEST = {
    "scene_kind": "motion_lab",
    "motion_target": "green_ball",
    "compare_pair": {"a": "blue_ball", "b": "green_ball",
                     "attribute": "friction", "levels": {"a": 2.0, "b": 0.05}},
    "families": ["motion_state_change", "friction_or_restitution_identification"],
}
