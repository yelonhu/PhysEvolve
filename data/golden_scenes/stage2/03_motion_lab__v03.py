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
        pos=(-0.054, 0.0, 0.0),
        size=(5.032, 5.0),
        rgba=(0.65, 0.65, 0.65, 1.0),
        friction=0.497,
    )

    # Flush contact surfaces: the wood deck is split around the rough pad so
    # the high-friction patch is embedded instead of raised above the table.
    api.add_box(
        "deck_west",
        pos=(-0.318, 0.0, 0.010),
        size=(0.474, 0.30, 0.010),
        mass=10.003,
        rgba=(0.42, 0.32, 0.22, 1.0),
        friction=0.301,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "ramp_surface",
        pos=(-1.029, 0.0, 0.215),
        size=(0.699, 0.22, 0.025),
        mass=9.897,
        rgba=(0.48, 0.36, 0.24, 1.0),
        friction=0.197,
        restitution=0.0,
        fixed=True,
    )
    api.set_rotation("ramp_surface", quaternion=(0.96593, 0, 0.25882, 0))

    api.add_box(
        "rough_pad",
        pos=(0.257, 0.0, 0.010),
        size=(0.072, 0.16, 0.010),
        mass=10.057,
        rgba=(0.18, 0.16, 0.14, 1.0),
        friction=0.36,
        restitution=0.019,
        fixed=True,
    )
    api.add_box(
        "deck_east",
        pos=(0.798, 0.0, 0.010),
        size=(0.4, 0.30, 0.010),
        mass=9.908,
        rgba=(0.42, 0.32, 0.22, 1.0),
        friction=0.222,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "dock_wall",
        pos=(1.366, 0.0, 0.120),
        size=(0.03, 0.20, 0.100),
        mass=9.989,
        rgba=(0.34, 0.37, 0.39, 1.0),
        friction=0.508,
        restitution=0.875,
        fixed=True,
    )

    api.add_sphere(
        "green_ball",
        pos=(-1.375, 0.0, 1.2),
        radius=0.016,
        mass=1.059,
        rgba=(0.22, 0.78, 0.32, 1.0),
        friction=0.008,
        restitution=0.945,
        fixed=False,
    )
    api.add_box(
        "amber_slider",
        pos=(-0.147, 0.0, 0.1),
        size=(0.18, 0.08, 0.09),
        mass=1.063,
        rgba=(0.95, 0.55, 0.15, 1.0),
        friction=0.091,
        restitution=0.942,
        fixed=False,
    )
    api.add_sphere(
        "blue_ball",
        pos=(0.713, 0.0, 0.110),
        radius=0.127,
        mass=1.194,
        rgba=(0.12, 0.28, 0.92, 1.0),
        friction=0.102,
        restitution=0.852,
        fixed=False,
    )

    # Down-ramp initial speed >= 0.10 m/s (30 deg ramp: +x, -z).
    api.set_velocity(
        "green_ball",
        linear=(0.168, 0.0, -0.16),
        angular=(0.0, 0.0, 0.0),
    )

    api.set_duration(3.5)
    api.add_camera("cam0", pos=(-0.138, -3.8, 0.75), target=(0.036, 0.0, 0.22))

SCENE_MANIFEST = {
    "scene_kind": "motion_lab",
    "motion_target": "green_ball",
    "compare_pair": {"a": "blue_ball", "b": "green_ball",
                     "attribute": "friction", "levels": {"a": 2.0, "b": 0.05}},
    "families": ["motion_state_change", "friction_or_restitution_identification"],
}