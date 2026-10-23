"""Golden scene variant: motion_lab — ramp dock v02

Target families: motion_state_change, friction_or_restitution_identification
Plan: same ramp-dock cascade as v01 with higher friction (+0.10), heavier balls
    (×2 mass), enlarged middle slider (×1.25 half-axes, mass=0.5), classic palette
    colours on dynamic bodies (red_ball / yellow_slider / purple_ball); red_ball
    initial velocity (0.50, 0, -0.80) restores slow_down + friction VQA at seed=7
    after executor geometry repair (slider/purple z clearance).
"""


def build_scene(api):
    api.add_plane(
        name="ground",
        pos=(0.0, 0.0, 0.0),
        size=(5.0, 5.0),
        rgba=(0.65, 0.65, 0.65, 1.0),
        friction=0.60,
    )

    # Flush contact surfaces: the wood deck is split around the rough pad so
    # the high-friction patch is embedded instead of raised above the table.
    api.add_box(
        "deck_west",
        pos=(-0.2725, 0.0, 0.010),
        size=(0.4275, 0.30, 0.010),
        mass=10.0,
        rgba=(0.42, 0.32, 0.22, 1.0),
        friction=0.40,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "ramp_surface",
        pos=(-0.95, 0.0, 0.215),
        size=(0.72, 0.22, 0.025),
        mass=10.0,
        rgba=(0.48, 0.36, 0.24, 1.0),
        friction=0.25,
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
        friction=0.40,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "deck_east",
        pos=(0.8725, 0.0, 0.010),
        size=(0.4275, 0.30, 0.010),
        mass=10.0,
        rgba=(0.42, 0.32, 0.22, 1.0),
        friction=0.25,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "dock_wall",
        pos=(1.29, 0.0, 0.120),
        size=(0.030, 0.20, 0.100),
        mass=10.0,
        rgba=(0.34, 0.37, 0.39, 1.0),
        friction=0.60,
        restitution=0.9,
        fixed=True,
    )

    api.add_sphere(
        "red_ball",
        pos=(-1.30, 0.0, 1.2),
        radius=0.09,
        mass=2.0,
        rgba=(0.9, 0.15, 0.15, 1.0),
        friction=0.15,
        restitution=0.9,
        fixed=False,
    )
    api.add_box(
        "yellow_slider",
        pos=(-0.08, 0.0, 0.16),
        size=(0.13, 0.10, 0.11),
        mass=1.7,
        rgba=(0.95, 0.9, 0.2, 1.0),
        friction=0.15,
        restitution=0.9,
        fixed=False,
    )
    api.add_sphere(
        "purple_ball",
        pos=(0.70, 0.0, 0.14),
        radius=0.09,
        mass=2.4,
        rgba=(0.6, 0.2, 0.8, 1.0),
        friction=0.15,
        restitution=0.9,
        fixed=False,
    )

    # Down-ramp initial speed >= 0.10 m/s (30 deg ramp: +x, -z).
    api.set_velocity(
        "red_ball",
        linear=(0.50, 0.0, -0.80),
        angular=(0.0, 0.0, 0.0),
    )

    api.set_duration(3.5)
    api.add_camera("cam0", pos=(-0.20, -3.8, 0.75), target=(0.05, 0.0, 0.22))

SCENE_MANIFEST = {
    "scene_kind": "motion_lab",
    "motion_target": "red_ball",
    "compare_pair": {"a": "purple_ball", "b": "red_ball",
                     "attribute": "friction", "levels": {"a": 2.0, "b": 0.05}},
    "families": ["motion_state_change", "friction_or_restitution_identification"],
}
