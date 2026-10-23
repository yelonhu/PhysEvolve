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
        pos=(0.073, 0.0, 0.0),
        size=(4.886, 5.0),
        rgba=(0.65, 0.65, 0.65, 1.0),
        friction=0.68,
    )

    # Flush contact surfaces: the wood deck is split around the rough pad so
    # the high-friction patch is embedded instead of raised above the table.
    api.add_box(
        "deck_west",
        pos=(-0.323, 0.0, 0.010),
        size=(0.367, 0.30, 0.010),
        mass=9.963,
        rgba=(0.42, 0.32, 0.22, 1.0),
        friction=0.462,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "ramp_surface",
        pos=(-0.876, 0.0, 0.215),
        size=(0.691, 0.22, 0.025),
        mass=10.104,
        rgba=(0.48, 0.36, 0.24, 1.0),
        friction=0.28,
        restitution=0.068,
        fixed=True,
    )
    api.set_rotation("ramp_surface", quaternion=(0.96593, 0, 0.25882, 0))

    api.add_box(
        "rough_pad",
        pos=(0.333, 0.0, 0.010),
        size=(0.073, 0.16, 0.010),
        mass=10.021,
        rgba=(0.18, 0.16, 0.14, 1.0),
        friction=0.37,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "deck_east",
        pos=(0.928, 0.0, 0.010),
        size=(0.441, 0.30, 0.010),
        mass=10.066,
        rgba=(0.42, 0.32, 0.22, 1.0),
        friction=0.324,
        restitution=0.011,
        fixed=True,
    )
    api.add_box(
        "dock_wall",
        pos=(1.317, 0.0, 0.120),
        size=(0.01, 0.20, 0.100),
        mass=9.955,
        rgba=(0.34, 0.37, 0.39, 1.0),
        friction=0.645,
        restitution=0.973,
        fixed=True,
    )

    api.add_sphere(
        "red_ball",
        pos=(-1.386, 0.0, 1.2),
        radius=0.142,
        mass=1.924,
        rgba=(0.9, 0.15, 0.15, 1.0),
        friction=0.078,
        restitution=0.905,
        fixed=False,
    )
    api.add_box(
        "yellow_slider",
        pos=(-0.155, 0.0, 0.16),
        size=(0.198, 0.10, 0.11),
        mass=1.704,
        rgba=(0.95, 0.9, 0.2, 1.0),
        friction=0.136,
        restitution=0.926,
        fixed=False,
    )
    api.add_sphere(
        "purple_ball",
        pos=(0.767, 0.0, 0.14),
        radius=0.059,
        mass=2.42,
        rgba=(0.6, 0.2, 0.8, 1.0),
        friction=0.118,
        restitution=0.915,
        fixed=False,
    )

    # Down-ramp initial speed >= 0.10 m/s (30 deg ramp: +x, -z).
    api.set_velocity(
        "red_ball",
        linear=(0.507, 0.0, -0.80),
        angular=(0.0, 0.0, 0.0),
    )

    api.set_duration(3.5)
    api.add_camera("cam0", pos=(-0.211, -3.8, 0.75), target=(0.078, 0.0, 0.22))

SCENE_MANIFEST = {
    "scene_kind": "motion_lab",
    "motion_target": "red_ball",
    "compare_pair": {"a": "purple_ball", "b": "red_ball",
                     "attribute": "friction", "levels": {"a": 2.0, "b": 0.05}},
    "families": ["motion_state_change", "friction_or_restitution_identification"],
}