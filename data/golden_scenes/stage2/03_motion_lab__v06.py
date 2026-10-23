"""Golden scene: motion_lab dual-fall speed_pair.

Families: motion_state_change (forward + speed_profile + pairwise_speed)
Plan: two free spheres fall side-by-side under gravity from different heights /
initial downward speeds so one is consistently faster.
"""


def build_scene(api):
    api.add_plane(
        name="ground",
        pos=(-0.001, 0.009, 0.05),
        size=(4.957, 4.997),
        rgba=(0.968, 0.576, 0.182, 1.000),
        friction=0.377,
    )
    # Soft landing pads so rebound is visible but does not invert the ordering.
    api.add_box(
        "pad_a",
        pos=(-0.365, -0.027, 0.01),
        size=(0.271, 0.313, 0.05),
        mass=10.022,
        rgba=(0.236, 0.716, 0.819, 1.000),
        friction=0.274,
        restitution=0.139,
        fixed=True,
    )
    api.add_box(
        "pad_b",
        pos=(0.293, 0.037, 0.01),
        size=(0.217, 0.327, 0.01),
        mass=10.077,
        rgba=(0.975, 0.524, 0.159, 1.000),
        friction=0.319,
        restitution=0.192,
        fixed=True,
    )
    # Neutral names + distinct rgba (colour word derived from rgba).
    api.add_sphere(
        "ball_a",
        pos=(-0.33, -0.051, 1.292),
        radius=0.075,
        mass=0.941,
        rgba=(0.929, 0.877, 0.227, 1.000),
        friction=0.129,
        restitution=0.329,
        fixed=False,
    )
    api.add_sphere(
        "ball_b",
        pos=(0.272, -0.018, 1.006),
        radius=0.17,
        mass=0.988,
        rgba=(0.193, 0.749, 0.831, 1.000),
        friction=0.137,
        restitution=0.325,
        fixed=False,
    )
    # ball_a starts higher with a stronger downward kick -> consistently faster.
    api.set_velocity("ball_a", linear=(-0.07, -0.045, -0.831), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_b", linear=(0.019, -0.075, -0.252), angular=(0.0, 0.0, 0.0))
    api.set_gravity((0.0, 0.0, -9.8))
    api.set_duration(2.4)
    api.add_camera("cam0", pos=(-0.04, -2.751, 0.966), target=(0.03, 0.024, 0.571))


SCENE_MANIFEST = {
    "scene_kind": "motion_lab",
    "motion_target": "ball_a",
    "speed_pair": {"a": "ball_a", "b": "ball_b"},
    "families": ["motion_state_change"],
}