"""Golden scene: motion_lab dual-fall speed_pair.

Families: motion_state_change (forward + speed_profile + pairwise_speed)
Plan: two free spheres fall side-by-side under gravity from different heights /
initial downward speeds so one is consistently faster.
"""


def build_scene(api):
    api.add_plane(
        name="ground",
        pos=(0.0, 0.0, 0.0),
        size=(5.0, 5.0),
        rgba=(0.65, 0.65, 0.65, 1.0),
        friction=0.4,
    )
    # Soft landing pads so rebound is visible but does not invert the ordering.
    api.add_box(
        "pad_a",
        pos=(-0.35, 0.0, 0.02),
        size=(0.25, 0.25, 0.02),
        mass=10.0,
        rgba=(0.40, 0.40, 0.42, 1.0),
        friction=0.3,
        restitution=0.15,
        fixed=True,
    )
    api.add_box(
        "pad_b",
        pos=(0.35, 0.0, 0.02),
        size=(0.25, 0.25, 0.02),
        mass=10.0,
        rgba=(0.40, 0.40, 0.42, 1.0),
        friction=0.3,
        restitution=0.15,
        fixed=True,
    )
    # Neutral names + distinct rgba (colour word derived from rgba).
    api.add_sphere(
        "ball_a",
        pos=(-0.35, 0.0, 1.35),
        radius=0.09,
        mass=1.0,
        rgba=(0.22, 0.78, 0.32, 1.0),
        friction=0.2,
        restitution=0.35,
        fixed=False,
    )
    api.add_sphere(
        "ball_b",
        pos=(0.35, 0.0, 0.95),
        radius=0.09,
        mass=1.0,
        rgba=(0.95, 0.55, 0.15, 1.0),
        friction=0.2,
        restitution=0.35,
        fixed=False,
    )
    # ball_a starts higher with a stronger downward kick -> consistently faster.
    api.set_velocity("ball_a", linear=(0.0, 0.0, -0.85), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_b", linear=(0.0, 0.0, -0.25), angular=(0.0, 0.0, 0.0))
    api.set_gravity((0.0, 0.0, -9.8))
    api.set_duration(2.4)
    api.add_camera("cam0", pos=(0.0, -2.80, 1.10), target=(0.0, 0.0, 0.55))


SCENE_MANIFEST = {
    "scene_kind": "motion_lab",
    "motion_target": "ball_a",
    "speed_pair": {"a": "ball_a", "b": "ball_b"},
    "families": ["motion_state_change"],
}
