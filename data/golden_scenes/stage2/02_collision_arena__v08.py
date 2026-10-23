"""Golden scene variant: collision_arena

Plan: a hinged striker hits a block, which transfers motion through a ball
    toward an upright panel.
"""


def build_scene(api):
    api.add_plane(
        name="ground",
        pos=(-0.068, 0.015, 0.078),
        size=(5.011, 5.058),
        rgba=(0.577, 0.223, 0.857, 1.000),
        friction=0.53,
    )

    # Cantilever pivot: thin riser on the far left + bracket + pin (swing clears riser).
    _pivot = (-1.05, 0.0, 0.56)
    api.add_box(
        "gray_anchor",
        pos=(-1.166, 0.032, 0.275),
        size=(0.011, 0.051, 0.331),
        mass=9.956,
        rgba=(0.598, 0.237, 0.894, 1.000),
        friction=0.549,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "pivot_bracket",
        pos=(-1.233, -0.029, 0.483),
        size=(0.057, 0.01, 0.01),
        mass=9.955,
        rgba=(0.190, 0.773, 0.290, 1.000),
        friction=0.556,
        restitution=0.022,
        fixed=True,
    )
    api.add_box(
        "pivot_hub",
        pos=_pivot,
        size=(0.01, 0.06, 0.02),
        mass=10.071,
        rgba=(0.589, 0.237, 0.891, 1.000),
        friction=0.565,
        restitution=0.011,
        fixed=True,
    )

    # Vertical capsule arm (top near pivot); world hinge at bracket tip.
    api.add_capsule(
        "swing_striker",
        pos=(-1.105, -0.045, 0.229),
        radius=0.032,
        half_height=0.18,
        mass=12.056,
        rgba=(0.609, 0.238, 0.886, 1.000),
        friction=0.243,
        restitution=0.0,
        fixed=False,
        axis="z",
    )
    api.add_hinge(
        name="hammer_pivot",
        parent="world",
        child="swing_striker",
        pos=_pivot,
        axis=(0.0, 1.0, 0.0),
    )

    # Stationary yellow target in the swing arc (+x of the pivot).
    api.add_box(
        "yellow_block",
        pos=(-0.749, -0.073, 0.126),
        size=(0.208, 0.14, 0.17),
        mass=0.957,
        rgba=(0.902, 0.125, 0.119, 1.000),
        friction=0.046,
        restitution=0.28,
        fixed=False,
    )

    # Ball sits in the rebound lane toward the green panel.
    api.add_sphere(
        "blue_ball",
        pos=(-0.308, -0.067, 0.103),
        radius=0.187,
        mass=0.929,
        rgba=(0.969, 0.416, 0.597, 1.000),
        friction=0.119,
        restitution=0.37,
        fixed=False,
    )

    api.add_box(
        "green_plate",
        pos=(0.218, 0.079, 0.231),
        size=(0.077, 0.323, 0.34),
        mass=0.293,
        rgba=(0.919, 0.422, 0.570, 1.000),
        friction=0.436,
        restitution=0.078,
        fixed=False,
    )

    api.add_box(
        "orange_bar",
        pos=(0.482, 0.023, 0.098),
        size=(0.053, 0.041, 0.155),
        mass=0.678,
        rgba=(0.106, 0.317, 0.881, 1.000),
        friction=0.59,
        restitution=0.0,
        fixed=False,
    )
    api.add_box(
        "purple_bar",
        pos=(0.813, -0.042, 0.16),
        size=(0.085, 0.113, 0.16),
        mass=0.65,
        rgba=(0.610, 0.248, 0.884, 1.000),
        friction=0.495,
        restitution=0.061,
        fixed=False,
    )
    api.add_box(
        "cyan_bar",
        pos=(0.575, 0.203, 0.22),
        size=(0.064, 0.01, 0.162),
        mass=0.598,
        rgba=(0.116, 0.246, 0.930, 1.000),
        friction=0.514,
        restitution=0.057,
        fixed=False,
    )

    api.set_duration(4.0)
    api.add_camera("cam0", pos=(-0.218, -3.639, 0.884), target=(0.131, 0.053, 0.236))


SCENE_MANIFEST = {
    "scene_kind": "collision_arena",
    "target": "green_plate",
    "trigger": "swing_striker",
    "movers": ["yellow_block", "blue_ball", "green_plate", "orange_bar"],
    "relation_pair": {
        "a": "blue_ball",
        "b": "green_plate",
        "expected": "contact",
    },
    "families": [
        "motion_attribution",
        "collision_final_state",
        "collision_final_relation",
    ],
}