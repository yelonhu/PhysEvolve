"""Golden scene variant: collision_arena

Plan: a hinged striker hits a block, which transfers motion through a ball
    toward an upright panel.
"""


def build_scene(api):
    api.add_plane(
        name="ground",
        pos=(-0.063, 0.0, 0.0),
        size=(5.036, 5.0),
        rgba=(0.65, 0.65, 0.65, 1.0),
        friction=0.57,
    )

    # Cantilever pivot: thin riser on the far left + bracket + pin (swing clears riser).
    _pivot = (-1.05, 0.0, 0.56)
    api.add_box(
        "gray_anchor",
        pos=(-1.339, 0.0, 0.30),
        size=(0.067, 0.03, 0.30),
        mass=9.953,
        rgba=(0.5, 0.5, 0.5, 1.0),
        friction=0.529,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "pivot_bracket",
        pos=(-1.227, 0.0, 0.56),
        size=(0.026, 0.04, 0.025),
        mass=9.966,
        rgba=(0.95, 0.95, 0.95, 1.0),
        friction=0.462,
        restitution=0.049,
        fixed=True,
    )
    api.add_box(
        "pivot_hub",
        pos=_pivot,
        size=(0.051, 0.06, 0.03),
        mass=10.012,
        rgba=(0.08, 0.08, 0.08, 1.0),
        friction=0.526,
        restitution=0.0,
        fixed=True,
    )

    # Vertical capsule arm (top near pivot); world hinge at bracket tip.
    api.add_capsule(
        "swing_striker",
        pos=(-1.304, 0.0, 0.30),
        radius=0.019,
        half_height=0.133,
        mass=12.115,
        rgba=(0.9, 0.15, 0.15, 1.0),
        friction=0.285,
        restitution=0.128,
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
        pos=(-0.75, 0.0, 0.10),
        size=(0.227, 0.08, 0.10),
        mass=1.06,
        rgba=(0.95, 0.9, 0.2, 1.0),
        friction=0.03,
        restitution=0.218,
        fixed=False,
    )

    # Ball sits in the rebound lane toward the green panel.
    api.add_sphere(
        "blue_ball",
        pos=(-0.366, 0.0, 0.110),
        radius=0.131,
        mass=0.949,
        rgba=(0.2, 0.3, 0.9, 1.0),
        friction=0.085,
        restitution=0.46,
        fixed=False,
    )

    api.add_box(
        "green_plate",
        pos=(0.242, 0.0, 0.28),
        size=(0.024, 0.36, 0.28),
        mass=0.342,
        rgba=(0.2, 0.75, 0.25, 1.0),
        friction=0.455,
        restitution=0.048,
        fixed=False,
    )

    api.add_box(
        "orange_bar",
        pos=(0.56, 0.0, 0.145),
        size=(0.024, 0.070, 0.135),
        mass=0.701,
        rgba=(0.95, 0.55, 0.15, 1.0),
        friction=0.484,
        restitution=0.0,
        fixed=False,
    )
    api.add_box(
        "purple_bar",
        pos=(0.684, 0.0, 0.145),
        size=(0.073, 0.070, 0.155),
        mass=0.588,
        rgba=(0.6, 0.2, 0.8, 1.0),
        friction=0.554,
        restitution=0.0,
        fixed=False,
    )
    api.add_box(
        "cyan_bar",
        pos=(0.557, 0.16, 0.145),
        size=(0.062, 0.032, 0.165),
        mass=0.656,
        rgba=(0.2, 0.8, 0.85, 1.0),
        friction=0.543,
        restitution=0.0,
        fixed=False,
    )

    api.set_duration(4.0)
    api.add_camera("cam0", pos=(-0.13, -3.55, 0.88), target=(0.095, 0.0, 0.26))


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