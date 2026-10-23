"""Golden scene variant: collision_arena

Plan: a hinged striker hits a block, which transfers motion through a ball
    toward an upright panel.
"""


def build_scene(api):
    api.add_plane(
        name="ground",
        pos=(-0.022, 0.0, 0.0),
        size=(4.98, 5.0),
        rgba=(0.65, 0.65, 0.65, 1.0),
        friction=0.491,
    )

    # Cantilever pivot: thin riser on the far left + bracket + pin (swing clears riser).
    _pivot = (-1.05, 0.0, 0.56)
    api.add_box(
        "gray_anchor",
        pos=(-1.242, 0.0, 0.30),
        size=(0.01, 0.03, 0.30),
        mass=9.882,
        rgba=(0.5, 0.5, 0.5, 1.0),
        friction=0.468,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "pivot_bracket",
        pos=(-1.077, 0.0, 0.56),
        size=(0.04, 0.04, 0.025),
        mass=10.109,
        rgba=(0.95, 0.95, 0.95, 1.0),
        friction=0.461,
        restitution=0.063,
        fixed=True,
    )
    api.add_box(
        "pivot_hub",
        pos=_pivot,
        size=(0.079, 0.06, 0.03),
        mass=9.887,
        rgba=(0.08, 0.08, 0.08, 1.0),
        friction=0.493,
        restitution=0.02,
        fixed=True,
    )

    # Vertical capsule arm (top near pivot); world hinge at bracket tip.
    api.add_capsule(
        "swing_striker",
        pos=(-1.28, 0.0, 0.30),
        radius=0.015,
        half_height=0.161,
        mass=11.963,
        rgba=(0.9, 0.15, 0.15, 1.0),
        friction=0.288,
        restitution=0.073,
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
        pos=(-0.668, 0.0, 0.10),
        size=(0.177, 0.08, 0.10),
        mass=0.927,
        rgba=(0.95, 0.9, 0.2, 1.0),
        friction=0.076,
        restitution=0.333,
        fixed=False,
    )

    # Ball sits in the rebound lane toward the green panel.
    api.add_sphere(
        "blue_ball",
        pos=(-0.334, 0.0, 0.110),
        radius=0.114,
        mass=0.958,
        rgba=(0.2, 0.3, 0.9, 1.0),
        friction=0.095,
        restitution=0.369,
        fixed=False,
    )

    api.add_box(
        "green_plate",
        pos=(0.234, 0.0, 0.28),
        size=(0.044, 0.36, 0.28),
        mass=0.382,
        rgba=(0.2, 0.75, 0.25, 1.0),
        friction=0.536,
        restitution=0.0,
        fixed=False,
    )

    api.add_box(
        "orange_bar",
        pos=(0.615, 0.0, 0.145),
        size=(0.039, 0.070, 0.135),
        mass=0.717,
        rgba=(0.95, 0.55, 0.15, 1.0),
        friction=0.546,
        restitution=0.047,
        fixed=False,
    )
    api.add_box(
        "purple_bar",
        pos=(0.753, 0.0, 0.145),
        size=(0.053, 0.070, 0.155),
        mass=0.65,
        rgba=(0.6, 0.2, 0.8, 1.0),
        friction=0.474,
        restitution=0.0,
        fixed=False,
    )
    api.add_box(
        "cyan_bar",
        pos=(0.517, 0.16, 0.145),
        size=(0.017, 0.032, 0.165),
        mass=0.629,
        rgba=(0.2, 0.8, 0.85, 1.0),
        friction=0.556,
        restitution=0.025,
        fixed=False,
    )

    api.set_duration(4.0)
    api.add_camera("cam0", pos=(-0.136, -3.55, 0.88), target=(0.051, 0.0, 0.26))


SCENE_MANIFEST = {
    "scene_kind": "collision_arena",
    "target": "green_plate",
    "trigger": "swing_striker",
    "movers": ["yellow_block", "blue_ball", "green_plate", "orange_bar"],
    "relation_pair": {
        "a": "blue_ball",
        "b": "green_plate",
        "expected": "no_contact",
    },
    "families": [
        "motion_attribution",
        "collision_final_state",
        "collision_final_relation",
    ],
}