"""Golden scene variant: collision_arena

Plan: a hinged striker hits a block, which transfers motion through a ball
    toward an upright panel.
"""


def build_scene(api):
    api.add_plane(
        name="ground",
        pos=(0.0, 0.0, 0.0),
        size=(5.0, 5.0),
        rgba=(0.65, 0.65, 0.65, 1.0),
        friction=0.5,
    )

    # Cantilever pivot: thin riser on the far left + bracket + pin (swing clears riser).
    _pivot = (-1.05, 0.0, 0.56)
    api.add_box(
        "gray_anchor",
        pos=(-1.28, 0.0, 0.30),
        size=(0.03, 0.03, 0.30),
        mass=10.0,
        rgba=(0.5, 0.5, 0.5, 1.0),
        friction=0.5,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "pivot_bracket",
        pos=(-1.135, 0.0, 0.56),
        size=(0.070, 0.04, 0.025),
        mass=10.0,
        rgba=(0.95, 0.95, 0.95, 1.0),
        friction=0.5,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "pivot_hub",
        pos=_pivot,
        size=(0.03, 0.06, 0.03),
        mass=10.0,
        rgba=(0.08, 0.08, 0.08, 1.0),
        friction=0.5,
        restitution=0.0,
        fixed=True,
    )

    # Vertical capsule arm (top near pivot); world hinge at bracket tip.
    api.add_capsule(
        "swing_striker",
        pos=(-1.214, 0.0, 0.30),
        radius=0.045,
        half_height=0.20,
        mass=12.0,
        rgba=(0.9, 0.15, 0.15, 1.0),
        friction=0.30,
        restitution=0.05,
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
        pos=(-0.74, 0.0, 0.10),
        size=(0.21, 0.08, 0.10),
        mass=1.0,
        rgba=(0.95, 0.9, 0.2, 1.0),
        friction=0.00,
        restitution=0.28,
        fixed=False,
    )

    # Ball sits in the rebound lane toward the green panel.
    api.add_sphere(
        "blue_ball",
        pos=(-0.38, 0.0, 0.110),
        radius=0.110,
        mass=1.0,
        rgba=(0.2, 0.3, 0.9, 1.0),
        friction=0.15,
        restitution=0.40,
        fixed=False,
    )

    api.add_box(
        "green_plate",
        pos=(0.24, 0.0, 0.28),
        size=(0.025, 0.36, 0.28),
        mass=0.35,
        rgba=(0.2, 0.75, 0.25, 1.0),
        friction=0.50,
        restitution=0.00,
        fixed=False,
    )

    api.add_box(
        "orange_bar",
        pos=(0.56, 0.0, 0.145),
        size=(0.040, 0.070, 0.135),
        mass=0.65,
        rgba=(0.95, 0.55, 0.15, 1.0),
        friction=0.55,
        restitution=0.00,
        fixed=False,
    )
    api.add_box(
        "purple_bar",
        pos=(0.76, 0.0, 0.145),
        size=(0.035, 0.070, 0.155),
        mass=0.65,
        rgba=(0.6, 0.2, 0.8, 1.0),
        friction=0.55,
        restitution=0.00,
        fixed=False,
    )
    api.add_box(
        "cyan_bar",
        pos=(0.56, 0.16, 0.145),
        size=(0.032, 0.032, 0.165),
        mass=0.65,
        rgba=(0.2, 0.8, 0.85, 1.0),
        friction=0.55,
        restitution=0.00,
        fixed=False,
    )

    api.set_duration(4.0)
    api.add_camera("cam0", pos=(-0.12, -3.55, 0.88), target=(0.08, 0.0, 0.26))


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
