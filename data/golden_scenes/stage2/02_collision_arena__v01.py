"""Golden scene: collision_arena — hinged striker chain (stable t=0).

Families: motion_attribution, collision_final_state, collision_final_relation.

Plan: a gray cantilever (`gray_anchor` riser + `pivot_bracket` + `pivot_hub`)
supports a y-axis pivot; the vertical capsule `swing_striker` swings down and
strikes the stationary `brown_hammer`, which knocks the orange ball toward the
blue panel.
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
        rgba=(0.55, 0.55, 0.58, 1.0),
        friction=0.5,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "pivot_bracket",
        pos=(-1.135, 0.0, 0.56),
        size=(0.070, 0.04, 0.025),
        mass=10.0,
        rgba=(0.50, 0.50, 0.53, 1.0),
        friction=0.5,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "pivot_hub",
        pos=_pivot,
        size=(0.03, 0.06, 0.03),
        mass=10.0,
        rgba=(0.45, 0.45, 0.48, 1.0),
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
        rgba=(0.40, 0.40, 0.44, 1.0),
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

    # Stationary brown target in the swing arc (+x of the pivot).
    api.add_box(
        "brown_hammer",
        pos=(-0.74, 0.0, 0.10),
        size=(0.18, 0.05, 0.07),
        mass=1.0,
        rgba=(0.46, 0.24, 0.10, 1.0),
        friction=0.00,
        restitution=0.28,
        fixed=False,
    )

    # Ball sits in the brown rebound lane toward the blue panel.
    api.add_sphere(
        "orange_ball",
        pos=(-0.38, 0.0, 0.110),
        radius=0.110,
        mass=1.0,
        rgba=(0.95, 0.52, 0.12, 1.0),
        friction=0.15,
        restitution=0.40,
        fixed=False,
    )

    api.add_box(
        "blue_plate",
        pos=(0.24, 0.0, 0.28),
        size=(0.065, 0.36, 0.28),
        mass=0.35,
        rgba=(0.12, 0.28, 0.92, 1.0),
        friction=0.50,
        restitution=0.00,
        fixed=False,
    )

    api.add_box(
        "purple_bar",
        pos=(0.56, 0.0, 0.145),
        size=(0.032, 0.070, 0.145),
        mass=0.35,
        rgba=(0.58, 0.22, 0.88, 1.0),
        friction=0.55,
        restitution=0.00,
        fixed=False,
    )
    api.add_box(
        "green_bar",
        pos=(0.76, 0.0, 0.145),
        size=(0.032, 0.070, 0.145),
        mass=0.35,
        rgba=(0.22, 0.78, 0.32, 1.0),
        friction=0.55,
        restitution=0.00,
        fixed=False,
    )
    api.add_box(
        "pink_bar",
        pos=(0.56, 0.16, 0.145),
        size=(0.032, 0.032, 0.145),
        mass=0.35,
        rgba=(0.94, 0.45, 0.56, 1.0),
        friction=0.55,
        restitution=0.00,
        fixed=False,
    )

    api.set_duration(4.0)
    api.add_camera("cam0", pos=(-0.15, -3.55, 0.88), target=(0.08, 0.0, 0.26))


SCENE_MANIFEST = {
    "scene_kind": "collision_arena",
    "target": "blue_plate",
    "trigger": "swing_striker",
    "movers": ["brown_hammer", "orange_ball", "blue_plate", "purple_bar"],
    "relation_pair": {
        "a": "orange_ball",
        "b": "blue_plate",
        "expected": "contact",
    },
    "families": [
        "motion_attribution",
        "collision_final_state",
        "collision_final_relation",
    ],
}
