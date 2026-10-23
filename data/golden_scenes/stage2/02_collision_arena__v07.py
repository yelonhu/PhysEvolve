"""Golden scene: collision_arena — hinged striker chain (stable t=0).

Families: motion_attribution, collision_final_state, collision_final_relation.

Plan: a gray cantilever (`gray_anchor` riser + `pivot_bracket` + `pivot_hub`)
supports a y-axis pivot; the vertical capsule `swing_striker` swings down and
strikes the stationary `brown_hammer`, which knocks the orange ball toward the
blue panel.
Relation: the orange ball and blue plate remain separated in this perturbation.
"""


def build_scene(api):
    api.add_plane(
        name="ground",
        pos=(-0.028, 0.019, 0.01),
        size=(5.077, 4.943),
        rgba=(0.958, 0.462, 0.586, 1.000),
        friction=0.452,
    )

    # Cantilever pivot: thin riser on the far left + bracket + pin (swing clears riser).
    _pivot = (-1.05, 0.0, 0.56)
    api.add_box(
        "gray_anchor",
        pos=(-1.353, -0.076, 0.331),
        size=(0.01, 0.08, 0.342),
        mass=9.976,
        rgba=(0.936, 0.914, 0.180, 1.000),
        friction=0.447,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "pivot_bracket",
        pos=(-1.099, -0.05, 0.504),
        size=(0.132, 0.03, 0.096),
        mass=10.077,
        rgba=(0.225, 0.789, 0.286, 1.000),
        friction=0.526,
        restitution=0.0,
        fixed=True,
    )
    api.add_box(
        "pivot_hub",
        pos=_pivot,
        size=(0.026, 0.019, 0.068),
        mass=10.002,
        rgba=(0.940, 0.449, 0.593, 1.000),
        friction=0.468,
        restitution=0.0,
        fixed=True,
    )

    # Vertical capsule arm (top near pivot); world hinge at bracket tip.
    api.add_capsule(
        "swing_striker",
        pos=(-1.24, -0.069, 0.26),
        radius=0.012,
        half_height=0.187,
        mass=11.999,
        rgba=(0.942, 0.896, 0.200, 1.000),
        friction=0.311,
        restitution=0.106,
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
        pos=(-0.773, -0.065, 0.069),
        size=(0.135, 0.062, 0.066),
        mass=0.971,
        rgba=(0.129, 0.250, 0.925, 1.000),
        friction=0.0,
        restitution=0.32,
        fixed=False,
    )

    # Ball sits in the brown rebound lane toward the blue panel.
    api.add_sphere(
        "orange_ball",
        pos=(-0.36, 0.02, 0.142),
        radius=0.167,
        mass=0.926,
        rgba=(0.179, 0.762, 0.824, 1.000),
        friction=0.1,
        restitution=0.395,
        fixed=False,
    )

    api.add_box(
        "blue_plate",
        pos=(0.272, 0.001, 0.281),
        size=(0.034, 0.319, 0.34),
        mass=0.427,
        rgba=(0.220, 0.739, 0.811, 1.000),
        friction=0.5,
        restitution=0.0,
        fixed=False,
    )

    api.add_box(
        "purple_bar",
        pos=(0.501, 0.016, 0.099),
        size=(0.052, 0.035, 0.14),
        mass=0.408,
        rgba=(0.090, 0.301, 0.932, 1.000),
        friction=0.52,
        restitution=0.006,
        fixed=False,
    )
    api.add_box(
        "green_bar",
        pos=(0.731, -0.002, 0.204),
        size=(0.031, 0.144, 0.13),
        mass=0.343,
        rgba=(0.937, 0.148, 0.110, 1.000),
        friction=0.573,
        restitution=0.0,
        fixed=False,
    )
    api.add_box(
        "pink_bar",
        pos=(0.588, 0.126, 0.067),
        size=(0.02, 0.038, 0.105),
        mass=0.366,
        rgba=(0.890, 0.142, 0.160, 1.000),
        friction=0.58,
        restitution=0.066,
        fixed=False,
    )

    api.set_duration(4.0)
    api.add_camera("cam0", pos=(-0.082, -3.645, 0.948), target=(0.046, -0.064, 0.266))


SCENE_MANIFEST = {
    "scene_kind": "collision_arena",
    "target": "blue_plate",
    "trigger": "swing_striker",
    "movers": ["brown_hammer", "orange_ball", "blue_plate", "purple_bar"],
    "relation_pair": {
        "a": "orange_ball",
        "b": "blue_plate",
        "expected": "no_contact",
    },
    "families": [
        "motion_attribution",
        "collision_final_state",
        "collision_final_relation",
    ],
}