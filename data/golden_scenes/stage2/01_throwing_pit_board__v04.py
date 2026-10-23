"""Golden scene variant: throwing_pit_board — asymmetric_pits v04

Target families: throwing_final_destination, throwing_current_run_judgment,
    counterfactual_destination, inverse_intervention
Plan: middle_platform x half 0.36; orange/cyan/green/purple walls; red/blue/yellow balls.
"""

def build_scene(api):
    api.add_plane(name="ground", pos=(0.0, 0.0, 0.0), size=(5.0, 5.0),
                  rgba=(0.65, 0.65, 0.65, 1.0), friction=0.5)
    # Bottom platforms; the gaps between them are the pits.
    api.add_box("left_platform", pos=(-0.92, 0, 0.105), size=(0.34, 0.13, 0.105),
                mass=10.0, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.5,
                restitution=0.0, fixed=True)
    api.add_box("middle_platform", pos=(0.06, 0, 0.105), size=(0.39, 0.13, 0.105),
                mass=10.0, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.5,
                restitution=0.0, fixed=True)
    api.add_box("right_platform", pos=(1, 0, 0.105), size=(0.33, 0.13, 0.105),
                mass=10.0, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.5,
                restitution=0.0, fixed=True)
    # Slanted / vertical floating walls (deflectors; removable for cf / inverse).
    api.add_box("orange_left_wall", pos=(-1.24, 0, 0.62), size=(0.5, 0.055, 0.055),
                mass=10.0, rgba=(0.95, 0.55, 0.15, 1.0), friction=0.5,
                restitution=0.5, fixed=True)
    api.set_rotation("orange_left_wall", quaternion=(0.9239, 0.0, 0.3827, 0.0))
    api.add_box("cyan_wall", pos=(0.6, 0, 0.75), size=(0.035, 0.06, 0.315),
                mass=10.0, rgba=(0.2, 0.8, 0.85, 1.0), friction=0.5,
                restitution=0.5, fixed=True)
    api.add_box("green_wall", pos=(-0.15, 0, 0.7), size=(0.39, 0.055, 0.055),
                mass=10.0, rgba=(0.2, 0.75, 0.25, 1.0), friction=0.5,
                restitution=0.5, fixed=True)
    api.set_rotation("green_wall", quaternion=(0.9135, 0.0, 0.4067, 0.0))
    api.add_box("purple_right_wall", pos=(1.29, 0, 0.64), size=(0.55, 0.06, 0.055),
                mass=10.0, rgba=(0.6, 0.2, 0.8, 1.0), friction=0.5,
                restitution=0.5, fixed=True)
    api.set_rotation("purple_right_wall", quaternion=(0.9304, 0.0, -0.3665, 0.0))
    # Three launched balls. NEUTRAL names (ball_a/b/c) + distinct rgba: the
    # colour word in the question ("the green ball") is derived from the rgba,
    # never from the body name, so colour-permuted variants stay correct.
    api.add_sphere("ball_a", pos=(-1.34, 0, 1.3), radius=0.085, mass=1.0,
                   rgba=(0.9, 0.15, 0.15, 1.0), friction=0.2, restitution=0.7,
                   fixed=False)
    api.add_sphere("ball_b", pos=(-0.05, 0, 1.48), radius=0.085, mass=1.0,
                   rgba=(0.2, 0.3, 0.9, 1.0), friction=0.2, restitution=0.9,
                   fixed=False)
    api.add_sphere("ball_c", pos=(1.04, 0, 1.25), radius=0.085, mass=1.0,
                   rgba=(0.95, 0.9, 0.2, 1.0), friction=0.2, restitution=0.9,
                   fixed=False)
    api.set_velocity("ball_a", linear=(0.26, 0.0, -0.18), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_b", linear=(0.10, 0.0, -0.24), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_c", linear=(-0.06, 0.0, -0.40), angular=(0.0, 0.0, 0.0))
    api.set_gravity((0.0, 0.0, -1.8))
    api.set_duration(4.9)
    api.add_camera("cam0", pos=(0.0, -3.70, 1.05), target=(0.0, 0.0, 0.72))

SCENE_MANIFEST = {
    "scene_kind": "throwing_pit_board",
    "balls": ["ball_a", "ball_b", "ball_c"],
    "platforms": ["left_platform", "middle_platform", "right_platform"],
    "pits": {
        "left": {"x": [-0.58, -0.30]},
        "middle": {"x": [0.50, 0.67]},
        "right": {"x": [1.33, 1.50]},
    },
    "walls": ["orange_left_wall", "cyan_wall", "green_wall", "purple_right_wall"],
    "region_kind": "pit",
    "families": ["throwing_final_destination", "throwing_current_run_judgment",
                 "counterfactual_destination", "inverse_intervention"],
}
