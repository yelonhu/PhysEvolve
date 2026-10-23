"""Golden scene variant: throwing_pit_board — asymmetric_pits v04

Target families: throwing_final_destination, throwing_current_run_judgment,
    counterfactual_destination, inverse_intervention
Plan: middle_platform x half 0.36; orange/cyan/green/purple walls; red/blue/yellow balls.
"""

def build_scene(api):
    api.add_plane(name="ground", pos=(-0.044, 0.0, 0.0), size=(4.91, 5.0),
                  rgba=(0.65, 0.65, 0.65, 1.0), friction=0.533)
    # Bottom platforms; the gaps between them are the pits.
    api.add_box("left_platform", pos=(-0.986, 0, 0.105), size=(0.3, 0.13, 0.105),
                mass=9.93, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.523,
                restitution=0.0, fixed=True)
    api.add_box("middle_platform", pos=(0.053, 0, 0.105), size=(0.389, 0.13, 0.105),
                mass=10.079, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.434,
                restitution=0.0, fixed=True)
    api.add_box("right_platform", pos=(1, 0, 0.105), size=(0.253, 0.13, 0.105),
                mass=9.978, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.564,
                restitution=0.0, fixed=True)
    # Slanted / vertical floating walls (deflectors; removable for cf / inverse).
    api.add_box("orange_left_wall", pos=(-1.298, 0, 0.62), size=(0.579, 0.055, 0.055),
                mass=10.029, rgba=(0.95, 0.55, 0.15, 1.0), friction=0.48,
                restitution=0.526, fixed=True)
    api.set_rotation("orange_left_wall", quaternion=(0.9239, 0.0, 0.3827, 0.0))
    api.add_box("cyan_wall", pos=(0.574, 0, 0.75), size=(0.066, 0.06, 0.315),
                mass=10.036, rgba=(0.2, 0.8, 0.85, 1.0), friction=0.564,
                restitution=0.513, fixed=True)
    api.add_box("green_wall", pos=(-0.207, 0, 0.7), size=(0.32, 0.055, 0.055),
                mass=9.997, rgba=(0.2, 0.75, 0.25, 1.0), friction=0.451,
                restitution=0.571, fixed=True)
    api.set_rotation("green_wall", quaternion=(0.9135, 0.0, 0.4067, 0.0))
    api.add_box("purple_right_wall", pos=(1.345, 0, 0.64), size=(0.611, 0.06, 0.055),
                mass=9.966, rgba=(0.6, 0.2, 0.8, 1.0), friction=0.56,
                restitution=0.442, fixed=True)
    api.set_rotation("purple_right_wall", quaternion=(0.9304, 0.0, -0.3665, 0.0))
    # Three launched balls. NEUTRAL names (ball_a/b/c) + distinct rgba: the
    # colour word in the question ("the green ball") is derived from the rgba,
    # never from the body name, so colour-permuted variants stay correct.
    api.add_sphere("ball_a", pos=(-1.437, 0, 1.3), radius=0.115, mass=1.032,
                   rgba=(0.9, 0.15, 0.15, 1.0), friction=0.272, restitution=0.755,
                   fixed=False)
    api.add_sphere("ball_b", pos=(-0.049, 0, 1.48), radius=0.037, mass=0.944,
                   rgba=(0.2, 0.3, 0.9, 1.0), friction=0.205, restitution=0.902,
                   fixed=False)
    api.add_sphere("ball_c", pos=(1.137, 0, 1.25), radius=0.086, mass=1.032,
                   rgba=(0.95, 0.9, 0.2, 1.0), friction=0.155, restitution=0.859,
                   fixed=False)
    api.set_velocity("ball_a", linear=(0.182, 0.0, -0.18), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_b", linear=(0.075, 0.0, -0.24), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_c", linear=(-0.097, 0.0, -0.40), angular=(0.0, 0.0, 0.0))
    api.set_gravity((0.0, 0.0, -1.8))
    api.set_duration(4.9)
    api.add_camera("cam0", pos=(-0.012, -3.70, 1.05), target=(-0.02, 0.0, 0.72))

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