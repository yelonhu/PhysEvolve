"""Golden scene variant: throwing_pit_board — wall_dense

Target families: throwing_final_destination, throwing_current_run_judgment,
    counterfactual_destination, inverse_intervention
Plan: denser floating-wall deflectors; all platforms / walls / balls raised +0.1 z;
    one white-gray wall plus red / blue / green walls; green / orange / purple balls.
"""

def build_scene(api):
    api.add_plane(name="ground", pos=(0.047, 0.0, 0.0), size=(4.996, 5.0),
                  rgba=(0.65, 0.65, 0.65, 1.0), friction=0.462)
    # Bottom platforms; the gaps between them are the pits.
    api.add_box("left_platform", pos=(-1.0, 0, 0.205), size=(0.366, 0.13, 0.105),
                mass=10.062, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.48,
                restitution=0.043, fixed=True)
    api.add_box("middle_platform", pos=(0.064, 0, 0.205), size=(0.448, 0.13, 0.105),
                mass=9.979, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.506,
                restitution=0.029, fixed=True)
    api.add_box("right_platform", pos=(1, 0, 0.205), size=(0.281, 0.13, 0.105),
                mass=10.073, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.462,
                restitution=0.049, fixed=True)
    # Slanted / vertical floating walls (deflectors; removable for cf / inverse).
    api.add_box("gray_left_wall", pos=(-1.227, 0, 0.72), size=(0.474, 0.055, 0.055),
                mass=10.072, rgba=(0.82, 0.82, 0.86, 1.0), friction=0.549,
                restitution=0.491, fixed=True)
    api.set_rotation("gray_left_wall", quaternion=(0.9239, 0.0, 0.3827, 0.0))
    api.add_box("red_wall", pos=(-0.815, 0, 0.85), size=(0.013, 0.06, 0.315),
                mass=9.95, rgba=(0.9, 0.15, 0.15, 1.0), friction=0.572,
                restitution=0.514, fixed=True)
    api.add_box("blue_wall", pos=(0.252, 0, 0.8), size=(0.415, 0.055, 0.055),
                mass=10.104, rgba=(0.2, 0.3, 0.9, 1.0), friction=0.566,
                restitution=0.502, fixed=True)
    api.set_rotation("blue_wall", quaternion=(0.9455, 0.0, -0.3256, 0.0))
    api.add_box("green_right_wall", pos=(1.337, 0, 0.74), size=(0.599, 0.06, 0.055),
                mass=9.887, rgba=(0.2, 0.75, 0.25, 1.0), friction=0.478,
                restitution=0.516, fixed=True)
    api.set_rotation("green_right_wall", quaternion=(0.9304, 0.0, -0.3665, 0.0))
    # Three launched balls. NEUTRAL names (ball_a/b/c) + distinct rgba: the
    # colour word in the question ("the green ball") is derived from the rgba,
    # never from the body name, so colour-permuted variants stay correct.
    api.add_sphere("ball_a", pos=(-1.319, 0, 1.4), radius=0.019, mass=1.061,
                   rgba=(0.95, 0.55, 0.15, 1.0), friction=0.204, restitution=0.639,
                   fixed=False)
    api.add_sphere("ball_b", pos=(0.076, 0, 1.58), radius=0.055, mass=0.951,
                   rgba=(0.6, 0.2, 0.8, 1.0), friction=0.197, restitution=0.842,
                   fixed=False)
    api.add_sphere("ball_c", pos=(1.128, 0, 1.35), radius=0.117, mass=0.922,
                   rgba=(0.95, 0.9, 0.2, 1.0), friction=0.244, restitution=0.823,
                   fixed=False)
    api.set_velocity("ball_a", linear=(0.233, 0.0, -0.18), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_b", linear=(0.165, 0.0, -0.24), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_c", linear=(-0.041, 0.0, -0.40), angular=(0.0, 0.0, 0.0))
    api.set_gravity((0.0, 0.0, -1.8))
    api.set_duration(4.9)
    api.add_camera("cam0", pos=(-0.031, -3.70, 1.15), target=(-0.02, 0.0, 0.82))

SCENE_MANIFEST = {
    "scene_kind": "throwing_pit_board",
    "balls": ["ball_a", "ball_b", "ball_c"],
    "platforms": ["left_platform", "middle_platform", "right_platform"],
    "pits": {
        "left": {"x": [-0.58, -0.30]},
        "middle": {"x": [0.50, 0.67]},
        "right": {"x": [1.33, 1.50]},
    },
    "walls": ["gray_left_wall", "red_wall", "blue_wall", "green_right_wall"],
    "region_kind": "pit",
    "families": ["throwing_final_destination", "throwing_current_run_judgment",
                 "counterfactual_destination", "inverse_intervention"],
}