"""Golden scene: throwing_pit_board

Source: physevolve/skills/scene_templates/throwing_pit_board.py
Families: throwing_final_destination, throwing_current_run_judgment, counterfactual_destination, inverse_intervention
Plan: three balls fall through a board of slanted coloured walls and drop into
the gaps between three bottom platforms; the same scene supports a destination /
yes-no / counterfactual / inverse question per ball.
"""

def build_scene(api):
    api.add_plane(name="ground", pos=(-0.067, -0.052, 0.029), size=(5.108, 5.058),
                  rgba=(0.143, 0.280, 0.953, 1.000), friction=0.494)
    # Bottom platforms; the gaps between them are the pits.
    api.add_box("left_platform", pos=(-0.868, -0.041, 0.122), size=(0.319, 0.128, 0.179),
                mass=9.955, rgba=(0.223, 0.781, 0.289, 1.000), friction=0.541,
                restitution=0.043, fixed=True)
    api.add_box("middle_platform", pos=(0.034, 0.079, 0.097), size=(0.407, 0.098, 0.114),
                mass=10.074, rgba=(0.562, 0.229, 0.877, 1.000), friction=0.476,
                restitution=0.037, fixed=True)
    api.add_box("right_platform", pos=(1.045, 0.059, 0.136), size=(0.319, 0.104, 0.086),
                mass=9.98, rgba=(0.201, 0.743, 0.291, 1.000), friction=0.562,
                restitution=0.053, fixed=True)
    # Slanted / vertical floating walls (deflectors; removable for cf / inverse).
    api.add_box("black_left_wall", pos=(-1.446, -0.073, 0.689), size=(0.429, 0.038, 0.01),
                mass=9.952, rgba=(0.913, 0.579, 0.176, 1.000), friction=0.571,
                restitution=0.49, fixed=True)
    api.set_rotation("black_left_wall", quaternion=(0.9426, 0.0008, 0.3988, 0.0134))
    api.add_box("orange_wall", pos=(-0.429, 0.053, 0.754), size=(0.086, 0.103, 0.332),
                mass=10.01, rgba=(0.969, 0.566, 0.126, 1.000), friction=0.562,
                restitution=0.528, fixed=True)
    api.add_box("blue_wall", pos=(0.175, 0.064, 0.751), size=(0.462, 0.082, 0.046),
                mass=9.977, rgba=(0.114, 0.249, 0.882, 1.000), friction=0.52,
                restitution=0.449, fixed=True)
    api.set_rotation("blue_wall", quaternion=(0.9443, 0.0196, -0.3351, -0.0077))
    api.add_box("pink_right_wall", pos=(1.34, 0.061, 0.595), size=(0.563, 0.068, 0.071),
                mass=9.985, rgba=(0.966, 0.517, 0.146, 1.000), friction=0.421,
                restitution=0.568, fixed=True)
    api.set_rotation("pink_right_wall", quaternion=(0.9194, 0.0102, -0.3824, -0.0023))
    # Three launched balls. NEUTRAL names (ball_a/b/c) + distinct rgba: the
    # colour word in the question ("the green ball") is derived from the rgba,
    # never from the body name, so colour-permuted variants stay correct.
    api.add_sphere("ball_a", pos=(-1.422, 0.066, 1.182), radius=0.107, mass=0.968,
                   rgba=(0.606, 0.235, 0.867, 1.000), friction=0.18, restitution=0.637,
                   fixed=False)
    api.add_sphere("ball_b", pos=(0.003, 0.024, 1.497), radius=0.073, mass=1.077,
                   rgba=(0.224, 0.820, 0.356, 1.000), friction=0.153, restitution=0.929,
                   fixed=False)
    api.add_sphere("ball_c", pos=(1.133, -0.037, 1.173), radius=0.029, mass=1.019,
                   rgba=(0.932, 0.522, 0.187, 1.000), friction=0.264, restitution=0.886,
                   fixed=False)
    api.set_velocity("ball_a", linear=(0.208, -0.064, -0.194), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_b", linear=(0.159, 0.021, -0.243), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_c", linear=(-0.056, -0.047, -0.443), angular=(0.0, 0.0, 0.0))
    api.set_gravity((0.0, 0.0, -1.8))
    api.set_duration(4.9)
    api.add_camera("cam0", pos=(0.044, -3.598, 1.01), target=(0.073, -0.068, 0.666))

SCENE_MANIFEST = {
    "scene_kind": "throwing_pit_board",
    "balls": ["ball_a", "ball_b", "ball_c"],
    "platforms": ["left_platform", "middle_platform", "right_platform"],
    "pits": {
        "left": {"x": [-0.58, -0.30]},
        "middle": {"x": [0.50, 0.67]},
        "right": {"x": [1.33, 1.50]},
    },
    "walls": ["black_left_wall", "orange_wall", "blue_wall", "pink_right_wall"],
    "region_kind": "pit",
    "families": ["throwing_final_destination", "throwing_current_run_judgment",
                 "counterfactual_destination", "inverse_intervention"],
}