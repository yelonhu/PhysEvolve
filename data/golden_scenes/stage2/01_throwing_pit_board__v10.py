"""Golden scene variant: throwing_pit_board — wall_dense

Target families: throwing_final_destination, throwing_current_run_judgment,
    counterfactual_destination, inverse_intervention
Plan: denser floating-wall deflectors; all platforms / walls / balls raised +0.1 z;
    one white-gray wall plus red / blue / green walls; green / orange / purple balls.
"""

def build_scene(api):
    api.add_plane(name="ground", pos=(-0.013, 0.01, 0.01), size=(5.072, 5.041),
                  rgba=(0.583, 0.251, 0.865, 1.000), friction=0.549)
    # Bottom platforms; the gaps between them are the pits.
    api.add_box("left_platform", pos=(-0.909, -0.005, 0.259), size=(0.419, 0.142, 0.126),
                mass=9.93, rgba=(0.965, 0.456, 0.577, 1.000), friction=0.558,
                restitution=0.023, fixed=True)
    api.add_box("middle_platform", pos=(0.121, -0.054, 0.218), size=(0.417, 0.075, 0.091),
                mass=10.039, rgba=(0.541, 0.257, 0.857, 1.000), friction=0.545,
                restitution=0.034, fixed=True)
    api.add_box("right_platform", pos=(1.028, 0.072, 0.252), size=(0.261, 0.108, 0.031),
                mass=10.065, rgba=(0.980, 0.895, 0.197, 1.000), friction=0.53,
                restitution=0.0, fixed=True)
    # Slanted / vertical floating walls (deflectors; removable for cf / inverse).
    api.add_box("gray_left_wall", pos=(-1.375, 0.017, 0.717), size=(0.421, 0.08, 0.021),
                mass=9.966, rgba=(0.220, 0.730, 0.812, 1.000), friction=0.58,
                restitution=0.565, fixed=True)
    api.set_rotation("gray_left_wall", quaternion=(0.9067, 0.0128, 0.3709, 0.01))
    api.add_box("red_wall", pos=(-0.69, 0.062, 0.861), size=(0.036, 0.04, 0.332),
                mass=10.069, rgba=(0.907, 0.419, 0.591, 1.000), friction=0.472,
                restitution=0.432, fixed=True)
    api.add_box("blue_wall", pos=(0.373, 0.035, 0.854), size=(0.388, 0.022, 0.026),
                mass=9.942, rgba=(0.213, 0.769, 0.881, 1.000), friction=0.479,
                restitution=0.51, fixed=True)
    api.set_rotation("blue_wall", quaternion=(0.939, -0.0137, -0.3303, -0.0075))
    api.add_box("green_right_wall", pos=(1.401, -0.002, 0.694), size=(0.624, 0.019, 0.013),
                mass=10.028, rgba=(0.984, 0.540, 0.115, 1.000), friction=0.516,
                restitution=0.577, fixed=True)
    api.set_rotation("green_right_wall", quaternion=(0.948, -0.0185, -0.3724, -0.003))
    # Three launched balls. NEUTRAL names (ball_a/b/c) + distinct rgba: the
    # colour word in the question ("the green ball") is derived from the rgba,
    # never from the body name, so colour-permuted variants stay correct.
    api.add_sphere("ball_a", pos=(-1.265, 0.061, 1.492), radius=0.086, mass=1.022,
                   rgba=(0.225, 0.746, 0.300, 1.000), friction=0.243, restitution=0.624,
                   fixed=False)
    api.add_sphere("ball_b", pos=(0.054, -0.015, 1.507), radius=0.139, mass=0.956,
                   rgba=(0.936, 0.113, 0.158, 1.000), friction=0.194, restitution=0.908,
                   fixed=False)
    api.add_sphere("ball_c", pos=(1.093, 0.048, 1.328), radius=0.153, mass=1.016,
                   rgba=(0.928, 0.578, 0.149, 1.000), friction=0.248, restitution=0.849,
                   fixed=False)
    api.set_velocity("ball_a", linear=(0.252, 0.042, -0.158), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_b", linear=(0.05, 0.014, -0.197), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_c", linear=(-0.004, -0.074, -0.465), angular=(0.0, 0.0, 0.0))
    api.set_gravity((0.0, 0.0, -1.8))
    api.set_duration(4.9)
    api.add_camera("cam0", pos=(-0.028, -3.758, 1.308), target=(0.01, -0.029, 0.814))

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