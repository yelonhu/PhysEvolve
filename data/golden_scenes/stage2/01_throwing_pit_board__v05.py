"""Golden scene: throwing_pit_board

Source: physevolve/skills/scene_templates/throwing_pit_board.py
Families: throwing_final_destination, throwing_current_run_judgment, counterfactual_destination, inverse_intervention
Plan: three balls fall through a board of slanted coloured walls and drop into
the gaps between three bottom platforms; the same scene supports a destination /
yes-no / counterfactual / inverse question per ball.
"""

def build_scene(api):
    api.add_plane(name="ground", pos=(0.02, 0.0, 0.0), size=(5.071, 5.0),
                  rgba=(0.65, 0.65, 0.65, 1.0), friction=0.571)
    # Bottom platforms; the gaps between them are the pits.
    api.add_box("left_platform", pos=(-0.882, 0, 0.105), size=(0.408, 0.13, 0.105),
                mass=9.992, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.571,
                restitution=0.024, fixed=True)
    api.add_box("middle_platform", pos=(0.164, 0, 0.105), size=(0.338, 0.13, 0.105),
                mass=9.939, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.507,
                restitution=0.012, fixed=True)
    api.add_box("right_platform", pos=(1, 0, 0.105), size=(0.252, 0.13, 0.105),
                mass=9.947, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.567,
                restitution=0.043, fixed=True)
    # Slanted / vertical floating walls (deflectors; removable for cf / inverse).
    api.add_box("black_left_wall", pos=(-1.319, 0, 0.62), size=(0.442, 0.055, 0.055),
                mass=9.91, rgba=(0.01, 0.015, 0.018, 1.0), friction=0.42,
                restitution=0.559, fixed=True)
    api.set_rotation("black_left_wall", quaternion=(0.9239, 0.0, 0.3827, 0.0))
    api.add_box("orange_wall", pos=(-0.496, 0, 0.75), size=(0.011, 0.06, 0.315),
                mass=10.089, rgba=(0.95, 0.72, 0.10, 1.0), friction=0.466,
                restitution=0.574, fixed=True)
    api.add_box("blue_wall", pos=(0.206, 0, 0.7), size=(0.418, 0.055, 0.055),
                mass=10.106, rgba=(0.02, 0.20, 0.90, 1.0), friction=0.531,
                restitution=0.575, fixed=True)
    api.set_rotation("blue_wall", quaternion=(0.9455, 0.0, -0.3256, 0.0))
    api.add_box("pink_right_wall", pos=(1.242, 0, 0.64), size=(0.528, 0.06, 0.055),
                mass=9.915, rgba=(0.94, 0.45, 0.56, 1.0), friction=0.43,
                restitution=0.468, fixed=True)
    api.set_rotation("pink_right_wall", quaternion=(0.9304, 0.0, -0.3665, 0.0))
    # Three launched balls. NEUTRAL names (ball_a/b/c) + distinct rgba: the
    # colour word in the question ("the green ball") is derived from the rgba,
    # never from the body name, so colour-permuted variants stay correct.
    api.add_sphere("ball_a", pos=(-1.459, 0, 1.3), radius=0.113, mass=0.974,
                   rgba=(0.2, 0.75, 0.25, 1.0), friction=0.17, restitution=0.751,
                   fixed=False)
    api.add_sphere("ball_b", pos=(-0.053, 0, 1.48), radius=0.056, mass=0.997,
                   rgba=(0.95, 0.55, 0.15, 1.0), friction=0.233, restitution=0.829,
                   fixed=False)
    api.add_sphere("ball_c", pos=(0.925, 0, 1.25), radius=0.125, mass=1.055,
                   rgba=(0.2, 0.3, 0.9, 1.0), friction=0.123, restitution=0.946,
                   fixed=False)
    api.set_velocity("ball_a", linear=(0.239, 0.0, -0.18), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_b", linear=(0.113, 0.0, -0.24), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_c", linear=(-0.139, 0.0, -0.40), angular=(0.0, 0.0, 0.0))
    api.set_gravity((0.0, 0.0, -1.8))
    api.set_duration(4.9)
    api.add_camera("cam0", pos=(-0.073, -3.70, 1.05), target=(-0.051, 0.0, 0.72))

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