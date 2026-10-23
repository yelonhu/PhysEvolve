"""Golden scene variant: throwing_pit_board — asymmetric_pits

Target families: throwing_final_destination, throwing_current_run_judgment,
    counterfactual_destination, inverse_intervention
Plan: middle_platform y half 0.08; purple_left_wall +0.15 x; steeper blue_wall (~50 deg);
    yellow vertical deflector at x=0.6; white / green / orange balls.
"""

def build_scene(api):
    api.add_plane(name="ground", pos=(-0.028, 0.0, 0.0), size=(5.036, 5.0),
                  rgba=(0.65, 0.65, 0.65, 1.0), friction=0.432)
    # Bottom platforms; the gaps between them are the pits.
    api.add_box("left_platform", pos=(-0.874, 0, 0.105), size=(0.339, 0.13, 0.105),
                mass=10.002, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.426,
                restitution=0.0, fixed=True)
    api.add_box("middle_platform", pos=(-0.019, 0, 0.105), size=(0.275, 0.13, 0.105),
                mass=10.078, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.44,
                restitution=0.0, fixed=True)
    api.add_box("right_platform", pos=(1, 0, 0.105), size=(0.35, 0.13, 0.105),
                mass=10.019, rgba=(0.34, 0.37, 0.39, 1.0), friction=0.483,
                restitution=0.076, fixed=True)
    # Slanted / vertical floating walls (deflectors; removable for cf / inverse).
    api.add_box("purple_left_wall", pos=(-1.154, 0, 0.62), size=(0.466, 0.055, 0.055),
                mass=9.908, rgba=(0.6, 0.2, 0.8, 1.0), friction=0.469,
                restitution=0.551, fixed=True)
    api.set_rotation("purple_left_wall", quaternion=(0.9239, 0.0, 0.3827, 0.0))
    api.add_box("yellow_wall", pos=(0.549, 0, 0.82), size=(0.048, 0.06, 0.315),
                mass=9.969, rgba=(0.95, 0.9, 0.2, 1.0), friction=0.508,
                restitution=0.43, fixed=True)
    api.add_box("blue_wall", pos=(-0.24, 0, 0.7), size=(0.343, 0.055, 0.055),
                mass=9.983, rgba=(0.2, 0.3, 0.9, 1.0), friction=0.47,
                restitution=0.514, fixed=True)
    api.set_rotation("blue_wall", quaternion=(0.9063, 0.0, -0.4226, 0.0))
    api.add_box("red_right_wall", pos=(1.242, 0, 0.64), size=(0.597, 0.06, 0.055),
                mass=9.939, rgba=(0.9, 0.15, 0.15, 1.0), friction=0.512,
                restitution=0.504, fixed=True)
    api.set_rotation("red_right_wall", quaternion=(0.9304, 0.0, -0.3665, 0.0))
    # Three launched balls. NEUTRAL names (ball_a/b/c) + distinct rgba: the
    # colour word in the question ("the green ball") is derived from the rgba,
    # never from the body name, so colour-permuted variants stay correct.
    api.add_sphere("ball_a", pos=(-1.285, 0, 1.3), radius=0.051, mass=1.077,
                   rgba=(0.95, 0.95, 0.95, 1.0), friction=0.139, restitution=0.687,
                   fixed=False)
    api.add_sphere("ball_b", pos=(-0.009, 0, 1.48), radius=0.029, mass=0.998,
                   rgba=(0.2, 0.75, 0.25, 1.0), friction=0.126, restitution=0.927,
                   fixed=False)
    api.add_sphere("ball_c", pos=(1.058, 0, 1.25), radius=0.145, mass=0.97,
                   rgba=(0.95, 0.55, 0.15, 1.0), friction=0.231, restitution=0.915,
                   fixed=False)
    api.set_velocity("ball_a", linear=(0.273, 0.0, -0.18), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_b", linear=(0.093, 0.0, -0.24), angular=(0.0, 0.0, 0.0))
    api.set_velocity("ball_c", linear=(-0.006, 0.0, -0.40), angular=(0.0, 0.0, 0.0))
    api.set_gravity((0.0, 0.0, -1.8))
    api.set_duration(4.9)
    api.add_camera("cam0", pos=(0.071, -3.70, 1.05), target=(-0.004, 0.0, 0.72))

SCENE_MANIFEST = {
    "scene_kind": "throwing_pit_board",
    "balls": ["ball_a", "ball_b", "ball_c"],
    "platforms": ["left_platform", "middle_platform", "right_platform"],
    "pits": {
        "left": {"x": [-0.58, -0.30]},
        "middle": {"x": [0.50, 0.67]},
        "right": {"x": [1.33, 1.50]},
    },
    "walls": ["purple_left_wall", "yellow_wall", "blue_wall", "red_right_wall"],
    "region_kind": "pit",
    "families": ["throwing_final_destination", "throwing_current_run_judgment",
                 "counterfactual_destination", "inverse_intervention"],
}