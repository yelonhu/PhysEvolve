"""Reusable Stage-2 Proposer physics construction modules.

These strings are prompt-facing only. They translate the owner-approved
few-shot examples into small, named construction patterns so main training still
gets useful code shape after the full golden program is removed.
"""

from __future__ import annotations


MODULES: dict[str, str] = {
    "collision_chain": """\
### collision_chain

Causal chain:
  A dynamic striker with moderate speed touches an intermediate body; contact
  transfers momentum into a target. Friction and restitution decide whether the
  target stops, keeps sliding, topples, or ends in contact with another body.
  Keep the chain nearly collinear in x, with positive gaps at t = 0, so the
  first contact is caused by motion rather than initial overlap.

Use as inspiration only; change >=3 of names, positions, sizes, friction.
Code fragment (inside build_scene; vary all names, coordinates, and parameters):
  api.add_box("striker_box", pos=(-0.85, 0.0, 0.11), size=(0.10, 0.06, 0.10),
              mass=1.0, rgba=(0.75, 0.35, 0.10, 1.0), friction=0.2,
              restitution=0.3, fixed=False)
  api.add_sphere("link_ball", pos=(-0.45, 0.0, 0.11), radius=0.10, mass=0.8,
                 rgba=(0.95, 0.55, 0.10, 1.0), friction=0.2,
                 restitution=0.4, fixed=False)
  api.add_box("target_panel", pos=(0.05, 0.0, 0.24), size=(0.05, 0.22, 0.24),
              mass=0.5, rgba=(0.10, 0.25, 0.90, 1.0), friction=0.5,
              restitution=0.0, fixed=False)
  api.set_velocity("striker_box", linear=(0.9, 0.0, 0.0),
                   angular=(0.0, 0.0, 0.0))
""",
    "hinge_striker": """\
### hinge_striker

Causal chain:
  A world-parent hinge converts gravity and angular motion into a repeatable
  striker arc. The hinge child must be dynamic and the parent must be "world";
  never use a plane name as a hinge parent. Place the struck body just outside
  the striker's resting pose so the contact happens after the swing begins.

Use as inspiration only; change >=3 of names, positions, sizes, friction.
Code fragment (inside build_scene; vary all names, coordinates, and parameters):
  pivot = (-0.95, 0.0, 0.62)
  api.add_box("pivot_post", pos=(-1.10, 0.0, 0.32), size=(0.04, 0.04, 0.32),
              mass=10.0, rgba=(0.55, 0.55, 0.58, 1.0), friction=0.5,
              restitution=0.0, fixed=True)
  api.add_capsule("hinge_arm", pos=(-0.98, 0.0, 0.36), radius=0.04,
                  half_height=0.22, mass=8.0,
                  rgba=(0.42, 0.42, 0.46, 1.0), friction=0.3,
                  restitution=0.05, fixed=False, axis="z")
  api.add_hinge("arm_pivot", parent="world", child="hinge_arm",
                pos=pivot, axis=(0.0, 1.0, 0.0))
""",
    "ramp_launch": """\
### ramp_launch

Causal chain:
  A fixed tilted ramp changes vertical drop into horizontal speed. A small
  down-ramp initial velocity keeps the motion reliable; gravity supplies the
  visible acceleration. The dynamic body must start above the ramp surface and
  move in x and z, not y-as-height.

Use as inspiration only; change >=3 of names, positions, sizes, friction.
Code fragment (inside build_scene; vary all names, coordinates, and parameters):
  api.add_box("ramp_surface", pos=(-0.85, 0.0, 0.24), size=(0.60, 0.22, 0.025),
              mass=10.0, rgba=(0.48, 0.36, 0.24, 1.0), friction=0.15,
              restitution=0.0, fixed=True)
  api.set_rotation("ramp_surface", quaternion=(0.96593, 0.0, 0.25882, 0.0))
  api.add_sphere("rolling_ball", pos=(-1.15, 0.0, 0.95), radius=0.09,
                 mass=1.0, rgba=(0.20, 0.75, 0.30, 1.0), friction=0.05,
                 restitution=0.8, fixed=False)
  api.set_velocity("rolling_ball", linear=(0.10, 0.0, -0.16),
                   angular=(0.0, 0.0, 0.0))
""",
    "friction_lane": """\
### friction_lane

Causal chain:
  Matched bodies with the same starting speed separate because friction changes
  energy loss. Higher friction -> lower final speed and shorter travel. Keep
  the lanes parallel or use a connected apparatus, but do not make one body
  start faster than the other.

Use as inspiration only; change >=3 of names, positions, sizes, friction.
Code fragment (inside build_scene; vary all names, coordinates, and parameters):
  api.add_box("smooth_lane", pos=(-0.20, -0.18, 0.01), size=(0.55, 0.08, 0.01),
              mass=10.0, rgba=(0.45, 0.35, 0.25, 1.0), friction=0.05,
              restitution=0.0, fixed=True)
  api.add_box("rough_lane", pos=(-0.20, 0.18, 0.01), size=(0.55, 0.08, 0.01),
              mass=10.0, rgba=(0.18, 0.16, 0.14, 1.0), friction=2.0,
              restitution=0.0, fixed=True)
  api.add_sphere("smooth_ball", pos=(-0.70, -0.18, 0.10), radius=0.09,
                 mass=1.0, rgba=(0.20, 0.75, 0.30, 1.0), friction=0.05,
                 restitution=0.5, fixed=False)
  api.add_sphere("rough_ball", pos=(-0.70, 0.18, 0.10), radius=0.09,
                 mass=1.0, rgba=(0.10, 0.25, 0.90, 1.0), friction=2.0,
                 restitution=0.5, fixed=False)
  api.set_velocity("smooth_ball", linear=(0.8, 0.0, 0.0),
                   angular=(0.0, 0.0, 0.0))
  api.set_velocity("rough_ball", linear=(0.8, 0.0, 0.0),
                   angular=(0.0, 0.0, 0.0))
""",
}


IDIOM_CARDS: dict[str, str] = {
    "slanted_deflector": """\
### slanted_deflector

Syntax idiom only - not a complete scene:
  api.add_box("<deflector>", pos=(x, y, z), size=(length, thickness_y, thickness_z),
              mass=<fixed_mass>, rgba=(...), friction=<f>, restitution=<r>,
              fixed=True)
  api.set_rotation("<deflector>", quaternion=(qw, 0.0, qy, 0.0))

Use this when a wall should change a ball's path. Choose a new angle, position,
length, colour, and restitution for each scene; do not leave every wall
axis-aligned.
""",
    "gap_pit_board": """\
### gap_pit_board

Syntax idiom only - not a complete scene:
  left_edge = platform_left.center_x + platform_left.half_x
  right_edge = platform_right.center_x - platform_right.half_x
  SCENE_MANIFEST["pits"][<region>] = {"x": [left_edge, right_edge]}

Non-runnable invariant:
  pit_between_ab = [center_x_a + half_x_a, center_x_b - half_x_b]

Use visible platform edges to define each pit interval. The interval must be a
gap between platforms, not a platform footprint and not an off-screen region.
Never use [center_x_a, center_x_b] as a pit interval, and keep every platform
body centre inside |x| < 1.5 with margin.
""",
    "multi_lane_ball_launch": """\
### multi_lane_ball_launch

Syntax idiom only - not a complete scene:
  api.add_sphere("<ball_i>", pos=(lane_x_i, lane_y_i, spawn_z_i), ...)
  api.set_velocity("<ball_i>", linear=(vx_i, 0.0, vz_i), angular=(wx_i, wy_i, wz_i))

Use separated lanes and non-identical velocities. Each ball should have its own
x start, colour, restitution, and modest x/z velocity so trajectories diverge.
""",
    "collision_chain": """\
### collision_chain

Syntax idiom only - not a complete scene:
  api.add_box("<striker>", pos=(x0, y0, z0), size=(sx0, sy0, sz0), fixed=False, ...)
  api.add_sphere("<intermediate>", pos=(x1, y1, z1), radius=r1, fixed=False, ...)
  api.add_box("<target>", pos=(x2, y2, z2), size=(sx2, sy2, sz2), fixed=False, ...)
  api.set_velocity("<striker>", linear=(vx, vy, vz), angular=(wx, wy, wz))

Keep positive t=0 gaps and one dominant lane of impact. Vary body shapes,
masses, restitution, and offsets so only one candidate receives the main impulse.
""",
    "hinge_striker": """\
### hinge_striker

Syntax idiom only - not a complete scene:
  pivot = (pivot_x, pivot_y, pivot_z)
  api.add_capsule("<arm>", pos=(arm_x, arm_y, arm_z), axis="<axis>", fixed=False, ...)
  api.add_hinge("<hinge>", parent="world", child="<arm>", pos=pivot, axis=(ax, ay, az))
  api.set_velocity("<arm>", linear=(0.0, 0.0, 0.0), angular=(wx, wy, wz))

Use a non-zero angular velocity or an off-rest pose. The struck body must sit in
the swing arc with a small positive initial gap.
""",
    "ramp_contact_transfer": """\
### ramp_contact_transfer

Syntax idiom only - not a complete scene:
  api.add_box("<ramp>", pos=(x, y, z), size=(length, width, thickness), fixed=True, ...)
  api.set_rotation("<ramp>", quaternion=(qw, qx, qy, qz))
  api.add_sphere("<rolling_body>", pos=(start_x, start_y, start_z), fixed=False, ...)
  api.add_box("<transfer_body>", pos=(contact_x, contact_y, contact_z), fixed=False, ...)

Use a ramp or surface to create motion, then a contact body or pad to change
speed. Do not reduce motion_lab to one isolated ball on one axis-aligned block.
""",
    "friction_compare_pair": """\
### friction_compare_pair

Syntax idiom only - not a complete scene:
  api.add_box("<lane_a>", pos=(x_a, y_a, z), friction=low_or_high, fixed=True, ...)
  api.add_box("<lane_b>", pos=(x_b, y_b, z), friction=other_level, fixed=True, ...)
  api.add_sphere("<body_a>", pos=(start_x_a, y_a, start_z), friction=level_a, fixed=False, ...)
  api.add_sphere("<body_b>", pos=(start_x_b, y_b, start_z), friction=level_b, fixed=False, ...)
  api.set_velocity("<body_a>", linear=(vx, vy, vz), angular=(wx, wy, wz))
  api.set_velocity("<body_b>", linear=(vx, vy, vz), angular=(wx, wy, wz))

Match geometry and starting speed; the answer should come from friction, not
from different launch speeds or same-position overlap.
""",
    "restitution_compare_pair": """\
### restitution_compare_pair

Syntax idiom only - not a complete scene:
  api.add_sphere("<body_a>", pos=(x_a, y_a, drop_z), restitution=level_a, fixed=False, ...)
  api.add_sphere("<body_b>", pos=(x_b, y_b, drop_z), restitution=level_b, fixed=False, ...)

Use matched drop height and matched body geometry. The higher-restitution body
must visibly rebound higher within the clip.
""",
}


SCENE_MODULE_KEYS: dict[str, tuple[str, ...]] = {
    "collision_arena": ("collision_chain", "hinge_striker"),
    "motion_lab": ("ramp_launch", "friction_lane"),
    "throwing_pit_board": (),
}


SCENE_IDIOM_KEYS: dict[str, tuple[str, ...]] = {
    "collision_arena": ("collision_chain", "hinge_striker"),
    "motion_lab": (
        "ramp_contact_transfer",
        "friction_compare_pair",
        "restitution_compare_pair",
    ),
    "throwing_pit_board": (
        "gap_pit_board",
        "slanted_deflector",
        "multi_lane_ball_launch",
    ),
}


def format_modules_for_scene(
    scene_kind: str,
    *,
    include_code_fragments: bool = True,
) -> str:
    """Return prompt-facing physics modules for ``scene_kind``."""

    if include_code_fragments:
        keys = SCENE_MODULE_KEYS.get(scene_kind, ())
        if not keys:
            return ""
        rendered = "\n\n".join(MODULES[key].rstrip() for key in keys)
        header = "## Physics Modules (optional partial snippets - NOT a full scene)"
        intro = (
            "Each module below is ONE reusable construction pattern you MAY include.\n"
            "Pick zero, one, or combine several - but never paste a module verbatim.\n"
            "A complete scene still needs Playbook geometry, manifest wiring, and "
            "your own layout.\n"
        )
    else:
        keys = SCENE_IDIOM_KEYS.get(scene_kind, ())
        if not keys:
            return ""
        rendered = "\n\n".join(IDIOM_CARDS[key].rstrip() for key in keys)
        header = "## Physics Modules (API idiom cards - syntax only)"
        intro = (
            "These cards teach local API call order without giving a runnable "
            "scene or manifest.\n"
            "Use placeholders as roles, then invent all concrete names, positions, "
            "sizes, colours, friction, restitution, velocities, camera, and timing.\n"
            "Prompt literal numbers are not defaults; repeated bodies must differ "
            "in at least one role-relevant value.\n"
        )
    return (
        f"\n{header}\n\n"
        f"{intro}"
        f"{rendered}\n"
    )


__all__ = [
    "IDIOM_CARDS",
    "MODULES",
    "SCENE_IDIOM_KEYS",
    "SCENE_MODULE_KEYS",
    "format_modules_for_scene",
]
