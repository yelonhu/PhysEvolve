"""Canonical data schema for the engine layer.

This module is the single source of truth for every dataclass and enum that
crosses an engine boundary. All other modules import from here; nothing
here imports from anywhere else inside ``physevolve``.

Grid discretization, parameter levels, and observation fields are defined
here. Changes must preserve the executor and training contracts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

# ─── Enums ──────────────────────────────────────────────────────────────


class GeomType(str, Enum):
    """Primitive geom kinds exposed by the DSL.

    The string value matches MuJoCo MJCF ``type="..."`` tokens so the
    compiler can consume ``GeomType.BOX.value`` directly.
    """

    BOX = "box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CAPSULE = "capsule"
    PLANE = "plane"


class JointType(str, Enum):
    """Joint kinds exposed by the DSL.

    ``FIXED`` is a logical marker; it is rendered as the absence of any
    ``<joint>`` element under the body in MJCF (welded to parent).
    """

    FREE = "free"
    HINGE = "hinge"
    SLIDE = "slide"
    BALL = "ball"
    FIXED = "fixed"


class BodyStatus(str, Enum):
    """Final-frame classification used by the dense forward reward.

    See ``simulator.py`` for the classification algorithm and thresholds.
    """

    STABLE = "stable"
    MOVING = "moving"
    FALLEN = "fallen"
    EJECTED = "ejected"


# ─── Discretization constants ────────────────────

# 16³ workspace grid. Integer cell coords ∈ [0, 15].
GRID_RES: int = 16
WORKSPACE_X_HALF: float = 1.5
WORKSPACE_Y_HALF: float = 1.5
WORKSPACE_Z_MAX: float = 3.0
GRID_CELL: float = 0.1875  # 3.0 m / 16 cells; also (2*1.5)/16 for x/y.

# Discrete mass/friction levels (level 1..5).
MASS_LEVELS: tuple[float, ...] = (0.1, 0.5, 1.0, 5.0, 10.0)
FRICTION_LEVELS: tuple[float, ...] = (0.05, 0.2, 0.5, 1.0, 2.0)
# Discrete restitution levels. The DSL
# snaps any in-range [0, 1] value to the nearest level, analogously to
# ``MASS_LEVELS`` / ``FRICTION_LEVELS``.
RESTITUTION_LEVELS: tuple[float, ...] = (0.0, 0.5, 0.9)


def world_to_grid(pos: tuple[float, float, float]) -> tuple[int, int, int]:
    """Map a 3D world-frame position (meters) to a 16³ integer grid cell.

    Out-of-range coordinates are clipped into ``[0, 15]``; callers that
    need "out of workspace" semantics should consult
    :class:`BodyStatus.EJECTED` instead of reading the grid coord.

    Args:
        pos: ``(x, y, z)`` in meters. Workspace is
            ``x, y ∈ [-1.5, 1.5]`` and ``z ∈ [0, 3.0]``.

    Returns:
        Tuple of three ints in ``[0, 15]``.
    """

    x, y, z = pos
    gx = int(np.clip((x + WORKSPACE_X_HALF) / GRID_CELL, 0, GRID_RES - 1))
    gy = int(np.clip((y + WORKSPACE_Y_HALF) / GRID_CELL, 0, GRID_RES - 1))
    gz = int(np.clip(z / GRID_CELL, 0, GRID_RES - 1))
    return (gx, gy, gz)


def grid_to_world(grid: tuple[int, int, int]) -> tuple[float, float, float]:
    """Return the world-frame CENTRE of the ``grid`` cell.

    This is the canonical inverse of :func:`world_to_grid` for round-
    trip tests: every cell centre ``c`` satisfies
    ``world_to_grid(grid_to_world(g)) == g`` for ``g ∈ [0, 15]^3``.

    Out-of-range integer inputs raise :class:`ValueError` rather than
    clipping, since the cell-centre semantics are undefined outside
    the workspace.
    """

    gx, gy, gz = grid
    for name, g in (("gx", gx), ("gy", gy), ("gz", gz)):
        if not (0 <= int(g) < GRID_RES):
            raise ValueError(f"{name}={g} out of [0, {GRID_RES})")
    # Cell centres: ``(g + 0.5) * GRID_CELL`` in each axis, shifted
    # back to world coordinates.
    x = (int(gx) + 0.5) * GRID_CELL - WORKSPACE_X_HALF
    y = (int(gy) + 0.5) * GRID_CELL - WORKSPACE_Y_HALF
    z = (int(gz) + 0.5) * GRID_CELL
    return (float(x), float(y), float(z))


# ─── DSL primitives (MUTABLE builder state) ────────────────────────────
#
# These hold state while the Proposer's ``build_scene(api)`` runs. They are
# copied into the frozen ``SceneDef`` at ``PhysDSLApi.freeze()`` time.
# As mutable builder state, they use ``slots=True`` but not
# ``frozen=True``.


@dataclass(slots=True)
class BodyDef:
    """A single rigid body plus its primary geom and initial state."""

    name: str
    geom_type: GeomType
    pos: tuple[float, float, float]
    size: tuple[float, ...]
    mass: float
    rgba: tuple[float, float, float, float]
    friction_slide: float
    restitution: float
    joint_type: JointType = JointType.FREE
    fixed: bool = False
    initial_lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Orientation quaternion in MuJoCo w-first convention.
    initial_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(slots=True)
class JointDef:
    """A non-free joint linking two existing bodies (hinge / slide / ball)."""

    name: str
    joint_type: JointType
    parent: str
    child: str
    pos: tuple[float, float, float]
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    limit_range: Optional[tuple[float, float]] = None


@dataclass(slots=True)
class CameraDef:
    """Named camera defined by eye position and look-at target."""

    name: str
    pos: tuple[float, float, float]
    target: tuple[float, float, float]


@dataclass(slots=True)
class SpringDef:
    """A damped spring (spatial tendon) connecting two bodies.

    Emitted in MJCF as a ``<tendon><spatial .../></tendon>`` block with
    two ``<site>`` anchors (one per body, at the respective
    body-local ``anchor_*`` offset). ``rest_length`` maps to the MuJoCo
    ``springlength`` attribute; ``None`` lets MuJoCo infer the rest
    length from the initial pose.
    """

    name: str
    body_a: str
    body_b: str
    stiffness: float
    damping: float
    anchor_a: tuple[float, float, float] = (0.0, 0.0, 0.0)
    anchor_b: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rest_length: Optional[float] = None


@dataclass(slots=True)
class EnvConfig:
    """Global simulation settings (gravity, timestep, duration, frames)."""

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    timestep: float = 0.002
    duration: float = 8.0
    n_frames: int = 32


# ─── Compiled scene (FROZEN, passed to compiler/simulator) ─────────────


@dataclass(frozen=True, slots=True)
class SceneDef:
    """Immutable snapshot of the DSL builder state.

    This is what the compiler consumes. Ordering inside each tuple is
    significant (preserves Proposer-authored body order, which is part of
    the scene hash).
    """

    bodies: tuple[BodyDef, ...]
    joints: tuple[JointDef, ...]
    cameras: tuple[CameraDef, ...]
    env: EnvConfig
    # (x_half, y_half, z_max) in meters.
    workspace_bounds: tuple[float, float, float]
    # Empty by default so spring-less
    # scenes continue to hash identically in structure (though ``repr``
    # gains a trailing ``springs=()`` field that rebases scene_ids once).
    springs: tuple["SpringDef", ...] = ()


# ─── Simulation output ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BodyTrajectory:
    """Full per-step trajectory of a single body.

    Shapes:
        positions      (T, 3)       float64, world-frame meters
        orientations   (T, 4)       float64, quaternion w,x,y,z
        lin_velocities (T, 3)       float64, m/s
        ang_velocities (T, 3)       float64, rad/s

    ``T`` is the number of MuJoCo simulation steps
    (``env.duration / env.timestep``), not the number of rendered frames.
    """

    name: str
    positions: np.ndarray
    orientations: np.ndarray
    lin_velocities: np.ndarray
    ang_velocities: np.ndarray
    final_status: BodyStatus


@dataclass(frozen=True, slots=True)
class CollisionEvent:
    """Summary of a collision between two named bodies.

    Multiple contact-solver rows within a 10-step window are aggregated
    into one event; ``contact_force_norm`` is the max across that window.
    """

    t_step: int
    body_a: str
    body_b: str
    contact_force_norm: float


@dataclass(frozen=True, slots=True)
class SimState:
    """Canonical ground-truth output of :func:`simulate`.

    This is the ONLY object that crosses the engine→executor boundary.
    Every reward, every probe target, every Solver observation is derived
    from a ``SimState``.

    ``initial_positions`` / ``initial_lin_velocities`` are the **true
    t = 0** per-body state (sampled right after
    ``_apply_initial_conditions`` + ``mj_forward``, before the first
    ``mj_step``). Trajectory row 0, by contrast, is the state after the
    first step — close but not identical. Stage-2 ``SimStateView`` uses
    these fields to honour the observation contract that
    ``initial_*`` means "what the DSL author specified".
    """

    scene_id: str
    scene_def: SceneDef
    trajectories: dict[str, BodyTrajectory]
    collisions: tuple[CollisionEvent, ...]
    # Layout: (n_cameras, n_frames, H, W, 3) uint8.
    video_frames: np.ndarray
    camera_names: tuple[str, ...]
    sim_walltime_s: float
    # Per-body pre-step initial state. Keys match ``trajectories``.
    initial_positions: dict[str, np.ndarray]
    initial_lin_velocities: dict[str, np.ndarray]


# ─── Scene identity ────────────────────────────────────────────────────


def compute_scene_id(scene_def: SceneDef, seed: int) -> str:
    """Deterministic 16-character hex ID for ``(scene_def, seed)``.

    Uses ``sha256(repr(scene_def) + str(seed))[:16]``. Using ``repr`` relies
    on the frozen+slots dataclasses having a stable, recursive repr; this
    holds as long as every leaf field has a deterministic repr. All leaf
    fields here are tuples of primitives, enums, or floats — safe.

    Args:
        scene_def: Frozen scene description.
        seed: RNG seed that will be used by the simulator.

    Returns:
        16-character lowercase hex string.
    """

    payload = (repr(scene_def) + str(seed)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


__all__ = [
    # Enums
    "GeomType",
    "JointType",
    "BodyStatus",
    # Grid / discretization
    "GRID_RES",
    "WORKSPACE_X_HALF",
    "WORKSPACE_Y_HALF",
    "WORKSPACE_Z_MAX",
    "GRID_CELL",
    "MASS_LEVELS",
    "FRICTION_LEVELS",
    "RESTITUTION_LEVELS",
    "world_to_grid",
    "grid_to_world",
    # Builder state
    "BodyDef",
    "JointDef",
    "CameraDef",
    "SpringDef",
    "EnvConfig",
    # Frozen scene / sim output
    "SceneDef",
    "BodyTrajectory",
    "CollisionEvent",
    "SimState",
    "compute_scene_id",
]
