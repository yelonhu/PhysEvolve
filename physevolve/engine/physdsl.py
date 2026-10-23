"""PhysDSL scene-building API.

Builder methods validate names, types, workspace bounds, and physical parameter
levels before updating the scene. Mutators return ``None``. ``freeze``
validates the completed scene and returns the engine representation.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Literal, Optional

from physevolve.engine.failure import DSLValidationError
from physevolve.engine.schema import (
    FRICTION_LEVELS,
    MASS_LEVELS,
    RESTITUTION_LEVELS,
    WORKSPACE_X_HALF,
    WORKSPACE_Y_HALF,
    WORKSPACE_Z_MAX,
    BodyDef,
    CameraDef,
    EnvConfig,
    GeomType,
    JointDef,
    JointType,
    SceneDef,
    SpringDef,
)

# ─── Module-level constants ───────────────────────────

_NAME_PATTERN = re.compile(r"[a-z_][a-z0-9_]*")
_RESERVED_NAMES: frozenset[str] = frozenset({"ground", "world", "sky"})

# Sentinel name accepted as the ``parent`` of a hinge/slide joint to mean
# "weld this body to the immovable worldbody root", saving authors from
# having to register a fake fixed anchor body. ``"world"`` is already in
# ``_RESERVED_NAMES`` so it can never collide with a user-registered body.
_WORLD_SENTINEL: str = "world"

# Body-center workspace. Position constraints are strict.
_POS_Z_MIN: float = 0.0
_POS_Z_MAX: float = 2.5

_MASS_MIN: float = 0.01
_MASS_MAX: float = 100.0
_FRICTION_MIN: float = 0.0
_FRICTION_MAX: float = 2.0
_RESTITUTION_MIN: float = 0.0
_RESTITUTION_MAX: float = 1.0
_SIZE_MIN: float = 0.01
_SIZE_MAX: float = 1.0
_VEL_MAX: float = 20.0  # per-component

_MIN_BODIES: int = 1
_MAX_BODIES: int = 20

# Spring tendon bounds. The stiffness/damping ranges are
# intentionally wide to accommodate both soft couplings (stiffness ~ 1
# N/m) and stiff chains. Anchors are body-local and typically small; the
# 2.0 m cap prevents pathological configurations.
_SPRING_STIFFNESS_MIN: float = 0.1
_SPRING_STIFFNESS_MAX: float = 1.0e4
_SPRING_DAMPING_MIN: float = 0.0
_SPRING_DAMPING_MAX: float = 1.0e3
_SPRING_ANCHOR_MAX: float = 2.0

# Auto-default camera.
_DEFAULT_CAMERA_POS: tuple[float, float, float] = (1.8, -1.8, 1.8)
_DEFAULT_CAMERA_TARGET: tuple[float, float, float] = (0.0, 0.0, 0.5)
_DEFAULT_CAMERA_NAME: str = "cam0"

# Auto-default ground plane.
_DEFAULT_GROUND_NAME: str = "ground"
_DEFAULT_GROUND_SIZE: tuple[float, float] = (5.0, 5.0)
_DEFAULT_GROUND_RGBA: tuple[float, float, float, float] = (0.85, 0.85, 0.85, 1.0)
_DEFAULT_GROUND_FRICTION: float = 0.5


def _approx_in(value: float, levels: tuple[float, ...]) -> bool:
    """Return True if ``value`` matches any entry in ``levels`` (abs tol 1e-6)."""

    return any(math.isclose(value, lv, rel_tol=1e-6, abs_tol=1e-6) for lv in levels)


def _snap_to_level(value: float, levels: tuple[float, ...]) -> float:
    """Snap ``value`` onto its matching level value (canonicalizes 0.10 → 0.1).

    Assumes ``_approx_in(value, levels)`` already returned True.
    """

    for lv in levels:
        if math.isclose(value, lv, rel_tol=1e-6, abs_tol=1e-6):
            return lv
    return value  # unreachable if caller validated first


# ─── Silent coercion layer ─────────────────────
#
# Design contract: coerce silently at the *syntactic / type / shape*
# boundary, but keep every physical bound (workspace, velocity, body
# count, level ranges) as a hard rejection. See docs/architecture.md for the
# full policy.


def _to_float(value: Any, *, label: str) -> float:
    """Return ``float(value)`` or raise ``DSLValidationError``.

    Rejects ``bool`` (Python's bool is an int subclass but is never a
    valid physics scalar), non-numeric types, NaN, and infinities.
    """

    if isinstance(value, bool):
        raise DSLValidationError(f"{label}={value!r} is not a number")
    if not isinstance(value, (int, float)):
        raise DSLValidationError(
            f"{label}={value!r} is not a number (got {type(value).__name__})"
        )
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise DSLValidationError(f"{label} is NaN or inf")
    return v


def _is_numeric_scalar(value: Any) -> bool:
    """True if ``value`` is an ``int`` or ``float`` (but not ``bool``)."""

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coerce_vec_n(value: Any, n: int, *, label: str) -> tuple[float, ...]:
    """Silently coerce ``value`` to an ``n``-tuple of floats.

    Accepts ``tuple | list | numpy.ndarray | any sized iterable`` of
    numeric elements. Over-long sequences are silently truncated to the
    first ``n`` components. Under-long sequences and non-iterable inputs
    raise :class:`DSLValidationError` (a missing axis is a real error,
    not a syntactic typo).
    """

    # Scalar numerics are never a valid length-n vector. Reject clearly.
    if _is_numeric_scalar(value):
        raise DSLValidationError(
            f"{label} must be a sequence of {n} numbers, got scalar {value!r}"
        )
    # ``numpy.ndarray`` without ``__len__`` (0-D) or strings should not
    # pass as vectors.
    if isinstance(value, (str, bytes)):
        raise DSLValidationError(
            f"{label} must be a numeric sequence, got {type(value).__name__}"
        )
    try:
        seq = list(value)
    except TypeError as exc:  # pragma: no cover - defensive
        raise DSLValidationError(
            f"{label} is not iterable ({type(value).__name__})"
        ) from exc
    if len(seq) < n:
        raise DSLValidationError(
            f"{label} must have at least {n} components, got {len(seq)} ({seq!r})"
        )
    coerced = tuple(_to_float(seq[i], label=f"{label}[{i}]") for i in range(n))
    return coerced


def _coerce_rgba(value: Any) -> tuple[float, float, float, float]:
    """Coerce to a 4-tuple RGBA. 3-tuples gain ``alpha=1.0`` silently."""

    if _is_numeric_scalar(value) or isinstance(value, (str, bytes)):
        raise DSLValidationError(
            f"rgba must be a 3- or 4-tuple of numbers, got {value!r}"
        )
    try:
        seq = list(value)
    except TypeError as exc:  # pragma: no cover - defensive
        raise DSLValidationError(
            f"rgba is not iterable ({type(value).__name__})"
        ) from exc
    if len(seq) == 3:
        seq = list(seq) + [1.0]
    if len(seq) != 4:
        raise DSLValidationError(
            f"rgba must have 3 or 4 components, got {len(seq)} ({seq!r})"
        )
    rgba = tuple(_to_float(seq[i], label=f"rgba[{i}]") for i in range(4))
    for i, c in enumerate(rgba):
        if not (0.0 <= c <= 1.0):
            raise DSLValidationError(f"rgba[{i}]={c!r} outside [0, 1]")
    return rgba  # type: ignore[return-value]


def _coerce_size(
    value: Any,
    *,
    expected_len: int,
    allow_max: bool = True,
    label: str = "size",
) -> tuple[float, ...]:
    """Coerce size to an ``expected_len``-tuple and range-check each component.

    Scalar values are silently expanded to a uniform tuple of length
    ``expected_len`` (e.g. ``size=0.5`` for a cube → ``(0.5, 0.5, 0.5)``).
    Over-long sequences are silently truncated. Range checks still
    hard-reject anything outside ``[_SIZE_MIN, _SIZE_MAX]``.
    """

    if _is_numeric_scalar(value):
        v = _to_float(value, label=label)
        size = tuple([v] * expected_len)
    else:
        size = _coerce_vec_n(value, expected_len, label=label)
    for i, s in enumerate(size):
        if s < _SIZE_MIN:
            raise DSLValidationError(
                f"{label}[{i}]={s!r} below minimum {_SIZE_MIN}"
            )
        if allow_max and s > _SIZE_MAX:
            raise DSLValidationError(
                f"{label}[{i}]={s!r} above maximum {_SIZE_MAX}"
            )
    return size


def _coerce_level(
    value: Any,
    levels: tuple[float, ...],
    *,
    lo: float,
    hi: float,
    label: str,
) -> float:
    """Range-check then silently snap ``value`` to the nearest of ``levels``.

    Hard-rejects values outside ``[lo, hi]``. Inside the range, returns
    the element of ``levels`` closest to ``value`` (ties broken by
    smallest level value, matching ``min(..., key=...)`` semantics).
    """

    v = _to_float(value, label=label)
    if not (lo <= v <= hi):
        raise DSLValidationError(
            f"{label} {v!r} outside [{lo}, {hi}]"
        )
    return min(levels, key=lambda lv: abs(lv - v))


def _coerce_continuous(
    value: Any,
    *,
    lo: float,
    hi: float,
    label: str,
) -> float:
    """Range-check ``value`` into ``[lo, hi]`` WITHOUT snapping to a level.

    The Stage-2 continuous-physics path uses this instead of
    :func:`_coerce_level`: mass / friction / restitution become free
    floats inside their physical bounds (hard-rejected outside), so the
    Proposer can express continuous values (e.g. ``mass=2.0``)
    rather than the discrete Stage-1 ladder.
    """

    v = _to_float(value, label=label)
    if not (lo <= v <= hi):
        raise DSLValidationError(f"{label} {v!r} outside [{lo}, {hi}]")
    return v


class PhysDSLApi:
    """Mutable DSL builder.

    Instantiated once per Proposer rollout; ``build_scene(api)`` is the
    single call that populates it; :meth:`freeze` turns it into an
    immutable :class:`SceneDef` consumed by the compiler and simulator.
    """

    __slots__ = (
        "_bodies",
        "_joints",
        "_cameras",
        "_env",
        "_used_names",
        "_plane_added",
        "_bodies_with_explicit_joint",
        "_springs",
        "continuous_levels",
    )

    def __init__(self, *, continuous_levels: bool = False) -> None:
        # Insertion-ordered dict preserves Proposer-authored body order
        # (which participates in the scene hash via SceneDef repr).
        self._bodies: dict[str, BodyDef] = {}
        self._joints: list[JointDef] = []
        self._cameras: list[CameraDef] = []
        self._env: EnvConfig = EnvConfig()
        # Global unique-name set across bodies, joints, cameras, springs.
        self._used_names: set[str] = set()
        self._plane_added: bool = False
        # Bodies that are the "child" of a non-free joint; they must NOT
        # receive a <freejoint/> at compile time.
        self._bodies_with_explicit_joint: set[str] = set()
        # Damped spring tendons.
        self._springs: list[SpringDef] = []
        # Stage-2 continuous-physics switch: when True, mass / friction /
        # restitution are range-checked but NOT snapped to discrete levels
        # (so continuous values like mass=2.0 survive). Stage-1 keeps
        # the discrete ladder (default False). Set by the executor per stage.
        self.continuous_levels: bool = bool(continuous_levels)

    # ───────────────────────── Geometry ────────────────────────────────

    def _phys_level(
        self,
        value: Any,
        levels: tuple[float, ...],
        *,
        lo: float,
        hi: float,
        label: str,
    ) -> float:
        """Coerce a physical scalar, honouring the continuous/discrete mode."""

        if self.continuous_levels:
            return _coerce_continuous(value, lo=lo, hi=hi, label=label)
        return _coerce_level(value, levels, lo=lo, hi=hi, label=label)

    def add_box(
        self,
        name: str,
        pos: tuple[float, float, float],
        size: tuple[float, float, float],
        mass: float = 1.0,
        rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0),
        friction: float = 0.5,
        restitution: float = 0.0,
        fixed: bool = False,
    ) -> None:
        """Add a box body.

        Args:
            name: Unique lowercase identifier.
            pos: World-frame center in meters.
            size: Half-extents ``(sx, sy, sz)`` in meters.
            mass: One of :data:`MASS_LEVELS` kg.
            rgba: Colour in ``[0, 1]^4``.
            friction: One of :data:`FRICTION_LEVELS`.
            restitution: ``[0, 1]``; 0 = perfectly inelastic.
            fixed: If True, body is welded to world.
        """

        self._validate_body_name(name)
        pos_t = self._validate_pos(pos)
        size_t = _coerce_size(size, expected_len=3, label="size")
        mass_level = self._phys_level(
            mass, MASS_LEVELS, lo=_MASS_MIN, hi=_MASS_MAX, label="mass"
        )
        rgba_t = _coerce_rgba(rgba)
        friction_level = self._phys_level(
            friction,
            FRICTION_LEVELS,
            lo=_FRICTION_MIN,
            hi=_FRICTION_MAX,
            label="friction",
        )
        rest_level = self._phys_level(
            restitution,
            RESTITUTION_LEVELS,
            lo=_RESTITUTION_MIN,
            hi=_RESTITUTION_MAX,
            label="restitution",
        )
        self._register_body(
            BodyDef(
                name=name,
                geom_type=GeomType.BOX,
                pos=pos_t,
                size=size_t,
                mass=mass_level,
                rgba=rgba_t,
                friction_slide=friction_level,
                restitution=rest_level,
                joint_type=JointType.FIXED if fixed else JointType.FREE,
                fixed=fixed,
            )
        )

    def add_sphere(
        self,
        name: str,
        pos: tuple[float, float, float],
        radius: float,
        mass: float = 1.0,
        rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0),
        friction: float = 0.5,
        restitution: float = 0.0,
        fixed: bool = False,
    ) -> None:
        """Add a sphere body. ``radius`` is stored as a 1-tuple in ``size``."""

        self._validate_body_name(name)
        pos_t = self._validate_pos(pos)
        r = _to_float(radius, label="radius")
        size_t = _coerce_size((r,), expected_len=1, label="size")
        mass_level = self._phys_level(
            mass, MASS_LEVELS, lo=_MASS_MIN, hi=_MASS_MAX, label="mass"
        )
        rgba_t = _coerce_rgba(rgba)
        friction_level = self._phys_level(
            friction,
            FRICTION_LEVELS,
            lo=_FRICTION_MIN,
            hi=_FRICTION_MAX,
            label="friction",
        )
        rest_level = self._phys_level(
            restitution,
            RESTITUTION_LEVELS,
            lo=_RESTITUTION_MIN,
            hi=_RESTITUTION_MAX,
            label="restitution",
        )
        self._register_body(
            BodyDef(
                name=name,
                geom_type=GeomType.SPHERE,
                pos=pos_t,
                size=size_t,
                mass=mass_level,
                rgba=rgba_t,
                friction_slide=friction_level,
                restitution=rest_level,
                joint_type=JointType.FIXED if fixed else JointType.FREE,
                fixed=fixed,
            )
        )

    def add_cylinder(
        self,
        name: str,
        pos: tuple[float, float, float],
        radius: float,
        half_height: float,
        mass: float = 1.0,
        rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0),
        friction: float = 0.5,
        restitution: float = 0.0,
        fixed: bool = False,
        axis: Literal["x", "y", "z"] = "z",
    ) -> None:
        """Add a cylinder body with axis aligned to ``axis`` in the body frame.

        The axis is realised by a unit-quaternion rotation stored on the
        body. At compile time the geom is emitted aligned with the body
        z-axis (MuJoCo convention) and the body frame itself is rotated.
        """

        self._validate_body_name(name)
        pos_t = self._validate_pos(pos)
        r = _to_float(radius, label="radius")
        h = _to_float(half_height, label="half_height")
        size_t = _coerce_size((r, h), expected_len=2, label="size")
        mass_level = self._phys_level(
            mass, MASS_LEVELS, lo=_MASS_MIN, hi=_MASS_MAX, label="mass"
        )
        rgba_t = _coerce_rgba(rgba)
        friction_level = self._phys_level(
            friction,
            FRICTION_LEVELS,
            lo=_FRICTION_MIN,
            hi=_FRICTION_MAX,
            label="friction",
        )
        rest_level = self._phys_level(
            restitution,
            RESTITUTION_LEVELS,
            lo=_RESTITUTION_MIN,
            hi=_RESTITUTION_MAX,
            label="restitution",
        )
        self._register_body(
            BodyDef(
                name=name,
                geom_type=GeomType.CYLINDER,
                pos=pos_t,
                size=size_t,
                mass=mass_level,
                rgba=rgba_t,
                friction_slide=friction_level,
                restitution=rest_level,
                joint_type=JointType.FIXED if fixed else JointType.FREE,
                fixed=fixed,
                initial_quat=_axis_to_quat(axis),
            )
        )

    def add_capsule(
        self,
        name: str,
        pos: tuple[float, float, float],
        radius: float,
        half_height: float,
        mass: float = 1.0,
        rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0),
        friction: float = 0.5,
        restitution: float = 0.0,
        fixed: bool = False,
        axis: Literal["x", "y", "z"] = "z",
    ) -> None:
        """Add a capsule body. See :meth:`add_cylinder` for axis semantics."""

        self._validate_body_name(name)
        pos_t = self._validate_pos(pos)
        r = _to_float(radius, label="radius")
        h = _to_float(half_height, label="half_height")
        size_t = _coerce_size((r, h), expected_len=2, label="size")
        mass_level = self._phys_level(
            mass, MASS_LEVELS, lo=_MASS_MIN, hi=_MASS_MAX, label="mass"
        )
        rgba_t = _coerce_rgba(rgba)
        friction_level = self._phys_level(
            friction,
            FRICTION_LEVELS,
            lo=_FRICTION_MIN,
            hi=_FRICTION_MAX,
            label="friction",
        )
        rest_level = self._phys_level(
            restitution,
            RESTITUTION_LEVELS,
            lo=_RESTITUTION_MIN,
            hi=_RESTITUTION_MAX,
            label="restitution",
        )
        self._register_body(
            BodyDef(
                name=name,
                geom_type=GeomType.CAPSULE,
                pos=pos_t,
                size=size_t,
                mass=mass_level,
                rgba=rgba_t,
                friction_slide=friction_level,
                restitution=rest_level,
                joint_type=JointType.FIXED if fixed else JointType.FREE,
                fixed=fixed,
                initial_quat=_axis_to_quat(axis),
            )
        )

    def add_plane(
        self,
        name: str = _DEFAULT_GROUND_NAME,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        size: tuple[float, float] = _DEFAULT_GROUND_SIZE,
        rgba: tuple[float, float, float, float] = _DEFAULT_GROUND_RGBA,
        friction: float = _DEFAULT_GROUND_FRICTION,
    ) -> None:
        """Add a ground plane.

        ``name`` defaults to the reserved identifier ``"ground"``. Planes
        are always fixed; they have no mass and no velocity state.
        """

        # Planes are allowed to reuse the reserved "ground" name, but
        # nothing else; any other name goes through the full name check.
        if name != _DEFAULT_GROUND_NAME:
            self._validate_body_name(name)
        else:
            if name in self._used_names:
                raise DSLValidationError(
                    f"duplicate name {name!r}: a body/joint/camera is already "
                    "using this identifier"
                )
        # Plane pos is allowed at z == 0 exactly (workspace floor).
        pos_t = self._validate_pos(pos, allow_z_zero=True)
        size_t = _coerce_size(
            size, expected_len=2, allow_max=False, label="size"
        )
        rgba_t = _coerce_rgba(rgba)
        friction_level = self._phys_level(
            friction,
            FRICTION_LEVELS,
            lo=_FRICTION_MIN,
            hi=_FRICTION_MAX,
            label="friction",
        )
        # Planes do not enforce mass levels (massless in MuJoCo).
        self._register_body(
            BodyDef(
                name=name,
                geom_type=GeomType.PLANE,
                pos=pos_t,
                size=size_t,
                mass=0.0,
                rgba=rgba_t,
                friction_slide=friction_level,
                restitution=0.0,
                joint_type=JointType.FIXED,
                fixed=True,
            )
        )
        self._plane_added = True

    # ──────────────── Initial conditions (per-body setters) ────────────

    def set_velocity(
        self,
        body_name: str,
        linear: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Set initial linear + angular velocity of an existing body."""

        body = self._require_body(body_name)
        if body.fixed:
            raise DSLValidationError(
                f"cannot set velocity on fixed body {body_name!r}"
            )
        lin_t = self._validate_velocity(linear, label="velocity")
        ang_t = self._validate_velocity(angular, label="velocity")
        body.initial_lin_vel = lin_t
        body.initial_ang_vel = ang_t

    def set_rotation(
        self,
        body_name: str,
        quaternion: tuple[float, float, float, float],
    ) -> None:
        """Set initial orientation quaternion (w, x, y, z) on an existing body.

        The quaternion is normalised before storage.
        """

        body = self._require_body(body_name)
        q = self._validate_quaternion(quaternion)
        body.initial_quat = q

    # ─────────── Joints (link two already-registered bodies) ───────────

    def add_hinge(
        self,
        name: str,
        parent: str,
        child: str,
        pos: tuple[float, float, float],
        axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        limit: Optional[tuple[float, float]] = None,
    ) -> None:
        """Add a 1-DOF rotational joint between two bodies."""

        self._add_joint(JointType.HINGE, name, parent, child, pos, axis, limit)

    def add_slide(
        self,
        name: str,
        parent: str,
        child: str,
        pos: tuple[float, float, float],
        axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        limit: Optional[tuple[float, float]] = None,
    ) -> None:
        """Add a 1-DOF prismatic joint between two bodies."""

        self._add_joint(JointType.SLIDE, name, parent, child, pos, axis, limit)

    # ────────────────── Springs (damped spatial tendons) ──────────────

    def add_spring(
        self,
        name: str,
        body_a: str,
        body_b: str,
        stiffness: float,
        damping: float,
        anchor_a: tuple[float, float, float] = (0.0, 0.0, 0.0),
        anchor_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rest_length: Optional[float] = None,
    ) -> None:
        """Connect two bodies with a damped spring (MJCF spatial tendon).

        Args:
            name: Unique identifier (shares the joint/body namespace).
            body_a: Name of the first body (already registered).
            body_b: Name of the second body (already registered; must
                differ from ``body_a``).
            stiffness: Spring constant in ``[_SPRING_STIFFNESS_MIN,
                _SPRING_STIFFNESS_MAX]``.
            damping: Damping coefficient in
                ``[_SPRING_DAMPING_MIN, _SPRING_DAMPING_MAX]``.
            anchor_a / anchor_b: Body-local attachment offsets (meters,
                each component in ``[-2.0, 2.0]``).
            rest_length: Optional natural length (meters, > 0); if
                omitted, MuJoCo infers from the initial pose.
        """

        self._validate_joint_name(name)
        if body_a == body_b:
            raise DSLValidationError(
                f"spring {name!r}: body_a and body_b must differ (both were {body_a!r})"
            )
        if body_a not in self._bodies:
            raise DSLValidationError(
                f"spring {name!r}: body_a {body_a!r} not registered"
            )
        if body_b not in self._bodies:
            raise DSLValidationError(
                f"spring {name!r}: body_b {body_b!r} not registered"
            )
        if self._bodies[body_a].fixed and self._bodies[body_b].fixed:
            raise DSLValidationError(
                f"spring {name!r}: at least one connected body must be dynamic"
            )
        # Planes are emitted as bare <geom> (no wrapping <body>) so they
        # cannot host the <site> anchors that a spatial tendon requires.
        # Reject them as spring endpoints.
        for side, ref in (("body_a", body_a), ("body_b", body_b)):
            if self._bodies[ref].geom_type is GeomType.PLANE:
                raise DSLValidationError(
                    f"spring {name!r}: {side}={ref!r} is a plane; planes "
                    "cannot be spring anchors"
                )
        k = _to_float(stiffness, label=f"spring {name!r} stiffness")
        if not (_SPRING_STIFFNESS_MIN <= k <= _SPRING_STIFFNESS_MAX):
            raise DSLValidationError(
                f"spring {name!r} stiffness {k!r} outside "
                f"[{_SPRING_STIFFNESS_MIN}, {_SPRING_STIFFNESS_MAX}]"
            )
        c = _to_float(damping, label=f"spring {name!r} damping")
        if not (_SPRING_DAMPING_MIN <= c <= _SPRING_DAMPING_MAX):
            raise DSLValidationError(
                f"spring {name!r} damping {c!r} outside "
                f"[{_SPRING_DAMPING_MIN}, {_SPRING_DAMPING_MAX}]"
            )
        anchor_a_t = _coerce_vec_n(
            anchor_a, 3, label=f"spring {name!r} anchor_a"
        )
        anchor_b_t = _coerce_vec_n(
            anchor_b, 3, label=f"spring {name!r} anchor_b"
        )
        for axis_name, anchor in (("anchor_a", anchor_a_t), ("anchor_b", anchor_b_t)):
            for i, comp in enumerate(anchor):
                if abs(comp) > _SPRING_ANCHOR_MAX:
                    raise DSLValidationError(
                        f"spring {name!r} {axis_name}[{i}]={comp!r} "
                        f"exceeds |v| <= {_SPRING_ANCHOR_MAX}"
                    )
        rest_t: Optional[float] = None
        if rest_length is not None:
            rest_t = _to_float(
                rest_length, label=f"spring {name!r} rest_length"
            )
            if rest_t <= 0.0:
                raise DSLValidationError(
                    f"spring {name!r}: rest_length must be > 0, got {rest_t!r}"
                )

        self._springs.append(
            SpringDef(
                name=name,
                body_a=body_a,
                body_b=body_b,
                stiffness=k,
                damping=c,
                anchor_a=anchor_a_t,  # type: ignore[arg-type]
                anchor_b=anchor_b_t,  # type: ignore[arg-type]
                rest_length=rest_t,
            )
        )
        self._used_names.add(name)

    # ─────────────────────── Environment setters ───────────────────────

    def set_gravity(self, vector: tuple[float, float, float]) -> None:
        """Override the gravity vector (m/s^2). Magnitude is unconstrained.

        Accepts any 3-length numeric sequence (tuple, list, ndarray, ...).
        """

        g = _coerce_vec_n(vector, 3, label="gravity")
        self._env.gravity = g  # type: ignore[assignment]

    def set_duration(self, seconds: float) -> None:
        """Override total simulation duration in seconds."""

        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
            raise DSLValidationError(f"duration must be a number, got {seconds!r}")
        if not (0.1 <= float(seconds) <= 60.0):
            raise DSLValidationError(
                f"duration {seconds!r} outside [0.1, 60.0] seconds"
            )
        self._env.duration = float(seconds)

    def add_camera(
        self,
        name: str,
        pos: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> None:
        """Add a named camera defined by eye position and look-at target."""

        self._validate_camera_name(name)
        pos_t = _coerce_vec_n(pos, 3, label=f"camera {name!r} pos")
        target_t = _coerce_vec_n(target, 3, label=f"camera {name!r} target")
        if pos_t == target_t:
            raise DSLValidationError(
                f"camera {name!r}: pos must differ from target"
            )
        self._cameras.append(
            CameraDef(name=name, pos=pos_t, target=target_t)  # type: ignore[arg-type]
        )
        self._used_names.add(name)

    # ───────────────────── Freeze into SceneDef ────────────────────────

    def freeze(self) -> SceneDef:
        """Snapshot the builder into a frozen :class:`SceneDef`.

        Auto-adds the default ground plane and default camera if the
        Proposer did not register any. Enforces body-count bounds
        (``1 <= n <= 20``)
        """

        if not self._plane_added:
            # Call via add_plane so the name-uniqueness machinery is
            # consistent; if the Proposer already used "ground" for
            # something non-plane, add_plane will correctly raise.
            self.add_plane()

        if not self._cameras:
            self.add_camera(
                _DEFAULT_CAMERA_NAME, _DEFAULT_CAMERA_POS, _DEFAULT_CAMERA_TARGET
            )

        n_dynamic = sum(1 for b in self._bodies.values() if not b.fixed)
        total_bodies = len(self._bodies)
        if total_bodies < _MIN_BODIES:
            raise DSLValidationError(
                f"scene must contain at least {_MIN_BODIES} body; got {total_bodies}"
            )
        # Body-count cap applies to non-fixed bodies (plane + fixed
        # platforms are scene furniture, not "bodies the Proposer is
        # testing"). The body-count cap includes such anchors so
        # e.g. a platform + 20 balls remains legal.
        if n_dynamic > _MAX_BODIES:
            raise DSLValidationError(
                f"too many dynamic bodies: {n_dynamic} > {_MAX_BODIES}"
            )

        # Spring sanity: every referenced body must exist (redundant
        # with add_spring but cheap and defensive).
        for spring in self._springs:
            for ref in (spring.body_a, spring.body_b):
                if ref not in self._bodies:
                    raise DSLValidationError(
                        f"spring {spring.name!r} references unknown body {ref!r}"
                    )

        return SceneDef(
            bodies=tuple(self._bodies.values()),
            joints=tuple(self._joints),
            cameras=tuple(self._cameras),
            env=EnvConfig(
                gravity=self._env.gravity,
                timestep=self._env.timestep,
                duration=self._env.duration,
                n_frames=self._env.n_frames,
            ),
            workspace_bounds=(WORKSPACE_X_HALF, WORKSPACE_Y_HALF, WORKSPACE_Z_MAX),
            springs=tuple(self._springs),
        )

    # ══════════════════════ Private helpers ════════════════════════════

    def _register_body(self, body: BodyDef) -> None:
        self._bodies[body.name] = body
        self._used_names.add(body.name)

    def _require_body(self, name: str) -> BodyDef:
        body = self._bodies.get(name)
        if body is None:
            raise DSLValidationError(f"unknown body {name!r}")
        return body

    def _add_joint(
        self,
        joint_type: JointType,
        name: str,
        parent: str,
        child: str,
        pos: tuple[float, float, float],
        axis: tuple[float, float, float],
        limit: Optional[tuple[float, float]],
    ) -> None:
        self._validate_joint_name(name)
        if parent == child:
            raise DSLValidationError(
                f"joint {name!r}: parent and child must differ (both were {parent!r})"
            )
        if child == _WORLD_SENTINEL:
            raise DSLValidationError(
                f"joint {name!r}: child cannot be {_WORLD_SENTINEL!r}; "
                f"only parent may use the world sentinel"
            )
        # ``parent="world"`` is the worldbody-root shortcut: no registered
        # body is required, the compiler keeps the child at <worldbody>
        # level and emits the non-free joint inline.
        if parent != _WORLD_SENTINEL and parent not in self._bodies:
            raise DSLValidationError(
                f"joint {name!r}: parent body {parent!r} not registered"
            )
        if child not in self._bodies:
            raise DSLValidationError(
                f"joint {name!r}: child body {child!r} not registered"
            )
        if child in self._bodies_with_explicit_joint:
            raise DSLValidationError(
                f"joint {name!r}: child {child!r} already has another joint"
            )
        pos_t = _coerce_vec_n(pos, 3, label=f"joint {name!r} pos")
        axis_t = _coerce_vec_n(axis, 3, label=f"joint {name!r} axis")
        if math.sqrt(sum(a * a for a in axis_t)) < 1e-6:
            raise DSLValidationError(f"joint {name!r}: axis must be non-zero")
        limit_t: Optional[tuple[float, float]] = None
        if limit is not None:
            limit_t = _coerce_vec_n(limit, 2, label=f"joint {name!r} limit")  # type: ignore[assignment]
            if limit_t[0] >= limit_t[1]:
                raise DSLValidationError(
                    f"joint {name!r}: limit must be (lo, hi) with lo < hi"
                )

        self._joints.append(
            JointDef(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                pos=pos_t,  # type: ignore[arg-type]
                axis=axis_t,  # type: ignore[arg-type]
                limit_range=limit_t,
            )
        )
        self._used_names.add(name)
        self._bodies_with_explicit_joint.add(child)
        # The child body is no longer a free-floating body; it owns a
        # non-free joint instead.
        child_body = self._bodies[child]
        child_body.joint_type = joint_type

    # ───────────── Validation primitives (raise DSLValidationError) ────

    def _validate_body_name(self, name: str) -> None:
        self._check_name_format(name)
        if name in _RESERVED_NAMES:
            raise DSLValidationError(
                f"name {name!r} is reserved (cannot be used for a body)"
            )
        if name in self._used_names:
            raise DSLValidationError(f"duplicate name {name!r}")

    def _validate_joint_name(self, name: str) -> None:
        self._check_name_format(name)
        if name in _RESERVED_NAMES:
            raise DSLValidationError(
                f"name {name!r} is reserved (cannot be used for a joint)"
            )
        if name in self._used_names:
            raise DSLValidationError(f"duplicate name {name!r}")

    def _validate_camera_name(self, name: str) -> None:
        self._check_name_format(name)
        if name in _RESERVED_NAMES:
            raise DSLValidationError(
                f"name {name!r} is reserved (cannot be used for a camera)"
            )
        if name in self._used_names:
            raise DSLValidationError(f"duplicate name {name!r}")

    @staticmethod
    def _check_name_format(name: str) -> None:
        if not isinstance(name, str):
            raise DSLValidationError(f"name must be a string, got {type(name).__name__}")
        if _NAME_PATTERN.fullmatch(name) is None:
            raise DSLValidationError(
                f"invalid name {name!r}: must match [a-z_][a-z0-9_]*"
            )

    @staticmethod
    def _validate_pos(
        pos: tuple[float, float, float],
        *,
        allow_z_zero: bool = False,
    ) -> tuple[float, float, float]:
        """Coerce ``pos`` to a float 3-tuple and enforce workspace bounds.

        Returns the canonicalized tuple so the caller can store it on
        the resulting :class:`BodyDef`.
        """

        pos_t = _coerce_vec_n(pos, 3, label="pos")
        x, y, z = pos_t
        if not (abs(x) < WORKSPACE_X_HALF):
            raise DSLValidationError(
                f"pos.x={x!r} outside workspace |x| < {WORKSPACE_X_HALF}"
            )
        if not (abs(y) < WORKSPACE_Y_HALF):
            raise DSLValidationError(
                f"pos.y={y!r} outside workspace |y| < {WORKSPACE_Y_HALF}"
            )
        z_lo_ok = (z >= _POS_Z_MIN) if allow_z_zero else (z > _POS_Z_MIN)
        if not z_lo_ok or not (z < _POS_Z_MAX):
            lo_symbol = ">=" if allow_z_zero else ">"
            raise DSLValidationError(
                f"pos.z={z!r} outside workspace "
                f"({lo_symbol} {_POS_Z_MIN} and < {_POS_Z_MAX})"
            )
        return pos_t  # type: ignore[return-value]

    @staticmethod
    def _validate_velocity(
        v: tuple[float, float, float], *, label: str = "velocity"
    ) -> tuple[float, float, float]:
        v_t = _coerce_vec_n(v, 3, label=label)
        for i, comp in enumerate(v_t):
            if abs(comp) > _VEL_MAX:
                raise DSLValidationError(
                    f"{label}[{i}]={comp!r} exceeds |v| <= {_VEL_MAX}"
                )
        return v_t  # type: ignore[return-value]

    @staticmethod
    def _validate_quaternion(
        q: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        q_t = _coerce_vec_n(q, 4, label="quaternion")
        norm = math.sqrt(sum(c * c for c in q_t))
        if norm < 1e-8:
            raise DSLValidationError(
                f"quaternion {q!r} has zero norm; cannot normalise"
            )
        return (q_t[0] / norm, q_t[1] / norm, q_t[2] / norm, q_t[3] / norm)


# ─── Axis → quaternion helper (module private) ─────────────────────────

_AXIS_QUATS: dict[str, tuple[float, float, float, float]] = {
    # Rotate body frame so its local z-axis aligns with the requested world
    # axis. "z" is identity; "x" is a +90° rotation about y; "y" is a -90°
    # rotation about x.
    "z": (1.0, 0.0, 0.0, 0.0),
    "x": (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0),
    "y": (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0),
}


def _axis_to_quat(axis: str) -> tuple[float, float, float, float]:
    """Map an ``"x" | "y" | "z"`` token to a unit quaternion."""

    if axis not in _AXIS_QUATS:
        raise DSLValidationError(
            f"axis {axis!r} must be one of {sorted(_AXIS_QUATS)}"
        )
    return _AXIS_QUATS[axis]


__all__ = ["PhysDSLApi"]
