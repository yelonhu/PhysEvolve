"""Shared AST-level structural checks for PhysDSL fixtures and tests.

These helpers keep prompt smoke checks, fixture validation, and probing seed
checks aligned on how many bodies / joints / springs a ``build_scene`` emits.

Exposed surface:

* :func:`count_api_calls` — ``{method_name: call_count}``. Replaces
  older duplicated AST walkers and keeps structural checks in one place.
* :func:`classify_hinge_parent` — returns ``"world"`` / ``"body"`` /
  ``None`` for a single ``add_hinge`` call, used to discriminate the
  ``simple_pendulum_explicit`` (parent must be a registered body)
  vs. ``simple_pendulum_worldfixed`` (parent is the ``"world"``
  sentinel) categories.
* :func:`extract_scene_stats` — richer AST walk returning the
  per-body / per-joint / per-spring / per-velocity summary needed by
  the full 12-category constraint grid.
* :class:`StructuralConstraint` + :data:`CONSTRAINTS` + :func:`check_constraint`
  — structural invariant grid for the curated PhysDSL example categories.
  ``check_constraint`` returns ``(True, "")`` or
  ``(False, "<actionable violation message>")`` so the generator can
  log why a candidate was rejected.

Everything here is pure AST — no executor, no ``mujoco``.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Optional


# ─── AST parse helper ────────────────────────────────────────────────


def parse_build_scene(code: str) -> Optional[ast.AST]:
    """Return an :class:`ast.Module` for ``code`` or ``None`` on failure.

    Supports both shapes we see in practice:

    * Full ``<propose>`` body starting at ``def build_scene(api):`` —
      parses directly.
    * Indented body of ``build_scene`` only. ``ast.parse`` rejects indented
      top-level statements, so we retry with a synthetic
      ``def build_scene(api):`` header.

    Returns ``None`` if the code fails to parse under both strategies.
    """

    if not code:
        return None
    try:
        return ast.parse(code)
    except SyntaxError:
        pass
    lines = code.splitlines()
    if not any(line.startswith((" ", "\t")) for line in lines if line.strip()):
        lines = ["    " + line if line.strip() else line for line in lines]
    wrapped = "def build_scene(api):\n" + "\n".join(lines) + "\n"
    try:
        return ast.parse(wrapped)
    except SyntaxError:
        return None


def count_api_calls(code: str) -> Optional[dict[str, int]]:
    """Return ``{method_name: count}`` for each ``api.<method>(...)`` call.

    Walks the AST looking for :class:`ast.Call` nodes whose ``func`` is
    an :class:`ast.Attribute` rooted on the bare name ``api``. Returns
    ``None`` when the code fails to parse under both strategies in
    :func:`parse_build_scene`.
    """

    tree = parse_build_scene(code)
    if tree is None:
        return None
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        value = func.value
        if not isinstance(value, ast.Name) or value.id != "api":
            continue
        counts[func.attr] = counts.get(func.attr, 0) + 1
    return counts


# ─── Literal evaluator (constants + unary +/- + list/tuple) ──────────


def _literal_value(node: ast.AST) -> Any:
    """Return the Python value of a literal AST ``node`` if recoverable.

    Supports the shapes the Proposer emits:

    * :class:`ast.Constant` — ``5``, ``0.3``, ``"world"``, ``True`` …
    * :class:`ast.UnaryOp` with ``USub`` / ``UAdd`` wrapping a constant
      (the parser encodes ``-0.3`` this way).
    * :class:`ast.List` / :class:`ast.Tuple` of recursively-resolvable
      elements.

    Returns a sentinel ``_UNKNOWN`` object on anything else (variables,
    function calls, binary ops, …) so callers can tell "couldn't
    recover" apart from legitimate ``None`` / ``0`` values.
    """

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        operand = node.operand.value
        if isinstance(node.op, ast.USub):
            return -operand if isinstance(operand, (int, float)) else _UNKNOWN
        if isinstance(node.op, ast.UAdd):
            return +operand if isinstance(operand, (int, float)) else _UNKNOWN
        return _UNKNOWN
    if isinstance(node, (ast.List, ast.Tuple)):
        values = []
        for elt in node.elts:
            resolved = _literal_value(elt)
            if resolved is _UNKNOWN:
                return _UNKNOWN
            values.append(resolved)
        return tuple(values) if isinstance(node, ast.Tuple) else values
    return _UNKNOWN


class _UnknownSentinel:
    """Singleton marker — see :func:`_literal_value`."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover — diagnostic only.
        return "<unknown>"


_UNKNOWN: Any = _UnknownSentinel()


# ─── Rich scene stats ────────────────────────────────────────────────


_API_METHOD_SIGNATURES: dict[str, tuple[str, ...]] = {
    "add_box": (
        "name", "pos", "size", "mass", "rgba", "friction", "restitution", "fixed",
    ),
    "add_sphere": (
        "name", "pos", "radius", "mass", "rgba", "friction", "restitution", "fixed",
    ),
    "add_cylinder": (
        "name", "pos", "radius", "half_height", "mass", "rgba", "friction",
        "restitution", "fixed",
    ),
    "add_capsule": (
        "name", "pos", "radius", "half_height", "mass", "rgba", "friction",
        "restitution", "fixed",
    ),
    "add_plane": ("name", "pos", "size", "rgba", "friction"),
    "set_velocity": ("body_name", "linear", "angular"),
    "set_rotation": ("body_name", "quaternion"),
    "add_hinge": ("name", "parent", "child", "pos", "axis", "limit"),
    "add_slide": ("name", "parent", "child", "pos", "axis", "limit"),
    "add_spring": (
        "name", "body_a", "body_b", "stiffness", "damping",
        "anchor_a", "anchor_b", "rest_length",
    ),
    "set_gravity": ("vector",),
    "set_duration": ("seconds",),
    "add_camera": ("name", "pos", "target"),
}


_DYNAMIC_BODY_METHODS: frozenset[str] = frozenset(
    {"add_box", "add_sphere", "add_cylinder", "add_capsule"}
)


def _call_args_as_kwargs(call: ast.Call, method: str) -> dict[str, ast.AST]:
    """Flatten positional + keyword args of ``call`` into a name→node map.

    Uses :data:`_API_METHOD_SIGNATURES` to map positional args to the
    canonical parameter names. Returns an empty dict for unknown
    methods.
    """

    sig = _API_METHOD_SIGNATURES.get(method, ())
    out: dict[str, ast.AST] = {}
    for i, arg in enumerate(call.args):
        if i < len(sig):
            out[sig[i]] = arg
    for kw in call.keywords:
        if kw.arg is not None:
            out[kw.arg] = kw.value
    return out


@dataclass(frozen=True, slots=True)
class BodyInfo:
    """One registered dynamic body (``add_box`` / ``add_sphere`` / …)."""

    name: str
    kind: str
    fixed: bool
    pos: Optional[tuple[float, float, float]]
    size_hint: Optional[float]  # Half-extent for box, radius for sphere/etc.


@dataclass(frozen=True, slots=True)
class PlaneInfo:
    """One ``add_plane`` call."""

    name: str


@dataclass(frozen=True, slots=True)
class JointInfo:
    """One ``add_hinge`` or ``add_slide`` call."""

    name: str
    kind: Literal["hinge", "slide"]
    parent: str
    child: str


@dataclass(frozen=True, slots=True)
class SpringInfo:
    """One ``add_spring`` call."""

    name: str
    body_a: str
    body_b: str


@dataclass(frozen=True, slots=True)
class VelocityInfo:
    """One ``set_velocity`` call; components default to 0 when omitted."""

    body: str
    linear: tuple[float, float, float]  # (vx, vy, vz)
    has_literal_linear: bool            # False if couldn't recover literal


@dataclass(frozen=True, slots=True)
class SceneStats:
    """Aggregate AST-derived view of a ``build_scene`` body."""

    bodies: tuple[BodyInfo, ...]
    planes: tuple[PlaneInfo, ...]
    hinges: tuple[JointInfo, ...]
    slides: tuple[JointInfo, ...]
    springs: tuple[SpringInfo, ...]
    velocities: tuple[VelocityInfo, ...]
    api_call_counts: dict[str, int] = field(default_factory=dict)

    # ─── Derived accessors ──────────────────────────────────────────

    @property
    def dynamic_bodies(self) -> tuple[BodyInfo, ...]:
        return tuple(b for b in self.bodies if not b.fixed)

    @property
    def fixed_bodies(self) -> tuple[BodyInfo, ...]:
        return tuple(b for b in self.bodies if b.fixed)

    def body_by_name(self, name: str) -> Optional[BodyInfo]:
        for b in self.bodies:
            if b.name == name:
                return b
        return None


def _coerce_pos(node: ast.AST) -> Optional[tuple[float, float, float]]:
    """Return ``(x, y, z)`` if ``node`` evaluates to a 3-vector of floats."""

    resolved = _literal_value(node)
    if resolved is _UNKNOWN or resolved is None:
        return None
    if isinstance(resolved, (list, tuple)) and len(resolved) == 3:
        try:
            return (float(resolved[0]), float(resolved[1]), float(resolved[2]))
        except (TypeError, ValueError):
            return None
    return None


def _coerce_bool(node: ast.AST) -> Optional[bool]:
    val = _literal_value(node)
    return bool(val) if isinstance(val, bool) else None


def _coerce_str(node: ast.AST) -> Optional[str]:
    val = _literal_value(node)
    return val if isinstance(val, str) else None


def _size_hint_from_kwargs(method: str, kw: dict[str, ast.AST]) -> Optional[float]:
    """Best-effort per-body "characteristic size" literal.

    Used only for heuristic z-stacking detection (we don't need
    sub-centimetre accuracy — we just want "are these two bodies
    vertically stacked?"). Returns ``None`` when no literal is
    recoverable.
    """

    if method == "add_box":
        size = kw.get("size")
        if size is not None:
            val = _literal_value(size)
            if isinstance(val, (list, tuple)) and len(val) == 3:
                try:
                    return max(float(val[0]), float(val[1]), float(val[2]))
                except (TypeError, ValueError):
                    return None
    if method in ("add_sphere", "add_cylinder", "add_capsule"):
        r = kw.get("radius")
        if r is not None:
            val = _literal_value(r)
            if isinstance(val, (int, float)):
                return float(val)
    return None


def extract_scene_stats(code: str) -> Optional[SceneStats]:
    """Parse ``code`` and pull every Proposer-emitted primitive into a stats object.

    Returns ``None`` on unparsable input. Malformed individual calls
    (e.g. ``add_hinge`` with a non-literal ``parent=``) are tolerated:
    we simply skip the unreadable pieces of that call.
    """

    tree = parse_build_scene(code)
    if tree is None:
        return None

    bodies: list[BodyInfo] = []
    planes: list[PlaneInfo] = []
    hinges: list[JointInfo] = []
    slides: list[JointInfo] = []
    springs: list[SpringInfo] = []
    velocities: list[VelocityInfo] = []
    counts: dict[str, int] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        receiver = func.value
        if not isinstance(receiver, ast.Name) or receiver.id != "api":
            continue
        method = func.attr
        counts[method] = counts.get(method, 0) + 1
        kw = _call_args_as_kwargs(node, method)

        if method in _DYNAMIC_BODY_METHODS:
            name_node = kw.get("name")
            name = _coerce_str(name_node) if name_node is not None else None
            if not name:
                continue  # Can't reference an unknown name elsewhere.
            pos_node = kw.get("pos")
            pos = _coerce_pos(pos_node) if pos_node is not None else None
            fixed_node = kw.get("fixed")
            fixed_val = _coerce_bool(fixed_node) if fixed_node is not None else None
            fixed = bool(fixed_val) if fixed_val is not None else False
            size_hint = _size_hint_from_kwargs(method, kw)
            bodies.append(
                BodyInfo(
                    name=name,
                    kind=method.removeprefix("add_"),
                    fixed=fixed,
                    pos=pos,
                    size_hint=size_hint,
                )
            )
        elif method == "add_plane":
            name_node = kw.get("name")
            name = _coerce_str(name_node) if name_node is not None else ""
            planes.append(PlaneInfo(name=name or ""))
        elif method in ("add_hinge", "add_slide"):
            name_node = kw.get("name")
            parent_node = kw.get("parent")
            child_node = kw.get("child")
            name = _coerce_str(name_node) if name_node is not None else ""
            parent = _coerce_str(parent_node) if parent_node is not None else ""
            child = _coerce_str(child_node) if child_node is not None else ""
            info = JointInfo(
                name=name or "",
                kind="hinge" if method == "add_hinge" else "slide",
                parent=parent or "",
                child=child or "",
            )
            (hinges if method == "add_hinge" else slides).append(info)
        elif method == "add_spring":
            name_node = kw.get("name")
            a_node = kw.get("body_a")
            b_node = kw.get("body_b")
            name = _coerce_str(name_node) if name_node is not None else ""
            body_a = _coerce_str(a_node) if a_node is not None else ""
            body_b = _coerce_str(b_node) if b_node is not None else ""
            springs.append(
                SpringInfo(name=name or "", body_a=body_a or "", body_b=body_b or "")
            )
        elif method == "set_velocity":
            body_node = kw.get("body_name")
            linear_node = kw.get("linear")
            body = _coerce_str(body_node) if body_node is not None else ""
            linear_val: Optional[tuple[float, float, float]] = None
            if linear_node is not None:
                linear_val = _coerce_pos(linear_node)
            has_literal = linear_val is not None
            vel = linear_val if linear_val is not None else (0.0, 0.0, 0.0)
            velocities.append(
                VelocityInfo(
                    body=body or "",
                    linear=vel,
                    has_literal_linear=has_literal,
                )
            )

    return SceneStats(
        bodies=tuple(bodies),
        planes=tuple(planes),
        hinges=tuple(hinges),
        slides=tuple(slides),
        springs=tuple(springs),
        velocities=tuple(velocities),
        api_call_counts=counts,
    )


# ─── Hinge-parent classifier (Step 2 API, used by eval + generator) ──


def classify_hinge_parent(
    code: str, hinge_name: str
) -> Optional[Literal["world", "body"]]:
    """Classify one hinge's ``parent=`` argument as ``"world"`` or ``"body"``.

    Returns ``None`` when:

    * ``code`` doesn't parse, OR
    * no ``api.add_hinge(...)`` with the requested ``name=`` exists, OR
    * the parent kwarg/positional arg is not a literal string.

    Used by the generator to tell ``simple_pendulum_explicit``
    (``"body"`` required) apart from ``simple_pendulum_worldfixed``
    (``"world"`` required).
    """

    stats = extract_scene_stats(code)
    if stats is None:
        return None
    for h in stats.hinges:
        if h.name == hinge_name:
            return "world" if h.parent == "world" else (
                "body" if h.parent else None
            )
    return None


# ─── StructuralConstraint + CONSTRAINTS grid + check_constraint ──────


@dataclass(frozen=True, slots=True)
class StructuralConstraint:
    """Per-category AST-checkable invariant.

    The numeric fields (min/max body / joint / spring counts) are the
    bread-and-butter of the grid; the flag fields capture the subtle
    per-category requirements that the guidance calls out explicitly
    (world-sentinel discrimination, stacking, velocity direction,
    hinge-chain shape, spring-chain connectivity, …).
    """

    category_id: str

    # Body / joint / spring counts.
    min_dynamic_bodies: int = 0
    max_dynamic_bodies: Optional[int] = None
    min_joints: int = 0                       # hinge + slide
    max_joints: Optional[int] = None
    min_hinges: int = 0
    max_hinges: Optional[int] = None
    min_slides: int = 0
    max_slides: Optional[int] = None
    min_springs: int = 0
    max_springs: Optional[int] = None

    # Hinge-parent policy.
    required_hinge_parent: Optional[Literal["world", "body"]] = None

    # Broad shape flags.
    requires_spring_and_joint: bool = False
    forbids_joints: bool = False
    forbids_springs: bool = False
    requires_plane: bool = False

    # Velocity policy (single_body_* + two_body_collision).
    # "any" / "none_or_vertical" / "has_nonvertical" / "aimed_at_other"
    velocity_policy: str = "any"

    # Per-category custom invariants.
    requires_z_stacking: bool = False
    requires_fixed_anchor: bool = False
    requires_double_hinge_chain: bool = False
    requires_hinge_plus_free_body: bool = False
    requires_spring_chain_connected: bool = False


# ─── Core checker ────────────────────────────────────────────────────


_VEL_EPS: float = 1e-6


def _is_zero(v: float) -> bool:
    return abs(v) < _VEL_EPS


def _nonzero_horizontal(v: tuple[float, float, float]) -> bool:
    return not _is_zero(v[0]) or not _is_zero(v[1])


def _check_velocity_policy(
    stats: SceneStats, policy: str
) -> Optional[str]:
    """Validate :attr:`StructuralConstraint.velocity_policy`.

    Returns an error message on violation or ``None`` on success.
    """

    if policy == "any":
        return None

    # "none_or_vertical" allows 0 set_velocity calls OR all calls have
    # zero x / y components.
    if policy == "none_or_vertical":
        for v in stats.velocities:
            if v.has_literal_linear and _nonzero_horizontal(v.linear):
                return (
                    f"set_velocity for {v.body!r} has non-vertical component "
                    f"{v.linear} — this category forbids lateral motion"
                )
            if not v.has_literal_linear:
                return (
                    f"set_velocity for {v.body!r} uses a non-literal "
                    "linear argument; unable to verify it is vertical-only"
                )
        return None

    if policy == "has_nonvertical":
        any_nonvertical = False
        for v in stats.velocities:
            if not v.has_literal_linear:
                continue
            if _nonzero_horizontal(v.linear):
                any_nonvertical = True
                break
        if not any_nonvertical:
            return (
                "expected at least one set_velocity with a non-zero "
                "horizontal (x or y) component"
            )
        return None

    if policy == "aimed_at_other":
        # Weaker but tractable proxy: require at least one set_velocity
        # with a non-zero horizontal component whose body is one of the
        # registered dynamic bodies. The executor will of course decide
        # whether the scene actually collides, but this catches the
        # common collapse mode "two bodies on a plane with zero
        # velocity" at AST time.
        dyn_names = {b.name for b in stats.dynamic_bodies}
        aimed = False
        for v in stats.velocities:
            if v.body in dyn_names and v.has_literal_linear and _nonzero_horizontal(v.linear):
                aimed = True
                break
        if not aimed:
            return (
                "expected at least one set_velocity on a dynamic body with "
                "a non-zero horizontal component aimed at the other body"
            )
        return None

    return f"unknown velocity_policy {policy!r}"


def _check_z_stacking(stats: SceneStats) -> Optional[str]:
    """Return an error message iff the bodies do NOT look z-stacked.

    Heuristic: some pair of dynamic bodies must share a
    ``|dx|, |dy| < 0.35 m`` footprint with ``dz > 0.05 m``. This is a
    deliberately loose proxy — we only want to reject the
    ``two bodies side by side on the ground`` collapse.
    """

    dyn = [b for b in stats.dynamic_bodies if b.pos is not None]
    if len(dyn) < 2:
        return "stacking requires at least two dynamic bodies with literal positions"
    for i in range(len(dyn)):
        for j in range(i + 1, len(dyn)):
            ax, ay, az = dyn[i].pos  # type: ignore[misc]
            bx, by, bz = dyn[j].pos  # type: ignore[misc]
            if abs(ax - bx) < 0.35 and abs(ay - by) < 0.35 and abs(az - bz) > 0.05:
                return None
    return (
        "no pair of dynamic bodies is vertically stacked "
        "(expected |dx|,|dy| < 0.35 m and |dz| > 0.05 m for some pair)"
    )


def _check_fixed_anchor(stats: SceneStats) -> Optional[str]:
    if any(b.fixed for b in stats.bodies):
        return None
    return "expected at least one body with fixed=True to serve as the anchor"


def _check_double_hinge_chain(stats: SceneStats) -> Optional[str]:
    """Invariant for ``double_pendulum``.

    * exactly 2 hinges.
    * first hinge's parent is a fixed body OR the ``"world"`` sentinel.
    * second hinge's parent is the first hinge's child.
    """

    if len(stats.hinges) < 2:
        return f"expected ≥2 hinges for a double pendulum, got {len(stats.hinges)}"
    h1, h2 = stats.hinges[0], stats.hinges[1]
    p1 = h1.parent
    if p1 == "world":
        pass
    else:
        parent_body = stats.body_by_name(p1)
        if parent_body is None or not parent_body.fixed:
            return (
                f"first hinge parent must be a fixed body or 'world' "
                f"(got parent={p1!r}, "
                f"{'unknown body' if parent_body is None else 'body.fixed=False'})"
            )
    if h2.parent != h1.child:
        return (
            f"second hinge parent must equal first hinge's child "
            f"(got parent={h2.parent!r}, expected {h1.child!r})"
        )
    return None


def _check_hinge_plus_free_body(stats: SceneStats) -> Optional[str]:
    """Invariant for ``pendulum_strikes_free``.

    * ≥ 3 bodies total.
    * ≥ 1 hinge whose (parent, child) are both body-names.
    * at least one dynamic body is NOT mentioned by any hinge / slide.
    """

    if len(stats.hinges) < 1:
        return "pendulum_strikes_free requires at least one hinge"
    hinge_bodies: set[str] = set()
    for h in stats.hinges:
        hinge_bodies.add(h.parent)
        hinge_bodies.add(h.child)
    for s in stats.slides:
        hinge_bodies.add(s.parent)
        hinge_bodies.add(s.child)
    free_dyn = [b for b in stats.dynamic_bodies if b.name not in hinge_bodies]
    if not free_dyn:
        return (
            "no dynamic body is free of the hinge/slide graph — "
            "pendulum_strikes_free needs a third body in the swing path"
        )
    return None


def _check_spring_chain_connected(stats: SceneStats) -> Optional[str]:
    """Invariant for ``spring_chain``.

    * ≥ 3 dynamic bodies and ≥ 2 spring edges.
    * spring-edge-induced undirected graph contains a connected
      component spanning ≥ 3 of those bodies.
    """

    if len(stats.springs) < 2:
        return f"spring_chain requires ≥2 springs, got {len(stats.springs)}"
    dyn_names = {b.name for b in stats.dynamic_bodies}
    adj: dict[str, set[str]] = {}
    for s in stats.springs:
        if not s.body_a or not s.body_b:
            continue
        adj.setdefault(s.body_a, set()).add(s.body_b)
        adj.setdefault(s.body_b, set()).add(s.body_a)
    # BFS over each node to find the largest connected component size.
    best = 0
    seen_all: set[str] = set()
    for start in adj:
        if start in seen_all:
            continue
        comp: set[str] = set()
        frontier = [start]
        while frontier:
            node = frontier.pop()
            if node in comp:
                continue
            comp.add(node)
            frontier.extend(n for n in adj.get(node, ()) if n not in comp)
        seen_all |= comp
        dyn_comp = comp & dyn_names
        if len(dyn_comp) > best:
            best = len(dyn_comp)
    if best < 3:
        return (
            f"spring graph does not connect ≥3 dynamic bodies "
            f"(largest connected component has {best})"
        )
    return None


def check_constraint(
    code: str, constraint: StructuralConstraint
) -> tuple[bool, str]:
    """Validate ``code`` against one category's :class:`StructuralConstraint`.

    Returns ``(True, "")`` on success. On first violation returns
    ``(False, "<actionable message>")``. The generator logs the
    message so a failed attempt is diagnosable without re-running the
    AST walk by hand.
    """

    stats = extract_scene_stats(code)
    if stats is None:
        return False, "could not parse build_scene body"

    # Body counts.
    n_dyn = len(stats.dynamic_bodies)
    if n_dyn < constraint.min_dynamic_bodies:
        return False, (
            f"expected ≥{constraint.min_dynamic_bodies} dynamic bodies, "
            f"got {n_dyn}"
        )
    if (
        constraint.max_dynamic_bodies is not None
        and n_dyn > constraint.max_dynamic_bodies
    ):
        return False, (
            f"expected ≤{constraint.max_dynamic_bodies} dynamic bodies, "
            f"got {n_dyn}"
        )

    # Planes.
    if constraint.requires_plane and len(stats.planes) < 1:
        return False, "expected at least one add_plane call"

    # Joint counts.
    n_hinge = len(stats.hinges)
    n_slide = len(stats.slides)
    n_joint = n_hinge + n_slide
    if n_joint < constraint.min_joints:
        return False, (
            f"expected ≥{constraint.min_joints} joints (hinge+slide), "
            f"got {n_joint}"
        )
    if constraint.max_joints is not None and n_joint > constraint.max_joints:
        return False, (
            f"expected ≤{constraint.max_joints} joints (hinge+slide), "
            f"got {n_joint}"
        )
    if constraint.forbids_joints and n_joint > 0:
        return False, (
            f"this category forbids hinges/slides; found {n_joint}"
        )

    if n_hinge < constraint.min_hinges:
        return False, f"expected ≥{constraint.min_hinges} hinges, got {n_hinge}"
    if constraint.max_hinges is not None and n_hinge > constraint.max_hinges:
        return False, f"expected ≤{constraint.max_hinges} hinges, got {n_hinge}"
    if n_slide < constraint.min_slides:
        return False, f"expected ≥{constraint.min_slides} slides, got {n_slide}"
    if constraint.max_slides is not None and n_slide > constraint.max_slides:
        return False, f"expected ≤{constraint.max_slides} slides, got {n_slide}"

    # Springs.
    n_spring = len(stats.springs)
    if n_spring < constraint.min_springs:
        return False, f"expected ≥{constraint.min_springs} springs, got {n_spring}"
    if constraint.max_springs is not None and n_spring > constraint.max_springs:
        return False, f"expected ≤{constraint.max_springs} springs, got {n_spring}"
    if constraint.forbids_springs and n_spring > 0:
        return False, f"this category forbids springs; found {n_spring}"

    # Spring × joint cross.
    if constraint.requires_spring_and_joint:
        if n_spring == 0:
            return False, (
                "mixed_constraint requires ≥1 add_spring; none found"
            )
        if n_joint == 0:
            return False, (
                "mixed_constraint requires ≥1 hinge/slide; none found"
            )

    # Hinge-parent policy.
    if constraint.required_hinge_parent is not None:
        if n_hinge == 0:
            return False, (
                f"required_hinge_parent={constraint.required_hinge_parent!r} "
                f"but no add_hinge call present"
            )
        is_world = stats.hinges[0].parent == "world"
        parent_kind: Literal["world", "body"] = "world" if is_world else "body"
        if parent_kind != constraint.required_hinge_parent:
            got = stats.hinges[0].parent or "<unknown>"
            return False, (
                f"hinge parent must be "
                f"{constraint.required_hinge_parent!r}, got {got!r}"
            )

    # Velocity policy.
    vp_error = _check_velocity_policy(stats, constraint.velocity_policy)
    if vp_error is not None:
        return False, vp_error

    # Per-category custom checks.
    if constraint.requires_z_stacking:
        err = _check_z_stacking(stats)
        if err is not None:
            return False, err
    if constraint.requires_fixed_anchor:
        err = _check_fixed_anchor(stats)
        if err is not None:
            return False, err
    if constraint.requires_double_hinge_chain:
        err = _check_double_hinge_chain(stats)
        if err is not None:
            return False, err
    if constraint.requires_hinge_plus_free_body:
        err = _check_hinge_plus_free_body(stats)
        if err is not None:
            return False, err
    if constraint.requires_spring_chain_connected:
        err = _check_spring_chain_connected(stats)
        if err is not None:
            return False, err

    return True, ""


# ─── 12-category preset grid ─────────────────────────────────────────


CONSTRAINTS: dict[str, StructuralConstraint] = {
    "single_body_freefall": StructuralConstraint(
        category_id="single_body_freefall",
        min_dynamic_bodies=1,
        max_dynamic_bodies=1,
        requires_plane=True,
        forbids_joints=True,
        forbids_springs=True,
        velocity_policy="none_or_vertical",
    ),
    "single_body_projectile": StructuralConstraint(
        category_id="single_body_projectile",
        min_dynamic_bodies=1,
        max_dynamic_bodies=1,
        requires_plane=True,
        forbids_joints=True,
        forbids_springs=True,
        velocity_policy="has_nonvertical",
    ),
    "two_body_collision": StructuralConstraint(
        category_id="two_body_collision",
        min_dynamic_bodies=2,
        max_dynamic_bodies=2,
        requires_plane=True,
        forbids_joints=True,
        forbids_springs=True,
        velocity_policy="aimed_at_other",
    ),
    "stacking_free": StructuralConstraint(
        category_id="stacking_free",
        min_dynamic_bodies=2,
        max_dynamic_bodies=4,
        requires_plane=True,
        forbids_joints=True,
        forbids_springs=True,
        requires_z_stacking=True,
    ),
    "simple_pendulum_explicit": StructuralConstraint(
        category_id="simple_pendulum_explicit",
        min_dynamic_bodies=1,  # anchor is fixed, bob is dynamic
        max_dynamic_bodies=2,
        min_hinges=1,
        max_hinges=1,
        min_slides=0,
        max_slides=0,
        max_springs=0,
        forbids_springs=True,
        required_hinge_parent="body",
        requires_fixed_anchor=True,
    ),
    "simple_pendulum_worldfixed": StructuralConstraint(
        category_id="simple_pendulum_worldfixed",
        min_dynamic_bodies=1,
        max_dynamic_bodies=1,
        min_hinges=1,
        max_hinges=1,
        min_slides=0,
        max_slides=0,
        max_springs=0,
        forbids_springs=True,
        required_hinge_parent="world",
    ),
    "double_pendulum": StructuralConstraint(
        category_id="double_pendulum",
        min_dynamic_bodies=2,  # 3 bodies incl. optional anchor
        max_dynamic_bodies=3,
        min_hinges=2,
        max_hinges=2,
        min_slides=0,
        max_slides=0,
        max_springs=0,
        forbids_springs=True,
        requires_double_hinge_chain=True,
    ),
    "slider_mechanism": StructuralConstraint(
        category_id="slider_mechanism",
        min_dynamic_bodies=1,  # anchor may be fixed
        max_dynamic_bodies=4,
        min_slides=1,
        max_slides=2,
        min_hinges=0,
        max_hinges=0,
        max_springs=0,
        forbids_springs=True,
    ),
    "pendulum_strikes_free": StructuralConstraint(
        category_id="pendulum_strikes_free",
        min_dynamic_bodies=2,  # free body + bob (anchor may be fixed)
        min_hinges=1,
        max_hinges=2,
        min_slides=0,
        max_slides=0,
        max_springs=0,
        forbids_springs=True,
        requires_hinge_plus_free_body=True,
    ),
    "spring_two_body": StructuralConstraint(
        category_id="spring_two_body",
        min_dynamic_bodies=2,
        max_dynamic_bodies=2,
        min_springs=1,
        max_springs=1,
        min_hinges=0,
        max_hinges=0,
        min_slides=0,
        max_slides=0,
        forbids_joints=True,
    ),
    "spring_chain": StructuralConstraint(
        category_id="spring_chain",
        min_dynamic_bodies=3,
        min_springs=2,
        min_hinges=0,
        max_hinges=0,
        min_slides=0,
        max_slides=0,
        forbids_joints=True,
        requires_spring_chain_connected=True,
    ),
    "mixed_constraint": StructuralConstraint(
        category_id="mixed_constraint",
        min_dynamic_bodies=2,  # 3+ bodies total (may include fixed anchor)
        min_joints=1,
        min_springs=1,
        requires_spring_and_joint=True,
    ),
}


CATEGORY_TARGETS: dict[str, int] = {
    "single_body_freefall": 30,
    "single_body_projectile": 30,
    "two_body_collision": 40,
    "stacking_free": 30,
    "simple_pendulum_explicit": 30,
    "simple_pendulum_worldfixed": 20,
    "double_pendulum": 20,
    "slider_mechanism": 25,
    "pendulum_strikes_free": 30,
    "spring_two_body": 25,
    "spring_chain": 20,
    "mixed_constraint": 30,
}


__all__ = [
    "BodyInfo",
    "CATEGORY_TARGETS",
    "CONSTRAINTS",
    "JointInfo",
    "PlaneInfo",
    "SceneStats",
    "SpringInfo",
    "StructuralConstraint",
    "VelocityInfo",
    "check_constraint",
    "classify_hinge_parent",
    "count_api_calls",
    "extract_scene_stats",
    "parse_build_scene",
]
