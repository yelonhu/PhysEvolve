"""Deterministic Stage-2 executor repair utilities.

The repair layer is intentionally bounded: it fixes small compiler hygiene
issues (unsupported harmless kwargs, spawn clearance, simple t=0 overlaps) and
records every action. It never adds causal bodies, velocities, joints, or scene
topology.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from physevolve.engine.failure import DSLValidationError
from physevolve.engine.physdsl import PhysDSLApi
from physevolve.engine.schema import BodyDef, GeomType, JointDef, SceneDef


@dataclass(frozen=True, slots=True)
class RepairAction:
    """One deterministic repair action applied by the executor."""

    kind: str
    target: str
    tier: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict."""

        return {
            "kind": self.kind,
            "target": self.target,
            "tier": self.tier,
            "detail": dict(self.detail),
        }


@dataclass(slots=True)
class RepairTrace:
    """Mutable audit record for bounded repair attempts."""

    actions: list[RepairAction] = field(default_factory=list)

    def add(
        self,
        kind: str,
        target: str,
        *,
        tier: str = "minor",
        **detail: Any,
    ) -> None:
        """Append a repair action."""

        self.actions.append(
            RepairAction(kind=kind, target=target, tier=tier, detail=dict(detail))
        )

    @property
    def applied(self) -> bool:
        """Whether any repair action was applied."""

        return bool(self.actions)

    @property
    def max_tier(self) -> str:
        """Return the strongest tier present in the trace."""

        order = {"none": 0, "minor": 1, "manifest": 2, "geometry": 3}
        strongest = "none"
        for action in self.actions:
            if order.get(action.tier, 0) > order.get(strongest, 0):
                strongest = action.tier
        return strongest

    @property
    def reward_discount(self) -> float:
        """Conservative Proposer credit discount for repaired OK scenes."""

        if not self.applied:
            return 1.0
        if self.max_tier == "geometry":
            return 0.75
        if self.max_tier == "manifest":
            return 0.80
        return 0.90

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe trace record."""

        return {
            "applied": self.applied,
            "tier": self.max_tier,
            "reward_discount": self.reward_discount,
            "actions": [action.as_dict() for action in self.actions],
        }


@dataclass(frozen=True, slots=True)
class SceneKindRepairOverride:
    """Optional geometry-repair overrides for one Stage-2 scene kind."""

    scene_kind: str
    overlap_margin: Optional[float] = None
    overlap_iterations: Optional[int] = None


_DEFAULT_SCENE_KIND_REPAIR_OVERRIDES: tuple[SceneKindRepairOverride, ...] = (
    SceneKindRepairOverride(
        scene_kind="collision_arena",
        overlap_margin=0.03,
        overlap_iterations=6,
    ),
)


@dataclass(frozen=True, slots=True)
class RepairPolicy:
    """Knobs for deterministic bounded repair."""

    enabled: bool = True
    floor_margin: float = 0.02
    workspace_margin: float = 0.02
    overlap_margin: float = 0.02
    overlap_iterations: int = 4
    repair_fixed_floor_clearance: bool = False
    scene_kind_overrides: tuple[
        SceneKindRepairOverride, ...
    ] = _DEFAULT_SCENE_KIND_REPAIR_OVERRIDES

    def for_scene_kind(self, scene_kind: Optional[str]) -> "RepairPolicy":
        """Return this policy with only the matching scene-kind override applied."""

        if not scene_kind:
            return self
        for override in self.scene_kind_overrides:
            if override.scene_kind != scene_kind:
                continue
            changes: dict[str, Any] = {"scene_kind_overrides": ()}
            if override.overlap_margin is not None:
                changes["overlap_margin"] = float(override.overlap_margin)
            if override.overlap_iterations is not None:
                changes["overlap_iterations"] = int(override.overlap_iterations)
            return dataclasses.replace(self, **changes)
        return self


class RepairingPhysDSLApi:
    """A narrow proxy that tolerates harmless API shape mistakes.

    The wrapped :class:`PhysDSLApi` still performs all hard physics validation.
    This proxy only drops unsupported kwargs on ``add_plane`` and nudges obvious
    spawn coordinates away from strict workspace/floor boundaries before the
    inner API sees them.
    """

    def __init__(
        self,
        inner: PhysDSLApi,
        trace: RepairTrace,
        policy: RepairPolicy | None = None,
    ) -> None:
        self._inner = inner
        self._trace = trace
        self._policy = policy or RepairPolicy()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def freeze(self) -> SceneDef:
        """Forward freeze to the wrapped API."""

        return self._inner.freeze()

    def add_plane(self, *args: Any, **kwargs: Any) -> None:
        """Forward ``add_plane`` after dropping harmless non-plane kwargs."""

        method = self._inner.add_plane
        allowed = set(inspect.signature(method).parameters)
        dropped = sorted(k for k in kwargs if k not in allowed)
        if dropped:
            for key in dropped:
                kwargs.pop(key, None)
            self._trace.add(
                "drop_unsupported_kwarg",
                "add_plane",
                tier="minor",
                kwargs=dropped,
            )
        method(*args, **kwargs)

    def add_box(self, *args: Any, **kwargs: Any) -> None:
        self._call_geom("add_box", *args, **kwargs)

    def add_sphere(self, *args: Any, **kwargs: Any) -> None:
        self._call_geom("add_sphere", *args, **kwargs)

    def add_cylinder(self, *args: Any, **kwargs: Any) -> None:
        self._call_geom("add_cylinder", *args, **kwargs)

    def add_capsule(self, *args: Any, **kwargs: Any) -> None:
        self._call_geom("add_capsule", *args, **kwargs)

    def _call_geom(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        method = getattr(self._inner, method_name)
        bound = inspect.signature(method).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        name = str(bound.arguments.get("name", method_name))
        pos = _coerce_pos_tuple(bound.arguments.get("pos"))
        if pos is not None:
            new_pos = _repair_spawn_pos(
                method_name,
                pos,
                bound.arguments,
                self._policy,
            )
            if new_pos != pos:
                bound.arguments["pos"] = new_pos
                self._trace.add(
                    "repair_spawn_pos",
                    name,
                    tier="minor",
                    before=list(pos),
                    after=list(new_pos),
                )
        method(*bound.args, **bound.kwargs)


def repair_scene_def(
    scene_def: SceneDef,
    *,
    trace: RepairTrace,
    policy: RepairPolicy | None = None,
) -> SceneDef:
    """Return a repaired ``SceneDef`` copy.

    This is a pure transform: the incoming scene is not mutated.
    """

    policy = policy or RepairPolicy()
    if not policy.enabled:
        return scene_def

    repaired = _repair_body_positions(scene_def, trace=trace, policy=policy)
    repaired = _repair_simple_overlaps(repaired, trace=trace, policy=policy)
    return repaired


def _repair_body_positions(
    scene_def: SceneDef,
    *,
    trace: RepairTrace,
    policy: RepairPolicy,
) -> SceneDef:
    updated: list[BodyDef] = []
    for body in scene_def.bodies:
        if body.geom_type is GeomType.PLANE:
            updated.append(body)
            continue
        pos = body.pos
        new_pos = _clamp_workspace(pos, policy.workspace_margin)
        extent = _vertical_extent(body)
        if body.fixed and not policy.repair_fixed_floor_clearance:
            min_z = extent
        else:
            min_z = extent + policy.floor_margin
        if new_pos[2] < min_z:
            new_pos = (new_pos[0], new_pos[1], min_z)
        if new_pos != pos:
            trace.add(
                "repair_body_pos",
                body.name,
                tier="geometry",
                before=list(pos),
                after=list(new_pos),
            )
            updated.append(dataclasses.replace(body, pos=new_pos))
        else:
            updated.append(body)
    if all(new is old for new, old in zip(updated, scene_def.bodies, strict=True)):
        return scene_def
    return dataclasses.replace(scene_def, bodies=tuple(updated))


def _repair_simple_overlaps(
    scene_def: SceneDef,
    *,
    trace: RepairTrace,
    policy: RepairPolicy,
) -> SceneDef:
    bodies = list(scene_def.bodies)
    movable = _movable_body_names(scene_def)
    joints = list(scene_def.joints)

    for _ in range(max(0, int(policy.overlap_iterations))):
        changed = False
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                a, b = bodies[i], bodies[j]
                if a.geom_type is GeomType.PLANE or b.geom_type is GeomType.PLANE:
                    continue
                shift = _separation_shift(a, b, policy.overlap_margin)
                if shift is None:
                    continue
                move_idx: Optional[int] = None
                if b.name in movable:
                    move_idx = j
                elif a.name in movable:
                    move_idx = i
                    shift = (-shift[0], -shift[1], -shift[2])
                if move_idx is None:
                    continue
                old_body = bodies[move_idx]
                new_pos = _clamp_workspace(
                    (
                        old_body.pos[0] + shift[0],
                        old_body.pos[1] + shift[1],
                        old_body.pos[2] + shift[2],
                    ),
                    policy.workspace_margin,
                )
                min_z = _vertical_extent(old_body) + policy.floor_margin
                if new_pos[2] < min_z:
                    new_pos = (new_pos[0], new_pos[1], min_z)
                if new_pos == old_body.pos:
                    continue
                delta = (
                    new_pos[0] - old_body.pos[0],
                    new_pos[1] - old_body.pos[1],
                    new_pos[2] - old_body.pos[2],
                )
                bodies[move_idx] = dataclasses.replace(old_body, pos=new_pos)
                joints = _translate_body_joints(joints, old_body.name, delta)
                trace.add(
                    "separate_initial_overlap",
                    old_body.name,
                    tier="geometry",
                    other=b.name if old_body.name == a.name else a.name,
                    delta=list(delta),
                    after=list(new_pos),
                )
                changed = True
        if not changed:
            break
    return dataclasses.replace(scene_def, bodies=tuple(bodies), joints=tuple(joints))


def _movable_body_names(scene_def: SceneDef) -> set[str]:
    joint_children = {joint.child for joint in scene_def.joints}
    spring_refs = {
        ref
        for spring in scene_def.springs
        for ref in (spring.body_a, spring.body_b)
    }
    return {
        body.name
        for body in scene_def.bodies
        if (not body.fixed)
        and body.geom_type is not GeomType.PLANE
        and body.name not in joint_children
        and body.name not in spring_refs
    }


def _translate_body_joints(
    joints: list[JointDef],
    body_name: str,
    delta: tuple[float, float, float],
) -> list[JointDef]:
    out: list[JointDef] = []
    for joint in joints:
        if joint.child == body_name or joint.parent == body_name:
            out.append(
                dataclasses.replace(
                    joint,
                    pos=(
                        joint.pos[0] + delta[0],
                        joint.pos[1] + delta[1],
                        joint.pos[2] + delta[2],
                    ),
                )
            )
        else:
            out.append(joint)
    return out


def _separation_shift(
    a: BodyDef,
    b: BodyDef,
    margin: float,
) -> Optional[tuple[float, float, float]]:
    ea = _aabb_extents(a)
    eb = _aabb_extents(b)
    deltas = [b.pos[k] - a.pos[k] for k in range(3)]
    overlaps = [ea[k] + eb[k] + margin - abs(deltas[k]) for k in range(3)]
    if any(overlap <= 0.0 for overlap in overlaps):
        return None
    axis = min(range(3), key=lambda k: overlaps[k])
    sign = 1.0 if deltas[axis] >= 0.0 else -1.0
    if abs(deltas[axis]) < 1e-9:
        sign = 1.0
    shift = [0.0, 0.0, 0.0]
    shift[axis] = sign * overlaps[axis]
    return (shift[0], shift[1], shift[2])


def _aabb_extents(body: BodyDef) -> tuple[float, float, float]:
    if body.geom_type is GeomType.BOX and len(body.size) >= 3:
        return (float(body.size[0]), float(body.size[1]), float(body.size[2]))
    if body.geom_type is GeomType.SPHERE:
        r = float(body.size[0])
        return (r, r, r)
    if body.geom_type in (GeomType.CYLINDER, GeomType.CAPSULE) and len(body.size) >= 2:
        r = float(body.size[0])
        h = float(body.size[1])
        # Conservative axis-agnostic bound; avoids under-estimating rotated rods.
        return (r + h, r + h, r + h)
    return (0.0, 0.0, 0.0)


def _vertical_extent(body: BodyDef) -> float:
    if body.geom_type is GeomType.BOX and len(body.size) >= 3:
        return float(body.size[2])
    if body.geom_type is GeomType.SPHERE:
        return float(body.size[0])
    if body.geom_type is GeomType.CYLINDER and len(body.size) >= 2:
        return float(body.size[1])
    if body.geom_type is GeomType.CAPSULE and len(body.size) >= 2:
        return float(body.size[0]) + float(body.size[1])
    return 0.0


def _repair_spawn_pos(
    method_name: str,
    pos: tuple[float, float, float],
    arguments: dict[str, Any],
    policy: RepairPolicy,
) -> tuple[float, float, float]:
    new_pos = _clamp_workspace(pos, policy.workspace_margin)
    extent = _spawn_vertical_extent(method_name, arguments)
    if extent is not None:
        fixed = bool(arguments.get("fixed", False))
        if fixed and not policy.repair_fixed_floor_clearance:
            min_z = extent
        else:
            min_z = extent + policy.floor_margin
        if new_pos[2] < min_z:
            new_pos = (new_pos[0], new_pos[1], min_z)
    return new_pos


def _spawn_vertical_extent(
    method_name: str,
    arguments: dict[str, Any],
) -> Optional[float]:
    try:
        if method_name == "add_box":
            size = _coerce_float_tuple(arguments.get("size"))
            return float(size[2]) if size is not None and len(size) >= 3 else None
        if method_name == "add_sphere":
            return float(arguments["radius"])
        if method_name == "add_cylinder":
            axis = str(arguments.get("axis", "z"))
            return float(arguments["half_height"]) if axis == "z" else float(arguments["radius"])
        if method_name == "add_capsule":
            axis = str(arguments.get("axis", "z"))
            radius = float(arguments["radius"])
            half_height = float(arguments["half_height"])
            return radius + half_height if axis == "z" else radius
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _clamp_workspace(
    pos: tuple[float, float, float],
    margin: float,
) -> tuple[float, float, float]:
    x, y, z = pos
    hi_xy = 1.5 - margin
    lo_z = margin
    hi_z = 2.5 - margin
    return (
        min(max(float(x), -hi_xy), hi_xy),
        min(max(float(y), -hi_xy), hi_xy),
        min(max(float(z), lo_z), hi_z),
    )


def _coerce_pos_tuple(value: Any) -> Optional[tuple[float, float, float]]:
    vec = _coerce_float_tuple(value)
    if vec is None or len(vec) < 3:
        return None
    return (vec[0], vec[1], vec[2])


def _coerce_float_tuple(value: Any) -> Optional[tuple[float, ...]]:
    if isinstance(value, (str, bytes)) or value is None:
        return None
    try:
        seq = list(value)
    except TypeError:
        return None
    try:
        return tuple(float(v) for v in seq)
    except (TypeError, ValueError):
        return None


def wrap_physdsl_api(
    api: PhysDSLApi,
    *,
    trace: RepairTrace,
    policy: RepairPolicy | None = None,
) -> PhysDSLApi | RepairingPhysDSLApi:
    """Return a repairing proxy when enabled, else the original API."""

    active_policy = policy or RepairPolicy()
    if not active_policy.enabled:
        return api
    return RepairingPhysDSLApi(api, trace, active_policy)


__all__ = [
    "RepairAction",
    "RepairPolicy",
    "RepairTrace",
    "RepairingPhysDSLApi",
    "SceneKindRepairOverride",
    "repair_scene_def",
    "wrap_physdsl_api",
]
