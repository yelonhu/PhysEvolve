"""Pure ``SceneDef`` to MJCF compilation using :mod:`lxml.etree`.

XML construction requires no MuJoCo runtime. Numeric values use
locale-independent ``%.6g`` formatting, and insertion-ordered attributes keep
serialization stable. Golden XML tests cover the generated structure.
"""

from __future__ import annotations

import math
from typing import Optional

from lxml import etree

from physevolve.engine.contact_params import restitution_to_mjcf_contact
from physevolve.engine.schema import (
    BodyDef,
    CameraDef,
    EnvConfig,
    GeomType,
    JointDef,
    JointType,
    SceneDef,
    SpringDef,
)

# ─── Locked MJCF constants ────────────────────────────

_NCONMAX: str = "500"
_NJMAX: str = "1000"
_OFFWIDTH: str = "480"
_OFFHEIGHT: str = "480"
# Key light: above + in front of the scene (dir has -Z for downward fill).
_LIGHT_POS: tuple[float, float, float] = (0.0, -2.0, 3.0)
_LIGHT_DIR: tuple[float, float, float] = (0.0, 0.5, -1.0)
_LIGHT_DIFFUSE: tuple[float, float, float] = (0.95, 0.95, 0.95)
_LIGHT_SPECULAR: tuple[float, float, float] = (0.2, 0.2, 0.2)
# Camera-aligned headlight fill (MJCF <visual><headlight>).
_HEADLIGHT_DIFFUSE: tuple[float, float, float] = (0.35, 0.35, 0.35)
_HEADLIGHT_AMBIENT: tuple[float, float, float] = (0.25, 0.25, 0.25)
_HEADLIGHT_SPECULAR: tuple[float, float, float] = (0.05, 0.05, 0.05)
# Offscreen render background (skybox; darker than stage2 ground 0.65).
_VISUAL_BACKGROUND_RGBA: tuple[float, float, float, float] = (0.45, 0.45, 0.45, 1.0)
_SKYBOX_TEXTURE_SIZE: str = "256"


# ─── Number formatting ─────────────────────────────────────────────────


def _fmt_float(v: float) -> str:
    """Format a float for MJCF.

    Uses ``%.6g`` for compact, locale-independent output while keeping
    integers distinguishable as floats (``1`` → ``1``, ``1.5`` → ``1.5``).
    Negative zero is folded to ``0`` to keep diffs clean.
    """

    if v == 0.0:
        return "0"
    s = f"{float(v):.6g}"
    # Normalise NumPy / locale-specific tokens (belt and braces).
    if s == "-0":
        return "0"
    return s


def _fmt_vec(vec: tuple[float, ...]) -> str:
    """Space-join a tuple of floats using :func:`_fmt_float`."""

    return " ".join(_fmt_float(v) for v in vec)


# ─── Look-at helper ────────────────────────────────────────────────────


def _lookat_to_xyaxes(
    pos: tuple[float, float, float],
    target: tuple[float, float, float],
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float, float, float]:
    """Convert an eye + look-at + world-up to MuJoCo's ``xyaxes`` format.

    MuJoCo cameras specify orientation by the first two rows of the
    camera→world rotation matrix. Row 0 is the camera's right axis (x);
    row 1 is the camera's up axis (y). MuJoCo infers the third axis (z,
    pointing backwards from the subject) by cross product.

    Args:
        pos: Camera eye in world frame.
        target: Point the camera looks at, in world frame.
        world_up: Which world direction is "up". Defaults to +z.

    Returns:
        ``(rx, ry, rz, ux, uy, uz)`` unit-length vectors as a flat tuple.
    """

    fx, fy, fz = (target[0] - pos[0], target[1] - pos[1], target[2] - pos[2])
    fnorm = math.sqrt(fx * fx + fy * fy + fz * fz)
    if fnorm < 1e-9:
        raise ValueError("look-at vector has zero length (pos == target)")
    # Forward (from eye into the scene) as unit vector.
    fx, fy, fz = fx / fnorm, fy / fnorm, fz / fnorm

    # Right = normalize(forward × world_up). When forward ∥ world_up (camera
    # pointing straight up/down), the cross product vanishes; fall back to
    # world +y as the up reference.
    upx, upy, upz = world_up
    rx = fy * upz - fz * upy
    ry = fz * upx - fx * upz
    rz = fx * upy - fy * upx
    rnorm = math.sqrt(rx * rx + ry * ry + rz * rz)
    if rnorm < 1e-6:
        upx, upy, upz = 0.0, 1.0, 0.0
        rx = fy * upz - fz * upy
        ry = fz * upx - fx * upz
        rz = fx * upy - fy * upx
        rnorm = math.sqrt(rx * rx + ry * ry + rz * rz)
    rx, ry, rz = rx / rnorm, ry / rnorm, rz / rnorm

    # Camera up = right × forward, guaranteed orthonormal.
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx
    return (rx, ry, rz, ux, uy, uz)


# ─── Sub-tree builders ─────────────────────────────────────────────────


def _build_option(env: EnvConfig) -> etree._Element:
    return etree.Element(
        "option",
        timestep=_fmt_float(env.timestep),
        gravity=_fmt_vec(env.gravity),
    )


def _build_size() -> etree._Element:
    return etree.Element("size", nconmax=_NCONMAX, njmax=_NJMAX)


def _build_visual() -> etree._Element:
    visual = etree.Element("visual")
    etree.SubElement(visual, "global", offwidth=_OFFWIDTH, offheight=_OFFHEIGHT)
    etree.SubElement(
        visual,
        "headlight",
        diffuse=_fmt_vec(_HEADLIGHT_DIFFUSE),
        ambient=_fmt_vec(_HEADLIGHT_AMBIENT),
        specular=_fmt_vec(_HEADLIGHT_SPECULAR),
    )
    return visual


def _build_asset() -> etree._Element:
    """Emit a flat skybox texture for the offscreen renderer background.

    MuJoCo 3.6 no longer accepts ``rgba`` on ``<visual><global>``; the first
    skybox texture sets the clear/background colour in ``mujoco.Renderer``.
    """

    r, g, b, _ = _VISUAL_BACKGROUND_RGBA
    rgb = _fmt_vec((r, g, b))
    asset = etree.Element("asset")
    etree.SubElement(
        asset,
        "texture",
        name="skybox",
        type="skybox",
        builtin="gradient",
        rgb1=rgb,
        rgb2=rgb,
        width=_SKYBOX_TEXTURE_SIZE,
        height=_SKYBOX_TEXTURE_SIZE,
    )
    return asset


def _build_default_light(parent: etree._Element) -> etree._Element:
    return etree.SubElement(
        parent,
        "light",
        pos=_fmt_vec(_LIGHT_POS),
        dir=_fmt_vec(_LIGHT_DIR),
        diffuse=_fmt_vec(_LIGHT_DIFFUSE),
        specular=_fmt_vec(_LIGHT_SPECULAR),
    )


def _geom_attrs(body: BodyDef) -> dict[str, str]:
    """Geom-element attribute dict for a single body."""

    attrs: dict[str, str] = {
        "name": f"{body.name}_geom",
        "type": body.geom_type.value,
        "rgba": _fmt_vec(body.rgba),
    }
    # Size encoding per MJCF geom semantics.
    if body.geom_type is GeomType.BOX:
        attrs["size"] = _fmt_vec(body.size)
    elif body.geom_type is GeomType.SPHERE:
        attrs["size"] = _fmt_float(body.size[0])
    elif body.geom_type is GeomType.CYLINDER or body.geom_type is GeomType.CAPSULE:
        # (radius, half-height)
        attrs["size"] = _fmt_vec(body.size)
    elif body.geom_type is GeomType.PLANE:
        # MJCF planes need a third "spacing" field; 0.1 is a reasonable default.
        sx, sy = body.size
        attrs["size"] = _fmt_vec((sx, sy, 0.1))
    else:  # pragma: no cover - enum is exhaustive
        raise ValueError(f"unsupported geom type {body.geom_type!r}")

    # Planes are massless / frictionless-default per MJCF conventions; adding
    # extra attributes on them is harmless but creates noise, so skip.
    if body.geom_type is not GeomType.PLANE:
        attrs["mass"] = _fmt_float(body.mass)
        # Three-component friction vector: slide, spin, roll. We expose a
        # single slide value on the DSL and fill the other two with the
        # MuJoCo defaults (0.005 spin, 0.0001 roll).
        attrs["friction"] = _fmt_vec((body.friction_slide, 0.005, 0.0001))
        if body.restitution > 0.0:
            for key, values in restitution_to_mjcf_contact(
                body.restitution
            ).items():
                attrs[key] = _fmt_vec(values)

    return attrs


def _spring_site_name(spring_name: str, side: str) -> str:
    """Deterministic ``<site>`` name for a spring anchor ("a" or "b" side)."""

    return f"spring__{spring_name}__{side}"


def _build_body(
    body: BodyDef,
    *,
    inner_joint: Optional[JointDef],
    parent_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spring_sites: tuple[tuple[str, tuple[float, float, float]], ...] = (),
) -> etree._Element:
    """Build a ``<body>`` element with its geom and (optional) joint.

    The element is returned WITHOUT any nested child bodies: the
    caller (:func:`_emit_body_subtree`) is responsible for recursing
    into joint-children and appending them at the right place. The
    joint element (if any) must always come BEFORE child ``<body>``
    elements per MJCF's expected order; we enforce this by inserting
    it here while the element is still child-free.

    Args:
        body: Frozen body description.
        inner_joint: Non-free joint that welds ``body`` to its parent
            (``None`` for root / free bodies).
        parent_pos: Absolute world-frame position of the enclosing
            ``<body>``, used to compute parent-relative ``pos`` for
            nested children. For root bodies this is ``(0, 0, 0)``.
    """

    rel_pos = (
        body.pos[0] - parent_pos[0],
        body.pos[1] - parent_pos[1],
        body.pos[2] - parent_pos[2],
    )

    body_attrs: dict[str, str] = {
        "name": body.name,
        "pos": _fmt_vec(rel_pos),
    }
    # Only emit quat when it's non-identity so snapshots stay tidy.
    if body.initial_quat != (1.0, 0.0, 0.0, 0.0):
        body_attrs["quat"] = _fmt_vec(body.initial_quat)

    body_el = etree.Element("body", **body_attrs)

    # Joint emission order:
    # - fixed body or body with a non-free inner joint → that joint (or none).
    # - dynamic body without any joint → <freejoint/>.
    # The joint MUST be emitted before the <geom> and before any nested
    # <body> children so the MJCF parser attaches it correctly.
    if inner_joint is not None:
        _attach_joint(body_el, inner_joint, child_pos=body.pos)
    elif not body.fixed and body.joint_type is JointType.FREE:
        etree.SubElement(body_el, "freejoint")

    # Spring anchors live on the body as <site>. They must precede the
    # <geom> so the MJCF ordering stays conventional (joint → site → geom).
    for site_name, anchor in spring_sites:
        etree.SubElement(
            body_el,
            "site",
            name=site_name,
            pos=_fmt_vec(anchor),
        )

    etree.SubElement(body_el, "geom", **_geom_attrs(body))
    return body_el


def _attach_joint(
    body_el: etree._Element,
    joint: JointDef,
    *,
    child_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Attach a non-free joint element as a child of ``body_el``.

    MJCF joint ``pos`` is interpreted in the **child** body's local
    frame (i.e. relative to the ``<body>`` element that contains it).
    The DSL authors joint positions in the absolute world frame
    (matching how humans think about pivots), so we subtract the
    child body's absolute position here.

    For world-welded joint children (no enclosing dynamic body tree —
    either because the parent body is itself fixed at the worldbody
    root, or because the DSL used the ``parent="world"`` shortcut),
    ``child_pos == body.pos`` (the body's world-frame pos) and the
    translation is straightforward.
    """

    type_token = joint.joint_type.value
    local_pos = (
        joint.pos[0] - child_pos[0],
        joint.pos[1] - child_pos[1],
        joint.pos[2] - child_pos[2],
    )
    attrs: dict[str, str] = {
        "name": joint.name,
        "type": type_token,
        "pos": _fmt_vec(local_pos),
        "axis": _fmt_vec(joint.axis),
    }
    if joint.limit_range is not None:
        attrs["limited"] = "true"
        attrs["range"] = _fmt_vec(joint.limit_range)
    etree.SubElement(body_el, "joint", **attrs)


def _build_camera(cam: CameraDef) -> etree._Element:
    xyaxes = _lookat_to_xyaxes(cam.pos, cam.target)
    return etree.Element(
        "camera",
        name=cam.name,
        pos=_fmt_vec(cam.pos),
        xyaxes=_fmt_vec(xyaxes),
    )


# ─── Public API ────────────────────────────────────────────────────────


def compile_to_mjcf(scene_def: SceneDef) -> str:
    """Compile a frozen :class:`SceneDef` into an MJCF XML string.

    This function is pure. Same input → byte-identical output. It does
    not instantiate :class:`mujoco.MjModel`; that happens in the
    simulator.

    Args:
        scene_def: Frozen scene description from :meth:`PhysDSLApi.freeze`.

    Returns:
        Unicode XML string. Starts with the root ``<mujoco>`` element
        (no ``<?xml ...?>`` declaration — MuJoCo parses either).
    """

    root = etree.Element("mujoco")
    root.append(_build_option(scene_def.env))
    root.append(_build_size())
    root.append(_build_visual())
    root.append(_build_asset())

    worldbody = etree.SubElement(root, "worldbody")
    _build_default_light(worldbody)

    # Build the hierarchy: a child body (per a JointDef) must be nested
    # inside its parent's <body> element, with joint pos/body pos in the
    # parent's local frame. Free / un-jointed bodies
    # remain flat at the worldbody level.
    by_name: dict[str, BodyDef] = {b.name: b for b in scene_def.bodies}
    joint_by_child: dict[str, JointDef] = {j.child: j for j in scene_def.joints}
    children_of: dict[str, list[str]] = {}
    for j in scene_def.joints:
        children_of.setdefault(j.parent, []).append(j.child)

    # Spring sites, grouped by body. Each entry is
    # ``(site_name, anchor_xyz)`` in the body-local frame. Emission is
    # order-preserving (spring declaration order).
    sites_by_body: dict[
        str, list[tuple[str, tuple[float, float, float]]]
    ] = {}
    for spring in scene_def.springs:
        sites_by_body.setdefault(spring.body_a, []).append(
            (_spring_site_name(spring.name, "a"), spring.anchor_a)
        )
        sites_by_body.setdefault(spring.body_b, []).append(
            (_spring_site_name(spring.name, "b"), spring.anchor_b)
        )

    # A "nested child" is any body that appears as ``child`` of a joint
    # whose ``parent`` is a registered body; such bodies MUST NOT be
    # emitted again at the worldbody level. Joints with ``parent="world"``
    # (world-parent joint) keep their child at the worldbody level instead;
    # ``_build_body`` will attach the non-free joint inline via the
    # ``inner_joint`` lookup, suppressing the default ``<freejoint/>``.
    nested_names: set[str] = {
        j.child for j in scene_def.joints if j.parent != "world"
    }

    # Planes are emitted as bare <geom> on the worldbody (MuJoCo convention)
    # rather than wrapped in a <body>; this saves a level and matches most
    # tutorial scenes.
    for body in scene_def.bodies:
        if body.name in nested_names:
            continue
        if body.geom_type is GeomType.PLANE:
            attrs = _geom_attrs(body)
            attrs["pos"] = _fmt_vec(body.pos)
            etree.SubElement(worldbody, "geom", **attrs)
            continue
        worldbody.append(
            _emit_body_subtree(
                body,
                by_name=by_name,
                joint_by_child=joint_by_child,
                children_of=children_of,
                sites_by_body=sites_by_body,
                parent_pos=(0.0, 0.0, 0.0),
            )
        )

    for cam in scene_def.cameras:
        worldbody.append(_build_camera(cam))

    # Tendons (damped springs). Emitted as a sibling of ``<worldbody>``
    # only when there is at least one spring; this keeps spring-less
    # scenes byte-identical to the golden XML.
    if scene_def.springs:
        tendon_block = etree.SubElement(root, "tendon")
        for spring in scene_def.springs:
            spatial_attrs: dict[str, str] = {
                "name": spring.name,
                "stiffness": _fmt_float(spring.stiffness),
                "damping": _fmt_float(spring.damping),
            }
            if spring.rest_length is not None:
                spatial_attrs["springlength"] = _fmt_float(spring.rest_length)
            spatial = etree.SubElement(tendon_block, "spatial", **spatial_attrs)
            etree.SubElement(
                spatial,
                "site",
                site=_spring_site_name(spring.name, "a"),
            )
            etree.SubElement(
                spatial,
                "site",
                site=_spring_site_name(spring.name, "b"),
            )

    xml_bytes = etree.tostring(root, pretty_print=True, encoding="utf-8")
    return xml_bytes.decode("utf-8")


def _emit_body_subtree(
    body: BodyDef,
    *,
    by_name: dict[str, BodyDef],
    joint_by_child: dict[str, JointDef],
    children_of: dict[str, list[str]],
    sites_by_body: dict[str, list[tuple[str, tuple[float, float, float]]]],
    parent_pos: tuple[float, float, float],
) -> etree._Element:
    """Recursively emit ``<body>`` plus all its joint-children.

    Joint-children are appended AFTER the geom element so the resulting
    MJCF satisfies MuJoCo's expected ordering (``joint``, ``geom``,
    then child ``<body>`` elements). The child's ``pos`` is rebased
    into this body's local frame; the grand-child recursion uses the
    **absolute** position of this body as its ``parent_pos``.
    """

    inner_joint = joint_by_child.get(body.name)
    spring_sites = tuple(sites_by_body.get(body.name, ()))
    body_el = _build_body(
        body,
        inner_joint=inner_joint,
        parent_pos=parent_pos,
        spring_sites=spring_sites,
    )

    for child_name in children_of.get(body.name, ()):
        child_body = by_name[child_name]
        child_el = _emit_body_subtree(
            child_body,
            by_name=by_name,
            joint_by_child=joint_by_child,
            children_of=children_of,
            sites_by_body=sites_by_body,
            # The child's `pos` field is absolute; we pass THIS body's
            # absolute pos as the new parent frame so the recursion
            # subtracts it.
            parent_pos=body.pos,
        )
        body_el.append(child_el)

    return body_el


__all__ = ["compile_to_mjcf"]
