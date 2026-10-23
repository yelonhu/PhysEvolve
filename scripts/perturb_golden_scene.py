#!/usr/bin/env python3
"""Generate perturbed golden scene variants from a base Stage-2 golden file.

Axes (comma-separated via ``--axes``):
  geometry  — numeric jitter on pos/size/mass/friction/restitution/velocity
  camera    — jitter ``add_camera`` pos/target
  rgba      — remap rgba tuples to nearby palette colours

Usage:
    python scripts/perturb_golden_scene.py \\
        --base data/golden_scenes/stage2/01_throwing_pit_board__v01.py \\
        --out data/golden_scenes/stage2/01_throwing_pit_board__v09.py \\
        --seed 9 --axes camera,rgba,geometry
"""

from __future__ import annotations

import argparse
import ast
import random
import re
from pathlib import Path

_FLOAT_RE = re.compile(
    r"(?P<prefix>\b(?:pos|linear|mass|friction|restitution|target|size|"
    r"radius|half_height)=)"
    r"(?P<body>\([^)]*\)|-?\d+\.\d+)"
)

_QUAT_RE = re.compile(
    r"quaternion=\((?P<a>-?\d+\.?\d*),\s*(?P<b>-?\d+\.?\d*),\s*"
    r"(?P<c>-?\d+\.?\d*),\s*(?P<d>-?\d+\.?\d*)\)"
)

_RGBA_RE = re.compile(
    r"rgba=\((?P<r>-?\d+\.?\d*),\s*(?P<g>-?\d+\.?\d*),\s*"
    r"(?P<b>-?\d+\.?\d*),\s*(?P<a>-?\d+\.?\d*)\)"
)

_CAMERA_CALL_RE = re.compile(
    r"api\.add_camera\([^)]*\)",
    re.DOTALL,
)

_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.90, 0.15, 0.15),
    (0.22, 0.78, 0.32),
    (0.12, 0.28, 0.92),
    (0.95, 0.55, 0.15),
    (0.94, 0.45, 0.56),
    (0.58, 0.22, 0.88),
    (0.95, 0.90, 0.20),
    (0.20, 0.75, 0.85),
)


def _split_doc_body(source: str) -> tuple[str, str]:
    lines = source.splitlines()
    if lines and lines[0].startswith('"""'):
        end = 0
        for idx, line in enumerate(lines[1:], start=1):
            if line.strip().endswith('"""'):
                end = idx + 1
                break
        return "\n".join(lines[:end]), "\n".join(lines[end:])
    return "", source


def _jitter_tuple_nums(text: str, rng: random.Random, *, scale: float) -> str:
    def _one(match: re.Match[str]) -> str:
        value = float(match.group(0))
        return str(round(value + rng.uniform(-scale, scale), 3))

    return re.sub(r"-?\d+\.\d+", _one, text)


def _perturb_geometry(body: str, rng: random.Random) -> str:
    def _replace(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        payload = match.group("body")
        if payload.startswith("("):
            nums = [float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", payload)]
            out: list[str] = []
            for idx, value in enumerate(nums):
                delta = rng.uniform(-0.08, 0.08)
                if abs(value) > 1.0:
                    delta = rng.uniform(-0.12, 0.12)
                new_val = round(value + delta, 3)
                key = prefix.lower()
                if "size" in key or "radius" in key or "half_height" in key:
                    new_val = max(0.01, abs(new_val))
                if "restitution" in key or "friction" in key:
                    new_val = min(1.0, max(0.0, new_val))
                if "mass" in key:
                    new_val = max(0.01, new_val)
                # PhysDSL workspace: non-plane pos.z must be in (0, 2.5);
                # clamp to a small positive floor so jitter never hits z=0.
                if "pos" in key and idx == 2:
                    new_val = min(2.49, max(0.01, new_val))
                out.append(str(new_val))
            return f"{prefix}({', '.join(out)})"
        value = float(payload)
        delta = rng.uniform(-0.08, 0.08)
        new_val = round(value + delta, 3)
        key = prefix.lower()
        if "restitution" in key or "friction" in key:
            new_val = min(1.0, max(0.0, new_val))
        if "mass" in key:
            new_val = max(0.01, new_val)
        return f"{prefix}{new_val}"

    body = _FLOAT_RE.sub(_replace, body)

    def _quat(match: re.Match[str]) -> str:
        vals = [float(match.group(k)) for k in ("a", "b", "c", "d")]
        vals = [round(v + rng.uniform(-0.02, 0.02), 4) for v in vals]
        return f"quaternion=({vals[0]}, {vals[1]}, {vals[2]}, {vals[3]})"

    return _QUAT_RE.sub(_quat, body)


def _perturb_camera(body: str, rng: random.Random) -> str:
    def _cam(match: re.Match[str]) -> str:
        text = match.group(0)
        return _jitter_tuple_nums(text, rng, scale=0.15)

    return _CAMERA_CALL_RE.sub(_cam, body)


def _perturb_rgba(body: str, rng: random.Random) -> str:
    def _rgba(match: re.Match[str]) -> str:
        alpha = float(match.group("a"))
        rgb = rng.choice(_PALETTE)
        # Tiny jitter so variants are not identical palette clones.
        rgb = tuple(max(0.0, min(1.0, c + rng.uniform(-0.04, 0.04))) for c in rgb)
        return (
            f"rgba=({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}, {alpha:.3f})"
        )

    return _RGBA_RE.sub(_rgba, body)


def _perturb_source(source: str, seed: int, axes: set[str]) -> str:
    """Apply deterministic perturbations along selected axes."""

    rng = random.Random(seed)
    head, body = _split_doc_body(source)
    if "geometry" in axes:
        body = _perturb_geometry(body, rng)
    if "camera" in axes:
        body = _perturb_camera(body, rng)
    if "rgba" in axes:
        body = _perturb_rgba(body, rng)
    if head:
        return head + "\n" + body
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--axes",
        type=str,
        default="camera,rgba,geometry",
        help="Comma-separated axes: camera,rgba,geometry",
    )
    args = parser.parse_args()

    axes = {a.strip().lower() for a in args.axes.split(",") if a.strip()}
    allowed = {"camera", "rgba", "geometry"}
    unknown = axes - allowed
    if unknown:
        raise SystemExit(f"unknown axes {sorted(unknown)}; allowed={sorted(allowed)}")

    source = args.base.read_text(encoding="utf-8")
    ast.parse(source)
    perturbed = _perturb_source(source, args.seed, axes)
    ast.parse(perturbed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(perturbed, encoding="utf-8")
    print(f"wrote {args.out} axes={sorted(axes)}")


if __name__ == "__main__":
    main()
