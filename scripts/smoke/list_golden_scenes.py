"""List Stage-2 golden scenes from ``data/golden_scenes/manifest.json``.

Usage::

    python scripts/smoke/list_golden_scenes.py
    python scripts/smoke/list_golden_scenes.py --render-cmd
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_SETUP = Path(__file__).resolve().parent / "_path_setup.py"
_spec = importlib.util.spec_from_file_location("_smoke_path_setup", _SETUP)
assert _spec is not None and _spec.loader is not None
_setup_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup_mod)
_setup_mod.bootstrap_smoke_sys_path()

from golden_scenes_common import MANIFEST_PATH  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List exported Stage-2 golden scenes."
    )
    parser.add_argument(
        "--render-cmd",
        action="store_true",
        help="Print a copy-paste render_golden_scene.py command per scene.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help=f"Manifest path (default: {MANIFEST_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    """Print the golden scene index."""

    args = _parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        print(
            f"error: manifest not found: {manifest_path}\n"
            "Run: python scripts/smoke/export_golden_fewshots.py",
            file=sys.stderr,
        )
        sys.exit(1)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = data.get("scenes") or []
    print(f"Stage-2 golden scenes ({len(scenes)}):\n")
    for rec in scenes:
        order = rec.get("order", "?")
        scene_kind = rec.get("scene_kind", "?")
        rel = rec.get("relative_path", "?")
        families = rec.get("families") or []
        upstream = rec.get("upstream_source", "?")
        print(f"{order:>2}. {scene_kind}")
        print(f"    file: {rel}")
        print(f"    upstream: {upstream}")
        print(f"    families ({len(families)}): {', '.join(families)}")
        if args.render_cmd:
            print(
                f"    cmd: python scripts/smoke/render_golden_scene.py "
                f"--scene {rel} --seed 7 --print-vqa --print-branches"
            )
        print()


if __name__ == "__main__":
    main()
