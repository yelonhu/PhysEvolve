"""Export Stage-2 scene-template golden fewshots to ``data/golden_scenes/stage2/``.

Reads each template's ``fewshot_example`` from
:func:`physevolve.skills.scene_templates.get_scene_template_registry`,
extracts the ``<propose>`` block, and writes standalone runnable ``.py`` files
plus ``data/golden_scenes/manifest.json``.

Usage::

    python scripts/smoke/export_golden_fewshots.py
    python scripts/smoke/export_golden_fewshots.py --force
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
_REPO_ROOT, _ = _setup_mod.bootstrap_smoke_sys_path()

from physevolve.inference.parsing import extract_propose_block  # noqa: E402
from physevolve.skills.scene_templates import (  # noqa: E402
    get_scene_template_registry,
)
from golden_scenes_common import (  # noqa: E402
    GOLDEN_SCENES_ROOT,
    MANIFEST_PATH,
    ORDERED_SCENE_KINDS,
    STAGE2_DIR,
    build_golden_scene_source,
    manifest_entry,
    stage2_filename,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Stage-2 golden fewshots to data/golden_scenes/stage2/."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing golden scene files (default: skip existing).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=STAGE2_DIR,
        help=f"Output directory (default: {STAGE2_DIR})",
    )
    return parser.parse_args()


def export_golden_fewshots(*, force: bool, out_dir: Path) -> list[dict]:
    """Export all ordered scene kinds; return manifest scene records."""

    registry = get_scene_template_registry()
    out_dir.mkdir(parents=True, exist_ok=True)
    GOLDEN_SCENES_ROOT.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for order, scene_kind in enumerate(ORDERED_SCENE_KINDS, start=1):
        template = registry.lookup(scene_kind)
        fewshot = (template.fewshot_example or "").strip()
        if not fewshot:
            raise ValueError(f"template {scene_kind!r} has empty fewshot_example")

        propose_code = extract_propose_block(fewshot)
        if propose_code is None:
            raise ValueError(f"template {scene_kind!r} fewshot has no <propose> block")
        if "SCENE_MANIFEST" not in propose_code:
            raise ValueError(
                f"template {scene_kind!r} propose block missing SCENE_MANIFEST"
            )

        filename = stage2_filename(order, scene_kind)
        out_path = out_dir / filename
        rel_path = str(out_path.relative_to(_REPO_ROOT))

        if out_path.exists() and not force:
            print(f"skip  {rel_path} (exists; use --force to overwrite)")
        else:
            body = build_golden_scene_source(
                scene_kind=scene_kind,
                families=template.families,
                fewshot=fewshot,
                propose_code=propose_code,
            )
            out_path.write_text(body, encoding="utf-8")
            print(f"write {rel_path}")

        records.append(
            manifest_entry(
                order=order,
                scene_kind=scene_kind,
                families=template.families,
                relative_path=rel_path,
            )
        )

    manifest = {
        "version": 1,
        "description": "Stage-2 scene-template golden fewshots (export of fewshot_example).",
        "stage2_dir": str(out_dir.relative_to(_REPO_ROOT)),
        "scenes": records,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"write {MANIFEST_PATH.relative_to(_REPO_ROOT)}")
    return records


def main() -> None:
    """CLI entry point."""

    args = _parse_args()
    records = export_golden_fewshots(force=bool(args.force), out_dir=args.out_dir.resolve())
    print(f"\nExported {len(records)} golden scene(s).")
    for rec in records:
        print(
            f"  {rec['order']:02d} {rec['scene_kind']}: "
            f"{len(rec['families'])} families -> {rec['relative_path']}"
        )


if __name__ == "__main__":
    main()
