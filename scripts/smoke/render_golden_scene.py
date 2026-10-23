"""Render a Stage-2 golden scene file and optionally print derived VQAs.

Runs :func:`run_executor` with ``stage=2`` on a standalone golden ``.py``
(``build_scene`` + ``SCENE_MANIFEST``), writes a structured artifact tree
(per-family VQA JSON, branches, MP4), and can print diagnostics to stdout.

Usage::

    python scripts/smoke/render_golden_scene.py \\
      --scene data/golden_scenes/stage2/01_throwing_pit_board.py \\
      --seed 7 --print-vqa --print-branches
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

_SETUP = Path(__file__).resolve().parent / "_path_setup.py"
_spec = importlib.util.spec_from_file_location("_smoke_path_setup", _SETUP)
assert _spec is not None and _spec.loader is not None
_setup_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup_mod)
_REPO_ROOT, _ = _setup_mod.bootstrap_smoke_sys_path()

from physevolve.engine.failure import FailureReason  # noqa: E402
from physevolve.executor import run_executor  # noqa: E402
from golden_scenes_common import (  # noqa: E402
    make_run_dir,
    vqa_item_to_record,
    write_run_artifacts,
)

_DEFAULT_OUT_DIR = Path("outputs")
_LETTERS = "ABCD"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Stage-2 golden scene (SCENE_MANIFEST) to artifacts."
    )
    parser.add_argument(
        "--scene",
        type=Path,
        required=True,
        help="Path to golden scene Python source.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help=f"Output root directory (default: {_DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Simulation seed (default: 7, matches scene template tests).",
    )
    parser.add_argument(
        "--camera-idx",
        type=int,
        default=0,
        help="Camera index when video_frames is 5-D (default: 0).",
    )
    parser.add_argument(
        "--sim-timeout-s",
        type=float,
        default=30.0,
        help="Physics simulation wall-clock budget in seconds.",
    )
    parser.add_argument(
        "--print-vqa",
        action="store_true",
        help="Print each auto-derived VQA (question, options, correct letter).",
    )
    parser.add_argument(
        "--print-branches",
        action="store_true",
        help="Print branch-simulation diagnostics from the executor trace.",
    )
    parser.add_argument(
        "--color-variants",
        type=int,
        default=0,
        help="Pass color_variants_per_scene to run_executor (default: 0).",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip baseline.mp4 and per-family video.mp4 (JSON artifacts still written).",
    )
    parser.add_argument(
        "--no-save-artifacts",
        action="store_true",
        help="Do not write the run directory (stdout --print-* only).",
    )
    return parser.parse_args()


def _print_vqa_payloads(result: object) -> None:
    """Print vqa_payloads from an OK executor result."""

    payloads = getattr(result, "vqa_payloads", None) or []
    if not payloads:
        print("vqa: (none)")
        return
    print(f"vqa: {len(payloads)} question(s)")
    for i, item in enumerate(payloads):
        record = vqa_item_to_record(item, i)
        family = record["family"] or "?"
        phrasing = record["phrasing"] or "?"
        letter = record["correct_letter"]
        print(f"\n--- [{i}] family={family} phrasing={phrasing} correct={letter} ---")
        print(record["question"])
        for j, opt in enumerate(record["options"]):
            mark = " *" if j == int(record["correct_idx"]) else ""
            print(f"  {_LETTERS[j]}. {opt}{mark}")


def _print_branch_trace(trace: dict) -> None:
    """Pretty-print branch diagnostics from the executor trace dict."""

    print("\n=== branch trace ===")
    print(f"scene_kind: {trace.get('scene_kind')}")
    baseline = trace.get("baseline_outcomes") or {}
    if baseline:
        print("baseline outcomes:")
        for ball, outcome in baseline.items():
            print(f"  {ball}: {json.dumps(outcome, ensure_ascii=False)}")
    branches = trace.get("branches") or []
    print(f"branches: {len(branches)} spec(s)")
    for rec in branches:
        key = rec.get("key")
        removed = rec.get("removed")
        ran = rec.get("ran")
        print(f"\n  key={key!r} removed={removed} ran={ran}")
        outcomes = rec.get("outcomes")
        if outcomes:
            for ball, outcome in outcomes.items():
                print(f"    {ball}: {json.dumps(outcome, ensure_ascii=False)}")


def main() -> None:
    """Run simulation, write artifacts, optionally print VQA/branch diagnostics."""

    args = _parse_args()
    scene_path = args.scene.resolve()
    if not scene_path.is_file():
        print(f"error: scene file not found: {scene_path}", file=sys.stderr)
        sys.exit(1)

    save_artifacts = not bool(args.no_save_artifacts)
    trace: dict = {}
    code = scene_path.read_text(encoding="utf-8")
    result = run_executor(
        code,
        stage=2,
        seed=int(args.seed),
        sim_timeout_s=float(args.sim_timeout_s),
        color_variants_per_scene=int(args.color_variants),
        trace=trace if (save_artifacts or args.print_branches) else None,
    )

    if result.status not in (FailureReason.OK, FailureReason.TRIVIAL):
        print(f"status={result.status.value}", file=sys.stderr)
        if result.error_message:
            print(f"error: {result.error_message}", file=sys.stderr)
        sys.exit(1)

    print(f"status={result.status.value}")
    print(f"scene_id={result.scene_id}")
    n_vqa = len(result.vqa_payloads or [])
    print(f"vqa_count={n_vqa}")
    print(f"walltime_s={result.walltime_s:.2f}")

    if args.print_vqa:
        _print_vqa_payloads(result)
    if args.print_branches and trace:
        _print_branch_trace(trace)
    elif args.print_branches:
        print("\n(branch trace empty — executor did not fill trace)")

    if not save_artifacts:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = make_run_dir(
        args.out_dir.resolve(),
        scene_path.stem,
        int(args.seed),
        stamp=stamp,
    )
    write_video = not bool(args.no_video)
    if write_video and (
        result.sim_state is None or result.sim_state.video_frames is None
    ):
        print(
            "error: simulation succeeded but video_frames is missing "
            "(use --no-video to save JSON only)",
            file=sys.stderr,
        )
        sys.exit(1)

    write_run_artifacts(
        run_dir=run_dir,
        result=result,
        trace=trace,
        scene_path=scene_path,
        seed=int(args.seed),
        status=result.status.value,
        write_video=write_video,
        camera_idx=int(args.camera_idx),
    )

    print(f"artifact_dir={run_dir}")
    if write_video:
        print(f"baseline_video={run_dir / 'baseline.mp4'}")
    print(f"summary={run_dir / 'summary.json'}")
    print(f"branches={run_dir / 'branches.json'}")


if __name__ == "__main__":
    main()
