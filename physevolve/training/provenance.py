"""Record the code, dependencies, and construction seeds behind a run."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from pathlib import Path

from physevolve.smoke.io import write_json


def write_run_provenance(run_dir: Path, *, model: str, stage: int) -> None:
    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
                text=True, timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    versions = {}
    for name in ("torch", "transformers", "vllm", "verl", "mujoco", "numpy", "ray"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None

    from physevolve.training.online_step_builder import PhysEvolveOnlineStepBuilderConfig
    from physevolve.training.online_step_builder_stage2 import PhysEvolveStage2StepBuilderConfig

    defaults = PhysEvolveStage2StepBuilderConfig() if stage == 2 else PhysEvolveOnlineStepBuilderConfig()
    write_json(run_dir / "provenance.json", {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": versions,
        "model": model,
        "stage": stage,
        "construction_seeds": {
            name: getattr(defaults, name)
            for name in ("seed_start", "rng_seed") if hasattr(defaults, name)
        },
        "verl_base_commit": (root / "third_party/verl/base_commit.txt").read_text().strip(),
    })
