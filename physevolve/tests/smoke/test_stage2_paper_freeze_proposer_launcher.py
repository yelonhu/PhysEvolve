"""CPU-only recipe lock for the Stage-2 paper freeze-proposer launcher."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAUNCHER = _REPO_ROOT / "scripts" / "train" / "ablations" / "freeze_proposer.sh"


def test_paper_freeze_proposer_launcher_dry_run_locks_recipe(tmp_path: Path) -> None:
    """Assert that the thin launcher adds only the frozen-Proposer ablation."""

    run_id = "paper_freeze_proposer_recipe_lock"
    env = dict(os.environ)
    env["PYTHON_BIN"] = sys.executable
    subprocess.run(
        [
            "bash",
            str(_LAUNCHER),
            "--dry-run",
            "--out-root",
            str(tmp_path),
            "--run-id",
            run_id,
        ],
        cwd=_REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    run_dir = tmp_path / run_id
    config = json.loads((run_dir / "verl_config.json").read_text(encoding="utf-8"))
    physevolve = config["physevolve"]
    rollout = config["actor_rollout_ref"]["rollout"]
    actor = config["actor_rollout_ref"]["actor"]

    assert config["trainer"]["total_training_steps"] == 100
    assert config["trainer"]["save_freq"] == 100
    assert actor["kl_loss_coef"] == pytest.approx(0.03)
    assert "freeze_vision_tower" not in actor
    assert physevolve["freeze_proposer"] is True
    assert "disable_branch_elision" not in physevolve
    assert config["algorithm"]["kl_ctrl"]["kl_coef"] == pytest.approx(0.03)
    assert rollout["temperature"] == pytest.approx(0.7)
    assert rollout["top_p"] == pytest.approx(0.95)
    assert rollout["top_k"] == 20
    assert physevolve["generation"]["temperature"] == pytest.approx(0.7)
    assert physevolve["generation"]["top_p"] == pytest.approx(0.95)
    assert physevolve["generation"]["top_k"] == 20
    assert physevolve["stage2_branch_family_weight_multiplier"] == pytest.approx(1.2)
    assert physevolve["stage2_cold_start_fewshot_steps"] == 100
    assert physevolve["stage2_min_scene_kind_picks_per_step"] == {
        "throwing_pit_board": 1,
        "collision_arena": 1,
        "motion_lab": 1,
    }
    assert physevolve["stage2_max_solver_questions_per_scene"] == 5
    assert physevolve["stage2_max_solver_groups_per_step"] == 32
    assert physevolve["stage2_camera_depth_bypass_enabled"] is True
    assert physevolve["stage2_reference_buffer_enabled"] is True
    assert physevolve["stage2_reference_buffer_during_fewshot"] is True
    assert physevolve["stage2_copy_penalty_enabled"] is True
    assert physevolve["stage2_batch_dedup_enabled"] is True
    assert physevolve["stage2_anti_copy_start_step"] == 0
