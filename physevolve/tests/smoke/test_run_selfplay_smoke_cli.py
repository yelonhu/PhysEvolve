"""Tests for the training self-play smoke CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI_PATH = _REPO_ROOT / "scripts" / "smoke" / "run_selfplay_2step_smoke.py"
_SPEC = importlib.util.spec_from_file_location("run_selfplay_2step_smoke_cli", _CLI_PATH)
assert _SPEC is not None and _SPEC.loader is not None
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)
_STAGE2_CLI_PATH = _REPO_ROOT / "scripts" / "smoke" / "run_stage2_2step_smoke.py"
_STAGE2_SPEC = importlib.util.spec_from_file_location(
    "run_stage2_2step_smoke_cli",
    _STAGE2_CLI_PATH,
)
assert _STAGE2_SPEC is not None and _STAGE2_SPEC.loader is not None
stage2_cli = importlib.util.module_from_spec(_STAGE2_SPEC)
_STAGE2_SPEC.loader.exec_module(stage2_cli)


def test_selfplay_cli_defaults_match_training_100step_v3(monkeypatch, tmp_path) -> None:
    captured = {}

    def _fake_prepare(config):
        captured["config"] = config
        return SimpleNamespace(
            run_dir=Path(tmp_path) / "training_cli",
            factory_config={"output_dir": str(Path(tmp_path) / "training_cli")},
        )

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(cli, "prepare_physevolve_trainer_smoke", _fake_prepare)
    monkeypatch.setattr(
        cli,
        "write_dry_run_summary",
        lambda config, run_dir: {"completed": False, "executed": False},
    )

    exit_code = cli.main(["--dry-run", "--out-root", str(tmp_path)])

    assert exit_code == 0
    assert captured["config"].execute is False
    assert captured["config"].max_steps == 2
    assert captured["config"].scenes_per_batch == cli.V3_SCENES_PER_BATCH
    assert captured["config"].group_size == cli.V3_GROUP_SIZE
    assert captured["config"].n_gpus_per_node == cli.V3_N_GPUS_PER_NODE
    assert captured["config"].ray_num_cpus == cli.V3_RAY_NUM_CPUS
    assert captured["config"].checkpoint_freq == cli.V3_CHECKPOINT_FREQ
    assert captured["config"].experiment_name == cli.V3_EXPERIMENT_NAME
    assert "CUDA_VISIBLE_DEVICES" not in cli.os.environ


def test_selfplay_cli_forwards_native_knobs(monkeypatch, tmp_path) -> None:
    captured = {}

    def _fake_prepare(config):
        captured["config"] = config
        return SimpleNamespace(
            run_dir=Path(tmp_path) / "training_cli",
            factory_config={"output_dir": str(Path(tmp_path) / "training_cli")},
        )

    monkeypatch.setattr(cli, "prepare_physevolve_trainer_smoke", _fake_prepare)
    monkeypatch.setattr(
        cli,
        "write_dry_run_summary",
        lambda config, run_dir: {"completed": False, "executed": False},
    )

    exit_code = cli.main(
        [
            "--dry-run",
            "--out-root",
            str(tmp_path),
            "--group-size",
            "4",
            "--checkpoint-freq",
            "20",
        ]
    )

    assert exit_code == 0
    assert captured["config"].group_size == 4
    assert captured["config"].checkpoint_freq == 20


def test_stage2_cli_forwards_branch_family_weight_multiplier(
    monkeypatch,
    tmp_path,
) -> None:
    captured = {}

    def _fake_prepare(config):
        captured["config"] = config
        return SimpleNamespace(
            run_dir=Path(tmp_path) / "stage2_cli",
            factory_config={"output_dir": str(Path(tmp_path) / "stage2_cli")},
        )

    monkeypatch.setattr(stage2_cli, "prepare_physevolve_trainer_smoke", _fake_prepare)
    monkeypatch.setattr(
        stage2_cli,
        "write_dry_run_summary",
        lambda config, run_dir: {"completed": False, "executed": False},
    )

    exit_code = stage2_cli.main(
        [
            "--dry-run",
            "--out-root",
            str(tmp_path),
            "--stage2-branch-family-weight-multiplier",
            "1.2",
        ]
    )

    assert exit_code == 0
    assert captured["config"].stage2_branch_family_weight_multiplier == 1.2


def test_stage2_cli_forwards_opt_in_stage2_ablations(monkeypatch, tmp_path) -> None:
    captured = []

    def _fake_prepare(config):
        captured.append(config)
        return SimpleNamespace(
            run_dir=Path(tmp_path) / "stage2_cli",
            factory_config={"output_dir": str(Path(tmp_path) / "stage2_cli")},
        )

    monkeypatch.setattr(stage2_cli, "prepare_physevolve_trainer_smoke", _fake_prepare)
    monkeypatch.setattr(
        stage2_cli,
        "write_dry_run_summary",
        lambda config, run_dir: {"completed": False, "executed": False},
    )

    default_exit_code = stage2_cli.main(
        ["--dry-run", "--out-root", str(tmp_path)]
    )
    branch_exit_code = stage2_cli.main(
        [
            "--dry-run",
            "--out-root",
            str(tmp_path),
            "--disable-branch-elision",
        ]
    )
    proposer_exit_code = stage2_cli.main(
        [
            "--dry-run",
            "--out-root",
            str(tmp_path),
            "--freeze-proposer",
        ]
    )

    assert default_exit_code == 0
    assert branch_exit_code == 0
    assert proposer_exit_code == 0
    assert captured[0].disable_branch_elision is False
    assert captured[0].freeze_proposer is False
    assert captured[1].disable_branch_elision is True
    assert captured[1].freeze_proposer is False
    assert captured[2].disable_branch_elision is False
    assert captured[2].freeze_proposer is True


def test_selfplay_cli_forwards_wandb_flags(monkeypatch, tmp_path) -> None:
    captured = {}

    def _fake_prepare(config):
        captured["config"] = config
        return SimpleNamespace(
            run_dir=Path(tmp_path) / "training_cli",
            factory_config={"output_dir": str(Path(tmp_path) / "training_cli")},
        )

    monkeypatch.setattr(cli, "prepare_physevolve_trainer_smoke", _fake_prepare)
    monkeypatch.setattr(
        cli,
        "write_dry_run_summary",
        lambda config, run_dir: {"completed": False, "executed": False},
    )

    exit_code = cli.main(
        [
            "--dry-run",
            "--out-root",
            str(tmp_path),
            "--wandb",
            "--wandb-project",
            "nz-project",
            "--wandb-run-name",
            "training-20",
        ]
    )

    assert exit_code == 0
    assert captured["config"].wandb_enabled is True
    assert captured["config"].wandb_project == "nz-project"
    assert captured["config"].wandb_run_name == "training-20"


def test_selfplay_cli_requires_trainer_factory_for_execution(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="trainer-factory"):
        cli.main(["--out-root", str(tmp_path)])


def test_selfplay_cli_factory_preflight(monkeypatch, tmp_path) -> None:
    captured = {}

    def _fake_preflight(config):
        captured["config"] = config
        return {"ok": True, "loads_model": False, "starts_ray": False}

    monkeypatch.setattr(cli, "preflight_physevolve_verl_factory", _fake_preflight)

    exit_code = cli.main(
        [
            "--factory-preflight",
            "--out-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert captured["config"].execute is True


def test_selfplay_cli_real_execution_calls_factory_with_config_keyword(
    monkeypatch,
    tmp_path,
) -> None:
    captured = {}
    summary_path = Path(tmp_path) / "training_cli" / "summary.json"
    trainer = SimpleNamespace(
        name="trainer",
        fit=lambda: summary_path.write_text('{"completed": true}\n', encoding="utf-8"),
    )

    def _fake_load_trainer_factory(spec):
        captured["factory_spec"] = spec

        def _fake_factory(**kwargs):
            captured["factory_kwargs"] = kwargs
            return trainer

        return _fake_factory

    def _fake_prepare(config):
        captured["run_config"] = config
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            run_dir=summary_path.parent,
            factory_config={"output_dir": summary_path.parent.as_posix()},
        )

    monkeypatch.setattr(cli, "_load_trainer_factory", _fake_load_trainer_factory)
    monkeypatch.setattr(cli, "prepare_physevolve_trainer_smoke", _fake_prepare)

    exit_code = cli.main(
        [
            "--out-root",
            str(tmp_path),
            "--trainer-factory",
            "physevolve.training.verl_factory:create_initialized_physevolve_verl_trainer",
        ]
    )

    assert exit_code == 0
    assert captured["factory_spec"] == (
        "physevolve.training.verl_factory:create_initialized_physevolve_verl_trainer"
    )
    assert set(captured["factory_kwargs"]) == {"config"}
    assert captured["factory_kwargs"]["config"]["output_dir"] == summary_path.parent.as_posix()
    assert captured["run_config"].execute is True


def test_selfplay_cli_validates_20_step_run_after_execution(monkeypatch, tmp_path) -> None:
    captured = {}
    run_dir = Path(tmp_path) / "training_cli"
    summary_path = run_dir / "summary.json"

    def _fit() -> None:
        from physevolve.smoke.io import append_jsonl, write_json

        write_json(
            summary_path,
            {
                "completed": True,
                "executed": True,
                "ppo_update_count": 20,
                "reward_tensor_finite": True,
            },
        )
        write_json(run_dir / "run_status.json", {"completed": True, "error": None})
        (run_dir / "global_step_20" / "actor").mkdir(parents=True)
        for step in range(1, 21):
            append_jsonl(
                run_dir / "metrics.jsonl",
                {
                    "step": step,
                    "time/step_total_s": 1.0,
                    "reward_batch": {
                        "reward_tensor_finite": True,
                        "reward_mean": float(step),
                    },
                    "online": {
                        "all_rewards_finite": True,
                        "n_executor_ok": 1,
                        "selected_solver_groups": 1,
                        "answer_parse_rate": 1.0,
                        "schema_ok_rate": 1.0,
                        "solver_backfill_shortfall": 0,
                        "planned_solver_groups": 1,
                        "unique_scene_ratio": 1.0,
                    },
                    "update": {
                        "actor/loss": 0.0,
                        "actor/pg_loss": 0.0,
                        "actor/kl_loss": 0.0,
                        "actor/grad_norm": 1.0,
                    },
                },
            )

    trainer = SimpleNamespace(name="trainer", fit=_fit)

    def _fake_load_trainer_factory(spec):
        captured["factory_spec"] = spec

        def _fake_factory(**kwargs):
            captured["factory_kwargs"] = kwargs
            return trainer

        return _fake_factory

    def _fake_prepare(config):
        captured["run_config"] = config
        run_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            run_dir=run_dir,
            factory_config={"output_dir": run_dir.as_posix()},
        )

    monkeypatch.setattr(cli, "_load_trainer_factory", _fake_load_trainer_factory)
    monkeypatch.setattr(cli, "prepare_physevolve_trainer_smoke", _fake_prepare)

    exit_code = cli.main(
        [
            "--max-steps",
            "20",
            "--out-root",
            str(tmp_path),
            "--trainer-factory",
            "physevolve.training.verl_factory:create_initialized_physevolve_verl_trainer",
        ]
    )

    assert exit_code == 0
    assert (run_dir / "training_validation.json").exists()
