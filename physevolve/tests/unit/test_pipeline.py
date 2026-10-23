"""Unit tests for :mod:`physevolve.executor.pipeline`.

Every :class:`FailureReason` value from docs/architecture.md must be reachable
through ``run_executor`` — these tests exercise one path per value.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

import physevolve.engine  # noqa: F401 — ensures EGL env is set up.
from physevolve.engine.failure import PROPOSER_PENALTY, FailureReason
from physevolve.executor import run_executor
from physevolve.executor.pipeline import _validate_vqa_payload
from physevolve.executor.result import ExecutorResult


# ─── Shared tiny-sim knobs: keep every test below ~1-2 seconds. ──────
_TINY_SIM_S = 8.0


def _ok_sphere_src() -> str:
    return (
        "def build_scene(api):\n"
        "    api.add_sphere('ball', pos=(0.0, 0.0, 1.5), radius=0.1, mass=1.0)\n"
        "    api.set_duration(0.5)\n"
    )


# ─── OK path ─────────────────────────────────────────────────────────


class TestOkPath:
    def test_ok_status(self) -> None:
        r = run_executor(_ok_sphere_src(), stage=1, sim_timeout_s=_TINY_SIM_S)
        assert r.status == FailureReason.OK
        assert r.ok is True
        assert r.sim_state is not None
        assert r.vqa_payload is None
        assert r.proposer_penalty == PROPOSER_PENALTY[FailureReason.OK]
        assert r.scene_id != ""
        assert r.walltime_s > 0

    def test_scene_id_deterministic(self) -> None:
        r1 = run_executor(_ok_sphere_src(), stage=1, sim_timeout_s=_TINY_SIM_S)
        r2 = run_executor(_ok_sphere_src(), stage=1, sim_timeout_s=_TINY_SIM_S)
        assert r1.scene_id == r2.scene_id


# ─── Per-FailureReason coverage ──────────────────────────────────────


def _expect_status(res: ExecutorResult, status: FailureReason) -> None:
    assert res.status == status, f"expected {status}, got {res.status}: {res.error_message}"
    assert res.ok is (status == FailureReason.OK)
    assert res.proposer_penalty == PROPOSER_PENALTY[status]


class TestFailureTaxonomy:
    def test_failed_syntax(self) -> None:
        src = "def build_scene(api:\n"
        r = run_executor(src, stage=1, sim_timeout_s=_TINY_SIM_S)
        _expect_status(r, FailureReason.FAILED_SYNTAX)
        assert r.sim_state is None
        assert r.scene_id == ""

    def test_failed_semantic_via_ast(self) -> None:
        src = "import os\n"
        r = run_executor(src, stage=1, sim_timeout_s=_TINY_SIM_S)
        _expect_status(r, FailureReason.FAILED_SEMANTIC)

    def test_failed_semantic_via_dsl_validation(self) -> None:
        src = (
            "def build_scene(api):\n"
            "    api.add_box('x', pos=(99.0, 0.0, 1.0), size=(0.1, 0.1, 0.1), mass=1.0)\n"
        )
        r = run_executor(src, stage=1, sim_timeout_s=_TINY_SIM_S)
        _expect_status(r, FailureReason.FAILED_SEMANTIC)

    def test_failed_runtime(self) -> None:
        src = (
            "def build_scene(api):\n"
            "    raise RuntimeError('boom')\n"
        )
        r = run_executor(src, stage=1, sim_timeout_s=_TINY_SIM_S)
        _expect_status(r, FailureReason.FAILED_RUNTIME)

    def test_failed_overlap(self) -> None:
        src = (
            "def build_scene(api):\n"
            "    api.add_sphere('a', pos=(0.0, 0.0, 0.5), radius=0.3, mass=1.0)\n"
            "    api.add_sphere('b', pos=(0.0, 0.0, 0.5), radius=0.3, mass=1.0)\n"
        )
        r = run_executor(src, stage=1, sim_timeout_s=_TINY_SIM_S)
        _expect_status(r, FailureReason.FAILED_OVERLAP)

    def test_failed_nan_injection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inject NaN into qpos after a few steps; expect FAILED_NAN."""

        import mujoco

        from physevolve.engine import simulator as sim_module

        real_mj_step = mujoco.mj_step
        call_count = {"n": 0}

        def fake_mj_step(model, data, *args, **kwargs):  # type: ignore[no-untyped-def]
            real_mj_step(model, data, *args, **kwargs)
            call_count["n"] += 1
            if call_count["n"] == 1:
                data.qpos[0] = np.nan

        monkeypatch.setattr(sim_module.mujoco, "mj_step", fake_mj_step)
        r = run_executor(_ok_sphere_src(), stage=1, sim_timeout_s=_TINY_SIM_S)
        _expect_status(r, FailureReason.FAILED_NAN)

    def test_failed_timeout_sim(self) -> None:
        """Simulator hits its wall-clock budget on a long-duration scene."""

        src = (
            "def build_scene(api):\n"
            "    api.add_sphere('ball', pos=(0.0, 0.0, 1.0), radius=0.1, mass=1.0)\n"
            "    api.set_duration(8.0)\n"
        )
        r = run_executor(src, stage=1, sim_timeout_s=0.001)
        _expect_status(r, FailureReason.FAILED_TIMEOUT)

    def test_failed_timeout_sandbox(self) -> None:
        """Infinite loop in Proposer code is killed by sandbox SIGALRM."""

        src = "while True:\n    pass\n"
        r = run_executor(src, stage=1, sandbox_timeout_s=0.2, sim_timeout_s=_TINY_SIM_S)
        _expect_status(r, FailureReason.FAILED_TIMEOUT)

    def test_failed_render(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force ``render_multiview`` to raise; expect FAILED_RENDER."""

        from physevolve.engine import simulator as sim_module

        def _broken_render(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated render failure")

        monkeypatch.setattr(sim_module, "render_multiview", _broken_render)
        r = run_executor(_ok_sphere_src(), stage=1, sim_timeout_s=_TINY_SIM_S)
        _expect_status(r, FailureReason.FAILED_RENDER)

    def test_failed_compile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force ``MjModel.from_xml_string`` to raise ValueError."""

        import mujoco

        from physevolve.engine import simulator as sim_module

        def _broken_from_xml(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise ValueError("simulated compile failure")

        monkeypatch.setattr(
            sim_module.mujoco.MjModel, "from_xml_string", _broken_from_xml
        )
        r = run_executor(_ok_sphere_src(), stage=1, sim_timeout_s=_TINY_SIM_S)
        _expect_status(r, FailureReason.FAILED_COMPILE)
        _ = mujoco  # keep flake8 quiet; we need the import for monkeypatch side-effect.

    def test_failed_compile_simulate_runtime_error(self) -> None:
        """A joint child under a plane parent should fail softly, not crash."""

        src = (
            "def build_scene(api):\n"
            "    api.add_plane('floor', pos=(0.0, 0.0, 0.0), size=(3.0, 3.0),\n"
            "                  rgba=(0.8, 0.8, 0.8, 1.0), friction=0.5)\n"
            "    api.add_capsule('brown_hammer', pos=(0.0, 0.5, 1.0),\n"
            "                    radius=0.1, half_height=0.3, mass=0.5,\n"
            "                    rgba=(0.6, 0.3, 0.1, 1.0), friction=1.0,\n"
            "                    restitution=0.0, fixed=False)\n"
            "    api.add_hinge('hinge_hammer', 'floor', 'brown_hammer',\n"
            "                  pos=(0.0, 0.5, 1.0), axis=(0.0, 1.0, 0.0))\n"
            "    api.set_duration(0.5)\n"
        )
        r = run_executor(src, stage=2, sim_timeout_s=_TINY_SIM_S)
        _expect_status(r, FailureReason.FAILED_COMPILE)
        assert r.sim_state is None

    def test_trivial(self) -> None:
        """Single fixed sphere with no dynamics triggers TRIVIAL."""

        src = (
            "def build_scene(api):\n"
            "    api.add_sphere('ball', pos=(0.0, 0.0, 1.0), radius=0.1,\n"
            "                   mass=1.0, fixed=True)\n"
            "    api.set_duration(0.5)\n"
        )
        r = run_executor(src, stage=1, sim_timeout_s=_TINY_SIM_S)
        _expect_status(r, FailureReason.TRIVIAL)
        # TRIVIAL preserves sim_state for diagnostics.
        assert r.sim_state is not None


# ─── VQA validation ─────────────────────────────────────────────────


# Manifest-path failure sources. Each has a valid build_scene (so the sim runs)
# but a SCENE_MANIFEST the executor rejects -> FAILED_VQA. Only scene manifests are accepted.
_VQA_SOURCES: dict[str, str] = {
    "unknown_scene_kind": (
        "def build_scene(api):\n"
        "    api.add_box('blue_box', pos=(-0.7, 0.0, 0.10), size=(0.10, 0.10, 0.10),\n"
        "                mass=1.0, rgba=(0.2, 0.4, 0.9, 1.0), friction=2.0,\n"
        "                restitution=0.0, fixed=False)\n"
        "    api.set_velocity('blue_box', linear=(1.8, 0.0, 0.0))\n"
        "    api.set_duration(0.5)\n"
        "\n"
        "SCENE_MANIFEST = {\n"
        "    'scene_kind': 'not_a_real_scene_kind',\n"
        "    'motion_target': 'blue_box',\n"
        "    'families': ['motion_state_change'],\n"
        "}\n"
    ),
    "unknown_family": (
        "def build_scene(api):\n"
        "    api.add_box('blue_box', pos=(-0.7, 0.0, 0.10), size=(0.10, 0.10, 0.10),\n"
        "                mass=1.0, rgba=(0.2, 0.4, 0.9, 1.0), friction=2.0,\n"
        "                restitution=0.0, fixed=False)\n"
        "    api.set_velocity('blue_box', linear=(1.8, 0.0, 0.0))\n"
        "    api.set_duration(0.5)\n"
        "\n"
        "SCENE_MANIFEST = {\n"
        "    'scene_kind': 'motion_lab',\n"
        "    'motion_target': 'blue_box',\n"
        "    'families': ['collision_first_contact'],\n"
        "}\n"
    ),
    "missing_body_ref": (
        "def build_scene(api):\n"
        "    api.add_box('blue_box', pos=(-0.7, 0.0, 0.10), size=(0.10, 0.10, 0.10),\n"
        "                mass=1.0, rgba=(0.2, 0.4, 0.9, 1.0), friction=2.0,\n"
        "                restitution=0.0, fixed=False)\n"
        "    api.set_velocity('blue_box', linear=(1.8, 0.0, 0.0))\n"
        "    api.set_duration(0.5)\n"
        "\n"
        "SCENE_MANIFEST = {\n"
        "    'scene_kind': 'motion_lab',\n"
        "    'motion_target': 'does_not_exist',\n"
        "    'families': ['motion_state_change'],\n"
        "}\n"
    ),
    "non_dict_manifest": (
        "def build_scene(api):\n"
        "    api.add_box('blue_box', pos=(-0.7, 0.0, 0.10), size=(0.10, 0.10, 0.10),\n"
        "                mass=1.0, rgba=(0.2, 0.4, 0.9, 1.0), friction=2.0,\n"
        "                restitution=0.0, fixed=False)\n"
        "    api.set_velocity('blue_box', linear=(1.8, 0.0, 0.0))\n"
        "    api.set_duration(0.5)\n"
        "\n"
        "SCENE_MANIFEST = 5\n"
    ),
}


@pytest.mark.parametrize("key,src", list(_VQA_SOURCES.items()))
def test_failed_vqa(key: str, src: str) -> None:
    r = run_executor(src, stage=2, sim_timeout_s=_TINY_SIM_S)
    _expect_status(r, FailureReason.FAILED_VQA)


def test_stage2_ok_path() -> None:
    """Unified Stage-2 source emits build_scene + SCENE_MANIFEST literal.

    The ``motion_lab`` scene_kind is robust: a box launched on a high-friction
    surface starts with clear non-zero speed (feasibility) and the executor
    reads its actual motion-change bucket as the GT.
    """

    src = (
        "def build_scene(api):\n"
        "    api.add_box('blue_box', pos=(-0.7, 0.0, 0.10), size=(0.10, 0.10, 0.10),\n"
        "                mass=1.0, rgba=(0.2, 0.4, 0.9, 1.0), friction=2.0,\n"
        "                restitution=0.0, fixed=False)\n"
        "    api.set_velocity('blue_box', linear=(1.8, 0.0, 0.0))\n"
        "    api.set_duration(0.5)\n"
        "\n"
        "SCENE_MANIFEST = {\n"
        "    'scene_kind': 'motion_lab',\n"
        "    'motion_target': 'blue_box',\n"
        "    'families': ['motion_state_change'],\n"
        "}\n"
    )
    r = run_executor(src, stage=2, sim_timeout_s=_TINY_SIM_S)
    assert r.status == FailureReason.OK
    assert r.vqa_payload is not None
    assert r.vqa_payloads is not None and len(r.vqa_payloads) >= 1
    families = {
        str((item.get("spec") or {}).get("family", ""))
        for item in r.vqa_payloads
    }
    assert families == {"motion_state_change"}
    question, options, idx = r.vqa_payload
    assert isinstance(question, str) and question
    assert len(options) == 4
    assert 0 <= idx < 4


def test_stage2_repair_tolerates_plane_kwarg_and_spawn_edges() -> None:
    src = (
        "def build_scene(api):\n"
        "    api.add_plane('ground', pos=(0.0, 0.0, 0.0), size=(3.0, 3.0),\n"
        "                  rgba=(0.8, 0.8, 0.8, 1.0), friction=0.5,\n"
        "                  restitution=0.0)\n"
        "    api.add_box('blue_box', pos=(-1.5, 0.0, 0.0), size=(0.10, 0.10, 0.10),\n"
        "                mass=1.0, rgba=(0.2, 0.4, 0.9, 1.0), friction=2.0,\n"
        "                restitution=0.0, fixed=False)\n"
        "    api.set_velocity('blue_box', linear=(1.8, 0.0, 0.0))\n"
        "    api.set_duration(0.5)\n"
        "\n"
        "SCENE_MANIFEST = {\n"
        "    'scene_kind': 'motion_lab',\n"
        "    'motion_target': 'blue_box',\n"
        "    'families': ['motion_state_change'],\n"
        "}\n"
    )
    r = run_executor(src, stage=2, sim_timeout_s=_TINY_SIM_S)
    assert r.status == FailureReason.OK, r.error_message
    assert r.repair_trace is not None
    actions = {item["kind"] for item in r.repair_trace["actions"]}
    assert "drop_unsupported_kwarg" in actions
    assert "repair_spawn_pos" in actions


def test_stage2_missing_manifest() -> None:
    """Stage=2 without a top-level ``SCENE_MANIFEST`` literal must FAILED_VQA."""

    r = run_executor(_ok_sphere_src(), stage=2, sim_timeout_s=_TINY_SIM_S)
    _expect_status(r, FailureReason.FAILED_VQA)


# ─── Pure VQA-validator unit tests (fast, no physics) ────────────────


class TestValidateVqaPayload:
    def test_valid(self) -> None:
        q, opts, idx = _validate_vqa_payload(("q?", ["a", "b", "c", "d"], 2))
        assert q == "q?"
        assert opts == ["a", "b", "c", "d"]
        assert idx == 2

    @pytest.mark.parametrize(
        "payload",
        [
            ("q?", ["a", "b", "c", "d"]),
            (None, ["a", "b", "c", "d"], 0),
            ("q?", ["a", "b", "c"], 0),
            ("q?", ["a", "b", "c", 5], 0),
            ("q?", ["a", "b", "c", "d"], -1),
            ("q?", ["a", "b", "c", "d"], 4),
            ("q?", ["a", "b", "c", "d"], True),  # bool, not int
        ],
    )
    def test_invalid(self, payload) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError):
            _validate_vqa_payload(payload)


# ─── Coverage guard: every FailureReason has a test. ─────────────────


def test_every_failure_reason_has_a_test() -> None:
    """Meta-test: every ``FailureReason`` must be covered above.

    If someone adds a new reason to docs/architecture.md, this test forces us to
    also add a pipeline test for it.
    """

    covered: set[FailureReason] = {
        FailureReason.OK,
        FailureReason.FAILED_SYNTAX,
        FailureReason.FAILED_SEMANTIC,
        FailureReason.FAILED_RUNTIME,
        FailureReason.FAILED_COMPILE,
        FailureReason.FAILED_OVERLAP,
        FailureReason.FAILED_NAN,
        FailureReason.FAILED_TIMEOUT,
        FailureReason.FAILED_RENDER,
        FailureReason.FAILED_VQA,
        FailureReason.TRIVIAL,
    }
    assert covered == set(FailureReason)


def test_collect_branch_views_skip_does_not_call_run_branch_sims(monkeypatch) -> None:
    from physevolve.executor.pipeline import _collect_branch_views

    called = {"n": 0}

    def _fake_run(*_args: object, **_kwargs: object) -> dict:
        called["n"] += 1
        return {"k": object()}

    monkeypatch.setattr(
        "physevolve.executor.branch_sim.run_branch_sims",
        _fake_run,
    )

    class _FakeTemplate:
        def required_branches(self, _manifest: dict) -> list[str]:
            return ["remove_wall"]

    views = _collect_branch_views(
        _FakeTemplate(),
        {},
        object(),
        seed=0,
        sim_timeout_s=1.0,
        max_branches=4,
        skip_branch_sims=True,
    )
    assert views == {}
    assert called["n"] == 0


def test_collect_branch_views_default_still_calls_run_branch_sims(monkeypatch) -> None:
    from physevolve.executor.pipeline import _collect_branch_views

    called = {"n": 0}
    sentinel = {"k": object()}

    def _fake_run(*_args: object, **_kwargs: object) -> dict:
        called["n"] += 1
        return sentinel

    monkeypatch.setattr(
        "physevolve.executor.branch_sim.run_branch_sims",
        _fake_run,
    )

    class _FakeTemplate:
        def required_branches(self, _manifest: dict) -> list[str]:
            return ["remove_wall"]

    views = _collect_branch_views(
        _FakeTemplate(),
        {},
        object(),
        seed=0,
        sim_timeout_s=1.0,
        max_branches=4,
    )
    assert views is sentinel
    assert called["n"] == 1


# Convenience so pytest discovers helper type without warning.
_ = Callable


@pytest.mark.parametrize("skip_branches", [False, True])
def test_scene_trace_preserves_payloads(skip_branches: bool) -> None:
    """Diagnostic branch traces must not turn a valid scene into FAILED_VQA."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[3]
        / "data/golden_scenes/stage2/01_throwing_pit_board__v01.py"
    ).read_text()
    plain = run_executor(source, stage=2, seed=7, skip_branch_sims=skip_branches)
    trace: dict = {}
    traced = run_executor(source, stage=2, seed=7,
                          skip_branch_sims=skip_branches, trace=trace)
    assert plain.status == traced.status == FailureReason.OK, traced.error_message
    assert [row["payload"] for row in traced.vqa_payloads] == [
        row["payload"] for row in plain.vqa_payloads
    ]
    assert bool(trace["branches"]) is (not skip_branches)
    if not skip_branches:
        assert any(branch["ran"] for branch in trace["branches"])
