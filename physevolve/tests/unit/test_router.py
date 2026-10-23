"""Unit tests for :mod:`physevolve.tasks.router` and the 3 mode builders."""

from __future__ import annotations

import json
import random

import pytest

import physevolve.engine  # noqa: F401 — EGL env.
from physevolve.executor import run_executor
from physevolve.tasks import (
    SolverObservation,
    TaskMode,
    assemble_observation,
    parse_solver_output,
    select_task_mode,
)
from physevolve.tasks.inverse import (
    VELOCITY_BINS,
    bin_to_velocity,
    velocity_to_bin,
)


# ─── Shared tiny scene fixture ────────────────────────────────────────


_OK_SRC = '''
def build_scene(api):
    api.add_sphere('ball', pos=(0.0, 0.0, 1.5), radius=0.1, mass=1.0,
                   friction=0.2)
    api.set_velocity('ball', linear=(1.0, 0.0, 0.0))
    api.set_duration(0.5)
'''


@pytest.fixture(scope="module")
def sim_state():
    r = run_executor(_OK_SRC, stage=1, seed=0, sim_timeout_s=10.0)
    assert r.ok, r.error_message
    return r.sim_state


# ─── Mode selection ──────────────────────────────────────────────────


class TestSelectTaskMode:
    def test_all_modes_reachable(self, sim_state) -> None:
        """Across 2000 draws, every mode should fire at least once."""

        rng = random.Random(123)
        seen: set[TaskMode] = set()
        for _ in range(2000):
            seen.add(select_task_mode(sim_state, rng))
        assert seen == {TaskMode.FORWARD, TaskMode.INVERSE, TaskMode.PARAMETER}

    def test_weights_roughly_match(self, sim_state) -> None:
        """Forward should be drawn ~40% of the time (tolerance ±5%)."""

        rng = random.Random(42)
        counts = {m: 0 for m in TaskMode}
        n = 10000
        for _ in range(n):
            counts[select_task_mode(sim_state, rng)] += 1
        assert abs(counts[TaskMode.FORWARD] / n - 0.40) < 0.05
        assert abs(counts[TaskMode.INVERSE] / n - 0.30) < 0.05
        assert abs(counts[TaskMode.PARAMETER] / n - 0.30) < 0.05


# ─── Observation structure per mode ───────────────────────────────────


class TestAssembleObservation:
    def test_forward_shapes(self, sim_state) -> None:
        obs = assemble_observation(sim_state, TaskMode.FORWARD)
        assert isinstance(obs, SolverObservation)
        assert obs.mode is TaskMode.FORWARD
        assert obs.scene_id == sim_state.scene_id
        # First half of frames.
        n_cam, n_full = sim_state.video_frames.shape[:2]
        assert obs.video_frames.shape == (n_cam, n_full // 2, *sim_state.video_frames.shape[2:])
        # GT schema.
        assert set(obs.ground_truth["ball"].keys()) == {"grid", "status"}
        grid = obs.ground_truth["ball"]["grid"]
        assert len(grid) == 3 and all(0 <= g <= 15 for g in grid)
        assert obs.ground_truth["ball"]["status"] in {"stable", "moving", "fallen", "ejected"}

    def test_inverse_shapes(self, sim_state) -> None:
        obs = assemble_observation(sim_state, TaskMode.INVERSE)
        # Second half.
        n_cam, n_full = sim_state.video_frames.shape[:2]
        expected_frames = n_full - n_full // 2
        assert obs.video_frames.shape == (n_cam, expected_frames, *sim_state.video_frames.shape[2:])
        vg = obs.ground_truth["ball"]["initial_vel_grid"]
        assert len(vg) == 3 and all(0 <= v <= 6 for v in vg)

    def test_parameter_shapes(self, sim_state) -> None:
        obs = assemble_observation(sim_state, TaskMode.PARAMETER)
        # Full video.
        assert obs.video_frames.shape == sim_state.video_frames.shape
        gt = obs.ground_truth["ball"]
        assert 1 <= gt["mass_level"] <= 5
        assert 1 <= gt["friction_level"] <= 5

    def test_prompt_contains_mode_cue(self, sim_state) -> None:
        """Sanity: each mode's prompt says what the Solver is predicting."""

        fwd = assemble_observation(sim_state, TaskMode.FORWARD).text_prompt
        inv = assemble_observation(sim_state, TaskMode.INVERSE).text_prompt
        par = assemble_observation(sim_state, TaskMode.PARAMETER).text_prompt
        assert "final state" in fwd.lower()
        assert "initial velocity" in inv.lower()
        assert "hidden physical parameters" in par.lower()


# ─── Parse Solver output ─────────────────────────────────────────────


class TestParseSolverOutput:
    def test_forward_happy(self) -> None:
        raw = (
            "<think>stuff</think>"
            "<answer>"
            '{"a": {"grid": [1, 2, 3], "status": "moving"}}'
            "</answer>"
        )
        out = parse_solver_output(raw, TaskMode.FORWARD)
        assert out == {"a": {"grid": [1, 2, 3], "status": "moving"}}

    def test_inverse_happy(self) -> None:
        raw = '<answer>{"a": {"initial_vel_grid": [3, 3, 5]}}</answer>'
        out = parse_solver_output(raw, TaskMode.INVERSE)
        assert out == {"a": {"initial_vel_grid": [3, 3, 5]}}

    def test_parameter_happy(self) -> None:
        raw = '<answer>{"a": {"mass_level": 2, "friction_level": 4}}</answer>'
        out = parse_solver_output(raw, TaskMode.PARAMETER)
        assert out == {"a": {"mass_level": 2, "friction_level": 4}}

    @pytest.mark.parametrize(
        "raw",
        [
            "no tags here",
            "<answer>not json</answer>",
            "<answer>{\"a\": {\"grid\": [1, 2], \"status\": \"moving\"}}</answer>",
            "<answer>{\"a\": {\"grid\": [1, 2, 99], \"status\": \"moving\"}}</answer>",
            "<answer>{\"a\": {\"grid\": [1, 2, 3], \"status\": \"zooming\"}}</answer>",
            "<answer>[\"not a dict\"]</answer>",
        ],
    )
    def test_forward_invalid(self, raw: str) -> None:
        assert parse_solver_output(raw, TaskMode.FORWARD) is None

    def test_inverse_invalid_bin_out_of_range(self) -> None:
        raw = '<answer>{"a": {"initial_vel_grid": [0, 7, 3]}}</answer>'
        assert parse_solver_output(raw, TaskMode.INVERSE) is None

    def test_parameter_invalid_level_out_of_range(self) -> None:
        raw = '<answer>{"a": {"mass_level": 6, "friction_level": 1}}</answer>'
        assert parse_solver_output(raw, TaskMode.PARAMETER) is None

    def test_parameter_rejects_bool_level(self) -> None:
        """Python ``bool`` is a subclass of ``int`` — must be rejected."""

        raw = '<answer>{"a": {"mass_level": true, "friction_level": 1}}</answer>'
        assert parse_solver_output(raw, TaskMode.PARAMETER) is None

    def test_forward_extracts_innermost_block(self) -> None:
        """Multiple ``<answer>`` tags: use the first (regex is DOTALL non-greedy)."""

        raw = (
            '<answer>{"a": {"grid": [0, 0, 0], "status": "stable"}}</answer>'
            '<answer>{"b": {"grid": [1, 1, 1], "status": "moving"}}</answer>'
        )
        out = parse_solver_output(raw, TaskMode.FORWARD)
        # Regex greedy on outer fences + non-greedy body picks the first block.
        assert out is not None
        assert "a" in out


# ─── Velocity bin helpers ────────────────────────────────────────────


class TestVelocityBins:
    def test_roundtrip_on_bin_centres(self) -> None:
        for i, v in enumerate(VELOCITY_BINS):
            assert velocity_to_bin(v) == i
            assert bin_to_velocity(i) == v

    def test_velocity_to_bin_nearest(self) -> None:
        # 2.0 is halfway between bins 1 (3.0→idx 5) and -1 (idx 2);
        # |2 - 1| = 1, |2 - 3| = 1 → tie, numpy argmin picks the first
        # which is idx 4 (v=1.0).
        assert velocity_to_bin(2.0) == 4

    def test_bin_to_velocity_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            bin_to_velocity(-1)
        with pytest.raises(ValueError):
            bin_to_velocity(len(VELOCITY_BINS))


# ─── Grid discretization round-trip ────────────────


class TestGridRoundTrip:
    """Covers the ``tasks/router`` row: grid round-trip.

    Lives here because the tasks layer is the only consumer that needs
    to reason about grid semantics end-to-end; the schema module owns
    the primitives but the contract is exercised by the router.
    """

    def test_world_to_grid_is_exact_on_cell_centres(self) -> None:
        from physevolve.engine.schema import (
            GRID_RES,
            grid_to_world,
            world_to_grid,
        )

        for gx in range(GRID_RES):
            for gy in range(GRID_RES):
                for gz in range(GRID_RES):
                    world = grid_to_world((gx, gy, gz))
                    assert world_to_grid(world) == (gx, gy, gz)

    def test_grid_to_world_rejects_out_of_range(self) -> None:
        from physevolve.engine.schema import grid_to_world

        with pytest.raises(ValueError):
            grid_to_world((-1, 0, 0))
        with pytest.raises(ValueError):
            grid_to_world((0, 16, 0))
        with pytest.raises(ValueError):
            grid_to_world((0, 0, 16))

    def test_world_to_grid_clips_out_of_workspace(self) -> None:
        from physevolve.engine.schema import world_to_grid

        # Far beyond +x bound: must clip to the last cell index.
        assert world_to_grid((100.0, 0.0, 1.5))[0] == 15
        # Deep below the floor: must clip to 0 in z.
        assert world_to_grid((0.0, 0.0, -99.0))[2] == 0


# ─── Anti-pattern check: SceneDef never leaks to the Solver ──────────


def test_ground_truth_does_not_contain_scene_def(sim_state) -> None:
    # Stage-2 VQA requires an explicit vqa_payload; supply a synthetic one
    # so the per-mode iteration covers VQA too.
    vqa_payload = ("Where is the ball?", ["here", "there", "up", "down"], 0)
    for mode in TaskMode:
        if mode is TaskMode.VQA:
            obs = assemble_observation(sim_state, mode, vqa_payload=vqa_payload)
        else:
            obs = assemble_observation(sim_state, mode)
        # SolverObservation exposes four fields; none should carry a
        # SceneDef or SimState. Check via repr.
        gt_repr = json.dumps(obs.ground_truth)  # JSON-serialisable => no python objects
        assert "SceneDef" not in gt_repr
        assert "SimState" not in gt_repr
