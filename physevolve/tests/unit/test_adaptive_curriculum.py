"""Unit tests for the Stage-1 adaptive curriculum primitives.

CPU-only. Exercises :class:`RollingAccTracker`, :func:`adaptive_pick`,
:func:`derive_difficulty`, and :class:`ReferenceBuffer` directly without
involving the executor, the backend, or veRL.

tracker / sampler key on skill only. Difficulty
is a deterministic per-skill knob via :func:`derive_difficulty`.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from physevolve.engine.schema import (
    BodyDef,
    BodyStatus,
    BodyTrajectory,
    EnvConfig,
    GeomType,
    JointType,
    SceneDef,
    SimState,
    compute_scene_id,
)
from physevolve.training.adaptive_curriculum import (
    DEFAULT_DIFFICULTY_EASY_BELOW,
    DEFAULT_DIFFICULTY_HARD_AT_OR_ABOVE,
    NEUTRAL_ACC,
    WEIGHT_FLOOR,
    ReferenceBuffer,
    ReferenceEntry,
    RollingAccTracker,
    adaptive_pick,
    build_one_line_summary,
    derive_difficulty,
    format_reference_examples,
    pick_weight_snapshot,
)


# ─── shared sim-state fixture ─────────────────────────────────────────


def _make_sim_state(*, friction: float = 0.5, mass: float = 1.0) -> SimState:
    body = BodyDef(
        name="ball",
        geom_type=GeomType.SPHERE,
        pos=(0.0, 0.0, 1.0),
        size=(0.1,),
        mass=mass,
        rgba=(1.0, 0.0, 0.0, 1.0),
        friction_slide=friction,
        restitution=0.0,
        joint_type=JointType.FREE,
    )
    scene = SceneDef(
        bodies=(body,),
        joints=(),
        cameras=(),
        env=EnvConfig(duration=1.0, n_frames=8),
        workspace_bounds=(1.5, 1.5, 3.0),
    )
    positions = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (4, 1))
    orientations = np.tile(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (4, 1)
    )
    trajectory = BodyTrajectory(
        name="ball",
        positions=positions,
        orientations=orientations,
        lin_velocities=np.zeros((4, 3), dtype=np.float64),
        ang_velocities=np.zeros((4, 3), dtype=np.float64),
        final_status=BodyStatus.STABLE,
    )
    return SimState(
        scene_id=compute_scene_id(scene, seed=1),
        scene_def=scene,
        trajectories={"ball": trajectory},
        collisions=(),
        video_frames=np.zeros((1, 8, 8, 8, 3), dtype=np.uint8),
        camera_names=("main",),
        sim_walltime_s=0.01,
        initial_positions={"ball": np.array([0.0, 0.0, 1.0], dtype=np.float64)},
        initial_lin_velocities={"ball": np.zeros(3, dtype=np.float64)},
    )


# ─── RollingAccTracker (skill-only keying) ────────────────


def test_rolling_acc_tracker_cold_start_returns_neutral_value() -> None:
    tracker = RollingAccTracker(window_size=8)

    assert tracker.mean("dynamics_throwing") == NEUTRAL_ACC
    assert tracker.count("dynamics_throwing") == 0
    assert tracker.snapshot() == {}


def test_rolling_acc_tracker_window_eviction() -> None:
    tracker = RollingAccTracker(window_size=3)

    for value in (1.0, 1.0, 1.0, 0.0):
        tracker.update("dynamics_collision", value)

    assert tracker.count("dynamics_collision") == 3
    # The earliest 1.0 should have been evicted; remaining window is (1, 1, 0).
    assert tracker.mean("dynamics_collision") == pytest.approx(2.0 / 3.0)


def test_rolling_acc_tracker_clamps_out_of_range_updates() -> None:
    tracker = RollingAccTracker(window_size=4)

    tracker.update("optics_motion", -0.5)
    tracker.update("optics_motion", 1.7)

    # -0.5 -> 0.0, 1.7 -> 1.0, mean 0.5.
    assert tracker.mean("optics_motion") == pytest.approx(0.5)


def test_rolling_acc_tracker_snapshot_is_sorted_and_omits_empty() -> None:
    tracker = RollingAccTracker(window_size=4)
    tracker.update("dynamics_throwing", 0.4)
    tracker.update("dynamics_collision", 0.7)

    snap = tracker.snapshot()

    assert list(snap.keys()) == ["dynamics_collision", "dynamics_throwing"]
    assert snap["dynamics_collision"] == pytest.approx(0.7)
    assert snap["dynamics_throwing"] == pytest.approx(0.4)


# ─── adaptive_pick (skill-only return) ────────────────────


_SKILLS = (
    "dynamics_collision",
    "dynamics_throwing",
    "dynamics_motion_state",
    "property_mass_contrast",
)


def test_adaptive_pick_returns_a_skill_string() -> None:
    tracker = RollingAccTracker(window_size=4)
    rng = random.Random(0)

    pick = adaptive_pick(rng, tracker, skills=_SKILLS, step=1)

    assert isinstance(pick, str)
    assert pick in _SKILLS


def test_adaptive_pick_uniform_during_cold_start() -> None:
    """During cold start, every skill should be reachable with deterministic rngs."""

    tracker = RollingAccTracker(window_size=4)
    # Pollute one skill with extreme data — should be ignored during cold start.
    for _ in range(4):
        tracker.update("dynamics_collision", 1.0)

    seen: set[str] = set()
    for trial in range(2000):
        rng = random.Random(trial)
        seen.add(
            adaptive_pick(
                rng,
                tracker,
                skills=_SKILLS,
                step=1,
                cold_start_uniform_steps=8,
            )
        )
    assert seen == set(_SKILLS)


def test_adaptive_pick_concentrates_on_middle_acc_after_warmup() -> None:
    """Post-cold-start, skills near acc=0.5 should dominate."""

    tracker = RollingAccTracker(window_size=8)
    for _ in range(8):
        tracker.update("dynamics_collision", 1.0)  # saturated
        tracker.update("dynamics_throwing", 0.5)  # learnable
        tracker.update("dynamics_motion_state", 0.0)  # hopeless
        tracker.update("property_mass_contrast", 0.5)  # learnable

    picks: dict[str, int] = {s: 0 for s in _SKILLS}
    for trial in range(5000):
        rng = random.Random(trial)
        s = adaptive_pick(
            rng,
            tracker,
            skills=_SKILLS,
            step=100,
            cold_start_uniform_steps=8,
        )
        picks[s] += 1

    # Learnable skills (acc=0.5, weight ~1.0) vs floored (acc=1.0 / 0.0, w=0.05):
    # the ratio should be ~20x in expectation. Allow a generous margin.
    assert picks["dynamics_throwing"] > picks["dynamics_collision"] * 5
    assert picks["property_mass_contrast"] > picks["dynamics_motion_state"] * 5


def test_adaptive_pick_keeps_floor_weight_for_extreme_skills() -> None:
    """An acc=1.0 skill must still be sampled occasionally (exploration floor)."""

    tracker = RollingAccTracker(window_size=4)
    # Every skill saturated to acc=1.0 -> all weights collapse to the floor.
    for skill in _SKILLS:
        for _ in range(4):
            tracker.update(skill, 1.0)

    extreme_count = 0
    for trial in range(2000):
        rng = random.Random(trial)
        s = adaptive_pick(
            rng,
            tracker,
            skills=_SKILLS,
            step=100,
            cold_start_uniform_steps=4,
        )
        if s == "dynamics_collision":
            extreme_count += 1
    # With all 4 skills at floor weight, each receives ~25% of picks.
    assert extreme_count > 250


def test_adaptive_pick_rejects_empty_vocabulary() -> None:
    tracker = RollingAccTracker(window_size=4)
    rng = random.Random(0)

    with pytest.raises(ValueError):
        adaptive_pick(rng, tracker, skills=(), step=1)


def test_pick_weight_snapshot_sums_to_one_and_is_positive() -> None:
    tracker = RollingAccTracker(window_size=4)
    tracker.update("dynamics_collision", 0.5)
    tracker.update("dynamics_throwing", 0.9)

    snap = pick_weight_snapshot(tracker, skills=_SKILLS)

    assert set(snap) == set(_SKILLS)
    assert sum(snap.values()) == pytest.approx(1.0)
    for weight in snap.values():
        assert weight > 0.0


# ─── derive_difficulty (accuracy-based difficulty) ───────────────────


def test_derive_difficulty_cold_start_defaults_to_easy() -> None:
    assert derive_difficulty(None) == "easy"


@pytest.mark.parametrize(
    "acc,expected",
    [
        (0.0, "easy"),
        (0.1, "easy"),
        (DEFAULT_DIFFICULTY_EASY_BELOW - 1e-9, "easy"),
        (DEFAULT_DIFFICULTY_EASY_BELOW, "medium"),
        (0.5, "medium"),
        (DEFAULT_DIFFICULTY_HARD_AT_OR_ABOVE - 1e-9, "medium"),
        (DEFAULT_DIFFICULTY_HARD_AT_OR_ABOVE, "hard"),
        (0.8, "hard"),
        (1.0, "hard"),
    ],
)
def test_derive_difficulty_thresholds(acc: float, expected: str) -> None:
    assert derive_difficulty(acc) == expected


def test_derive_difficulty_clamps_out_of_range() -> None:
    assert derive_difficulty(-1.0) == "easy"
    assert derive_difficulty(2.0) == "hard"


def test_derive_difficulty_custom_thresholds() -> None:
    # Easier escalation: hard kicks in at 0.5.
    assert derive_difficulty(0.55, easy_below=0.2, hard_at_or_above=0.5) == "hard"
    assert derive_difficulty(0.3, easy_below=0.2, hard_at_or_above=0.5) == "medium"


def test_derive_difficulty_rejects_inconsistent_thresholds() -> None:
    with pytest.raises(ValueError, match="hard_at_or_above > easy_below"):
        derive_difficulty(0.5, easy_below=0.6, hard_at_or_above=0.5)


# ─── ReferenceBuffer ─────────────────────────────────────────────────


def test_reference_buffer_sample_returns_k_or_fewer_when_buffer_small() -> None:
    buf = ReferenceBuffer(maxlen=8)
    buf.record("dynamics_collision", _make_sim_state(), "scene_a")

    rng = random.Random(0)
    sampled = buf.sample(rng, "dynamics_collision", k=3)

    assert len(sampled) == 1
    assert sampled[0].scene_id == "scene_a"


def test_reference_buffer_one_line_summary_excludes_full_scene_def() -> None:
    """Summaries must not leak raw PhysDSL source / body names / positions."""

    sim_state = _make_sim_state()
    summary = build_one_line_summary(sim_state)

    assert "ball" not in summary  # body name is private
    assert "pos=" not in summary  # raw position must not leak
    assert "def build_scene" not in summary  # no PhysDSL source
    assert "sphere" in summary
    assert "n_dynamic" in summary
    assert "mass_levels" in summary


def test_reference_buffer_per_skill_isolation() -> None:
    buf = ReferenceBuffer(maxlen=8)
    for i in range(3):
        buf.record(
            "dynamics_collision",
            _make_sim_state(),
            f"collision_scene_{i}",
            summary=f"collision summary {i}",
            structural_signature=f"collision_sig_{i}",
        )
    buf.record(
        "dynamics_throwing",
        _make_sim_state(),
        "throwing_scene_0",
        summary="throwing summary 0",
        structural_signature="throwing_sig_0",
    )

    rng = random.Random(0)
    sampled = buf.sample(rng, "dynamics_collision", k=2)
    assert all(entry.scene_id.startswith("collision_scene_") for entry in sampled)

    rng = random.Random(1)
    sampled = buf.sample(rng, "dynamics_throwing", k=3)
    throwing = [e for e in sampled if e.scene_id.startswith("throwing")]
    collision = [e for e in sampled if e.scene_id.startswith("collision")]
    assert len(throwing) == 1
    assert len(collision) >= 1


def test_reference_buffer_can_disable_cross_skill_fallback() -> None:
    buf = ReferenceBuffer(maxlen=8)
    buf.record("collision_arena", _make_sim_state(), "collision_scene")

    rng = random.Random(0)
    sampled = buf.sample(
        rng,
        "throwing_pit_board",
        k=2,
        allow_cross_skill_fallback=False,
    )

    assert sampled == []


def test_reference_buffer_eviction_respects_maxlen() -> None:
    buf = ReferenceBuffer(maxlen=2)
    for i in range(5):
        buf.record(
            "dynamics_collision",
            _make_sim_state(),
            f"scene_{i}",
            summary=f"summary {i}",
            structural_signature=f"sig_{i}",
        )

    sizes = buf.snapshot_sizes()
    assert sizes["dynamics_collision"] == 2

    rng = random.Random(0)
    sampled = buf.sample(rng, "dynamics_collision", k=2)
    sampled_ids = {entry.scene_id for entry in sampled}
    assert sampled_ids <= {"scene_3", "scene_4"}


def test_reference_buffer_record_accepts_explicit_summary_override() -> None:
    buf = ReferenceBuffer(maxlen=4)
    entry = buf.record(
        "dynamics_collision",
        _make_sim_state(),
        "scene_explicit",
        summary="injected summary",
        structural_signature="explicit_sig",
    )

    assert isinstance(entry, ReferenceEntry)
    assert entry.one_line_summary == "injected summary"

    rng = random.Random(0)
    sampled = buf.sample(rng, "dynamics_collision", k=1)
    assert sampled[0].one_line_summary == "injected summary"


def test_reference_buffer_summary_preserves_repair_metadata() -> None:
    buf = ReferenceBuffer(maxlen=4)
    entry = buf.record(
        "throwing_pit_board",
        _make_sim_state(),
        "scene_repaired",
        scene_kind="throwing_pit_board",
        families=("throwing_final_destination",),
        n_questions=2,
        repair_tier="geometry",
    )

    assert entry.repair_tier == "geometry"
    assert entry.n_questions == 2
    assert entry.families == ("throwing_final_destination",)
    assert "repair=geometry" in entry.one_line_summary


def test_format_reference_examples_strips_to_strings() -> None:
    buf = ReferenceBuffer(maxlen=2)
    buf.record(
        "dynamics_collision",
        _make_sim_state(),
        "scene_a",
        summary="alpha summary",
        structural_signature="alpha_sig",
    )
    buf.record(
        "dynamics_collision",
        _make_sim_state(),
        "scene_b",
        summary="beta summary",
        structural_signature="beta_sig",
    )

    rng = random.Random(0)
    sampled = buf.sample(rng, "dynamics_collision", k=2)
    strings = format_reference_examples(sampled)

    assert isinstance(strings, tuple)
    assert set(strings) == {"alpha summary", "beta summary"}


def test_reference_entry_rejects_empty_fields() -> None:
    with pytest.raises(ValueError):
        ReferenceEntry(scene_id="", one_line_summary="x")
    with pytest.raises(ValueError):
        ReferenceEntry(scene_id="x", one_line_summary="")


def test_reference_buffer_deduplicates_structural_signatures() -> None:
    buf = ReferenceBuffer(maxlen=8)
    first = buf.record(
        "collision_arena",
        _make_sim_state(),
        "scene_a",
        summary="same apparatus",
        scene_kind="collision_arena",
        families=("collision_final_state",),
        n_questions=1,
        repair_tier="none",
        structural_signature="shared_sig",
    )
    second = buf.record(
        "collision_arena",
        _make_sim_state(),
        "scene_b",
        summary="different words",
        scene_kind="collision_arena",
        families=("collision_final_state",),
        n_questions=1,
        repair_tier="none",
        structural_signature="shared_sig",
    )

    assert second is first
    assert buf.snapshot_sizes()["collision_arena"] == 1


def test_weight_floor_constant_is_positive() -> None:
    """Guard against accidental floor=0 regressions that would starve skills."""

    assert WEIGHT_FLOOR > 0.0
