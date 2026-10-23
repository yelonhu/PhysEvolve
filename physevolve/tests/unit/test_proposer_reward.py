"""Unit tests for :mod:`physevolve.verification.proposer_reward`."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from physevolve.engine.failure import PROPOSER_PENALTY, FailureReason
from physevolve.engine.schema import (
    BodyDef,
    BodyStatus,
    BodyTrajectory,
    CollisionEvent,
    EnvConfig,
    GeomType,
    JointDef,
    JointType,
    SceneDef,
    SimState,
    compute_scene_id,
)
from physevolve.executor.result import ExecutorResult, make_failure, make_ok
from physevolve.verification.proposer_reward import (
    DEFAULT_INTRINSIC_COMPONENT_WEIGHTS,
    DEFAULT_INTRINSIC_WEIGHT,
    CopyControl,
    IntrinsicInputs,
    apply_copy_control,
    compute_proposer_reward,
    compute_proposer_reward_with_intrinsic,
    compute_scene_signature,
    family_difficulty_weight,
    intrinsic_reward_terms,
    learnability,
)


def _fake_ok() -> ExecutorResult:
    return make_ok(
        scene_id="deadbeef", sim_state=None, vqa_payload=None, walltime_s=0.01  # type: ignore[arg-type]
    )


# ─── Anti-copy / saturation / diversity control ──────────────────────


class TestCopyControl:
    def test_disabled_control_is_noop(self) -> None:
        assert apply_copy_control(0.7, 0.5, None) == pytest.approx(0.7)
        off = CopyControl(enabled=False, copy_penalty=-0.5, is_exact_copy=True)
        assert apply_copy_control(0.7, 0.5, off) == pytest.approx(0.7)

    def test_exact_copy_forces_penalty(self) -> None:
        ctrl = CopyControl(enabled=True, is_exact_copy=True, copy_penalty=-0.5)
        assert apply_copy_control(1.0, 0.5, ctrl) == pytest.approx(-0.5)

    def test_batch_duplicate_forces_penalty(self) -> None:
        ctrl = CopyControl(enabled=True, is_batch_duplicate=True, copy_penalty=-0.5)
        assert apply_copy_control(0.9, 0.5, ctrl) == pytest.approx(-0.5)

    def test_near_copy_caps_reward(self) -> None:
        ctrl = CopyControl(
            enabled=True,
            near_copy_similarity=0.97,
            near_copy_threshold=0.95,
            copy_cap=0.1,
        )
        assert apply_copy_control(0.8, 0.5, ctrl) == pytest.approx(0.1)

    def test_diversity_bonus_added(self) -> None:
        ctrl = CopyControl(enabled=True, diversity_bonus=0.4)
        assert apply_copy_control(0.5, 0.5, ctrl) == pytest.approx(0.9)

    def test_saturation_penalty(self) -> None:
        ctrl = CopyControl(
            enabled=True, saturation_threshold=0.85, saturation_weight=1.0
        )
        # mean_acc=1.0 -> overshoot=1.0 -> reward drops by 1.0.
        assert apply_copy_control(0.5, 1.0, ctrl) == pytest.approx(-0.5)

    def test_compute_reward_exact_copy_overrides_learnability(self) -> None:
        ctrl = CopyControl(enabled=True, is_exact_copy=True, copy_penalty=-0.5)
        reward = compute_proposer_reward(
            _fake_ok(),
            solver_rewards=[1.0, 1.0, 0.0, 0.0],
            copy_control=ctrl,
        )
        assert reward == pytest.approx(-0.5)

    def test_compute_reward_batch_duplicate_without_solver_rewards(self) -> None:
        ctrl = CopyControl(enabled=True, is_batch_duplicate=True, copy_penalty=-0.5)
        # No Solver rewards (duplicates are never sent to the Solver), but the
        # override still forces the penalty rather than the 0.0 short-circuit.
        reward = compute_proposer_reward(
            _fake_ok(), solver_rewards=[], copy_control=ctrl
        )
        assert reward == pytest.approx(-0.5)


def _fake_failure(status: FailureReason) -> ExecutorResult:
    return make_failure(status, "deadbeef", RuntimeError("x"), walltime_s=0.01)


# ─── Learnability tent function ───────────────────────────────────────


class TestLearnability:
    def test_peak_at_half(self) -> None:
        assert learnability(0.5) == pytest.approx(1.0)

    def test_zero_at_extremes(self) -> None:
        assert learnability(0.0) == pytest.approx(0.0)
        assert learnability(1.0) == pytest.approx(0.0)

    def test_symmetric(self) -> None:
        assert learnability(0.25) == pytest.approx(learnability(0.75))

    def test_clips_out_of_range(self) -> None:
        assert learnability(-0.5) == pytest.approx(learnability(0.0))
        assert learnability(1.5) == pytest.approx(learnability(1.0))


# ─── Full reward ──────────────────────────────────────────────────────


class TestComputeProposerReward:
    def test_ok_peak(self) -> None:
        r = _fake_ok()
        assert compute_proposer_reward(r, [0.5, 0.5, 0.5, 0.5]) == pytest.approx(1.0)

    def test_ok_zero_accuracy(self) -> None:
        r = _fake_ok()
        assert compute_proposer_reward(r, [0.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_ok_perfect_accuracy(self) -> None:
        r = _fake_ok()
        assert compute_proposer_reward(r, [1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_ok_insufficient_rollouts_returns_zero(self) -> None:
        r = _fake_ok()
        assert compute_proposer_reward(r, [0.5, 0.5]) == 0.0

    def test_ok_weight_scales_learnability(self) -> None:
        r = _fake_ok()
        val = compute_proposer_reward(
            r, [0.5] * 4, learnability_weight=2.5
        )
        assert val == pytest.approx(2.5)

    def test_family_difficulty_weight_clips_and_noop(self) -> None:
        assert family_difficulty_weight(0.0, 0.0) == pytest.approx(1.0)
        assert family_difficulty_weight(0.0, 1.0) == pytest.approx(1.5)
        assert family_difficulty_weight(1.0, 1.0) == pytest.approx(0.5)
        assert family_difficulty_weight(0.5, 1.0) == pytest.approx(1.0)

    def test_edit_beats_exact_copy_when_learnability_zero(self) -> None:
        """recipe: verbatim copy cannot outscore a template edit via learnability."""

        r = _fake_ok()
        copy_ctrl = CopyControl(
            enabled=True,
            is_exact_copy=True,
            copy_penalty=-0.8,
            diversity_bonus=0.0,
        )
        edit_ctrl = CopyControl(
            enabled=True,
            is_exact_copy=False,
            near_copy_similarity=0.2,
            near_copy_threshold=0.95,
            copy_cap=0.1,
            diversity_bonus=1.5 * (1.0 - 0.2),
        )
        # Even with perfect solver acc, exact copy is forced to -0.8.
        copy_r = compute_proposer_reward(
            r,
            [1.0] * 4,
            learnability_weight=0.0,
            copy_control=copy_ctrl,
        )
        edit_r = compute_proposer_reward(
            r,
            [1.0] * 4,
            learnability_weight=0.0,
            copy_control=edit_ctrl,
        )
        assert copy_r == pytest.approx(-0.8)
        assert edit_r == pytest.approx(1.2)
        assert edit_r > copy_r

    @pytest.mark.parametrize(
        "status",
        [
            FailureReason.FAILED_SYNTAX,
            FailureReason.FAILED_SEMANTIC,
            FailureReason.FAILED_RUNTIME,
            FailureReason.FAILED_COMPILE,
            FailureReason.FAILED_OVERLAP,
            FailureReason.FAILED_NAN,
            FailureReason.FAILED_TIMEOUT,
            FailureReason.FAILED_RENDER,
            FailureReason.FAILED_VQA,
        ],
    )
    def test_failure_returns_exact_penalty(
        self, status: FailureReason
    ) -> None:
        r = _fake_failure(status)
        reward = compute_proposer_reward(r, [0.5, 0.5, 0.5, 0.5])
        assert reward == PROPOSER_PENALTY[status]

    def test_failure_weight_scales_penalty(self) -> None:
        r = _fake_failure(FailureReason.FAILED_SYNTAX)
        val = compute_proposer_reward(r, [0.5] * 4, failure_weight=0.5)
        assert val == pytest.approx(0.5 * PROPOSER_PENALTY[FailureReason.FAILED_SYNTAX])

    def test_trivial_is_zero(self) -> None:
        """TRIVIAL is a non-OK status with penalty 0.0 per docs/architecture.md."""

        from physevolve.executor.result import make_trivial

        r = make_trivial("deadbeef", sim_state=None, walltime_s=0.01)  # type: ignore[arg-type]
        assert compute_proposer_reward(r, [0.5] * 4) == 0.0

    def test_repaired_ok_is_discounted(self) -> None:
        r = make_ok(
            scene_id="deadbeef",
            sim_state=None,  # type: ignore[arg-type]
            vqa_payload=None,
            walltime_s=0.01,
            repair_trace={
                "applied": True,
                "tier": "geometry",
                "reward_discount": 0.75,
                "actions": [],
            },
        )
        assert compute_proposer_reward(r, [0.5] * 4) == pytest.approx(0.75)


# ─── Intrinsic shaping (physics-version intrinsic reward) ─────────────


def _scene_with(
    *,
    bodies: tuple[BodyDef, ...],
    joints: tuple[JointDef, ...] = (),
    duration: float = 2.0,
) -> SceneDef:
    env = EnvConfig(gravity=(0.0, 0.0, -9.81), timestep=0.002, duration=duration, n_frames=32)
    return SceneDef(
        bodies=bodies,
        joints=joints,
        cameras=(),
        env=env,
        workspace_bounds=(1.5, 1.5, 3.0),
    )


def _trajectory(
    name: str,
    *,
    displacement: float,
    max_speed: float,
    final_status: BodyStatus = BodyStatus.STABLE,
) -> BodyTrajectory:
    positions = np.zeros((4, 3), dtype=np.float64)
    positions[-1, 0] = float(displacement)
    orient = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (4, 1))
    vels = np.zeros((4, 3), dtype=np.float64)
    vels[1, 0] = float(max_speed)
    return BodyTrajectory(
        name=name,
        positions=positions,
        orientations=orient,
        lin_velocities=vels,
        ang_velocities=np.zeros((4, 3), dtype=np.float64),
        final_status=final_status,
    )


def _make_sim_state(
    *,
    bodies: tuple[BodyDef, ...],
    joints: tuple[JointDef, ...] = (),
    trajectories: Optional[dict[str, BodyTrajectory]] = None,
    collisions: tuple[CollisionEvent, ...] = (),
) -> SimState:
    scene = _scene_with(bodies=bodies, joints=joints)
    if trajectories is None:
        trajectories = {
            b.name: _trajectory(b.name, displacement=0.3, max_speed=1.0)
            for b in bodies
        }
    video = np.zeros((1, scene.env.n_frames, 8, 8, 3), dtype=np.uint8)
    initial_positions = {b.name: np.asarray(b.pos, dtype=np.float64) for b in bodies}
    initial_vels = {b.name: np.zeros(3, dtype=np.float64) for b in bodies}
    return SimState(
        scene_id=compute_scene_id(scene, 0),
        scene_def=scene,
        trajectories=trajectories,
        collisions=collisions,
        video_frames=video,
        camera_names=("cam0",),
        sim_walltime_s=0.01,
        initial_positions=initial_positions,
        initial_lin_velocities=initial_vels,
    )


def _body(
    name: str,
    *,
    geom: GeomType = GeomType.SPHERE,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.2),
    mass: float = 1.0,
    friction: float = 0.5,
    restitution: float = 0.0,
) -> BodyDef:
    return BodyDef(
        name=name,
        geom_type=geom,
        pos=pos,
        size=(0.1,) if geom == GeomType.SPHERE else (0.1, 0.1, 0.1),
        mass=mass,
        rgba=(0.5, 0.5, 0.5, 1.0),
        friction_slide=friction,
        restitution=restitution,
        joint_type=JointType.FREE,
    )


def _ok_with(sim_state: SimState) -> ExecutorResult:
    return make_ok(
        scene_id=sim_state.scene_id,
        sim_state=sim_state,
        vqa_payload=None,
        walltime_s=0.01,
    )


class TestSceneSignature:
    def test_signature_is_hashable(self) -> None:
        sim = _make_sim_state(bodies=(_body("a"),))
        sig = compute_scene_signature(sim)
        # Must round-trip through a set.
        assert sig in {sig}

    def test_signature_is_invariant_to_value_jitter(self) -> None:
        """Two scenes with identical structural features share a signature."""

        sim_a = _make_sim_state(bodies=(_body("a", mass=1.0), _body("b", pos=(0.4, 0, 0.2))))
        sim_b = _make_sim_state(bodies=(_body("c", mass=1.0), _body("d", pos=(0.4, 0, 0.2))))
        assert compute_scene_signature(sim_a) == compute_scene_signature(sim_b)

    def test_signature_differs_when_geometry_mix_differs(self) -> None:
        sim_two_spheres = _make_sim_state(
            bodies=(_body("a"), _body("b", pos=(0.4, 0, 0.2)))
        )
        sim_sphere_and_box = _make_sim_state(
            bodies=(_body("a"), _body("b", geom=GeomType.BOX, pos=(0.4, 0, 0.2)))
        )
        assert compute_scene_signature(sim_two_spheres) != compute_scene_signature(
            sim_sphere_and_box
        )


class TestIntrinsicRewardTerms:
    def _intrinsic(
        self, sim: SimState, *, history: tuple = ()
    ) -> IntrinsicInputs:
        return IntrinsicInputs(
            sim_state=sim,
            signature=compute_scene_signature(sim),
            history=history,
            weight=DEFAULT_INTRINSIC_WEIGHT,
            component_weights=DEFAULT_INTRINSIC_COMPONENT_WEIGHTS,
        )

    def test_components_in_unit_interval(self) -> None:
        sim = _make_sim_state(
            bodies=(_body("a"), _body("b", pos=(0.4, 0, 0.2), mass=5.0, friction=2.0)),
            collisions=(CollisionEvent(t_step=10, body_a="a", body_b="b", contact_force_norm=5.0),),
        )
        bd = intrinsic_reward_terms(self._intrinsic(sim))
        for name, value in bd.components.items():
            assert 0.0 <= value <= 1.0, f"{name}={value}"
        assert 0.0 <= bd.weighted_sum <= 1.0
        assert bd.weight == pytest.approx(DEFAULT_INTRINSIC_WEIGHT)
        assert bd.total == pytest.approx(bd.weight * bd.weighted_sum)

    def test_novelty_is_one_when_history_empty(self) -> None:
        sim = _make_sim_state(bodies=(_body("a"),))
        bd = intrinsic_reward_terms(self._intrinsic(sim, history=()))
        assert bd.components["diversity"] == pytest.approx(1.0)

    def test_novelty_is_zero_when_signature_repeats(self) -> None:
        sim = _make_sim_state(bodies=(_body("a"),))
        sig = compute_scene_signature(sim)
        bd = intrinsic_reward_terms(self._intrinsic(sim, history=(sig,)))
        assert bd.components["diversity"] == pytest.approx(0.0)

    def test_motion_monotonic_in_displacement(self) -> None:
        body = _body("ball", mass=1.0)
        sim_low = _make_sim_state(
            bodies=(body,),
            trajectories={"ball": _trajectory("ball", displacement=0.06, max_speed=0.2)},
        )
        sim_high = _make_sim_state(
            bodies=(body,),
            trajectories={"ball": _trajectory("ball", displacement=0.5, max_speed=2.0)},
        )
        bd_low = intrinsic_reward_terms(self._intrinsic(sim_low))
        bd_high = intrinsic_reward_terms(self._intrinsic(sim_high))
        assert bd_high.components["motion"] > bd_low.components["motion"]

    def test_complexity_monotonic_in_body_count_within_band(self) -> None:
        sim_one = _make_sim_state(bodies=(_body("a"),))
        sim_three = _make_sim_state(
            bodies=(
                _body("a", pos=(-0.4, 0, 0.2)),
                _body("b", pos=(0.0, 0, 0.2)),
                _body("c", pos=(0.4, 0, 0.2)),
            )
        )
        bd_one = intrinsic_reward_terms(self._intrinsic(sim_one))
        bd_three = intrinsic_reward_terms(self._intrinsic(sim_three))
        assert bd_three.components["complexity"] > bd_one.components["complexity"]

    def test_parameter_identifiability_recognises_two_families(self) -> None:
        sim = _make_sim_state(
            bodies=(
                _body("a", mass=0.5, friction=0.2),
                _body("b", pos=(0.4, 0, 0.2), mass=5.0, friction=2.0),
            )
        )
        bd = intrinsic_reward_terms(self._intrinsic(sim))
        assert bd.components["parameter"] == pytest.approx(1.0)

    def test_rejects_bad_component_weight_count(self) -> None:
        sim = _make_sim_state(bodies=(_body("a"),))
        bad = IntrinsicInputs(
            sim_state=sim,
            signature=compute_scene_signature(sim),
            history=(),
            weight=DEFAULT_INTRINSIC_WEIGHT,
            component_weights=(0.5, 0.5, 0.0),  # wrong length
        )
        with pytest.raises(ValueError, match="component_weights"):
            intrinsic_reward_terms(bad)


class TestComputeProposerRewardWithIntrinsic:
    def _intrinsic(self, sim: SimState) -> IntrinsicInputs:
        return IntrinsicInputs(
            sim_state=sim,
            signature=compute_scene_signature(sim),
            history=(),
            weight=DEFAULT_INTRINSIC_WEIGHT,
            component_weights=DEFAULT_INTRINSIC_COMPONENT_WEIGHTS,
        )

    def test_ok_branch_bounded_above(self) -> None:
        sim = _make_sim_state(
            bodies=(
                _body("a", mass=0.5, friction=0.2),
                _body("b", pos=(0.4, 0, 0.2), mass=5.0, friction=2.0),
            ),
            collisions=(CollisionEvent(t_step=10, body_a="a", body_b="b", contact_force_norm=5.0),),
        )
        r = _ok_with(sim)
        total = compute_proposer_reward(
            r,
            [0.5] * 4,
            intrinsic=self._intrinsic(sim),
        )
        assert 0.0 <= total <= 1.0 + DEFAULT_INTRINSIC_WEIGHT + 1e-9

    def test_intrinsic_strictly_adds_on_top_of_learnability(self) -> None:
        sim = _make_sim_state(
            bodies=(
                _body("a", mass=0.5, friction=0.2),
                _body("b", pos=(0.4, 0, 0.2), mass=5.0, friction=2.0),
            ),
            collisions=(CollisionEvent(t_step=10, body_a="a", body_b="b", contact_force_norm=5.0),),
        )
        r = _ok_with(sim)
        base = compute_proposer_reward(r, [0.5] * 4)
        boosted = compute_proposer_reward(r, [0.5] * 4, intrinsic=self._intrinsic(sim))
        assert boosted > base
        assert boosted - base <= DEFAULT_INTRINSIC_WEIGHT + 1e-9

    def test_intrinsic_skipped_below_min_rollouts(self) -> None:
        sim = _make_sim_state(bodies=(_body("a"),))
        r = _ok_with(sim)
        val, breakdown = compute_proposer_reward_with_intrinsic(
            r, [0.5, 0.5], intrinsic=self._intrinsic(sim)
        )
        assert val == 0.0
        assert breakdown is None

    def test_failure_branch_ignores_intrinsic(self) -> None:
        sim = _make_sim_state(bodies=(_body("a"),))
        r = make_failure(FailureReason.FAILED_SYNTAX, "x", RuntimeError("y"), walltime_s=0.0)
        val, breakdown = compute_proposer_reward_with_intrinsic(
            r, [0.5] * 4, intrinsic=self._intrinsic(sim)
        )
        assert val == PROPOSER_PENALTY[FailureReason.FAILED_SYNTAX]
        assert breakdown is None
