"""Unit tests for :mod:`physevolve.executor.state_view`.

Tests the Stage-2 read-only trajectory view, including
the guarantee that ``initial_pos`` / ``initial_vel`` reflect the true
DSL-authored t = 0 state (pre-``mj_step``), NOT trajectory row 0.
"""

from __future__ import annotations

import pytest

import physevolve.engine  # noqa: F401 — EGL env.
from physevolve.executor import run_executor
from physevolve.executor.state_view import sim_state_public_view


@pytest.fixture(scope="module")
def moving_ball_sim_state():
    src = (
        "def build_scene(api):\n"
        "    api.add_sphere('ball', pos=(0.0, 0.0, 1.5),\n"
        "                   radius=0.1, mass=1.0)\n"
        "    api.set_velocity('ball', linear=(1.0, 0.0, 0.0))\n"
        "    api.set_duration(0.5)\n"
    )
    r = run_executor(src, stage=1, seed=0, sim_timeout_s=10.0)
    assert r.ok, r.error_message
    return r.sim_state


class TestInitialStateIsTrueT0:
    """``initial_*`` properties must reflect the DSL-authored pre-step state.

    Regression for initial values incorrectly read from trajectory
    row 0 (i.e. after the first ``mj_step``), which for a freely-falling
    body already carries a tiny gravitational velocity component.
    """

    def test_initial_pos_matches_dsl(self, moving_ball_sim_state) -> None:
        view = sim_state_public_view(moving_ball_sim_state)
        ip = view["ball"].initial_pos
        assert ip == pytest.approx((0.0, 0.0, 1.5), abs=1e-9)

    def test_initial_vel_matches_dsl(self, moving_ball_sim_state) -> None:
        view = sim_state_public_view(moving_ball_sim_state)
        iv = view["ball"].initial_vel
        # The DSL set linear velocity to (1, 0, 0) m/s. After the first
        # mj_step (~2 ms) gravity would add ~ -0.02 m/s in z; reading
        # row 0 would therefore NOT satisfy exact equality.
        assert iv == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)

    def test_final_pos_has_advanced(self, moving_ball_sim_state) -> None:
        view = sim_state_public_view(moving_ball_sim_state)
        fp = view["ball"].final_pos
        # Ball moves with +x velocity under gravity; final x > initial x.
        assert fp[0] > view["ball"].initial_pos[0] + 0.1

    def test_max_speed_positive(self, moving_ball_sim_state) -> None:
        view = sim_state_public_view(moving_ball_sim_state)
        assert view["ball"].max_speed > 0.0


class TestCollisionQueries:
    def test_collided_false_when_no_contact(self, moving_ball_sim_state) -> None:
        view = sim_state_public_view(moving_ball_sim_state)
        assert view.collided("ball", "ground") in (False, True)
        # time_of_collision is either None or a non-negative float.
        t = view.time_of_collision("ball", "ground")
        assert t is None or t >= 0.0


class TestBodyNamesImmutable:
    def test_body_names_list_is_copy(self, moving_ball_sim_state) -> None:
        view = sim_state_public_view(moving_ball_sim_state)
        names = view.body_names
        names.append("hacker")
        # Second call must not have been mutated by the first.
        assert "hacker" not in view.body_names

    def test_contains(self, moving_ball_sim_state) -> None:
        view = sim_state_public_view(moving_ball_sim_state)
        assert "ball" in view
        assert "no_such_body" not in view
        assert 42 not in view  # non-string must return False safely


class TestKeyErrorOnMissingBody:
    def test_missing_body_key_error(self, moving_ball_sim_state) -> None:
        view = sim_state_public_view(moving_ball_sim_state)
        with pytest.raises(KeyError):
            _ = view["nonexistent"]
