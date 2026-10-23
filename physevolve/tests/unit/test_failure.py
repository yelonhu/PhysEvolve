"""Unit tests for :mod:`physevolve.engine.failure`.

Tests ``engine/failure.py``: every FailureReason
reachable-from-table, penalty completeness, and exception subclass sanity.
"""

from __future__ import annotations

import pytest

from physevolve.engine.failure import (
    PROPOSER_PENALTY,
    DSLValidationError,
    FailureReason,
    InitialOverlapError,
    SimulationDivergedError,
    SimulationTimeoutError,
)


class TestFailureReasonEnum:
    def test_expected_members(self) -> None:
        expected = {
            "OK",
            "FAILED_SYNTAX",
            "FAILED_SEMANTIC",
            "FAILED_RUNTIME",
            "FAILED_COMPILE",
            "FAILED_OVERLAP",
            "FAILED_NAN",
            "FAILED_TIMEOUT",
            "FAILED_RENDER",
            "FAILED_VQA",
            "TRIVIAL",
        }
        assert {r.name for r in FailureReason} == expected
        assert len(FailureReason) == 11

    def test_is_str_enum(self) -> None:
        assert FailureReason.OK == "ok"
        assert FailureReason.FAILED_VQA == "vqa"


class TestProposerPenalty:
    def test_covers_every_reason(self) -> None:
        # Missing any FailureReason would silently yield KeyError in the
        # pipeline; guard against that here.
        assert set(PROPOSER_PENALTY.keys()) == set(FailureReason)

    def test_locked_values(self) -> None:
        # Check the default failure penalties.
        assert PROPOSER_PENALTY[FailureReason.OK] == 0.0
        assert PROPOSER_PENALTY[FailureReason.TRIVIAL] == 0.0
        assert PROPOSER_PENALTY[FailureReason.FAILED_SYNTAX] == -1.0
        assert PROPOSER_PENALTY[FailureReason.FAILED_SEMANTIC] == -1.0
        assert PROPOSER_PENALTY[FailureReason.FAILED_VQA] == -1.0
        assert PROPOSER_PENALTY[FailureReason.FAILED_RUNTIME] == -0.8
        assert PROPOSER_PENALTY[FailureReason.FAILED_COMPILE] == -0.5
        assert PROPOSER_PENALTY[FailureReason.FAILED_OVERLAP] == -0.5
        assert PROPOSER_PENALTY[FailureReason.FAILED_NAN] == pytest.approx(-0.3)
        assert PROPOSER_PENALTY[FailureReason.FAILED_TIMEOUT] == pytest.approx(-0.3)
        assert PROPOSER_PENALTY[FailureReason.FAILED_RENDER] == pytest.approx(-0.3)

    def test_all_penalties_non_positive(self) -> None:
        for reason, penalty in PROPOSER_PENALTY.items():
            assert penalty <= 0.0, f"{reason} has positive penalty {penalty}"

    def test_all_penalties_above_minus_one(self) -> None:
        # -1.0 is the worst allowed penalty (LOCKED floor).
        for reason, penalty in PROPOSER_PENALTY.items():
            assert penalty >= -1.0, f"{reason} below floor: {penalty}"

    def test_ok_and_trivial_are_zero(self) -> None:
        assert PROPOSER_PENALTY[FailureReason.OK] == 0.0
        assert PROPOSER_PENALTY[FailureReason.TRIVIAL] == 0.0


class TestEngineExceptions:
    def test_dsl_validation_is_value_error(self) -> None:
        assert issubclass(DSLValidationError, ValueError)
        with pytest.raises(DSLValidationError):
            raise DSLValidationError("bad mass")

    def test_sim_diverged_is_runtime_error(self) -> None:
        assert issubclass(SimulationDivergedError, RuntimeError)
        with pytest.raises(SimulationDivergedError):
            raise SimulationDivergedError("qpos NaN at step 42")

    def test_sim_timeout_is_timeout_error(self) -> None:
        assert issubclass(SimulationTimeoutError, TimeoutError)
        with pytest.raises(SimulationTimeoutError):
            raise SimulationTimeoutError("wall-clock exceeded")

    def test_initial_overlap_is_value_error(self) -> None:
        assert issubclass(InitialOverlapError, ValueError)
        with pytest.raises(InitialOverlapError):
            raise InitialOverlapError("box_a and box_b overlap")
