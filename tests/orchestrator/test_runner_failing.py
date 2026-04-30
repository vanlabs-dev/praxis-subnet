"""Tests verifying run_validator failure patterns for adversarial environments.

Empirically-observed failure patterns (verified before writing):

TrivialEnv (reward=+1 on step 1, terminates immediately):
    - reset_correctness: PASSED (valid tuple, obs in space, idempotent)
    - reward_bounds:     PASSED (reward=+1.0 within [-1.0, 1.0] step bounds)
    - determinism_anchor: FAILED (make_adversarial_manifest uses fake hashes "0"*64)
    - determinism_self_consistency: PASSED (env IS deterministic at validator seeds)
    - solver_baseline:  FAILED (failure_reason="trivial_random_baseline";
                                random policy easily clears T=0.5)

LiarTupleShape (reset returns bare ndarray, not a 2-tuple):
    - reset_correctness: FAILED (TUPLE_SHAPE violations)
    - reward_bounds:     ERRORED (iter_rollout cannot unpack non-tuple return)
    - determinism_anchor: ERRORED (rollout crashes on env.reset unpack)
    - determinism_self_consistency: ERRORED (same crash)
    - solver_baseline:  ERRORED (same crash)

Both environments produce overall passed=False.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "tests"))

from checks._adversarial_envs import make_adversarial_manifest  # type: ignore[import]

from praxis.orchestrator import CheckId, ValidatorReport, run_validator


@pytest.fixture(scope="module")
def trivial_report() -> ValidatorReport:
    m = make_adversarial_manifest("trivial-env", "TrivialEnv")
    return run_validator(m)


@pytest.fixture(scope="module")
def liar_tuple_report() -> ValidatorReport:
    m = make_adversarial_manifest("liar-tuple", "LiarTupleShape")
    return run_validator(m)


class TestTrivialEnvReport:
    """TrivialEnv: determinism_anchor and solver_baseline fail; others pass."""

    def test_overall_not_passed(self, trivial_report: ValidatorReport) -> None:
        assert trivial_report.passed is False

    def test_failure_summary_contains_expected_checks(
        self, trivial_report: ValidatorReport
    ) -> None:
        assert CheckId.DETERMINISM_ANCHOR.value in trivial_report.failure_summary
        assert CheckId.SOLVER_BASELINE.value in trivial_report.failure_summary

    def test_reset_correctness_passed(self, trivial_report: ValidatorReport) -> None:
        assert trivial_report.check_results[CheckId.RESET_CORRECTNESS].outcome == "passed"

    def test_reward_bounds_passed(self, trivial_report: ValidatorReport) -> None:
        assert trivial_report.check_results[CheckId.REWARD_BOUNDS].outcome == "passed"

    def test_determinism_anchor_failed(self, trivial_report: ValidatorReport) -> None:
        # Fake hashes in make_adversarial_manifest cause all anchors to mismatch.
        assert trivial_report.check_results[CheckId.DETERMINISM_ANCHOR].outcome == "failed"

    def test_determinism_self_consistency_passed(
        self, trivial_report: ValidatorReport
    ) -> None:
        # TrivialEnv IS deterministic; the self-consistency check passes.
        assert (
            trivial_report.check_results[CheckId.DETERMINISM_SELF_CONSISTENCY].outcome
            == "passed"
        )

    def test_solver_baseline_failed(self, trivial_report: ValidatorReport) -> None:
        # Random policy clears T=0.5 on TrivialEnv (trivial_random_baseline).
        assert trivial_report.check_results[CheckId.SOLVER_BASELINE].outcome == "failed"

    def test_exactly_two_failures(self, trivial_report: ValidatorReport) -> None:
        assert len(trivial_report.failure_summary) == 2


class TestLiarTupleShapeReport:
    """LiarTupleShape: reset fails; all other checks error out."""

    def test_overall_not_passed(self, liar_tuple_report: ValidatorReport) -> None:
        assert liar_tuple_report.passed is False

    def test_failure_summary_has_all_five_checks(
        self, liar_tuple_report: ValidatorReport
    ) -> None:
        assert len(liar_tuple_report.failure_summary) == 5
        for check_id in CheckId:
            assert check_id.value in liar_tuple_report.failure_summary

    def test_reset_correctness_failed(self, liar_tuple_report: ValidatorReport) -> None:
        assert liar_tuple_report.check_results[CheckId.RESET_CORRECTNESS].outcome == "failed"

    def test_reward_bounds_errored(self, liar_tuple_report: ValidatorReport) -> None:
        outcome = liar_tuple_report.check_results[CheckId.REWARD_BOUNDS]
        assert outcome.outcome == "errored"

    def test_determinism_anchor_errored(self, liar_tuple_report: ValidatorReport) -> None:
        outcome = liar_tuple_report.check_results[CheckId.DETERMINISM_ANCHOR]
        assert outcome.outcome == "errored"

    def test_determinism_self_consistency_errored(
        self, liar_tuple_report: ValidatorReport
    ) -> None:
        outcome = liar_tuple_report.check_results[CheckId.DETERMINISM_SELF_CONSISTENCY]
        assert outcome.outcome == "errored"

    def test_solver_baseline_errored(self, liar_tuple_report: ValidatorReport) -> None:
        outcome = liar_tuple_report.check_results[CheckId.SOLVER_BASELINE]
        assert outcome.outcome == "errored"
