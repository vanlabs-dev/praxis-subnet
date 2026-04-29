"""F-023 closure tests.

RT-004 finding F-023 (CRITICAL): manifest.reference_solver was creator-
declared, letting Phase 2 attackers stay on the weakest solver they over-
fit against. Closure: validator iterates SOLVER_REGISTRY and runs every
applicable solver, conjunctively aggregating results.

These tests exercise the F-023 contract:

1. test_validator_ignores_manifest_reference_solver: validator selects
   solvers from registry, not from manifest declaration.
2. test_no_compatible_solver_fails: empty registry yields
   passed=False with failure_reason='no_compatible_solver'.
3. test_conjunctive_aggregation_with_mock_solvers: when the registry has
   multiple applicable solvers, a single failing solver flips the
   aggregate to fail. This test exercises Phase 2's multi-solver
   semantics today; it is the F-023 closure invariant.

Do not delete test 3 even though Phase 1 only ships one solver -- it is
the regression guard for the contract Phase 2 will rely on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from build_gridworld_manifest import build_easy_manifest

import praxis.checks.solver_baseline as solver_baseline_module
from praxis.checks.solver_baseline import (
    BandConfig,
    SolverBaselineConfig,
    check_solver_baseline,
)
from praxis.protocol import DifficultyBand
from praxis.protocol.types import SolverId
from praxis.solver._protocol import EvalResult


# ---------------------------------------------------------------------------
# Mock solvers implementing the Solver protocol (duck-typed).
#
# Gridworld easy reward bounds (from build_gridworld_manifest._reward_bounds(5)):
#   min_per_episode = -0.01 * 4*5*5 = -1.0
#   max_per_episode = 1.0 - 0.01 * 2*(5-1) = 0.92
#   span = 0.92 - (-1.0) = 1.92
#
# Threshold = 0.5 (from BandConfig in test 3).
# Random baseline on easy ~= 0.460 < 0.5, so random_fail=False.
#
# Passing mock: raw=0.85 -> norm=(0.85+1.0)/1.92 = 1.85/1.92 ~= 0.964 > 0.5 (pass)
# Failing mock: raw=-0.5 -> norm=(-0.5+1.0)/1.92 = 0.5/1.92 ~= 0.260 < 0.5 (fail)
# ---------------------------------------------------------------------------


class _MockPassingSolver:
    """Canned high-return solver; always passes at threshold=0.5."""

    def train(self, env: gym.Env[Any, Any], seed: int, budget: int) -> None:
        return None

    def evaluate(
        self, env: gym.Env[Any, Any], state: Any, seed: int, n_episodes: int
    ) -> EvalResult:
        per_ep = tuple(0.85 for _ in range(n_episodes))
        return EvalResult(
            mean_episodic_return=0.85,
            per_episode_returns=per_ep,
            terminated_count=n_episodes,
            truncated_count=0,
        )


class _MockFailingSolver:
    """Canned low-return solver; always fails at threshold=0.5."""

    def train(self, env: gym.Env[Any, Any], seed: int, budget: int) -> None:
        return None

    def evaluate(
        self, env: gym.Env[Any, Any], state: Any, seed: int, n_episodes: int
    ) -> EvalResult:
        per_ep = tuple(-0.5 for _ in range(n_episodes))
        return EvalResult(
            mean_episodic_return=-0.5,
            per_episode_returns=per_ep,
            terminated_count=n_episodes,
            truncated_count=0,
        )


# ---------------------------------------------------------------------------
# Helper: inject a temporary SolverId member for multi-solver tests.
#
# SolverId is a StrEnum. Pydantic v2 validates SolverId fields by calling
# SolverId(value) which hits _value2member_map_. We inject a pseudo-member
# via direct attribute assignment (Python allows this on StrEnum instances
# since they have no __slots__) and register it in the enum's lookup dicts.
# The caller must call _remove_temp_solver_id() in a finally block.
#
# This mutation is process-local and safe within a single test; pytest's
# per-test isolation prevents cross-test leakage provided cleanup runs.
# ---------------------------------------------------------------------------

_MOCK_FAILING_VALUE = "mock_failing_test_only"
_MOCK_FAILING_NAME = "MOCK_FAILING_TEST_ONLY"


def _add_temp_solver_id() -> SolverId:
    """Inject a temporary MOCK_FAILING_TEST_ONLY member into SolverId.

    Returns the new pseudo-member. Caller must call _remove_temp_solver_id()
    to restore enum state after the test.
    """
    member: SolverId = str.__new__(SolverId, _MOCK_FAILING_VALUE)  # type: ignore[arg-type]
    member._name_ = _MOCK_FAILING_NAME  # type: ignore[attr-defined]
    member._value_ = _MOCK_FAILING_VALUE  # type: ignore[attr-defined]
    SolverId._value2member_map_[_MOCK_FAILING_VALUE] = member  # type: ignore[attr-defined]
    SolverId._member_map_[_MOCK_FAILING_NAME] = member  # type: ignore[attr-defined]
    return member


def _remove_temp_solver_id() -> None:
    """Remove the MOCK_FAILING_TEST_ONLY member added by _add_temp_solver_id."""
    SolverId._value2member_map_.pop(_MOCK_FAILING_VALUE, None)  # type: ignore[attr-defined]
    SolverId._member_map_.pop(_MOCK_FAILING_NAME, None)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Test 1: validator ignores manifest.reference_solver
# ---------------------------------------------------------------------------


def test_validator_ignores_manifest_reference_solver() -> None:
    """Validator selects solvers from SOLVER_REGISTRY, not from the manifest field.

    This test passes trivially in Phase 1 because only one solver exists in the
    registry. It documents the contract: the manifest's reference_solver field
    has no effect on which solvers run. A creator cannot steer validation to a
    weaker solver by manipulating reference_solver.
    """
    manifest = build_easy_manifest()
    # Use full default config (10_000 training steps, threshold=0.5) which is
    # empirically calibrated to pass on gridworld easy.
    report = check_solver_baseline(manifest)

    # In Phase 1 the registry has exactly one solver: TABULAR_Q_LEARNING.
    # The manifest declares reference_solver=TABULAR_Q_LEARNING, but even if it
    # declared something else, the validator would still run TABULAR_Q_LEARNING
    # because it is in SOLVER_REGISTRY.
    assert SolverId.TABULAR_Q_LEARNING in report.solver_results
    assert len(report.solver_results) == 1
    assert report.passed is True


# ---------------------------------------------------------------------------
# Test 2: empty registry yields no_compatible_solver failure
# ---------------------------------------------------------------------------


def test_no_compatible_solver_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero applicable solvers -> passed=False, failure_reason='no_compatible_solver'.

    Monkeypatching the SOLVER_REGISTRY name in solver_baseline's module namespace
    simulates a registry with no solver applicable to this env. Note: we patch
    praxis.checks.solver_baseline.SOLVER_REGISTRY because solver_baseline.py uses
    'from praxis.solver.registry import SOLVER_REGISTRY' (binds the name locally).
    """
    manifest = build_easy_manifest()
    cfg = SolverBaselineConfig(
        band_configs={
            DifficultyBand.EASY: BandConfig(
                training_budget=200, eval_episodes=5, threshold_normalized=0.5
            ),
        }
    )

    empty_registry: dict[SolverId, Any] = {}
    monkeypatch.setattr(solver_baseline_module, "SOLVER_REGISTRY", empty_registry)

    report = check_solver_baseline(manifest, cfg)

    assert report.passed is False
    assert report.failure_reason == "no_compatible_solver"
    assert len(report.solver_results) == 0


# ---------------------------------------------------------------------------
# Test 3: conjunctive aggregation -- one failing solver flips aggregate to fail
# ---------------------------------------------------------------------------


def test_conjunctive_aggregation_with_mock_solvers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conjunctive contract: a single failing solver makes the aggregate fail.

    This test exercises Phase 2's multi-solver semantics today using two mock
    solvers injected via monkeypatch. It is the F-023 closure invariant and
    must remain even though Phase 1 ships only one solver.

    We inject a temporary SolverId pseudo-member (MOCK_FAILING_TEST_ONLY) so
    that Pydantic validates PerSolverResult.solver_id for both registry entries.
    The member is removed in a finally block regardless of test outcome.

    Registry insertion order:
      1. SolverId.TABULAR_Q_LEARNING  -> _MockPassingSolver  (norm~=0.964 > 0.5)
      2. SolverId.MOCK_FAILING_TEST_ONLY -> _MockFailingSolver (norm~=0.260 < 0.5)

    Random baseline on gridworld easy ~= 0.460 < threshold 0.5 -> random_fail=False.
    Failing solver's failure_reason = "solver_below_threshold".
    Aggregate expected: passed=False, failure_reason="solver_below_threshold".
    """
    manifest = build_easy_manifest()
    cfg = SolverBaselineConfig(
        band_configs={
            DifficultyBand.EASY: BandConfig(
                training_budget=200, eval_episodes=5, threshold_normalized=0.5
            ),
        }
    )

    failing_id = _add_temp_solver_id()
    try:
        mock_registry: dict[SolverId, Any] = {
            SolverId.TABULAR_Q_LEARNING: _MockPassingSolver(),
            failing_id: _MockFailingSolver(),
        }
        monkeypatch.setattr(solver_baseline_module, "SOLVER_REGISTRY", mock_registry)

        report = check_solver_baseline(manifest, cfg)
    finally:
        _remove_temp_solver_id()

    # Conjunctive contract: one failing solver flips the aggregate.
    assert report.passed is False, (
        f"Expected aggregate fail but got passed=True; "
        f"solver_results={report.solver_results}"
    )
    assert report.failure_reason == "solver_below_threshold"
    assert len(report.solver_results) == 2

    # Passing mock result (TABULAR_Q_LEARNING).
    passing_result = report.solver_results[SolverId.TABULAR_Q_LEARNING]
    assert passing_result.passed is True
    assert passing_result.failure_reason is None
    assert passing_result.normalized_mean_return == pytest.approx(
        (0.85 - (-1.0)) / 1.92, abs=1e-6
    )

    # Failing mock result (MOCK_FAILING_TEST_ONLY).
    # We use the string value for lookup since the pseudo-member may not
    # survive serialization; the dict key identity is the same object.
    failing_result_key = next(
        k for k in report.solver_results if str(k) == _MOCK_FAILING_VALUE
    )
    failing_result = report.solver_results[failing_result_key]
    assert failing_result.passed is False
    assert failing_result.failure_reason == "solver_below_threshold"
    assert failing_result.normalized_mean_return == pytest.approx(
        (-0.5 - (-1.0)) / 1.92, abs=1e-6
    )
