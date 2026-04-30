"""Tests for conjunctive aggregation logic in run_validator.

Uses monkeypatch to swap _CHECK_RUNNERS with lightweight mock runners so
aggregation behavior can be tested without running real sub-checks.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from build_gridworld_manifest import build_easy_manifest  # type: ignore[import]

from praxis.orchestrator import CheckId, run_validator
import praxis.orchestrator.runner as _runner_module
from praxis.protocol import EnvManifest

_DUMMY_MANIFEST = build_easy_manifest()


# ---------------------------------------------------------------------------
# Mock sub-check report helpers
# ---------------------------------------------------------------------------


@dataclass
class _MockSubReport:
    passed: bool
    failure_reason: str | None = None

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return {"passed": self.passed, "failure_reason": self.failure_reason}


def _mock_passing_runner(manifest: EnvManifest) -> _MockSubReport:
    return _MockSubReport(passed=True)


def _mock_failing_runner(manifest: EnvManifest) -> _MockSubReport:
    return _MockSubReport(passed=False, failure_reason="mock_failure")


def _mock_erroring_runner(manifest: EnvManifest) -> _MockSubReport:
    raise RuntimeError("mock error")


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------


def test_all_pass_yields_overall_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_runners = {cid: _mock_passing_runner for cid in CheckId}  # type: ignore[dict-item]
    monkeypatch.setattr(_runner_module, "_CHECK_RUNNERS", mock_runners)

    report = run_validator(_DUMMY_MANIFEST)

    assert report.passed is True
    assert report.failure_summary == []
    for outcome in report.check_results.values():
        assert outcome.outcome == "passed"


def test_one_fail_yields_overall_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_runners = {  # type: ignore[dict-item]
        CheckId.RESET_CORRECTNESS: _mock_failing_runner,
        CheckId.REWARD_BOUNDS: _mock_passing_runner,
        CheckId.DETERMINISM_ANCHOR: _mock_passing_runner,
        CheckId.DETERMINISM_SELF_CONSISTENCY: _mock_passing_runner,
        CheckId.SOLVER_BASELINE: _mock_passing_runner,
    }
    monkeypatch.setattr(_runner_module, "_CHECK_RUNNERS", mock_runners)

    report = run_validator(_DUMMY_MANIFEST)

    assert report.passed is False
    assert CheckId.RESET_CORRECTNESS.value in report.failure_summary
    assert len(report.failure_summary) == 1


def test_one_error_yields_overall_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_runners = {  # type: ignore[dict-item]
        CheckId.RESET_CORRECTNESS: _mock_erroring_runner,
        CheckId.REWARD_BOUNDS: _mock_passing_runner,
        CheckId.DETERMINISM_ANCHOR: _mock_passing_runner,
        CheckId.DETERMINISM_SELF_CONSISTENCY: _mock_passing_runner,
        CheckId.SOLVER_BASELINE: _mock_passing_runner,
    }
    monkeypatch.setattr(_runner_module, "_CHECK_RUNNERS", mock_runners)

    report = run_validator(_DUMMY_MANIFEST)

    assert report.passed is False
    assert CheckId.RESET_CORRECTNESS.value in report.failure_summary


def test_all_fail_yields_all_in_failure_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_runners = {cid: _mock_failing_runner for cid in CheckId}  # type: ignore[dict-item]
    monkeypatch.setattr(_runner_module, "_CHECK_RUNNERS", mock_runners)

    report = run_validator(_DUMMY_MANIFEST)

    assert report.passed is False
    assert len(report.failure_summary) == len(CheckId)
    for cid in CheckId:
        assert cid.value in report.failure_summary


def test_mix_fail_and_error_yields_both_in_failure_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_runners = {  # type: ignore[dict-item]
        CheckId.RESET_CORRECTNESS: _mock_failing_runner,
        CheckId.REWARD_BOUNDS: _mock_erroring_runner,
        CheckId.DETERMINISM_ANCHOR: _mock_passing_runner,
        CheckId.DETERMINISM_SELF_CONSISTENCY: _mock_passing_runner,
        CheckId.SOLVER_BASELINE: _mock_passing_runner,
    }
    monkeypatch.setattr(_runner_module, "_CHECK_RUNNERS", mock_runners)

    report = run_validator(_DUMMY_MANIFEST)

    assert report.passed is False
    assert CheckId.RESET_CORRECTNESS.value in report.failure_summary
    assert CheckId.REWARD_BOUNDS.value in report.failure_summary
    assert len(report.failure_summary) == 2


def test_failure_summary_preserves_insertion_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """failure_summary must list check IDs in cheapest-first (insertion) order."""
    mock_runners = {cid: _mock_failing_runner for cid in CheckId}  # type: ignore[dict-item]
    monkeypatch.setattr(_runner_module, "_CHECK_RUNNERS", mock_runners)

    report = run_validator(_DUMMY_MANIFEST)

    expected_order = [cid.value for cid in CheckId]
    assert report.failure_summary == expected_order
