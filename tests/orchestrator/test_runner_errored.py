"""Tests verifying CheckErrored isolation: one erroring check must not crash others.

Uses monkeypatch to inject a crashing runner into _CHECK_RUNNERS and asserts
that the remaining runners still execute and the overall report is well-formed.
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

_EASY_MANIFEST = build_easy_manifest()


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class _MockPassReport:
    passed: bool = True

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return {"passed": self.passed}


@dataclass
class _MockFailReport:
    passed: bool = False

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return {"passed": self.passed}


def _mock_passing_runner(manifest: EnvManifest) -> _MockPassReport:
    return _MockPassReport(passed=True)


def _mock_failing_runner(manifest: EnvManifest) -> _MockFailReport:
    return _MockFailReport(passed=False)


def _mock_erroring_runner(manifest: EnvManifest) -> _MockPassReport:
    raise RuntimeError("intentional mock error")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_erroring_check_does_not_crash_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception from one runner must not prevent remaining runners from executing."""
    call_log: list[CheckId] = []

    def _logging_pass(manifest: EnvManifest) -> _MockPassReport:
        call_log.append(CheckId.REWARD_BOUNDS)  # reuse enum value; doesn't matter here
        return _MockPassReport()

    # Crash on RESET_CORRECTNESS; all others pass.
    mock_runners = {
        CheckId.RESET_CORRECTNESS: _mock_erroring_runner,  # type: ignore[dict-item]
        CheckId.REWARD_BOUNDS: _mock_passing_runner,  # type: ignore[dict-item]
        CheckId.DETERMINISM_ANCHOR: _mock_passing_runner,  # type: ignore[dict-item]
        CheckId.DETERMINISM_SELF_CONSISTENCY: _mock_passing_runner,  # type: ignore[dict-item]
        CheckId.SOLVER_BASELINE: _mock_passing_runner,  # type: ignore[dict-item]
    }
    monkeypatch.setattr(_runner_module, "_CHECK_RUNNERS", mock_runners)

    report = run_validator(_EASY_MANIFEST)

    # Errored check present and counted as failure.
    assert report.check_results[CheckId.RESET_CORRECTNESS].outcome == "errored"
    # All other checks ran and passed.
    for cid in [
        CheckId.REWARD_BOUNDS,
        CheckId.DETERMINISM_ANCHOR,
        CheckId.DETERMINISM_SELF_CONSISTENCY,
        CheckId.SOLVER_BASELINE,
    ]:
        assert report.check_results[cid].outcome == "passed"
    # Overall: failed (conjunctive).
    assert report.passed is False


def test_errored_outcome_captures_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CheckErrored must record error_type and error_message from the exception."""
    mock_runners = {cid: _mock_erroring_runner for cid in CheckId}  # type: ignore[dict-item]
    monkeypatch.setattr(_runner_module, "_CHECK_RUNNERS", mock_runners)

    report = run_validator(_EASY_MANIFEST)

    for outcome in report.check_results.values():
        assert outcome.outcome == "errored"
        assert outcome.error_type == "RuntimeError"  # type: ignore[union-attr]
        assert "intentional mock error" in outcome.error_message  # type: ignore[union-attr]


def test_all_errored_means_all_five_in_failure_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_runners = {cid: _mock_erroring_runner for cid in CheckId}  # type: ignore[dict-item]
    monkeypatch.setattr(_runner_module, "_CHECK_RUNNERS", mock_runners)

    report = run_validator(_EASY_MANIFEST)
    assert len(report.failure_summary) == len(CheckId)
    for cid in CheckId:
        assert cid.value in report.failure_summary
