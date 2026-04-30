"""Tests verifying run_validator passes on the three gridworld manifests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from build_gridworld_manifest import (  # type: ignore[import]
    build_easy_manifest,
    build_hard_manifest,
    build_medium_manifest,
)

from praxis.orchestrator import CheckId, ValidatorReport, run_validator


@pytest.fixture(scope="module")
def easy_report() -> ValidatorReport:
    return run_validator(build_easy_manifest())


@pytest.fixture(scope="module")
def medium_report() -> ValidatorReport:
    return run_validator(build_medium_manifest())


@pytest.fixture(scope="module")
def hard_report() -> ValidatorReport:
    return run_validator(build_hard_manifest())


class TestEasyPasses:
    def test_passed_true(self, easy_report: ValidatorReport) -> None:
        assert easy_report.passed is True

    def test_failure_summary_empty(self, easy_report: ValidatorReport) -> None:
        assert easy_report.failure_summary == []

    def test_all_checks_passed(self, easy_report: ValidatorReport) -> None:
        for check_id, outcome in easy_report.check_results.items():
            assert outcome.outcome == "passed", f"{check_id} did not pass"

    def test_five_checks_present(self, easy_report: ValidatorReport) -> None:
        assert set(easy_report.check_results.keys()) == set(CheckId)

    def test_env_id(self, easy_report: ValidatorReport) -> None:
        assert easy_report.env_id == "praxis-gridworld-easy"

    def test_manifest_hash_is_64_hex(self, easy_report: ValidatorReport) -> None:
        assert len(easy_report.manifest_hash) == 64
        assert all(c in "0123456789abcdef" for c in easy_report.manifest_hash)


class TestMediumPasses:
    def test_passed_true(self, medium_report: ValidatorReport) -> None:
        assert medium_report.passed is True

    def test_failure_summary_empty(self, medium_report: ValidatorReport) -> None:
        assert medium_report.failure_summary == []

    def test_env_id(self, medium_report: ValidatorReport) -> None:
        assert medium_report.env_id == "praxis-gridworld-medium"


class TestHardPasses:
    def test_passed_true(self, hard_report: ValidatorReport) -> None:
        assert hard_report.passed is True

    def test_failure_summary_empty(self, hard_report: ValidatorReport) -> None:
        assert hard_report.failure_summary == []

    def test_env_id(self, hard_report: ValidatorReport) -> None:
        assert hard_report.env_id == "praxis-gridworld-hard"
