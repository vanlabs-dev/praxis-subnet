"""Tests for ValidatorReport serialization and schema correctness.

Verifies:
- model_dump(mode="json") produces a valid dict with all expected fields.
- model_dump_json() round-trips through model_validate_json().
- Schema version fields are stable.
- check_results keys serialize as str values (not enum repr).
- CheckOutcome discriminated union deserializes correctly for each variant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from build_gridworld_manifest import build_easy_manifest  # type: ignore[import]

from praxis.orchestrator import (
    CheckErrored,
    CheckFailed,
    CheckId,
    CheckPassed,
    ValidatorReport,
    run_validator,
)
from praxis.orchestrator._models import REPORT_FORMAT, VALIDATOR_VERSION


@pytest.fixture(scope="module")
def easy_report() -> ValidatorReport:
    return run_validator(build_easy_manifest())


@pytest.fixture(scope="module")
def easy_report_dict(easy_report: ValidatorReport) -> dict:  # type: ignore[type-arg]
    return easy_report.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Layer 1 fields
# ---------------------------------------------------------------------------


def test_report_format_value(easy_report_dict: dict) -> None:  # type: ignore[type-arg]
    assert easy_report_dict["report_format"] == REPORT_FORMAT


def test_validator_version_value(easy_report_dict: dict) -> None:  # type: ignore[type-arg]
    assert easy_report_dict["validator_version"] == VALIDATOR_VERSION


def test_generated_at_utc_present_and_nonempty(easy_report_dict: dict) -> None:  # type: ignore[type-arg]
    ts = easy_report_dict["generated_at_utc"]
    assert isinstance(ts, str) and len(ts) > 0


# ---------------------------------------------------------------------------
# Layer 2 fields
# ---------------------------------------------------------------------------


def test_manifest_hash_64_hex_chars(easy_report_dict: dict) -> None:  # type: ignore[type-arg]
    h = easy_report_dict["manifest_hash"]
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_env_id_present(easy_report_dict: dict) -> None:  # type: ignore[type-arg]
    assert easy_report_dict["env_id"] == "praxis-gridworld-easy"


def test_env_version_present(easy_report_dict: dict) -> None:  # type: ignore[type-arg]
    assert easy_report_dict["env_version"] == "0.1.0"


# ---------------------------------------------------------------------------
# Layer 3 fields
# ---------------------------------------------------------------------------


def test_passed_field_present(easy_report_dict: dict) -> None:  # type: ignore[type-arg]
    assert isinstance(easy_report_dict["passed"], bool)


def test_check_results_keys_are_strings(easy_report_dict: dict) -> None:  # type: ignore[type-arg]
    """check_results keys must serialize as plain strings, not enum repr."""
    keys = list(easy_report_dict["check_results"].keys())
    assert all(isinstance(k, str) for k in keys)
    # Must include all five check IDs by value.
    assert set(keys) == {cid.value for cid in CheckId}


def test_check_results_each_has_outcome_field(easy_report_dict: dict) -> None:  # type: ignore[type-arg]
    for key, outcome_dict in easy_report_dict["check_results"].items():
        assert "outcome" in outcome_dict, f"{key} missing 'outcome' field"


def test_failure_summary_is_list(easy_report_dict: dict) -> None:  # type: ignore[type-arg]
    assert isinstance(easy_report_dict["failure_summary"], list)


# ---------------------------------------------------------------------------
# Round-trip serialization
# ---------------------------------------------------------------------------


def test_round_trip_json(easy_report: ValidatorReport) -> None:
    """model_dump_json -> model_validate_json must produce an equal report."""
    json_str = easy_report.model_dump_json()
    restored = ValidatorReport.model_validate_json(json_str)
    assert restored.model_dump(mode="json") == easy_report.model_dump(mode="json")


def test_round_trip_dict(easy_report: ValidatorReport) -> None:
    """model_dump -> model_validate must produce an equal report."""
    d = easy_report.model_dump(mode="json")
    restored = ValidatorReport.model_validate(d)
    assert restored.model_dump(mode="json") == d


# ---------------------------------------------------------------------------
# Discriminated union deserialization
# ---------------------------------------------------------------------------


def test_check_passed_deserializes() -> None:
    cp = CheckPassed(outcome="passed", report={"passed": True})
    d = cp.model_dump(mode="json")
    restored = CheckPassed.model_validate(d)
    assert restored.outcome == "passed"
    assert restored.report["passed"] is True


def test_check_failed_deserializes() -> None:
    cf = CheckFailed(outcome="failed", report={"passed": False, "violations": []})
    d = cf.model_dump(mode="json")
    restored = CheckFailed.model_validate(d)
    assert restored.outcome == "failed"


def test_check_errored_deserializes() -> None:
    ce = CheckErrored(
        outcome="errored",
        error_type="ValueError",
        error_message="not enough values to unpack",
    )
    d = ce.model_dump(mode="json")
    restored = CheckErrored.model_validate(d)
    assert restored.outcome == "errored"
    assert restored.error_type == "ValueError"
    assert "not enough values" in restored.error_message
