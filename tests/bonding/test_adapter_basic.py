"""Basic adapter tests: justification_hash_for_report and slash_for_report.

Uses module-scoped fixtures to avoid running run_validator more than once.
TrivialEnv produces passed=False (determinism_anchor + solver_baseline fail).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "tests"))

from checks._adversarial_envs import make_adversarial_manifest  # type: ignore[import]

from praxis.bonding import (
    InMemoryBondLedger,
    InsufficientBalanceError,
    SlashEvent,
    UnknownMinerError,
    justification_hash_for_report,
    slash_for_report,
)
from praxis.orchestrator import ValidatorReport, run_validator

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def failing_report() -> ValidatorReport:
    """TrivialEnv produces passed=False (determinism_anchor + solver_baseline fail)."""
    manifest = make_adversarial_manifest("trivial-env", "TrivialEnv")
    return run_validator(manifest)


# ---------------------------------------------------------------------------
# justification_hash_for_report
# ---------------------------------------------------------------------------


class TestJustificationHash:
    def test_returns_64_hex_lowercase(self, failing_report: ValidatorReport) -> None:
        h = justification_hash_for_report(failing_report)
        assert _HEX64_RE.match(h), f"Expected 64 lowercase hex, got: {h!r}"

    def test_same_report_same_hash(self, failing_report: ValidatorReport) -> None:
        h1 = justification_hash_for_report(failing_report)
        h2 = justification_hash_for_report(failing_report)
        assert h1 == h2

    def test_hash_excludes_generated_at_utc(self, failing_report: ValidatorReport) -> None:
        """Mutating generated_at_utc on a copy must not change the hash."""
        # model_dump returns a fresh dict -- safe to modify without affecting source.
        payload = failing_report.model_dump(mode="json")
        payload["generated_at_utc"] = "1970-01-01T00:00:00Z"
        from praxis.protocol.hashing import hash_payload

        hash_with_dummy_ts = hash_payload({k: v for k, v in payload.items()
                                           if k != "generated_at_utc"})
        assert justification_hash_for_report(failing_report) == hash_with_dummy_ts

    def test_hash_length_is_64(self, failing_report: ValidatorReport) -> None:
        assert len(justification_hash_for_report(failing_report)) == 64


# ---------------------------------------------------------------------------
# slash_for_report -- guard: refuses passed=True
# ---------------------------------------------------------------------------


class TestSlashForReportRefusePassed:
    def test_raises_value_error_on_passed_report(self) -> None:
        """slash_for_report must raise ValueError if report.passed is True."""
        # Build a minimal passing report directly without running the orchestrator.
        from praxis.orchestrator._models import REPORT_FORMAT, VALIDATOR_VERSION, CheckId
        from praxis.orchestrator import CheckPassed

        report = ValidatorReport(
            report_format=REPORT_FORMAT,
            validator_version=VALIDATOR_VERSION,
            generated_at_utc="2026-01-01T00:00:00Z",
            manifest_hash="0" * 64,
            env_id="dummy",
            env_version="0.1.0",
            passed=True,
            check_results={
                cid: CheckPassed(outcome="passed", report={"passed": True})
                for cid in CheckId
            },
            failure_summary=[],
        )
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 1000)

        with pytest.raises(ValueError, match="refusing to slash on a passed report"):
            slash_for_report(ledger, "miner_a", 100, report, "https://example.com")

    def test_error_message_includes_manifest_hash(self) -> None:
        from praxis.orchestrator._models import REPORT_FORMAT, VALIDATOR_VERSION, CheckId
        from praxis.orchestrator import CheckPassed

        manifest_hash = "f" * 64
        report = ValidatorReport(
            report_format=REPORT_FORMAT,
            validator_version=VALIDATOR_VERSION,
            generated_at_utc="2026-01-01T00:00:00Z",
            manifest_hash=manifest_hash,
            env_id="dummy",
            env_version="0.1.0",
            passed=True,
            check_results={
                cid: CheckPassed(outcome="passed", report={"passed": True})
                for cid in CheckId
            },
            failure_summary=[],
        )
        ledger = InMemoryBondLedger()
        with pytest.raises(ValueError, match=manifest_hash):
            slash_for_report(ledger, "miner_a", 100, report, "https://example.com")


# ---------------------------------------------------------------------------
# slash_for_report -- full slash flow on a failing report
# ---------------------------------------------------------------------------


class TestSlashForReportFlow:
    def test_returns_slash_event(self, failing_report: ValidatorReport) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 1000)
        event = slash_for_report(ledger, "miner_a", 100, failing_report, "https://example.com")
        assert isinstance(event, SlashEvent)

    def test_slash_event_miner_id(self, failing_report: ValidatorReport) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 1000)
        event = slash_for_report(ledger, "miner_a", 200, failing_report, "https://example.com")
        assert event.miner_id == "miner_a"

    def test_slash_event_amount(self, failing_report: ValidatorReport) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 1000)
        event = slash_for_report(ledger, "miner_a", 300, failing_report, "https://example.com")
        assert event.amount_slashed == 300

    def test_slash_event_hash_is_64_hex(self, failing_report: ValidatorReport) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 1000)
        event = slash_for_report(ledger, "miner_a", 50, failing_report, "https://example.com")
        assert _HEX64_RE.match(event.justification_hash)

    def test_slash_event_url_matches_input(self, failing_report: ValidatorReport) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 1000)
        url = "https://reports.example.com/slash/42"
        event = slash_for_report(ledger, "miner_a", 50, failing_report, url)
        assert event.justification_url == url

    def test_slash_deducts_balance(self, failing_report: ValidatorReport) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 1000)
        slash_for_report(ledger, "miner_a", 400, failing_report, "https://example.com")
        assert ledger.get_balance("miner_a") == 600

    def test_slash_on_unknown_miner_raises(self, failing_report: ValidatorReport) -> None:
        ledger = InMemoryBondLedger()
        with pytest.raises(UnknownMinerError):
            slash_for_report(ledger, "ghost", 100, failing_report, "https://example.com")

    def test_slash_exceeding_balance_raises(self, failing_report: ValidatorReport) -> None:
        ledger = InMemoryBondLedger()
        ledger.deposit("miner_a", 50)
        with pytest.raises(InsufficientBalanceError):
            slash_for_report(ledger, "miner_a", 100, failing_report, "https://example.com")
