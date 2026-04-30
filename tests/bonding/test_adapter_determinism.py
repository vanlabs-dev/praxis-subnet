"""Adapter determinism: two run_validator calls on the same manifest produce
the same justification_hash.

Uses a module-scoped fixture so run_validator runs twice (two separate calls
on the same manifest), and the resulting justification_hash values are compared.
This is the consumer-view determinism test; the orchestrator's own
test_runner_determinism.py tests byte-identical report bodies (modulo
generated_at_utc). Here we verify that justification_hash_for_report is
stable across independent runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "tests"))

from checks._adversarial_envs import make_adversarial_manifest  # type: ignore[import]

from praxis.bonding import justification_hash_for_report
from praxis.orchestrator import ValidatorReport, run_validator


@pytest.fixture(scope="module")
def two_failing_reports() -> tuple[ValidatorReport, ValidatorReport]:
    """Run run_validator twice on TrivialEnv. Both should have passed=False."""
    manifest = make_adversarial_manifest("trivial-env", "TrivialEnv")
    report_a = run_validator(manifest)
    report_b = run_validator(manifest)
    return report_a, report_b


class TestAdapterDeterminism:
    def test_both_reports_failed(
        self, two_failing_reports: tuple[ValidatorReport, ValidatorReport]
    ) -> None:
        a, b = two_failing_reports
        assert a.passed is False
        assert b.passed is False

    def test_justification_hash_identical_across_runs(
        self, two_failing_reports: tuple[ValidatorReport, ValidatorReport]
    ) -> None:
        """Two independent run_validator calls must produce the same hash."""
        a, b = two_failing_reports
        hash_a = justification_hash_for_report(a)
        hash_b = justification_hash_for_report(b)
        assert hash_a == hash_b, (
            f"justification_hash diverged across runs:\n  run_a: {hash_a}\n  run_b: {hash_b}\n"
            "This indicates a determinism regression in the orchestrator."
        )

    def test_hash_is_64_hex(
        self, two_failing_reports: tuple[ValidatorReport, ValidatorReport]
    ) -> None:
        a, _ = two_failing_reports
        h = justification_hash_for_report(a)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_manifest_hashes_identical(
        self, two_failing_reports: tuple[ValidatorReport, ValidatorReport]
    ) -> None:
        """Sanity: same manifest -> same manifest_hash in both reports."""
        a, b = two_failing_reports
        assert a.manifest_hash == b.manifest_hash

    def test_generated_at_utc_may_differ(
        self, two_failing_reports: tuple[ValidatorReport, ValidatorReport]
    ) -> None:
        """generated_at_utc is the ONLY field allowed to differ.

        This test is informational (we cannot guarantee they differ on fast
        hardware), but it documents the contract: justification_hash is stable
        BECAUSE we exclude generated_at_utc.
        """
        a, b = two_failing_reports
        # Hashes are equal even though generated_at_utc may differ.
        # The assertion is that the hash function does not depend on it.
        hash_a = justification_hash_for_report(a)
        hash_b = justification_hash_for_report(b)
        assert hash_a == hash_b  # redundant; makes the intent explicit
