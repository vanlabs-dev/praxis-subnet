"""Validator pipeline orchestrator.

Runs all five Phase 1 sub-checks against a manifest and returns a
structured ValidatorReport.

Design decisions (locked; see commit message for rationale):

Run-all semantics:
    Every sub-check runs even after earlier failures. The operator gets the
    full failure surface in one report, not just the first failure.

Cheapest-first ordering:
    reset_correctness -> reward_bounds -> determinism_anchor ->
    determinism_self_consistency -> solver_baseline.
    The order is fixed; it cannot be changed by callers.

Conjunctive aggregation:
    overall passed = all sub-checks passed. CheckErrored counts as failure.
    An honest check should not raise on an honest environment.

Failure isolation:
    Each sub-check runs inside a try/except. An unhandled exception from a
    sub-check produces a CheckErrored outcome rather than crashing the
    pipeline. This ensures one buggy check cannot suppress results from the
    remaining checks.

F-040 closure note:
    RT-005 finding F-040 (LOW: "inconsistent strictness across checks
    engineerable into pass-everywhere-fail-cheapest IF orchestration treats
    passed non-conjunctively") is closed by design: conjunctive aggregation
    is built in from day one, so relaxing any single check cannot make the
    aggregate pass.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from praxis.checks.determinism import check_determinism, check_determinism_self_consistency
from praxis.checks.reset_correctness import check_reset_correctness
from praxis.checks.reward_bounds import check_reward_bounds
from praxis.checks.solver_baseline import check_solver_baseline
from praxis.protocol import EnvManifest, hash_payload

from praxis.orchestrator._models import (
    REPORT_FORMAT,
    VALIDATOR_VERSION,
    CheckErrored,
    CheckFailed,
    CheckId,
    CheckOutcome,
    CheckPassed,
    SubCheckReport,
    ValidatorReport,
)

# ---------------------------------------------------------------------------
# Check registry
# ---------------------------------------------------------------------------

# Cheapest-first ordering: the dict preserves insertion order (Python 3.7+).
# Each entry is a Callable that takes an EnvManifest and returns a value
# satisfying the SubCheckReport Protocol (passed: bool + model_dump).
_CHECK_RUNNERS: dict[CheckId, Callable[[EnvManifest], SubCheckReport]] = {
    CheckId.RESET_CORRECTNESS: check_reset_correctness,
    CheckId.REWARD_BOUNDS: check_reward_bounds,
    CheckId.DETERMINISM_ANCHOR: check_determinism,
    CheckId.DETERMINISM_SELF_CONSISTENCY: check_determinism_self_consistency,
    CheckId.SOLVER_BASELINE: check_solver_baseline,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _manifest_hash(manifest: EnvManifest) -> str:
    """Blake2b-256 hex digest of the canonical JSON representation of the manifest.

    Uses hash_payload(manifest.model_dump(mode="json")) which calls
    canonical_bytes internally. The 32-byte digest produces 64 hex chars.

    The hash covers all manifest fields. generated_at_utc is not in the
    manifest so it does not affect this hash.
    """
    return hash_payload(manifest.model_dump(mode="json"))


def _run_one(
    check_id: CheckId,
    runner: Callable[[EnvManifest], SubCheckReport],
    manifest: EnvManifest,
) -> CheckOutcome:
    """Run a single sub-check and return the appropriate CheckOutcome variant.

    Failure isolation: any exception from the runner is caught and returned
    as CheckErrored. The exception is NOT re-raised; run_validator continues
    with the remaining checks.

    Parameters
    ----------
    check_id:
        The CheckId for this runner. Used in error messages only.
    runner:
        A callable satisfying Callable[[EnvManifest], SubCheckReport].
    manifest:
        The manifest to validate.

    Returns
    -------
    CheckOutcome
        CheckPassed, CheckFailed, or CheckErrored.
    """
    try:
        result = runner(manifest)
        report_dict = result.model_dump(mode="json")
        if result.passed:
            return CheckPassed(outcome="passed", report=report_dict)
        return CheckFailed(outcome="failed", report=report_dict)
    except Exception as exc:
        return CheckErrored(
            outcome="errored",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_validator(manifest: EnvManifest) -> ValidatorReport:
    """Run all five Phase 1 sub-checks against the manifest.

    Executes checks in cheapest-first order (reset_correctness, reward_bounds,
    determinism_anchor, determinism_self_consistency, solver_baseline). All
    checks run regardless of earlier failures (run-all semantics).

    Aggregation is conjunctive: the report's `passed` field is True iff every
    sub-check outcome is CheckPassed. CheckErrored counts as failure.

    Parameters
    ----------
    manifest:
        A validated EnvManifest describing the environment to check.

    Returns
    -------
    ValidatorReport
        Three-layer structured report. The only non-deterministic field is
        generated_at_utc (wall-clock UTC). All other fields are fully
        determined by the manifest content and check results.
    """
    generated_at_utc = datetime.now(tz=timezone.utc).isoformat()
    manifest_hash = _manifest_hash(manifest)

    check_results: dict[CheckId, CheckOutcome] = {}
    for check_id, runner in _CHECK_RUNNERS.items():
        check_results[check_id] = _run_one(check_id, runner, manifest)

    # Conjunctive aggregation: all outcomes must be CheckPassed.
    passed = all(outcome.outcome == "passed" for outcome in check_results.values())

    failure_summary = [
        check_id.value
        for check_id, outcome in check_results.items()
        if outcome.outcome != "passed"
    ]

    return ValidatorReport(
        report_format=REPORT_FORMAT,
        validator_version=VALIDATOR_VERSION,
        generated_at_utc=generated_at_utc,
        manifest_hash=manifest_hash,
        env_id=manifest.env_id,
        env_version=manifest.env_version,
        passed=passed,
        check_results=check_results,
        failure_summary=failure_summary,
    )
