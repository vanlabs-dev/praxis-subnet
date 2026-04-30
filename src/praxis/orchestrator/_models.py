"""Data models for the validator pipeline orchestrator.

Defines the three-layer attestation-shaped ValidatorReport and its
constituent types: CheckId, CheckOutcome (discriminated union), and
the SubCheckReport Protocol.

Layer 1 (format): report_format, validator_version, generated_at_utc.
Layer 2 (subject): manifest_hash, env_id, env_version.
Layer 3 (verdict): passed, check_results, failure_summary.

CheckId placement note: CheckId lives here (praxis.orchestrator._models)
in Phase 1. If Phase 2 makes the report schema a hard protocol contract,
CheckId will migrate to protocol/types.py so that validators and consumers
share a single definition without importing from the orchestrator.

Non-determinism note: generated_at_utc is the ONLY non-deterministic field
in this schema. It records wall-clock time from the node running the
validator. Phase 2 will replace it with a chain-beacon-derived timestamp so
that two validators evaluating the same manifest at the same chain height
produce byte-identical reports. Until then, generated_at_utc MUST be
excluded from any canonical signing surface or hash comparison.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# CheckId
# ---------------------------------------------------------------------------


class CheckId(StrEnum):
    """Stable identifiers for the five Phase 1 sub-checks.

    Ordering here is documentation only; execution order is defined in
    runner._CHECK_RUNNERS (cheapest-first: reset -> reward_bounds ->
    determinism_anchor -> determinism_self_consistency -> solver_baseline).

    Phase 2 migration note: if this enum is promoted to protocol/types.py,
    update imports in runner.py and all consumers accordingly.
    """

    RESET_CORRECTNESS = "reset_correctness"
    REWARD_BOUNDS = "reward_bounds"
    DETERMINISM_ANCHOR = "determinism_anchor"
    DETERMINISM_SELF_CONSISTENCY = "determinism_self_consistency"
    SOLVER_BASELINE = "solver_baseline"


# ---------------------------------------------------------------------------
# SubCheckReport Protocol
# ---------------------------------------------------------------------------


class SubCheckReport(Protocol):
    """Structural Protocol that every sub-check report satisfies.

    Each of the four Phase 1 check modules returns a Pydantic v2 BaseModel
    whose fields include `passed: bool` and whose class exposes
    `model_dump(*, mode=...)`. This Protocol captures those two members so
    the orchestrator can type-safely access them without a Union type or
    explicit inheritance.

    mypy's structural subtyping matches each Pydantic BaseModel report against
    this Protocol automatically. No `# type: ignore` needed in src/.
    """

    @property
    def passed(self) -> bool: ...

    def model_dump(
        self, *, mode: str | Literal["json", "python"] = "python"
    ) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# CheckOutcome discriminated union
# ---------------------------------------------------------------------------


class CheckPassed(BaseModel):
    """The sub-check ran and returned passed=True."""

    outcome: Literal["passed"] = "passed"
    report: dict[str, Any]


class CheckFailed(BaseModel):
    """The sub-check ran and returned passed=False."""

    outcome: Literal["failed"] = "failed"
    report: dict[str, Any]


class CheckErrored(BaseModel):
    """The sub-check raised an unhandled exception.

    An honest check should not raise on an honest environment. CheckErrored
    is treated as a failure for the purposes of conjunctive aggregation.

    Attributes
    ----------
    error_type:
        The exception class name (e.g. "ValueError", "RuntimeError").
    error_message:
        The str() of the exception.
    """

    outcome: Literal["errored"] = "errored"
    error_type: str
    error_message: str


CheckOutcome = Annotated[
    CheckPassed | CheckFailed | CheckErrored,
    Field(discriminator="outcome"),
]


# ---------------------------------------------------------------------------
# ValidatorReport
# ---------------------------------------------------------------------------

VALIDATOR_VERSION: str = "1.0.0"
REPORT_FORMAT: Literal["praxis-validator-report-v1"] = "praxis-validator-report-v1"


class ValidatorReport(BaseModel):
    """Three-layer attestation-shaped report produced by run_validator().

    Layer 1 -- format metadata (stable across runs):
        report_format: identifies schema version.
        validator_version: semver string of this orchestrator.

    Layer 2 -- subject identification (deterministic given the manifest):
        manifest_hash: blake2b-256 hex digest of canonical_bytes of
            manifest.model_dump(mode="json"). 64 hex chars.
        env_id: from manifest.env_id.
        env_version: from manifest.env_version.

    Layer 3 -- verdict (deterministic given the manifest and check results):
        passed: True iff every check_result has outcome="passed".
            Conjunctive; CheckErrored counts as failure.
        check_results: ordered dict mapping CheckId to CheckOutcome.
            Insertion order matches the execution schedule (cheapest-first).
        failure_summary: human-readable list of failed/errored check names.
            Empty when passed=True.

    Non-deterministic field (OUTSIDE canonical signing surface):
        generated_at_utc: wall-clock ISO-8601 UTC timestamp. Non-deterministic;
            excluded from any signing or hash comparison. Phase 2 will replace
            this with a chain-beacon-derived value so two validators at the
            same height produce byte-identical reports.

    Backward-compatibility notes:
        - Adding new fields with defaults is backward-compatible.
        - Renaming or removing fields requires a report_format version bump.
        - Phase 2 Bundle wrapper will envelope this report; do not add
          outer-envelope fields here.
    """

    # Layer 1
    report_format: Literal["praxis-validator-report-v1"] = REPORT_FORMAT
    validator_version: str = VALIDATOR_VERSION
    generated_at_utc: str

    # Layer 2
    manifest_hash: str
    env_id: str
    env_version: str

    # Layer 3
    passed: bool
    check_results: dict[CheckId, CheckOutcome]
    failure_summary: list[str]
