"""Validator pipeline orchestrator for Praxis Phase 1.

Public surface:

    run_validator(manifest: EnvManifest) -> ValidatorReport
        Run all five sub-checks and return a structured report.

    ValidatorReport
        Three-layer attestation-shaped Pydantic model.

    CheckId
        StrEnum of stable sub-check identifiers.

    CheckPassed, CheckFailed, CheckErrored
        The three CheckOutcome variants (discriminated union on "outcome").

    CheckOutcome
        Annotated union type alias for type annotations.

    VALIDATOR_VERSION
        Semver string for this orchestrator release.
"""

from praxis.orchestrator._models import (
    VALIDATOR_VERSION,
    CheckErrored,
    CheckFailed,
    CheckId,
    CheckOutcome,
    CheckPassed,
    SubCheckReport,
    ValidatorReport,
)
from praxis.orchestrator.runner import run_validator

__all__ = [
    "CheckErrored",
    "CheckFailed",
    "CheckId",
    "CheckOutcome",
    "CheckPassed",
    "SubCheckReport",
    "VALIDATOR_VERSION",
    "ValidatorReport",
    "run_validator",
]
