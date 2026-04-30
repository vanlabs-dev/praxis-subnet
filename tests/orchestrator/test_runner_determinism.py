"""Tests verifying run_validator pipeline determinism.

Two successive calls with the same manifest must produce byte-identical
reports EXCEPT for generated_at_utc. If any other field differs, there is
a non-determinism bug in the pipeline that must be surfaced, not papered
over.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from build_gridworld_manifest import build_easy_manifest  # type: ignore[import]

from praxis.orchestrator import run_validator


def test_two_runs_identical_except_generated_at_utc() -> None:
    """Two successive run_validator calls must agree on every field except generated_at_utc."""
    manifest = build_easy_manifest()

    r1 = run_validator(manifest).model_dump(mode="json")
    r2 = run_validator(manifest).model_dump(mode="json")

    ts1 = r1.pop("generated_at_utc")
    ts2 = r2.pop("generated_at_utc")

    # generated_at_utc is the ONLY allowed source of divergence.
    assert r1 == r2, (
        "Two run_validator calls on the same manifest produced different reports "
        "(field(s) other than generated_at_utc differ). "
        f"Keys that diverge: {[k for k in r1 if r1.get(k) != r2.get(k)]}"
    )

    # Timestamps are valid ISO-8601 strings; we don't assert they're equal since
    # wall time advances between the two calls.
    assert isinstance(ts1, str) and len(ts1) > 0
    assert isinstance(ts2, str) and len(ts2) > 0


def test_generated_at_utc_is_the_only_non_deterministic_field() -> None:
    """Cross-check: removing generated_at_utc leaves no other non-deterministic fields."""
    manifest = build_easy_manifest()

    r1 = run_validator(manifest).model_dump(mode="json")
    r2 = run_validator(manifest).model_dump(mode="json")

    # Remove only generated_at_utc; assert nothing else varies.
    r1.pop("generated_at_utc")
    r2.pop("generated_at_utc")

    diverging = {k for k in set(r1) | set(r2) if r1.get(k) != r2.get(k)}
    assert not diverging, f"Unexpected non-deterministic fields: {diverging}"
