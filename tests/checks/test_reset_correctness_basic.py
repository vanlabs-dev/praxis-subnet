"""Basic integration tests for check_reset_correctness against PraxisGridworld."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing build helpers from scripts/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_gridworld_manifest import (  # type: ignore[import-not-found]
    build_easy_manifest,
    build_hard_manifest,
    build_medium_manifest,
)

from praxis.checks._seeds import derive_validator_seeds
from praxis.checks.reset_correctness import ResetReport, check_reset_correctness


def test_easy_manifest_passes() -> None:
    """Easy (5x5) gridworld: check passes with 8 seeds, zero violations."""
    manifest = build_easy_manifest()
    report: ResetReport = check_reset_correctness(manifest)

    assert report.passed is True
    assert len(report.violations) == 0
    assert len(report.seeds_tested) == 8


def test_medium_manifest_passes() -> None:
    """Medium (10x10) gridworld: check passes with 8 seeds, zero violations."""
    manifest = build_medium_manifest()
    report: ResetReport = check_reset_correctness(manifest)

    assert report.passed is True
    assert len(report.violations) == 0
    assert len(report.seeds_tested) == 8


def test_hard_manifest_passes() -> None:
    """Hard (20x20) gridworld: check passes with 8 seeds, zero violations."""
    manifest = build_hard_manifest()
    report: ResetReport = check_reset_correctness(manifest)

    assert report.passed is True
    assert len(report.violations) == 0
    assert len(report.seeds_tested) == 8


def test_reset_seeds_disjoint_from_reward_bounds_seeds() -> None:
    """Reset seeds (salt=b"reset_correctness") are disjoint from reward_bounds seeds.

    Statistically the two 8-seed sets should have near-zero overlap; we allow
    at most 1 coincidental collision to avoid a brittle test.
    """
    manifest = build_easy_manifest()
    report = check_reset_correctness(manifest)

    reset_seeds = set(report.seeds_tested)
    reward_bounds_seeds = set(derive_validator_seeds(manifest, 8, b"reward_bounds"))

    overlap = reset_seeds & reward_bounds_seeds
    assert len(overlap) <= 1, (
        f"Unexpected seed overlap between reset_correctness and reward_bounds: {overlap}"
    )
