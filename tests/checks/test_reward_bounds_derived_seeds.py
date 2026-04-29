"""Tests verifying that check_reward_bounds uses manifest-derived seeds."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing build helpers from scripts/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_gridworld_manifest import build_easy_manifest  # type: ignore[import-not-found]

from praxis.checks._seeds import derive_validator_seeds
from praxis.checks.reward_bounds import check_reward_bounds
from praxis.protocol import RewardBounds

_SALT = b"reward_bounds"


def test_default_uses_derived_seeds() -> None:
    """Default config: seeds in report match derive_validator_seeds output."""
    manifest = build_easy_manifest()
    report = check_reward_bounds(manifest)

    assert report.passed is True

    expected_seeds = list(derive_validator_seeds(manifest, 8, _SALT))
    actual_seeds = [s.seed for s in report.samples]
    assert actual_seeds == expected_seeds


def test_seeds_invariant_to_declared_bounds() -> None:
    """Two manifests differing only in declared_reward_bounds produce identical seed lists.

    Collusion-resistance: tweaking declared bounds must NOT shift the seeds
    the validator samples.
    """
    manifest_a = build_easy_manifest()

    loose_bounds = RewardBounds(
        min_per_step=-1.0,
        max_per_step=1.0,
        min_per_episode=-100.0,
        max_per_episode=100.0,
    )
    manifest_b = manifest_a.model_copy(update={"declared_reward_bounds": loose_bounds})

    report_a = check_reward_bounds(manifest_a)
    report_b = check_reward_bounds(manifest_b)

    seeds_a = [s.seed for s in report_a.samples]
    seeds_b = [s.seed for s in report_b.samples]
    assert seeds_a == seeds_b
