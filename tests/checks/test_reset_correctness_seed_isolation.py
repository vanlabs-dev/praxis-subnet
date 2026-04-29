"""Tests verifying salt-based seed isolation for check_reset_correctness."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing build helpers from scripts/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_gridworld_manifest import build_easy_manifest  # type: ignore[import-not-found]

from praxis.checks._seeds import derive_validator_seeds
from praxis.checks.reset_correctness import check_reset_correctness
from praxis.protocol import RewardBounds, TrajectoryAnchor
from praxis.protocol.types import ActionPolicyId

_RESET_SALT = b"reset_correctness"
_REWARD_SALT = b"reward_bounds"


def test_reset_seeds_disjoint_from_reward_bounds_seeds() -> None:
    """Seeds from reset_correctness (salt=b"reset_correctness") have at most
    1 overlap with reward_bounds seeds (salt=b"reward_bounds").

    Statistically distinct salts over the same manifest should produce
    near-zero overlap in the 64-bit seed space.
    """
    manifest = build_easy_manifest()
    report = check_reset_correctness(manifest)

    reset_seeds = set(report.seeds_tested)
    reward_seeds = set(derive_validator_seeds(manifest, 8, _REWARD_SALT))

    overlap = reset_seeds & reward_seeds
    assert len(overlap) <= 1, (
        f"Unexpected overlap between reset_correctness and reward_bounds seeds: {overlap}"
    )


def test_seeds_invariant_to_declared_reward_bounds() -> None:
    """Modifying declared_reward_bounds only must not change seeds_tested.

    Collusion-resistance: declared_reward_bounds is excluded from the
    seed-derivation hash input.
    """
    manifest_a = build_easy_manifest()
    loose_bounds = RewardBounds(
        min_per_step=-99.0,
        max_per_step=99.0,
        min_per_episode=-9999.0,
        max_per_episode=9999.0,
    )
    manifest_b = manifest_a.model_copy(update={"declared_reward_bounds": loose_bounds})

    report_a = check_reset_correctness(manifest_a)
    report_b = check_reset_correctness(manifest_b)

    assert report_a.seeds_tested == report_b.seeds_tested


def test_seeds_invariant_to_anchor_trajectories() -> None:
    """Modifying anchor_trajectories only must not change seeds_tested.

    Collusion-resistance: anchor_trajectories is excluded from the
    seed-derivation hash input.
    """
    manifest_a = build_easy_manifest()
    different_anchors = [
        TrajectoryAnchor(
            seed=i + 100,
            action_policy=ActionPolicyId.SEEDED_RANDOM,
            n_steps=5,
            expected_hash="f" * 64,
        )
        for i in range(4)
    ]
    manifest_b = manifest_a.model_copy(update={"anchor_trajectories": different_anchors})

    report_a = check_reset_correctness(manifest_a)
    report_b = check_reset_correctness(manifest_b)

    assert report_a.seeds_tested == report_b.seeds_tested
