"""Basic integration tests for check_determinism against PraxisGridworld-Easy."""

from __future__ import annotations

import praxis.envs  # noqa: F401 -- ensure env registrations are loaded

from praxis.checks.determinism import (
    RolloutResult,
    check_determinism,
    rollout,
)
from praxis.protocol import (
    ActionPolicyId,
    DifficultyBand,
    EnvManifest,
    RewardBounds,
    TrajectoryAnchor,
)

_ENV_ID = "praxisgridworld-easy-v0"
_N_STEPS = 50
_SEEDS = [1, 2, 3, 4]

# Realistic reward bounds for the 5x5 gridworld:
#   per-step: [-0.01, 0.99]  (step penalty + optional goal bonus)
#   per-episode: [-0.01 * 100, 0.99]  (100 = 4 * 5^2 max steps)
_BOUNDS = RewardBounds(
    min_per_step=-0.01,
    max_per_step=0.99,
    min_per_episode=-1.0,
    max_per_episode=0.99,
)


def _compute_anchors(seeds: list[int], corrupt_index: int | None = None) -> list[TrajectoryAnchor]:
    """Build TrajectoryAnchor list by computing real hashes, optionally corrupting one."""
    anchors: list[TrajectoryAnchor] = []
    for i, seed in enumerate(seeds):
        result: RolloutResult = rollout(
            env_id=_ENV_ID,
            seed=seed,
            action_policy=ActionPolicyId.SEEDED_RANDOM,
            n_steps=_N_STEPS,
        )
        if corrupt_index is not None and i == corrupt_index:
            declared = "0" * 64
        else:
            declared = result.computed_hash
        anchors.append(
            TrajectoryAnchor(
                seed=seed,
                action_policy=ActionPolicyId.SEEDED_RANDOM,
                n_steps=_N_STEPS,
                expected_hash=declared,
            )
        )
    return anchors


def _build_manifest(anchors: list[TrajectoryAnchor]) -> EnvManifest:
    return EnvManifest(
        protocol_version="0.1.0",
        env_id=_ENV_ID,
        entry_point="praxis.envs.gridworld:PraxisGridworld",
        difficulty_band=DifficultyBand.EASY,
        max_episode_steps=100,
        declared_reward_bounds=_BOUNDS,
        anchor_trajectories=anchors,
    )


def test_all_anchors_pass_with_correct_hashes() -> None:
    """All correct hashes produce passed=True and all anchors matched."""
    anchors = _compute_anchors(_SEEDS)
    manifest = _build_manifest(anchors)
    report = check_determinism(manifest)

    assert report.passed is True
    assert report.anchor_count == len(_SEEDS)
    assert report.matched_count == len(_SEEDS)
    for anchor_result in report.anchors:
        assert anchor_result.matched is True
        assert anchor_result.declared_hash == anchor_result.computed_hash


def test_one_corrupted_anchor_fails() -> None:
    """One corrupted hash causes passed=False with exactly one mismatch."""
    corrupt_index = 2  # corrupt the third anchor
    anchors = _compute_anchors(_SEEDS, corrupt_index=corrupt_index)
    manifest = _build_manifest(anchors)
    report = check_determinism(manifest)

    assert report.passed is False
    assert report.matched_count == len(_SEEDS) - 1

    mismatched = [a for a in report.anchors if not a.matched]
    assert len(mismatched) == 1
    assert mismatched[0].seed == _SEEDS[corrupt_index]
    assert mismatched[0].declared_hash == "0" * 64

    # All other anchors still matched.
    matched = [a for a in report.anchors if a.matched]
    assert len(matched) == len(_SEEDS) - 1


def test_all_corrupted_anchors_fail() -> None:
    """All corrupted hashes cause passed=False with matched_count=0."""
    # Build anchors with all declared hashes set to zeros.
    real_anchors = _compute_anchors(_SEEDS)
    corrupt_anchors = [
        TrajectoryAnchor(
            seed=anchor.seed,
            action_policy=anchor.action_policy,
            n_steps=anchor.n_steps,
            expected_hash="0" * 64,
        )
        for anchor in real_anchors
    ]
    manifest = _build_manifest(corrupt_anchors)
    report = check_determinism(manifest)

    assert report.passed is False
    assert report.matched_count == 0
    assert report.anchor_count == len(_SEEDS)
    for anchor_result in report.anchors:
        assert anchor_result.matched is False
