"""Tests for the truncation-only scenario in check_reward_bounds.

Verifies that when every sampled episode truncates (no natural termination),
the report sets per_episode_unverified=True and still passes -- because
no per-step bounds are violated by the gridworld's normal step reward.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing build helpers from scripts/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from praxis.checks.reward_bounds import (
    RewardBoundsConfig,
    RewardBoundsReport,
    check_reward_bounds,
)
from praxis.protocol import DifficultyBand, EnvManifest, RewardBounds, TrajectoryAnchor
from praxis.protocol.types import ActionPolicyId

_ENTRY_POINT = "praxis.envs.gridworld:PraxisGridworld"
_PROTOCOL_VERSION = "0.2.0"
_ENV_VERSION = "0.1.0"

# 20x20 grid, but we cap max_episode_steps=10 so every random walk truncates.
# On a 20x20 grid the Manhattan distance to the goal is 38 steps -- a purely
# random walk will never reach the goal in 10 steps.
_GRID_SIZE = 20
_MAX_EPISODE_STEPS = 10


def _manifest() -> EnvManifest:
    anchors = [
        TrajectoryAnchor(
            seed=i,
            action_policy=ActionPolicyId.SEEDED_RANDOM,
            n_steps=10,
            expected_hash="a" * 64,
        )
        for i in range(1, 5)
    ]
    return EnvManifest(
        protocol_version=_PROTOCOL_VERSION,
        env_id="praxis-gridworld-truncation-test",
        entry_point=_ENTRY_POINT,
        env_version=_ENV_VERSION,
        kwargs={"grid_size": _GRID_SIZE},
        difficulty_band=DifficultyBand.HARD,
        max_episode_steps=_MAX_EPISODE_STEPS,
        declared_reward_bounds=RewardBounds(
            min_per_step=-0.01,
            max_per_step=0.99,
            min_per_episode=-0.01 * _MAX_EPISODE_STEPS,
            max_per_episode=0.99,
        ),
        anchor_trajectories=anchors,
    )


def test_all_truncated_episodes_set_unverified_flag() -> None:
    """All 8 episodes truncate -- per_episode_unverified=True, passed=True."""
    manifest = _manifest()
    report: RewardBoundsReport = check_reward_bounds(manifest)

    assert report.per_episode_unverified is True
    assert report.terminated_episode_count == 0
    assert report.passed is True


def test_truncated_episodes_no_step_violations() -> None:
    """The gridworld's -0.01 step reward is within declared bounds -> no violations."""
    manifest = _manifest()
    report = check_reward_bounds(manifest)

    assert len(report.step_violations) == 0
    assert len(report.episode_violations) == 0


def test_sample_count_matches_config() -> None:
    """Custom config with 4 seeds produces sample_count=4, all truncated."""
    manifest = _manifest()
    cfg = RewardBoundsConfig(override_seeds=(1000, 1001, 1002, 1003))
    report = check_reward_bounds(manifest, config=cfg)

    assert report.sample_count == 4
    for sample in report.samples:
        assert sample.truncated is True
        assert sample.terminated is False
