"""Basic integration tests for check_reward_bounds against PraxisGridworld."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing build helpers from scripts/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_gridworld_manifest import build_easy_manifest  # type: ignore[import-not-found]

from praxis.checks.reward_bounds import (
    RewardBoundsReport,
    check_reward_bounds,
)
from praxis.protocol import DifficultyBand, EnvManifest, RewardBounds, TrajectoryAnchor
from praxis.protocol.types import ActionPolicyId

_ENTRY_POINT = "praxis.envs.gridworld:PraxisGridworld"
_PROTOCOL_VERSION = "0.2.0"
_ENV_VERSION = "0.1.0"


def _minimal_anchors() -> list[TrajectoryAnchor]:
    """Return 4 placeholder anchors (needed by EnvManifest validation)."""
    return [
        TrajectoryAnchor(
            seed=i,
            action_policy=ActionPolicyId.SEEDED_RANDOM,
            n_steps=10,
            expected_hash="a" * 64,
        )
        for i in range(1, 5)
    ]


def test_easy_manifest_passes() -> None:
    """Easy gridworld with correct declared bounds: passed=True, sample_count=8."""
    manifest = build_easy_manifest()
    report: RewardBoundsReport = check_reward_bounds(manifest)

    assert report.passed is True
    assert report.sample_count == 8
    assert len(report.step_violations) == 0
    assert len(report.episode_violations) == 0


def test_step_violation_max_too_tight() -> None:
    """max_per_step=-0.5 causes violations: every -0.01 step reward exceeds it.

    The gridworld always emits -0.01 on non-goal steps. Setting max_per_step
    below -0.01 (e.g. -0.5) guarantees violations on every step.
    """
    manifest = build_easy_manifest()
    tight_bounds = RewardBounds(
        min_per_step=-1.0,
        max_per_step=-0.5,  # tighter than -0.01 -> every step is a violation
        min_per_episode=-200.0,
        max_per_episode=1.0,
    )
    corrupt = manifest.model_copy(update={"declared_reward_bounds": tight_bounds})
    report = check_reward_bounds(corrupt)

    assert report.passed is False
    assert len(report.step_violations) > 0
    for v in report.step_violations:
        assert v.observed_reward > v.bound_max


def test_step_violation_min_too_tight() -> None:
    """min_per_step=+0.5 causes violations: -0.01 is below it."""
    manifest = build_easy_manifest()
    tight_bounds = RewardBounds(
        min_per_step=0.5,  # -0.01 < 0.5 -> every non-goal step is a violation
        max_per_step=1.0,
        min_per_episode=-200.0,
        max_per_episode=1.0,
    )
    corrupt = manifest.model_copy(update={"declared_reward_bounds": tight_bounds})
    report = check_reward_bounds(corrupt)

    assert report.passed is False
    assert len(report.step_violations) > 0
    for v in report.step_violations:
        assert v.observed_reward < v.bound_min


def test_episode_violation_on_small_grid() -> None:
    """3x3 grid with a max_per_episode bound below the actual best total
    triggers an episode violation on terminated episodes.

    On a 3x3 grid (max_episode_steps=36), random-walk seeds terminate
    naturally in the majority of rollouts. The shortest possible terminal
    total is 1.0 - 0.01 * 4 = 0.96. Setting max_per_episode=0.5 forces
    violations on all terminals.
    """
    grid_size = 3
    max_steps = 4 * grid_size * grid_size  # 36

    bounds = RewardBounds(
        min_per_step=-0.01,
        max_per_step=0.99,
        min_per_episode=-0.36,
        max_per_episode=0.5,  # all terminal totals (>= 0.73) exceed this
    )
    manifest = EnvManifest(
        protocol_version=_PROTOCOL_VERSION,
        env_id="praxis-gridworld-test",
        entry_point=_ENTRY_POINT,
        env_version=_ENV_VERSION,
        kwargs={"grid_size": grid_size},
        difficulty_band=DifficultyBand.EASY,
        max_episode_steps=max_steps,
        declared_reward_bounds=bounds,
        anchor_trajectories=_minimal_anchors(),
    )
    report = check_reward_bounds(manifest)

    # At least some episodes must have terminated.
    assert report.terminated_episode_count > 0
    assert report.passed is False
    assert len(report.episode_violations) > 0
    for v in report.episode_violations:
        assert v.episode_total > v.bound_max
