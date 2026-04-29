"""Tests for RewardBoundsConfig and its effect on check_reward_bounds."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing build helpers from scripts/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_gridworld_manifest import build_easy_manifest  # type: ignore[import-not-found]

from praxis.checks.reward_bounds import RewardBoundsConfig, check_reward_bounds
from praxis.protocol import ActionPolicyId


def test_default_config_has_eight_seeds() -> None:
    """Default RewardBoundsConfig requests 8 derived seeds."""
    cfg = RewardBoundsConfig()
    assert cfg.sample_seed_count == 8


def test_default_config_override_seeds_is_none() -> None:
    """Default RewardBoundsConfig has no override_seeds."""
    cfg = RewardBoundsConfig()
    assert cfg.override_seeds is None


def test_custom_override_seeds_two_samples() -> None:
    """Config with override_seeds=(2000, 2001) produces sample_count=2."""
    manifest = build_easy_manifest()
    cfg = RewardBoundsConfig(override_seeds=(2000, 2001))
    report = check_reward_bounds(manifest, cfg)

    assert report.sample_count == 2
    assert len(report.samples) == 2
    assert {s.seed for s in report.samples} == {2000, 2001}


def test_default_config_action_policy() -> None:
    """Default action policy is SEEDED_RANDOM."""
    cfg = RewardBoundsConfig()
    assert cfg.action_policy == ActionPolicyId.SEEDED_RANDOM


def test_config_is_frozen() -> None:
    """RewardBoundsConfig is immutable (frozen dataclass)."""
    cfg = RewardBoundsConfig()
    try:
        cfg.sample_seed_count = 4  # type: ignore[misc]
        raise AssertionError("expected FrozenInstanceError")
    except Exception as exc:
        assert "frozen" in type(exc).__name__.lower() or "FrozenInstance" in type(exc).__name__
