"""Tests for ResetCorrectnessConfig and its effect on check_reset_correctness."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing build helpers from scripts/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_gridworld_manifest import build_easy_manifest  # type: ignore[import-not-found]

from praxis.checks.reset_correctness import ResetCorrectnessConfig, check_reset_correctness


def test_default_config_sample_seed_count() -> None:
    """Default ResetCorrectnessConfig requests 8 derived seeds."""
    cfg = ResetCorrectnessConfig()
    assert cfg.sample_seed_count == 8


def test_default_config_override_seeds_is_none() -> None:
    """Default ResetCorrectnessConfig has no override_seeds."""
    cfg = ResetCorrectnessConfig()
    assert cfg.override_seeds is None


def test_default_config_mid_episode_steps() -> None:
    """Default mid_episode_steps is 5."""
    cfg = ResetCorrectnessConfig()
    assert cfg.mid_episode_steps == 5


def test_config_is_frozen() -> None:
    """ResetCorrectnessConfig is immutable (frozen dataclass)."""
    cfg = ResetCorrectnessConfig()
    try:
        cfg.sample_seed_count = 4  # type: ignore[misc]
        raise AssertionError("expected FrozenInstanceError")
    except Exception as exc:
        assert "frozen" in type(exc).__name__.lower() or "FrozenInstance" in type(exc).__name__


def test_override_seeds_respected() -> None:
    """Config with override_seeds=(1, 2, 3) -> seeds_tested == (1, 2, 3)."""
    manifest = build_easy_manifest()
    cfg = ResetCorrectnessConfig(override_seeds=(1, 2, 3))
    report = check_reset_correctness(manifest, cfg)

    assert report.seeds_tested == (1, 2, 3)
    assert report.passed is True


def test_mid_episode_steps_zero_still_passes() -> None:
    """Setting mid_episode_steps=0 skips the mid-episode phase; check still passes."""
    manifest = build_easy_manifest()
    cfg = ResetCorrectnessConfig(mid_episode_steps=0)
    report = check_reset_correctness(manifest, cfg)

    assert report.passed is True
    assert len(report.violations) == 0
