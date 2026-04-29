"""Configuration tests for check_solver_baseline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from build_gridworld_manifest import build_easy_manifest

from praxis.checks.solver_baseline import (
    BandConfig,
    DEFAULT_BAND_CONFIGS,
    SolverBaselineConfig,
    check_solver_baseline,
)
from praxis.protocol import DifficultyBand


def test_default_config() -> None:
    cfg = SolverBaselineConfig()
    assert DifficultyBand.EASY in cfg.band_configs
    assert DifficultyBand.MEDIUM in cfg.band_configs
    assert DifficultyBand.HARD in cfg.band_configs
    assert cfg.override_train_seed is None
    assert cfg.override_eval_seeds is None


def test_default_bands_have_expected_constants() -> None:
    """Pin the Phase 1 uniform-threshold values (T=0.5 across all bands)."""
    e = DEFAULT_BAND_CONFIGS[DifficultyBand.EASY]
    assert e.training_budget == 10_000
    assert e.eval_episodes == 20
    assert e.threshold_normalized == 0.5

    m = DEFAULT_BAND_CONFIGS[DifficultyBand.MEDIUM]
    assert m.training_budget == 30_000
    assert m.eval_episodes == 20
    assert m.threshold_normalized == 0.5

    h = DEFAULT_BAND_CONFIGS[DifficultyBand.HARD]
    assert h.training_budget == 100_000
    assert h.eval_episodes == 20
    assert h.threshold_normalized == 0.5


def test_custom_band_config_overrides_threshold() -> None:
    """Custom threshold is respected: threshold in (random_norm, solver_norm) gives passed=True.

    Gridworld EASY calibration: random_norm~0.499, solver_norm~1.0 at full budget.
    threshold=0.6 sits above random_norm and below solver_norm, so the check passes.
    """
    manifest = build_easy_manifest()
    cfg = SolverBaselineConfig(
        band_configs={
            DifficultyBand.EASY: BandConfig(training_budget=10_000, eval_episodes=3, threshold_normalized=0.6),
        }
    )
    report = check_solver_baseline(manifest, cfg)
    assert report.passed is True
    assert report.threshold_normalized == 0.6
    assert report.failure_reason is None


def test_override_eval_seeds_sets_episode_count() -> None:
    """Eval episodes follows band_config.eval_episodes; override_eval_seeds must agree."""
    manifest = build_easy_manifest()
    cfg = SolverBaselineConfig(
        band_configs={
            DifficultyBand.EASY: BandConfig(training_budget=200, eval_episodes=3, threshold_normalized=0.0),
        },
        override_eval_seeds=(11, 22, 33),
    )
    report = check_solver_baseline(manifest, cfg)
    assert report.eval_episodes == 3
    assert len(report.per_episode_returns_random) == 3
