"""Adversarial env tests for check_solver_baseline."""

from __future__ import annotations

from praxis.checks.solver_baseline import BandConfig, SolverBaselineConfig, check_solver_baseline
from praxis.protocol import ActionPolicyId, DifficultyBand, EnvManifest, RewardBounds, TrajectoryAnchor
from praxis.protocol.types import SolverId

_FAKE_HASH = "0" * 64


def test_lazy_env_fails_baseline() -> None:
    """LazyEnv: env is harder than declared -> baseline fails."""
    manifest = EnvManifest(
        protocol_version="0.3.0",
        env_id="lazy-env",
        env_version="0.1.0",
        entry_point="tests.checks._adversarial_envs:LazyEnv",
        kwargs={},
        difficulty_band=DifficultyBand.EASY,
        max_episode_steps=20,
        declared_reward_bounds=RewardBounds(
            min_per_step=-1.0,
            max_per_step=0.0,
            min_per_episode=-20.0,
            max_per_episode=0.0,
        ),
        anchor_trajectories=[
            TrajectoryAnchor(
                seed=i,
                action_policy=ActionPolicyId.SEEDED_RANDOM,
                n_steps=10,
                expected_hash=_FAKE_HASH,
            )
            for i in range(4)
        ],
        reference_solver=SolverId.TABULAR_Q_LEARNING,
    )
    # Smaller budget for fast test
    cfg = SolverBaselineConfig(
        band_configs={
            DifficultyBand.EASY: BandConfig(training_budget=500, eval_episodes=5, threshold_normalized=0.7),
        }
    )
    report = check_solver_baseline(manifest, cfg)
    assert report.passed is False
    assert report.normalized_mean_return < 0.7


def test_trivial_env_passes_with_warning() -> None:
    """TrivialEnv: declared HARD but actually trivial -> passes + warning."""
    manifest = EnvManifest(
        protocol_version="0.3.0",
        env_id="trivial-env",
        env_version="0.1.0",
        entry_point="tests.checks._adversarial_envs:TrivialEnv",
        kwargs={},
        difficulty_band=DifficultyBand.HARD,
        max_episode_steps=10,
        declared_reward_bounds=RewardBounds(
            min_per_step=0.0,
            max_per_step=1.0,
            min_per_episode=0.0,
            max_per_episode=1.0,
        ),
        anchor_trajectories=[
            TrajectoryAnchor(
                seed=i,
                action_policy=ActionPolicyId.SEEDED_RANDOM,
                n_steps=5,
                expected_hash=_FAKE_HASH,
            )
            for i in range(4)
        ],
        reference_solver=SolverId.TABULAR_Q_LEARNING,
    )
    cfg = SolverBaselineConfig(
        band_configs={
            DifficultyBand.HARD: BandConfig(training_budget=200, eval_episodes=5, threshold_normalized=0.1),
        }
    )
    report = check_solver_baseline(manifest, cfg)
    assert report.passed is True
    assert report.trivial_random_warning is True
    # Random policy gets +1 every episode in 1 step -> normalized = 1.0
    assert report.random_baseline_normalized >= 0.99
