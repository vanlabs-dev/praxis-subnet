"""Adversarial tests for check_reset_correctness violation categories.

Adversarial envs are defined in tests/checks/_adversarial_envs.py and loaded
via importlib entry_points. No gymnasium.register() calls appear anywhere in
this file or in the fixture module.
"""

from __future__ import annotations

from praxis.checks.reset_correctness import (
    ResetCheckCategory,
    ResetCorrectnessConfig,
    check_reset_correctness,
)
from praxis.protocol import (
    ActionPolicyId,
    DifficultyBand,
    EnvManifest,
    RewardBounds,
    TrajectoryAnchor,
)

_PROTOCOL_VERSION = "0.2.0"
_ADV_MODULE = "tests.checks._adversarial_envs"
_FAKE_HASH = "0" * 64


def _adversarial_manifest(env_id: str, class_name: str) -> EnvManifest:
    """Build a minimal valid EnvManifest pointing at an adversarial env class."""
    return EnvManifest(
        protocol_version=_PROTOCOL_VERSION,
        env_id=env_id,
        env_version="0.1.0",
        entry_point=f"{_ADV_MODULE}:{class_name}",
        kwargs={},
        difficulty_band=DifficultyBand.EASY,
        max_episode_steps=20,
        declared_reward_bounds=RewardBounds(
            min_per_step=-1.0,
            max_per_step=1.0,
            min_per_episode=-100.0,
            max_per_episode=100.0,
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
    )


# Use a single seed to keep tests fast and deterministic.
_SINGLE_SEED_CFG = ResetCorrectnessConfig(override_seeds=(42,), mid_episode_steps=3)


def test_tuple_shape_violation() -> None:
    """LiarTupleShape: reset returns bare array -> TUPLE_SHAPE violation."""
    manifest = _adversarial_manifest("liar-tuple-shape", "LiarTupleShape")
    report = check_reset_correctness(manifest, _SINGLE_SEED_CFG)

    assert report.passed is False
    categories = [v.category for v in report.violations]
    assert ResetCheckCategory.TUPLE_SHAPE in categories


def test_obs_in_space_violation() -> None:
    """LiarObsInSpace: obs=999 outside Box(0, 10) -> OBS_IN_SPACE violation."""
    manifest = _adversarial_manifest("liar-obs-in-space", "LiarObsInSpace")
    report = check_reset_correctness(manifest, _SINGLE_SEED_CFG)

    assert report.passed is False
    categories = [v.category for v in report.violations]
    assert ResetCheckCategory.OBS_IN_SPACE in categories


def test_info_is_dict_violation() -> None:
    """LiarInfoIsDict: info is a string -> INFO_IS_DICT violation."""
    manifest = _adversarial_manifest("liar-info-is-dict", "LiarInfoIsDict")
    report = check_reset_correctness(manifest, _SINGLE_SEED_CFG)

    assert report.passed is False
    categories = [v.category for v in report.violations]
    assert ResetCheckCategory.INFO_IS_DICT in categories


def test_seed_idempotency_violation() -> None:
    """LiarIdempotency: reset(seed=s) returns different obs each call -> SEED_IDEMPOTENCY."""
    manifest = _adversarial_manifest("liar-idempotency", "LiarIdempotency")
    report = check_reset_correctness(manifest, _SINGLE_SEED_CFG)

    assert report.passed is False
    categories = [v.category for v in report.violations]
    assert ResetCheckCategory.SEED_IDEMPOTENCY in categories


def test_mid_episode_reset_violation() -> None:
    """LiarMidEpisode: post-step reset returns different obs -> MID_EPISODE_RESET."""
    manifest = _adversarial_manifest("liar-mid-episode", "LiarMidEpisode")
    report = check_reset_correctness(manifest, _SINGLE_SEED_CFG)

    assert report.passed is False
    categories = [v.category for v in report.violations]
    assert ResetCheckCategory.MID_EPISODE_RESET in categories


def test_reset_crashed_violation() -> None:
    """CrashOnReset: reset() raises RuntimeError -> RESET_CRASHED violation."""
    manifest = _adversarial_manifest("crash-on-reset", "CrashOnReset")
    report = check_reset_correctness(manifest, _SINGLE_SEED_CFG)

    assert report.passed is False
    categories = [v.category for v in report.violations]
    assert ResetCheckCategory.RESET_CRASHED in categories


def test_step_crashed_violation() -> None:
    """CrashOnStep: step() raises RuntimeError -> STEP_CRASHED violation."""
    manifest = _adversarial_manifest("crash-on-step", "CrashOnStep")
    report = check_reset_correctness(manifest, _SINGLE_SEED_CFG)

    assert report.passed is False
    categories = [v.category for v in report.violations]
    assert ResetCheckCategory.STEP_CRASHED in categories


def test_no_exception_escapes_on_crash() -> None:
    """Buggy envs must never propagate exceptions; check always returns a report."""
    for class_name, env_id in [
        ("CrashOnReset", "crash-no-escape-reset"),
        ("CrashOnStep", "crash-no-escape-step"),
    ]:
        manifest = _adversarial_manifest(env_id, class_name)
        # Must not raise
        report = check_reset_correctness(manifest, _SINGLE_SEED_CFG)
        assert isinstance(report, object)
