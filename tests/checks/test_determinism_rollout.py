"""Tests for the rollout() function in the determinism check."""

from __future__ import annotations

import re

import praxis.envs  # noqa: F401 -- ensure env registrations are loaded

from praxis.checks.determinism import RolloutResult, rollout
from praxis.protocol import ActionPolicyId

_ENV_ID = "praxisgridworld-easy-v0"
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def test_rollout_returns_valid_hash() -> None:
    """rollout produces a 64-char lowercase hex hash."""
    result: RolloutResult = rollout(
        env_id=_ENV_ID,
        seed=42,
        action_policy=ActionPolicyId.SEEDED_RANDOM,
        n_steps=200,
    )
    assert _HASH_PATTERN.match(result.computed_hash) is not None
    assert result.actual_steps <= 200


def test_rollout_is_deterministic() -> None:
    """Two rollouts with identical arguments produce the same hash."""
    kwargs = dict(
        env_id=_ENV_ID,
        seed=42,
        action_policy=ActionPolicyId.SEEDED_RANDOM,
        n_steps=200,
    )
    r1: RolloutResult = rollout(**kwargs)  # type: ignore[arg-type]
    r2: RolloutResult = rollout(**kwargs)  # type: ignore[arg-type]
    assert r1.computed_hash == r2.computed_hash
    assert r1.actual_steps == r2.actual_steps
    assert r1.terminated_early == r2.terminated_early
    assert r1.truncated_early == r2.truncated_early


def test_rollout_early_stop_with_large_n_steps() -> None:
    """With large n_steps, the episode terminates or truncates before completion.

    The 5x5 gridworld has max_episode_steps = 4 * 25 = 100. A random walk
    will always stop at or before 100 steps, so n_steps=10_000 guarantees
    early stopping.
    """
    result: RolloutResult = rollout(
        env_id=_ENV_ID,
        seed=0,
        action_policy=ActionPolicyId.SEEDED_RANDOM,
        n_steps=10_000,
    )
    assert result.actual_steps < 10_000
    assert result.terminated_early or result.truncated_early


def test_rollout_different_seeds_produce_different_hashes() -> None:
    """Different seeds lead to different trajectories and different hashes."""
    r1: RolloutResult = rollout(
        env_id=_ENV_ID,
        seed=100,
        action_policy=ActionPolicyId.SEEDED_RANDOM,
        n_steps=50,
    )
    r2: RolloutResult = rollout(
        env_id=_ENV_ID,
        seed=200,
        action_policy=ActionPolicyId.SEEDED_RANDOM,
        n_steps=50,
    )
    # Different seeds should almost certainly differ (not a hard guarantee,
    # but with 50 steps and 4 actions the probability of collision is negligible).
    assert r1.computed_hash != r2.computed_hash


def test_rollout_result_is_frozen() -> None:
    """RolloutResult is immutable (frozen dataclass)."""
    result: RolloutResult = rollout(
        env_id=_ENV_ID,
        seed=1,
        action_policy=ActionPolicyId.SEEDED_RANDOM,
        n_steps=20,
    )
    try:
        result.computed_hash = "mutated"  # type: ignore[misc]
        raise AssertionError("expected FrozenInstanceError")
    except Exception as exc:
        # dataclasses.FrozenInstanceError is what frozen=True raises.
        assert "frozen" in type(exc).__name__.lower() or "FrozenInstance" in type(exc).__name__
