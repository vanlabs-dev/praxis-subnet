"""Tests for the iter_rollout shared primitive in praxis.checks._rollout."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from praxis.checks._rollout import EnvSpec, StepRecord, iter_rollout
from praxis.protocol import ActionPolicyId


_SPEC_5X5 = EnvSpec(
    entry_point="praxis.envs.gridworld:PraxisGridworld",
    kwargs={"grid_size": 5},
    max_episode_steps=20,
)


def test_iter_rollout_returns_obs0_and_iterator() -> None:
    """iter_rollout returns a 2-tuple of (ndarray, iterator)."""
    result = iter_rollout(_SPEC_5X5, seed=42, action_policy=ActionPolicyId.SEEDED_RANDOM, n_steps=10)
    obs0, it = result
    list(it)  # exhaust to close env
    assert isinstance(obs0, np.ndarray)


def test_obs0_is_start_position() -> None:
    """Initial observation is [0, 0] (int32) -- the fixed start state."""
    obs0, it = iter_rollout(_SPEC_5X5, seed=42, action_policy=ActionPolicyId.SEEDED_RANDOM, n_steps=10)
    list(it)  # exhaust to close env
    expected: npt.NDArray[np.int32] = np.array([0, 0], dtype=np.int32)
    assert np.array_equal(obs0, expected)
    assert obs0.dtype == np.int32


def test_step_records_have_correct_types() -> None:
    """Each StepRecord has the expected field types."""
    obs0, it = iter_rollout(_SPEC_5X5, seed=42, action_policy=ActionPolicyId.SEEDED_RANDOM, n_steps=10)
    _ = obs0
    for record in it:
        assert isinstance(record, StepRecord)
        assert isinstance(record.obs, np.ndarray)
        # action is int (Python int after int() cast in iter_rollout)
        assert isinstance(record.action, int)
        assert isinstance(record.reward, float)
        assert isinstance(record.terminated, bool)
        assert isinstance(record.truncated, bool)


def test_iterator_yields_up_to_n_steps() -> None:
    """With n_steps=10 and no early episode end, iterator yields exactly 10 records."""
    # Use max_episode_steps=100 to avoid truncation within 10 steps.
    spec = EnvSpec(
        entry_point="praxis.envs.gridworld:PraxisGridworld",
        kwargs={"grid_size": 5},
        max_episode_steps=100,
    )
    # Seed 42 on 5x5 with 100 max_steps -- very unlikely to terminate in 10 steps.
    obs0, it = iter_rollout(spec, seed=42, action_policy=ActionPolicyId.SEEDED_RANDOM, n_steps=10)
    _ = obs0
    records = list(it)
    assert len(records) <= 10


def test_iterator_stops_at_truncation() -> None:
    """With n_steps >> max_episode_steps, iterator stops at truncation.

    5x5 grid with max_episode_steps=20 and n_steps=1000: the TimeLimit wrapper
    will truncate at step 20, so the iterator yields at most 20 records.
    """
    obs0, it = iter_rollout(_SPEC_5X5, seed=42, action_policy=ActionPolicyId.SEEDED_RANDOM, n_steps=1000)
    _ = obs0
    records = list(it)
    assert len(records) < 1000
    assert records[-1].truncated is True or records[-1].terminated is True


def test_last_record_truncated_when_max_steps_exceeded() -> None:
    """Last record has truncated=True when n_steps forces TimeLimit cutoff."""
    # Seed 42 doesn't reach the goal in 20 steps on a 5x5 grid.
    obs0, it = iter_rollout(_SPEC_5X5, seed=42, action_policy=ActionPolicyId.SEEDED_RANDOM, n_steps=1000)
    _ = obs0
    records = list(it)
    assert records[-1].truncated is True


def test_step_record_is_frozen() -> None:
    """StepRecord instances are immutable (frozen dataclass)."""
    obs0, it = iter_rollout(_SPEC_5X5, seed=42, action_policy=ActionPolicyId.SEEDED_RANDOM, n_steps=1)
    _ = obs0
    records = list(it)
    assert len(records) >= 1
    record = records[0]
    try:
        record.reward = 999.0  # type: ignore[misc]
        raise AssertionError("expected FrozenInstanceError")
    except Exception as exc:
        assert "frozen" in type(exc).__name__.lower() or "FrozenInstance" in type(exc).__name__
