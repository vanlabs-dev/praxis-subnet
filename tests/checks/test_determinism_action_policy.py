"""Unit tests for SeededRandomPolicy and the ActionPolicy protocol."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from praxis.checks.determinism import ActionPolicy, SeededRandomPolicy


def test_same_seed_returns_identical_arrays() -> None:
    """Identical (seed, n_steps, action_space) calls return equal arrays."""
    policy = SeededRandomPolicy()
    space = Discrete(4)
    arr1 = policy.actions(seed=42, n_steps=100, action_space=space)
    arr2 = policy.actions(seed=42, n_steps=100, action_space=space)
    assert np.array_equal(arr1, arr2)


def test_different_seeds_return_different_arrays() -> None:
    """Different seeds produce different action arrays (with overwhelming probability)."""
    policy = SeededRandomPolicy()
    space = Discrete(4)
    arr1 = policy.actions(seed=0, n_steps=100, action_space=space)
    arr2 = policy.actions(seed=1, n_steps=100, action_space=space)
    assert not np.array_equal(arr1, arr2)


def test_non_discrete_space_raises_not_implemented() -> None:
    """Non-Discrete action space raises NotImplementedError with 'Discrete' in message."""
    policy = SeededRandomPolicy()
    box_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
    with pytest.raises(NotImplementedError, match="Discrete"):
        policy.actions(seed=0, n_steps=10, action_space=box_space)


def test_seeded_random_policy_satisfies_action_policy_protocol() -> None:
    """SeededRandomPolicy is recognized as an ActionPolicy at runtime."""
    policy = SeededRandomPolicy()
    assert isinstance(policy, ActionPolicy)


def test_output_dtype_is_int64() -> None:
    """Returned array has int64 dtype per the canonical policy spec."""
    policy = SeededRandomPolicy()
    arr = policy.actions(seed=7, n_steps=50, action_space=Discrete(4))
    assert arr.dtype == np.int64


def test_output_values_within_action_space() -> None:
    """All returned actions are valid indices for the given Discrete space."""
    policy = SeededRandomPolicy()
    n_actions = 4
    arr = policy.actions(seed=99, n_steps=200, action_space=Discrete(n_actions))
    assert int(arr.min()) >= 0
    assert int(arr.max()) < n_actions


def test_output_length_matches_n_steps() -> None:
    """Returned array has exactly n_steps elements."""
    policy = SeededRandomPolicy()
    n = 77
    arr = policy.actions(seed=3, n_steps=n, action_space=Discrete(4))
    assert len(arr) == n
