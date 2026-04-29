"""Validation/error-path tests for TabularQLearning and _obs_to_key."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from praxis.solver.tabular_q import TabularQLearning, _obs_to_key

from tests.solver._toy_env import ToyChainEnv


class _BoxActionEnv(gym.Env):  # type: ignore[type-arg]
    """Minimal stub env with a continuous Box action space."""

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return np.array([0.0], dtype=np.float32), 0.0, False, False, {}

    def render(self) -> None:  # type: ignore[override]
        return None

    def close(self) -> None:
        return None


def test_train_rejects_non_discrete_action_space() -> None:
    """Box action space raises NotImplementedError mentioning 'Discrete'."""
    solver = TabularQLearning()
    env = _BoxActionEnv()

    with pytest.raises(NotImplementedError) as exc_info:
        solver.train(env, seed=0, budget=10)

    assert "Discrete" in str(exc_info.value)


def test_evaluate_rejects_wrong_state_type() -> None:
    """evaluate(state=<not TabularQState>) raises TypeError."""
    solver = TabularQLearning()
    env = ToyChainEnv()

    with pytest.raises(TypeError):
        solver.evaluate(env, {"wrong": "type"}, seed=0, n_episodes=1)

    with pytest.raises(TypeError):
        solver.evaluate(env, 42, seed=0, n_episodes=1)


def test_obs_to_key_string_raises() -> None:
    """_obs_to_key on a string raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        _obs_to_key("invalid")


def test_obs_to_key_ndarray_returns_int_tuple() -> None:
    """_obs_to_key on np.array([1, 2, 3], dtype=int32) returns (1, 2, 3)."""
    result = _obs_to_key(np.array([1, 2, 3], dtype=np.int32))
    assert result == (1, 2, 3)
    assert all(isinstance(v, int) for v in result)


def test_obs_to_key_numpy_scalar_returns_one_tuple() -> None:
    """_obs_to_key on numpy integer scalars returns a 1-tuple."""
    assert _obs_to_key(np.int32(5)) == (5,)
    assert _obs_to_key(np.int64(7)) == (7,)
