"""Tests for gymnasium registration of PraxisGridworld variants."""

import gymnasium as gym
import numpy as np
import pytest

import praxis.envs  # noqa: F401 -- triggers registration as a side effect


@pytest.mark.parametrize(
    "env_id,expected_grid_size",
    [
        ("PraxisGridworld-Easy-v0", 5),
        ("PraxisGridworld-Medium-v0", 10),
        ("PraxisGridworld-Hard-v0", 20),
    ],
)
def test_registered_env_makes_and_has_correct_grid_size(
    env_id: str, expected_grid_size: int
) -> None:
    env = gym.make(env_id)
    assert env.unwrapped.grid_size == expected_grid_size  # type: ignore[attr-defined]
    env.close()


@pytest.mark.parametrize(
    "env_id",
    [
        "PraxisGridworld-Easy-v0",
        "PraxisGridworld-Medium-v0",
        "PraxisGridworld-Hard-v0",
    ],
)
def test_registered_env_has_discrete4_action_space(env_id: str) -> None:
    env = gym.make(env_id)
    assert env.action_space == gym.spaces.Discrete(4)
    env.close()


@pytest.mark.parametrize(
    "env_id",
    [
        "PraxisGridworld-Easy-v0",
        "PraxisGridworld-Medium-v0",
        "PraxisGridworld-Hard-v0",
    ],
)
def test_registered_env_has_correct_obs_space_shape(env_id: str) -> None:
    env = gym.make(env_id)
    assert env.observation_space.shape == (2,)
    env.close()


@pytest.mark.parametrize(
    "env_id",
    [
        "PraxisGridworld-Easy-v0",
        "PraxisGridworld-Medium-v0",
        "PraxisGridworld-Hard-v0",
    ],
)
def test_registered_env_obs_dtype_is_int32(env_id: str) -> None:
    env = gym.make(env_id)
    assert env.observation_space.dtype == np.int32  # type: ignore[union-attr]
    env.close()
