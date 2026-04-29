"""Basic behavioural tests for PraxisGridworld."""

import numpy as np
import pytest

from praxis.envs.gridworld import PraxisGridworld


def test_instantiation() -> None:
    env = PraxisGridworld(5)
    assert env.grid_size == 5


def test_reset_returns_start_state() -> None:
    env = PraxisGridworld(5)
    obs, info = env.reset()

    assert obs.dtype == np.int32
    np.testing.assert_array_equal(obs, np.array([0, 0], dtype=np.int32))
    assert info == {"grid_size": 5}


def test_step_right_from_origin() -> None:
    env = PraxisGridworld(5)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(1)  # action=1 is right

    np.testing.assert_array_equal(obs, np.array([0, 1], dtype=np.int32))
    assert reward == pytest.approx(-0.01)
    assert terminated is False
    assert truncated is False
    assert info == {}


def test_optimal_path_terminates_with_correct_reward() -> None:
    """Walk the optimal path (right then down) to the goal and verify
    the final step gives reward +0.99 (-0.01 + 1.0)."""
    n = 5
    env = PraxisGridworld(n)
    env.reset()

    # Optimal path: (n-1) rights then (n-1) downs = 2*(n-1) = 8 steps
    actions = [1] * (n - 1) + [2] * (n - 1)
    last_reward: float = 0.0
    last_terminated = False

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        last_reward = reward
        last_terminated = terminated

    np.testing.assert_array_equal(obs, np.array([n - 1, n - 1], dtype=np.int32))
    assert last_reward == pytest.approx(0.99)
    assert last_terminated is True
    assert truncated is False


def test_wall_bump_keeps_position() -> None:
    """Stepping up from (0,0) should keep the agent at (0,0) with -0.01 reward."""
    env = PraxisGridworld(5)
    env.reset()

    obs, reward, terminated, truncated, _ = env.step(0)  # action=0 is up

    np.testing.assert_array_equal(obs, np.array([0, 0], dtype=np.int32))
    assert reward == pytest.approx(-0.01)
    assert terminated is False
    assert truncated is False


def test_truncation_fires_at_max_episode_steps() -> None:
    """With max_episode_steps=3, the third wall-bump step should be truncated."""
    env = PraxisGridworld(5, max_episode_steps=3)
    env.reset()

    # Bump the top-left wall three times (action=0 up from (0,0))
    for step_idx in range(1, 4):
        obs, reward, terminated, truncated, _ = env.step(0)
        if step_idx < 3:
            assert truncated is False, f"truncated too early at step {step_idx}"
        else:
            assert truncated is True
            assert terminated is False

    np.testing.assert_array_equal(obs, np.array([0, 0], dtype=np.int32))


def test_invalid_grid_size_raises() -> None:
    with pytest.raises(ValueError, match="grid_size must be >= 2"):
        PraxisGridworld(1)


def test_invalid_action_raises() -> None:
    env = PraxisGridworld(5)
    env.reset()
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(99)
