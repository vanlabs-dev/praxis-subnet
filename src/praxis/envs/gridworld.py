"""Discrete NxN gridworld reference environment for the Praxis subnet.

Reward bounds derivation
------------------------
Per-step bounds:
  - Step penalty is always applied: -0.01
  - Goal bonus is applied only on the step that reaches the goal: +1.0
  - Min per step: -0.01 (no goal reached this step)
  - Max per step: -0.01 + 1.0 = +0.99 (goal reached this step)

Per-episode bounds (given max_episode_steps = 4 * grid_size^2 by default):
  - Min per episode: agent never reaches goal, every step incurs -0.01
    => min_per_episode = -0.01 * max_episode_steps
  - Max per episode: agent takes the shortest possible path to the goal.
    Shortest path length (Manhattan distance from (0,0) to (N-1,N-1)):
    = (N-1) + (N-1) = 2*(grid_size - 1) steps
    => max_per_episode = 1.0 - 0.01 * 2 * (grid_size - 1)
                       = 1.02 - 0.02 * grid_size

    Example for grid_size=5: max = 1.02 - 0.10 = 0.92
    Example for grid_size=10: max = 1.02 - 0.20 = 0.82
    Example for grid_size=20: max = 1.02 - 0.40 = 0.62
"""

from typing import Any

import numpy as np
import numpy.typing as npt

import gymnasium as gym
from gymnasium import spaces


# Action constants
_UP = 0
_RIGHT = 1
_DOWN = 2
_LEFT = 3

# Direction deltas: (row_delta, col_delta)
_DELTAS: dict[int, tuple[int, int]] = {
    _UP: (-1, 0),
    _RIGHT: (0, 1),
    _DOWN: (1, 0),
    _LEFT: (0, -1),
}


class PraxisGridworld(gym.Env[npt.NDArray[np.int32], int]):
    """Parameterised deterministic NxN gridworld.

    The agent starts at (0, 0) and must reach the goal at (N-1, N-1).
    Transitions are fully deterministic. Bumping into a wall keeps the
    agent in place (no extra penalty beyond the step cost).

    Parameters
    ----------
    grid_size:
        Side length N of the square grid. Must be >= 2.
    max_episode_steps:
        Maximum number of steps before truncation. Defaults to
        4 * grid_size^2 if not provided. Set explicitly to override.
    """

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(
        self,
        grid_size: int = 5,
        max_episode_steps: int | None = None,
    ) -> None:
        super().__init__()

        if grid_size < 2:
            raise ValueError(f"grid_size must be >= 2, got {grid_size}")

        self._grid_size = grid_size
        self._max_episode_steps: int = (
            max_episode_steps if max_episode_steps is not None else 4 * grid_size * grid_size
        )

        if self._max_episode_steps <= 0:
            raise ValueError(
                f"max_episode_steps must be > 0, got {self._max_episode_steps}"
            )

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=grid_size - 1,
            shape=(2,),
            dtype=np.int32,
        )

        # Internal state -- initialised properly in reset()
        self._pos: tuple[int, int] = (0, 0)
        self._steps: int = 0

    @property
    def grid_size(self) -> int:
        """Public accessor for the grid size parameter."""
        return self._grid_size

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.int32], dict[str, Any]]:
        """Reset the environment to the start state.

        The start state is always (0, 0) regardless of seed -- this env is
        fully deterministic. The seed is forwarded to super().reset() so
        that action_space.sample() respects it.
        """
        super().reset(seed=seed)
        self._pos = (0, 0)
        self._steps = 0
        return self._obs(), {"grid_size": self._grid_size}

    def step(
        self, action: int
    ) -> tuple[npt.NDArray[np.int32], float, bool, bool, dict[str, Any]]:
        """Apply one action and return the resulting transition.

        Parameters
        ----------
        action:
            Integer in {0, 1, 2, 3}: 0=up, 1=right, 2=down, 3=left.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        if action not in _DELTAS:
            raise ValueError(
                f"Invalid action {action!r}. Must be one of {list(_DELTAS.keys())}."
            )

        dr, dc = _DELTAS[action]
        row, col = self._pos
        new_row = max(0, min(self._grid_size - 1, row + dr))
        new_col = max(0, min(self._grid_size - 1, col + dc))
        self._pos = (new_row, new_col)
        self._steps += 1

        goal = self._grid_size - 1
        terminated: bool = (new_row == goal and new_col == goal)

        reward: float = -0.01
        if terminated:
            reward += 1.0

        truncated: bool = (not terminated) and (self._steps >= self._max_episode_steps)

        return self._obs(), reward, terminated, truncated, {}

    def _obs(self) -> npt.NDArray[np.int32]:
        """Return the current observation as a (2,) int32 array [row, col]."""
        return np.array([self._pos[0], self._pos[1]], dtype=np.int32)

    def render(self) -> None:
        """Rendering is not supported in Phase 1."""
        return None

    def close(self) -> None:
        """No resources to release."""
        return None
