"""3-state deterministic chain MDP for solver unit tests.

States: 0 (start), 1 (middle), 2 (terminal).
Actions: 0=left, 1=right.

Transitions and rewards:
  - right from 0 -> 1, reward -0.01
  - right from 1 -> 2, reward +0.99 (step penalty -0.01 + goal bonus +1.0)
  - left  from 0 -> 0 (wall bump), reward -0.01
  - left  from 1 -> 0, reward -0.01
  - reaching state 2 sets terminated=True

Optimal "always right" trajectory: -0.01 + 0.99 = +0.98.

NOT registered with gymnasium. NO gymnasium.register() calls.
Leading underscore keeps pytest from collecting this as a test module.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt


class ToyChainEnv(gym.Env):  # type: ignore[type-arg]
    """Minimal 3-state chain for testing TabularQLearning."""

    metadata: dict[str, list[str]] = {"render_modes": []}

    _MAX_STEPS: int = 100

    def __init__(self) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0, high=2, shape=(1,), dtype=np.int32
        )
        self._state: int = 0
        self._step_count: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.int32], dict[str, Any]]:
        super().reset(seed=seed)
        self._state = 0
        self._step_count = 0
        return self._obs(), {}

    def step(
        self, action: int
    ) -> tuple[npt.NDArray[np.int32], float, bool, bool, dict[str, Any]]:
        if self._state == 2:
            # Already terminal -- should not be called, but be safe.
            return self._obs(), 0.0, True, False, {}

        if action == 1:  # right
            if self._state == 0:
                self._state = 1
                reward = -0.01
            else:  # state == 1
                self._state = 2
                reward = 0.99
        else:  # action == 0: left
            if self._state == 0:
                pass  # wall bump, stay
            else:  # state == 1
                self._state = 0
            reward = -0.01

        self._step_count += 1
        terminated: bool = self._state == 2
        truncated: bool = (not terminated) and (self._step_count >= self._MAX_STEPS)
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self) -> npt.NDArray[np.int32]:
        return np.array([self._state], dtype=np.int32)

    def render(self) -> None:  # type: ignore[override]
        return None

    def close(self) -> None:
        return None
