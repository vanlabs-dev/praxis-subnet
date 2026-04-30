"""Fixture module for sys.modules guard tests.

This module is deliberately not imported by any production code path or
by any other test. It exists solely as a target for _load_env in
test_rollout_isolation.py to verify F-032 closure (creator modules do
not leak across _load_env calls).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class IsolationTarget(gym.Env):  # type: ignore[type-arg]
    """Trivial single-state env. Just enough to satisfy gym.Env contract."""

    metadata: dict[str, Any] = {}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}
