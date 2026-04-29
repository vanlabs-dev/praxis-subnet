"""Adversarial gymnasium.Env subclasses used by reset correctness violation tests.

These classes are NOT registered with the gymnasium registry -- they are
loaded exclusively via importlib (entry_point "tests.checks._adversarial_envs:ClassName").
No gymnasium.register() calls must appear in this file or in the test files
that use it.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class _BaseTinyEnv(gym.Env):  # type: ignore[type-arg]
    metadata: dict[str, list[str]] = {"render_modes": []}
    observation_space: gym.spaces.Space[Any] = gym.spaces.Box(  # type: ignore[assignment]
        low=0, high=10, shape=(1,), dtype=np.int32
    )
    action_space: gym.spaces.Space[Any] = gym.spaces.Discrete(2)  # type: ignore[assignment]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._call_count: int = 0

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return np.array([0], dtype=np.int32), 0.0, False, False, {}

    def render(self) -> None:  # type: ignore[override]
        pass


class LiarTupleShape(_BaseTinyEnv):
    """reset() returns a bare array instead of the required (obs, info) tuple."""

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Any:
        super().reset(seed=seed)
        return np.array([0], dtype=np.int32)  # not a tuple


class LiarObsInSpace(_BaseTinyEnv):
    """reset() returns an obs outside the declared Box(0, 10) space."""

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        return np.array([999], dtype=np.int32), {}  # outside Box(low=0, high=10)


class LiarInfoIsDict(_BaseTinyEnv):
    """reset() returns a non-dict info."""

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, Any]:
        super().reset(seed=seed)
        return np.array([0], dtype=np.int32), "not a dict"


class LiarIdempotency(_BaseTinyEnv):
    """reset(seed=s) returns a different obs each call (non-idempotent)."""

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        self._call_count += 1
        return np.array([self._call_count % 11], dtype=np.int32), {}


class LiarMidEpisode(_BaseTinyEnv):
    """reset(seed=s) returns different obs depending on whether steps occurred."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._has_stepped: bool = False

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        if self._has_stepped:
            self._has_stepped = False
            return np.array([5], dtype=np.int32), {}  # different obs post-step
        return np.array([0], dtype=np.int32), {}

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self._has_stepped = True
        return np.array([0], dtype=np.int32), 0.0, False, False, {}


class CrashOnReset(_BaseTinyEnv):
    """reset() always raises RuntimeError."""

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        raise RuntimeError("intentional crash in reset")


class CrashOnStep(_BaseTinyEnv):
    """step() always raises RuntimeError; reset() works fine."""

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        return np.array([0], dtype=np.int32), {}

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        raise RuntimeError("intentional crash in step")
