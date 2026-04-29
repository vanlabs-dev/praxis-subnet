"""Adversarial gymnasium.Env subclasses used by reset correctness and determinism tests.

These classes are NOT registered with the gymnasium registry -- they are
loaded exclusively via importlib (entry_point "tests.checks._adversarial_envs:ClassName").
No gymnasium.register() calls must appear in this file or in the test files
that use it.

Shared builder
--------------
``make_adversarial_manifest(env_id, class_name)`` builds a minimal valid
EnvManifest pointing at one of the adversarial classes defined here. Import it
from this module rather than duplicating the helper in each test file.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from praxis.protocol import (
    ActionPolicyId,
    DifficultyBand,
    EnvManifest,
    RewardBounds,
    TrajectoryAnchor,
)

_PROTOCOL_VERSION = "0.2.0"
_ADV_MODULE = "tests.checks._adversarial_envs"
_FAKE_HASH = "0" * 64


def make_adversarial_manifest(env_id: str, class_name: str) -> EnvManifest:
    """Build a minimal valid EnvManifest pointing at an adversarial env class."""
    return EnvManifest(
        protocol_version=_PROTOCOL_VERSION,
        env_id=env_id,
        env_version="0.1.0",
        entry_point=f"{_ADV_MODULE}:{class_name}",
        kwargs={},
        difficulty_band=DifficultyBand.EASY,
        max_episode_steps=20,
        declared_reward_bounds=RewardBounds(
            min_per_step=-1.0,
            max_per_step=1.0,
            min_per_episode=-100.0,
            max_per_episode=100.0,
        ),
        anchor_trajectories=[
            TrajectoryAnchor(
                seed=i,
                action_policy=ActionPolicyId.SEEDED_RANDOM,
                n_steps=10,
                expected_hash=_FAKE_HASH,
            )
            for i in range(4)
        ],
    )


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


class NondeterministicReward(_BaseTinyEnv):
    """Adversarial: reward = base + small uniform noise via numpy global RNG.

    Deterministic obs/action/term/trunc, but reward jitters between calls
    even at the same seed. Used to verify check_determinism_self_consistency
    catches envs that are deterministic at declared anchors but not at
    validator-derived seeds.
    """

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        return np.array([0], dtype=np.int32), {}

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # global RNG, NOT seeded by reset -- intentionally non-deterministic
        noise = float(np.random.default_rng().uniform(-0.001, 0.001))
        return np.array([0], dtype=np.int32), 0.0 + noise, False, False, {}


class LeakyInfoEnv(_BaseTinyEnv):
    """Adversarial: deterministic obs/action/reward/term/trunc, but step's
    info dict includes a monotonically incrementing process-global counter.
    Demonstrates the F-004 info side channel: hidden under hash_infos=False
    (infos excluded from hash), exposed when hash_infos=True (infos included;
    hashes differ between two rollouts because the counter advances).
    """

    # Module-level counter so each step call gets a unique value even when
    # two rollouts of the same env run back-to-back in the same process.
    _global_step_counter: int = 0

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        return np.array([0], dtype=np.int32), {}

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        LeakyInfoEnv._global_step_counter += 1
        return (
            np.array([0], dtype=np.int32),
            0.0,
            False,
            False,
            {"leak": LeakyInfoEnv._global_step_counter},
        )
