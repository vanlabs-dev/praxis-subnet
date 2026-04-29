"""Shared rollout primitive for Praxis validator checks.

This module is internal to ``praxis.checks`` -- the leading underscore signals
that it is not part of any public API. Both ``determinism.py`` and
``reward_bounds.py`` depend on it; it must not import from either.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Protocol, runtime_checkable

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.wrappers import TimeLimit

from praxis.protocol import ActionPolicyId

__all__ = [
    "EnvSpec",
    "StepRecord",
    "iter_rollout",
    "ActionPolicy",
    "SeededRandomPolicy",
    "POLICY_REGISTRY",
]


# ---------------------------------------------------------------------------
# Environment specification and loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnvSpec:
    """Immutable specification for loading an environment via importlib.

    Attributes
    ----------
    entry_point:
        Dotted module path and class name separated by a colon, e.g.
        ``"praxis.envs.gridworld:PraxisGridworld"``.
    kwargs:
        Keyword arguments forwarded to the environment constructor.
    max_episode_steps:
        Maximum steps before the TimeLimit wrapper truncates the episode.
        Callers are responsible for providing a sensible positive value.
    """

    entry_point: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    max_episode_steps: int = 0


def _load_env(spec: EnvSpec) -> gym.Env[Any, Any]:
    """Resolve and instantiate an environment from an EnvSpec.

    Imports the module, retrieves the class, instantiates it with spec.kwargs,
    and wraps the result in a TimeLimit with spec.max_episode_steps.

    Parameters
    ----------
    spec:
        The EnvSpec describing how to load the environment.

    Returns
    -------
    gym.Env
        A TimeLimit-wrapped environment instance.

    Raises
    ------
    ValueError
        If entry_point does not contain a colon separator.
    ImportError
        If the module path cannot be imported.
    AttributeError
        If the class name does not exist on the imported module.
    TypeError
        If the resolved attribute is not callable.
    """
    if ":" not in spec.entry_point:
        raise ValueError(
            f"entry_point must be of the form 'module.path:ClassName', got {spec.entry_point!r}"
        )
    module_path, class_name = spec.entry_point.split(":", 1)
    module = import_module(module_path)  # raises ImportError on failure
    env_cls = getattr(module, class_name)  # raises AttributeError on missing attr
    if not callable(env_cls):
        raise TypeError(
            f"entry_point {spec.entry_point!r} did not resolve to a callable,"
            f" got {type(env_cls).__name__}"
        )
    env = env_cls(**spec.kwargs)
    return TimeLimit(env, max_episode_steps=spec.max_episode_steps)


# ---------------------------------------------------------------------------
# Action policies
# ---------------------------------------------------------------------------


@runtime_checkable
class ActionPolicy(Protocol):
    """Protocol for action-generation policies used by validator checks.

    Implementations must be stateless and produce a deterministic action
    sequence given only (seed, n_steps, action_space). The seed must fully
    determine the sequence -- no hidden state.
    """

    def actions(
        self,
        seed: int,
        n_steps: int,
        action_space: gym.Space[Any],
    ) -> npt.NDArray[np.int64]:
        """Return an array of n_steps actions drawn from action_space."""
        ...


class SeededRandomPolicy:
    """Canonical PCG64-seeded uniform-random policy for Discrete action spaces.

    Uses np.random.Generator(np.random.PCG64(seed)) so that action sequences
    are independent of gymnasium internals and stable across gymnasium version
    bumps.

    Phase 1 restriction: only Discrete action spaces are supported.
    """

    def actions(
        self,
        seed: int,
        n_steps: int,
        action_space: gym.Space[Any],
    ) -> npt.NDArray[np.int64]:
        """Return n_steps actions sampled uniformly from a Discrete space.

        Parameters
        ----------
        seed:
            RNG seed. Must fully determine the returned sequence.
        n_steps:
            Number of actions to generate. Precomputed in full so that action
            generation is decoupled from environment stepping.
        action_space:
            Must be a Discrete space. Any other type raises NotImplementedError.

        Raises
        ------
        NotImplementedError
            If action_space is not a gymnasium.spaces.Discrete instance.
        """
        if not isinstance(action_space, gym.spaces.Discrete):
            raise NotImplementedError(
                "SEEDED_RANDOM Phase 1 supports only Discrete action spaces"
            )
        rng = np.random.Generator(np.random.PCG64(seed))
        return rng.integers(low=0, high=int(action_space.n), size=n_steps, dtype=np.int64)


POLICY_REGISTRY: dict[ActionPolicyId, ActionPolicy] = {
    ActionPolicyId.SEEDED_RANDOM: SeededRandomPolicy(),
}


# ---------------------------------------------------------------------------
# Step record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StepRecord:
    """Immutable record of a single environment step.

    Attributes
    ----------
    obs:
        Observation returned by env.step(). Type is Any because the
        observation space shape varies by environment.
    action:
        The integer action that was applied. May be a Python int or a
        numpy integer depending on how the policy array was indexed --
        callers should not rely on the exact int subtype.
    reward:
        Step reward as a Python float.
    terminated:
        True if the episode reached a natural terminal state.
    truncated:
        True if the episode was cut off by a step-limit wrapper.
    info:
        Info dict returned by env.step(). Defaults to empty mapping.
        Exposed so downstream checks can optionally include it in
        trajectory hashes (see DeterminismConfig.hash_infos).
    """

    obs: Any
    action: int
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# iter_rollout -- shared inner step loop
# ---------------------------------------------------------------------------


def iter_rollout(
    env_spec: EnvSpec,
    seed: int,
    action_policy: ActionPolicyId,
    n_steps: int,
) -> tuple[npt.NDArray[Any], Mapping[str, Any], Iterator[StepRecord]]:
    """Load an environment, reset it, and return an iterator over step records.

    The function returns ``(obs0, info_0, iterator)`` where ``obs0`` is the
    initial observation from env.reset(), ``info_0`` is the info dict from
    env.reset(), and ``iterator`` yields one StepRecord per step until the
    episode ends or ``n_steps`` is exhausted.

    The iterator manages env lifecycle -- the environment is closed in a
    ``finally`` block when the generator is exhausted **or** garbage-collected.
    Callers should exhaust the iterator (or use it in a ``for`` loop) to ensure
    timely cleanup. In tests, ``list(iterator)`` is sufficient.

    Parameters
    ----------
    env_spec:
        Specification for loading the environment via importlib.
    seed:
        Seed forwarded to env.reset(seed=seed) and to the action policy.
    action_policy:
        Policy ID looked up in POLICY_REGISTRY.
    n_steps:
        Maximum number of steps to execute. The iterator may yield fewer
        records if the episode terminates or truncates early.

    Returns
    -------
    obs0:
        Initial observation from env.reset(seed=seed).
    info_0:
        Info dict from env.reset(seed=seed).
    iterator:
        Yields StepRecord for each step taken. The last record will have
        terminated=True or truncated=True (or both False if n_steps is
        exhausted without episode end). Each StepRecord.info carries the
        info dict from the corresponding env.step() call.

    Raises
    ------
    KeyError
        If action_policy is not found in POLICY_REGISTRY.
    NotImplementedError
        If the action policy does not support the environment's action space.
    """
    policy = POLICY_REGISTRY[action_policy]
    env = _load_env(env_spec)

    # Precompute full action sequence before reset so generation is decoupled
    # from stepping (same convention as the original rollout() function).
    action_seq = policy.actions(seed=seed, n_steps=n_steps, action_space=env.action_space)

    obs0_raw, info_0_raw = env.reset(seed=seed)
    obs0: npt.NDArray[Any] = obs0_raw
    info_0: Mapping[str, Any] = info_0_raw if isinstance(info_0_raw, Mapping) else {}

    def _step_generator() -> Iterator[StepRecord]:
        try:
            for t in range(n_steps):
                action = int(action_seq[t])
                obs, reward, terminated, truncated, step_info = env.step(action)
                yield StepRecord(
                    obs=obs,
                    action=action,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=step_info if isinstance(step_info, Mapping) else {},
                )
                if terminated or truncated:
                    break
        finally:
            env.close()

    return obs0, info_0, _step_generator()
