"""Determinism check for Praxis validator.

Verifies that an environment produced via importlib resolves to bit-identical
trajectories when replayed with the same seed and canonical action policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Protocol, runtime_checkable

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.wrappers import TimeLimit
from pydantic import BaseModel

from praxis.protocol import ActionPolicyId, EnvManifest, trajectory_hash

__all__ = [
    "EnvSpec",
    "ActionPolicy",
    "SeededRandomPolicy",
    "POLICY_REGISTRY",
    "RolloutResult",
    "rollout",
    "AnchorResult",
    "DeterminismReport",
    "check_determinism",
]


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


def _load_env(spec: EnvSpec) -> gym.Env:  # type: ignore[type-arg]
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


@runtime_checkable
class ActionPolicy(Protocol):
    """Protocol for action-generation policies used by the determinism check.

    Implementations must be stateless and produce a deterministic action
    sequence given only (seed, n_steps, action_space). The seed must fully
    determine the sequence -- no hidden state.
    """

    def actions(
        self,
        seed: int,
        n_steps: int,
        action_space: gym.Space,  # type: ignore[type-arg]
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
        action_space: gym.Space,  # type: ignore[type-arg]
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


@dataclass(frozen=True)
class RolloutResult:
    """Result of a single trajectory rollout.

    Attributes
    ----------
    computed_hash:
        blake2b-256 hex digest of the trajectory (excluding infos).
    actual_steps:
        Number of steps actually executed. May be less than the requested
        n_steps when the episode terminates or truncates early.
    terminated_early:
        True if the episode reached a terminal state before n_steps.
    truncated_early:
        True if the episode was truncated (e.g. max_episode_steps) before
        n_steps.
    """

    computed_hash: str
    actual_steps: int
    terminated_early: bool
    truncated_early: bool


def rollout(
    env_spec: EnvSpec,
    seed: int,
    action_policy: ActionPolicyId,
    n_steps: int,
) -> RolloutResult:
    """Execute a single trajectory rollout and return its hash.

    Implements the canonical Praxis rollout protocol:

    1. Load the environment via _load_env(env_spec) (importlib, no gym.make).
    2. Precompute the full action sequence using the specified policy.
    3. Call env.reset(seed=seed). The initial observation obs0 is included
       in the observations sequence, so len(observations) == actual_steps + 1,
       while actions/rewards/terminations/truncations are each of length
       actual_steps.
    4. Step through the environment, collecting obs/action/reward/terminated/
       truncated at each step. Stop early on termination or truncation.
    5. Hash the trajectory with include_infos=False (infos excluded per
       protocol decision).

    Parameters
    ----------
    env_spec:
        Specification for loading the environment. Must have a valid
        entry_point of the form 'module.path:ClassName'.
    seed:
        Seed forwarded to env.reset(seed=seed) and to the action policy.
    action_policy:
        Policy ID for action generation. Must be a key in POLICY_REGISTRY.
    n_steps:
        Maximum number of steps to execute. Actual steps may be fewer due to
        early termination or truncation.

    Returns
    -------
    RolloutResult
        Computed hash and episode metadata.

    Raises
    ------
    KeyError
        If action_policy is not found in POLICY_REGISTRY.
    NotImplementedError
        If the action policy does not support the environment's action space.
    """
    policy = POLICY_REGISTRY[action_policy]

    env = _load_env(env_spec)
    try:
        # Precompute actions up front to decouple generation from stepping.
        action_seq = policy.actions(seed=seed, n_steps=n_steps, action_space=env.action_space)

        obs0, _info0 = env.reset(seed=seed)

        observations: list[object] = [obs0]
        actions: list[int] = []
        rewards: list[float] = []
        terminations: list[bool] = []
        truncations: list[bool] = []

        terminated_early = False
        truncated_early = False

        for t in range(n_steps):
            action = int(action_seq[t])
            obs, reward, terminated, truncated, _info = env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            terminations.append(bool(terminated))
            truncations.append(bool(truncated))

            if terminated:
                terminated_early = True
                break
            if truncated:
                truncated_early = True
                break
    finally:
        env.close()

    actual_steps = len(actions)
    computed = trajectory_hash(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        truncations=truncations,
        infos=[],
        include_infos=False,
    )

    return RolloutResult(
        computed_hash=computed,
        actual_steps=actual_steps,
        terminated_early=terminated_early,
        truncated_early=truncated_early,
    )


class AnchorResult(BaseModel):
    """Per-anchor outcome from a determinism check.

    Attributes
    ----------
    seed:
        The seed used for this anchor.
    declared_hash:
        The expected hash from the manifest's TrajectoryAnchor.
    computed_hash:
        The hash produced by re-running the rollout.
    matched:
        True iff declared_hash == computed_hash.
    actual_steps:
        Number of steps actually executed in the rollout.
    terminated_early:
        True if the episode terminated before n_steps.
    truncated_early:
        True if the episode was truncated before n_steps.
    """

    seed: int
    declared_hash: str
    computed_hash: str
    matched: bool
    actual_steps: int
    terminated_early: bool
    truncated_early: bool


class DeterminismReport(BaseModel):
    """Aggregate report from check_determinism.

    Attributes
    ----------
    env_id:
        The environment ID from the manifest that was checked.
    passed:
        True iff every anchor hash matched (strict pass/fail).
    anchor_count:
        Total number of anchors evaluated.
    matched_count:
        Number of anchors whose hash matched.
    anchors:
        Per-anchor results in manifest order.
    """

    env_id: str
    passed: bool
    anchor_count: int
    matched_count: int
    anchors: list[AnchorResult]


def check_determinism(manifest: EnvManifest) -> DeterminismReport:
    """Run all anchor trajectories in the manifest and compare hashes.

    Each anchor is re-executed from scratch using the declared seed, action
    policy, and n_steps. The computed hash is compared against the declared
    expected_hash. A single mismatch causes passed=False.

    Parameters
    ----------
    manifest:
        A validated EnvManifest whose anchor_trajectories list all anchors to
        check. The environment is loaded via importlib using manifest.entry_point
        and manifest.kwargs -- gym.make is not used.

    Returns
    -------
    DeterminismReport
        Strict pass/fail result with per-anchor diagnostics.
    """
    env_spec = EnvSpec(
        entry_point=manifest.entry_point,
        kwargs=dict(manifest.kwargs),
        max_episode_steps=manifest.max_episode_steps,
    )

    anchor_results: list[AnchorResult] = []

    for anchor in manifest.anchor_trajectories:
        result = rollout(
            env_spec=env_spec,
            seed=anchor.seed,
            action_policy=anchor.action_policy,
            n_steps=anchor.n_steps,
        )
        matched = result.computed_hash == anchor.expected_hash
        anchor_results.append(
            AnchorResult(
                seed=anchor.seed,
                declared_hash=anchor.expected_hash,
                computed_hash=result.computed_hash,
                matched=matched,
                actual_steps=result.actual_steps,
                terminated_early=result.terminated_early,
                truncated_early=result.truncated_early,
            )
        )

    matched_count = sum(1 for a in anchor_results if a.matched)
    passed = matched_count == len(anchor_results)

    return DeterminismReport(
        env_id=manifest.env_id,
        passed=passed,
        anchor_count=len(anchor_results),
        matched_count=matched_count,
        anchors=anchor_results,
    )
