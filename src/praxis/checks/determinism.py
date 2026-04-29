"""Determinism check for Praxis validator.

Verifies that a registered gymnasium environment produces bit-identical
trajectories when replayed with the same seed and canonical action policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from praxis.protocol import ActionPolicyId, EnvManifest, trajectory_hash

__all__ = [
    "ActionPolicy",
    "SeededRandomPolicy",
    "POLICY_REGISTRY",
    "RolloutResult",
    "rollout",
    "AnchorResult",
    "DeterminismReport",
    "check_determinism",
]


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
    env_id: str,
    seed: int,
    action_policy: ActionPolicyId,
    n_steps: int,
) -> RolloutResult:
    """Execute a single trajectory rollout and return its hash.

    Implements the canonical Praxis rollout protocol:

    1. Make the environment via gym.make(env_id).
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
    env_id:
        Registered gymnasium environment ID. praxis.envs is imported to ensure
        Praxis environments are registered before gym.make is called.
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
    # Import here to guarantee env registrations are loaded regardless of
    # import order in the calling process.
    import praxis.envs  # noqa: F401

    policy = POLICY_REGISTRY[action_policy]

    env = gym.make(env_id)
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
        The gymnasium environment ID that was checked.
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
        check. praxis.envs is imported to ensure env registrations are loaded.

    Returns
    -------
    DeterminismReport
        Strict pass/fail result with per-anchor diagnostics.
    """
    # Ensure Praxis env registrations are loaded.
    import praxis.envs  # noqa: F401

    anchor_results: list[AnchorResult] = []

    for anchor in manifest.anchor_trajectories:
        result = rollout(
            env_id=manifest.env_id,
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
