"""Determinism check for Praxis validator.

Verifies that an environment produced via importlib resolves to bit-identical
trajectories when replayed with the same seed and canonical action policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from praxis.checks._rollout import (
    POLICY_REGISTRY,
    ActionPolicy,
    EnvSpec,
    SeededRandomPolicy,
    iter_rollout,
)
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

# Re-export so that existing callers (scripts/build_gridworld_manifest.py and
# any downstream consumers) can continue to import these names from
# praxis.checks.determinism without modification.
__all__ += []  # names already listed above


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
    obs0, it = iter_rollout(env_spec, seed, action_policy, n_steps)

    observations: list[Any] = [obs0]
    actions: list[int] = []
    rewards: list[float] = []
    terminations: list[bool] = []
    truncations: list[bool] = []
    terminated_early = False
    truncated_early = False

    for record in it:
        observations.append(record.obs)
        actions.append(record.action)
        rewards.append(record.reward)
        terminations.append(record.terminated)
        truncations.append(record.truncated)
        if record.terminated:
            terminated_early = True
        if record.truncated:
            truncated_early = True

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
        actual_steps=len(actions),
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
