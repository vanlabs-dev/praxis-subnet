"""Determinism is verified by two independent checks:

1. check_determinism (anchor-match): the env produces the trajectory
   hashes declared in manifest.anchor_trajectories. This is the bonded
   creator claim.

2. check_determinism_self_consistency: the env is reproducible at
   validator-derived seeds (run twice, hashes must equal). Catches envs
   that are deterministic at declared anchor seeds but non-deterministic
   at validator-chosen seeds. Closes RT-001 finding F-001.

Both checks must pass for the env to be considered deterministic.

DeterminismConfig.hash_infos toggles whether step-level info dicts are
folded into the trajectory hash. Default False matches the policy that
infos are non-deterministic by spec; True enables paranoid-mode audits
that detect info-channel side effects (RT-001 finding F-004).
"""

from __future__ import annotations

from collections.abc import Mapping
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
from praxis.checks._seeds import derive_validator_seeds
from praxis.protocol import ActionPolicyId, EnvManifest, trajectory_hash

__all__ = [
    "EnvSpec",
    "ActionPolicy",
    "SeededRandomPolicy",
    "POLICY_REGISTRY",
    "DeterminismConfig",
    "RolloutResult",
    "rollout",
    "AnchorResult",
    "DeterminismReport",
    "check_determinism",
    "SelfConsistencyResult",
    "DeterminismSelfConsistencyReport",
    "check_determinism_self_consistency",
]

# Re-export so that existing callers (scripts/build_gridworld_manifest.py and
# any downstream consumers) can continue to import these names from
# praxis.checks.determinism without modification.
__all__ += []  # names already listed above


@dataclass(frozen=True, slots=True)
class DeterminismConfig:
    """Configuration for determinism checks.

    Attributes:
        sample_seed_count: number of seeds to derive for the
            self-consistency check. Default 8. Not used by the anchor-match
            check (check_determinism), whose seeds come from the manifest.
        override_seeds: explicit seed tuple for the self-consistency check.
            For tests and red-team experiments only; production code paths
            leave this None.
        hash_infos: when True, info dicts from each step are included in
            the trajectory hash. Default False matches "infos are
            non-deterministic by spec." When True, deterministic-at-seeds
            envs that leak walltime/pid/global state into infos will fail.
            Used for paranoid-mode audits and as a knob for the F-004 PoC.
            Both check_determinism and check_determinism_self_consistency
            respect this flag so a paranoid run flips both to strict mode
            together.
    """

    sample_seed_count: int = 8
    override_seeds: tuple[int, ...] | None = None
    hash_infos: bool = False


@dataclass(frozen=True)
class RolloutResult:
    """Result of a single trajectory rollout.

    Attributes
    ----------
    computed_hash:
        blake2b-256 hex digest of the trajectory (excluding infos by default).
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
    *,
    hash_infos: bool = False,
) -> RolloutResult:
    """Execute a single trajectory rollout and return its hash.

    Implements the canonical Praxis rollout protocol:

    1. Load the environment via _load_env(env_spec) (importlib, no gym.make).
    2. Precompute the full action sequence using the specified policy.
    3. Call env.reset(seed=seed). The initial observation obs0 and reset
       info_0 are included: len(observations) == actual_steps + 1 and
       len(infos) == actual_steps + 1, while actions/rewards/terminations/
       truncations are each of length actual_steps.
    4. Step through the environment, collecting obs/action/reward/terminated/
       truncated/info at each step. Stop early on termination or truncation.
    5. Hash the trajectory. infos are included when hash_infos=True.

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
    hash_infos:
        When True, info dicts from reset and each step are folded into the
        trajectory hash. Default False preserves backward compatibility.

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
    obs0, info_0, it = iter_rollout(env_spec, seed, action_policy, n_steps)

    observations: list[Any] = [obs0]
    actions: list[int] = []
    rewards: list[float] = []
    terminations: list[bool] = []
    truncations: list[bool] = []
    infos: list[Mapping[str, Any]] = [info_0]
    terminated_early = False
    truncated_early = False

    for record in it:
        observations.append(record.obs)
        actions.append(record.action)
        rewards.append(record.reward)
        terminations.append(record.terminated)
        truncations.append(record.truncated)
        infos.append(record.info)
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
        infos=infos,
        include_infos=hash_infos,
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


def check_determinism(
    manifest: EnvManifest,
    config: DeterminismConfig | None = None,
) -> DeterminismReport:
    """Run all anchor trajectories in the manifest and compare hashes.

    Each anchor is re-executed from scratch using the declared seed, action
    policy, and n_steps. The computed hash is compared against the declared
    expected_hash. A single mismatch causes passed=False.

    Note: cfg.sample_seed_count and cfg.override_seeds are NOT used here --
    anchor seeds come from the manifest. Only cfg.hash_infos is respected.
    The asymmetry is intentional: a paranoid run (hash_infos=True) flips both
    this check and check_determinism_self_consistency to strict mode together.

    Parameters
    ----------
    manifest:
        A validated EnvManifest whose anchor_trajectories list all anchors to
        check. The environment is loaded via importlib using manifest.entry_point
        and manifest.kwargs -- gym.make is not used.
    config:
        Optional configuration. Defaults to DeterminismConfig() when None.

    Returns
    -------
    DeterminismReport
        Strict pass/fail result with per-anchor diagnostics.
    """
    cfg = config if config is not None else DeterminismConfig()
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
            hash_infos=cfg.hash_infos,
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


class SelfConsistencyResult(BaseModel):
    """Per-seed outcome from a self-consistency check.

    Attributes
    ----------
    seed:
        The validator-derived seed used for this pair of rollouts.
    hash_a:
        Hash from the first rollout.
    hash_b:
        Hash from the second rollout.
    matched:
        True iff hash_a == hash_b.
    actual_steps_a:
        Steps executed in the first rollout.
    actual_steps_b:
        Steps executed in the second rollout.
    """

    seed: int
    hash_a: str
    hash_b: str
    matched: bool
    actual_steps_a: int
    actual_steps_b: int


class DeterminismSelfConsistencyReport(BaseModel):
    """Aggregate report from check_determinism_self_consistency.

    Attributes
    ----------
    env_id:
        The environment ID from the manifest that was checked.
    passed:
        True iff every per-seed pair of hashes matched (strict pass/fail).
    seeds_tested:
        The seeds used for the check, in order.
    per_seed_results:
        Per-seed pair results in seeds_tested order.
    """

    env_id: str
    passed: bool
    seeds_tested: tuple[int, ...]
    per_seed_results: list[SelfConsistencyResult]


def check_determinism_self_consistency(
    manifest: EnvManifest,
    config: DeterminismConfig | None = None,
) -> DeterminismSelfConsistencyReport:
    """Verify the env is reproducible at validator-derived seeds.

    For each derived seed s, run rollout twice with the same arguments
    and assert the resulting trajectory hashes match. Catches envs that
    are deterministic at declared anchor seeds but non-deterministic at
    validator-chosen seeds (RT-001 finding F-001).

    Sample seeds are derived via derive_validator_seeds with
    salt=b"determinism_self_consistency", giving disjoint sets from
    reward_bounds (b"reward_bounds") and reset_correctness
    (b"reset_correctness").

    Strict pass: every per-seed pair must match.

    Parameters
    ----------
    manifest:
        A validated EnvManifest. The environment is loaded via importlib.
    config:
        Optional configuration. Defaults to DeterminismConfig() when None.
        cfg.override_seeds, if set, replaces the derived seeds entirely.
        cfg.hash_infos controls whether info dicts enter the trajectory hash.

    Returns
    -------
    DeterminismSelfConsistencyReport
        Strict pass/fail result with per-seed pair diagnostics.
    """
    cfg = config if config is not None else DeterminismConfig()
    spec = EnvSpec(
        entry_point=manifest.entry_point,
        kwargs=dict(manifest.kwargs),
        max_episode_steps=manifest.max_episode_steps,
    )
    seeds: tuple[int, ...] = (
        cfg.override_seeds
        if cfg.override_seeds is not None
        else derive_validator_seeds(manifest, cfg.sample_seed_count, salt=b"determinism_self_consistency")
    )
    results: list[SelfConsistencyResult] = []
    for seed in seeds:
        a = rollout(spec, seed, ActionPolicyId.SEEDED_RANDOM, manifest.max_episode_steps, hash_infos=cfg.hash_infos)
        b = rollout(spec, seed, ActionPolicyId.SEEDED_RANDOM, manifest.max_episode_steps, hash_infos=cfg.hash_infos)
        results.append(SelfConsistencyResult(
            seed=seed,
            hash_a=a.computed_hash,
            hash_b=b.computed_hash,
            matched=a.computed_hash == b.computed_hash,
            actual_steps_a=a.actual_steps,
            actual_steps_b=b.actual_steps,
        ))
    return DeterminismSelfConsistencyReport(
        env_id=manifest.env_id,
        passed=all(r.matched for r in results),
        seeds_tested=seeds,
        per_seed_results=results,
    )
