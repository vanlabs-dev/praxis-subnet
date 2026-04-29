"""Reward bounds check for Praxis validator.

Verifies that an environment's per-step and per-episode rewards stay within
the declared bounds across a fixed sample of seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pydantic import BaseModel

from praxis.checks._rollout import EnvSpec, iter_rollout
from praxis.protocol import ActionPolicyId, EnvManifest

__all__ = [
    "RewardBoundsConfig",
    "StepViolation",
    "EpisodeViolation",
    "SeedSample",
    "RewardBoundsReport",
    "check_reward_bounds",
]


@dataclass(frozen=True, slots=True)
class RewardBoundsConfig:
    """Sampling configuration for the reward bounds check.

    Phase 1 limitation: ``sample_seeds`` is a fixed validator-side range, which
    means a sufficiently motivated creator could pre-tune their env to keep
    rewards inside declared bounds for these specific seeds and violate them
    elsewhere. Phase 2 will replace this with commit-reveal or beacon-derived
    seeds; see the relevant DR.

    Attributes
    ----------
    sample_seeds:
        Tuple of integer seeds to sample. Default is ``range(1000, 1008)``
        (eight seeds). Configurable to support targeted or expanded sampling.
    action_policy:
        Policy ID used to generate actions during sampling. Must be present
        in ``POLICY_REGISTRY``.
    """

    sample_seeds: tuple[int, ...] = field(default_factory=lambda: tuple(range(1000, 1008)))
    action_policy: ActionPolicyId = ActionPolicyId.SEEDED_RANDOM


# ---------------------------------------------------------------------------
# Pydantic report models
# ---------------------------------------------------------------------------


class StepViolation(BaseModel):
    """A single per-step reward bound violation.

    Attributes
    ----------
    seed:
        The sample seed during which the violation occurred.
    step_index:
        Zero-based index of the offending step within the episode.
    observed_reward:
        The reward that was outside the declared bounds.
    bound_min:
        Declared ``min_per_step`` at the time of the check.
    bound_max:
        Declared ``max_per_step`` at the time of the check.
    """

    seed: int
    step_index: int
    observed_reward: float
    bound_min: float
    bound_max: float


class EpisodeViolation(BaseModel):
    """A single per-episode cumulative reward bound violation.

    Only naturally-terminated episodes contribute to this list; truncated
    episodes are excluded from per-episode checking.

    Attributes
    ----------
    seed:
        The sample seed for the offending episode.
    episode_total:
        Cumulative reward sum for the episode.
    bound_min:
        Declared ``min_per_episode`` at the time of the check.
    bound_max:
        Declared ``max_per_episode`` at the time of the check.
    n_steps:
        Number of steps taken in the episode.
    """

    seed: int
    episode_total: float
    bound_min: float
    bound_max: float
    n_steps: int


class SeedSample(BaseModel):
    """Summary statistics for a single seed's rollout.

    Attributes
    ----------
    seed:
        The seed used for this rollout.
    n_steps:
        Number of steps actually executed.
    terminated:
        True if the episode reached a natural terminal state.
    truncated:
        True if the episode was cut off by the step-limit wrapper.
    min_reward_seen:
        Minimum per-step reward observed. Set to ``0.0`` if ``n_steps == 0``
        (degenerate episode that ended before any step was taken).
    max_reward_seen:
        Maximum per-step reward observed. Set to ``0.0`` if ``n_steps == 0``.
    episode_total:
        Cumulative reward sum. Zero when ``n_steps == 0``.
    """

    seed: int
    n_steps: int
    terminated: bool
    truncated: bool
    min_reward_seen: float
    max_reward_seen: float
    episode_total: float


class RewardBoundsReport(BaseModel):
    """Aggregate result from check_reward_bounds.

    Attributes
    ----------
    env_id:
        The environment ID from the manifest.
    passed:
        True iff there are zero step violations AND zero episode violations
        (strict pass/fail).
    sample_count:
        Number of seeds sampled (equals ``len(config.sample_seeds)``).
    terminated_episode_count:
        Number of sampled episodes that terminated naturally (not truncated).
    per_episode_unverified:
        True when ``terminated_episode_count == 0``. The per-episode bounds
        cannot be empirically verified because no episode terminated -- the
        check still passes in this case (no false negatives), but the caller
        should be aware that per-episode validation was skipped.
    step_violations:
        All per-step bound violations across all seeds, in the order they
        were observed. Empty on a passing check.
    episode_violations:
        All per-episode bound violations. Empty on a passing check.
    samples:
        Per-seed summary records in the order the seeds were processed.
    """

    env_id: str
    passed: bool
    sample_count: int
    terminated_episode_count: int
    per_episode_unverified: bool
    step_violations: list[StepViolation]
    episode_violations: list[EpisodeViolation]
    samples: list[SeedSample]


# ---------------------------------------------------------------------------
# check_reward_bounds
# ---------------------------------------------------------------------------


def check_reward_bounds(
    manifest: EnvManifest,
    config: RewardBoundsConfig | None = None,
) -> RewardBoundsReport:
    """Sample rollouts and verify rewards stay within declared bounds.

    For each seed in ``config.sample_seeds``, one full episode is rolled out
    (up to ``manifest.max_episode_steps`` steps). Every per-step reward is
    checked against ``declared_reward_bounds.min_per_step`` and
    ``declared_reward_bounds.max_per_step``. For naturally-terminated episodes,
    the cumulative reward is also checked against the per-episode bounds.

    Phase 1 limitation: ``config.sample_seeds`` is a fixed validator-side
    range (default ``range(1000, 1008)``), which means a motivated creator
    could pre-tune their environment to behave correctly for exactly these
    seeds while violating bounds on other seeds. Phase 2 will switch to
    commit-reveal or beacon-derived seeds to remove this attack surface.

    Bounds are strict:
    - Any per-step reward outside ``[min_per_step, max_per_step]`` is a
      violation. All violations are collected; the check does not bail on the
      first hit.
    - Any naturally-terminated episode whose cumulative reward is outside
      ``[min_per_episode, max_per_episode]`` is a violation.
    - Truncated episodes contribute only to per-step checking.
    - If zero episodes terminate naturally, ``per_episode_unverified=True``
      is set but the check does NOT fail.

    Parameters
    ----------
    manifest:
        Validated environment manifest. Bounds are read from
        ``manifest.declared_reward_bounds``. The env is loaded via
        importlib using ``manifest.entry_point`` and ``manifest.kwargs``.
    config:
        Sampling configuration. Defaults to ``RewardBoundsConfig()`` if
        not provided (8 seeds in range 1000..1007).

    Returns
    -------
    RewardBoundsReport
        Structured report with pass/fail status and full violation details.
    """
    if config is None:
        config = RewardBoundsConfig()

    bounds = manifest.declared_reward_bounds
    env_spec = EnvSpec(
        entry_point=manifest.entry_point,
        kwargs=dict(manifest.kwargs),
        max_episode_steps=manifest.max_episode_steps,
    )

    step_violations: list[StepViolation] = []
    episode_violations: list[EpisodeViolation] = []
    samples: list[SeedSample] = []

    for seed in config.sample_seeds:
        _obs0, it = iter_rollout(env_spec, seed, config.action_policy, manifest.max_episode_steps)

        min_r = math.inf
        max_r = -math.inf
        total = 0.0
        n = 0
        terminated = False
        truncated = False

        for record in it:
            n += 1
            r = record.reward
            if r < min_r:
                min_r = r
            if r > max_r:
                max_r = r
            total += r

            if r < bounds.min_per_step or r > bounds.max_per_step:
                step_violations.append(
                    StepViolation(
                        seed=seed,
                        step_index=n - 1,
                        observed_reward=r,
                        bound_min=bounds.min_per_step,
                        bound_max=bounds.max_per_step,
                    )
                )

            # Track termination/truncation from each record; the last record's
            # values are what matter after the loop.
            terminated = record.terminated
            truncated = record.truncated

        if terminated:
            if total < bounds.min_per_episode or total > bounds.max_per_episode:
                episode_violations.append(
                    EpisodeViolation(
                        seed=seed,
                        episode_total=total,
                        bound_min=bounds.min_per_episode,
                        bound_max=bounds.max_per_episode,
                        n_steps=n,
                    )
                )

        # For degenerate episodes where no step was taken, sentinel to 0.0.
        samples.append(
            SeedSample(
                seed=seed,
                n_steps=n,
                terminated=terminated,
                truncated=truncated,
                min_reward_seen=min_r if n > 0 else 0.0,
                max_reward_seen=max_r if n > 0 else 0.0,
                episode_total=total,
            )
        )

    terminated_episode_count = sum(1 for s in samples if s.terminated)
    per_episode_unverified = terminated_episode_count == 0
    passed = not step_violations and not episode_violations

    return RewardBoundsReport(
        env_id=manifest.env_id,
        passed=passed,
        sample_count=len(samples),
        terminated_episode_count=terminated_episode_count,
        per_episode_unverified=per_episode_unverified,
        step_violations=step_violations,
        episode_violations=episode_violations,
        samples=samples,
    )
