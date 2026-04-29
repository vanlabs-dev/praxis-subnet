"""Solver-baseline validator check.

Looks up the manifest's reference_solver from SOLVER_REGISTRY, trains it
on a band-appropriate budget, evaluates the trained policy, and asserts
the normalized mean episodic return clears a per-band threshold.

Phase 1 ships a uniform threshold T=0.5 across all bands (closes RT-004
F-021). Bands differ only in training budget. Phase 2 will refine via
multi-seed statistical calibration.

Both lower-bound and random-policy-floor are enforced: a manifest fails if
the solver's normalized return is below the threshold (failure_reason=
"solver_below_threshold"), if the random-policy baseline clears the same
threshold (failure_reason="trivial_random_baseline"), or both. The
random-policy floor is a hard failure across all bands; there is no
EASY-band exemption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

from pydantic import BaseModel

from praxis.checks._rollout import EnvSpec, _load_env, iter_rollout, spec_from_manifest
from praxis.checks._seeds import derive_validator_seeds
from praxis.protocol import ActionPolicyId, DifficultyBand, EnvManifest
from praxis.protocol.types import SolverId
from praxis.solver.registry import SOLVER_REGISTRY

__all__ = [
    "BandConfig",
    "DEFAULT_BAND_CONFIGS",
    "FailureReason",
    "SolverBaselineConfig",
    "SolverBaselineReport",
    "check_solver_baseline",
]

# Four mutually-exclusive reasons a SolverBaselineReport can have passed=False.
FailureReason = Literal["solver_below_threshold", "trivial_random_baseline", "both"]


# ---------------------------------------------------------------------------
# Band configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BandConfig:
    """Per-band training budget, evaluation count, and pass threshold."""

    training_budget: int
    eval_episodes: int
    threshold_normalized: float


# Phase 1: uniform threshold T=0.5 across all bands. Bands differ only in
# training budget. Closes RT-004 F-021 (creator declaring HARD on actually-EASY
# env to lower threshold from 0.7 to 0.1). Residual gap: declaring HARD on an
# actually-MEDIUM env to gain 100K budget vs 30K is documented as Phase 2 work
# under Option D (two-budget compute verification: train at the declared band's
# budget AND at the next-easier band's budget; pass requires the easier-band run
# to be below threshold).
DEFAULT_BAND_CONFIGS: Final[dict[DifficultyBand, BandConfig]] = {
    DifficultyBand.EASY:   BandConfig(training_budget=10_000,  eval_episodes=20, threshold_normalized=0.5),
    DifficultyBand.MEDIUM: BandConfig(training_budget=30_000,  eval_episodes=20, threshold_normalized=0.5),
    DifficultyBand.HARD:   BandConfig(training_budget=100_000, eval_episodes=20, threshold_normalized=0.5),
}


@dataclass(frozen=True, slots=True)
class SolverBaselineConfig:
    """Configuration for check_solver_baseline.

    Attributes:
        band_configs: per-band training budget / eval episodes / threshold.
            Defaults to DEFAULT_BAND_CONFIGS. Override to adjust calibration.
        override_train_seed: explicit training seed. For tests and
            red-team experiments only; production paths leave None and
            use derive_validator_seeds(manifest, 1, b"solver_baseline")[0].
        override_eval_seeds: explicit eval seed tuple. For tests and
            red-team only; production paths leave None and use
            derive_validator_seeds(manifest, eval_episodes, b"solver_baseline_eval").
    """

    band_configs: dict[DifficultyBand, BandConfig] = field(
        default_factory=lambda: dict(DEFAULT_BAND_CONFIGS)
    )
    override_train_seed: int | None = None
    override_eval_seeds: tuple[int, ...] | None = None


# ---------------------------------------------------------------------------
# Report schema
# ---------------------------------------------------------------------------


class SolverBaselineReport(BaseModel):
    """Structured report from check_solver_baseline.

    Attributes
    ----------
    env_id:
        The environment ID from the manifest.
    passed:
        True iff solver_normalized >= threshold AND random_normalized < threshold.
    difficulty_band:
        The band declared in the manifest.
    reference_solver:
        The solver used for training and evaluation.
    training_budget:
        Number of env steps used during training.
    eval_episodes:
        Number of evaluation episodes run.
    raw_mean_return:
        Mean episodic return from the solver's greedy policy (unnormalized).
    normalized_mean_return:
        raw_mean_return normalized to [0, 1] via declared reward bounds.
        Clamped at 0 from below; no upper clamp.
    threshold_normalized:
        The band's pass threshold.
    random_baseline_normalized:
        Normalized mean return of a seeded-random policy over the same
        eval_episodes. Used as a hard-fail signal; if this clears the
        threshold the report has passed=False regardless of the solver score.
    trivial_random_warning:
        True when random_baseline_normalized >= threshold_normalized.
        Diagnostic flag retained for visibility; the actual pass/fail signal
        from the random floor is captured in failure_reason. No EASY-band
        exemption applies.
    failure_reason:
        Set when passed=False. One of "solver_below_threshold",
        "trivial_random_baseline", or "both". None when passed=True.
    per_episode_returns_solver:
        Per-episode returns from the solver's greedy policy.
    per_episode_returns_random:
        Per-episode returns from the random baseline.
    """

    env_id: str
    passed: bool
    difficulty_band: DifficultyBand
    reference_solver: SolverId
    training_budget: int
    eval_episodes: int
    raw_mean_return: float
    normalized_mean_return: float
    threshold_normalized: float
    random_baseline_normalized: float
    trivial_random_warning: bool
    failure_reason: FailureReason | None = None
    per_episode_returns_solver: tuple[float, ...]
    per_episode_returns_random: tuple[float, ...]


# ---------------------------------------------------------------------------
# Random baseline helper
# ---------------------------------------------------------------------------


def _random_baseline_returns(
    spec: EnvSpec,
    eval_seeds: tuple[int, ...],
    max_episode_steps: int,
) -> list[float]:
    """One random-policy episode per seed; return the per-episode total reward.

    Uses iter_rollout to keep the loop semantics aligned with the rest of
    the validator's rollout primitives.
    """
    returns: list[float] = []
    for seed in eval_seeds:
        _obs0, _info0, it = iter_rollout(
            spec, int(seed), ActionPolicyId.SEEDED_RANDOM, max_episode_steps
        )
        ep_return = 0.0
        for record in it:
            ep_return += float(record.reward)
        returns.append(ep_return)
    return returns


# ---------------------------------------------------------------------------
# check_solver_baseline
# ---------------------------------------------------------------------------


def check_solver_baseline(
    manifest: EnvManifest,
    config: SolverBaselineConfig | None = None,
) -> SolverBaselineReport:
    """Train the manifest's reference_solver and verify normalized return clears the band threshold.

    See module docstring for the lower-bound + random-baseline semantics.

    Parameters
    ----------
    manifest:
        Validated environment manifest. The env is loaded via importlib
        using manifest.entry_point and manifest.kwargs.
    config:
        Check configuration. Defaults to SolverBaselineConfig() if not
        provided (DEFAULT_BAND_CONFIGS thresholds, derived seeds).

    Returns
    -------
    SolverBaselineReport
        Structured report with pass/fail status and full diagnostics.
    """
    cfg = config if config is not None else SolverBaselineConfig()
    band_cfg = cfg.band_configs[manifest.difficulty_band]
    solver = SOLVER_REGISTRY[manifest.reference_solver]

    train_seed = (
        cfg.override_train_seed
        if cfg.override_train_seed is not None
        else derive_validator_seeds(manifest, 1, b"solver_baseline")[0]
    )
    eval_seeds = (
        cfg.override_eval_seeds
        if cfg.override_eval_seeds is not None
        else derive_validator_seeds(manifest, band_cfg.eval_episodes, b"solver_baseline_eval")
    )

    spec = spec_from_manifest(manifest)

    # Train
    env_train = _load_env(spec)
    try:
        state = solver.train(env_train, train_seed, band_cfg.training_budget)
    finally:
        try:
            env_train.close()
        except Exception:
            pass

    # Solver eval
    env_eval = _load_env(spec)
    try:
        eval_result = solver.evaluate(
            env_eval, state, seed=eval_seeds[0], n_episodes=band_cfg.eval_episodes
        )
    finally:
        try:
            env_eval.close()
        except Exception:
            pass

    # Random baseline eval -- one episode per eval_seed
    random_returns = _random_baseline_returns(
        spec, eval_seeds[: band_cfg.eval_episodes], manifest.max_episode_steps
    )

    bounds = manifest.declared_reward_bounds
    span = bounds.max_per_episode - bounds.min_per_episode  # > 0 by manifest validator

    def _normalize(raw: float) -> float:
        norm = (raw - bounds.min_per_episode) / span
        return norm if norm >= 0.0 else 0.0  # clamp lower bound only

    raw_mean_solver = float(eval_result.mean_episodic_return)
    normalized_solver = _normalize(raw_mean_solver)
    raw_mean_random = float(sum(random_returns) / len(random_returns)) if random_returns else 0.0
    normalized_random = _normalize(raw_mean_random)

    solver_pass = normalized_solver >= band_cfg.threshold_normalized
    random_fail = normalized_random >= band_cfg.threshold_normalized

    if solver_pass and not random_fail:
        passed = True
        reason: FailureReason | None = None
    elif not solver_pass and random_fail:
        passed = False
        reason = "both"
    elif not solver_pass:
        passed = False
        reason = "solver_below_threshold"
    else:  # solver_pass and random_fail
        passed = False
        reason = "trivial_random_baseline"

    # Retained as a diagnostic flag; does not drive pass/fail directly.
    # No EASY-band exemption: random_fail is symmetric across all bands.
    trivial_warning = random_fail

    return SolverBaselineReport(
        env_id=manifest.env_id,
        passed=passed,
        difficulty_band=manifest.difficulty_band,
        reference_solver=manifest.reference_solver,
        training_budget=band_cfg.training_budget,
        eval_episodes=band_cfg.eval_episodes,
        raw_mean_return=raw_mean_solver,
        normalized_mean_return=normalized_solver,
        threshold_normalized=band_cfg.threshold_normalized,
        random_baseline_normalized=normalized_random,
        trivial_random_warning=trivial_warning,
        failure_reason=reason,
        per_episode_returns_solver=eval_result.per_episode_returns,
        per_episode_returns_random=tuple(random_returns),
    )
