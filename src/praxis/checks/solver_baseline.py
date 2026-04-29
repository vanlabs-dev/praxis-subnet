"""Solver-baseline validator check.

Iterates SOLVER_REGISTRY in insertion order, trains every applicable solver,
and conjunctively aggregates results. Pass requires (>=1 applicable solver)
AND (all applicable solvers pass).

Closes RT-004 F-023 (CRITICAL): manifest.reference_solver was creator-declared,
so a creator could over-fit their env to TabularQLearning's quirks, declare
reference_solver=TABULAR_Q_LEARNING, and the old validator would honor the
declaration -- never running PPO (Phase 2), never catching the over-fit.

Closure: check_solver_baseline ignores manifest.reference_solver entirely. Solver
selection is driven exclusively by SOLVER_REGISTRY + runtime compatibility
(NotImplementedError signals incompatibility). A creator can no longer steer
validation to a weaker solver by manipulating their manifest.

Phase 1 ships a uniform threshold T=0.5 across all bands (closes RT-004
F-021). Bands differ only in training budget. Phase 2 will refine via
multi-seed statistical calibration.

Both lower-bound and random-policy-floor are enforced: a manifest fails if
the solver's normalized return is below the threshold (failure_reason=
"solver_below_threshold"), if the random-policy baseline clears the same
threshold (failure_reason="trivial_random_baseline"), or both. The
random-policy floor is a hard failure across all bands; there is no
EASY-band exemption.

Random baseline computation is env-property (depends only on manifest + salt),
so it is computed once per check_solver_baseline call and shared across all
solver evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

from pydantic import BaseModel, Field

from praxis.checks._rollout import EnvSpec, _load_env, iter_rollout, spec_from_manifest
from praxis.checks._seeds import derive_validator_seeds
from praxis.protocol import ActionPolicyId, DifficultyBand, EnvManifest
from praxis.protocol.types import SolverId
from praxis.solver.registry import SOLVER_REGISTRY

__all__ = [
    "BandConfig",
    "DEFAULT_BAND_CONFIGS",
    "FailureReason",
    "PerSolverFailureReason",
    "AggregateFailureReason",
    "PerSolverResult",
    "SolverBaselineConfig",
    "SolverBaselineReport",
    "check_solver_baseline",
]

# Per-solver failure reasons (three values + None).
PerSolverFailureReason = Literal["solver_below_threshold", "trivial_random_baseline", "both"]

# Aggregate failure reasons (four values + None; adds "no_compatible_solver").
AggregateFailureReason = Literal[
    "solver_below_threshold", "trivial_random_baseline", "both", "no_compatible_solver"
]

# Legacy alias kept for backward compatibility with existing consumers.
FailureReason = PerSolverFailureReason


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


class PerSolverResult(BaseModel):
    """Per-solver evaluation result. Aggregated into SolverBaselineReport.

    All fields are populated by _run_one_solver after a successful
    train + evaluate cycle. Solvers that raise NotImplementedError during
    train are considered incompatible and never produce a PerSolverResult
    -- they are silently skipped in the registry iteration loop.

    Attributes
    ----------
    solver_id:
        The SolverId enum value that identifies this solver.
    passed:
        True iff normalized_mean_return >= threshold_normalized AND
        random_baseline_normalized < threshold_normalized.
    failure_reason:
        Set when passed=False. One of "solver_below_threshold",
        "trivial_random_baseline", or "both". None when passed=True.
    raw_mean_return:
        Mean episodic return from the solver's greedy policy (unnormalized).
    normalized_mean_return:
        raw_mean_return normalized to [0, 1] via declared reward bounds.
        Clamped at 0 from below; no upper clamp.
    threshold_normalized:
        The band's pass threshold used for this evaluation.
    random_baseline_normalized:
        Normalized mean return of the seeded-random policy. Precomputed
        once per check_solver_baseline call and copied here for traceability.
    trivial_random_warning:
        True when random_baseline_normalized >= threshold_normalized.
    per_episode_returns_solver:
        Per-episode returns from this solver's greedy policy.
    per_episode_returns_random:
        Per-episode returns from the random baseline (shared across solvers).
    """

    solver_id: SolverId
    passed: bool
    failure_reason: PerSolverFailureReason | None = None
    raw_mean_return: float
    normalized_mean_return: float
    threshold_normalized: float
    random_baseline_normalized: float
    trivial_random_warning: bool
    per_episode_returns_solver: tuple[float, ...]
    per_episode_returns_random: tuple[float, ...]


class SolverBaselineReport(BaseModel):
    """Structured report from check_solver_baseline.

    F-023 closure: this report reflects all applicable solvers from
    SOLVER_REGISTRY, not just the one declared in manifest.reference_solver.
    The manifest field is preserved for Phase 2 forward-compatibility but
    is ignored by the validator.

    Top-level fields (raw_mean_return, normalized_mean_return, etc.) reflect
    the FIRST applicable solver's result (earliest insertion order in
    SOLVER_REGISTRY). This is a Phase 1 backward-compat hack -- Phase 1 has
    exactly one solver, so it is unambiguous. Phase 2 consumers should migrate
    to the solver_results dict instead of relying on the top-level fields.

    Attributes
    ----------
    env_id:
        The environment ID from the manifest.
    passed:
        Conjunctive aggregate. True iff len(solver_results) >= 1 AND all
        per-solver results have passed=True.
    difficulty_band:
        The band declared in the manifest.
    reference_solver:
        The first applicable solver's SolverId. In Phase 1 this is always
        TABULAR_Q_LEARNING. Preserved for backward compatibility with
        existing report consumers; Phase 2 consumers should use
        solver_results.keys() instead.
    training_budget:
        Number of env steps used during training (from band config).
    eval_episodes:
        Number of evaluation episodes run (from band config).
    raw_mean_return:
        First applicable solver's raw mean episodic return. Phase 1 hack;
        see class docstring.
    normalized_mean_return:
        First applicable solver's normalized mean return. Phase 1 hack.
    threshold_normalized:
        The band's pass threshold.
    random_baseline_normalized:
        Normalized mean return of the seeded-random policy. Env-property;
        identical across all solvers.
    trivial_random_warning:
        True when random_baseline_normalized >= threshold_normalized.
    failure_reason:
        Aggregated. "no_compatible_solver" if zero applicable solvers;
        the first failing solver's reason (insertion order) if any fail;
        None if all passed.
    per_episode_returns_solver:
        First applicable solver's per-episode returns. Phase 1 hack.
    per_episode_returns_random:
        Per-episode returns from the random baseline (env-property).
    solver_results:
        Per-solver results keyed on SolverId. Empty iff no applicable solver.
        This is the authoritative record for Phase 2 multi-solver consumers.
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
    failure_reason: AggregateFailureReason | None = None
    per_episode_returns_solver: tuple[float, ...]
    per_episode_returns_random: tuple[float, ...]
    solver_results: dict[SolverId, PerSolverResult] = Field(default_factory=dict)
    """Per-solver PerSolverResult records keyed on SolverId. Empty iff no
    applicable solver exists in SOLVER_REGISTRY for this env. Phase 2
    consumers should read results from here rather than the top-level
    backward-compat fields."""


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


def _compute_random_baseline(
    spec: EnvSpec,
    manifest: EnvManifest,
    band_cfg: BandConfig,
    eval_seeds: tuple[int, ...],
) -> tuple[float, tuple[float, ...]]:
    """Compute the random-policy baseline normalized return and raw per-episode returns.

    This is env-property: the result depends only on (manifest, eval_seeds, band_cfg)
    and is identical for every solver evaluated in the same check_solver_baseline call.
    Compute once; pass results into each _run_one_solver call.

    Returns
    -------
    (random_baseline_normalized, per_episode_returns_random)
    """
    bounds = manifest.declared_reward_bounds
    span = bounds.max_per_episode - bounds.min_per_episode

    def _normalize(raw: float) -> float:
        norm = (raw - bounds.min_per_episode) / span
        return norm if norm >= 0.0 else 0.0

    random_returns = _random_baseline_returns(
        spec, eval_seeds[: band_cfg.eval_episodes], manifest.max_episode_steps
    )
    raw_mean_random = float(sum(random_returns) / len(random_returns)) if random_returns else 0.0
    normalized_random = _normalize(raw_mean_random)
    return normalized_random, tuple(random_returns)


# ---------------------------------------------------------------------------
# Per-solver helper
# ---------------------------------------------------------------------------


def _run_one_solver(
    solver_id: SolverId,
    solver_instance: object,
    spec: EnvSpec,
    manifest: EnvManifest,
    band_cfg: BandConfig,
    train_seed: int,
    eval_seeds: tuple[int, ...],
    random_baseline_normalized: float,
    per_episode_returns_random: tuple[float, ...],
) -> PerSolverResult:
    """Train solver_instance and evaluate it; return a PerSolverResult.

    The caller must catch NotImplementedError to detect solver incompatibility:

        try:
            result = _run_one_solver(...)
        except NotImplementedError:
            continue  # solver not applicable to this env

    All other exceptions propagate to the caller unchanged. A ValueError or
    RuntimeError from the env mid-train is a genuine failure, not a
    compatibility signal, and must fail the check.

    Parameters
    ----------
    solver_id:
        SolverId enum value for this solver (written into PerSolverResult).
    solver_instance:
        The Solver object from SOLVER_REGISTRY. Typed as object to avoid
        importing the Protocol; duck-typed via train/evaluate calls.
    spec:
        EnvSpec for loading fresh env instances.
    manifest:
        Validated environment manifest. Reward bounds are read for normalization.
    band_cfg:
        BandConfig controlling training_budget, eval_episodes, threshold.
    train_seed:
        Seed passed to solver.train().
    eval_seeds:
        Seeds tuple. eval_seeds[0] is passed to solver.evaluate() and
        eval_seeds[:band_cfg.eval_episodes] to the random baseline (already
        computed by caller).
    random_baseline_normalized:
        Precomputed normalized random-policy return (env-property).
    per_episode_returns_random:
        Precomputed per-episode random returns (env-property).

    Returns
    -------
    PerSolverResult

    Raises
    ------
    NotImplementedError
        If the solver signals incompatibility with this env via train().
    """
    from praxis.solver._protocol import Solver  # local to avoid circular-import risk

    # We use duck-typed calls; cast to Solver protocol type for mypy only.
    typed_solver: Solver = solver_instance  # type: ignore[assignment]

    bounds = manifest.declared_reward_bounds
    span = bounds.max_per_episode - bounds.min_per_episode

    def _normalize(raw: float) -> float:
        norm = (raw - bounds.min_per_episode) / span
        return norm if norm >= 0.0 else 0.0

    # Train -- NotImplementedError propagates to caller (incompatibility signal).
    env_train = _load_env(spec)
    try:
        state = typed_solver.train(env_train, train_seed, band_cfg.training_budget)
    finally:
        try:
            env_train.close()
        except Exception:
            pass

    # Evaluate trained policy.
    env_eval = _load_env(spec)
    try:
        eval_result = typed_solver.evaluate(
            env_eval, state, eval_seeds[0], band_cfg.eval_episodes
        )
    finally:
        try:
            env_eval.close()
        except Exception:
            pass

    raw_mean_solver = float(eval_result.mean_episodic_return)
    normalized_solver = _normalize(raw_mean_solver)

    solver_pass = normalized_solver >= band_cfg.threshold_normalized
    random_fail = random_baseline_normalized >= band_cfg.threshold_normalized

    per_solver_reason: PerSolverFailureReason | None
    if solver_pass and not random_fail:
        per_solver_passed: bool = True
        per_solver_reason = None
    elif not solver_pass and random_fail:
        per_solver_passed = False
        per_solver_reason = "both"
    elif not solver_pass:
        per_solver_passed = False
        per_solver_reason = "solver_below_threshold"
    else:  # solver_pass and random_fail
        per_solver_passed = False
        per_solver_reason = "trivial_random_baseline"

    trivial_warning = random_fail

    return PerSolverResult(
        solver_id=solver_id,
        passed=per_solver_passed,
        failure_reason=per_solver_reason,
        raw_mean_return=raw_mean_solver,
        normalized_mean_return=normalized_solver,
        threshold_normalized=band_cfg.threshold_normalized,
        random_baseline_normalized=random_baseline_normalized,
        trivial_random_warning=trivial_warning,
        per_episode_returns_solver=eval_result.per_episode_returns,
        per_episode_returns_random=per_episode_returns_random,
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def _aggregate_report(
    manifest: EnvManifest,
    band_cfg: BandConfig,
    solver_results: dict[SolverId, PerSolverResult],
    random_baseline_normalized: float,
    per_episode_returns_random: tuple[float, ...],
) -> SolverBaselineReport:
    """Conjunctively aggregate per-solver results into SolverBaselineReport.

    Aggregation rules:
    - Zero applicable solvers: passed=False, failure_reason="no_compatible_solver".
    - Any solver failed: passed=False, failure_reason = first failing solver's
      reason (insertion order = SOLVER_REGISTRY insertion order).
    - All passed: passed=True, failure_reason=None.

    Top-level per-band fields (raw_mean_return, normalized_mean_return, etc.)
    reflect the FIRST applicable solver's result. This is a Phase 1 backward-
    compat hack; in Phase 1 there is exactly one solver so it is unambiguous.
    Phase 2 consumers should migrate to reading solver_results directly.
    The reference_solver field is set to the first applicable solver's SolverId.
    """
    if not solver_results:
        # No compatible solver found in SOLVER_REGISTRY for this env.
        aggregate_passed: bool = False
        aggregate_reason: AggregateFailureReason | None = "no_compatible_solver"
        # Top-level fields zeroed; threshold still surfaced so consumers know the bar.
        first_solver_id = SolverId.TABULAR_Q_LEARNING  # fallback for reference_solver field
        raw_mean = 0.0
        norm_mean = 0.0
        trivial_warn = False
        per_ep_solver: tuple[float, ...] = ()
    else:
        # Determine aggregate pass/fail using SOLVER_REGISTRY insertion order.
        # Phase 1 note: only one solver, so this loop exits after one iteration.
        # Phase 2: first failing solver (insertion order) sets the aggregate reason.
        aggregate_reason = None
        for per_solver in solver_results.values():
            if not per_solver.passed and per_solver.failure_reason is not None:
                aggregate_reason = per_solver.failure_reason  # PerSolverFailureReason is a subset
                break

        aggregate_passed = aggregate_reason is None

        # Top-level fields from FIRST applicable solver (Phase 1 backward-compat hack).
        # Consumers should migrate to solver_results in Phase 2.
        first_result = next(iter(solver_results.values()))
        first_solver_id = first_result.solver_id
        raw_mean = first_result.raw_mean_return
        norm_mean = first_result.normalized_mean_return
        trivial_warn = first_result.trivial_random_warning
        per_ep_solver = first_result.per_episode_returns_solver

    return SolverBaselineReport(
        env_id=manifest.env_id,
        passed=aggregate_passed,
        difficulty_band=manifest.difficulty_band,
        reference_solver=first_solver_id,
        training_budget=band_cfg.training_budget,
        eval_episodes=band_cfg.eval_episodes,
        raw_mean_return=raw_mean,
        normalized_mean_return=norm_mean,
        threshold_normalized=band_cfg.threshold_normalized,
        random_baseline_normalized=random_baseline_normalized,
        trivial_random_warning=trivial_warn,
        failure_reason=aggregate_reason,
        per_episode_returns_solver=per_ep_solver,
        per_episode_returns_random=per_episode_returns_random,
        solver_results=solver_results,
    )


# ---------------------------------------------------------------------------
# check_solver_baseline
# ---------------------------------------------------------------------------


def check_solver_baseline(
    manifest: EnvManifest,
    config: SolverBaselineConfig | None = None,
) -> SolverBaselineReport:
    """Iterate SOLVER_REGISTRY, train each applicable solver, conjunctively aggregate.

    F-023 closure: manifest.reference_solver is IGNORED. Solver selection is
    determined entirely by SOLVER_REGISTRY iteration + runtime compatibility.
    Each solver's train() is called; if it raises NotImplementedError the solver
    is skipped (not applicable to this env type). Any other exception propagates
    as a check failure.

    Pass requires: (>=1 applicable solver) AND (all applicable solvers pass).
    An applicable solver passes iff its normalized return clears the band
    threshold AND the random-policy baseline does NOT clear the threshold.

    Phase 1 invariant: SOLVER_REGISTRY contains exactly one solver
    (TABULAR_Q_LEARNING), so the "conjunctive" aggregation degenerates to a
    single-solver check. The Phase 1 fix-pass "no_compatible_solver" failure
    reason should never appear for well-formed gridworld envs.

    Random baseline is env-property (depends on manifest + salt, not on any
    solver). It is computed once outside the solver loop and shared across all
    PerSolverResult records. This avoids N-fold redundant computation in Phase 2.

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
        Structured report with conjunctive pass/fail status, per-solver
        PerSolverResult records, and full diagnostics.
    """
    cfg = config if config is not None else SolverBaselineConfig()
    band_cfg = cfg.band_configs[manifest.difficulty_band]
    spec = spec_from_manifest(manifest)

    train_seed = (
        cfg.override_train_seed
        if cfg.override_train_seed is not None
        else int(derive_validator_seeds(manifest, 1, salt=b"solver_baseline")[0])
    )
    eval_seeds: tuple[int, ...] = (
        cfg.override_eval_seeds
        if cfg.override_eval_seeds is not None
        else derive_validator_seeds(manifest, band_cfg.eval_episodes, salt=b"solver_baseline_eval")
    )

    # Random baseline is env-property; computed ONCE outside the per-solver loop.
    # Each PerSolverResult receives a copy for traceability. Avoids N-fold compute
    # when Phase 2 adds multiple solvers.
    random_baseline_normalized, per_episode_returns_random = _compute_random_baseline(
        spec, manifest, band_cfg, eval_seeds
    )

    solver_results: dict[SolverId, PerSolverResult] = {}
    for solver_id, solver_instance in SOLVER_REGISTRY.items():
        try:
            per_solver = _run_one_solver(
                solver_id=solver_id,
                solver_instance=solver_instance,
                spec=spec,
                manifest=manifest,
                band_cfg=band_cfg,
                train_seed=train_seed,
                eval_seeds=eval_seeds,
                random_baseline_normalized=random_baseline_normalized,
                per_episode_returns_random=per_episode_returns_random,
            )
        except NotImplementedError:
            continue  # solver not applicable to this env; skip cleanly
        solver_results[solver_id] = per_solver

    return _aggregate_report(
        manifest=manifest,
        band_cfg=band_cfg,
        solver_results=solver_results,
        random_baseline_normalized=random_baseline_normalized,
        per_episode_returns_random=per_episode_returns_random,
    )
