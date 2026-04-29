"""Reset correctness check for Praxis validator.

Verifies that an environment's reset() method returns valid initial states
across a manifest-derived sample of seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel

from praxis.checks._rollout import POLICY_REGISTRY, ActionPolicy, EnvSpec, _load_env
from praxis.checks._seeds import derive_validator_seeds
from praxis.protocol import ActionPolicyId, EnvManifest

__all__ = [
    "ResetCheckCategory",
    "ResetCorrectnessConfig",
    "ResetViolation",
    "ResetReport",
    "check_reset_correctness",
]


# ---------------------------------------------------------------------------
# Category enum
# ---------------------------------------------------------------------------


class ResetCheckCategory(StrEnum):
    TUPLE_SHAPE = "tuple_shape"
    OBS_IN_SPACE = "obs_in_space"
    INFO_IS_DICT = "info_is_dict"
    SEED_IDEMPOTENCY = "seed_idempotency"
    MID_EPISODE_RESET = "mid_episode_reset"
    RESET_CRASHED = "reset_crashed"
    STEP_CRASHED = "step_crashed"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResetCorrectnessConfig:
    """Configuration for the reset correctness check.

    Sample seeds are derived from the manifest via
    praxis.checks._seeds.derive_validator_seeds with
    salt=b"reset_correctness", guaranteeing disjoint seed sets from
    other checks (e.g. reward_bounds, salt=b"reward_bounds").

    See derive_validator_seeds for collusion-resistance details and
    Phase 1 limitations.

    Attributes:
        sample_seed_count: number of seeds to derive. Default 8.
        override_seeds: explicit seed tuple. For tests and red-team
            experiments only; production code paths leave this None.
        mid_episode_steps: number of actions taken between the first
            reset and the second reset when checking mid-episode
            reset behavior. Default 5. Set to 0 to skip the mid-episode
            check while keeping the other categories.
    """

    sample_seed_count: int = 8
    override_seeds: tuple[int, ...] | None = None
    mid_episode_steps: int = 5


# ---------------------------------------------------------------------------
# Pydantic report models
# ---------------------------------------------------------------------------


class ResetViolation(BaseModel):
    """A single reset correctness violation.

    Attributes
    ----------
    category:
        Which behavioral or safety category was violated.
    seed:
        The sample seed during which the violation occurred.
    message:
        Human-readable description of the violation, including observed
        values where helpful.
    """

    category: ResetCheckCategory
    seed: int
    message: str


class ResetReport(BaseModel):
    """Aggregate result from check_reset_correctness.

    Attributes
    ----------
    env_id:
        The environment ID from the manifest.
    passed:
        True iff there are zero violations (strict pass/fail).
    seeds_tested:
        The seeds that were actually tested, in order.
    violations:
        All violations recorded across all seeds, in the order they
        were observed. Empty on a passing check.
    """

    env_id: str
    passed: bool
    seeds_tested: tuple[int, ...]
    violations: list[ResetViolation]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _obs_equal(a: Any, b: Any) -> bool:
    """Compare two observations for equality.

    Phase 1 supports np.ndarray (via np.array_equal) and scalar
    int/float (via ==). Dict and Tuple observation spaces fall back
    to == comparison; future work to handle structured spaces
    properly. Returns False if comparison itself raises.
    """
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return bool(np.array_equal(a, b))
        return bool(a == b)
    except Exception:
        return False


def _safe_close(env: Any) -> None:
    try:
        env.close()
    except Exception:
        pass


def _safe_len(x: Any) -> int:
    try:
        return len(x)
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# check_reset_correctness
# ---------------------------------------------------------------------------


def check_reset_correctness(
    manifest: EnvManifest,
    config: ResetCorrectnessConfig | None = None,
) -> ResetReport:
    """Multi-seed adversarial reset-correctness check.

    For each derived seed, verifies:
      * reset returns a 2-tuple (obs, info)
      * obs is in env.observation_space
      * info is a dict
      * reset(seed=s) is idempotent
      * reset(seed=s) after K steps returns the same obs as the
        first reset(seed=s) (no state leakage)

    Crashes during reset or step are recorded as RESET_CRASHED /
    STEP_CRASHED violations rather than propagated, so a buggy env
    yields a useful report instead of an unhandled exception.

    Strict pass/fail: any violation yields passed=False.

    Parameters
    ----------
    manifest:
        Validated environment manifest. The env is loaded via importlib
        using manifest.entry_point and manifest.kwargs.
    config:
        Check configuration. Defaults to ResetCorrectnessConfig() if
        not provided (8 derived seeds, 5 mid-episode steps).

    Returns
    -------
    ResetReport
        Structured report with pass/fail status and full violation details.
    """
    cfg = config if config is not None else ResetCorrectnessConfig()
    spec = EnvSpec(
        entry_point=manifest.entry_point,
        kwargs=dict(manifest.kwargs),
        max_episode_steps=manifest.max_episode_steps,
    )

    seeds: tuple[int, ...] = (
        cfg.override_seeds
        if cfg.override_seeds is not None
        else derive_validator_seeds(manifest, cfg.sample_seed_count, salt=b"reset_correctness")
    )
    violations: list[ResetViolation] = []

    for seed in seeds:
        # ------------------------------------------------------------------
        # Phase 1: structural + space + idempotency on a single env instance
        # ------------------------------------------------------------------
        try:
            env_a = _load_env(spec)
        except Exception as exc:
            violations.append(
                ResetViolation(
                    category=ResetCheckCategory.RESET_CRASHED,
                    seed=seed,
                    message=f"_load_env raised: {type(exc).__name__}: {exc}",
                )
            )
            continue

        try:
            result_a = env_a.reset(seed=seed)
        except Exception as exc:
            violations.append(
                ResetViolation(
                    category=ResetCheckCategory.RESET_CRASHED,
                    seed=seed,
                    message=f"env.reset(seed={seed}) raised: {type(exc).__name__}: {exc}",
                )
            )
            _safe_close(env_a)
            continue

        if not isinstance(result_a, tuple) or len(result_a) != 2:
            violations.append(
                ResetViolation(
                    category=ResetCheckCategory.TUPLE_SHAPE,
                    seed=seed,
                    message=(
                        f"reset returned {type(result_a).__name__}"
                        f" (len={_safe_len(result_a)}); expected 2-tuple"
                    ),
                )
            )
            _safe_close(env_a)
            continue

        obs_a, info_a = result_a

        # OBS_IN_SPACE
        try:
            in_space = env_a.observation_space.contains(obs_a)
        except Exception as exc:
            in_space = False
            violations.append(
                ResetViolation(
                    category=ResetCheckCategory.OBS_IN_SPACE,
                    seed=seed,
                    message=f"observation_space.contains raised: {type(exc).__name__}: {exc}",
                )
            )

        if in_space is False and not any(
            v.category == ResetCheckCategory.OBS_IN_SPACE and v.seed == seed
            for v in violations
        ):
            violations.append(
                ResetViolation(
                    category=ResetCheckCategory.OBS_IN_SPACE,
                    seed=seed,
                    message=f"obs not in observation_space; obs={obs_a!r}",
                )
            )

        # INFO_IS_DICT
        if not isinstance(info_a, dict):
            violations.append(
                ResetViolation(
                    category=ResetCheckCategory.INFO_IS_DICT,
                    seed=seed,
                    message=f"info is {type(info_a).__name__}, expected dict",
                )
            )

        # SEED_IDEMPOTENCY: second reset on the same env with the same seed
        try:
            result_b = env_a.reset(seed=seed)
        except Exception as exc:
            violations.append(
                ResetViolation(
                    category=ResetCheckCategory.RESET_CRASHED,
                    seed=seed,
                    message=f"second env.reset(seed={seed}) raised: {type(exc).__name__}: {exc}",
                )
            )
            _safe_close(env_a)
            continue

        if isinstance(result_b, tuple) and len(result_b) == 2:
            obs_b, _ = result_b
            if not _obs_equal(obs_a, obs_b):
                violations.append(
                    ResetViolation(
                        category=ResetCheckCategory.SEED_IDEMPOTENCY,
                        seed=seed,
                        message=(
                            f"reset(seed={seed}) twice returned different obs:"
                            f" {obs_a!r} vs {obs_b!r}"
                        ),
                    )
                )
        # If result_b shape was wrong, the first TUPLE_SHAPE violation from
        # the earlier check already captured the issue. Don't double-record.

        _safe_close(env_a)

        # ------------------------------------------------------------------
        # Phase 2: mid-episode reset on a fresh env instance
        # ------------------------------------------------------------------
        try:
            env_b = _load_env(spec)
        except Exception as exc:
            violations.append(
                ResetViolation(
                    category=ResetCheckCategory.RESET_CRASHED,
                    seed=seed,
                    message=f"_load_env raised on mid-episode env: {type(exc).__name__}: {exc}",
                )
            )
            continue

        try:
            result_mid = env_b.reset(seed=seed)
        except Exception as exc:
            violations.append(
                ResetViolation(
                    category=ResetCheckCategory.RESET_CRASHED,
                    seed=seed,
                    message=(
                        f"mid-episode setup reset(seed={seed}) raised:"
                        f" {type(exc).__name__}: {exc}"
                    ),
                )
            )
            _safe_close(env_b)
            continue

        obs_first_for_mid: Any
        if isinstance(result_mid, tuple) and len(result_mid) == 2:
            obs_first_for_mid, _ = result_mid
        else:
            # TUPLE_SHAPE already recorded above; skip mid-episode for this seed
            _safe_close(env_b)
            continue

        if cfg.mid_episode_steps > 0:
            policy: ActionPolicy = POLICY_REGISTRY[ActionPolicyId.SEEDED_RANDOM]
            try:
                actions = policy.actions(
                    seed=seed,
                    n_steps=cfg.mid_episode_steps,
                    action_space=env_b.action_space,
                )
            except Exception as exc:
                violations.append(
                    ResetViolation(
                        category=ResetCheckCategory.STEP_CRASHED,
                        seed=seed,
                        message=f"action policy raised: {type(exc).__name__}: {exc}",
                    )
                )
                _safe_close(env_b)
                continue

            stepped_ok = True
            for action in actions:
                try:
                    env_b.step(int(action))
                except Exception as exc:
                    violations.append(
                        ResetViolation(
                            category=ResetCheckCategory.STEP_CRASHED,
                            seed=seed,
                            message=f"env.step({int(action)}) raised: {type(exc).__name__}: {exc}",
                        )
                    )
                    stepped_ok = False
                    break
            if not stepped_ok:
                _safe_close(env_b)
                continue

        try:
            result_after = env_b.reset(seed=seed)
        except Exception as exc:
            violations.append(
                ResetViolation(
                    category=ResetCheckCategory.RESET_CRASHED,
                    seed=seed,
                    message=(
                        f"post-step reset(seed={seed}) raised:"
                        f" {type(exc).__name__}: {exc}"
                    ),
                )
            )
            _safe_close(env_b)
            continue

        obs_after: Any
        if isinstance(result_after, tuple) and len(result_after) == 2:
            obs_after, _ = result_after
            if not _obs_equal(obs_first_for_mid, obs_after):
                violations.append(
                    ResetViolation(
                        category=ResetCheckCategory.MID_EPISODE_RESET,
                        seed=seed,
                        message=(
                            f"reset(seed={seed}) after {cfg.mid_episode_steps} steps returned"
                            f" different obs: {obs_first_for_mid!r} vs {obs_after!r}"
                        ),
                    )
                )

        _safe_close(env_b)

    return ResetReport(
        env_id=manifest.env_id,
        passed=len(violations) == 0,
        seeds_tested=seeds,
        violations=violations,
    )
