"""Tests for check_determinism_self_consistency.

Closes RT-001 finding F-001: validator now spot-checks at validator-derived
seeds, not just creator-declared anchor seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_gridworld_manifest import (  # type: ignore[import-not-found]
    build_easy_manifest,
    build_medium_manifest,
    build_hard_manifest,
)

from tests.checks._adversarial_envs import make_adversarial_manifest

from praxis.checks._seeds import derive_validator_seeds
from praxis.checks.determinism import (
    DeterminismConfig,
    DeterminismSelfConsistencyReport,
    check_determinism_self_consistency,
)
from praxis.protocol import RewardBounds


def test_self_consistency_gridworld_easy_passes() -> None:
    """Easy gridworld self-consistency check passes at all 8 derived seeds."""
    manifest = build_easy_manifest()
    report = check_determinism_self_consistency(manifest)
    assert isinstance(report, DeterminismSelfConsistencyReport)
    assert report.passed is True
    assert len(report.seeds_tested) == 8
    assert all(r.matched for r in report.per_seed_results)


def test_self_consistency_gridworld_medium_passes() -> None:
    """Medium gridworld self-consistency check passes at all 8 derived seeds."""
    manifest = build_medium_manifest()
    report = check_determinism_self_consistency(manifest)
    assert report.passed is True
    assert len(report.seeds_tested) == 8
    assert all(r.matched for r in report.per_seed_results)


def test_self_consistency_gridworld_hard_passes() -> None:
    """Hard gridworld self-consistency check passes at all 8 derived seeds."""
    manifest = build_hard_manifest()
    report = check_determinism_self_consistency(manifest)
    assert report.passed is True
    assert len(report.seeds_tested) == 8
    assert all(r.matched for r in report.per_seed_results)


def test_self_consistency_override_seeds() -> None:
    """override_seeds replaces derived seeds; report reflects exact tuple."""
    manifest = build_easy_manifest()
    cfg = DeterminismConfig(override_seeds=(1, 2, 3))
    report = check_determinism_self_consistency(manifest, cfg)
    assert report.seeds_tested == (1, 2, 3)
    assert len(report.per_seed_results) == 3


def test_self_consistency_seed_isolation_from_other_checks() -> None:
    """Self-consistency seeds must be disjoint (in expectation) from
    reward_bounds and reset_correctness seeds for the same manifest."""
    manifest = build_easy_manifest()
    sc_seeds = set(derive_validator_seeds(manifest, 8, b"determinism_self_consistency"))
    rb_seeds = set(derive_validator_seeds(manifest, 8, b"reward_bounds"))
    rc_seeds = set(derive_validator_seeds(manifest, 8, b"reset_correctness"))
    assert len(sc_seeds & rb_seeds) <= 1
    assert len(sc_seeds & rc_seeds) <= 1


def test_self_consistency_seeds_invariant_to_declared_bounds() -> None:
    """Tweaking declared bounds does not shift self-consistency seeds."""
    manifest_a = build_easy_manifest()
    manifest_b = manifest_a.model_copy(update={"declared_reward_bounds": RewardBounds(
        min_per_step=-100.0,
        max_per_step=100.0,
        min_per_episode=-1000.0,
        max_per_episode=1000.0,
    )})
    seeds_a = derive_validator_seeds(manifest_a, 8, b"determinism_self_consistency")
    seeds_b = derive_validator_seeds(manifest_b, 8, b"determinism_self_consistency")
    assert seeds_a == seeds_b


def test_self_consistency_catches_nondeterminism() -> None:
    """An env that is non-deterministic at validator-derived seeds must fail."""
    manifest = make_adversarial_manifest("nondeterministic-reward", "NondeterministicReward")
    report = check_determinism_self_consistency(manifest)
    assert report.passed is False
    # Most or all seeds should have mismatched hashes (reward noise is tiny
    # but non-zero, so hashes will virtually always differ).
    assert sum(1 for r in report.per_seed_results if not r.matched) >= 4  # at least half


def test_self_consistency_report_fields_are_populated() -> None:
    """Report contains env_id, per-seed hash pairs, and matched flags."""
    manifest = build_easy_manifest()
    cfg = DeterminismConfig(override_seeds=(42,))
    report = check_determinism_self_consistency(manifest, cfg)
    assert report.env_id == manifest.env_id
    assert len(report.per_seed_results) == 1
    r = report.per_seed_results[0]
    assert r.seed == 42
    assert len(r.hash_a) == 64
    assert len(r.hash_b) == 64
    assert r.matched is True
    assert r.actual_steps_a == r.actual_steps_b
