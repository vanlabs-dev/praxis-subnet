"""Tests for praxis.checks._seeds.derive_validator_seeds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow importing build helpers from scripts/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_gridworld_manifest import build_easy_manifest  # type: ignore[import-not-found]

from praxis.checks._seeds import derive_validator_seeds
from praxis.protocol import (
    ActionPolicyId,
    RewardBounds,
    TrajectoryAnchor,
)

_SALT = b"reward_bounds"
_SALT_B = b"reset_correctness"


def _minimal_anchors(n: int = 4) -> list[TrajectoryAnchor]:
    return [
        TrajectoryAnchor(
            seed=i,
            action_policy=ActionPolicyId.SEEDED_RANDOM,
            n_steps=10,
            expected_hash="a" * 64,
        )
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_reproducibility() -> None:
    """Same manifest + same n + same salt -> same seeds."""
    manifest = build_easy_manifest()
    seeds_a = derive_validator_seeds(manifest, 8, _SALT)
    seeds_b = derive_validator_seeds(manifest, 8, _SALT)
    assert seeds_a == seeds_b


# ---------------------------------------------------------------------------
# Salt isolation
# ---------------------------------------------------------------------------


def test_salt_isolation() -> None:
    """Same manifest + same n + different salt -> different seeds."""
    manifest = build_easy_manifest()
    seeds_a = derive_validator_seeds(manifest, 8, _SALT)
    seeds_b = derive_validator_seeds(manifest, 8, _SALT_B)
    assert seeds_a != seeds_b


# ---------------------------------------------------------------------------
# Env-defining fields change seeds
# ---------------------------------------------------------------------------


def test_env_id_changes_seeds() -> None:
    """Different env_id -> different seeds."""
    manifest = build_easy_manifest()
    other = manifest.model_copy(update={"env_id": "praxis-gridworld-alt"})
    assert derive_validator_seeds(manifest, 8, _SALT) != derive_validator_seeds(other, 8, _SALT)


def test_env_version_changes_seeds() -> None:
    """Different env_version -> different seeds."""
    manifest = build_easy_manifest()
    other = manifest.model_copy(update={"env_version": "0.2.0"})
    assert derive_validator_seeds(manifest, 8, _SALT) != derive_validator_seeds(other, 8, _SALT)


def test_entry_point_changes_seeds() -> None:
    """Different entry_point -> different seeds."""
    manifest = build_easy_manifest()
    other = manifest.model_copy(update={"entry_point": "praxis.envs.gridworld:AltGridworld"})
    assert derive_validator_seeds(manifest, 8, _SALT) != derive_validator_seeds(other, 8, _SALT)


def test_kwargs_change_seeds() -> None:
    """Different kwargs -> different seeds."""
    manifest = build_easy_manifest()
    other = manifest.model_copy(update={"kwargs": {"grid_size": 6}})
    assert derive_validator_seeds(manifest, 8, _SALT) != derive_validator_seeds(other, 8, _SALT)


# ---------------------------------------------------------------------------
# Excluded fields do NOT change seeds (collusion-resistance)
# ---------------------------------------------------------------------------


def test_declared_reward_bounds_do_NOT_change_seeds() -> None:
    """Different declared_reward_bounds -> SAME seeds."""
    manifest = build_easy_manifest()
    loose_bounds = RewardBounds(
        min_per_step=-1.0,
        max_per_step=1.0,
        min_per_episode=-100.0,
        max_per_episode=100.0,
    )
    other = manifest.model_copy(update={"declared_reward_bounds": loose_bounds})
    assert derive_validator_seeds(manifest, 8, _SALT) == derive_validator_seeds(other, 8, _SALT)


def test_anchor_trajectories_do_NOT_change_seeds() -> None:
    """Different anchor_trajectories -> SAME seeds."""
    manifest = build_easy_manifest()
    new_anchors = _minimal_anchors(4)
    other = manifest.model_copy(update={"anchor_trajectories": new_anchors})
    assert derive_validator_seeds(manifest, 8, _SALT) == derive_validator_seeds(other, 8, _SALT)


def test_creator_metadata_does_NOT_change_seeds() -> None:
    """Different creator_metadata -> SAME seeds."""
    manifest = build_easy_manifest()
    other = manifest.model_copy(update={"creator_metadata": {"author": "adversary"}})
    assert derive_validator_seeds(manifest, 8, _SALT) == derive_validator_seeds(other, 8, _SALT)


# ---------------------------------------------------------------------------
# n variants
# ---------------------------------------------------------------------------


def test_n_one_returns_one_seed() -> None:
    manifest = build_easy_manifest()
    seeds = derive_validator_seeds(manifest, 1, _SALT)
    assert len(seeds) == 1


def test_n_eight_returns_eight_seeds() -> None:
    manifest = build_easy_manifest()
    seeds = derive_validator_seeds(manifest, 8, _SALT)
    assert len(seeds) == 8


def test_n_sixteen_returns_sixteen_seeds() -> None:
    """n=16 exercises the multi-block path (two blake2b digests)."""
    manifest = build_easy_manifest()
    seeds = derive_validator_seeds(manifest, 16, _SALT)
    assert len(seeds) == 16


# ---------------------------------------------------------------------------
# Range check
# ---------------------------------------------------------------------------


def test_seeds_in_int64_range() -> None:
    """All derived seeds are in [0, 2**63 - 1]."""
    manifest = build_easy_manifest()
    seeds = derive_validator_seeds(manifest, 16, _SALT)
    for s in seeds:
        assert 0 <= s <= (1 << 63) - 1


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_n_zero_raises() -> None:
    manifest = build_easy_manifest()
    with pytest.raises(ValueError, match="n must be >= 1"):
        derive_validator_seeds(manifest, 0, _SALT)


def test_n_negative_raises() -> None:
    manifest = build_easy_manifest()
    with pytest.raises(ValueError, match="n must be >= 1"):
        derive_validator_seeds(manifest, -1, _SALT)


def test_empty_salt_raises() -> None:
    manifest = build_easy_manifest()
    with pytest.raises(ValueError, match="salt must be non-empty"):
        derive_validator_seeds(manifest, 8, b"")
