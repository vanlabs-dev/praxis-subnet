"""Utility script: build a valid EnvManifest for the three gridworld bands.

Computes real trajectory hashes via rollout() and prints the resulting
manifest JSON to stdout. This is a fixture-builder / CLI utility.

Usage:
    uv run python scripts/build_gridworld_manifest.py

Output:
    One JSON manifest per gridworld band, printed to stdout.

Importable API (used by tests):
    build_easy_manifest()
    build_medium_manifest()
    build_hard_manifest()
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src/ is on the path when run as a script outside the installed package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from praxis.checks.determinism import EnvSpec, rollout
from praxis.protocol import (
    ActionPolicyId,
    DifficultyBand,
    EnvManifest,
    RewardBounds,
    TrajectoryAnchor,
)

_ENTRY_POINT = "praxis.envs.gridworld:PraxisGridworld"
_ENV_VERSION = "0.1.0"
_PROTOCOL_VERSION = "0.2.0"
_ANCHOR_SEEDS = [1, 2, 3, 4]
_N_STEPS = 200


def _reward_bounds(grid_size: int) -> RewardBounds:
    """Derive RewardBounds from the gridworld spec.

    Per-step: [-0.01, 0.99]
    Per-episode min: -0.01 * max_episode_steps  (max_steps = 4 * grid_size^2)
    Per-episode max: 1.0 - 0.01 * 2 * (grid_size - 1)  (shortest path)
    """
    max_steps = 4 * grid_size * grid_size
    return RewardBounds(
        min_per_step=-0.01,
        max_per_step=0.99,
        min_per_episode=-0.01 * max_steps,
        max_per_episode=1.0 - 0.01 * 2 * (grid_size - 1),
    )


def build_manifest(
    env_id: str,
    difficulty_band: DifficultyBand,
    grid_size: int,
    n_steps: int = _N_STEPS,
    seeds: list[int] | None = None,
) -> EnvManifest:
    """Compute hashes and return a valid EnvManifest for the given band.

    Parameters
    ----------
    env_id:
        Slug-form env ID, e.g. ``"praxis-gridworld-easy"``.
    difficulty_band:
        DifficultyBand enum value.
    grid_size:
        Side length of the gridworld.
    n_steps:
        Steps per anchor rollout.
    seeds:
        Seeds to use for anchor generation. Defaults to [1, 2, 3, 4].
    """
    if seeds is None:
        seeds = _ANCHOR_SEEDS

    max_episode_steps = 4 * grid_size * grid_size
    env_spec = EnvSpec(
        entry_point=_ENTRY_POINT,
        kwargs={"grid_size": grid_size},
        max_episode_steps=max_episode_steps,
    )

    anchors: list[TrajectoryAnchor] = []
    for seed in seeds:
        result = rollout(
            env_spec=env_spec,
            seed=seed,
            action_policy=ActionPolicyId.SEEDED_RANDOM,
            n_steps=n_steps,
        )
        anchors.append(
            TrajectoryAnchor(
                seed=seed,
                action_policy=ActionPolicyId.SEEDED_RANDOM,
                n_steps=n_steps,
                expected_hash=result.computed_hash,
            )
        )

    return EnvManifest(
        protocol_version=_PROTOCOL_VERSION,
        env_id=env_id,
        entry_point=_ENTRY_POINT,
        env_version=_ENV_VERSION,
        kwargs={"grid_size": grid_size},
        difficulty_band=difficulty_band,
        max_episode_steps=max_episode_steps,
        declared_reward_bounds=_reward_bounds(grid_size),
        anchor_trajectories=anchors,
    )


def build_easy_manifest() -> EnvManifest:
    """Build a manifest for the easy (5x5) gridworld band."""
    return build_manifest(
        env_id="praxis-gridworld-easy",
        difficulty_band=DifficultyBand.EASY,
        grid_size=5,
    )


def build_medium_manifest() -> EnvManifest:
    """Build a manifest for the medium (10x10) gridworld band."""
    return build_manifest(
        env_id="praxis-gridworld-medium",
        difficulty_band=DifficultyBand.MEDIUM,
        grid_size=10,
    )


def build_hard_manifest() -> EnvManifest:
    """Build a manifest for the hard (20x20) gridworld band."""
    return build_manifest(
        env_id="praxis-gridworld-hard",
        difficulty_band=DifficultyBand.HARD,
        grid_size=20,
    )


if __name__ == "__main__":
    for builder, label in [
        (build_easy_manifest, "praxis-gridworld-easy"),
        (build_medium_manifest, "praxis-gridworld-medium"),
        (build_hard_manifest, "praxis-gridworld-hard"),
    ]:
        manifest = builder()
        print(f"# {label}")
        print(json.dumps(json.loads(manifest.model_dump_json()), indent=2))
        print()
