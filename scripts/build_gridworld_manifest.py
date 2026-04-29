"""Utility script: build a valid EnvManifest for the three gridworld bands.

Computes real trajectory hashes via rollout() and prints the resulting
manifest JSON to stdout. This is a fixture-builder / CLI utility -- it is
NOT part of the public praxis API.

Usage:
    uv run python scripts/build_gridworld_manifest.py

Output:
    One JSON manifest per gridworld band, printed to stdout.
"""

from __future__ import annotations

import json
import sys

# Ensure src/ is on the path when run as a script outside the installed package.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import praxis.envs  # noqa: F401 -- register gymnasium environments

from praxis.checks.determinism import rollout
from praxis.protocol import (
    ActionPolicyId,
    DifficultyBand,
    EnvManifest,
    RewardBounds,
    TrajectoryAnchor,
)

# (lowercase_env_id, difficulty_band, grid_size, n_steps_per_anchor)
_BANDS: list[tuple[str, DifficultyBand, int, int]] = [
    ("praxisgridworld-easy-v0", DifficultyBand.EASY, 5, 50),
    ("praxisgridworld-medium-v0", DifficultyBand.MEDIUM, 10, 100),
    ("praxisgridworld-hard-v0", DifficultyBand.HARD, 20, 200),
]

_ANCHOR_SEEDS = [1, 2, 3, 4]


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
    n_steps: int,
    seeds: list[int] | None = None,
) -> EnvManifest:
    """Compute hashes and return a valid EnvManifest for the given band."""
    if seeds is None:
        seeds = _ANCHOR_SEEDS

    anchors: list[TrajectoryAnchor] = []
    for seed in seeds:
        result = rollout(
            env_id=env_id,
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
        protocol_version="0.1.0",
        env_id=env_id,
        entry_point="praxis.envs.gridworld:PraxisGridworld",
        difficulty_band=difficulty_band,
        max_episode_steps=4 * grid_size * grid_size,
        declared_reward_bounds=_reward_bounds(grid_size),
        anchor_trajectories=anchors,
    )


if __name__ == "__main__":
    for env_id, band, grid_size, n_steps in _BANDS:
        manifest = build_manifest(
            env_id=env_id,
            difficulty_band=band,
            grid_size=grid_size,
            n_steps=n_steps,
        )
        print(f"# {env_id}")
        print(json.dumps(json.loads(manifest.model_dump_json()), indent=2))
        print()
