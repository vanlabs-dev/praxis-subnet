"""Basic integration tests for check_determinism against PraxisGridworld-Easy."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing build helpers from scripts/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_gridworld_manifest import build_easy_manifest  # type: ignore[import-not-found]

from praxis.checks.determinism import (
    EnvSpec,
    RolloutResult,
    check_determinism,
    rollout,
)
from praxis.protocol import (
    ActionPolicyId,
    TrajectoryAnchor,
)

_N_STEPS = 200
_SEEDS = [1, 2, 3, 4]

# EnvSpec for the easy (5x5) gridworld.
_EASY_SPEC = EnvSpec(
    entry_point="praxis.envs.gridworld:PraxisGridworld",
    kwargs={"grid_size": 5},
    max_episode_steps=100,
)


def _compute_anchors(seeds: list[int], corrupt_index: int | None = None) -> list[TrajectoryAnchor]:
    """Build TrajectoryAnchor list by computing real hashes, optionally corrupting one."""
    anchors: list[TrajectoryAnchor] = []
    for i, seed in enumerate(seeds):
        result: RolloutResult = rollout(
            env_spec=_EASY_SPEC,
            seed=seed,
            action_policy=ActionPolicyId.SEEDED_RANDOM,
            n_steps=_N_STEPS,
        )
        if corrupt_index is not None and i == corrupt_index:
            declared = "0" * 64
        else:
            declared = result.computed_hash
        anchors.append(
            TrajectoryAnchor(
                seed=seed,
                action_policy=ActionPolicyId.SEEDED_RANDOM,
                n_steps=_N_STEPS,
                expected_hash=declared,
            )
        )
    return anchors


def test_all_anchors_pass_with_correct_hashes() -> None:
    """All correct hashes produce passed=True and all anchors matched."""
    manifest = build_easy_manifest()
    report = check_determinism(manifest)

    assert report.passed is True
    assert report.anchor_count == len(_SEEDS)
    assert report.matched_count == len(_SEEDS)
    for anchor_result in report.anchors:
        assert anchor_result.matched is True
        assert anchor_result.declared_hash == anchor_result.computed_hash


def test_one_corrupted_anchor_fails() -> None:
    """One corrupted hash causes passed=False with exactly one mismatch."""
    corrupt_index = 2  # corrupt the third anchor
    anchors = _compute_anchors(_SEEDS, corrupt_index=corrupt_index)
    manifest = build_easy_manifest()
    # Replace anchors with one corrupted entry.
    manifest_with_corrupt = manifest.model_copy(
        update={"anchor_trajectories": anchors}
    )
    report = check_determinism(manifest_with_corrupt)

    assert report.passed is False
    assert report.matched_count == len(_SEEDS) - 1

    mismatched = [a for a in report.anchors if not a.matched]
    assert len(mismatched) == 1
    assert mismatched[0].seed == _SEEDS[corrupt_index]
    assert mismatched[0].declared_hash == "0" * 64

    # All other anchors still matched.
    matched = [a for a in report.anchors if a.matched]
    assert len(matched) == len(_SEEDS) - 1


def test_all_corrupted_anchors_fail() -> None:
    """All corrupted hashes cause passed=False with matched_count=0."""
    real_anchors = _compute_anchors(_SEEDS)
    corrupt_anchors = [
        TrajectoryAnchor(
            seed=anchor.seed,
            action_policy=anchor.action_policy,
            n_steps=anchor.n_steps,
            expected_hash="0" * 64,
        )
        for anchor in real_anchors
    ]
    manifest = build_easy_manifest()
    manifest_with_corrupt = manifest.model_copy(
        update={"anchor_trajectories": corrupt_anchors}
    )
    report = check_determinism(manifest_with_corrupt)

    assert report.passed is False
    assert report.matched_count == 0
    assert report.anchor_count == len(_SEEDS)
    for anchor_result in report.anchors:
        assert anchor_result.matched is False
