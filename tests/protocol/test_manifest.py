"""Tests for praxis.protocol.manifest (EnvManifest, TrajectoryAnchor)."""

import pytest
from pydantic import ValidationError

from praxis.protocol.manifest import EnvManifest, TrajectoryAnchor
from praxis.protocol.types import ActionPolicyId, DifficultyBand, RewardBounds

# A hex string that matches the 64-char blake2b-256 pattern.
_FAKE_HASH = "a" * 64

_ANCHORS = [
    TrajectoryAnchor(
        seed=i,
        action_policy=ActionPolicyId.SEEDED_RANDOM,
        n_steps=100,
        expected_hash=_FAKE_HASH,
    )
    for i in range(4)
]

_BOUNDS = RewardBounds(
    min_per_step=-1.0,
    max_per_step=1.0,
    min_per_episode=-100.0,
    max_per_episode=100.0,
)


def _valid_manifest() -> dict[str, object]:
    return {
        "protocol_version": "0.2.0",
        "env_id": "test-env-v1",
        "entry_point": "praxis.envs.test_env:TestEnv",
        "difficulty_band": DifficultyBand.EASY,
        "max_episode_steps": 500,
        "declared_reward_bounds": _BOUNDS,
        "anchor_trajectories": _ANCHORS,
        "env_version": "0.1.0",
        "kwargs": {},
    }


def test_valid_manifest_validates() -> None:
    m = EnvManifest(**_valid_manifest())  # type: ignore[arg-type]
    assert m.env_id == "test-env-v1"
    assert m.protocol_version == "0.2.0"
    assert len(m.anchor_trajectories) == 4


def test_creator_metadata_defaults_to_empty_dict() -> None:
    m = EnvManifest(**_valid_manifest())  # type: ignore[arg-type]
    assert m.creator_metadata == {}


def test_bad_hash_format_raises() -> None:
    with pytest.raises(ValidationError):
        TrajectoryAnchor(
            seed=0,
            action_policy=ActionPolicyId.SEEDED_RANDOM,
            n_steps=10,
            expected_hash="ZZZZ",  # not valid hex and wrong length
        )


def test_hash_uppercase_raises() -> None:
    with pytest.raises(ValidationError):
        TrajectoryAnchor(
            seed=0,
            action_policy=ActionPolicyId.SEEDED_RANDOM,
            n_steps=10,
            expected_hash="A" * 64,  # uppercase fails the ^[0-9a-f]{64}$ pattern
        )


def test_duplicate_seed_policy_pair_raises() -> None:
    anchors = [
        TrajectoryAnchor(
            seed=0,
            action_policy=ActionPolicyId.SEEDED_RANDOM,
            n_steps=100,
            expected_hash=_FAKE_HASH,
        )
        for _ in range(4)  # all four have seed=0, same policy -- duplicates
    ]
    data = _valid_manifest()
    data["anchor_trajectories"] = anchors
    with pytest.raises(ValidationError, match="duplicate anchor"):
        EnvManifest(**data)  # type: ignore[arg-type]


def test_too_few_anchors_raises() -> None:
    data = _valid_manifest()
    data["anchor_trajectories"] = _ANCHORS[:2]  # only 2, min is 4
    with pytest.raises(ValidationError):
        EnvManifest(**data)  # type: ignore[arg-type]


def test_protocol_version_mismatch_raises() -> None:
    data = _valid_manifest()
    data["protocol_version"] = "9.9.9"
    with pytest.raises(ValidationError):
        EnvManifest(**data)  # type: ignore[arg-type]


def test_round_trip_json_preserves_all_fields() -> None:
    m = EnvManifest(**_valid_manifest())  # type: ignore[arg-type]
    m2 = EnvManifest.from_json_bytes(m.to_json_bytes())

    assert m2.protocol_version == m.protocol_version
    assert m2.env_id == m.env_id
    assert m2.entry_point == m.entry_point
    assert m2.difficulty_band == m.difficulty_band
    assert m2.max_episode_steps == m.max_episode_steps
    assert m2.declared_reward_bounds == m.declared_reward_bounds
    assert len(m2.anchor_trajectories) == len(m.anchor_trajectories)
    for a1, a2 in zip(m.anchor_trajectories, m2.anchor_trajectories):
        assert a1 == a2
    assert m2.creator_metadata == m.creator_metadata
    assert m2.env_version == m.env_version
    assert m2.kwargs == m.kwargs


def test_reward_bounds_min_ge_max_raises() -> None:
    with pytest.raises(ValidationError):
        RewardBounds(
            min_per_step=1.0,
            max_per_step=0.0,  # max < min
            min_per_episode=-100.0,
            max_per_episode=100.0,
        )


def test_negative_seed_raises() -> None:
    with pytest.raises(ValidationError):
        TrajectoryAnchor(
            seed=-1,
            action_policy=ActionPolicyId.SEEDED_RANDOM,
            n_steps=10,
            expected_hash=_FAKE_HASH,
        )


def test_zero_n_steps_raises() -> None:
    with pytest.raises(ValidationError):
        TrajectoryAnchor(
            seed=0,
            action_policy=ActionPolicyId.SEEDED_RANDOM,
            n_steps=0,
            expected_hash=_FAKE_HASH,
        )


# --- New tests for env_version ---


@pytest.mark.parametrize("version", ["0.1.0", "1.2.3.dev1", "2.0.0a1"])
def test_env_version_pep440_valid(version: str) -> None:
    data = _valid_manifest()
    data["env_version"] = version
    m = EnvManifest(**data)  # type: ignore[arg-type]
    assert m.env_version == version


def test_env_version_malformed_raises() -> None:
    data = _valid_manifest()
    data["env_version"] = "not-a-version"
    with pytest.raises(ValidationError):
        EnvManifest(**data)  # type: ignore[arg-type]


# --- New tests for kwargs ---


def test_kwargs_non_json_serialisable_raises() -> None:
    data = _valid_manifest()
    data["kwargs"] = {"x": object()}  # type: ignore[assignment]
    with pytest.raises(ValidationError, match="JSON-serialisable"):
        EnvManifest(**data)  # type: ignore[arg-type]


# --- New tests for protocol_version ---


def test_protocol_version_010_rejected() -> None:
    data = _valid_manifest()
    data["protocol_version"] = "0.1.0"
    with pytest.raises(ValidationError):
        EnvManifest(**data)  # type: ignore[arg-type]


def test_protocol_version_020_required() -> None:
    data = _valid_manifest()
    data["protocol_version"] = "0.2.0"
    m = EnvManifest(**data)  # type: ignore[arg-type]
    assert m.protocol_version == "0.2.0"
