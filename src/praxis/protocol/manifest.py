import json
from typing import Any, Literal, Self

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field, field_validator, model_validator

from praxis.protocol.types import ActionPolicyId, DifficultyBand, RewardBounds

ENV_ID_PATTERN = r"^[a-z][a-z0-9_-]{2,63}$"
ENTRY_POINT_PATTERN = r"^[\w\.]+:[\w]+$"
HASH_PATTERN = r"^[0-9a-f]{64}$"


class TrajectoryAnchor(BaseModel):
    seed: int = Field(ge=0)
    action_policy: ActionPolicyId
    n_steps: int = Field(gt=0)
    expected_hash: str = Field(pattern=HASH_PATTERN)


class EnvManifest(BaseModel):
    protocol_version: Literal["0.2.0"]
    env_id: str = Field(pattern=ENV_ID_PATTERN)
    entry_point: str = Field(pattern=ENTRY_POINT_PATTERN)
    difficulty_band: DifficultyBand
    max_episode_steps: int = Field(gt=0)
    declared_reward_bounds: RewardBounds
    anchor_trajectories: list[TrajectoryAnchor] = Field(min_length=4, max_length=32)
    creator_metadata: dict[str, str] = Field(default_factory=dict)
    env_version: str
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("env_version", mode="after")
    @classmethod
    def _env_version_must_be_pep440(cls, value: str) -> str:
        try:
            Version(value)
        except InvalidVersion:
            raise ValueError(f"env_version must be a PEP 440 version, got: {value}")
        return value

    @model_validator(mode="after")
    def _kwargs_must_be_json_serialisable(self) -> Self:
        try:
            json.dumps(self.kwargs)
        except TypeError as exc:
            raise ValueError(f"kwargs must be JSON-serialisable: {exc}") from exc
        return self

    @model_validator(mode="after")
    def _anchors_must_be_unique(self) -> Self:
        seen: set[tuple[int, ActionPolicyId]] = set()
        for anchor in self.anchor_trajectories:
            key = (anchor.seed, anchor.action_policy)
            if key in seen:
                raise ValueError(
                    f"duplicate anchor trajectory: seed={anchor.seed}, "
                    f"action_policy={anchor.action_policy.value}"
                )
            seen.add(key)
        return self

    def to_json_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_json_bytes(cls, data: bytes) -> Self:
        return cls.model_validate_json(data)
