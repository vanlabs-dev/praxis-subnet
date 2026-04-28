from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

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
    protocol_version: Literal["0.1.0"]
    env_id: str = Field(pattern=ENV_ID_PATTERN)
    entry_point: str = Field(pattern=ENTRY_POINT_PATTERN)
    difficulty_band: DifficultyBand
    max_episode_steps: int = Field(gt=0)
    declared_reward_bounds: RewardBounds
    anchor_trajectories: list[TrajectoryAnchor] = Field(min_length=4, max_length=32)
    creator_metadata: dict[str, str] = Field(default_factory=dict)

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
