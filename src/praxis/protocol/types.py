from enum import StrEnum
from typing import Self

from pydantic import BaseModel, model_validator


class DifficultyBand(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ActionPolicyId(StrEnum):
    SEEDED_RANDOM = "seeded_random"


class RewardBounds(BaseModel):
    min_per_step: float
    max_per_step: float
    min_per_episode: float
    max_per_episode: float

    @model_validator(mode="after")
    def _max_must_exceed_min(self) -> Self:
        if self.max_per_step <= self.min_per_step:
            raise ValueError("max_per_step must be strictly greater than min_per_step")
        if self.max_per_episode <= self.min_per_episode:
            raise ValueError("max_per_episode must be strictly greater than min_per_episode")
        return self
