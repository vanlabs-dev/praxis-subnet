from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import gymnasium as gym
from pydantic import BaseModel

from praxis.protocol.types import SolverId

__all__ = ["EvalResult", "Solver", "SolverId"]


class EvalResult(BaseModel):
    mean_episodic_return: float
    per_episode_returns: tuple[float, ...]
    terminated_count: int
    truncated_count: int


@runtime_checkable
class Solver(Protocol):
    """Reference solver abstraction. Phase 1 ships TabularQLearning;
    Phase 2 will add cleanrl-style PPO under the same interface.

    Implementations must be deterministic given a seed.
    """

    def train(self, env: gym.Env[Any, Any], seed: int, budget: int) -> Any:
        """Train on env for at most `budget` env steps. Return opaque solver
        state (e.g. Q-table for tabular, weights for PPO). Type Any because
        the state shape is solver-specific."""
        ...

    def evaluate(
        self, env: gym.Env[Any, Any], state: Any, seed: int, n_episodes: int
    ) -> EvalResult:
        """Evaluate the trained state on env using a deterministic policy.
        Returns mean episodic return over n_episodes plus per-episode
        diagnostics."""
        ...
