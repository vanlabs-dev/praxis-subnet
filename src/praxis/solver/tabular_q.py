from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from praxis.solver._protocol import EvalResult


@dataclass(frozen=True)
class TabularQConfig:
    learning_rate: float = 0.5
    discount: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000  # linear decay over this many env steps


@dataclass(frozen=True)
class TabularQState:
    """Trained Q-table plus its config and action-space size.

    The dataclass is frozen, but `q_table` itself is a mutable dict; the
    training loop mutates it in place. Treat it as immutable post-training.
    """

    q_table: dict[tuple[int, ...], npt.NDArray[np.float64]]
    config: TabularQConfig
    n_actions: int


def _obs_to_key(obs: Any) -> tuple[int, ...]:
    """Convert observation to a hashable tuple. Phase 1 supports ndarray
    and int (incl. numpy integer scalars). Other types raise
    NotImplementedError.
    """
    if isinstance(obs, np.ndarray):
        return tuple(int(x) for x in obs.flatten().tolist())
    if isinstance(obs, (int, np.integer)):
        return (int(obs),)
    raise NotImplementedError(
        f"TabularQLearning Phase 1 supports ndarray and int observations only; "
        f"got {type(obs).__name__}"
    )


class TabularQLearning:
    """Epsilon-greedy tabular Q-learning. Discrete action space required.

    Implements the Solver protocol structurally. Determinism: given the
    same env + same seed + same budget, train returns an identical
    TabularQState.
    """

    def __init__(self, config: TabularQConfig | None = None) -> None:
        self.config: Final[TabularQConfig] = config or TabularQConfig()

    def train(self, env: gym.Env, seed: int, budget: int) -> TabularQState:  # type: ignore[type-arg]
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise NotImplementedError(
                "TabularQLearning requires a Discrete action space; "
                f"got {type(env.action_space).__name__}"
            )
        n_actions = int(env.action_space.n)
        q_table: dict[tuple[int, ...], npt.NDArray[np.float64]] = {}
        rng = np.random.default_rng(seed)
        cfg = self.config

        obs, _ = env.reset(seed=seed)
        key = _obs_to_key(obs)

        for step_idx in range(budget):
            # Linear epsilon decay; flat at epsilon_end after decay_steps.
            if step_idx >= cfg.epsilon_decay_steps:
                epsilon = cfg.epsilon_end
            else:
                frac = step_idx / cfg.epsilon_decay_steps
                epsilon = cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * frac

            q_values = q_table.get(key)
            if q_values is None:
                q_values = np.zeros(n_actions, dtype=np.float64)
                q_table[key] = q_values

            if rng.random() < epsilon:
                action = int(rng.integers(0, n_actions))
            else:
                action = int(np.argmax(q_values))  # stable: argmax returns first max

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_key = _obs_to_key(next_obs)

            next_q = q_table.get(next_key)
            if next_q is None:
                next_q = np.zeros(n_actions, dtype=np.float64)
                q_table[next_key] = next_q

            target = float(reward)
            if not terminated:
                # Truncation is NOT terminal -- bootstrap as usual.
                target += cfg.discount * float(np.max(next_q))

            q_values[action] += cfg.learning_rate * (target - q_values[action])

            if terminated or truncated:
                # Re-seed per episode so episode diversity is reproducible.
                obs, _ = env.reset(seed=seed + step_idx + 1)
                key = _obs_to_key(obs)
            else:
                key = next_key

        return TabularQState(q_table=q_table, config=cfg, n_actions=n_actions)

    def evaluate(
        self, env: gym.Env, state: Any, seed: int, n_episodes: int  # type: ignore[type-arg]
    ) -> EvalResult:
        if not isinstance(state, TabularQState):
            raise TypeError(
                f"TabularQLearning.evaluate expects TabularQState; "
                f"got {type(state).__name__}"
            )
        per_episode: list[float] = []
        terminated_count = 0
        truncated_count = 0

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0
            while not done:
                key = _obs_to_key(obs)
                q_values = state.q_table.get(key)
                if q_values is None:
                    # Unseen-state fallback: action 0 deterministically.
                    action = 0
                else:
                    action = int(np.argmax(q_values))
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)
                done = bool(terminated or truncated)
                if terminated:
                    terminated_count += 1
                elif truncated:
                    truncated_count += 1
            per_episode.append(ep_return)

        mean_return = float(np.mean(per_episode)) if per_episode else 0.0
        return EvalResult(
            mean_episodic_return=mean_return,
            per_episode_returns=tuple(per_episode),
            terminated_count=terminated_count,
            truncated_count=truncated_count,
        )
