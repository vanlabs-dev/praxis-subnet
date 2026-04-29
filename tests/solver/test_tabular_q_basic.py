"""Basic functional tests for TabularQLearning on the 3-state chain MDP."""

from __future__ import annotations

import numpy as np

from praxis.solver.tabular_q import TabularQLearning

from tests.solver._toy_env import ToyChainEnv


def test_tabular_q_solves_chain() -> None:
    """5000 training steps on the 3-state chain at seed=42 -> near-optimal eval."""
    solver = TabularQLearning()
    env = ToyChainEnv()

    state = solver.train(env, seed=42, budget=5000)
    result = solver.evaluate(env, state, seed=42, n_episodes=10)

    assert result.mean_episodic_return >= 0.95, (
        f"Expected mean return >= 0.95 (optimal ~0.98), got {result.mean_episodic_return}"
    )
    assert result.terminated_count == 10, (
        f"Expected all 10 episodes to terminate, got {result.terminated_count}"
    )


def test_tabular_q_train_eval_determinism() -> None:
    """Same training seed + same eval seed -> identical EvalResult."""
    solver_a = TabularQLearning()
    solver_b = TabularQLearning()
    env_a = ToyChainEnv()
    env_b = ToyChainEnv()

    state_a = solver_a.train(env_a, seed=42, budget=5000)
    state_b = solver_b.train(env_b, seed=42, budget=5000)

    result_a = solver_a.evaluate(env_a, state_a, seed=42, n_episodes=10)
    result_b = solver_b.evaluate(env_b, state_b, seed=42, n_episodes=10)

    assert result_a == result_b, (
        f"Expected identical EvalResults for same seed; got {result_a} vs {result_b}"
    )


def test_tabular_q_different_seeds_diverge() -> None:
    """Different training seeds -> different Q-tables (sanity).

    Uses a small budget (200) so the tables haven't converged. Compares key
    sets and at least one shared key's Q-values for inequality.
    """
    solver_1 = TabularQLearning()
    solver_2 = TabularQLearning()
    env_1 = ToyChainEnv()
    env_2 = ToyChainEnv()

    state_1 = solver_1.train(env_1, seed=1, budget=200)
    state_2 = solver_2.train(env_2, seed=2, budget=200)

    # Check that tables differ by comparing all shared keys.
    shared_keys = set(state_1.q_table.keys()) & set(state_2.q_table.keys())
    # At least one shared key must exist for a meaningful comparison (both
    # explore from state 0, so (0,) is always visited).
    assert len(shared_keys) > 0, "No shared keys found -- unexpected for any budget > 0"

    tables_equal = all(
        np.array_equal(state_1.q_table[k], state_2.q_table[k]) for k in shared_keys
    )
    assert not tables_equal, (
        "Q-tables are identical for different seeds -- exploration is not seed-dependent"
    )
