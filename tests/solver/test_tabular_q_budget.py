"""Budget edge-case tests for TabularQLearning."""

from __future__ import annotations

from praxis.solver.tabular_q import TabularQLearning

from tests.solver._toy_env import ToyChainEnv


def test_zero_budget_evaluate_does_not_crash() -> None:
    """Budget=0 -> empty Q-table; evaluate falls back to action 0; no crash."""
    solver = TabularQLearning()
    env = ToyChainEnv()

    state = solver.train(env, seed=0, budget=0)
    assert state.q_table == {}, f"Expected empty Q-table for budget=0, got {state.q_table}"

    # Should not raise -- unseen states fall back to action 0.
    solver.evaluate(env, state, seed=0, n_episodes=3)


def test_tiny_budget_runs() -> None:
    """Budget=50 -> partial training, no exceptions expected."""
    solver = TabularQLearning()
    env = ToyChainEnv()

    state = solver.train(env, seed=7, budget=50)
    solver.evaluate(env, state, seed=7, n_episodes=5)
