"""Determinism tests for PraxisGridworld.

Two independently created envs, both reset with the same seed, must produce
identical observations and rewards for an identical action sequence.
"""

import numpy as np

from praxis.envs.gridworld import PraxisGridworld

_N_STEPS = 50  # well below default max_episode_steps for N=10 (400)


def test_two_envs_same_seed_identical_rollout() -> None:
    env_a = PraxisGridworld(10)
    env_b = PraxisGridworld(10)

    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    np.testing.assert_array_equal(obs_a, obs_b)

    # Sample actions from env_a and replay on both
    actions: list[int] = [int(env_a.action_space.sample()) for _ in range(_N_STEPS)]

    for step_idx, action in enumerate(actions):
        result_a = env_a.step(action)
        result_b = env_b.step(action)

        obs_a, rew_a, term_a, trunc_a, _ = result_a
        obs_b, rew_b, term_b, trunc_b, _ = result_b

        np.testing.assert_array_equal(
            obs_a, obs_b, err_msg=f"obs mismatch at step {step_idx}"
        )
        assert rew_a == rew_b, f"reward mismatch at step {step_idx}"
        assert term_a == term_b, f"terminated mismatch at step {step_idx}"
        assert trunc_a == trunc_b, f"truncated mismatch at step {step_idx}"

        if term_a or trunc_a:
            # Reset both and keep going
            obs_a, _ = env_a.reset(seed=42)
            obs_b, _ = env_b.reset(seed=42)
            np.testing.assert_array_equal(obs_a, obs_b)

    env_a.close()
    env_b.close()
