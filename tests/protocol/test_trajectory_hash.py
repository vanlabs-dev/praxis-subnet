"""Tests for praxis.protocol.hashing.trajectory_hash."""

import numpy as np

from praxis.protocol.hashing import trajectory_hash


def _make_trajectory(
    n: int = 4,
) -> tuple[
    list[np.ndarray],
    list[int],
    list[float],
    list[bool],
    list[bool],
    list[dict[str, object]],
]:
    obs = [np.array([float(i)], dtype=np.float32) for i in range(n)]
    actions = list(range(n))
    rewards = [float(i) * 0.1 for i in range(n)]
    terminations = [False] * (n - 1) + [True]
    truncations = [False] * n
    infos: list[dict[str, object]] = [{} for _ in range(n)]
    return obs, actions, rewards, terminations, truncations, infos


def test_identical_inputs_same_hash() -> None:
    obs, actions, rewards, terminations, truncations, infos = _make_trajectory()
    h1 = trajectory_hash(obs, actions, rewards, terminations, truncations, infos)
    h2 = trajectory_hash(obs, actions, rewards, terminations, truncations, infos)
    assert h1 == h2


def test_reordering_steps_changes_hash() -> None:
    obs, actions, rewards, terminations, truncations, infos = _make_trajectory()
    h1 = trajectory_hash(obs, actions, rewards, terminations, truncations, infos)
    # Reverse all sequences to simulate a different ordering.
    h2 = trajectory_hash(
        obs[::-1],
        actions[::-1],
        rewards[::-1],
        terminations[::-1],
        truncations[::-1],
        infos[::-1],
    )
    assert h1 != h2


def test_different_rewards_change_hash() -> None:
    obs, actions, rewards, terminations, truncations, infos = _make_trajectory()
    rewards2 = [r + 1.0 for r in rewards]
    h1 = trajectory_hash(obs, actions, rewards, terminations, truncations, infos)
    h2 = trajectory_hash(obs, actions, rewards2, terminations, truncations, infos)
    assert h1 != h2


def test_empty_infos_same_hash_with_or_without_flag() -> None:
    # All infos are empty dicts -- include_infos should not change the hash.
    obs, actions, rewards, terminations, truncations, infos = _make_trajectory()
    assert all(len(i) == 0 for i in infos)

    h_without = trajectory_hash(
        obs, actions, rewards, terminations, truncations, infos, include_infos=False
    )
    h_with = trajectory_hash(
        obs, actions, rewards, terminations, truncations, infos, include_infos=True
    )
    assert h_without == h_with


def test_nonempty_infos_change_hash_when_included() -> None:
    obs, actions, rewards, terminations, truncations, infos = _make_trajectory()
    infos_with_data = [{"step": i} for i in range(len(infos))]

    h_without = trajectory_hash(
        obs, actions, rewards, terminations, truncations, infos_with_data, include_infos=False
    )
    h_with = trajectory_hash(
        obs, actions, rewards, terminations, truncations, infos_with_data, include_infos=True
    )
    assert h_without != h_with


def test_hash_is_hex_string_of_correct_length() -> None:
    obs, actions, rewards, terminations, truncations, infos = _make_trajectory()
    h = trajectory_hash(obs, actions, rewards, terminations, truncations, infos)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
