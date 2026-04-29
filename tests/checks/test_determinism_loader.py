"""Tests for the _load_env() private loader in the determinism check.

_load_env is a private helper but is imported directly here -- the underscore
is a convention, not a hard barrier, and loader-error-path coverage requires
direct invocation.
"""

from __future__ import annotations

import pytest
from gymnasium.wrappers import TimeLimit

from praxis.checks._rollout import EnvSpec, _load_env
from praxis.envs.gridworld import PraxisGridworld


def test_load_env_returns_time_limit_wrapped_env() -> None:
    """_load_env returns a TimeLimit wrapper whose unwrapped env is PraxisGridworld."""
    spec = EnvSpec(
        entry_point="praxis.envs.gridworld:PraxisGridworld",
        kwargs={"grid_size": 5},
        max_episode_steps=100,
    )
    env = _load_env(spec)
    try:
        assert isinstance(env, TimeLimit)
        assert isinstance(env.unwrapped, PraxisGridworld)
    finally:
        env.close()


def test_load_env_malformed_entry_point_raises_value_error() -> None:
    """entry_point without a colon raises ValueError mentioning 'module.path:ClassName'."""
    spec = EnvSpec(
        entry_point="nocolon",
        kwargs={},
        max_episode_steps=10,
    )
    with pytest.raises(ValueError, match="module.path:ClassName"):
        _load_env(spec)


def test_load_env_missing_module_raises_import_error() -> None:
    """Unimportable module path raises ImportError (or its subclass ModuleNotFoundError)."""
    spec = EnvSpec(
        entry_point="praxis.envs.does_not_exist:Anything",
        kwargs={},
        max_episode_steps=10,
    )
    with pytest.raises(ImportError):
        _load_env(spec)


def test_load_env_missing_attribute_raises_attribute_error() -> None:
    """Valid module but nonexistent class name raises AttributeError."""
    spec = EnvSpec(
        entry_point="praxis.envs.gridworld:DoesNotExist",
        kwargs={},
        max_episode_steps=10,
    )
    with pytest.raises(AttributeError):
        _load_env(spec)


def test_load_env_non_callable_attribute_raises_type_error() -> None:
    """Resolved attribute that is not callable raises TypeError.

    ENV_ID_PATTERN in praxis.protocol.manifest is a module-level string
    constant -- a real non-callable exported by the protocol package.
    """
    spec = EnvSpec(
        entry_point="praxis.protocol.manifest:ENV_ID_PATTERN",
        kwargs={},
        max_episode_steps=10,
    )
    with pytest.raises(TypeError):
        _load_env(spec)
