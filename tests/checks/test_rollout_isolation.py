"""F-032 / F-003 closure tests for the sys.modules snapshot guard."""

from __future__ import annotations

import sys
import types

import pytest

from praxis.checks._rollout import EnvSpec, _isolated_import_namespace, _load_env


def test_guard_removes_new_modules_added_inside_block() -> None:
    """Modules imported inside the guard MUST be removed on exit."""
    fake_mod_name = "praxis_test_isolation_fake_module"
    sys.modules.pop(fake_mod_name, None)
    assert fake_mod_name not in sys.modules

    with _isolated_import_namespace():
        sys.modules[fake_mod_name] = types.ModuleType(fake_mod_name)
        assert fake_mod_name in sys.modules

    assert fake_mod_name not in sys.modules


def test_guard_restores_overwritten_modules() -> None:
    """Modules overwritten inside the guard MUST be restored to their
    original references on exit."""
    # json is always present in sys.modules after Python startup.
    assert "json" in sys.modules, "json must be pre-loaded for this test"
    original = sys.modules["json"]

    with _isolated_import_namespace():
        sys.modules["json"] = types.ModuleType("not_really_json")
        assert sys.modules["json"] is not original

    assert sys.modules["json"] is original


def test_guard_preserves_validator_dependencies() -> None:
    """Validator's own deps (numpy, gymnasium) MUST survive a guard cycle.
    They were imported BEFORE the snapshot, so they're in snapshot, so
    they survive update()."""
    original_numpy = sys.modules["numpy"]
    original_gym = sys.modules["gymnasium"]

    with _isolated_import_namespace():
        pass

    assert sys.modules["numpy"] is original_numpy
    assert sys.modules["gymnasium"] is original_gym


def test_guard_cleans_up_on_exception() -> None:
    """Exception inside the guard MUST still trigger cleanup (try/finally
    semantics)."""
    fake_mod_name = "praxis_test_isolation_exception_fake"
    sys.modules.pop(fake_mod_name, None)

    with pytest.raises(RuntimeError, match="boom"):
        with _isolated_import_namespace():
            sys.modules[fake_mod_name] = types.ModuleType(fake_mod_name)
            raise RuntimeError("boom")

    assert fake_mod_name not in sys.modules


def test_load_env_does_not_leak_modules_across_calls() -> None:
    """F-032 closure: _load_env MUST NOT leave the creator's module in
    sys.modules after the call returns.

    Force-eject the fixture module first to ensure the import inside
    _load_env is a fresh import (otherwise the test would be tautological).
    """
    target = "tests.checks._isolation_target"
    sys.modules.pop(target, None)

    spec = EnvSpec(
        entry_point="tests.checks._isolation_target:IsolationTarget",
        kwargs={},
        max_episode_steps=10,
    )

    pre_keys = set(sys.modules.keys())
    assert target not in pre_keys

    env = _load_env(spec)
    try:
        post_keys = set(sys.modules.keys())
    finally:
        env.close()

    new_keys = post_keys - pre_keys
    assert new_keys == set(), f"_load_env leaked into sys.modules: {new_keys}"
    assert target not in sys.modules


def test_load_env_propagates_constructor_exceptions() -> None:
    """The guard MUST NOT swallow exceptions from env_cls(**kwargs).
    A creator env that crashes in __init__ surfaces to the validator
    as a normal exception."""
    spec = EnvSpec(
        entry_point="tests.checks._adversarial_envs:CrashOnInit",
        kwargs={},
        max_episode_steps=10,
    )

    with pytest.raises(RuntimeError, match="construction failed"):
        _load_env(spec)
