"""F-004 PoC: hash_infos toggle exposes info-channel side effects.

When an env leaks walltime / pid / global state into info dicts:
    hash_infos=False (default): infos excluded; hashes match; check passes.
    hash_infos=True (paranoid): infos included; hashes differ; check fails.

This test IS the F-004 closure proof.
"""

from __future__ import annotations

from tests.checks._adversarial_envs import make_adversarial_manifest

from praxis.checks.determinism import (
    DeterminismConfig,
    check_determinism_self_consistency,
)


def test_hash_infos_default_false_hides_info_leak() -> None:
    """Default hash_infos=False: leaky info env passes self-consistency."""
    manifest = make_adversarial_manifest("leaky-info", "LeakyInfoEnv")
    report = check_determinism_self_consistency(manifest)
    assert report.passed is True


def test_hash_infos_true_detects_info_leak() -> None:
    """hash_infos=True: same leaky env fails self-consistency."""
    manifest = make_adversarial_manifest("leaky-info", "LeakyInfoEnv")
    cfg = DeterminismConfig(hash_infos=True)
    report = check_determinism_self_consistency(manifest, cfg)
    assert report.passed is False
    assert any(not r.matched for r in report.per_seed_results)


def test_hash_infos_clean_env_passes_in_paranoid_mode() -> None:
    """An env with empty info dicts still passes when hash_infos=True.

    trajectory_hash only folds infos into the payload when at least one info
    dict has content, so an env with no info fields is unaffected by the flag.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from build_gridworld_manifest import build_easy_manifest  # type: ignore[import-not-found]

    manifest = build_easy_manifest()
    cfg = DeterminismConfig(hash_infos=True, override_seeds=(7, 13))
    report = check_determinism_self_consistency(manifest, cfg)
    # Gridworld step() returns empty info -- paranoid mode must not break it
    # (trajectory_hash skips infos key when all infos are empty).
    # Note: gridworld reset returns {"grid_size": N} which is non-empty,
    # so hash_infos=True WILL include that in the hash. But it is
    # deterministic, so the two runs must still match.
    assert report.passed is True
