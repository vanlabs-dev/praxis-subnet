"""Empirical calibration tests: verify all three gridworld bands clear their thresholds."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from build_gridworld_manifest import build_easy_manifest, build_hard_manifest, build_medium_manifest

from praxis.checks.solver_baseline import check_solver_baseline
from praxis.protocol.types import SolverId
from praxis.solver.registry import SOLVER_REGISTRY


def test_solver_baseline_easy_passes() -> None:
    manifest = build_easy_manifest()
    t0 = time.perf_counter()
    report = check_solver_baseline(manifest)
    elapsed = time.perf_counter() - t0
    print(
        f"\n[EASY] elapsed={elapsed:.2f}s "
        f"raw={report.raw_mean_return:.3f} norm={report.normalized_mean_return:.3f} "
        f"thr={report.threshold_normalized} random_norm={report.random_baseline_normalized:.3f} "
        f"trivial_warn={report.trivial_random_warning}"
    )
    assert report.passed is True, (
        f"EASY failed: norm={report.normalized_mean_return:.3f} < thr={report.threshold_normalized}; "
        f"random_norm={report.random_baseline_normalized:.3f}"
    )
    assert report.failure_reason is None
    assert report.reference_solver == SolverId.TABULAR_Q_LEARNING
    assert report.eval_episodes == 20
    assert SolverId.TABULAR_Q_LEARNING in report.solver_results
    assert len(report.solver_results) == len(SOLVER_REGISTRY)
    assert report.solver_results[SolverId.TABULAR_Q_LEARNING].passed is True
    assert report.solver_results[SolverId.TABULAR_Q_LEARNING].failure_reason == report.failure_reason


def test_solver_baseline_medium_passes() -> None:
    manifest = build_medium_manifest()
    t0 = time.perf_counter()
    report = check_solver_baseline(manifest)
    elapsed = time.perf_counter() - t0
    print(
        f"\n[MEDIUM] elapsed={elapsed:.2f}s "
        f"raw={report.raw_mean_return:.3f} norm={report.normalized_mean_return:.3f} "
        f"thr={report.threshold_normalized} random_norm={report.random_baseline_normalized:.3f} "
        f"trivial_warn={report.trivial_random_warning}"
    )
    assert report.passed is True, (
        f"MEDIUM failed: norm={report.normalized_mean_return:.3f} < thr={report.threshold_normalized}; "
        f"random_norm={report.random_baseline_normalized:.3f}"
    )
    assert report.failure_reason is None
    assert report.reference_solver == SolverId.TABULAR_Q_LEARNING
    assert report.eval_episodes == 20
    assert SolverId.TABULAR_Q_LEARNING in report.solver_results
    assert len(report.solver_results) == len(SOLVER_REGISTRY)
    assert report.solver_results[SolverId.TABULAR_Q_LEARNING].passed is True
    assert report.solver_results[SolverId.TABULAR_Q_LEARNING].failure_reason == report.failure_reason


def test_solver_baseline_hard_passes() -> None:
    manifest = build_hard_manifest()
    t0 = time.perf_counter()
    report = check_solver_baseline(manifest)
    elapsed = time.perf_counter() - t0
    print(
        f"\n[HARD] elapsed={elapsed:.2f}s "
        f"raw={report.raw_mean_return:.3f} norm={report.normalized_mean_return:.3f} "
        f"thr={report.threshold_normalized} random_norm={report.random_baseline_normalized:.3f} "
        f"trivial_warn={report.trivial_random_warning}"
    )
    assert report.passed is True, (
        f"HARD failed: norm={report.normalized_mean_return:.3f} < thr={report.threshold_normalized}; "
        f"random_norm={report.random_baseline_normalized:.3f}"
    )
    assert report.failure_reason is None
    assert report.reference_solver == SolverId.TABULAR_Q_LEARNING
    assert report.eval_episodes == 20
    assert SolverId.TABULAR_Q_LEARNING in report.solver_results
    assert len(report.solver_results) == len(SOLVER_REGISTRY)
    assert report.solver_results[SolverId.TABULAR_Q_LEARNING].passed is True
    assert report.solver_results[SolverId.TABULAR_Q_LEARNING].failure_reason == report.failure_reason
