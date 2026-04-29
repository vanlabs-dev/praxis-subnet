"""Tests for SOLVER_REGISTRY contents and Protocol compliance."""

from __future__ import annotations

from praxis.solver import SOLVER_REGISTRY, Solver, SolverId


def test_registry_has_tabular_q() -> None:
    """SOLVER_REGISTRY contains TABULAR_Q_LEARNING."""
    assert SolverId.TABULAR_Q_LEARNING in SOLVER_REGISTRY


def test_registry_has_exactly_one_entry() -> None:
    """Phase 1 ships exactly one solver."""
    assert len(SOLVER_REGISTRY) == 1


def test_registered_solver_is_protocol_compliant() -> None:
    """Registered TabularQLearning satisfies the Solver Protocol.

    @runtime_checkable Protocol checks attribute existence only -- that is
    the structural typing contract.
    """
    assert isinstance(SOLVER_REGISTRY[SolverId.TABULAR_Q_LEARNING], Solver)
