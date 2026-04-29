from __future__ import annotations

from typing import Final

from praxis.solver._protocol import Solver, SolverId
from praxis.solver.tabular_q import TabularQLearning

SOLVER_REGISTRY: Final[dict[SolverId, Solver]] = {
    SolverId.TABULAR_Q_LEARNING: TabularQLearning(),
}
