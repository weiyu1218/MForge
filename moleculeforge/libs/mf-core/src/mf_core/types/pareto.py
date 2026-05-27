"""Pareto archive types."""
from __future__ import annotations

from pydantic import BaseModel, Field

from mf_core.types.molecule import MoleculeModel


class ParetoSolution(BaseModel):
    molecule: MoleculeModel
    objective_values: list[float] = Field(default_factory=list)


class ParetoArchive(BaseModel):
    archive_id: str
    run_id: str
    directions: list[float] = Field(default_factory=list)
    solutions: list[ParetoSolution] = Field(default_factory=list)

    def is_dominated(self, a: list[float], b: list[float]) -> bool:
        for av, bv, d in zip(a, b, self.directions):
            if d >= 0 and av <= bv:
                return False
            if d < 0 and av >= bv:
                return False
        return any(av != bv for av, bv in zip(a, b))

    def insert(self, solution: ParetoSolution) -> bool:
        vals = solution.objective_values
        for existing in self.solutions:
            if self.is_dominated(vals, existing.objective_values):
                return False
        self.solutions = [s for s in self.solutions if not self.is_dominated(s.objective_values, vals)]
        self.solutions.append(solution)
        return True
