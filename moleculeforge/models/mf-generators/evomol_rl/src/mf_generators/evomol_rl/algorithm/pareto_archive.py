"""Pareto archive for storing non-dominated solutions."""
import torch


class ParetoArchive:
    def __init__(self, n_objectives=4):
        self.n_obj = n_objectives
        self.points: list[torch.Tensor] = []
        self.molecules: list[str] = []

    def add(self, obj_values: torch.Tensor, smiles: str) -> bool:
        is_dominated = False
        for p in self.points:
            if (p >= obj_values).all() and (p > obj_values).any():
                is_dominated = True
                break
        if not is_dominated:
            self.points = [p for p in self.points if not ((obj_values >= p).all() and (obj_values > p).any())]
            self.points.append(obj_values.clone())
            self.molecules.append(smiles)
            return True
        return False

    def get_front(self) -> torch.Tensor:
        return torch.stack(self.points) if self.points else torch.zeros(0, self.n_obj)
