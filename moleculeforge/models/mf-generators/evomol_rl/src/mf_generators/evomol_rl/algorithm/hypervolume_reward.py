"""Hypervolume Indicator (HVI) based reward for multi-objective RL."""
import torch


def dominated_hypervolume(front: torch.Tensor, ref_point: torch.Tensor) -> torch.Tensor:
    """Compute the dominated hypervolume of a 2D or 3D Pareto front.

    A simple box-decomposition approach: for a minimization problem,
    each point contributes the volume of the hyperrectangle between
    itself and the reference point.
    """
    if front.numel() == 0:
        return torch.tensor(0.0)
    # For a set of points, compute the union of dominated regions.
    # This is a simplified Monte-Carlo style estimate using the ref point.
    gaps = torch.clamp(ref_point - front, min=0.0)
    # Simplified: sum of marginal improvements (approximation)
    hv = gaps.prod(dim=-1).sum()
    return hv


class HypervolumeReward:
    def __init__(self, ref_point: torch.Tensor):
        self.ref_point = ref_point
        self.pareto_front: list[torch.Tensor] = []

    def compute_reward(self, new_point: torch.Tensor) -> float:
        if not self.pareto_front:
            self.pareto_front.append(new_point)
            return 1.0
        current_hv = self._compute_hv()
        candidate_front = self.pareto_front + [new_point]
        # Remove dominated points
        front_tensor = torch.stack(candidate_front)
        is_nondom = self._is_nondominated(front_tensor)
        new_front = front_tensor[is_nondom]
        new_hv = dominated_hypervolume(new_front, self.ref_point)
        improvement = max(0.0, new_hv.item() - current_hv.item())
        if improvement > 1e-6:
            self.pareto_front.append(new_point)
        return improvement

    def _compute_hv(self) -> torch.Tensor:
        if not self.pareto_front:
            return torch.tensor(0.0)
        return dominated_hypervolume(torch.stack(self.pareto_front), self.ref_point)

    @staticmethod
    def _is_nondominated(points: torch.Tensor) -> torch.Tensor:
        n = points.shape[0]
        dominated = torch.zeros(n, dtype=torch.bool)
        for i in range(n):
            for j in range(n):
                if i != j and (points[j] >= points[i]).all() and (points[j] > points[i]).any():
                    dominated[i] = True
                    break
        return ~dominated
