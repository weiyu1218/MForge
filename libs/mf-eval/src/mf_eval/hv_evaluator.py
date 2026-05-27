"""Two-dimensional hypervolume evaluation utilities."""
from __future__ import annotations

import torch


def filter_non_dominated(points, maximize: bool = True) -> torch.Tensor:
    point_tensor = _as_points(points)
    keep = []
    for i, point in enumerate(point_tensor):
        others = torch.cat([point_tensor[:i], point_tensor[i + 1 :]], dim=0)
        if others.numel() == 0:
            keep.append(True)
            continue
        if maximize:
            dominated = ((others >= point).all(dim=1) & (others > point).any(dim=1)).any()
        else:
            dominated = ((others <= point).all(dim=1) & (others < point).any(dim=1)).any()
        keep.append(not bool(dominated.item()))
    filtered = point_tensor[torch.tensor(keep, dtype=torch.bool, device=point_tensor.device)]
    order = torch.argsort(filtered[:, 0])
    return filtered[order]


def hypervolume_2d(points, reference, maximize: bool = True) -> float:
    point_tensor = filter_non_dominated(points, maximize=maximize)
    reference_tensor = torch.tensor(reference, dtype=torch.float32)
    if point_tensor.shape[1] != 2 or reference_tensor.numel() != 2:
        raise ValueError("hypervolume_2d only supports 2D points")
    if not maximize:
        point_tensor = -point_tensor
        reference_tensor = -reference_tensor
    point_tensor = point_tensor[point_tensor[:, 0].argsort()]
    hv = 0.0
    previous_x = float(reference_tensor[0].item())
    for point in point_tensor:
        x_value = float(point[0].item())
        y_value = float(point[1].item())
        width = max(0.0, x_value - previous_x)
        height = max(0.0, y_value - float(reference_tensor[1].item()))
        hv += width * height
        previous_x = max(previous_x, x_value)
    return hv


def hypervolume_improvement(candidate, front, reference, maximize: bool = True) -> float:
    front_tensor = _as_points(front)
    candidate_tensor = _as_points([candidate])
    before = hypervolume_2d(front_tensor, reference, maximize=maximize)
    after = hypervolume_2d(
        torch.cat([front_tensor, candidate_tensor], dim=0),
        reference,
        maximize=maximize,
    )
    return max(0.0, after - before)


def _as_points(points) -> torch.Tensor:
    tensor = points if isinstance(points, torch.Tensor) else torch.tensor(points, dtype=torch.float32)
    tensor = tensor.float()
    if tensor.ndim != 2:
        raise ValueError("points must be a 2D array")
    if tensor.shape[1] != 2:
        raise ValueError("hypervolume evaluator currently supports only 2D points")
    return tensor
