"""Hypervolume improvement (HVI) — EvoMol-RL core reward function."""
from __future__ import annotations
import numpy as np


def _filter_non_dominated(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points

    n = len(points)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (points[j] >= points[i]).all() and (points[j] > points[i]).any():
                is_efficient[i] = False
                break
    return points[is_efficient]


def compute_hypervolume_improvement(
    new_point: dict,
    front: list[dict],
    ref: dict,
) -> float:
    keys = list(ref.keys())
    np_new = np.array([new_point[k] for k in keys], dtype=np.float64)
    np_ref = np.array([ref[k] for k in keys], dtype=np.float64)

    if len(front) == 0:
        return 1.0

    np_front = np.array([[p[k] for k in keys] for p in front], dtype=np.float64)

    current_hv = _hypervolume_2d(np_front, np_ref)
    augmented = np.vstack([np_front, np_new.reshape(1, -1)])
    new_front = _filter_non_dominated(augmented)
    new_hv = _hypervolume_2d(new_front, np_ref)

    return float(max(0.0, new_hv - current_hv))


def _hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0

    n_obj = points.shape[-1]
    if n_obj == 2:
        sorted_pts = points[points[:, 0].argsort()]
        hv = 0.0
        prev_y = ref[1]
        for pt in sorted_pts[::-1]:
            x, y = pt[0], pt[1]
            if x > ref[0] and y > ref[1]:
                hv += (x - ref[0]) * (y - prev_y)
                prev_y = max(prev_y, y)
        return hv

    n_samples = 5000
    lb = ref
    ub = points.max(axis=0)
    if (ub <= lb).any():
        return 0.0
    samples = lb + np.random.rand(n_samples, n_obj) * (ub - lb)
    dominated = np.zeros(n_samples, dtype=bool)
    for sol in points:
        dominated |= (sol >= samples).all(axis=-1)
    volume = float((ub - lb).prod())
    return volume * float(dominated.mean())
