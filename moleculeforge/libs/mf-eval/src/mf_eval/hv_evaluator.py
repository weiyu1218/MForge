"""Two-dimensional hypervolume evaluation utilities."""
from __future__ import annotations

import asyncio
import inspect

import torch
from mf_humu.gp.ehvi import ehvi as _ehvi
from mf_humu.manifold.lorentz import LorentzManifold


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


def probability_of_feasibility(
    mu,
    sigma,
    lower_bounds=None,
    upper_bounds=None,
) -> torch.Tensor:
    mu_tensor = _as_constraint_matrix(mu)
    sigma_tensor = torch.clamp(_as_constraint_matrix(sigma), min=1e-8)
    if mu_tensor.shape != sigma_tensor.shape:
        raise ValueError("constraint mu and sigma must have the same shape")
    probabilities = torch.ones(mu_tensor.shape[0], dtype=torch.float32, device=mu_tensor.device)
    if lower_bounds is not None:
        lower = _as_constraint_bounds(lower_bounds, mu_tensor.shape[1], mu_tensor.device)
        lower_probability = 1.0 - _normal_cdf((lower - mu_tensor) / sigma_tensor)
        probabilities = probabilities * lower_probability.prod(dim=1)
    if upper_bounds is not None:
        upper = _as_constraint_bounds(upper_bounds, mu_tensor.shape[1], mu_tensor.device)
        probabilities = probabilities * _normal_cdf((upper - mu_tensor) / sigma_tensor).prod(dim=1)
    return probabilities


def constrained_hypervolume_improvement(
    candidate,
    front,
    reference,
    constraint_mu,
    constraint_sigma,
    lower_bounds=None,
    upper_bounds=None,
    maximize: bool = True,
) -> float:
    hvi = hypervolume_improvement(candidate, front, reference, maximize=maximize)
    pof = probability_of_feasibility(
        [constraint_mu],
        [constraint_sigma],
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )[0]
    return hvi * float(pof.item())


def expected_hypervolume_improvement(
    mu,
    sigma,
    front,
    reference,
    maximize: bool = True,
) -> torch.Tensor:
    mu_tensor = _as_points(mu)
    sigma_tensor = torch.clamp(_as_points(sigma), min=1e-8)
    front_tensor = _as_points(front)
    reference_tensor = _as_reference_point(reference, mu_tensor.device)
    if mu_tensor.shape != sigma_tensor.shape:
        raise ValueError("objective mu and sigma must have the same shape")
    if maximize:
        return _ehvi(
            -mu_tensor,
            sigma_tensor,
            -reference_tensor,
            -front_tensor,
        )
    return _ehvi(
        mu_tensor,
        sigma_tensor,
        reference_tensor,
        front_tensor,
    )


def humu_logmap_tangent_features(
    humu_embeddings,
    *,
    base_embedding,
    curvature: float = 1.0,
) -> torch.Tensor:
    points = _as_feature_matrix(humu_embeddings, name="humu_embeddings")
    base = _as_single_embedding(base_embedding, name="base_embedding")
    if base.shape[1] != points.shape[1]:
        raise ValueError("base_embedding dimension must match HUMU embeddings")
    manifold = LorentzManifold(curvature=curvature)
    expanded_base = base.to(device=points.device).expand_as(points)
    return manifold.logmap(expanded_base, points)


def rank_constrained_hvi_candidates(
    candidates,
    front,
    reference,
    constraint_mu,
    constraint_sigma,
    lower_bounds=None,
    upper_bounds=None,
    maximize: bool = True,
) -> list[dict]:
    candidate_tensor = _as_points(candidates)
    mu_tensor = _as_constraint_matrix(constraint_mu)
    sigma_tensor = _as_constraint_matrix(constraint_sigma)
    if mu_tensor.shape != sigma_tensor.shape:
        raise ValueError("constraint mu and sigma must have the same shape")
    if mu_tensor.shape[0] != candidate_tensor.shape[0]:
        raise ValueError("constraint rows must match candidate rows")

    feasibility = probability_of_feasibility(
        mu_tensor,
        sigma_tensor,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    ranked = []
    for index, candidate in enumerate(candidate_tensor):
        improvement = hypervolume_improvement(
            candidate.tolist(),
            front,
            reference,
            maximize=maximize,
        )
        pof = float(feasibility[index].item())
        ranked.append(
            {
                "candidate_index": index,
                "candidate": [float(item) for item in candidate.tolist()],
                "hypervolume_improvement": improvement,
                "probability_of_feasibility": pof,
                "score": improvement * pof,
            }
        )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def tangent_space_gp_predict(
    train_embeddings,
    train_values,
    candidate_embeddings,
    lengthscale: float = 1.0,
    noise: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_x = _as_feature_matrix(train_embeddings, name="train_embeddings")
    train_y = _as_response_matrix(train_values)
    candidate_x = _as_feature_matrix(candidate_embeddings, name="candidate_embeddings")
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("train embedding rows must match train value rows")
    if train_x.shape[1] != candidate_x.shape[1]:
        raise ValueError("candidate embeddings must have the same dimension as train embeddings")
    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be positive")
    if noise <= 0.0:
        raise ValueError("noise must be positive")

    kernel = _rbf_kernel(train_x, train_x, lengthscale)
    jitter = torch.eye(kernel.shape[0], dtype=kernel.dtype, device=kernel.device) * noise
    train_kernel = kernel + jitter
    cross_kernel = _rbf_kernel(train_x, candidate_x, lengthscale)
    alpha = torch.linalg.solve(train_kernel, train_y)
    mean = cross_kernel.transpose(0, 1).matmul(alpha)
    solved_cross = torch.linalg.solve(train_kernel, cross_kernel)
    candidate_kernel = _rbf_kernel(candidate_x, candidate_x, lengthscale)
    variance = torch.diagonal(
        candidate_kernel - cross_kernel.transpose(0, 1).matmul(solved_cross),
        0,
    ).clamp_min(1e-8)
    sigma = torch.sqrt(variance).unsqueeze(1).expand_as(mean)
    return mean, sigma


def rank_tangent_gp_constrained_hvi_candidates(
    candidate_embeddings,
    observed_embeddings,
    observed_objectives,
    observed_constraints,
    front,
    reference,
    lower_bounds=None,
    upper_bounds=None,
    lengthscale: float = 1.0,
    noise: float = 1e-4,
    maximize: bool = True,
) -> list[dict]:
    objective_mu, objective_sigma = tangent_space_gp_predict(
        observed_embeddings,
        observed_objectives,
        candidate_embeddings,
        lengthscale=lengthscale,
        noise=noise,
    )
    constraint_mu, constraint_sigma = tangent_space_gp_predict(
        observed_embeddings,
        observed_constraints,
        candidate_embeddings,
        lengthscale=lengthscale,
        noise=noise,
    )
    ranked = rank_constrained_hvi_candidates(
        objective_mu,
        front,
        reference,
        constraint_mu,
        constraint_sigma,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        maximize=maximize,
    )
    for item in ranked:
        index = int(item["candidate_index"])
        item["predicted_objective"] = [
            float(value)
            for value in objective_mu[index].tolist()
        ]
        item["predicted_objective_sigma"] = [
            float(value)
            for value in objective_sigma[index].tolist()
        ]
        item["predicted_constraint"] = [
            float(value)
            for value in constraint_mu[index].tolist()
        ]
        item["predicted_constraint_sigma"] = [
            float(value)
            for value in constraint_sigma[index].tolist()
        ]
    return ranked


def rank_tangent_gp_constrained_ehvi_candidates(
    candidate_embeddings,
    observed_embeddings,
    observed_objectives,
    observed_constraints,
    front,
    reference,
    lower_bounds=None,
    upper_bounds=None,
    lengthscale: float = 1.0,
    noise: float = 1e-4,
    maximize: bool = True,
) -> list[dict]:
    objective_mu, objective_sigma = tangent_space_gp_predict(
        observed_embeddings,
        observed_objectives,
        candidate_embeddings,
        lengthscale=lengthscale,
        noise=noise,
    )
    constraint_mu, constraint_sigma = tangent_space_gp_predict(
        observed_embeddings,
        observed_constraints,
        candidate_embeddings,
        lengthscale=lengthscale,
        noise=noise,
    )
    feasibility = probability_of_feasibility(
        constraint_mu,
        constraint_sigma,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    ehvi_scores = expected_hypervolume_improvement(
        objective_mu,
        objective_sigma,
        front,
        reference,
        maximize=maximize,
    )
    ranked = []
    for index in range(objective_mu.shape[0]):
        ehvi_score = float(ehvi_scores[index].item())
        pof = float(feasibility[index].item())
        ranked.append(
            {
                "candidate_index": index,
                "predicted_objective": [
                    float(value)
                    for value in objective_mu[index].tolist()
                ],
                "predicted_objective_sigma": [
                    float(value)
                    for value in objective_sigma[index].tolist()
                ],
                "predicted_constraint": [
                    float(value)
                    for value in constraint_mu[index].tolist()
                ],
                "predicted_constraint_sigma": [
                    float(value)
                    for value in constraint_sigma[index].tolist()
                ],
                "expected_hypervolume_improvement": ehvi_score,
                "probability_of_feasibility": pof,
                "score": ehvi_score * pof,
            }
        )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def rank_humu_logmap_gp_constrained_ehvi_candidates(
    candidate_humu_embeddings,
    observed_humu_embeddings,
    observed_objectives,
    observed_constraints,
    *,
    base_embedding,
    front,
    reference,
    lower_bounds=None,
    upper_bounds=None,
    curvature: float = 1.0,
    lengthscale: float = 1.0,
    noise: float = 1e-4,
    maximize: bool = True,
) -> list[dict]:
    candidate_tangent = humu_logmap_tangent_features(
        candidate_humu_embeddings,
        base_embedding=base_embedding,
        curvature=curvature,
    )
    observed_tangent = humu_logmap_tangent_features(
        observed_humu_embeddings,
        base_embedding=base_embedding,
        curvature=curvature,
    )
    ranked = rank_tangent_gp_constrained_ehvi_candidates(
        candidate_tangent,
        observed_tangent,
        observed_objectives,
        observed_constraints,
        front,
        reference,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        lengthscale=lengthscale,
        noise=noise,
        maximize=maximize,
    )
    for item in ranked:
        index = int(item["candidate_index"])
        item["tangent_embedding"] = [
            float(value)
            for value in candidate_tangent[index].tolist()
        ]
    return ranked


async def async_pcbo_oracle_loop(
    candidate_embeddings,
    observed_embeddings,
    observed_objectives,
    observed_constraints,
    oracle_evaluate,
    *,
    reference,
    lower_bounds,
    upper_bounds,
    batch_size: int,
    n_iterations: int,
    lengthscale: float = 1.0,
    noise: float = 1e-4,
    maximize: bool = True,
) -> dict:
    candidate_tensor = _as_feature_matrix(
        candidate_embeddings,
        name="candidate_embeddings",
    )
    observed_x = _as_feature_matrix(
        observed_embeddings,
        name="observed_embeddings",
    )
    observed_y = _as_response_matrix(observed_objectives)
    observed_c = _as_constraint_matrix(observed_constraints)
    if observed_x.shape[0] != observed_y.shape[0]:
        raise ValueError("observed embedding rows must match observed objective rows")
    if observed_x.shape[0] != observed_c.shape[0]:
        raise ValueError("observed embedding rows must match observed constraint rows")
    if observed_x.shape[1] != candidate_tensor.shape[1]:
        raise ValueError("candidate embeddings must have the same dimension as observed embeddings")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive")

    remaining_indices = list(range(candidate_tensor.shape[0]))
    selected_indices: list[int] = []
    oracle_results: list[dict] = []
    ranked_batches: list[list[dict]] = []
    for _ in range(n_iterations):
        if not remaining_indices:
            break
        remaining_index_tensor = torch.tensor(
            remaining_indices,
            dtype=torch.long,
            device=candidate_tensor.device,
        )
        remaining_candidates = candidate_tensor.index_select(0, remaining_index_tensor)
        ranked = rank_tangent_gp_constrained_hvi_candidates(
            remaining_candidates,
            observed_x,
            observed_y,
            observed_c,
            filter_non_dominated(observed_y, maximize=maximize),
            reference,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            lengthscale=lengthscale,
            noise=noise,
            maximize=maximize,
        )
        batch = []
        requests = []
        for acquisition in ranked[: min(batch_size, len(ranked))]:
            remaining_position = int(acquisition["candidate_index"])
            candidate_index = int(remaining_indices[remaining_position])
            candidate_embedding = [
                float(value)
                for value in candidate_tensor[candidate_index].tolist()
            ]
            mapped_acquisition = dict(acquisition)
            mapped_acquisition["candidate_index"] = candidate_index
            mapped_acquisition["candidate_embedding"] = candidate_embedding
            batch.append(mapped_acquisition)
            requests.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_embedding": candidate_embedding,
                    "acquisition": mapped_acquisition,
                }
            )
        ranked_batches.append(batch)
        results = await asyncio.gather(
            *[_evaluate_pcbo_oracle(oracle_evaluate, request) for request in requests]
        )
        selected_in_batch = [item["candidate_index"] for item in batch]
        selected_indices.extend(selected_in_batch)
        oracle_results.extend(results)
        observed_x = torch.cat(
            [
                observed_x,
                candidate_tensor.index_select(
                    0,
                    torch.tensor(
                        selected_in_batch,
                        dtype=torch.long,
                        device=candidate_tensor.device,
                    ),
                ),
            ],
            dim=0,
        )
        observed_y = torch.cat(
            [observed_y, _oracle_result_matrix(results, "objectives", observed_y.shape[1])],
            dim=0,
        )
        observed_c = torch.cat(
            [observed_c, _oracle_result_matrix(results, "constraints", observed_c.shape[1])],
            dim=0,
        )
        selected_set = set(selected_in_batch)
        remaining_indices = [
            index
            for index in remaining_indices
            if index not in selected_set
        ]

    return {
        "selected_indices": selected_indices,
        "oracle_results": oracle_results,
        "observed_embeddings": observed_x,
        "observed_objectives": observed_y,
        "observed_constraints": observed_c,
        "ranked_batches": ranked_batches,
    }


class PCBOOptimizationScheduler:
    def __init__(
        self,
        *,
        candidate_provider,
        oracle_evaluate,
        reference,
        lower_bounds,
        upper_bounds,
        batch_size: int,
        n_rounds: int,
        lengthscale: float = 1.0,
        noise: float = 1e-4,
        maximize: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if n_rounds <= 0:
            raise ValueError("n_rounds must be positive")
        self.candidate_provider = candidate_provider
        self.oracle_evaluate = oracle_evaluate
        self.reference = reference
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.batch_size = batch_size
        self.n_rounds = n_rounds
        self.lengthscale = lengthscale
        self.noise = noise
        self.maximize = maximize

    async def run(
        self,
        *,
        observed_embeddings,
        observed_objectives,
        observed_constraints,
    ) -> dict:
        observed_x = _as_feature_matrix(
            observed_embeddings,
            name="observed_embeddings",
        )
        observed_y = _as_response_matrix(observed_objectives)
        observed_c = _as_constraint_matrix(observed_constraints)
        rounds = []
        for round_index in range(self.n_rounds):
            candidate_embeddings = await self._candidate_embeddings(
                round_index,
                observed_x,
                observed_y,
                observed_c,
            )
            loop_result = await async_pcbo_oracle_loop(
                candidate_embeddings,
                observed_x,
                observed_y,
                observed_c,
                self._round_oracle(round_index),
                reference=self.reference,
                lower_bounds=self.lower_bounds,
                upper_bounds=self.upper_bounds,
                batch_size=self.batch_size,
                n_iterations=1,
                lengthscale=self.lengthscale,
                noise=self.noise,
                maximize=self.maximize,
            )
            observed_x = loop_result["observed_embeddings"]
            observed_y = loop_result["observed_objectives"]
            observed_c = loop_result["observed_constraints"]
            rounds.append(
                {
                    "round_index": round_index,
                    "candidate_embeddings": candidate_embeddings,
                    "selected_indices": loop_result["selected_indices"],
                    "oracle_results": loop_result["oracle_results"],
                    "ranked_batches": loop_result["ranked_batches"],
                }
            )

        return {
            "rounds": rounds,
            "observed_embeddings": observed_x,
            "observed_objectives": observed_y,
            "observed_constraints": observed_c,
        }

    async def _candidate_embeddings(
        self,
        round_index: int,
        observed_embeddings: torch.Tensor,
        observed_objectives: torch.Tensor,
        observed_constraints: torch.Tensor,
    ) -> torch.Tensor:
        state = {
            "round_index": round_index,
            "observed_embeddings": observed_embeddings,
            "observed_objectives": observed_objectives,
            "observed_constraints": observed_constraints,
        }
        provider = self.candidate_provider
        if hasattr(provider, "propose"):
            candidates = provider.propose(state)
        else:
            candidates = provider(state)
        if inspect.isawaitable(candidates):
            candidates = await candidates
        candidate_tensor = _as_feature_matrix(candidates, name="candidate_embeddings")
        if candidate_tensor.shape[0] == 0:
            raise ValueError("candidate_provider must return at least one candidate")
        return candidate_tensor

    def _round_oracle(self, round_index: int):
        async def evaluate(request: dict) -> dict:
            enriched_request = dict(request)
            enriched_request["round_index"] = round_index
            return await _evaluate_pcbo_oracle(self.oracle_evaluate, enriched_request)

        return evaluate


def _as_points(points) -> torch.Tensor:
    tensor = (
        points
        if isinstance(points, torch.Tensor)
        else torch.tensor(points, dtype=torch.float32)
    )
    tensor = tensor.float()
    if tensor.ndim != 2:
        raise ValueError("points must be a 2D array")
    if tensor.shape[1] != 2:
        raise ValueError("hypervolume evaluator currently supports only 2D points")
    return tensor


def _normal_cdf(values: torch.Tensor) -> torch.Tensor:
    normalizer = torch.sqrt(torch.tensor(2.0, device=values.device))
    return 0.5 * (1.0 + torch.erf(values / normalizer))


def _as_constraint_matrix(values) -> torch.Tensor:
    tensor = (
        values
        if isinstance(values, torch.Tensor)
        else torch.tensor(values, dtype=torch.float32)
    )
    tensor = tensor.float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError("constraint values must be a vector or matrix")
    return tensor


def _as_constraint_bounds(values, n_constraints: int, device) -> torch.Tensor:
    tensor = (
        values
        if isinstance(values, torch.Tensor)
        else torch.tensor(values, dtype=torch.float32)
    )
    tensor = tensor.float().to(device=device)
    if tensor.ndim != 1 or tensor.numel() != n_constraints:
        raise ValueError("constraint bounds must match the number of constraints")
    return tensor.unsqueeze(0)


def _as_reference_point(values, device) -> torch.Tensor:
    tensor = (
        values
        if isinstance(values, torch.Tensor)
        else torch.tensor(values, dtype=torch.float32)
    )
    tensor = tensor.float().to(device=device)
    if tensor.ndim != 1 or tensor.numel() != 2:
        raise ValueError("reference point must contain two objectives")
    return tensor


def _as_single_embedding(values, name: str) -> torch.Tensor:
    tensor = (
        values
        if isinstance(values, torch.Tensor)
        else torch.tensor(values, dtype=torch.float32)
    )
    tensor = tensor.float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[0] != 1:
        raise ValueError(f"{name} must contain exactly one embedding")
    return tensor


def _as_feature_matrix(values, name: str) -> torch.Tensor:
    tensor = (
        values
        if isinstance(values, torch.Tensor)
        else torch.tensor(values, dtype=torch.float32)
    )
    tensor = tensor.float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a vector or matrix")
    return tensor


def _as_response_matrix(values) -> torch.Tensor:
    tensor = (
        values
        if isinstance(values, torch.Tensor)
        else torch.tensor(values, dtype=torch.float32)
    )
    tensor = tensor.float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 2:
        raise ValueError("train values must be a vector or matrix")
    return tensor


def _rbf_kernel(left: torch.Tensor, right: torch.Tensor, lengthscale: float) -> torch.Tensor:
    distances = torch.cdist(left, right, p=2.0).pow(2)
    return torch.exp(-0.5 * distances / float(lengthscale) ** 2)


async def _evaluate_pcbo_oracle(oracle_evaluate, request: dict) -> dict:
    result = oracle_evaluate(request)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, dict):
        raise TypeError("oracle_evaluate must return a dict")
    if "objectives" not in result or "constraints" not in result:
        raise ValueError("oracle result must include objectives and constraints")
    return result


def _oracle_result_matrix(results: list[dict], key: str, expected_width: int) -> torch.Tensor:
    matrix = _as_response_matrix([result[key] for result in results])
    if matrix.shape[1] != expected_width:
        raise ValueError(f"oracle result {key} width must match observed values")
    return matrix
