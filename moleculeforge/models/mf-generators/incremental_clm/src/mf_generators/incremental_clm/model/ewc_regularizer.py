"""Elastic Weight Consolidation (EWC) for continual learning."""

from collections.abc import Mapping

import torch
import torch.nn as nn


class EWCRegularizer:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.fisher_diag: dict[str, torch.Tensor] = {}
        self.optimal_params: dict[str, torch.Tensor] = {}

    def consolidate(
        self,
        task_losses: torch.Tensor,
        *,
        sample_weights: torch.Tensor | None = None,
    ) -> None:
        """Estimate task Fisher information and retain the current optimum."""
        if (
            not torch.is_tensor(task_losses)
            or task_losses.ndim > 1
            or task_losses.numel() == 0
            or not torch.isfinite(task_losses).all()
        ):
            raise ValueError("task_losses must be a finite scalar or one-dimensional tensor")
        task_losses = task_losses.reshape(-1)
        if sample_weights is None:
            weights = torch.ones_like(task_losses)
            task_strength = task_losses.new_tensor(1.0)
        else:
            if (
                not torch.is_tensor(sample_weights)
                or sample_weights.shape != task_losses.shape
                or not torch.isfinite(sample_weights).all()
                or torch.any(sample_weights < 0)
                or sample_weights.sum() <= 0
            ):
                raise ValueError(
                    "sample_weights must be finite non-negative values matching task_losses"
                )
            weights = sample_weights.to(
                device=task_losses.device,
                dtype=task_losses.dtype,
            )
            task_strength = weights.mean()
        weights = weights / weights.sum()
        named_parameters = list(self.model.named_parameters())
        trainable = [parameter for _, parameter in named_parameters if parameter.requires_grad]
        observed_fisher = {
            name: torch.zeros_like(parameter) for name, parameter in named_parameters
        }
        for index, (task_loss, weight) in enumerate(zip(task_losses, weights, strict=True)):
            gradients = torch.autograd.grad(
                task_loss,
                trainable,
                allow_unused=True,
                retain_graph=index + 1 < task_losses.numel(),
            )
            gradient_by_id = {
                id(parameter): gradient
                for parameter, gradient in zip(trainable, gradients, strict=True)
            }
            for name, parameter in named_parameters:
                gradient = gradient_by_id.get(id(parameter))
                if gradient is not None:
                    observed_fisher[name] += (
                        task_strength * weight * gradient.detach().pow(2)
                    )
        fisher: dict[str, torch.Tensor] = {}
        for name, parameter in named_parameters:
            observed = observed_fisher[name]
            previous = self.fisher_diag.get(name)
            if previous is not None:
                observed = observed + previous.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            fisher[name] = observed
        self.fisher_diag = fisher
        self.optimal_params = {
            name: parameter.detach().clone() for name, parameter in named_parameters
        }

    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss to prevent forgetting."""
        try:
            first_parameter = next(self.model.parameters())
        except StopIteration:
            return torch.zeros((), dtype=torch.float32)
        loss = first_parameter.new_zeros(())
        for n, p in self.model.named_parameters():
            if n in self.fisher_diag:
                loss += (self.fisher_diag[n] * (p - self.optimal_params[n]) ** 2).sum()
        return 0.5 * loss

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "fisher_diag": {
                name: value.detach().cpu().clone() for name, value in self.fisher_diag.items()
            },
            "optimal_params": {
                name: value.detach().cpu().clone() for name, value in self.optimal_params.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("EWC state must be a mapping")
        fisher = state.get("fisher_diag")
        optimal = state.get("optimal_params")
        if not isinstance(fisher, Mapping) or not isinstance(optimal, Mapping):
            raise ValueError("EWC state requires fisher_diag and optimal_params")
        parameters = dict(self.model.named_parameters())
        if set(fisher) != set(parameters) or set(optimal) != set(parameters):
            raise ValueError("EWC state parameters do not match the model")
        loaded_fisher: dict[str, torch.Tensor] = {}
        loaded_optimal: dict[str, torch.Tensor] = {}
        for name, parameter in parameters.items():
            fisher_value = fisher[name]
            optimal_value = optimal[name]
            if not torch.is_tensor(fisher_value) or not torch.is_tensor(optimal_value):
                raise ValueError("EWC state values must be tensors")
            if fisher_value.shape != parameter.shape or optimal_value.shape != parameter.shape:
                raise ValueError("EWC state tensor shapes do not match the model")
            if not torch.isfinite(fisher_value).all() or torch.any(fisher_value < 0):
                raise ValueError("EWC Fisher values must be finite and non-negative")
            if not torch.isfinite(optimal_value).all():
                raise ValueError("EWC optimal parameters must be finite")
            loaded_fisher[name] = fisher_value.to(
                device=parameter.device,
                dtype=parameter.dtype,
            ).clone()
            loaded_optimal[name] = optimal_value.to(
                device=parameter.device,
                dtype=parameter.dtype,
            ).clone()
        self.fisher_diag = loaded_fisher
        self.optimal_params = loaded_optimal
