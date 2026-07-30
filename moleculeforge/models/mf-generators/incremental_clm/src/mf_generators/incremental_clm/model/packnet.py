"""PackNet: iterative pruning for continual learning."""

from collections.abc import Iterator, Mapping

import torch
import torch.nn as nn


class PackNet:
    def __init__(self, model: nn.Module, prune_ratio: float = 0.5) -> None:
        if not 0.0 <= float(prune_ratio) < 1.0:
            raise ValueError("prune_ratio must be in [0, 1)")
        self.model = model
        self.prune_ratio = float(prune_ratio)
        self.masks: dict[str, torch.Tensor] = {}

    def prune(self) -> None:
        """Allocate the strongest pretrained weights and free the remainder."""
        if self.masks:
            raise RuntimeError("PackNet masks are already initialized")
        for name, parameter in self._eligible_parameters():
            mask = (
                self._allocation_mask(
                    parameter.detach(),
                    torch.ones_like(parameter, dtype=torch.bool),
                )
                if parameter.ndim >= 2
                else torch.ones_like(parameter, dtype=torch.bool)
            )
            self.masks[name] = mask.to(dtype=parameter.dtype)
            parameter.data *= self.masks[name]

    def apply_mask(self) -> None:
        for name, parameter in self.model.named_parameters():
            if name in self.masks:
                parameter.data *= self.masks[name]

    def capture_allocated_parameters(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if name in self.masks
        }

    def mask_gradients(self) -> None:
        for name, parameter in self.model.named_parameters():
            mask = self.masks.get(name)
            if mask is not None and parameter.grad is not None:
                parameter.grad.mul_(1.0 - mask)

    def restore_allocated_parameters(
        self,
        frozen_parameters: Mapping[str, torch.Tensor],
    ) -> None:
        if set(frozen_parameters) != set(self.masks):
            raise ValueError("PackNet frozen parameters do not match masks")
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                mask = self.masks.get(name)
                if mask is None:
                    continue
                frozen = frozen_parameters[name]
                if frozen.shape != parameter.shape:
                    raise ValueError("PackNet frozen parameter shape does not match model")
                parameter.copy_(
                    torch.where(
                        mask.bool(),
                        frozen.to(device=parameter.device, dtype=parameter.dtype),
                        parameter,
                    )
                )

    def allocate(self) -> None:
        """Allocate high-magnitude free weights to the completed task."""
        with torch.no_grad():
            for name, parameter in self._eligible_parameters():
                previous_mask = self.masks.get(name)
                if previous_mask is None:
                    raise RuntimeError("PackNet masks are not initialized")
                if parameter.ndim < 2:
                    continue
                free = ~previous_mask.bool()
                new_allocations = self._allocation_mask(parameter.detach(), free)
                merged = previous_mask.bool() | new_allocations
                self.masks[name] = merged.to(dtype=parameter.dtype)
                parameter.mul_(self.masks[name])

    def state_dict(self) -> dict[str, object]:
        return {
            "prune_ratio": self.prune_ratio,
            "masks": {name: mask.detach().cpu().clone() for name, mask in self.masks.items()},
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("PackNet state must be a mapping")
        prune_ratio = state.get("prune_ratio")
        if (
            isinstance(prune_ratio, bool)
            or not isinstance(prune_ratio, int | float)
            or float(prune_ratio) != self.prune_ratio
        ):
            raise ValueError("PackNet state prune_ratio does not match")
        masks = state.get("masks")
        if not isinstance(masks, Mapping):
            raise ValueError("PackNet state requires masks")
        eligible = dict(self._eligible_parameters())
        if set(masks) != set(eligible):
            raise ValueError("PackNet state parameters do not match the model")
        loaded: dict[str, torch.Tensor] = {}
        for name, parameter in eligible.items():
            mask = masks[name]
            if not torch.is_tensor(mask) or mask.shape != parameter.shape:
                raise ValueError("PackNet mask shapes do not match the model")
            if not torch.isfinite(mask).all() or not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("PackNet masks must contain only zero and one")
            loaded[name] = mask.to(
                device=parameter.device,
                dtype=parameter.dtype,
            ).clone()
        self.masks = loaded

    def _eligible_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        return (
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )

    def _allocation_mask(
        self,
        values: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        candidate_indices = candidates.reshape(-1).nonzero(as_tuple=False).reshape(-1)
        selected = torch.zeros_like(candidates, dtype=torch.bool).reshape(-1)
        if candidate_indices.numel() == 0:
            return selected.reshape_as(candidates)
        keep_count = max(
            1,
            candidate_indices.numel() - int(candidate_indices.numel() * self.prune_ratio),
        )
        candidate_values = values.abs().reshape(-1)[candidate_indices]
        strongest = torch.topk(candidate_values, k=keep_count).indices
        selected[candidate_indices[strongest]] = True
        return selected.reshape_as(candidates)
