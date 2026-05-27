"""PackNet: iterative pruning for continual learning."""
import torch
import torch.nn as nn


class PackNet:
    def __init__(self, model: nn.Module, prune_ratio=0.5):
        self.model = model
        self.prune_ratio = prune_ratio
        self.masks: dict[str, torch.Tensor] = {}

    def prune(self):
        """Prune the lowest-magnitude weights."""
        for n, p in self.model.named_parameters():
            if len(p.shape) >= 2:
                threshold = torch.quantile(p.data.abs(), self.prune_ratio)
                mask = (p.data.abs() > threshold).float()
                self.masks[n] = mask
                p.data *= mask

    def apply_mask(self):
        for n, p in self.model.named_parameters():
            if n in self.masks:
                p.data *= self.masks[n]
