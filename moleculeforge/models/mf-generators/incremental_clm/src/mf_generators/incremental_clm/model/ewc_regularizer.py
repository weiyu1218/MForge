"""Elastic Weight Consolidation (EWC) for continual learning."""
import torch
import torch.nn as nn


class EWCRegularizer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.fisher_diag: dict[str, torch.Tensor] = {}
        self.optimal_params: dict[str, torch.Tensor] = {}

    def compute_fisher(self, dataloader, n_samples=100):
        """Compute diagonal Fisher information matrix."""
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()
        for i, batch in enumerate(dataloader):
            if i >= n_samples:
                break
            self.model.zero_grad()
            output = self.model(batch)
            loss = output.sum() if isinstance(output, torch.Tensor) else output[0].sum()
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / n_samples
        self.fisher_diag = fisher
        self.optimal_params = {n: p.data.clone() for n, p in self.model.named_parameters()}

    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss to prevent forgetting."""
        loss = torch.tensor(0.0)
        for n, p in self.model.named_parameters():
            if n in self.fisher_diag:
                loss += (self.fisher_diag[n] * (p - self.optimal_params[n]) ** 2).sum()
        return 0.5 * loss
