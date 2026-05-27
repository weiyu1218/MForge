"""OOD-aware sampling for UAS."""
import torch


class OODAwareSampler:
    def __init__(
        self,
        unfamiliarity_estimator,
        rejection_threshold=0.5,
        candidate_source=None,
    ):
        self.estimator = unfamiliarity_estimator
        self.threshold = rejection_threshold
        self.candidate_source = candidate_source

    def sample(self, n_samples: int, max_attempts=10) -> torch.Tensor:
        if self.candidate_source is None:
            raise RuntimeError("UAS_CANDIDATE_SOURCE is required")
        accepted = []
        attempts = 0
        while len(accepted) < n_samples and attempts < max_attempts:
            candidates = self.candidate_source(n_samples)
            if not isinstance(candidates, torch.Tensor):
                candidates = torch.tensor(candidates, dtype=torch.float32)
            unfamiliarity = self.estimator(candidates)
            mask = unfamiliarity < self.threshold
            accepted.extend(candidates[mask])
            attempts += 1
        if len(accepted) < n_samples:
            raise RuntimeError("UAS candidate source did not produce enough in-domain samples")
        return torch.stack(accepted[:n_samples])
