"""OOD-aware sampling for UAS."""
import torch


class OODAwareSampler:
    def __init__(
        self,
        unfamiliarity_estimator,
        rejection_threshold=0.5,
        candidate_source=None,
        safety_temperature: float = 1.0,
        min_safety_probability: float = 0.5,
    ):
        self.estimator = unfamiliarity_estimator
        self.threshold = rejection_threshold
        self.candidate_source = candidate_source
        if safety_temperature <= 0.0:
            raise ValueError("UAS safety_temperature must be positive")
        self.safety_temperature = float(safety_temperature)
        self.min_safety_probability = float(min_safety_probability)
        self.last_safety_probabilities = torch.empty(0, dtype=torch.float32)

    def sample(self, n_samples: int, max_attempts=10) -> torch.Tensor:
        if self.candidate_source is None:
            raise RuntimeError("UAS_CANDIDATE_SOURCE is required")
        accepted = []
        accepted_probabilities = []
        attempts = 0
        while len(accepted) < n_samples and attempts < max_attempts:
            candidates = self.candidate_source(n_samples)
            if not isinstance(candidates, torch.Tensor):
                candidates = torch.tensor(candidates, dtype=torch.float32)
            unfamiliarity = self.estimator(candidates)
            safety_probability = _safety_probability(
                unfamiliarity,
                self.threshold,
                self.safety_temperature,
            )
            mask = safety_probability > self.min_safety_probability
            accepted.extend(candidates[mask])
            accepted_probabilities.extend(safety_probability[mask])
            attempts += 1
        if len(accepted) < n_samples:
            raise RuntimeError("UAS candidate source did not produce enough in-domain samples")
        self.last_safety_probabilities = torch.stack(accepted_probabilities[:n_samples])
        return torch.stack(accepted[:n_samples])


def _safety_probability(
    unfamiliarity: torch.Tensor,
    threshold: float,
    temperature: float,
) -> torch.Tensor:
    return torch.sigmoid((float(threshold) - unfamiliarity) / float(temperature))
