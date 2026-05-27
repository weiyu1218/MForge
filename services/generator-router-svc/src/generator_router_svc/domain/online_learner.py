"""TAR (Thompson-Aware Router) online learner for generator selection."""
import torch


class OnlineLearner:
    def __init__(self, n_generators=8):
        self.n_generators = n_generators
        self.weights = torch.ones(n_generators) / n_generators
        self.counts = torch.ones(n_generators)
        self.rewards = torch.zeros(n_generators)

    def update(self, generator_idx: int, reward: float) -> None:
        self.counts[generator_idx] += 1
        n = self.counts[generator_idx]
        self.rewards[generator_idx] += (reward - self.rewards[generator_idx]) / n
        # Thompson sampling weights
        alpha = self.rewards * self.counts + 1
        self.weights = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).sample()
        self.weights /= self.weights.sum()

    def select_generators(self, n_select: int) -> tuple[list[int], list[float]]:
        top_k = self.weights.topk(min(n_select, self.n_generators))
        return top_k.indices.tolist(), (top_k.values / top_k.values.sum()).tolist()
