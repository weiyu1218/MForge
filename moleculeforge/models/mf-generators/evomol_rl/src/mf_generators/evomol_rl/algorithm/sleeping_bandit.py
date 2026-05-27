"""Sleeping Bandit for generator selection in EvoMol-RL."""
import torch
import numpy as np


class SleepingBandit:
    def __init__(self, n_arms=8, exploration_weight=2.0):
        self.n_arms = n_arms
        self.counts = torch.ones(n_arms)
        self.values = torch.zeros(n_arms)
        self.active = torch.ones(n_arms, dtype=torch.bool)
        self.c = exploration_weight

    def select_arm(self) -> int:
        active_arms = self.active.nonzero(as_tuple=True)[0]
        if len(active_arms) == 0:
            return 0
        total = self.counts.sum()
        ucb = self.values + self.c * torch.sqrt(torch.log(total + 1) / self.counts)
        ucb[~self.active] = -float('inf')
        return int(ucb.argmax())

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def set_active(self, arm: int, active: bool) -> None:
        self.active[arm] = active
