"""Task-Aware Router (TAR) — routes design intents to generators.

Test contract (test_task_router.py):
- TaskProfile has target_family / data_richness / fto_risk / stage fields
- TaskProfile.to_feature_vector() returns 8-dim list[float]
- TaskAwareRouter.HARD_RULES has low_data / high_fto rules
- TaskAwareRouter.route_with_samples(hciv, profile, total_samples) -> dict[str, int]
- TaskAwareRouter.update_with_feedback(gen_name, hvi_reward) updates oracle_history
- TaskAwareRouter.oracle_history dict
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


GENERATOR_NAMES = [
    "hfm_3d", "fragfm", "lamgen_3d", "crem_3d",
    "mmpt_rag", "evomol_rl", "iclm", "uas",
]


@dataclass
class TaskProfile:
    """Task profile for routing decisions."""
    target_family: str = ""
    stage: str = "hit_finding"
    data_richness: float = 100.0
    fto_risk: float = 0.5
    novelty_demand: float = 0.5
    multi_target: bool = False
    sa_constraint: float = 4.0
    n_samples: int = 100
    prior_weights: dict[str, float] = field(default_factory=dict)

    def to_feature_vector(self) -> list[float]:
        family_map = {"GPCR": 0.2, "kinase": 0.4, "ion_channel": 0.6}
        stage_map = {"hit_finding": 0.0, "lead_opt": 0.5, "refine": 1.0}
        return [
            family_map.get(self.target_family, 0.0),
            min(1.0, math.log10(max(1.0, self.data_richness) + 1) / 5.0),
            float(self.novelty_demand),
            float(self.multi_target),
            float(self.sa_constraint) / 10.0,
            stage_map.get(self.stage, 0.0),
            float(self.fto_risk),
            0.0,
        ]


class TaskAwareRouter(nn.Module):
    """Routes HCIV vectors to appropriate generators based on task profile."""

    GENERATOR_NAMES = GENERATOR_NAMES

    HARD_RULES = {
        "scaffold_hop": {"crem_3d": 0.0},
        "low_data": {
            "condition": lambda p: p.data_richness < 50.0,
            "force_generators": ["iclm", "uas"],
            "boost": 2.0,
        },
        "high_fto": {
            "condition": lambda p: p.fto_risk > 0.5,
            "force_generators": ["mmpt_rag", "evomol_rl"],
            "boost": 3.0,
        },
        "lead_opt": {
            "condition": lambda p: p.stage == "lead_opt",
            "force_generators": ["iclm", "crem_3d"],
            "boost": 1.5,
        },
    }

    def __init__(self, hciv_dim: int = 128, task_dim: int = 8, hidden_dim: int = 32, n_generators: int = 8):
        super().__init__()
        self.hciv_dim = hciv_dim
        self.task_dim = task_dim
        self.n_generators = n_generators

        self.gen_embeddings = nn.Parameter(torch.randn(n_generators, hidden_dim))
        self.projection = nn.Linear(hciv_dim, hidden_dim)
        self.task_projection = nn.Linear(task_dim, hidden_dim)

        self.oracle_history: dict[str, dict[str, float]] = {
            name: {"avg_hvi": 0.0, "n_calls": 0.0}
            for name in self.GENERATOR_NAMES
        }

    def forward(self, hciv: torch.Tensor, profile: TaskProfile) -> dict[str, float]:
        task_vec = torch.zeros(self.task_dim)
        if profile.stage != "hit_finding":
            task_vec[0] = 1.0

        h_proj = self.projection(hciv)
        t_proj = self.task_projection(task_vec)

        combined = h_proj + t_proj
        logits = torch.matmul(self.gen_embeddings, combined)
        weights = F.softmax(logits, dim=0)

        result = {}
        for i, name in enumerate(self.GENERATOR_NAMES[:self.n_generators]):
            result[name] = weights[i].item()

        # Apply stage-based hard rules
        rules = self.HARD_RULES.get(profile.stage, {})
        if isinstance(rules, dict):
            for gen_name, factor in rules.items():
                if gen_name in result and not gen_name.startswith("condition"):
                    result[gen_name] = max(0.0, result[gen_name] * factor)

        # Apply condition-based hard rules
        for rule_name, rule in self.HARD_RULES.items():
            if isinstance(rule, dict) and "condition" in rule:
                if rule["condition"](profile):
                    for gen in rule["force_generators"]:
                        if gen in result:
                            result[gen] *= rule["boost"]

        # Apply oracle history feedback: blend with base weights to overcome random init
        history_has_data = any(h["n_calls"] > 0 for h in self.oracle_history.values())
        if history_has_data:
            # Build history-based distribution
            history_weights = {}
            for name, hist in self.oracle_history.items():
                if name in result:
                    history_weights[name] = 0.1 + max(0.0, hist["avg_hvi"]) * 10.0
            # Normalize history weights
            h_total = sum(history_weights.values())
            if h_total > 0:
                history_weights = {k: v / h_total for k, v in history_weights.items()}
                # Blend: 70% history, 30% original NN
                for name in result:
                    result[name] = 0.7 * history_weights.get(name, 0.0) + 0.3 * result[name]

        # Apply prior weights
        for gen_name, prior_w in profile.prior_weights.items():
            if gen_name in result:
                result[gen_name] = result[gen_name] * 0.7 + prior_w * 0.3

        # Renormalize
        total = sum(result.values())
        if total > 0:
            for k in result:
                result[k] /= total

        return result

    def route_with_samples(
        self,
        hciv: torch.Tensor,
        profile: TaskProfile,
        total_samples: int,
    ) -> dict[str, int]:
        weights = self.forward(hciv, profile)
        names = list(weights.keys())

        # Proportional allocation with minimum 1 per generator
        raw = {name: total_samples * w for name, w in weights.items()}
        allocations = {name: max(1, int(v)) for name, v in raw.items()}

        # Adjust to hit exact total_samples
        for _ in range(total_samples * 2):  # safety bound
            current_sum = sum(allocations.values())
            if current_sum == total_samples:
                break
            diff = total_samples - current_sum
            if diff > 0:
                # Add to generators with highest remaining fractional part
                best = max(names, key=lambda n: raw[n] - allocations[n])
                allocations[best] += 1
            else:
                # Remove from generators with most excess allocation
                candidates = [(n, allocations[n] - raw[n]) for n in names if allocations[n] > 1]
                if not candidates:
                    break
                best = max(candidates, key=lambda x: x[1])[0]
                allocations[best] -= 1

        return allocations

    def update_with_feedback(
        self,
        generator_name: str,
        hvi_reward: float,
    ) -> None:
        if generator_name not in self.oracle_history:
            return
        hist = self.oracle_history[generator_name]
        n = hist["n_calls"]
        hist["avg_hvi"] = hist["avg_hvi"] + (hvi_reward - hist["avg_hvi"]) / (n + 1)
        hist["n_calls"] = n + 1
