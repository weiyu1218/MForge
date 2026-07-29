"""Task-aware routing and deterministic generator allocation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as functional

GENERATOR_NAMES = [
    "hfm_3d",
    "fragfm",
    "crem_3d",
    "mmpt_rag",
    "iclm",
    "uas",
]


@dataclass
class TaskProfile:
    """Task profile for routing decisions."""

    target_family: str = ""
    stage: str = "hit_finding"
    data_richness: float = 100.0
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
            min(1.0, math.log10(self.data_richness + 1.0) / 5.0),
            float(self.novelty_demand),
            float(self.multi_target),
            float(self.sa_constraint) / 10.0,
            stage_map.get(self.stage, 0.0),
            0.0,
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
        "lead_opt": {
            "condition": lambda p: p.stage == "lead_opt",
            "force_generators": ["iclm", "crem_3d"],
            "boost": 1.5,
        },
    }

    def __init__(
        self,
        hciv_dim: int = 128,
        task_dim: int = 8,
        hidden_dim: int = 32,
        n_generators: int = len(GENERATOR_NAMES),
    ) -> None:
        super().__init__()
        self.hciv_dim = hciv_dim
        self.task_dim = task_dim
        self.n_generators = n_generators

        self.gen_embeddings = nn.Parameter(torch.randn(n_generators, hidden_dim))
        self.projection = nn.Linear(hciv_dim, hidden_dim)
        self.task_projection = nn.Linear(task_dim, hidden_dim)
        self.architecture_logits = nn.Parameter(torch.zeros(n_generators))
        self.register_buffer("policy_logits", torch.zeros(n_generators))

        self.oracle_history: dict[str, dict[str, float]] = {
            name: {"avg_hvi": 0.0, "n_calls": 0.0} for name in self.GENERATOR_NAMES
        }

    def forward(
        self,
        hciv: torch.Tensor,
        profile: TaskProfile,
        *,
        oracle_history: Mapping[str, Mapping[str, float]] | None = None,
    ) -> dict[str, float]:
        task_features = profile.to_feature_vector()
        task_vec = torch.zeros(
            self.task_dim,
            dtype=hciv.dtype,
            device=hciv.device,
        )
        usable = min(self.task_dim, len(task_features))
        if usable:
            task_vec[:usable] = torch.tensor(
                task_features[:usable],
                dtype=hciv.dtype,
                device=hciv.device,
            )

        h_proj = self.projection(hciv)
        t_proj = self.task_projection(task_vec)

        combined = h_proj + t_proj
        logits = torch.matmul(self.gen_embeddings, combined) / math.sqrt(
            self.gen_embeddings.shape[1]
        )
        logits = logits + self.policy_logits[: self.n_generators].to(
            device=logits.device,
            dtype=logits.dtype,
        )
        logits = logits + self.architecture_logits[: self.n_generators].to(
            device=logits.device,
            dtype=logits.dtype,
        )
        weights = functional.softmax(logits, dim=0)

        result = {}
        for i, name in enumerate(self.GENERATOR_NAMES[: self.n_generators]):
            result[name] = weights[i].item()

        # Apply stage-based hard rules
        rules = self.HARD_RULES.get(profile.stage, {})
        if isinstance(rules, dict):
            for gen_name, factor in rules.items():
                if gen_name in result and not gen_name.startswith("condition"):
                    result[gen_name] = max(0.0, result[gen_name] * factor)

        # Apply condition-based hard rules
        for _rule_name, rule in self.HARD_RULES.items():
            if isinstance(rule, dict) and "condition" in rule:
                if rule["condition"](profile):
                    for gen in rule["force_generators"]:
                        if gen in result:
                            result[gen] = result[gen] * rule["boost"] + rule["boost"]

        # Apply oracle history feedback: blend with base weights to overcome random init
        active_history = oracle_history if oracle_history is not None else self.oracle_history
        history_has_data = any(
            float(history.get("n_calls", 0.0)) > 0.0 for history in active_history.values()
        )
        if history_has_data:
            history_weights = {}
            for name in result:
                history = active_history.get(name, {})
                history_weights[name] = 0.1 + max(0.0, float(history.get("avg_hvi", 0.0))) * 10.0
            h_total = sum(history_weights.values())
            if h_total > 0:
                history_weights = {k: v / h_total for k, v in history_weights.items()}
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
        return minimum_one_largest_remainder(weights, total_samples)

    def proxyless_architecture_probabilities(
        self,
        temperature: float = 1.0,
    ) -> dict[str, float]:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        logits = self.architecture_logits[: self.n_generators] / float(temperature)
        probabilities = functional.softmax(logits, dim=0)
        return {
            name: float(probabilities[i].item())
            for i, name in enumerate(self.GENERATOR_NAMES[: self.n_generators])
        }

    def proxyless_expected_cost(
        self,
        generator_costs: dict[str, float],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        clean_costs = _validate_generator_values(
            generator_costs,
            generator_names=self.GENERATOR_NAMES[: self.n_generators],
            value_name="cost",
            non_negative=True,
        )
        logits = self.architecture_logits[: self.n_generators] / float(temperature)
        probabilities = functional.softmax(logits, dim=0)
        costs = torch.tensor(
            [clean_costs[name] for name in self.GENERATOR_NAMES[: self.n_generators]],
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        return torch.sum(probabilities * costs)

    def proxyless_architecture_optimizer_step(
        self,
        *,
        generator_rewards: dict[str, float],
        generator_costs: dict[str, float],
        cost_weight: float,
        learning_rate: float,
        temperature: float = 1.0,
    ) -> dict[str, float]:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if cost_weight < 0.0:
            raise ValueError("cost_weight must be non-negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        generator_names = self.GENERATOR_NAMES[: self.n_generators]
        clean_rewards = _validate_generator_values(
            generator_rewards,
            generator_names=generator_names,
            value_name="reward",
        )
        clean_costs = _validate_generator_values(
            generator_costs,
            generator_names=generator_names,
            value_name="cost",
            non_negative=True,
        )

        active_logits = self.architecture_logits[: self.n_generators]
        probabilities = functional.softmax(active_logits / float(temperature), dim=0)
        rewards = torch.tensor(
            [clean_rewards[name] for name in generator_names],
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        costs = torch.tensor(
            [clean_costs[name] for name in generator_names],
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        utility = rewards - float(cost_weight) * costs
        objective = torch.sum(probabilities * utility)
        loss = -objective
        (gradient,) = torch.autograd.grad(loss, self.architecture_logits)
        with torch.no_grad():
            self.architecture_logits -= float(learning_rate) * gradient
            active = self.architecture_logits[: self.n_generators]
            active -= active.mean()
        return {
            "objective": float(objective.detach().item()),
            "expected_reward": float(torch.sum(probabilities * rewards).detach().item()),
            "expected_cost": float(torch.sum(probabilities * costs).detach().item()),
        }

    def update_with_feedback(
        self,
        generator_name: str,
        hvi_reward: float,
        baseline: float | None = None,
        learning_rate: float = 0.05,
    ) -> None:
        if generator_name not in self.oracle_history:
            return
        hist = self.oracle_history[generator_name]
        n = hist["n_calls"]
        hist["avg_hvi"] = hist["avg_hvi"] + (hvi_reward - hist["avg_hvi"]) / (n + 1)
        hist["n_calls"] = n + 1
        self._apply_reinforce_update(generator_name, hvi_reward, baseline, learning_rate)

    def _apply_reinforce_update(
        self,
        generator_name: str,
        hvi_reward: float,
        baseline: float | None,
        learning_rate: float,
    ) -> None:
        if learning_rate <= 0.0:
            return
        generator_idx = self.GENERATOR_NAMES.index(generator_name)
        if generator_idx >= self.n_generators:
            return
        if baseline is None:
            observed = [
                hist["avg_hvi"] for hist in self.oracle_history.values() if hist["n_calls"] > 0
            ]
            baseline = sum(observed) / len(observed) if observed else 0.0
        advantage = float(hvi_reward) - float(baseline)
        if not math.isfinite(advantage):
            raise ValueError("hvi_reward and baseline must produce a finite advantage")
        with torch.no_grad():
            self.policy_logits[generator_idx] += float(learning_rate) * advantage
            active = self.policy_logits[: self.n_generators]
            active -= active.mean()


class ProxylessSearchScheduler:
    def __init__(
        self,
        *,
        router: TaskAwareRouter,
        generator_costs: dict[str, float],
        cost_weight: float,
        learning_rate: float,
        temperature: float = 1.0,
    ) -> None:
        if cost_weight < 0.0:
            raise ValueError("cost_weight must be non-negative")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.router = router
        self.generator_costs = _validate_generator_values(
            generator_costs,
            generator_names=router.GENERATOR_NAMES[: router.n_generators],
            value_name="cost",
            non_negative=True,
        )
        self.cost_weight = float(cost_weight)
        self.learning_rate = float(learning_rate)
        self.temperature = float(temperature)

    def run(
        self,
        reward_batches_by_dataset: Mapping[str, Sequence[dict[str, float]]],
    ) -> dict[str, object]:
        if not reward_batches_by_dataset:
            raise ValueError("reward_batches_by_dataset must not be empty")
        rounds = []
        for dataset, reward_batches in reward_batches_by_dataset.items():
            if not reward_batches:
                raise ValueError("each dataset must contain at least one reward batch")
            for round_index, rewards in enumerate(reward_batches):
                clean_rewards = self._clean_rewards(rewards)
                step = self.router.proxyless_architecture_optimizer_step(
                    generator_rewards=clean_rewards,
                    generator_costs=self.generator_costs,
                    cost_weight=self.cost_weight,
                    learning_rate=self.learning_rate,
                    temperature=self.temperature,
                )
                for generator_name, reward in clean_rewards.items():
                    self.router.update_with_feedback(generator_name, reward)
                rounds.append(
                    {
                        "dataset": str(dataset),
                        "round_index": round_index,
                        "generator_rewards": clean_rewards,
                        "objective": step["objective"],
                        "expected_reward": step["expected_reward"],
                        "expected_cost": step["expected_cost"],
                    }
                )
        return {
            "rounds": rounds,
            "architecture_probabilities": self.router.proxyless_architecture_probabilities(
                temperature=self.temperature
            ),
        }

    def _clean_rewards(self, rewards: Mapping[str, float]) -> dict[str, float]:
        return _validate_generator_values(
            rewards,
            generator_names=self.router.GENERATOR_NAMES[: self.router.n_generators],
            value_name="reward",
        )


def _validate_generator_values(
    values: Mapping[str, float],
    *,
    generator_names: Sequence[str],
    value_name: str,
    non_negative: bool = False,
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"generator {value_name}s must be an object")
    expected = list(generator_names)
    expected_set = set(expected)
    unknown = [str(name) for name in values if name not in expected_set]
    if unknown:
        raise ValueError(f"unknown generator in {value_name}s: {unknown[0]}")
    missing = [name for name in expected if name not in values]
    if missing:
        raise ValueError(f"missing generator {value_name}: {missing[0]}")
    clean: dict[str, float] = {}
    for name in expected:
        raw_value = values[name]
        if isinstance(raw_value, bool):
            raise ValueError(f"generator {value_name}s must be finite")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"generator {value_name}s must be finite")
        if non_negative and value < 0.0:
            raise ValueError(f"generator {value_name}s must be non-negative")
        clean[name] = value
    return clean


def minimum_one_largest_remainder(
    weights: Mapping[str, float],
    total_samples: int,
) -> dict[str, int]:
    """Allocate an exact sample count with one sample guaranteed per generator."""

    names = list(weights)
    if not names:
        raise ValueError("weights must contain at least one generator")
    if total_samples < len(names):
        raise ValueError("total_samples must be at least the number of generators")

    clean_weights: dict[str, float] = {}
    for name in names:
        value = float(weights[name])
        if not math.isfinite(value):
            raise ValueError("generator weights must be finite")
        if value < 0.0:
            raise ValueError("generator weights must be non-negative")
        clean_weights[name] = value

    weight_sum = sum(clean_weights.values())
    if weight_sum == 0.0:
        normalized = {name: 1.0 / len(names) for name in names}
    else:
        normalized = {name: clean_weights[name] / weight_sum for name in names}

    remaining = total_samples - len(names)
    quotas = {name: remaining * normalized[name] for name in names}
    allocations = {name: 1 + math.floor(quotas[name]) for name in names}
    unallocated = total_samples - sum(allocations.values())
    canonical_index = {
        name: (
            GENERATOR_NAMES.index(name) if name in GENERATOR_NAMES else len(GENERATOR_NAMES) + index
        )
        for index, name in enumerate(names)
    }
    remainder_order = sorted(
        names,
        key=lambda name: (
            -(quotas[name] - math.floor(quotas[name])),
            canonical_index[name],
        ),
    )
    for name in remainder_order[:unallocated]:
        allocations[name] += 1
    return allocations
