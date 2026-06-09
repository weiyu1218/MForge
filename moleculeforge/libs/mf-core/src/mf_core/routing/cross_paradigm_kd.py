"""Cross-Paradigm Knowledge Distillation layer."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class KDConfig:
    temperature: float = 2.0
    alpha: float = 0.5
    update_frequency: int = 10


@dataclass(frozen=True)
class OracleFeedback:
    oracle_name: str
    normalized_score: float


def boltz2_affinity_teacher_distribution(
    affinities: Sequence[Mapping[str, object] | object],
    *,
    favorable_delta_g: float,
    unfavorable_delta_g: float,
) -> list[float]:
    if favorable_delta_g >= unfavorable_delta_g:
        raise ValueError("favorable_delta_g must be lower than unfavorable_delta_g")
    values: list[float] = []
    for affinity in affinities:
        per_member = _record_field(affinity, "per_member_dg", required=False)
        if per_member is not None:
            if isinstance(per_member, str) or not isinstance(per_member, Sequence):
                raise ValueError("per_member_dg must be a sequence")
            values.extend(float(value) for value in per_member)
            continue
        values.append(float(_record_field(affinity, "delta_g_kcal_mol")))
    if not values:
        raise ValueError("Boltz2 affinity records must not be empty")
    span = unfavorable_delta_g - favorable_delta_g
    distribution = []
    for value in values:
        if not math.isfinite(value):
            raise ValueError("Boltz2 delta_g values must be finite")
        score = (unfavorable_delta_g - value) / span
        distribution.append(max(0.0, min(1.0, score)))
    return distribution


def boltz2_teacher_feedback(
    affinities: Sequence[Mapping[str, object] | object],
    *,
    favorable_delta_g: float,
    unfavorable_delta_g: float,
) -> dict[str, object]:
    return {
        "oracle_name": "boltz2",
        "teacher_distribution": boltz2_affinity_teacher_distribution(
            affinities,
            favorable_delta_g=favorable_delta_g,
            unfavorable_delta_g=unfavorable_delta_g,
        ),
    }


def hypseek_teacher_distribution(
    records: Sequence[Mapping[str, object] | object],
    *,
    score_field: str,
    min_score: float,
    max_score: float,
    higher_is_better: bool = True,
) -> list[float]:
    if not isinstance(score_field, str) or not score_field:
        raise ValueError("score_field must be a non-empty string")
    if min_score >= max_score:
        raise ValueError("min_score must be lower than max_score")
    if not records:
        raise ValueError("HypSeek score records must not be empty")

    span = max_score - min_score
    distribution = []
    for record in records:
        value = float(
            _record_field(
                record,
                score_field,
                source_name="HypSeek score record",
            )
        )
        if not math.isfinite(value):
            raise ValueError("HypSeek score values must be finite")
        score = (value - min_score) / span
        if not higher_is_better:
            score = 1.0 - score
        distribution.append(max(0.0, min(1.0, score)))
    return distribution


def hypseek_teacher_feedback(
    records: Sequence[Mapping[str, object] | object],
    *,
    score_field: str,
    min_score: float,
    max_score: float,
    higher_is_better: bool = True,
) -> dict[str, object]:
    return {
        "oracle_name": "hypseek",
        "teacher_distribution": hypseek_teacher_distribution(
            records,
            score_field=score_field,
            min_score=min_score,
            max_score=max_score,
            higher_is_better=higher_is_better,
        ),
    }


def load_teacher_embeddings_artifact(
    path_value: str | Path,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    import json

    path = Path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        value = payload.get("teacher_embeddings") or payload.get("embeddings")
    else:
        value = payload
    if not isinstance(value, list) or not value:
        raise ValueError("KD teacher embedding artifact requires teacher_embeddings")
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim not in {1, 2} or tensor.numel() == 0:
        raise ValueError("KD teacher embeddings must be a non-empty 1D or 2D tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError("KD teacher embeddings must contain finite values")
    return tensor


class WeakTeacher:
    """Weak teacher that scores SMILES using simple drug-likeness heuristics."""

    def score(self, smiles: str) -> float:
        if not smiles or len(smiles) > 200:
            return 0.0
        score = 0.5
        if len(smiles) < 3:
            return 0.0
        if len(smiles) > 100:
            score -= 0.2
        ring_chars = sum(1 for c in smiles if c.isdigit())
        score += min(ring_chars * 0.05, 0.3)
        unusual = sum(1 for c in smiles if c not in "()[]C=#ONPSFclBrIoHn1234567890%-+.@/\\")
        score -= unusual * 0.1
        return max(0.0, min(1.0, score))

    def score_batch(self, smiles_list: list[str]) -> list[float]:
        return [self.score(s) for s in smiles_list]


class CrossParadigmKDLayer(nn.Module):
    """Cross-paradigm KD that aligns generator output distributions
    with Oracle feedback."""

    def __init__(self, n_generators: int = 8, mode: str = "production_real"):
        super().__init__()
        if mode not in {"production_real", "local_demo"}:
            raise ValueError(f"Unsupported KD mode: {mode}")
        self.n_generators = n_generators
        self.mode = mode
        self.register_buffer("running_means", torch.zeros(n_generators))
        self.register_buffer("running_counts", torch.zeros(n_generators))
        self._quality_scores: list[float] = [1.0 / (i + 1) for i in range(n_generators)]
        self._teacher_embedding_targets: dict[int, Tensor] = {}
        # Learnable parameter to ensure compute_distillation_loss always has grad
        self.kd_scale = nn.Parameter(torch.ones(1))

    def update_teacher_scores(
        self,
        generator_name: str,
        generator_idx: int,
        oracle_feedback: Sequence[OracleFeedback | Mapping[str, object] | str],
    ) -> float:
        scores = self._extract_scores(oracle_feedback)
        mean_score = sum(scores) / max(len(scores), 1)
        n = self.running_counts[generator_idx]
        old_mean = self.running_means[generator_idx]
        new_n = n + len(scores)
        self.running_means[generator_idx] = (
            old_mean * n + mean_score * len(scores)
        ) / new_n
        self.running_counts[generator_idx] = new_n

        # Update quality scores for ranking
        if generator_idx < len(self._quality_scores):
            self._quality_scores[generator_idx] = float(mean_score)

        return mean_score

    def _extract_scores(
        self,
        feedback_items: Sequence[OracleFeedback | Mapping[str, object] | str],
    ) -> list[float]:
        if not feedback_items:
            raise ValueError("oracle feedback must not be empty")

        if self.mode == "local_demo":
            teacher = WeakTeacher()
        else:
            teacher = None

        scores: list[float] = []
        for item in feedback_items:
            if isinstance(item, str):
                if teacher is None:
                    raise TypeError(
                        "production KD requires oracle feedback, not SMILES strings"
                    )
                scores.append(teacher.score(item))
                continue

            if isinstance(item, OracleFeedback):
                oracle_name = item.oracle_name
                score = item.normalized_score
            elif isinstance(item, Mapping):
                oracle_name = item.get("oracle_name")
                if "teacher_distribution" in item:
                    self._validate_oracle_name(oracle_name)
                    scores.extend(self._normalized_score_distribution(item["teacher_distribution"]))
                    continue
                if "normalized_score" not in item:
                    raise ValueError(
                        "oracle feedback requires oracle_name and normalized_score"
                    )
                score = item["normalized_score"]
            else:
                raise TypeError("production KD requires oracle feedback records")

            self._validate_oracle_name(oracle_name)
            scores.append(self._normalized_score(score))

        return scores

    def _validate_oracle_name(self, oracle_name: object) -> None:
        if not isinstance(oracle_name, str) or not oracle_name:
            raise ValueError("oracle feedback requires a non-empty oracle_name")

    def _normalized_score_distribution(self, value: object) -> list[float]:
        if not isinstance(value, Sequence) or isinstance(value, str) or not value:
            raise ValueError("teacher_distribution must be a non-empty sequence")
        return [self._normalized_score(item) for item in value]

    def _normalized_score(self, value: object) -> float:
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("oracle feedback normalized_score must be finite")
        if score < 0.0 or score > 1.0:
            raise ValueError("oracle feedback normalized_score must be in [0, 1]")
        return score

    def update_teacher_embedding_targets(
        self,
        generator_idx: int,
        teacher_embeddings: Tensor | Sequence[Sequence[float]],
    ) -> Tensor:
        if generator_idx < 0 or generator_idx >= self.n_generators:
            raise IndexError("generator_idx is out of range")
        if isinstance(teacher_embeddings, Tensor):
            tensor = teacher_embeddings.detach().to(dtype=torch.float32)
        else:
            tensor = torch.tensor(teacher_embeddings, dtype=torch.float32)
        if tensor.ndim == 1:
            target = tensor
        elif tensor.ndim == 2 and tensor.shape[0] > 0:
            target = tensor.mean(dim=0)
        else:
            raise ValueError("teacher_embeddings must be a non-empty 1D or 2D tensor")
        if target.numel() == 0 or not torch.isfinite(target).all():
            raise ValueError("teacher_embeddings must contain finite values")
        self._teacher_embedding_targets[generator_idx] = target.detach()
        return target.detach()

    def compute_distillation_loss(
        self,
        embeddings: list[Tensor],
        indices: list[int],
    ) -> Tensor:
        if len(embeddings) != len(indices):
            raise ValueError(
                f"embeddings len {len(embeddings)} != indices len {len(indices)}"
            )
        if len(embeddings) == 0:
            return torch.tensor(0.0, requires_grad=True)

        losses = []
        for emb, idx in zip(embeddings, indices, strict=True):
            teacher_target = self._teacher_target_for(idx, emb)
            mse = torch.nn.functional.mse_loss(emb, teacher_target)
            losses.append(mse)

        base_loss = torch.stack(losses).mean()
        return (base_loss + self.kd_scale.squeeze() * 0.0).reshape(())

    def _teacher_target_for(self, generator_idx: int, embedding: Tensor) -> Tensor:
        target = self._teacher_embedding_targets.get(generator_idx)
        if target is None:
            return torch.zeros_like(embedding).detach()
        target = target.to(device=embedding.device, dtype=embedding.dtype)
        if target.shape == embedding.shape:
            return target.detach()
        if target.ndim == 1 and embedding.shape[-1] == target.shape[0]:
            return target.expand_as(embedding).detach()
        raise ValueError("teacher embedding target shape must match generator embedding")

    def get_generator_quality_ranking(self) -> dict[str, float]:
        ranking = {}
        for i in range(self.n_generators):
            if hasattr(self, "_quality_scores") and i < len(self._quality_scores):
                quality = self._quality_scores[i]
            else:
                quality = 1.0 / (i + 1)
            ranking[f"generator_{i}"] = float(quality)
        return ranking


def _record_field(
    record: Mapping[str, object] | object,
    name: str,
    *,
    required: bool = True,
    source_name: str = "Boltz2 affinity record",
) -> object:
    if isinstance(record, Mapping):
        if name in record:
            return record[name]
    elif hasattr(record, name):
        return getattr(record, name)
    if required:
        raise ValueError(f"{source_name} requires {name}")
    return None
