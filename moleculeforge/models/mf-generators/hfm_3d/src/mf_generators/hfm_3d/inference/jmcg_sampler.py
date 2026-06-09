"""Engineering skeleton for JMCG joint sample records.

This module builds explicit, serializable joint molecule / route / property /
pocket / intent records from existing HFM candidates and JMCG feedback context.
It is not a trained joint sampler; it is the W8-E local contract layer that keeps
engineering evidence separate from W8-R research quality.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from mf_core.geometry.lorentz import normalize_lorentz_embedding
from mf_core.types.molecule import Molecule
from mf_humu.manifold.lorentz import LorentzManifold


@dataclass
class JMCGContextRecord:
    kind: str
    subject: dict[str, object] = field(default_factory=dict)
    humu_embedding: list[float] | None = None
    weight: float = 1.0
    confidence: float = 1.0
    polarity: str = "attract"
    source: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": self.kind,
            "subject": self.subject,
            "weight": self.weight,
            "confidence": self.confidence,
            "polarity": self.polarity,
            "source": self.source,
            "metadata": self.metadata,
        }
        if self.humu_embedding is not None:
            payload["humu_embedding"] = self.humu_embedding
        return payload


@dataclass
class JMCGJointSample:
    molecule: dict[str, object]
    route: dict[str, object] | None
    property_profile: dict[str, object]
    pocket: dict[str, object] | None
    intent: dict[str, object] | None
    joint_score: float
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "molecule": self.molecule,
            "route": self.route,
            "property_profile": self.property_profile,
            "pocket": self.pocket,
            "intent": self.intent,
            "joint_score": self.joint_score,
            "metadata": self.metadata,
        }


def parse_jmcg_context(payload: object) -> list[JMCGContextRecord]:
    if payload in (None, "", b"", []):
        return []
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, Mapping):
        if "records" in payload:
            return parse_jmcg_context(payload["records"])
        return [_context_record_from_mapping(payload)]
    if isinstance(payload, list | tuple):
        records: list[JMCGContextRecord] = []
        for item in payload:
            records.extend(parse_jmcg_context(item))
        return records
    raise TypeError(f"Unsupported JMCG context payload: {type(payload)!r}")


class JMCGEngineeringSampler:
    """Build deterministic W8-E joint sample records from local context."""

    def __init__(self, embedding_dim: int = 129, curvature: float = 1.0) -> None:
        self.embedding_dim = int(embedding_dim)
        self.manifold = LorentzManifold(curvature=float(curvature))

    def sample(
        self,
        molecules: Sequence[object],
        *,
        routes: Sequence[Mapping[str, object]] | None = None,
        property_profile: Mapping[str, object] | None = None,
        jmcg_feedback: object = None,
        max_samples: int | None = None,
    ) -> list[JMCGJointSample]:
        molecule_items = [_normalize_molecule(item) for item in molecules]
        route_items = [_normalize_route(item) for item in (routes or [])]
        context_records = parse_jmcg_context(jmcg_feedback)
        property_context = [
            record for record in context_records if record.kind == "property"
        ]
        pocket_context = [
            record for record in context_records if record.kind == "pocket"
        ]
        intent_context = [
            record for record in context_records if record.kind == "intent"
        ]
        route_context = [
            record for record in context_records if record.kind == "route"
        ]

        limit = len(molecule_items) if max_samples is None else int(max_samples)
        samples = []
        for index, molecule in enumerate(molecule_items[:limit]):
            route = self._select_route(index, route_items, route_context)
            pocket = self._select_context(pocket_context)
            intent = self._select_context(intent_context)
            score, score_metadata = self._joint_alignment_score(
                molecule,
                route,
                pocket,
                intent,
            )
            samples.append(
                JMCGJointSample(
                    molecule=molecule,
                    route=route,
                    property_profile=self._property_profile(
                        property_profile,
                        property_context,
                    ),
                    pocket=pocket,
                    intent=intent,
                    joint_score=score,
                    metadata={
                        "mode": "engineering_skeleton",
                        "schema": "moleculeforge.jmcg.joint_sample.v1",
                        "embedding_dim": self.embedding_dim,
                        **score_metadata,
                    },
                )
            )
        return samples

    def _select_route(
        self,
        index: int,
        routes: list[dict[str, object]],
        route_context: list[JMCGContextRecord],
    ) -> dict[str, object] | None:
        if routes:
            return routes[index % len(routes)]
        if not route_context:
            return None
        context = max(route_context, key=lambda item: item.weight * item.confidence)
        route_id = str(context.subject.get("id") or context.metadata.get("route_id") or "")
        return {
            "route_id": route_id,
            "reactions": list(context.metadata.get("reactions") or []),
            "humu_embedding": context.humu_embedding,
            "source": context.source,
        }

    def _select_context(
        self,
        records: list[JMCGContextRecord],
    ) -> dict[str, object] | None:
        if not records:
            return None
        record = max(records, key=lambda item: item.weight * item.confidence)
        return record.to_dict()

    def _property_profile(
        self,
        property_profile: Mapping[str, object] | None,
        property_context: list[JMCGContextRecord],
    ) -> dict[str, object]:
        profile = dict(property_profile or {})
        if property_context:
            profile["context_records"] = [record.to_dict() for record in property_context]
        return profile

    def _joint_alignment_score(
        self,
        molecule: Mapping[str, object],
        route: Mapping[str, object] | None,
        pocket: Mapping[str, object] | None,
        intent: Mapping[str, object] | None,
    ) -> tuple[float, dict[str, object]]:
        molecule_embedding, molecule_ignored = self._embedding_from_mapping(molecule)
        ignored = molecule_ignored
        distances = []
        for kind, item in (
            ("route", route),
            ("pocket", pocket),
            ("intent", intent),
        ):
            embedding, item_ignored = self._embedding_from_mapping(item)
            ignored += item_ignored
            if molecule_embedding is None or embedding is None:
                continue
            distances.append(
                {
                    "kind": kind,
                    "distance": self._lorentz_distance(molecule_embedding, embedding),
                }
            )
        if not distances:
            return 0.0, {
                "alignment_distances": [],
                "alignment_pair_count": 0,
                "ignored_embedding_count": ignored,
                "non_steering_context": True,
            }
        mean_distance = sum(item["distance"] for item in distances) / float(len(distances))
        return 1.0 / (1.0 + mean_distance), {
            "alignment_distances": distances,
            "alignment_pair_count": len(distances),
            "ignored_embedding_count": ignored,
            "non_steering_context": False,
        }

    def _embedding_from_mapping(
        self,
        item: Mapping[str, object] | None,
    ) -> tuple[list[float] | None, int]:
        if item is None:
            return None, 0
        raw_embedding = item.get("humu_embedding")
        if raw_embedding is None:
            metadata = item.get("metadata")
            if isinstance(metadata, Mapping):
                raw_embedding = metadata.get("latent")
        embedding = _normalize_embedding(raw_embedding)
        if embedding is None:
            return None, int(raw_embedding is not None)
        normalized_embedding = normalize_lorentz_embedding(
            embedding,
            expected_dim=self.embedding_dim,
            curvature=self.manifold.k,
        )
        if normalized_embedding is None:
            return None, 1
        return normalized_embedding, 0

    def _lorentz_distance(self, left: list[float], right: list[float]) -> float:
        left_tensor = torch.tensor(left, dtype=torch.float32).unsqueeze(0)
        right_tensor = torch.tensor(right, dtype=torch.float32).unsqueeze(0)
        return float(self.manifold.distance(left_tensor, right_tensor).item())


def _context_record_from_mapping(payload: Mapping[str, object]) -> JMCGContextRecord:
    return JMCGContextRecord(
        kind=str(payload.get("kind") or "unspecified"),
        subject=_mapping_or_empty(payload.get("subject")),
        humu_embedding=_normalize_embedding(
            payload.get("humu_embedding", payload.get("route_humu_embedding"))
        ),
        weight=_float_or_default(payload.get("weight"), 1.0),
        confidence=max(0.0, min(_float_or_default(payload.get("confidence"), 1.0), 1.0)),
        polarity=str(payload.get("polarity") or "attract"),
        source=str(payload.get("source") or ""),
        metadata=_mapping_or_empty(payload.get("metadata")),
    )


def _normalize_molecule(item: object) -> dict[str, object]:
    if isinstance(item, Molecule):
        payload: dict[str, object] = {
            "smiles": item.smiles,
            "metadata": dict(item.metadata),
        }
        if item.humu_embedding:
            payload["humu_embedding"] = _embedding_from_bytes(item.humu_embedding)
        return payload
    if isinstance(item, Mapping):
        payload = dict(item)
        if "metadata" in payload and not isinstance(payload["metadata"], Mapping):
            payload["metadata"] = {}
        payload.setdefault("metadata", {})
        return payload
    raise TypeError(f"Unsupported JMCG molecule candidate: {type(item)!r}")


def _normalize_route(item: Mapping[str, object]) -> dict[str, object]:
    route_id = item.get("route_id", item.get("id", ""))
    return {
        "route_id": str(route_id),
        "reactions": list(item.get("reactions") or item.get("steps") or []),
        "humu_embedding": _normalize_embedding(item.get("humu_embedding")),
        "metadata": _mapping_or_empty(item.get("metadata")),
    }


def _normalize_embedding(value: object) -> list[float] | None:
    if value in (None, "", b""):
        return None
    if isinstance(value, bytes):
        return _embedding_from_bytes(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return _normalize_embedding(decoded)
    if isinstance(value, list | tuple):
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return None
    return None


def _embedding_from_bytes(value: bytes) -> list[float] | None:
    try:
        decoded = value.decode("utf-8")
    except UnicodeDecodeError:
        decoded = ""
    if decoded:
        embedding = _normalize_embedding(decoded)
        if embedding is not None:
            return embedding
    if len(value) % 4 != 0:
        return None
    return [float(item[0]) for item in struct.iter_unpack("<f", value)]


def _mapping_or_empty(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _float_or_default(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
