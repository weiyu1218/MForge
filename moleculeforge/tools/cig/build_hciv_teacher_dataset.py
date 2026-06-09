"""Build weak-supervised CIG-HCIV data from public text embeddings."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from cig_compiler_svc.domain.stages.stage2_cig_build import build_cig

from tools.cig.deepseek_semantic_parser import parse_semantic_text


EmbedText = Callable[[str], list[float]]
ParseText = Callable[[str], dict[str, Any]]


def lorentz_coordinates_from_embedding(
    embedding: Iterable[float],
    *,
    dim: int = 128,
    curvature: float = 1.0,
) -> list[float]:
    values = np.asarray([float(item) for item in embedding], dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise RuntimeError("teacher embedding must be a non-empty vector")
    if not np.isfinite(values).all():
        raise RuntimeError("teacher embedding must be finite")
    if dim <= 0:
        raise RuntimeError("dim must be positive")
    if not math.isfinite(curvature) or curvature <= 0.0:
        raise RuntimeError("curvature must be positive and finite")
    spatial = _project_to_dim(values, dim)
    norm = float(np.linalg.norm(spatial))
    if norm == 0.0:
        raise RuntimeError("teacher embedding projection must be non-zero")
    spatial = spatial / norm
    time_coord = math.sqrt(1.0 / curvature + float(np.dot(spatial, spatial)))
    return [float(time_coord), *[float(item) for item in spatial]]


def _project_to_dim(values: np.ndarray, dim: int) -> np.ndarray:
    if values.size == dim:
        return values.copy()
    rng = np.random.default_rng(17)
    projection = rng.normal(
        loc=0.0,
        scale=1.0 / math.sqrt(float(values.size)),
        size=(values.size, dim),
    )
    return values @ projection


def canonical_cig_text(cig, source_text: str) -> str:
    objectives = [
        {
            "id": node.id,
            "name": node.name,
            "type": getattr(node.type, "value", str(node.type)),
            "oracle": node.oracle,
            "weight": node.weight,
            "property": node.property,
        }
        for node in cig.objective_nodes
    ]
    payload = {
        "source": source_text,
        "objectives": objectives,
        "edges": [edge.model_dump(mode="json") for edge in cig.edges],
        "hyperedges": [edge.model_dump(mode="json") for edge in cig.hyperedges],
        "generative_priors": cig.generative_priors,
        "target_context": cig.target_context,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def build_teacher_records(
    intent_texts: list[str],
    *,
    parse_text: ParseText = parse_semantic_text,
    embed_text: EmbedText,
    dim: int = 128,
    curvature: float = 1.0,
) -> list[dict[str, Any]]:
    records = []
    for index, text in enumerate(intent_texts):
        if not text.strip():
            continue
        extracted = parse_text(text)
        cig = build_cig(extracted, source=text)
        teacher_text = canonical_cig_text(cig, text)
        target_hciv = lorentz_coordinates_from_embedding(
            embed_text(teacher_text),
            dim=dim,
            curvature=curvature,
        )
        records.append(
            {
                "id": f"h9-hciv-teacher-{index:06d}",
                "cig": cig.model_dump(mode="json", by_alias=True),
                "target_hciv": target_hciv,
                "weight": 1.0,
                "teacher": {
                    "type": "public_text_embedding",
                    "canonical_text": teacher_text,
                },
            }
        )
    if not records:
        raise RuntimeError("at least one intent text is required")
    return records


def load_public_text_embedder(model_name: str, *, device: str = "cpu") -> EmbedText:
    if model_name == "sklearn-hashing":
        from sklearn.feature_extraction.text import HashingVectorizer

        vectorizer = HashingVectorizer(
            n_features=384,
            alternate_sign=False,
            norm="l2",
            analyzer="word",
            ngram_range=(1, 2),
        )

        def embed(text: str) -> list[float]:
            vector = vectorizer.transform([text]).toarray().reshape(-1)
            return [float(item) for item in vector.tolist()]

        return embed

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def embed(text: str) -> list[float]:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        hidden = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        vector = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return [float(item) for item in vector.detach().cpu().reshape(-1).tolist()]

    return embed


def _read_intent_texts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        values = []
        for line in text.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
                raise RuntimeError("JSONL records must contain string field: text")
            values.append(payload["text"])
        return values
    if path.suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("texts")
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise RuntimeError("JSON input must be a list of strings or {texts: [...]}")
        return payload
    return [line.strip() for line in text.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-texts", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get(
            "HCIV_TEACHER_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
    )
    parser.add_argument("--device", default=os.environ.get("HCIV_TEACHER_DEVICE", "cpu"))
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--curvature", type=float, default=1.0)
    args = parser.parse_args(argv)

    texts = _read_intent_texts(Path(args.input_texts))
    embedder = load_public_text_embedder(args.embedding_model, device=args.device)
    records = build_teacher_records(
        texts,
        embed_text=embedder,
        dim=args.dim,
        curvature=args.curvature,
    )
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
