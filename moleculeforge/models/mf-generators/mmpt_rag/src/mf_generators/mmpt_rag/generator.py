"""MMPT-RAG (Matched Molecular Pair Transformer + Retrieval-Augmented Generation)."""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import shlex
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from mf_core.artifacts import CommandRequirement, check_command, require_available
from mf_core.types.molecule import MoleculeModel

_PATENT_RAG_COMMAND = CommandRequirement(
    "mmpt_patent_rag_command",
    "MMPT_PATENT_RAG_COMMAND",
    required=False,
)
_SEQ2SEQ_DECODER_COMMAND = CommandRequirement(
    "mmpt_seq2seq_decoder_command",
    "MMPT_SEQ2SEQ_DECODER_COMMAND",
    required=False,
)


class ExternalPatentRAGRetriever:
    def __init__(self, command: str):
        self.command = command
        self.timeout = float(os.getenv("MMPT_PATENT_RAG_TIMEOUT_SECONDS", "300"))

    async def retrieve(self, request: dict) -> list[dict]:
        return await asyncio.to_thread(self._run, request)

    def _run(self, request: dict) -> list[dict]:
        _require_command_available(_PATENT_RAG_COMMAND, self.command)
        completed = subprocess.run(
            shlex.split(self.command),
            input=json.dumps(request, sort_keys=True),
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(f"MMPT patent RAG command failed: {stderr}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("MMPT patent RAG command returned invalid JSON") from exc
        transforms = payload.get("transforms") if isinstance(payload, dict) else payload
        if not isinstance(transforms, list):
            raise RuntimeError("MMPT patent RAG command must return transforms")
        return _normalize_transforms(transforms)


class ExternalSeq2SeqDecoder:
    def __init__(self, command: str):
        self.command = command
        self.timeout = float(os.getenv("MMPT_SEQ2SEQ_DECODER_TIMEOUT_SECONDS", "300"))

    async def decode(self, request: dict) -> str:
        return await asyncio.to_thread(self._run, request)

    def _run(self, request: dict) -> str:
        _require_command_available(_SEQ2SEQ_DECODER_COMMAND, self.command)
        completed = subprocess.run(
            shlex.split(self.command),
            input=json.dumps(request, sort_keys=True),
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(f"MMPT Seq2Seq decoder command failed: {stderr}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("MMPT Seq2Seq decoder command returned invalid JSON") from exc
        smiles = _decoded_smiles(payload)
        if not smiles:
            raise RuntimeError("MMPT Seq2Seq decoder command must return smiles")
        return smiles


class MMPTRAGGenerator:
    name = "mmpt_rag"
    version = "0.1.0"
    supported_modes = ["lead_opt", "scaffold_hop"]

    def __init__(
        self,
        mmp_database: list[dict] | None = None,
        index_path: str = "",
        patent_negative_smiles: list[str] | None = None,
        patent_retriever: Any = None,
        seq2seq_decoder: Any = None,
    ):
        self.index_path = index_path
        loaded = _load_index(index_path) if index_path else None
        self.mmp_database = mmp_database or loaded or [
            {"pattern": "F", "replacement": "Cl"},
            {"pattern": "Cl", "replacement": "Br"},
            {"pattern": "OC", "replacement": "OCC"},
        ]
        self.patent_negative_smiles = set(patent_negative_smiles or [])
        self.patent_retriever = patent_retriever or _patent_retriever_from_env()
        self.seq2seq_decoder = seq2seq_decoder or _seq2seq_decoder_from_env()

    async def generate(
        self,
        hciv: Any,
        cone: Any,
        cig: Any,
        n_samples: int = 10,
        seed: int | None = None,
    ) -> AsyncIterator[MoleculeModel]:
        seeds = ["c1ccccc1F", "CC(=O)Oc1ccccc1F", "Fc1ccc(N)cc1"]
        generation_context = {
            "hciv": _jsonable(hciv),
            "intent_cone": _jsonable(cone),
            "cig": _jsonable(cig),
            "n_samples": n_samples,
            "seed": seed,
        }
        retrieved_transforms = await _retrieve_patent_transforms(
            self.patent_retriever,
            generation_context,
        )
        transforms = _dedupe_transforms([*retrieved_transforms, *self.mmp_database])

        decoded_candidates = []
        for mmp in transforms:
            transform_seeds = _transform_seed_smiles(mmp, seeds)
            for seed_smi in transform_seeds:
                new_smi = await _decode_transform(
                    self.seq2seq_decoder,
                    seed_smi,
                    mmp,
                    generation_context,
                )
                if new_smi is None:
                    new_smi = self._simple_replace(
                        seed_smi,
                        mmp["pattern"],
                        mmp["replacement"],
                    )
                if new_smi is None and seed_smi == mmp.get("seed_smiles"):
                    product_smiles = mmp.get("product_smiles")
                    new_smi = str(product_smiles) if product_smiles else None
                if new_smi is None:
                    continue
                if self._is_patent_negative(new_smi, mmp):
                    continue
                decoded_candidates.append(
                    {
                        "smiles": new_smi,
                        "seed": seed_smi,
                        "transform": mmp,
                        "score": self._contrastive_score(new_smi, mmp),
                    }
                )
        decoded_candidates.sort(
            key=lambda item: (
                item["score"],
                _seed_priority(str(item["seed"]), seeds),
            ),
            reverse=True,
        )
        for item in decoded_candidates[:n_samples]:
            yield MoleculeModel(
                smiles=item["smiles"],
                canonical_smiles=item["smiles"],
                generator_name=self.name,
                humu_embedding=None,
                properties={
                    "transform_id": str(item["transform"].get("id", "")),
                    "source_seed": item["seed"],
                    "contrastive_score": float(item["score"]),
                },
            )

    def _simple_replace(
        self,
        mol_smi: str,
        pattern_smi: str,
        replacement_smi: str,
    ) -> str | None:
        if pattern_smi not in mol_smi:
            return None
        return mol_smi.replace(pattern_smi, replacement_smi, 1)

    def _is_patent_negative(self, smiles: str, transform: dict) -> bool:
        transform_negatives = set(transform.get("negative_smiles", []) or [])
        return smiles in self.patent_negative_smiles or smiles in transform_negatives

    def _contrastive_score(self, smiles: str, transform: dict) -> float:
        positives = [str(item) for item in transform.get("positive_smiles", []) or []]
        negatives = [str(item) for item in transform.get("negative_smiles", []) or []]
        negatives.extend(self.patent_negative_smiles)
        positive_score = max(
            [_token_jaccard(smiles, item) for item in positives],
            default=0.0,
        )
        negative_score = max(
            [_token_jaccard(smiles, item) for item in negatives],
            default=0.0,
        )
        retrieval_score = float(transform.get("retrieval_score", 0.0) or 0.0)
        return retrieval_score + positive_score - negative_score


def _load_index(index_path: str) -> list[dict]:
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"MMPT index artifact not found: {index_path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MMPT index artifact must be a JSON object")
    transforms = payload.get("transforms")
    if not isinstance(transforms, list) or not transforms:
        raise ValueError("MMPT index artifact requires transforms")
    return _normalize_transforms(transforms)


def _normalize_transforms(transforms: list) -> list[dict]:
    normalized = []
    for index, transform in enumerate(transforms):
        normalized.append(_normalize_transform(transform, index))
    return normalized


def _normalize_transform(transform: object, index: int = 0) -> dict:
    if not isinstance(transform, dict):
        raise ValueError("MMPT transform must be a JSON object")
    pattern = transform.get("pattern")
    replacement = transform.get("replacement")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("MMPT transform requires pattern")
    if not isinstance(replacement, str) or not replacement:
        raise ValueError("MMPT transform requires replacement")
    record = {
        "id": str(transform.get("id", index)),
        "pattern": pattern,
        "replacement": replacement,
    }
    seed_smiles = transform.get("seed_smiles")
    if isinstance(seed_smiles, str) and seed_smiles:
        record["seed_smiles"] = seed_smiles
    product_smiles = transform.get("product_smiles")
    if isinstance(product_smiles, str) and product_smiles:
        record["product_smiles"] = product_smiles
    negative_smiles = [str(item) for item in transform.get("negative_smiles", []) or []]
    if negative_smiles:
        record["negative_smiles"] = negative_smiles
    positive_smiles = [str(item) for item in transform.get("positive_smiles", []) or []]
    if positive_smiles:
        record["positive_smiles"] = positive_smiles
    if "retrieval_score" in transform:
        record["retrieval_score"] = float(transform["retrieval_score"])
    return record


def _patent_retriever_from_env() -> ExternalPatentRAGRetriever | None:
    command = os.getenv("MMPT_PATENT_RAG_COMMAND", "").strip()
    if not command:
        return None
    return ExternalPatentRAGRetriever(command)


def _seq2seq_decoder_from_env() -> ExternalSeq2SeqDecoder | None:
    command = os.getenv("MMPT_SEQ2SEQ_DECODER_COMMAND", "").strip()
    if not command:
        return None
    return ExternalSeq2SeqDecoder(command)


def _require_command_available(
    requirement: CommandRequirement,
    command: str,
) -> None:
    required_requirement = CommandRequirement(
        requirement.name,
        requirement.env_var,
        required=True,
    )
    env = {**os.environ, requirement.env_var: command}
    require_available([check_command(required_requirement, env=env)])


async def _retrieve_patent_transforms(retriever: Any, request: dict) -> list[dict]:
    if retriever is None:
        return []
    if hasattr(retriever, "retrieve"):
        result = retriever.retrieve(request)
    elif callable(retriever):
        result = retriever(request)
    else:
        raise TypeError("patent_retriever must expose retrieve(request) or be callable")
    if inspect.isawaitable(result):
        result = await result
    return _normalize_transforms(list(result))


async def _decode_transform(
    decoder: Any,
    seed_smiles: str,
    transform: dict,
    generation_context: dict,
) -> str | None:
    if decoder is None:
        return None
    request = {
        **generation_context,
        "seed_smiles": seed_smiles,
        "transform": transform,
    }
    if hasattr(decoder, "decode"):
        result = decoder.decode(request)
    elif callable(decoder):
        result = decoder(request)
    else:
        raise TypeError("seq2seq_decoder must expose decode(request) or be callable")
    if inspect.isawaitable(result):
        result = await result
    smiles = _decoded_smiles(result)
    if not smiles:
        raise RuntimeError("seq2seq_decoder must return smiles")
    return smiles


def _decoded_smiles(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return ""
    value = payload.get("smiles") or payload.get("product_smiles")
    if not isinstance(value, str):
        return ""
    return value


def _dedupe_transforms(transforms: list[dict]) -> list[dict]:
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for transform in transforms:
        key = (
            str(transform.get("id", "")),
            str(transform["pattern"]),
            str(transform["replacement"]),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(transform)
    return unique


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, dict | list | str | int | float | bool):
        return value
    return str(value)


def _transform_seed_smiles(transform: dict, default_seeds: list[str]) -> list[str]:
    seed_smiles = transform.get("seed_smiles")
    if isinstance(seed_smiles, str) and seed_smiles:
        return [seed_smiles]
    return default_seeds


def _seed_priority(seed_smiles: str, default_seeds: list[str]) -> int:
    if seed_smiles in default_seeds:
        return -default_seeds.index(seed_smiles)
    return 0


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = _smiles_tokens(left)
    right_tokens = _smiles_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return intersection / union if union else 0.0


def _smiles_tokens(smiles: str) -> set[str]:
    tokens = set()
    for index, character in enumerate(smiles):
        tokens.add(character)
        if index + 2 <= len(smiles):
            tokens.add(smiles[index : index + 2])
    return tokens
