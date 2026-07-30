"""MMPT-RAG (Matched Molecular Pair Transformer + Retrieval-Augmented Generation)."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

from mf_core.artifacts import CommandRequirement, check_command, require_available
from mf_core.types.molecule import MoleculeModel

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None

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
_VALIDATION_ARTIFACT_SCHEMA = "moleculeforge.validation_artifact.v1"
_VALIDATION_ARTIFACT_PURPOSE = "synthetic_pipeline_validation_only"
_VALIDATION_ARTIFACT_METADATA_FILE = "moleculeforge_validation_artifact.json"
_VALIDATION_ARTIFACT_MARKER_KEY = "moleculeforge_validation_artifact"
_VALIDATION_ARTIFACT_SEED = 7
_VALIDATION_PROBE_SAMPLES = 256


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
        loaded, self.kd_projection = (
            _load_index(index_path) if index_path else (None, None)
        )
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
                canonical_smiles = _canonical_candidate_smiles(new_smi)
                if canonical_smiles is None:
                    continue
                decoded_candidates.append(
                    {
                        "smiles": new_smi,
                        "canonical_smiles": canonical_smiles,
                        "seed": seed_smi,
                        "transform": mmp,
                        "score": self._contrastive_score(new_smi, mmp),
                        "kd_score": _kd_weighted_alignment(mmp),
                    }
                )
        decoded_candidates.sort(
            key=lambda item: (
                item["score"] + item["kd_score"],
                _seed_priority(str(item["seed"]), seeds),
            ),
            reverse=True,
        )
        unique_candidates = []
        seen_smiles: set[str] = set()
        for item in decoded_candidates:
            canonical_smiles = str(item["canonical_smiles"])
            if canonical_smiles in seen_smiles:
                continue
            seen_smiles.add(canonical_smiles)
            unique_candidates.append(item)
        if not unique_candidates:
            raise RuntimeError("MMPT generation produced no valid candidates")
        if len(unique_candidates) < n_samples:
            raise RuntimeError(
                "MMPT generation produced "
                f"{len(unique_candidates)} unique valid candidates, requested {n_samples}"
            )
        for item in unique_candidates[:n_samples]:
            properties = {
                "transform_id": str(item["transform"].get("id", "")),
                "source_seed": item["seed"],
                "contrastive_score": float(item["score"]),
            }
            if "kd_alignment_score" in item["transform"]:
                properties["kd_alignment_score"] = float(
                    item["transform"]["kd_alignment_score"]
                )
                properties["kd_weight"] = float(item["transform"]["kd_weight"])
            yield MoleculeModel(
                smiles=item["smiles"],
                canonical_smiles=item["canonical_smiles"],
                generator_name=self.name,
                humu_embedding=None,
                properties=properties,
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


def _canonical_candidate_smiles(smiles: str) -> str | None:
    if Chem is None:
        raise ImportError("RDKit is required for MMPT candidate validation")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


async def bootstrap_validation_artifacts(
    target_directory: str | Path,
) -> dict[str, Path]:
    target = Path(target_directory).expanduser().resolve()
    if target.exists():
        return await _validated_existing_validation_artifacts(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
    )
    try:
        index_path = temporary / "mmpt_index.json"
        replacements = ["C" * length for length in range(1, _VALIDATION_PROBE_SAMPLES + 1)]
        _write_json(
            index_path,
            {
                "schema_version": "mmpt_mmp_index.v1",
                _VALIDATION_ARTIFACT_MARKER_KEY: _validation_artifact_marker(),
                "pairs": [],
                "transforms": [
                    {
                        "id": f"validation_f_to_{index}",
                        "pattern": "F",
                        "replacement": replacement,
                        "seed_smiles": "c1ccccc1F",
                        "product_smiles": f"c1ccccc1{replacement}",
                        "retrieval_score": float(len(replacements) - index),
                    }
                    for index, replacement in enumerate(replacements)
                ],
            },
        )
        _write_validation_metadata(temporary, {"index": index_path})
        paths = _validation_artifact_paths(temporary)
        await _probe_validation_artifacts(paths)
        _fsync_tree(temporary)
        if target.exists():
            return await _validated_existing_validation_artifacts(target)
        try:
            temporary.rename(target)
        except OSError:
            if not target.exists():
                raise
            return await _validated_existing_validation_artifacts(target)
        _fsync_directory(target.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _validation_artifact_paths(target)


async def _validated_existing_validation_artifacts(
    target: Path,
) -> dict[str, Path]:
    try:
        paths = _validation_artifact_paths(target)
        await _probe_validation_artifacts(paths)
    except Exception as exc:
        raise RuntimeError(
            f"MMPT validation bootstrap refuses to overwrite existing path: {target}"
        ) from exc
    return paths


def load_validation_artifact_metadata(
    artifact_path: str | Path,
) -> dict[str, object] | None:
    artifact = Path(artifact_path).expanduser().resolve()
    metadata_path = artifact.parent / _VALIDATION_ARTIFACT_METADATA_FILE
    if not metadata_path.is_file():
        return _read_embedded_validation_artifact_metadata(artifact)
    metadata = _read_validation_metadata(metadata_path)
    records = metadata["artifacts"]
    if not any(
        isinstance(record, Mapping) and record.get("file") == artifact.name
        for record in records.values()
    ):
        raise RuntimeError("MMPT validation metadata does not reference configured artifact")
    _validate_artifact_records(artifact.parent, records)
    return metadata


def _read_embedded_validation_artifact_metadata(
    artifact_path: Path,
) -> dict[str, object] | None:
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    marker_present = _VALIDATION_ARTIFACT_MARKER_KEY in payload
    marker = payload.get(_VALIDATION_ARTIFACT_MARKER_KEY)
    if not marker_present and (
        payload.get("schema_version") == _VALIDATION_ARTIFACT_SCHEMA
        or payload.get("purpose") == _VALIDATION_ARTIFACT_PURPOSE
    ):
        marker_present = True
        marker = payload
    if not marker_present:
        return None
    if (
        not isinstance(marker, Mapping)
        or marker.get("schema_version") != _VALIDATION_ARTIFACT_SCHEMA
        or marker.get("purpose") != _VALIDATION_ARTIFACT_PURPOSE
        or marker.get("generator") != "mmpt_rag"
        or marker.get("seed") != _VALIDATION_ARTIFACT_SEED
    ):
        raise RuntimeError("MMPT embedded validation artifact marker is invalid")
    return _validation_artifact_marker()


def _validation_artifact_marker() -> dict[str, object]:
    return {
        "schema_version": _VALIDATION_ARTIFACT_SCHEMA,
        "purpose": _VALIDATION_ARTIFACT_PURPOSE,
        "generator": "mmpt_rag",
        "seed": _VALIDATION_ARTIFACT_SEED,
    }


def _validation_artifact_paths(directory: Path) -> dict[str, Path]:
    metadata_path = directory / _VALIDATION_ARTIFACT_METADATA_FILE
    if not metadata_path.is_file():
        raise RuntimeError("MMPT validation artifact metadata is missing")
    metadata = _read_validation_metadata(metadata_path)
    records = metadata["artifacts"]
    if set(records) != {"index"}:
        raise RuntimeError("MMPT validation artifact metadata has invalid artifact set")
    _validate_artifact_records(directory, records)
    return {
        "index": directory / str(records["index"]["file"]),
        "metadata": metadata_path,
    }


def _read_validation_metadata(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("MMPT validation artifact metadata is invalid") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _VALIDATION_ARTIFACT_SCHEMA
        or payload.get("purpose") != _VALIDATION_ARTIFACT_PURPOSE
        or payload.get("generator") != "mmpt_rag"
        or payload.get("seed") != _VALIDATION_ARTIFACT_SEED
        or not isinstance(payload.get("artifacts"), dict)
    ):
        raise RuntimeError("MMPT validation artifact metadata is invalid")
    return payload


def _validate_artifact_records(
    directory: Path,
    records: Mapping[str, object],
) -> None:
    for record in records.values():
        if not isinstance(record, Mapping):
            raise RuntimeError("MMPT validation artifact record is invalid")
        filename = record.get("file")
        expected_sha256 = record.get("sha256")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
        ):
            raise RuntimeError("MMPT validation artifact record is invalid")
        artifact_path = directory / filename
        if not artifact_path.is_file():
            raise RuntimeError(f"MMPT validation artifact is missing: {artifact_path}")
        if _sha256(artifact_path) != expected_sha256:
            raise RuntimeError(f"MMPT validation artifact checksum mismatch: {artifact_path}")


def _write_validation_metadata(
    directory: Path,
    artifacts: Mapping[str, Path],
) -> None:
    _write_json(
        directory / _VALIDATION_ARTIFACT_METADATA_FILE,
        {
            "schema_version": _VALIDATION_ARTIFACT_SCHEMA,
            "purpose": _VALIDATION_ARTIFACT_PURPOSE,
            "generator": "mmpt_rag",
            "seed": _VALIDATION_ARTIFACT_SEED,
            "artifacts": {
                name: {
                    "file": path.name,
                    "sha256": _sha256(path),
                }
                for name, path in artifacts.items()
            },
        },
    )


async def _probe_validation_artifacts(paths: Mapping[str, Path]) -> None:
    generator = MMPTRAGGenerator(index_path=str(paths["index"]))
    molecules = [
        molecule
        async for molecule in generator.generate(
            None,
            None,
            None,
            n_samples=_VALIDATION_PROBE_SAMPLES,
            seed=_VALIDATION_ARTIFACT_SEED,
        )
    ]
    canonical_smiles = [molecule.canonical_smiles for molecule in molecules]
    if (
        len(molecules) != _VALIDATION_PROBE_SAMPLES
        or len(set(canonical_smiles)) != _VALIDATION_PROBE_SAMPLES
        or any(_canonical_candidate_smiles(smiles) is None for smiles in canonical_smiles)
    ):
        raise RuntimeError("MMPT validation artifact production probe failed")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            dict(payload),
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_tree(directory: Path) -> None:
    for artifact in sorted(path for path in directory.rglob("*") if path.is_file()):
        with artifact.open("rb") as handle:
            os.fsync(handle.fileno())
    _fsync_directory(directory)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_index(
    index_path: str,
) -> tuple[list[dict], dict[str, object] | None]:
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"MMPT index artifact not found: {index_path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MMPT index artifact must be a JSON object")
    transforms = payload.get("transforms")
    if not isinstance(transforms, list) or not transforms:
        raise ValueError("MMPT index artifact requires transforms")
    normalized_transforms = _normalize_transforms(transforms)
    kd_projection = _normalize_kd_projection(
        payload.get("kd_projection"),
        normalized_transforms,
    )
    return normalized_transforms, kd_projection


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
    record.update(_normalize_kd_alignment(transform))
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


def _normalize_kd_alignment(transform: dict) -> dict[str, float]:
    has_score = "kd_alignment_score" in transform
    has_weight = "kd_weight" in transform
    if has_score != has_weight:
        raise ValueError(
            "MMPT transform KD alignment requires kd_alignment_score and kd_weight"
        )
    if not has_score:
        return {}
    score_value = transform["kd_alignment_score"]
    weight_value = transform["kd_weight"]
    if isinstance(score_value, bool) or not isinstance(score_value, int | float):
        raise ValueError("MMPT transform kd_alignment_score must be a number")
    if isinstance(weight_value, bool) or not isinstance(weight_value, int | float):
        raise ValueError("MMPT transform kd_weight must be a number")
    score = float(score_value)
    weight = float(weight_value)
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError("MMPT transform kd_alignment_score must be finite and in [0, 1]")
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("MMPT transform kd_weight must be finite and positive")
    return {
        "kd_alignment_score": score,
        "kd_weight": weight,
    }


def _kd_weighted_alignment(transform: dict) -> float:
    normalized = _normalize_kd_alignment(transform)
    if not normalized:
        return 0.0
    return normalized["kd_alignment_score"] * normalized["kd_weight"]


def _normalize_kd_projection(
    value: object,
    transforms: list[dict],
) -> dict[str, object] | None:
    transforms_have_alignment = [
        "kd_alignment_score" in transform for transform in transforms
    ]
    if value is None:
        if any(transforms_have_alignment):
            raise ValueError("MMPT transform KD alignment requires kd_projection")
        return None
    if not isinstance(value, dict):
        raise ValueError("MMPT kd_projection must be a JSON object")
    if not transforms_have_alignment or not all(transforms_have_alignment):
        raise ValueError(
            "MMPT kd_projection requires KD alignment for every transform"
        )
    if value.get("schema_version") != "linear_kd_projection.v1":
        raise ValueError("MMPT kd_projection schema_version is unsupported")
    expected_features = [
        "seed_smiles_length",
        "product_smiles_length",
        "pattern_length",
        "replacement_length",
    ]
    if value.get("input_features") != expected_features:
        raise ValueError("MMPT kd_projection input_features are invalid")
    input_dim = _positive_int(value.get("input_dim"), "input_dim")
    if input_dim != len(expected_features):
        raise ValueError("MMPT kd_projection input_dim must be 4")
    teacher_dim = _positive_int(value.get("teacher_dim"), "teacher_dim")
    feature_mean = _finite_vector(
        value.get("feature_mean"),
        input_dim,
        "feature_mean",
    )
    feature_scale = _finite_vector(
        value.get("feature_scale"),
        input_dim,
        "feature_scale",
    )
    if any(item <= 0.0 for item in feature_scale):
        raise ValueError("MMPT KD projection feature_scale must be positive")
    weights_value = value.get("weights")
    if not isinstance(weights_value, list) or len(weights_value) != teacher_dim:
        raise ValueError("MMPT KD projection weights shape is invalid")
    weights = [
        _finite_vector(row, input_dim, "weights")
        for row in weights_value
    ]
    bias = _finite_vector(value.get("bias"), teacher_dim, "bias")
    regularization = _positive_float(
        value.get("regularization"),
        "regularization",
    )
    kd_weight = _positive_float(value.get("kd_weight"), "kd_weight")
    generator_idx = _non_negative_int(value.get("generator_idx"), "generator_idx")
    for transform in transforms:
        if not math.isclose(
            float(transform["kd_weight"]),
            kd_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "MMPT transform kd_weight must match kd_projection kd_weight"
            )
    return {
        "schema_version": "linear_kd_projection.v1",
        "input_features": expected_features,
        "input_dim": input_dim,
        "teacher_dim": teacher_dim,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "weights": weights,
        "bias": bias,
        "regularization": regularization,
        "kd_weight": kd_weight,
        "generator_idx": generator_idx,
    }


def _finite_vector(value: object, size: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"MMPT KD projection {name} shape is invalid")
    if any(
        isinstance(item, bool) or not isinstance(item, int | float)
        for item in value
    ):
        raise ValueError(f"MMPT KD projection {name} must contain numbers")
    normalized = [float(item) for item in value]
    if not all(math.isfinite(item) for item in normalized):
        raise ValueError(f"MMPT KD projection {name} must contain finite values")
    return normalized


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"MMPT KD projection {name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"MMPT KD projection {name} must be finite and positive")
    return normalized


def _positive_int(value: object, name: str) -> int:
    normalized = _non_negative_int(value, name)
    if normalized == 0:
        raise ValueError(f"MMPT KD projection {name} must be positive")
    return normalized


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"MMPT KD projection {name} must be a non-negative integer"
        )
    return value


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
