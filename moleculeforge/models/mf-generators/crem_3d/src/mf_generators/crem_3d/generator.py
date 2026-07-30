"""CReM-3D: Chemically Reasonable Mutations in 3D."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mf_core.plugins.generator import GeneratorPlugin
from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2, oracle_pb2_grpc
from mf_core.types.humu import IntentCone
from mf_core.types.molecule import Molecule
from mf_generators.crem_3d.fragment_replacement import get_attachment_points, replace_fragment

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None

_VALIDATION_ARTIFACT_SCHEMA = "moleculeforge.validation_artifact.v1"
_VALIDATION_ARTIFACT_PURPOSE = "synthetic_pipeline_validation_only"
_VALIDATION_ARTIFACT_METADATA_FILE = "moleculeforge_validation_artifact.json"
_VALIDATION_ARTIFACT_MARKER_KEY = "moleculeforge_validation_artifact"
_VALIDATION_ARTIFACT_SEED = 7


class CReM3DGenerator(GeneratorPlugin):
    name = "crem_3d"

    def __init__(
        self,
        mmp_db_path: str = "",
        mode: str = "production_real",
        docking_scorer: Any = None,
        pharmacophore_scorer: Any = None,
        humu_embedding_scorer: Any = None,
    ):
        if mode not in {"production_real", "local_demo"}:
            raise ValueError(f"Unknown CReM3DGenerator mode: {mode}")
        self.mode = mode
        self.mmp_db_path = mmp_db_path
        self.docking_scorer = docking_scorer
        self.pharmacophore_scorer = pharmacophore_scorer
        self.humu_embedding_scorer = humu_embedding_scorer
        self.kd_projection: dict[str, object] | None = None
        if mmp_db_path:
            self.mutations = self._load_mmp_database(mmp_db_path)
        elif mode == "local_demo":
            self.mutations = [
                {
                    "id": "local_demo_fluoro",
                    "seed_smiles": "c1ccccc1",
                    "product": "Fc1ccccc1",
                }
            ]
        else:
            raise RuntimeError(
                "CReM-3D production generation requires an MMP database artifact"
            )

    def _load_mmp_database(self, mmp_db_path: str) -> list[dict[str, object]]:
        path = Path(mmp_db_path)
        if not path.exists():
            raise FileNotFoundError(f"CReM MMP database artifact not found: {mmp_db_path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("CReM MMP database artifact must be a JSON object")
        mutations = payload.get("mutations")
        if not isinstance(mutations, list) or not mutations:
            raise ValueError("CReM MMP database artifact requires mutations")
        normalized_mutations = [
            self._normalize_mutation(idx, mutation)
            for idx, mutation in enumerate(mutations)
        ]
        self.kd_projection = _normalize_kd_projection(
            payload.get("kd_projection"),
            normalized_mutations,
        )
        return normalized_mutations

    def _normalize_mutation(
        self,
        idx: int,
        mutation: object,
    ) -> dict[str, object]:
        if not isinstance(mutation, Mapping):
            raise ValueError("CReM mutation record must be a JSON object")
        product = mutation.get("product")
        fragment_smiles = mutation.get("fragment_smiles")
        if (not isinstance(product, str) or not product) and (
            not isinstance(fragment_smiles, str) or not fragment_smiles
        ):
            raise ValueError("CReM mutation record requires product or fragment_smiles")
        seed_smiles = mutation.get("seed_smiles", "")
        if seed_smiles and not isinstance(seed_smiles, str):
            raise ValueError("CReM mutation seed_smiles must be a string")
        attachment_index = mutation.get("attachment_index")
        if attachment_index is not None and not isinstance(attachment_index, int):
            raise ValueError("CReM mutation attachment_index must be an integer")
        record = {
            "id": str(mutation.get("id", idx)),
            "seed_smiles": self._canonical_smiles(seed_smiles) if seed_smiles else "",
            "fragment_smiles": fragment_smiles or "",
            "attachment_index": attachment_index,
            "product": self._canonical_smiles(product) if product else "",
        }
        record.update(_normalize_kd_alignment(mutation))
        return record

    def _canonical_smiles(self, smiles: str) -> str:
        if Chem is None:
            raise ImportError("RDKit is required for CReM validity checks")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"CReM mutation produced invalid SMILES: {smiles}")
        return Chem.MolToSmiles(mol)

    async def generate(
        self,
        batch_size: int,
        intent_cone: IntentCone | None = None,
        **kwargs,
    ) -> list[Molecule]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        seed_smiles = kwargs.get("seed_smiles", "")
        if seed_smiles and not isinstance(seed_smiles, str):
            raise ValueError("seed_smiles must be a string")
        seed_key = self._canonical_smiles(seed_smiles) if seed_smiles else ""
        mutations = [
            mutation for mutation in self.mutations
            if not seed_key or mutation["seed_smiles"] in {"", seed_key}
        ]
        if not mutations:
            raise RuntimeError("CReM MMP database contains no mutation for seed_smiles")
        if self.kd_projection is not None:
            mutations.sort(key=_kd_rank_score, reverse=True)
        results = []
        for i in range(batch_size):
            mutation = mutations[i % len(mutations)]
            smiles = self._product_for_mutation(mutation, seed_key)
            fragment_replacement = bool(mutation.get("fragment_smiles"))
            metadata = {
                "generator_name": self.name,
                "mmp_database": self.mmp_db_path,
                "mutation_id": mutation["id"],
                "fragment_replacement": str(fragment_replacement).lower(),
            }
            if "kd_alignment_score" in mutation:
                metadata["kd_alignment_score"] = str(mutation["kd_alignment_score"])
                metadata["kd_weight"] = str(mutation["kd_weight"])
            results.append(
                Molecule(
                    smiles=smiles,
                    metadata=metadata,
                )
            )
        if self.docking_scorer is not None:
            results = await self._score_and_rank_with_docking(results)
        if self.pharmacophore_scorer is not None:
            results = await self._score_and_rank_with_pharmacophore(results, intent_cone)
        if self.humu_embedding_scorer is not None:
            results = await self._score_and_rank_with_humu_embeddings(results, intent_cone)
        return results

    async def _score_and_rank_with_docking(
        self,
        molecules: list[Molecule],
    ) -> list[Molecule]:
        if hasattr(self.docking_scorer, "score_batch"):
            records = await _score_batch_with_docking(
                self.docking_scorer,
                [molecule.smiles for molecule in molecules],
            )
            return _rank_molecules_with_docking_records(molecules, records)
        scored = []
        for molecule in molecules:
            record = await _score_with_docking(self.docking_scorer, molecule.smiles)
            metadata = dict(molecule.metadata)
            for key, value in record.items():
                metadata[str(key)] = str(value)
            scored.append(
                Molecule(
                    smiles=molecule.smiles,
                    sdf_bytes=molecule.sdf_bytes,
                    metadata=metadata,
                )
            )
        return sorted(scored, key=_docking_rank_key)

    async def _score_and_rank_with_pharmacophore(
        self,
        molecules: list[Molecule],
        intent_cone: IntentCone | None,
    ) -> list[Molecule]:
        records = await _score_batch_with_provider(
            self.pharmacophore_scorer,
            [molecule.smiles for molecule in molecules],
            intent_cone=intent_cone,
        )
        return _rank_molecules_with_score_records(
            molecules,
            records,
            score_key="pharmacophore_score",
            lower_is_better=False,
        )

    async def _score_and_rank_with_humu_embeddings(
        self,
        molecules: list[Molecule],
        intent_cone: IntentCone | None,
    ) -> list[Molecule]:
        records = await _score_batch_with_provider(
            self.humu_embedding_scorer,
            [molecule.smiles for molecule in molecules],
            intent_cone=intent_cone,
        )
        return _rank_molecules_with_score_records(
            molecules,
            records,
            score_key="humu_alignment_score",
            lower_is_better=False,
            humu_embedding_key="humu_embedding",
            intent_cone=intent_cone,
        )

    def _product_for_mutation(self, mutation: dict[str, object], seed_key: str) -> str:
        fragment_smiles = mutation.get("fragment_smiles")
        if not fragment_smiles:
            product = mutation.get("product")
            if not isinstance(product, str) or not product:
                raise RuntimeError("CReM mutation product is required")
            return product
        seed_smiles = seed_key or mutation.get("seed_smiles")
        if not isinstance(seed_smiles, str) or not seed_smiles:
            raise RuntimeError("CReM fragment replacement requires seed_smiles")
        seed_mol = Chem.MolFromSmiles(seed_smiles)
        if seed_mol is None:
            raise ValueError(f"CReM seed_smiles is invalid: {seed_smiles}")
        attachment_points = get_attachment_points(seed_mol)
        attachment_index = mutation.get("attachment_index")
        if attachment_index is None:
            attachment_index = attachment_points[0]
        if attachment_index not in attachment_points:
            raise RuntimeError("CReM attachment_index is not valid for seed_smiles")
        product_mol = replace_fragment(seed_mol, int(attachment_index), str(fragment_smiles))
        if product_mol is None:
            raise RuntimeError("CReM fragment replacement failed RDKit sanitization")
        return Chem.MolToSmiles(product_mol)

    async def info(self) -> dict:
        return {
            "name": "crem_3d",
            "version": "0.1.0",
            "supports_streaming": False,
            "requires_gpu": False,
        }


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
        database_path = temporary / "crem_mmp_database.json"
        _write_json(
            database_path,
            {
                "schema_version": "crem_mmp_database.v1",
                _VALIDATION_ARTIFACT_MARKER_KEY: _validation_artifact_marker(),
                "mutations": [
                    {
                        "id": "validation_benzene_fluoro",
                        "seed_smiles": "c1ccccc1",
                        "product": "Fc1ccccc1",
                    },
                    {
                        "id": "validation_benzene_chloro",
                        "seed_smiles": "c1ccccc1",
                        "product": "Clc1ccccc1",
                    },
                ],
            },
        )
        _write_validation_metadata(
            temporary,
            {"mmp_database": database_path},
        )
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
            f"CReM validation bootstrap refuses to overwrite existing path: {target}"
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
        raise RuntimeError("CReM validation metadata does not reference configured artifact")
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
        or marker.get("generator") != "crem_3d"
        or marker.get("seed") != _VALIDATION_ARTIFACT_SEED
    ):
        raise RuntimeError("CReM embedded validation artifact marker is invalid")
    return _validation_artifact_marker()


def _validation_artifact_marker() -> dict[str, object]:
    return {
        "schema_version": _VALIDATION_ARTIFACT_SCHEMA,
        "purpose": _VALIDATION_ARTIFACT_PURPOSE,
        "generator": "crem_3d",
        "seed": _VALIDATION_ARTIFACT_SEED,
    }


def _validation_artifact_paths(directory: Path) -> dict[str, Path]:
    metadata_path = directory / _VALIDATION_ARTIFACT_METADATA_FILE
    if not metadata_path.is_file():
        raise RuntimeError("CReM validation artifact metadata is missing")
    metadata = _read_validation_metadata(metadata_path)
    records = metadata["artifacts"]
    if set(records) != {"mmp_database"}:
        raise RuntimeError("CReM validation artifact metadata has invalid artifact set")
    _validate_artifact_records(directory, records)
    return {
        "mmp_database": directory / str(records["mmp_database"]["file"]),
        "metadata": metadata_path,
    }


def _read_validation_metadata(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("CReM validation artifact metadata is invalid") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _VALIDATION_ARTIFACT_SCHEMA
        or payload.get("purpose") != _VALIDATION_ARTIFACT_PURPOSE
        or payload.get("generator") != "crem_3d"
        or payload.get("seed") != _VALIDATION_ARTIFACT_SEED
        or not isinstance(payload.get("artifacts"), dict)
    ):
        raise RuntimeError("CReM validation artifact metadata is invalid")
    return payload


def _validate_artifact_records(
    directory: Path,
    records: Mapping[str, object],
) -> None:
    for record in records.values():
        if not isinstance(record, Mapping):
            raise RuntimeError("CReM validation artifact record is invalid")
        filename = record.get("file")
        expected_sha256 = record.get("sha256")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
        ):
            raise RuntimeError("CReM validation artifact record is invalid")
        artifact_path = directory / filename
        if not artifact_path.is_file():
            raise RuntimeError(f"CReM validation artifact is missing: {artifact_path}")
        if _sha256(artifact_path) != expected_sha256:
            raise RuntimeError(f"CReM validation artifact checksum mismatch: {artifact_path}")


def _write_validation_metadata(
    directory: Path,
    artifacts: Mapping[str, Path],
) -> None:
    _write_json(
        directory / _VALIDATION_ARTIFACT_METADATA_FILE,
        {
            "schema_version": _VALIDATION_ARTIFACT_SCHEMA,
            "purpose": _VALIDATION_ARTIFACT_PURPOSE,
            "generator": "crem_3d",
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
    generator = CReM3DGenerator(
        mmp_db_path=str(paths["mmp_database"]),
        mode="production_real",
    )
    molecules = await generator.generate(
        batch_size=2,
        seed_smiles="c1ccccc1",
    )
    if len(molecules) != 2 or any(
        Chem is None or Chem.MolFromSmiles(molecule.smiles) is None
        for molecule in molecules
    ):
        raise RuntimeError("CReM validation artifact production probe failed")


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


def _normalize_kd_alignment(mutation: Mapping[str, object]) -> dict[str, float]:
    has_score = "kd_alignment_score" in mutation
    has_weight = "kd_weight" in mutation
    if has_score != has_weight:
        raise ValueError(
            "CReM mutation KD alignment requires kd_alignment_score and kd_weight"
        )
    if not has_score:
        return {}
    score_value = mutation["kd_alignment_score"]
    weight_value = mutation["kd_weight"]
    if isinstance(score_value, bool) or not isinstance(score_value, int | float):
        raise ValueError("CReM mutation kd_alignment_score must be a number")
    if isinstance(weight_value, bool) or not isinstance(weight_value, int | float):
        raise ValueError("CReM mutation kd_weight must be a number")
    score = float(score_value)
    weight = float(weight_value)
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError("CReM mutation kd_alignment_score must be finite and in [0, 1]")
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("CReM mutation kd_weight must be finite and positive")
    return {
        "kd_alignment_score": score,
        "kd_weight": weight,
    }


def _kd_rank_score(mutation: Mapping[str, object]) -> float:
    return float(mutation.get("kd_alignment_score", 0.0)) * float(
        mutation.get("kd_weight", 0.0)
    )


def _normalize_kd_projection(
    value: object,
    mutations: list[dict[str, object]],
) -> dict[str, object] | None:
    mutations_have_alignment = [
        "kd_alignment_score" in mutation for mutation in mutations
    ]
    if value is None:
        if any(mutations_have_alignment):
            raise ValueError("CReM mutation KD alignment requires kd_projection")
        return None
    if not isinstance(value, Mapping):
        raise ValueError("CReM kd_projection must be a JSON object")
    if not mutations_have_alignment or not all(mutations_have_alignment):
        raise ValueError(
            "CReM kd_projection requires KD alignment for every mutation"
        )
    if value.get("schema_version") != "linear_kd_projection.v1":
        raise ValueError("CReM kd_projection schema_version is unsupported")
    expected_features = [
        "seed_smiles_length",
        "fragment_smiles_length",
        "attachment_index",
        "product_smiles_length",
    ]
    if value.get("input_features") != expected_features:
        raise ValueError("CReM kd_projection input_features are invalid")
    input_dim = _positive_int(value.get("input_dim"), "input_dim")
    if input_dim != len(expected_features):
        raise ValueError("CReM kd_projection input_dim must be 4")
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
        raise ValueError("CReM KD projection feature_scale must be positive")
    weights_value = value.get("weights")
    if not isinstance(weights_value, list) or len(weights_value) != teacher_dim:
        raise ValueError("CReM KD projection weights shape is invalid")
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
    for mutation in mutations:
        if not math.isclose(
            float(mutation["kd_weight"]),
            kd_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "CReM mutation kd_weight must match kd_projection kd_weight"
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
        raise ValueError(f"CReM KD projection {name} shape is invalid")
    if any(
        isinstance(item, bool) or not isinstance(item, int | float)
        for item in value
    ):
        raise ValueError(f"CReM KD projection {name} must contain numbers")
    normalized = [float(item) for item in value]
    if not all(math.isfinite(item) for item in normalized):
        raise ValueError(f"CReM KD projection {name} must contain finite values")
    return normalized


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"CReM KD projection {name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"CReM KD projection {name} must be finite and positive")
    return normalized


def _positive_int(value: object, name: str) -> int:
    normalized = _non_negative_int(value, name)
    if normalized == 0:
        raise ValueError(f"CReM KD projection {name} must be positive")
    return normalized


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"CReM KD projection {name} must be a non-negative integer"
        )
    return value


class DockOracleGrpcScorer:
    def __init__(
        self,
        target: str = "",
        *,
        stub=None,
        oracle_name: str = "diffdock_l",
        level: int = oracle_pb2.L2_DOCKING,
        requested_properties: list[str] | None = None,
    ) -> None:
        self.target = target
        self.oracle_name = oracle_name
        self.level = level
        self.requested_properties = requested_properties or ["docking_score"]
        if stub is not None:
            self.stub = stub
        else:
            if not target:
                raise ValueError("DockOracleGrpcScorer requires target or stub")
            import grpc

            _ensure_default_event_loop()
            self.channel = grpc.aio.insecure_channel(target)
            self.stub = oracle_pb2_grpc.OracleServiceStub(self.channel)

    async def score(self, smiles: str) -> dict:
        records = await self.score_batch([smiles])
        if smiles not in records:
            raise RuntimeError(f"Dock oracle returned no result for {smiles}")
        return records[smiles]

    async def score_batch(self, smiles_list: list[str]) -> dict[str, dict]:
        if not smiles_list:
            raise ValueError("smiles_list must not be empty")
        response = await self.stub.Evaluate(
            oracle_pb2.OracleBatchRequest(
                molecule_smiles=[str(smiles) for smiles in smiles_list],
                level=self.level,
                requested_properties=[
                    str(prop)
                    for prop in self.requested_properties
                ],
            )
        )
        records: dict[str, dict] = {}
        for evaluation in response.evaluations:
            if not evaluation.success:
                raise RuntimeError(evaluation.error_message or "dock oracle evaluation failed")
            record = {
                str(key): float(value)
                for key, value in evaluation.scores.items()
            }
            record["oracle_name"] = str(evaluation.oracle_name or self.oracle_name)
            records[str(evaluation.molecule_smiles)] = record
        return records


async def _score_batch_with_docking(docking_scorer: Any, smiles_list: list[str]) -> dict[str, dict]:
    result = docking_scorer.score_batch(smiles_list)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, dict):
        raise TypeError("docking scorer score_batch() must return a dictionary")
    records = {}
    for smiles in smiles_list:
        record = result.get(smiles)
        if not isinstance(record, dict):
            raise TypeError("docking scorer score_batch() must return records keyed by SMILES")
        records[smiles] = record
    return records


def _rank_molecules_with_docking_records(
    molecules: list[Molecule],
    records: dict[str, dict],
) -> list[Molecule]:
    scored = []
    for molecule in molecules:
        metadata = dict(molecule.metadata)
        for key, value in records[molecule.smiles].items():
            metadata[str(key)] = str(value)
        scored.append(
            Molecule(
                smiles=molecule.smiles,
                sdf_bytes=molecule.sdf_bytes,
                metadata=metadata,
            )
        )
    return sorted(scored, key=_docking_rank_key)


async def _score_batch_with_provider(
    provider: Any,
    smiles_list: list[str],
    *,
    intent_cone: IntentCone | None = None,
) -> dict[str, dict]:
    if provider is None:
        raise RuntimeError("CReM score provider is not configured")
    if hasattr(provider, "score_batch"):
        result = _call_provider(provider.score_batch, smiles_list, intent_cone=intent_cone)
    elif hasattr(provider, "score"):
        records = {}
        for smiles in smiles_list:
            records[smiles] = await _score_with_provider(provider, smiles, intent_cone=intent_cone)
        return records
    else:
        raise TypeError("CReM score provider must expose score_batch() or score()")
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, dict):
        raise TypeError("CReM score provider score_batch() must return a dictionary")
    records = {}
    for smiles in smiles_list:
        record = result.get(smiles)
        if not isinstance(record, dict):
            raise TypeError("CReM score provider must return records keyed by SMILES")
        records[smiles] = record
    return records


async def _score_with_provider(
    provider: Any,
    smiles: str,
    *,
    intent_cone: IntentCone | None = None,
) -> dict:
    result = _call_provider(provider.score, smiles, intent_cone=intent_cone)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, dict):
        raise TypeError("CReM score provider score() must return a dictionary")
    return result


def _call_provider(method: Any, payload: Any, *, intent_cone: IntentCone | None) -> Any:
    if _accepts_intent_cone(method):
        return method(payload, intent_cone=intent_cone)
    return method(payload)


def _accepts_intent_cone(method: Any) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return "intent_cone" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _rank_molecules_with_score_records(
    molecules: list[Molecule],
    records: dict[str, dict],
    *,
    score_key: str,
    lower_is_better: bool,
    humu_embedding_key: str = "",
    intent_cone: IntentCone | None = None,
) -> list[Molecule]:
    scored = []
    for molecule in molecules:
        record = dict(records[molecule.smiles])
        metadata = dict(molecule.metadata)
        humu_embedding = molecule.humu_embedding
        if humu_embedding_key:
            embedding = _normalize_vector(record.get(humu_embedding_key), humu_embedding_key)
            humu_embedding = json.dumps(embedding).encode("utf-8")
            metadata["humu_embedding_dim"] = str(len(embedding))
            if score_key not in record:
                record[score_key] = _humu_alignment_score(embedding, intent_cone)
        if score_key not in record:
            raise ValueError(f"CReM score provider result requires {score_key}")
        for key, value in record.items():
            if key == humu_embedding_key:
                continue
            metadata[str(key)] = _metadata_value(value)
        scored.append(
            Molecule(
                smiles=molecule.smiles,
                sdf_bytes=molecule.sdf_bytes,
                humu_embedding=humu_embedding,
                metadata=metadata,
            )
        )
    return sorted(
        scored,
        key=lambda molecule: _numeric_score_rank_key(
            molecule,
            score_key=score_key,
            lower_is_better=lower_is_better,
        ),
    )


async def _score_with_docking(docking_scorer: Any, smiles: str) -> dict:
    if hasattr(docking_scorer, "score"):
        result = docking_scorer.score(smiles)
    elif hasattr(docking_scorer, "evaluate"):
        result = docking_scorer.evaluate([smiles], ["docking_score"])
    elif callable(docking_scorer):
        result = docking_scorer(smiles)
    else:
        raise TypeError("docking scorer must expose score(smiles), evaluate(), or be callable")
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, dict) and smiles in result:
        result = result[smiles]
    if not isinstance(result, dict):
        raise TypeError("docking scorer must return a dictionary")
    return result


def _docking_rank_key(molecule: Molecule) -> float:
    value = molecule.metadata.get("docking_score")
    if value is None:
        return float("inf")
    return float(value)


def _numeric_score_rank_key(
    molecule: Molecule,
    *,
    score_key: str,
    lower_is_better: bool,
) -> tuple[float, str]:
    score = float(molecule.metadata[score_key])
    return (score if lower_is_better else -score, molecule.smiles)


def _normalize_vector(value: object, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"CReM score provider result requires {name}")
    if not all(isinstance(item, int | float) for item in value):
        raise ValueError(f"CReM {name} values must be numeric")
    return [float(item) for item in value]


def _humu_alignment_score(
    embedding: list[float],
    intent_cone: IntentCone | None,
) -> float:
    if intent_cone is None:
        raise ValueError("CReM HUMU alignment requires intent_cone")
    axis = intent_cone.axis
    if len(axis) == len(embedding) + 1:
        axis = axis[1:]
    dim = min(len(axis), len(embedding))
    if dim == 0:
        raise ValueError("CReM HUMU alignment requires non-empty overlap")
    axis_vector = [float(value) for value in axis[:dim]]
    embedding_vector = embedding[:dim]
    axis_norm = sum(value * value for value in axis_vector) ** 0.5
    embedding_norm = sum(value * value for value in embedding_vector) ** 0.5
    if axis_norm == 0.0 or embedding_norm == 0.0:
        return 0.0
    dot = sum(left * right for left, right in zip(axis_vector, embedding_vector, strict=True))
    return dot / (axis_norm * embedding_norm)


def _metadata_value(value: object) -> str:
    if isinstance(value, list | dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _ensure_default_event_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
