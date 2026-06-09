"""Benchmark test helpers for real resource-gated quality evaluation."""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
from pathlib import Path
from typing import Any

import pytest
from mf_core.artifacts import ArtifactRequirement, check_artifact

HFM_REQUIREMENTS = (
    ArtifactRequirement("hfm_checkpoint", "HFM_CHECKPOINT_PATH"),
    ArtifactRequirement("hfm_decoder", "HFM_DECODER_PATH"),
)


def require_artifacts(*requirements: ArtifactRequirement) -> None:
    missing = []
    for requirement in requirements:
        status = check_artifact(requirement)
        if not status.available:
            missing.append(_missing_status_name(status))
    if missing:
        pytest.skip("Missing benchmark resources: " + ", ".join(missing))


def require_hfm_artifacts() -> None:
    require_artifacts(*HFM_REQUIREMENTS)


def read_smiles_file(env_var: str) -> list[str]:
    path = _required_file_from_env(env_var)
    smiles: list[str] = []
    with _open_text(path) as handle:
        for line in handle:
            text = line.strip()
            if text and not text.startswith("#"):
                smiles.append(text.split()[0])
    if not smiles:
        pytest.skip(f"{env_var} contains no SMILES records: {path}")
    return smiles


def read_scored_smiles_table(env_var: str, score_column: str) -> dict[str, float]:
    path = _required_file_from_env(env_var)
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            pytest.skip(f"{env_var} must be a CSV file with a header")
        required = {"smiles", score_column}
        missing_columns = required.difference(reader.fieldnames)
        if missing_columns:
            pytest.skip(
                f"{env_var} missing CSV columns: {', '.join(sorted(missing_columns))}"
            )
        rows = {
            str(row["smiles"]): float(row[score_column])
            for row in reader
            if row.get("smiles") and row.get(score_column)
        }
    if not rows:
        pytest.skip(f"{env_var} contains no scored SMILES records: {path}")
    return rows


def read_jsonl_records(env_var: str) -> list[dict[str, Any]]:
    path = _required_file_from_env(env_var)
    records: list[dict[str, Any]] = []
    with _open_text(path) as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"{env_var}:{line_no} must be a JSON object")
            records.append(payload)
    if not records:
        pytest.skip(f"{env_var} contains no JSONL records: {path}")
    return records


async def generate_hfm_smiles(batch_size: int, seed: int = 42) -> list[str]:
    require_hfm_artifacts()
    from mf_generators.hfm_3d import HFM3DGenerator

    generator = HFM3DGenerator(
        checkpoint_path=os.environ["HFM_CHECKPOINT_PATH"],
        decoder_path=os.environ["HFM_DECODER_PATH"],
        mode="production_real",
    )
    molecules = await generator.generate(batch_size=batch_size, sampling_seed=seed)
    smiles = [str(molecule.smiles) for molecule in molecules if molecule.smiles]
    if not smiles:
        pytest.fail("HFM-3D generated no SMILES")
    return smiles


def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    from rdkit import Chem
    from rdkit.Chem import DataStructs, rdMolDescriptors

    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_a, 2, 2048)
    fp_b = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_b, 2, 2048)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _required_file_from_env(env_var: str) -> Path:
    path_value = os.environ.get(env_var)
    if not path_value:
        pytest.skip(f"{env_var} is required")
    path = Path(path_value)
    if not path.is_file():
        pytest.skip(f"{env_var} file does not exist: {path}")
    return path


def _missing_status_name(status) -> str:
    if not status.configured:
        return str(status.source)
    return f"{status.name}: {status.message}"


def _open_text(path: Path):
    """Open a plain or gzip-compressed text file for reading."""
    if path.suffix in {".gz", ".gzip"}:
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return path.open(encoding="utf-8")
