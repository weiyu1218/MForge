#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    try:
        request = _read_request()
        response = _run(request)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    json.dump(response, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _read_request() -> dict:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise RuntimeError("dock wrapper requires JSON stdin") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("dock wrapper request must be a JSON object")
    return payload


def _run(request: dict) -> dict:
    engine = str(request.get("engine") or "")
    if engine != "gnina":
        raise RuntimeError("dock wrapper currently supports engine=gnina")
    smiles = str(request.get("smiles") or "")
    if not smiles:
        raise RuntimeError("dock wrapper requires smiles")
    protein_pdb = str(request.get("protein_pdb") or "")
    if not protein_pdb:
        raise RuntimeError("dock wrapper requires protein_pdb")
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mforge-dock-") as work_dir:
        work_path = Path(work_dir)
        receptor = _materialize_receptor(protein_pdb, work_path)
        ligand = work_path / "ligand.sdf"
        _write_ligand_sdf(smiles, ligand)
        completed = _run_gnina(receptor, ligand)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    scores = _scores_from_gnina_stdout(completed.stdout)
    docking_score = scores.get("docking_score")
    if docking_score is None:
        raise RuntimeError("GNINA output did not contain a docking score")
    return {
        "engine": "gnina",
        "score": docking_score,
        "scores": scores,
        "uncertainties": {},
        "elapsed_ms": elapsed_ms,
    }


def _materialize_receptor(value: str, work_path: Path) -> Path:
    if "\n" not in value:
        path = Path(value).expanduser()
        if path.is_file():
            return path
        raise RuntimeError("protein_pdb must be an existing PDB path or PDB block")
    path = work_path / "protein.pdb"
    path.write_text(value, encoding="utf-8")
    return path


def _write_ligand_sdf(smiles: str, path: Path) -> None:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError("RDKit is required to materialize GNINA ligand input") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"invalid SMILES for docking: {smiles}")
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, randomSeed=61453)
    if status != 0:
        raise RuntimeError(f"RDKit failed to embed docking ligand: {smiles}")
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(mol)
    finally:
        writer.close()


def _run_gnina(receptor: Path, ligand: Path) -> subprocess.CompletedProcess[str]:
    binary = os.environ.get("GNINA_BINARY", "gnina").strip() or "gnina"
    command = [
        *shlex.split(binary),
        "-r",
        str(receptor),
        "-l",
        str(ligand),
        "--score_only",
    ]
    completed = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=float(os.environ.get("DOCK_ORACLE_TIMEOUT_SECONDS", "120")),
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(stderr or "GNINA score-only command failed")
    return completed


def _scores_from_gnina_stdout(stdout: str) -> dict[str, float]:
    json_scores = _scores_from_json(stdout)
    if json_scores:
        return json_scores
    scores: dict[str, float] = {}
    affinity = _first_float(
        stdout,
        (
            r"(?im)^\s*CNNaffinity\s*[:= ]\s*({number})",
            r"(?im)^\s*CNN_affinity\s*[:= ]\s*({number})",
            r"(?im)^\s*Affinity\s*[:= ]\s*({number})",
            r"(?im)^\s*docking_score\s*[:= ]\s*({number})",
        ),
    )
    if affinity is not None:
        scores["docking_score"] = affinity
    cnn_score = _first_float(
        stdout,
        (
            r"(?im)^\s*CNNscore\s*[:= ]\s*({number})",
            r"(?im)^\s*CNN_score\s*[:= ]\s*({number})",
        ),
    )
    if cnn_score is not None:
        scores["cnn_score"] = cnn_score
    return scores


def _scores_from_json(stdout: str) -> dict[str, float]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    raw_scores = payload.get("scores")
    if isinstance(raw_scores, dict):
        scores = _numeric_map(raw_scores)
    else:
        scores = {}
    score = payload.get("score", payload.get("docking_score"))
    if isinstance(score, int | float):
        scores.setdefault("docking_score", float(score))
    return scores


def _first_float(text: str, patterns: tuple[str, ...]) -> float | None:
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    for pattern in patterns:
        match = re.search(pattern.format(number=number), text)
        if match:
            return float(match.group(1))
    return None


def _numeric_map(values: dict) -> dict[str, float]:
    output = {}
    for key, value in values.items():
        if isinstance(value, int | float):
            output[str(key)] = float(value)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
