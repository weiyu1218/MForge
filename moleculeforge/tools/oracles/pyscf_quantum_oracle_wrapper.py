#!/usr/bin/env python3
from __future__ import annotations

import json
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from types import ModuleType


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
        raise RuntimeError("pyscf quantum wrapper requires JSON stdin") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("pyscf quantum wrapper request must be a JSON object")
    return payload


def _run(request: dict) -> dict:
    smiles = str(request.get("molecule_smiles") or "")
    if not smiles:
        raise RuntimeError("pyscf quantum wrapper requires molecule_smiles")
    properties = _requested_properties(request)
    unsupported = [item for item in properties if item != "quantum_correction"]
    if unsupported:
        raise RuntimeError("pyscf quantum wrapper unsupported properties: " + ", ".join(unsupported))
    method = os.environ.get("L4_PYSCF_METHOD", "").strip().upper()
    if method != "RHF":
        raise RuntimeError("L4_PYSCF_METHOD=RHF is required")
    basis = os.environ.get("L4_PYSCF_BASIS", "").strip()
    if not basis:
        raise RuntimeError("L4_PYSCF_BASIS is required")
    engine = _engine(request)

    start = time.perf_counter()
    atoms, charge = _geometry_from_smiles(smiles)
    if engine == "gpu4pyscf":
        energy_hartree, converged = _run_gpu4pyscf_rhf(atoms, charge, basis)
    elif engine == "pyscf":
        energy_hartree, converged = _run_rhf(atoms, charge, basis)
    else:
        raise RuntimeError(f"unsupported quantum engine: {engine}")
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    if not converged:
        raise RuntimeError(f"{engine} RHF calculation did not converge")
    return {
        "engine": engine,
        "method": method,
        "basis": basis,
        "scores": {"quantum_correction": energy_hartree},
        "metadata": {
            "energy_unit": "hartree",
            "elapsed_ms": elapsed_ms,
            "atom_count": len(atoms),
            "charge": charge,
        },
    }


def _engine(request: dict) -> str:
    raw = str(request.get("engine") or os.environ.get("L4_QUANTUM_ENGINE") or "").strip().lower()
    if not raw:
        raise RuntimeError("pyscf quantum wrapper requires engine or L4_QUANTUM_ENGINE")
    return raw


def _requested_properties(request: dict) -> list[str]:
    raw = request.get("requested_properties")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("pyscf quantum wrapper requires requested_properties")
    properties = [str(item) for item in raw if str(item)]
    if len(properties) != len(raw):
        raise RuntimeError("pyscf quantum wrapper requested_properties items must be non-empty")
    return properties


def _geometry_from_smiles(smiles: str) -> tuple[list[tuple[str, tuple[float, float, float]]], int]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for PySCF quantum wrapper") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"invalid SMILES for PySCF quantum wrapper: {smiles}")
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, randomSeed=61453)
    if status != 0:
        raise RuntimeError(f"RDKit failed to embed quantum molecule: {smiles}")
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    conformer = mol.GetConformer()
    atoms = []
    for atom in mol.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        atoms.append(
            (
                atom.GetSymbol(),
                (float(position.x), float(position.y), float(position.z)),
            )
        )
    charge = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    return atoms, charge


def _run_rhf(
    atoms: list[tuple[str, tuple[float, float, float]]],
    charge: int,
    basis: str,
) -> tuple[float, bool]:
    try:
        from pyscf import gto, scf
    except ImportError as exc:
        raise RuntimeError("PySCF is required for PySCF quantum wrapper") from exc
    mol = gto.M(
        atom=atoms,
        basis=basis,
        unit="Angstrom",
        charge=charge,
        spin=0,
        verbose=0,
    )
    if mol.nelectron % 2:
        raise RuntimeError("PySCF RHF wrapper requires an even-electron molecule")
    mf = scf.RHF(mol)
    energy = mf.kernel()
    return float(energy), bool(mf.converged)


def _run_gpu4pyscf_rhf(
    atoms: list[tuple[str, tuple[float, float, float]]],
    charge: int,
    basis: str,
) -> tuple[float, bool]:
    with _guard_hanging_lscpu_probe():
        try:
            _load_gpu4pyscf_patch()
            from pyscf import gto
            from gpu4pyscf.scf import hf as gpu_hf
        except ImportError as exc:
            raise RuntimeError("GPU4PySCF is required for GPU4PySCF quantum wrapper") from exc
        mol = gto.M(
            atom=atoms,
            basis=basis,
            unit="Angstrom",
            charge=charge,
            spin=0,
            verbose=0,
        )
        if mol.nelectron % 2:
            raise RuntimeError("GPU4PySCF RHF wrapper requires an even-electron molecule")
        mf = gpu_hf.RHF(mol)
        energy = mf.kernel()
    return float(energy), bool(mf.converged)


def _load_gpu4pyscf_patch() -> ModuleType:
    loaded = sys.modules.get("gpu4pyscf._patch_pyscf")
    if loaded is not None:
        return loaded
    package_spec = importlib.util.find_spec("gpu4pyscf")
    if package_spec is None or package_spec.origin is None:
        raise ImportError("gpu4pyscf")
    package_dir = os.path.dirname(package_spec.origin)
    patch_path = os.path.join(package_dir, "_patch_pyscf.py")
    patch_spec = importlib.util.spec_from_file_location("gpu4pyscf._patch_pyscf", patch_path)
    if patch_spec is None or patch_spec.loader is None:
        raise ImportError("gpu4pyscf._patch_pyscf")
    package = ModuleType("gpu4pyscf")
    package.__file__ = package_spec.origin
    package.__path__ = [package_dir]  # type: ignore[attr-defined]
    package.__package__ = "gpu4pyscf"
    package.__spec__ = package_spec
    try:
        package.__version__ = importlib.metadata.version("gpu4pyscf-cuda12x")
    except importlib.metadata.PackageNotFoundError:
        package.__version__ = "unknown"
    sys.modules["gpu4pyscf"] = package
    module = importlib.util.module_from_spec(patch_spec)
    sys.modules["gpu4pyscf._patch_pyscf"] = module
    patch_spec.loader.exec_module(module)
    setattr(package, "_patch_pyscf", module)
    return module


@contextmanager
def _guard_hanging_lscpu_probe():
    original_run = subprocess.run

    def guarded_run(command, *args, **kwargs):
        if command == "lscpu" or command == ["lscpu"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return original_run(command, *args, **kwargs)

    subprocess.run = guarded_run
    try:
        yield
    finally:
        subprocess.run = original_run


if __name__ == "__main__":
    raise SystemExit(main())
