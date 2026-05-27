"""ChEMBL 36 molecular data ingestion pipeline.

Reads ChEMBL SQLite/SDF data, filters by drug-likeness criteria,
generates 3D conformers, and outputs PyTorch Geometric format for
HUMU encoder training.

Usage:
    python data/ingestion/chembl_ingestion.py \
        --input zzzzz/Chembl/chembl_36/ \
        --output data/processing/chembl_mol_dataset/ \
        --max_mols 2400000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def filter_molecules(
    conn: sqlite3.Connection,
    mw_min: float = 200.0,
    mw_max: float = 700.0,
    max_mols: int = 2_400_000,
) -> list[dict]:
    """Filter ChEMBL molecules by drug-likeness criteria.

    Filters:
      - MW ∈ [200, 700]
      - No inorganic salts (C or N must be present)
      - Canonical SMILES not null
    """
    query = """
    SELECT DISTINCT
        cs.canonical_smiles AS smiles,
        md.chembl_id,
        md.molregno
    FROM compound_structures cs
    JOIN molecule_dictionary md ON cs.molregno = md.molregno
    JOIN compound_properties cp ON md.molregno = cp.molregno
    WHERE cs.canonical_smiles IS NOT NULL
      AND cp.mw BETWEEN ? AND ?
      AND (cs.canonical_smiles LIKE '%C%' OR cs.canonical_smiles LIKE '%N%')
    LIMIT ?
    """
    cursor = conn.execute(query, (mw_min, mw_max, max_mols))
    rows = cursor.fetchall()

    molecules = []
    for smiles, chembl_id, molregno in rows:
        molecules.append({
            "smiles": smiles,
            "chembl_id": chembl_id,
            "molregno": molregno,
        })

    logger.info("chembl.filtered", n_input=max_mols, n_filtered=len(molecules))
    return molecules


def deduplicate(molecules: list[dict]) -> list[dict]:
    """Deduplicate by canonical SMILES."""
    seen = set()
    unique = []
    for mol in molecules:
        smi = mol["smiles"]
        if smi not in seen:
            seen.add(smi)
            unique.append(mol)
    logger.info("chembl.deduplicated", n_before=len(molecules), n_after=len(unique))
    return unique


def generate_conformers(
    molecules: list[dict],
    n_confs: int = 1,
    seed: int = 42,
) -> list[dict]:
    """Generate 3D conformers using RDKit ETKDG.

    Adds 'conformers' field (list of SDF block bytes) to each molecule.
    Molecules that fail conformer generation are kept but with empty conformers.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        logger.warning("rdkit.not_installed — skipping conformer generation")
        for mol in molecules:
            mol["conformers"] = []
        return molecules

    success = 0
    for mol in molecules:
        rdmol = Chem.MolFromSmiles(mol["smiles"])
        if rdmol is None:
            mol["conformers"] = []
            continue
        try:
            rdmol = Chem.AddHs(rdmol)
            result = AllChem.EmbedMultipleConfs(rdmol, numConfs=n_confs, randomSeed=seed)
            if result == -1:
                mol["conformers"] = []
                continue
            confs = []
            for cid in range(rdmol.GetNumConformers()):
                sdf_block = Chem.MolToMolBlock(rdmol, confId=cid)
                confs.append(sdf_block.encode("utf-8"))
            mol["conformers"] = confs
            success += 1
        except Exception:
            mol["conformers"] = []

    logger.info("chembl.conformers", n_success=success, n_total=len(molecules))
    return molecules


def save_jsonl(molecules: list[dict], output_dir: str, shard_size: int = 50000) -> None:
    """Save molecules to JSONL shards."""
    os.makedirs(output_dir, exist_ok=True)

    for shard_idx in range(0, len(molecules), shard_size):
        shard = molecules[shard_idx:shard_idx + shard_size]
        shard_path = os.path.join(output_dir, f"shard_{shard_idx // shard_size:04d}.jsonl")

        with open(shard_path, "w") as f:
            for mol in shard:
                # Strip conformer bytes for JSON (store count only)
                record = {
                    "smiles": mol["smiles"],
                    "chembl_id": mol["chembl_id"],
                    "n_conformers": len(mol.get("conformers", [])),
                }
                f.write(json.dumps(record) + "\n")

    logger.info("chembl.saved", n_molecules=len(molecules), output_dir=output_dir)


def save_manifest(molecules: list[dict], output_dir: str) -> None:
    """Save dataset manifest."""
    manifest = {
        "source": "ChEMBL 36",
        "n_molecules": len(molecules),
        "filters": {
            "mw_min": 200,
            "mw_max": 700,
            "require_carbon_or_nitrogen": True,
        },
        "format": "jsonl_sharded",
    }
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="ChEMBL 36 ingestion pipeline")
    parser.add_argument("--input", required=True, help="Path to ChEMBL 36 SQLite database")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max_mols", type=int, default=2_400_000)
    parser.add_argument("--confs", type=int, default=1, help="Conformers per molecule")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info("chembl_ingestion.start", input=args.input, output=args.output)

    # Find SQLite database
    db_path = args.input
    if os.path.isdir(db_path):
        for f in os.listdir(db_path):
            if f.endswith(".db") or f.endswith(".sqlite"):
                db_path = os.path.join(db_path, f)
                break

    if not os.path.exists(db_path):
        logger.error("chembl.db_not_found", path=db_path)
        return

    conn = sqlite3.connect(db_path)
    try:
        molecules = filter_molecules(conn, max_mols=args.max_mols)
        molecules = deduplicate(molecules)
        molecules = generate_conformers(molecules, n_confs=args.confs, seed=args.seed)
        save_jsonl(molecules, args.output)
        save_manifest(molecules, args.output)
    finally:
        conn.close()

    logger.info("chembl_ingestion.complete", n_molecules=len(molecules))


if __name__ == "__main__":
    main()
