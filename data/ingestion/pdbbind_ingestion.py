"""PDBBind protein-ligand complex ingestion pipeline.

Reads PDBBind data, extracts pocket point clouds and binding affinity,
and produces training data for the HUMU pocket encoder.

Usage:
    python data/ingestion/pdbbind_ingestion.py \
        --input zzzzz/PDBBind/ \
        --output data/processing/pdbbind_dataset/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_index(index_path: str, resolution_threshold: float = 2.5) -> list[dict]:
    """Parse PDBBind index file (INDEX_general_PL_data.2020).

    Returns entries filtered by resolution ≤ threshold.
    """
    entries = []
    with open(index_path) as f:
        header_skipped = False
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                header_skipped = True
                continue
            if not header_skipped:
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            pdb_id = parts[0]
            year = parts[1]
            resolution = parts[2]
            kdscore = parts[3]
            logkd_ki = parts[4] if len(parts) > 4 else None

            # Parse resolution
            try:
                res = float(resolution)
            except ValueError:
                continue

            if res > resolution_threshold:
                continue

            entries.append({
                "pdb_id": pdb_id,
                "year": year,
                "resolution": res,
                "log_kd_ki": logkd_ki,
            })

    logger.info("pdbbind.parsed_index", n_entries=len(entries))
    return entries


def extract_pocket(
    pdb_dir: str,
    pdb_id: str,
    ligand_name: str | None = None,
    pocket_radius: float = 10.0,
) -> dict | None:
    """Extract binding pocket from PDB complex.

    Finds the ligand in the PDB file and extracts protein atoms
    within pocket_radius Angstroms.
    """
    pdb_path = os.path.join(pdb_dir, f"{pdb_id}_protein.pdb")
    ligand_path = os.path.join(pdb_dir, f"{pdb_id}_ligand.mol2")

    # Fallback paths
    if not os.path.exists(pdb_path):
        pdb_path = os.path.join(pdb_dir, pdb_id, f"{pdb_id}_protein.pdb")
    if not os.path.exists(ligand_path):
        ligand_path = os.path.join(pdb_dir, pdb_id, f"{pdb_id}_ligand.sdf")

    if not os.path.exists(pdb_path) or not os.path.exists(ligand_path):
        return None

    try:
        # Parse ligand to get binding center
        try:
            from rdkit import Chem
            ligand = Chem.SDMolSupplier(ligand_path, sanitize=False)
            if ligand and ligand[0]:
                mol = ligand[0]
                conf = mol.GetConformer()
                positions = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
                center_x = sum(p.x for p in positions) / len(positions)
                center_y = sum(p.y for p in positions) / len(positions)
                center_z = sum(p.z for p in positions) / len(positions)
            else:
                return None
        except Exception:
            return None

        # Extract pocket atoms from protein PDB
        pocket_atoms = []
        residues = set()

        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    dist = ((x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2) ** 0.5
                    if dist <= pocket_radius:
                        element = line[76:78].strip()
                        residue = line[17:20].strip()
                        chain = line[21].strip()
                        res_seq = line[22:26].strip()
                        pocket_atoms.append({
                            "element": element or "C",
                            "x": x, "y": y, "z": z,
                            "residue": residue,
                        })
                        residues.add((chain, res_seq, residue))
                except (ValueError, IndexError):
                    continue

        return {
            "pdb_id": pdb_id,
            "pocket_atoms": pocket_atoms,
            "n_residues": len(residues),
            "center": [center_x, center_y, center_z],
        }
    except Exception as e:
        logger.warning("pdbbind.extract_failed", pdb_id=pdb_id, error=str(e))
        return None


def main():
    parser = argparse.ArgumentParser(description="PDBBind ingestion pipeline")
    parser.add_argument("--input", required=True, help="PDBBind data directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--resolution", type=float, default=2.5, help="Max resolution (Å)")
    parser.add_argument("--max_entries", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Find index file
    index_path = None
    for candidate in ["INDEX_general_PL_data.2020", "index/INDEX_general_PL_data.2020"]:
        full_path = os.path.join(args.input, candidate)
        if os.path.exists(full_path):
            index_path = full_path
            break

    if index_path is None:
        logger.error("pdbbind.index_not_found", dir=args.input)
        return

    entries = parse_index(index_path, args.resolution)
    if args.max_entries > 0:
        entries = entries[:args.max_entries]

    os.makedirs(args.output, exist_ok=True)

    processed = []
    failed = 0

    for i, entry in enumerate(entries):
        if (i + 1) % 500 == 0:
            logger.info("pdbbind.progress", processed=i, failed=failed)

        pocket = extract_pocket(args.input, entry["pdb_id"])
        if pocket is None:
            failed += 1
            continue

        processed.append({
            "pdb_id": entry["pdb_id"],
            "resolution": entry["resolution"],
            "log_kd_ki": entry["log_kd_ki"],
            "n_pocket_residues": pocket["n_residues"],
            "pocket_center": pocket["center"],
        })

        # Save pocket data
        pocket_path = os.path.join(args.output, f"pocket_{entry['pdb_id']}.json")
        with open(pocket_path, "w") as f:
            json.dump(pocket, f)

    # Save index
    with open(os.path.join(args.output, "index.jsonl"), "w") as f:
        for rec in processed:
            f.write(json.dumps(rec) + "\n")

    manifest = {
        "source": "PDBBind",
        "n_entries": len(entries),
        "n_processed": len(processed),
        "n_failed": failed,
        "resolution_threshold": args.resolution,
    }
    with open(os.path.join(args.output, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("pdbbind.complete", n_processed=len(processed), n_failed=failed)


if __name__ == "__main__":
    main()
