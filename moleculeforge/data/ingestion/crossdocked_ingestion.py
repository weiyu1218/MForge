"""CrossDocked2020 pocket data ingestion pipeline.

Reads CrossDocked2020 v1.3 data, extracts pocket point clouds from
protein structures around ligand binding sites, and prepares training
data for the HUMU pocket encoder.

Usage:
    python data/ingestion/crossdocked_ingestion.py \
        --input zzzzz/CrossDocked2020/ \
        --output data/processing/crossdocked_pocket_dataset/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_types_file(types_path: str) -> list[dict]:
    """Parse CrossDocked2020 types file.

    Each line contains:
      <ligand_pdb> <protein_pdb> <vina_score> [<rmsd> ...]
    """
    entries = []
    with open(types_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            entries.append({
                "ligand_pdb": parts[0],
                "protein_pdb": parts[1],
                "vina_score": float(parts[2]),
                "rmsd": float(parts[3]) if len(parts) > 3 else None,
            })
    logger.info("crossdocked.parsed_types", n_entries=len(entries))
    return entries


def extract_pocket_residues(
    protein_path: str,
    ligand_path: str,
    pocket_radius: float = 10.0,
) -> dict | None:
    """Extract pocket residues within radius of ligand atoms.

    Returns dict with:
      - pocket_atoms: list of {element, x, y, z, residue}
      - n_residues: number of unique residues
      - center: [x, y, z] centroid of pocket
    """
    try:
        from rdkit import Chem
    except ImportError:
        return None

    if not os.path.exists(protein_path) or not os.path.exists(ligand_path):
        return None

    try:
        ligand = Chem.MolFromPDBFile(ligand_path, sanitize=False)
        if ligand is None:
            return None

        # Get ligand centroid
        conf = ligand.GetConformer()
        positions = [conf.GetAtomPosition(i) for i in range(ligand.GetNumAtoms())]
        center_x = sum(p.x for p in positions) / len(positions)
        center_y = sum(p.y for p in positions) / len(positions)
        center_z = sum(p.z for p in positions) / len(positions)

        # Parse protein PDB manually for pocket atoms
        pocket_atoms = []
        residues_seen = set()

        with open(protein_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                atom_name = line[12:16].strip()
                residue = line[17:20].strip()
                chain = line[21].strip()
                res_seq = line[22:26].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip()

                # Distance from ligand center
                dist = ((x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2) ** 0.5
                if dist <= pocket_radius:
                    pocket_atoms.append({
                        "element": element or atom_name[0],
                        "x": x,
                        "y": y,
                        "z": z,
                        "residue": residue,
                        "chain": chain,
                        "res_seq": int(res_seq),
                    })
                    residues_seen.add((chain, res_seq, residue))

        return {
            "pocket_atoms": pocket_atoms,
            "n_residues": len(residues_seen),
            "center": [center_x, center_y, center_z],
        }
    except Exception as e:
        logger.warning("crossdocked.extract_pocket_failed", error=str(e))
        return None


def process_split(
    data_dir: str,
    types_path: str,
    output_dir: str,
    max_entries: int = 0,
) -> None:
    """Process a CrossDocked2020 split.

    Args:
        data_dir: root directory containing PDB files.
        types_path: path to types file.
        output_dir: output directory for processed data.
        max_entries: max entries to process (0 = all).
    """
    entries = parse_types_file(types_path)

    if max_entries > 0:
        entries = entries[:max_entries]

    os.makedirs(output_dir, exist_ok=True)

    processed = []
    failed = 0

    for i, entry in enumerate(entries):
        if (i + 1) % 1000 == 0:
            logger.info("crossdocked.progress", processed=i, failed=failed)

        ligand_path = os.path.join(data_dir, entry["ligand_pdb"])
        protein_path = os.path.join(data_dir, entry["protein_pdb"])

        pocket = extract_pocket_residues(protein_path, ligand_path)
        if pocket is None:
            failed += 1
            continue

        processed.append({
            "index": i,
            "ligand_pdb": entry["ligand_pdb"],
            "protein_pdb": entry["protein_pdb"],
            "vina_score": entry["vina_score"],
            "rmsd": entry.get("rmsd"),
            "pocket_center": pocket["center"],
            "n_pocket_residues": pocket["n_residues"],
            "n_pocket_atoms": len(pocket["pocket_atoms"]),
        })

        # Save pocket atoms as separate JSON (too large for inline)
        pocket_path = os.path.join(output_dir, f"pocket_{i:06d}.json")
        with open(pocket_path, "w") as f:
            json.dump(pocket, f)

    # Save index
    index_path = os.path.join(output_dir, "index.jsonl")
    with open(index_path, "w") as f:
        for rec in processed:
            f.write(json.dumps(rec) + "\n")

    # Save manifest
    manifest = {
        "source": "CrossDocked2020 v1.3",
        "n_entries": len(entries),
        "n_processed": len(processed),
        "n_failed": failed,
        "pocket_radius": 10.0,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("crossdocked.complete", n_processed=len(processed), n_failed=failed)


def main():
    parser = argparse.ArgumentParser(description="CrossDocked2020 ingestion")
    parser.add_argument("--input", required=True, help="CrossDocked2020 data directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max_entries", type=int, default=0, help="Max entries (0=all)")
    parser.add_argument("--types", default=None, help="Types file path (auto-detect if None)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Auto-detect types file
    types_path = args.types
    if types_path is None:
        for f in os.listdir(args.input):
            if "types" in f.lower() and f.endswith(".txt"):
                types_path = os.path.join(args.input, f)
                break

    if types_path is None or not os.path.exists(types_path):
        logger.error("crossdocked.types_not_found", dir=args.input)
        return

    process_split(args.input, types_path, args.output, args.max_entries)


if __name__ == "__main__":
    main()
