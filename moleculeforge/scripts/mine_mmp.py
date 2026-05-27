"""Mine Matched Molecular Pairs from ChEMBL SQLite database.

Workflow:
  1. Extract SMILES from ChEMBL (filter: drug-like, has bioactivity)
  2. mmpdb fragment — fragment molecules at rotatable bonds
  3. mmpdb index   — build MMP pair database

Usage:
    python scripts/mine_mmp.py \
        --chembl-db C:/Users/guge0/Desktop/MForge/zzzzz/Chembl/chembl_36/chembl_36_sqlite/chembl_36.db \
        --output-dir data/mmpdb \
        --max-smiles 500000
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Drug-like filters (Lipinski-like)
_MAX_MW = 500
_MIN_HEAVY = 5


def extract_smiles(
    db_path: Path,
    output_smi: Path,
    max_smiles: int | None = None,
) -> int:
    """Extract drug-like SMILES from ChEMBL SQLite.

    Filters:
      - Non-null canonical_smiles
      - No '.' in SMILES (no mixtures/salts)
      - Heavy atom count >= 5
      - Has at least one bioactivity record

    Output format: one line per molecule, "SMILES\\tCHEMBL_ID"
    """
    logger.info("Connecting to ChEMBL: %s", db_path)
    db = sqlite3.connect(str(db_path))
    cur = db.cursor()

    query = """
        SELECT DISTINCT cs.canonical_smiles, md.chembl_id
        FROM compound_structures cs
        JOIN molecule_dictionary md ON cs.molregno = md.molregno
        JOIN activities a ON cs.molregno = a.molregno
        WHERE cs.canonical_smiles IS NOT NULL
          AND a.standard_value IS NOT NULL
    """
    if max_smiles:
        query += f" LIMIT {max_smiles}"

    logger.info("Executing query (may take a few minutes)...")
    cur.execute(query)

    count = 0
    skipped = 0
    with open(output_smi, "w", encoding="utf-8") as f:
        for smiles, chembl_id in cur:
            # Filter: no mixtures
            if "." in smiles:
                skipped += 1
                continue
            # Filter: minimum size (rough: >4 heavy atoms)
            if len(smiles) < 6:
                skipped += 1
                continue

            f.write(f"{smiles}\t{chembl_id}\n")
            count += 1

            if count % 100000 == 0:
                logger.info("  extracted %d molecules...", count)

    db.close()
    logger.info("Extracted %d SMILES (skipped %d) to %s", count, skipped, output_smi)
    return count


def run_mmpdb_fragment(
    smi_path: Path,
    fragdb_path: Path,
    num_jobs: int = 4,
) -> None:
    """Run mmpdb fragment command.

    mmpdb fragment breaks molecules at rotatable bonds and outputs
    a .fragdb file (not CSV).
    """
    logger.info("Running mmpdb fragment (num_jobs=%d)...", num_jobs)
    cmd = [
        sys.executable, "-m", "mmpdblib", "fragment",
        str(smi_path),
        "-o", str(fragdb_path),
        "--num-jobs", str(num_jobs),
        "--delimiter", "whitespace",
    ]
    logger.info("Command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"mmpdb fragment failed (exit {result.returncode}):\n{result.stderr[:2000]}"
        )
    logger.info("Fragmentation complete: %s", fragdb_path)


def run_mmpdb_index(
    fragdb_path: Path,
    mmpdb_path: Path,
) -> None:
    """Run mmpdb index command.

    mmpdb index builds the MMP pair database from fragments.
    """
    logger.info("Running mmpdb index...")
    cmd = [
        sys.executable, "-m", "mmpdblib", "index",
        str(fragdb_path),
        "-o", str(mmpdb_path),
    ]
    logger.info("Command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"mmpdb index failed (exit {result.returncode}):\n{result.stderr[:2000]}"
        )
    logger.info("MMP database created: %s", mmpdb_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine MMPs from ChEMBL")
    parser.add_argument(
        "--chembl-db",
        type=Path,
        default=Path("C:/Users/guge0/Desktop/MForge/zzzzz/Chembl/chembl_36/chembl_36_sqlite/chembl_36.db"),
        help="Path to ChEMBL SQLite database",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mmpdb"),
        help="Output directory for MMP database",
    )
    parser.add_argument(
        "--max-smiles",
        type=int,
        default=500000,
        help="Max molecules to extract (None = all)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip SMILES extraction (reuse existing .smi file)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    smi_path = args.output_dir / "chembl_molecules.smi"
    fragdb_path = args.output_dir / "chembl_molecules.fragdb"
    mmpdb_path = args.output_dir / "chembl_mmpdb.db"

    if not args.skip_extract or not smi_path.exists():
        extract_smiles(args.chembl_db, smi_path, args.max_smiles)

    if not fragdb_path.exists():
        run_mmpdb_fragment(smi_path, fragdb_path)

    if not mmpdb_path.exists():
        run_mmpdb_index(fragdb_path, mmpdb_path)

    logger.info("Done! MMP database at: %s", mmpdb_path)


if __name__ == "__main__":
    main()
