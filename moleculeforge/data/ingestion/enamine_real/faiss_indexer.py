"""Enamine REAL Space FAISS index builder.

Builds a FAISS IVF-PQ index over the Enamine REAL database (~49 billion
make-on-demand compounds) for supply-chain-aware molecular design.

Status: PLACEHOLDER — requires Enamine REAL access license.
To implement: contact Enamine Ltd. at https://enamine.net/
"""

import logging

logger = logging.getLogger(__name__)


def build_index(
    real_db_path: str,
    output_index_path: str,
    *,
    n_bits: int = 1024,
    ivf_nlist: int = 4096,
    pq_m: int = 64,
    pq_nbits: int = 8,
) -> dict:
    """Build FAISS IVF-PQ index over Enamine REAL Space.

    Args:
        real_db_path: Path to Enamine REAL database (SMILES + IDs)
        output_index_path: Path to save the FAISS index
        n_bits: ECFP4 fingerprint size (1024)
        ivf_nlist: Number of IVF clusters (4096 for 10^9 scale)
        pq_m: Product quantization sub-vectors (64)
        pq_nbits: Bits per PQ code (8)

    Returns:
        dict with keys: n_molecules, index_size_gb, build_time_seconds
    """
    logger.warning(
        "Enamine REAL FAISS indexer is a placeholder. Requires valid license."
    )
    # When implemented:
    # 1. Load Enamine REAL SMILES (streaming, too large for memory)
    # 2. Compute ECFP4 fingerprints in batches
    # 3. Train FAISS IVF-PQ index
    # 4. Add vectors in batches
    # 5. Save index to disk

    return {
        "n_molecules": 0,
        "index_size_gb": 0.0,
        "build_time_seconds": 0.0,
        "errors": ["PLACEHOLDER: Enamine REAL license not configured"],
    }
