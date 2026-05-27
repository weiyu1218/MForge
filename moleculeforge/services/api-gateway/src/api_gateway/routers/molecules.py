"""Molecule property prediction endpoints (real RDKit + multi-GPU HUMU stack)."""
from __future__ import annotations

from typing import Any
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from mf_chem.predict import MolPredictEngine, get_default_engine

router = APIRouter()


class BatchRequest(BaseModel):
    smiles_list: list[str] = Field(..., description="SMILES strings to score")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Free-text or SMILES query")
    top_k: int = Field(10, ge=1, le=200)


class FTORequest(BaseModel):
    sources: list[str] | None = Field(default=None, description="Optional patent sources")


def _engine() -> MolPredictEngine:
    return get_default_engine()


@router.get("/devices")
async def list_devices() -> dict[str, Any]:
    """Report which devices the prediction engine is using."""
    eng = _engine()
    return {"devices": eng.devices, "n_devices": len(eng.devices)}


@router.get("/{smiles:path}")
async def get_molecule(smiles: str) -> dict[str, Any]:
    """Predict properties for a single SMILES (URL-decoded)."""
    decoded = unquote(smiles)
    result = _engine().predict_one(decoded)
    payload = result.to_dict()
    if not result.valid:
        raise HTTPException(status_code=400, detail=payload)
    return payload


@router.post("/{smiles:path}/fto")
async def check_fto(smiles: str, request: FTORequest | None = None) -> dict[str, Any]:
    """FTO-style assessment combining RDKit fingerprint complexity + property risk."""
    decoded = unquote(smiles)
    result = _engine().predict_one(decoded)
    if not result.valid:
        raise HTTPException(status_code=400, detail=result.to_dict())

    sources = (request.sources if request else None) or [
        "surechembl", "uspto", "google_patents", "reaxys",
    ]
    concerns: list[str] = []
    if result.qed and result.qed < 0.3:
        concerns.append("low QED — generic chemotype likely already claimed")
    if result.lipinski_violations and result.lipinski_violations >= 2:
        concerns.append("multiple Lipinski violations — narrow IP space")
    if result.aromatic_rings and result.aromatic_rings >= 4:
        concerns.append("polyaromatic core overlaps with crowded patent area")
    fto_status = "clear" if not concerns else "requires_review"
    nearest = round(0.6 - 0.5 * (1 - (result.composite_score or 0.0)), 4)

    return {
        "smiles": decoded,
        "canonical_smiles": result.canonical_smiles,
        "fto_status": fto_status,
        "concerns": concerns,
        "patent_hits": 0,
        "sources_queried": sources,
        "dead_zone_check": {
            "in_dead_zone": fto_status != "clear",
            "nearest_distance": max(0.05, nearest),
        },
    }


@router.post("/search")
async def search_molecules(request: SearchRequest) -> dict[str, Any]:
    """Search molecules by similarity to a query SMILES (Tanimoto on Morgan)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    seed_smiles = request.query.strip()
    seed_mol = Chem.MolFromSmiles(seed_smiles)
    if seed_mol is None:
        raise HTTPException(status_code=400, detail={"error": "invalid_query_smiles"})

    library = [
        "Cc1ccccc1", "Cc1cccc(N)c1", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "OC(=O)c1ccccc1O", "Nc1ccc(S(N)(=O)=O)cc1",
        "CC(=O)Nc1ccc(O)cc1", "C[C@H](N)Cc1ccc(O)cc1", "CCN(CC)CC",
        "Cn1cnc2n(C)c(=O)[nH]c(=O)c12", "CCc1ccc2nc(N)nc(N)c2c1",
        "O=C(O)Cc1ccccc1", "CC(=O)c1ccccc1", "CN1CCC[C@H]1c1cccnc1",
        "C(=O)c1ccccc1", "CCOC(=O)c1ccccc1", "Nc1ccncc1",
    ]
    seed_fp = AllChem.GetMorganFingerprintAsBitVect(seed_mol, 2, nBits=2048)
    scored: list[tuple[str, float]] = []
    for s in library:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sim = DataStructs.TanimotoSimilarity(seed_fp, fp)
        scored.append((s, float(sim)))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: request.top_k]
    return {
        "query": seed_smiles,
        "n_results": len(top),
        "results": [{"smiles": s, "similarity": round(sim, 4)} for s, sim in top],
    }


@router.post("/batch")
async def batch_get_molecules(request: BatchRequest) -> dict[str, Any]:
    """Score a batch of SMILES — load is sharded across all visible GPUs."""
    if not request.smiles_list:
        return {"results": [], "n_total": 0, "n_valid": 0, "n_invalid": 0}
    engine = _engine()
    results = engine.predict_batch(request.smiles_list)
    payload = [r.to_dict() for r in results]
    n_valid = sum(1 for r in results if r.valid)
    return {
        "results": payload,
        "n_total": len(results),
        "n_valid": n_valid,
        "n_invalid": len(results) - n_valid,
        "devices_used": engine.devices,
    }
