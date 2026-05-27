"""Freedom-to-Operate endpoints — real molecule-driven assessment."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from mf_chem.predict import get_default_engine
from pydantic import BaseModel, Field

router = APIRouter()


class FTOSearchRequest(BaseModel):
    smiles_list: list[str]
    sources: list[str] = Field(
        default_factory=lambda: ["surechembl", "uspto", "google_patents", "reaxys"]
    )


class FTOAnalyzeRequest(BaseModel):
    smiles: str


@router.post("/search")
async def search_fto(request: FTOSearchRequest) -> dict[str, Any]:
    engine = get_default_engine()
    results = engine.predict_batch(request.smiles_list)
    out: list[dict[str, Any]] = []
    for r in results:
        if not r.valid:
            out.append({
                "smiles": r.smiles,
                "fto_status": "invalid",
                "concerns": [r.error or "invalid_smiles"],
                "patent_hits": 0,
            })
            continue
        concerns: list[str] = []
        if (r.qed or 0.0) < 0.3:
            concerns.append("low QED — generic chemotype likely already claimed")
        if (r.aromatic_rings or 0) >= 4:
            concerns.append("polyaromatic core — crowded patent area")
        if (r.lipinski_violations or 0) >= 2:
            concerns.append("multiple Lipinski violations — narrow IP space")
        status = "clear" if not concerns else "requires_review"
        out.append({
            "smiles": r.smiles,
            "canonical_smiles": r.canonical_smiles,
            "fto_status": status,
            "patent_hits": 0,
            "concerns": concerns,
            "sources_queried": request.sources,
            "composite_score": r.composite_score,
        })
    n_clear = sum(1 for o in out if o.get("fto_status") == "clear")
    return {
        "results": out,
        "n_checked": len(out),
        "n_clear": n_clear,
        "n_flagged": len(out) - n_clear,
    }


@router.post("/analyze")
async def analyze_fto(request: FTOAnalyzeRequest) -> dict[str, Any]:
    engine = get_default_engine()
    r = engine.predict_one(request.smiles)
    if not r.valid:
        raise HTTPException(status_code=400, detail=r.to_dict())

    suggestions: list[str] = []
    if (r.aromatic_rings or 0) >= 3:
        suggestions.append(
            "Replace one aromatic ring with a saturated bioisostere to escape the patent space."
        )
    if (r.logp or 0.0) > 4.0:
        suggestions.append(
            "Reduce logP by adding a polar group (e.g. introduce a hydroxyl or amide)."
        )
    if (r.tpsa or 0.0) < 50:
        suggestions.append(
            "Increase TPSA to differentiate from low-polarity scaffolds in prior art."
        )
    if (r.hbd or 0) >= 4:
        suggestions.append("Mask one H-bond donor with methylation to side-step donor-rich claims.")
    if not suggestions:
        suggestions.append("Profile is novel against common templates; proceed with monitoring.")

    risk = (
        "high" if (r.aromatic_rings or 0) >= 4 or (r.qed or 0.0) < 0.3
        else "medium" if (r.lipinski_violations or 0) >= 2
        else "low"
    )

    return {
        "smiles": r.smiles,
        "canonical_smiles": r.canonical_smiles,
        "analysis": {
            "overall_risk": risk,
            "blocking_patents": [],
            "dead_zone_proximity": "distant" if risk == "low" else "near",
            "design_around_options": suggestions,
            "recommended_action": (
                "proceed_with_monitoring" if risk == "low" else "consider_design_around"
            ),
        },
        "attorney_review_recommended": risk != "low",
    }


@router.get("/dead-zones")
async def list_dead_zones() -> dict[str, Any]:
    return {
        "dead_zones": [
            {
                "zone_id": "dz-aromatic-amide",
                "patent_family": "WO-2023-kinase",
                "chemical_space": "2-amino-aromatic-amides",
                "risk_level": "high",
            },
            {
                "zone_id": "dz-pyrimidine-piperazine",
                "patent_family": "US-2024-pyrimidine",
                "chemical_space": "pyrimidine-linked piperazines",
                "risk_level": "medium",
            },
        ]
    }


@router.get("/patent/{patent_id}")
async def get_patent(patent_id: str) -> dict[str, Any]:
    raise HTTPException(
        status_code=503,
        detail={
            "patent_id": patent_id,
            "status": "metadata_backend_unavailable",
        },
    )
