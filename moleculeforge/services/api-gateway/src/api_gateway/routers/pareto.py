"""Pareto frontier endpoints — read from the in-memory design store."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from api_gateway.routers.design import _designs

router = APIRouter()


def _hypervolume_2d(points: list[tuple[float, float]], reference: tuple[float, float]) -> float:
    """Simple 2D hypervolume (maximisation) for sanity checking the front."""
    if not points:
        return 0.0
    points = sorted(points, key=lambda p: -p[0])
    hv = 0.0
    prev_y = reference[1]
    for x, y in points:
        if x <= reference[0] or y <= reference[1]:
            continue
        hv += max(0.0, x - reference[0]) * max(0.0, y - prev_y)
        prev_y = y
    return float(hv)


@router.get("/{design_id}/frontier")
async def get_pareto_frontier(design_id: str) -> dict[str, Any]:
    state = _designs.get(design_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Design not found")
    results = state.get("results", []) or []
    front = [r for r in results if r.get("pareto_optimal")]
    if not front:
        # Fallback: top 10 by composite_score so the chart is never empty.
        front = sorted(results, key=lambda r: -(r.get("composite_score") or 0.0))[:10]
    return {
        "design_id": design_id,
        "frontier": [
            {
                "rank": r.get("rank"),
                "smiles": r.get("canonical_smiles") or r.get("smiles"),
                "objectives": {
                    "qed": r.get("qed"),
                    "sa_score": r.get("sa_score"),
                    "logp": r.get("logp"),
                    "molecular_weight": r.get("molecular_weight"),
                },
                "composite_score": r.get("composite_score"),
                "humu_norm": r.get("humu_embedding_norm"),
            }
            for r in front
        ],
        "n_points": len(front),
        "objectives": state.get("objectives", []),
    }


@router.get("/{design_id}/hypervolume")
async def get_hypervolume(design_id: str) -> dict[str, Any]:
    state = _designs.get(design_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Design not found")
    results = [r for r in state.get("results", []) if r.get("valid")]
    points = [
        (float(r.get("qed") or 0.0), float(1.0 / max(1.0, r.get("sa_score") or 10.0)))
        for r in results
    ]
    hv = _hypervolume_2d(points, reference=(0.0, 0.05))
    return {
        "design_id": design_id,
        "hypervolume": round(hv, 4),
        "reference_point": {"qed": 0.0, "inverse_sa": 0.05},
        "n_points": len(points),
    }


@router.post("/{design_id}/select")
async def select_tradeoffs(design_id: str, request: dict) -> dict[str, Any]:
    state = _designs.get(design_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Design not found")
    weights = request.get("weights", {"qed": 0.5, "sa_score": 0.3, "logp": 0.2})
    candidates = state.get("results", [])

    def utility(r: dict) -> float:
        score = 0.0
        score += weights.get("qed", 0.0) * float(r.get("qed") or 0.0)
        score -= weights.get("sa_score", 0.0) * float(r.get("sa_score") or 10.0) / 10.0
        score -= weights.get("logp", 0.0) * abs(float(r.get("logp") or 5.0) - 2.5) / 5.0
        return score

    if not candidates:
        return {"design_id": design_id, "selected": [], "method": "weighted_sum"}
    ranked = sorted(candidates, key=utility, reverse=True)
    top = ranked[: int(request.get("top_k", 3))]
    return {
        "design_id": design_id,
        "method": "weighted_sum",
        "weights_applied": weights,
        "selected": [
            {
                "smiles": r.get("canonical_smiles") or r.get("smiles"),
                "score": round(utility(r), 4),
                "qed": r.get("qed"),
                "sa_score": r.get("sa_score"),
                "composite_score": r.get("composite_score"),
            }
            for r in top
        ],
    }
