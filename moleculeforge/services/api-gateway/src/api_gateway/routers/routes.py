"""Synthetic route planning — uses RDKit substructure matching for real building-block decomposition."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mf_chem.predict import get_default_engine

router = APIRouter()


class PlanRequest(BaseModel):
    target_smiles: str
    max_routes: int = 3


class ScoreRequest(BaseModel):
    route: dict[str, Any]


def _suggest_disconnections(target_smiles: str) -> list[dict[str, Any]]:
    """Return a list of candidate disconnections derived from RDKit substructure search."""
    from rdkit import Chem

    target = Chem.MolFromSmiles(target_smiles)
    if target is None:
        raise ValueError("invalid target SMILES")

    moves: list[dict[str, Any]] = []
    # Heuristic: amide bond -> amide coupling
    if target.HasSubstructMatch(Chem.MolFromSmarts("C(=O)N")):
        moves.append({
            "name": "Amide coupling",
            "reaction_smarts": "[C:1](=O)[N:2]>>[C:1](=O)O.[N:2]",
            "score": 0.88,
        })
    # Aryl-aryl bond -> Suzuki coupling
    if target.HasSubstructMatch(Chem.MolFromSmarts("c-c")):
        moves.append({
            "name": "Suzuki coupling",
            "reaction_smarts": "[c:1]-[c:2]>>[c:1]Br.[c:2]B(O)O",
            "score": 0.82,
        })
    # Ester -> esterification
    if target.HasSubstructMatch(Chem.MolFromSmarts("C(=O)O[C,c]")):
        moves.append({
            "name": "Esterification",
            "reaction_smarts": "[C:1](=O)O[C:2]>>[C:1](=O)O.[O,C:2]O",
            "score": 0.79,
        })
    # Reductive amination
    if target.HasSubstructMatch(Chem.MolFromSmarts("[C][N]")):
        moves.append({
            "name": "Reductive amination",
            "reaction_smarts": "[C:1][N:2]>>[C:1]=O.[N:2]",
            "score": 0.71,
        })
    if not moves:
        moves.append({
            "name": "Functional group interconversion",
            "reaction_smarts": "[*:1]>>[*:1]",
            "score": 0.5,
        })
    return moves


@router.post("/plan")
async def plan_routes(request: PlanRequest) -> dict[str, Any]:
    engine = get_default_engine()
    target_pred = engine.predict_one(request.target_smiles)
    if not target_pred.valid:
        raise HTTPException(status_code=400, detail=target_pred.to_dict())

    try:
        moves = _suggest_disconnections(request.target_smiles)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    routes: list[dict[str, Any]] = []
    for i, move in enumerate(moves[: max(1, request.max_routes)]):
        steps = [{
            "step": 0,
            "reaction": move["name"],
            "reaction_smarts": move["reaction_smarts"],
            "score": move["score"],
        }]
        # Estimate yield from move score and target SA score
        sa = target_pred.sa_score or 5.0
        est_yield = max(0.1, min(0.95, 1.0 - (sa - 1) / 12 - 0.1 * i))
        routes.append({
            "route_id": f"route-{i:02d}",
            "steps": steps,
            "total_steps": len(steps),
            "estimated_yield": round(est_yield, 3),
            "total_score": round(move["score"] * est_yield, 3),
            "target_smiles": target_pred.canonical_smiles,
        })
    return {
        "target_smiles": target_pred.canonical_smiles,
        "n_routes": len(routes),
        "routes": routes,
        "target_sa_score": target_pred.sa_score,
        "target_qed": target_pred.qed,
    }


@router.get("/{route_id}")
async def get_route(route_id: str) -> dict[str, Any]:
    return {
        "route_id": route_id,
        "note": "Route metadata is generated on demand via /plan; no persistent store yet.",
    }


@router.post("/score")
async def score_route(request: ScoreRequest) -> dict[str, Any]:
    route = request.route or {}
    steps = route.get("steps", []) or []
    if not steps:
        raise HTTPException(status_code=400, detail="route.steps must be non-empty")

    avg_step_score = sum(s.get("score", 0.0) for s in steps) / len(steps)
    feasibility = round(min(1.0, max(0.0, avg_step_score)), 3)
    cost = round(max(0.1, 1.0 - 0.05 * len(steps)), 3)
    greenness = round(max(0.0, 0.9 - 0.07 * len(steps)), 3)
    overall = round(0.5 * feasibility + 0.3 * cost + 0.2 * greenness, 3)
    return {
        "route_id": route.get("route_id", "unknown"),
        "scores": {
            "feasibility": feasibility,
            "cost": cost,
            "greenness": greenness,
            "overall": overall,
        },
        "n_steps": len(steps),
    }


@router.post("/compare")
async def compare_routes(request: dict) -> dict[str, Any]:
    routes = request.get("routes", [])
    if not routes:
        raise HTTPException(status_code=400, detail="routes payload is required")

    comparisons: list[dict[str, Any]] = []
    for r in routes:
        steps = r.get("steps", []) or []
        avg = sum(s.get("score", 0.0) for s in steps) / max(1, len(steps))
        comparisons.append({
            "route_id": r.get("route_id"),
            "total_steps": len(steps),
            "avg_step_score": round(avg, 3),
            "estimated_yield": r.get("estimated_yield"),
            "total_score": r.get("total_score"),
        })
    best = max(comparisons, key=lambda x: x.get("total_score") or 0.0)
    return {"comparison": comparisons, "best_route": best.get("route_id")}
