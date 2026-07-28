"""Pareto frontier endpoints backed by canonical Orchestrator snapshots."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException

from api_gateway.routers.design import orchestrator_get

router = APIRouter()


async def _run_state(design_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    snapshot, _ = await orchestrator_get(f"/v1/orchestrator/runs/{design_id}")
    state_value = snapshot.get("state")
    state = state_value if isinstance(state_value, dict) else {}
    candidates_value = (
        snapshot.get("results")
        or state.get("results")
        or state.get("ranked")
        or state.get("candidates")
        or []
    )
    candidates = [
        dict(row)
        for row in candidates_value
        if isinstance(row, Mapping)
    ] if isinstance(candidates_value, list) else []
    validation = state.get("validation")
    validation_value = (
        validation.get("results") or []
        if isinstance(validation, dict)
        else []
    )
    validation_rows = [
        dict(row)
        for row in validation_value
        if isinstance(row, Mapping)
    ] if isinstance(validation_value, list) else []
    return {**snapshot, **state}, _merge_candidate_results(
        candidates,
        validation_rows,
    )


def _merge_candidate_results(
    candidates: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = [dict(candidate) for candidate in candidates]
    by_candidate_id = {
        str(candidate["candidate_id"]): index
        for index, candidate in enumerate(merged)
        if candidate.get("candidate_id")
    }
    by_smiles = {
        str(candidate.get("canonical_smiles") or candidate.get("smiles")): index
        for index, candidate in enumerate(merged)
        if candidate.get("canonical_smiles") or candidate.get("smiles")
    }
    for validation in validation_rows:
        index = None
        candidate_id = validation.get("candidate_id")
        canonical_smiles = validation.get("canonical_smiles") or validation.get("smiles")
        if candidate_id:
            index = by_candidate_id.get(str(candidate_id))
        if index is None and canonical_smiles:
            index = by_smiles.get(str(canonical_smiles))
        if index is None:
            merged.append(dict(validation))
            continue
        candidate = merged[index]
        candidate_properties = candidate.get("properties")
        validation_properties = validation.get("properties")
        merged[index] = {
            **candidate,
            **validation,
            "properties": {
                **(
                    candidate_properties
                    if isinstance(candidate_properties, dict)
                    else {}
                ),
                **(
                    validation_properties
                    if isinstance(validation_properties, dict)
                    else {}
                ),
            },
        }
    return merged


def _value(row: dict[str, Any], key: str, fallback: object = None) -> object:
    properties = row.get("properties")
    if isinstance(properties, dict) and properties.get(key) is not None:
        return properties[key]
    if row.get(key) is not None:
        return row[key]
    return fallback


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
    state, results = await _run_state(design_id)
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
                    "qed": _value(r, "qed"),
                    "sa_score": _value(r, "sa_score"),
                    "logp": _value(r, "logp"),
                    "molecular_weight": _value(r, "molecular_weight"),
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
    _, result_rows = await _run_state(design_id)
    results = [r for r in result_rows if _value(r, "valid", True)]
    points = [
        (
            float(_value(r, "qed", 0.0) or 0.0),
            float(1.0 / max(1.0, float(_value(r, "sa_score", 10.0) or 10.0))),
        )
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
    _, candidates = await _run_state(design_id)
    weights = request.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise HTTPException(status_code=400, detail="weights is required")
    if any(
        isinstance(weight, bool) or not isinstance(weight, (int, float))
        for weight in weights.values()
    ):
        raise HTTPException(status_code=400, detail="weights must be numeric")

    def utility(r: dict) -> float:
        score = 0.0
        score += weights.get("qed", 0.0) * float(_value(r, "qed", 0.0) or 0.0)
        score -= (
            weights.get("sa_score", 0.0)
            * float(_value(r, "sa_score", 10.0) or 10.0)
            / 10.0
        )
        score -= (
            weights.get("logp", 0.0)
            * abs(float(_value(r, "logp", 5.0) or 5.0) - 2.5)
            / 5.0
        )
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
                "qed": _value(r, "qed"),
                "sa_score": _value(r, "sa_score"),
                "composite_score": r.get("composite_score"),
            }
            for r in top
        ],
    }
