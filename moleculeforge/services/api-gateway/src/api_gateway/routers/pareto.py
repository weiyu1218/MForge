"""Pareto frontier endpoints backed by canonical Orchestrator snapshots."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException

from api_gateway.auth.oidc import AuthenticatedUser
from api_gateway.routers.design import orchestrator_get

router = APIRouter()


def _explicit_validation_passed(row: Mapping[str, Any]) -> bool:
    if "overall_passed" in row:
        return row.get("overall_passed") is True
    if "status" in row:
        return str(row.get("status")).strip().lower() == "validated"
    return row.get("valid") is True


def _order_by_validation_rank(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def rank_key(row: dict[str, Any]) -> tuple[int, float]:
        rank = row.get("rank")
        if isinstance(rank, bool):
            return (1, 0.0)
        try:
            return (0, float(rank))
        except (TypeError, ValueError):
            return (1, 0.0)

    return sorted(rows, key=rank_key)


async def _run_state(
    design_id: str,
    principal_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    snapshot, _ = await orchestrator_get(
        f"/v1/orchestrator/runs/{design_id}",
        principal_id=principal_id,
    )
    state_value = snapshot.get("state")
    state = state_value if isinstance(state_value, dict) else {}
    verified_value = next(
        (
            value
            for value in (
                snapshot.get("results"),
                state.get("results"),
                state.get("ranked"),
            )
            if isinstance(value, list)
        ),
        None,
    )
    if verified_value is not None:
        verified_rows = _order_by_validation_rank(
            [
                dict(row)
                for row in verified_value
                if isinstance(row, Mapping) and _explicit_validation_passed(row)
            ]
        )
        return {**snapshot, **state}, verified_rows
    candidates_value = state.get("candidates")
    candidates = (
        [dict(row) for row in candidates_value if isinstance(row, Mapping)]
        if isinstance(candidates_value, list)
        else []
    )
    validation = state.get("validation")
    validation_value = validation.get("results") or [] if isinstance(validation, dict) else []
    validation_rows = (
        [dict(row) for row in validation_value if isinstance(row, Mapping)]
        if isinstance(validation_value, list)
        else []
    )
    return {**snapshot, **state}, _merge_candidate_results(
        candidates,
        validation_rows,
        require_validated=True,
    )


def _merge_candidate_results(
    candidates: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    require_validated: bool = False,
) -> list[dict[str, Any]]:
    merged = [dict(candidate) for candidate in candidates]
    candidate_count = len(merged)
    by_candidate_id: dict[str, deque[int]] = {}
    by_candidate_id_and_smiles: dict[tuple[str, str], deque[int]] = {}
    by_smiles: dict[str, deque[int]] = {}
    for index, candidate in enumerate(merged):
        candidate_id = candidate.get("candidate_id")
        if candidate_id:
            by_candidate_id.setdefault(str(candidate_id), deque()).append(index)
        smiles = candidate.get("canonical_smiles") or candidate.get("smiles")
        if smiles:
            by_smiles.setdefault(str(smiles), deque()).append(index)
            if candidate_id:
                by_candidate_id_and_smiles.setdefault(
                    (str(candidate_id), str(smiles)),
                    deque(),
                ).append(index)
    explicit_matches: dict[int, int] = {}
    explicitly_matched_indices: set[int] = set()
    invalid_occurrence_rows: set[int] = set()
    for validation_index, validation in enumerate(validation_rows):
        if "candidate_index" not in validation:
            continue
        candidate_index = validation.get("candidate_index")
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or candidate_index < 0
            or candidate_index >= candidate_count
            or candidate_index in explicitly_matched_indices
        ):
            invalid_occurrence_rows.add(validation_index)
            continue
        candidate = merged[candidate_index]
        candidate_id = validation.get("candidate_id")
        validation_smiles = validation.get("canonical_smiles") or validation.get("smiles")
        if candidate_id not in (None, "") and str(candidate_id) != str(
            candidate.get("candidate_id") or ""
        ):
            invalid_occurrence_rows.add(validation_index)
            continue
        candidate_smiles = candidate.get("canonical_smiles") or candidate.get("smiles")
        if validation_smiles and str(validation_smiles) != str(candidate_smiles or ""):
            invalid_occurrence_rows.add(validation_index)
            continue
        explicit_matches[validation_index] = candidate_index
        explicitly_matched_indices.add(candidate_index)
    for validation_index, validation in enumerate(validation_rows):
        if validation_index in explicit_matches or validation_index in invalid_occurrence_rows:
            continue
        candidate_id = validation.get("candidate_id")
        validation_smiles = validation.get("canonical_smiles") or validation.get("smiles")
        if not candidate_id or not validation_smiles:
            continue
        candidates_for_id_and_smiles = by_candidate_id_and_smiles.get(
            (str(candidate_id), str(validation_smiles)),
            deque(),
        )
        while (
            candidates_for_id_and_smiles
            and candidates_for_id_and_smiles[0] in explicitly_matched_indices
        ):
            candidates_for_id_and_smiles.popleft()
        if candidates_for_id_and_smiles:
            index = candidates_for_id_and_smiles.popleft()
            explicit_matches[validation_index] = index
            explicitly_matched_indices.add(index)
    for validation_index, validation in enumerate(validation_rows):
        if validation_index in explicit_matches or validation_index in invalid_occurrence_rows:
            continue
        candidate_id = validation.get("candidate_id")
        if not candidate_id:
            continue
        candidates_for_id = by_candidate_id.get(str(candidate_id), deque())
        while candidates_for_id and candidates_for_id[0] in explicitly_matched_indices:
            candidates_for_id.popleft()
        if candidates_for_id:
            index = candidates_for_id.popleft()
            explicit_matches[validation_index] = index
            explicitly_matched_indices.add(index)
    reserved_indices = set(explicit_matches.values())
    claimed_indices: set[int] = set()
    validated_matches: list[tuple[int, int]] = []
    for validation_index, validation in enumerate(validation_rows):
        if validation_index in invalid_occurrence_rows:
            merged.append(dict(validation))
            continue
        matched_index = explicit_matches.get(validation_index)
        candidate_id = validation.get("candidate_id")
        canonical_smiles = validation.get("canonical_smiles") or validation.get("smiles")
        matched_by_candidate_id = matched_index is not None
        if candidate_id and matched_index is None:
            merged.append(dict(validation))
            continue
        if matched_index is None and canonical_smiles:
            candidates_for_smiles = by_smiles.get(str(canonical_smiles), deque())
            while candidates_for_smiles and (
                candidates_for_smiles[0] in reserved_indices
                or candidates_for_smiles[0] in claimed_indices
            ):
                candidates_for_smiles.popleft()
            if candidates_for_smiles:
                matched_index = candidates_for_smiles.popleft()
        if matched_index is None:
            merged.append(dict(validation))
            continue
        claimed_indices.add(matched_index)
        if _explicit_validation_passed(validation):
            validated_matches.append((validation_index, matched_index))
        candidate = merged[matched_index]
        candidate_properties = candidate.get("properties")
        validation_properties = validation.get("properties")
        merged_candidate = {
            **candidate,
            **validation,
            "properties": {
                **(candidate_properties if isinstance(candidate_properties, dict) else {}),
                **(validation_properties if isinstance(validation_properties, dict) else {}),
            },
        }
        if not matched_by_candidate_id and candidate.get("candidate_id"):
            merged_candidate["candidate_id"] = candidate["candidate_id"]
        merged[matched_index] = merged_candidate
    if require_validated:
        validated_rows = [
            merged[index] for _, index in validated_matches if index < candidate_count
        ]
        return _order_by_validation_rank(validated_rows)
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
async def get_pareto_frontier(
    design_id: str,
    authenticated_user: AuthenticatedUser,
) -> dict[str, Any]:
    state, results = await _run_state(design_id, str(authenticated_user["sub"]))
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
async def get_hypervolume(
    design_id: str,
    authenticated_user: AuthenticatedUser,
) -> dict[str, Any]:
    _, result_rows = await _run_state(design_id, str(authenticated_user["sub"]))
    results = [r for r in result_rows if _explicit_validation_passed(r)]
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
async def select_tradeoffs(
    design_id: str,
    request: dict,
    authenticated_user: AuthenticatedUser,
) -> dict[str, Any]:
    _, candidates = await _run_state(design_id, str(authenticated_user["sub"]))
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
        score -= weights.get("sa_score", 0.0) * float(_value(r, "sa_score", 10.0) or 10.0) / 10.0
        score -= weights.get("logp", 0.0) * abs(float(_value(r, "logp", 5.0) or 5.0) - 2.5) / 5.0
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
