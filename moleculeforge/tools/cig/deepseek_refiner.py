"""DeepSeek-backed CIG refinement command.

stdin: {"cig": {...}, "feedback": "...", "context": {...}}
stdout: {"cig": {...}, "hciv": {...}, "intent_cone": {...}, ...}
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cig_compiler_svc.domain.hciv_encoder import load_hciv_encoder_checkpoint
from cig_compiler_svc.domain.stages.stage2_cig_build import build_cig

from tools.cig.deepseek_semantic_parser import _deepseek_json, _validate_extracted_intent


_SYSTEM_PROMPT = """You are MoleculeForge's CIG refinement parser.
Return only a JSON object. Do not include markdown.
Given an existing CIG, feedback, and context, produce the refined extracted intent JSON
for downstream CIG building.
The top-level object must use exactly this schema:
- properties: list of objects with name, direction, priority
- targets: list of objects with name
- constraints: object such as max_mw, min_mw, lipinski_strict
- activity: object with type, direction, target_value
- admet_constraints: object such as oral_bioavailability_min, cyp3a4_ic50_min
- synthetic_constraints: object such as max_synthetic_steps
Do not wrap the response in a cig object. Do not return properties, constraints,
admet_constraints, or synthetic_constraints as arrays or strings. If feedback names
a molecular target, include it in targets with a name field.
"""


SemanticRefiner = Callable[[dict[str, Any]], dict[str, Any]]


def deepseek_refine_extracted_intent(payload: dict[str, Any]) -> dict[str, Any]:
    extracted = _deepseek_json(
        _SYSTEM_PROMPT,
        "Refine this CIG intent.\n"
        + json.dumps(payload, sort_keys=True, ensure_ascii=True),
        timeout_env="CIG_REFINEMENT_TIMEOUT_SECONDS",
    )
    extracted = _normalise_refined_intent(extracted)
    _validate_extracted_intent(extracted)
    return extracted


def refine_payload(
    payload: dict[str, Any],
    *,
    semantic_refiner: SemanticRefiner | None = None,
    hciv_checkpoint_path: str | None = None,
    dim: int = 128,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("refinement payload must be a JSON object")
    if not isinstance(payload.get("cig"), dict):
        raise RuntimeError("refinement payload requires cig object")
    if not isinstance(payload.get("feedback", ""), str):
        raise RuntimeError("refinement payload feedback must be a string")
    refiner = semantic_refiner or deepseek_refine_extracted_intent
    extracted = _normalise_refined_intent(refiner(payload))
    if not isinstance(extracted, dict):
        raise RuntimeError("semantic refiner must return a JSON object")
    _validate_extracted_intent(extracted)
    feedback = str(payload.get("feedback") or "")
    source = "refined CIG"
    if feedback:
        source = f"{source}: {feedback}"
    cig = build_cig(extracted, source=source)
    checkpoint_path = hciv_checkpoint_path or os.environ.get("HCIV_CHECKPOINT_PATH", "")
    if not checkpoint_path:
        raise RuntimeError("HCIV_CHECKPOINT_PATH is required for CIG refinement")
    encoder = load_hciv_encoder_checkpoint(checkpoint_path, dim=dim)
    hciv, cone = encoder.encode(cig)
    response = {
        "cig": cig.model_dump(mode="json", by_alias=True),
        "hciv": hciv.model_dump(mode="json"),
        "intent_cone": cone.model_dump(mode="json"),
        "ambiguities": [],
    }
    parse_confidence = extracted.get("parse_confidence")
    if parse_confidence is not None:
        response["parse_confidence"] = float(parse_confidence)
    return response


def _normalise_refined_intent(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("semantic refiner must return a JSON object")
    raw = payload.get("cig")
    if isinstance(raw, dict):
        payload = {key: value for key, value in raw.items() if key != "objective_nodes"}

    normalised = dict(payload)
    properties = normalised.get("properties")
    if properties is not None:
        if isinstance(properties, dict):
            properties = [
                {"name": str(name), **(value if isinstance(value, dict) else {})}
                for name, value in properties.items()
            ]
        if not isinstance(properties, list):
            raise RuntimeError("semantic parser properties must be a list")
        normalised["properties"] = [
            {"name": item}
            if isinstance(item, str)
            else item
            for item in properties
        ]

    targets = normalised.get("targets")
    if not targets:
        activity = normalised.get("activity")
        if isinstance(activity, dict) and isinstance(activity.get("target"), str):
            targets = [{"name": activity["target"]}]
            normalised["targets"] = targets
    if targets is not None:
        if not isinstance(targets, list):
            raise RuntimeError("semantic parser targets must be a list")
        normalised["targets"] = [
            _normalise_target(item)
            for item in targets
        ]

    constraints = normalised.get("constraints")
    if isinstance(constraints, list):
        normalised["constraints"] = _constraint_list_to_dict(constraints)
    elif isinstance(constraints, dict):
        normalised["constraints"] = {
            str(key): value
            for key, value in constraints.items()
            if value is not None
        }
    for field in ("admet_constraints", "synthetic_constraints"):
        constraints = normalised.get(field)
        if isinstance(constraints, list):
            normalised[field] = _constraint_list_to_dict(constraints)
        elif isinstance(constraints, dict):
            normalised[field] = {
                str(key): value
                for key, value in constraints.items()
                if value is not None
            }
    return normalised


def _constraint_list_to_dict(items: list[Any]) -> dict[str, Any]:
    return {
        str(item.get("name")): item.get("value", item.get("operator", True))
        for item in items
        if isinstance(item, dict) and item.get("name")
    }


def _normalise_target(item: Any) -> Any:
    if not isinstance(item, dict) or item.get("name"):
        return item
    for key in ("id", "target", "type"):
        value = item.get(key)
        if isinstance(value, str) and value:
            normalised = dict(item)
            normalised["name"] = value
            return normalised
    return item


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("stdin must be a JSON object") from exc
    print(json.dumps(refine_payload(payload), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
