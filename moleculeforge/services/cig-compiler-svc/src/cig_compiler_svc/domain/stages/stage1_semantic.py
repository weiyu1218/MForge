"""Stage 1: NL to structured entities via the shared nl2obj parser."""
from __future__ import annotations

from typing import Any

from nl2obj.parser import parse as parse_intent


PROPERTY_MAP = {
    "qed": ("qed", "maximize"),
    "sa": ("sa_score", "minimize"),
    "logp": ("logp", "target_range"),
    "solubility": ("solubility", "maximize"),
    "potency": ("binding_affinity", "maximize"),
    "selectivity": ("selectivity", "maximize"),
    "safety": ("safety", "maximize"),
}


def _heuristic_extract(nl_text: str) -> dict[str, Any]:
    parsed = parse_intent(nl_text)

    properties = _map_properties(parsed)
    constraints = _map_constraints(parsed.get("constraints", {}))

    return {
        "properties": properties,
        "constraints": constraints,
        "targets": parsed.get("target_details", []),
        "activity": parsed.get("activity", {"type": "", "direction": "", "target_value": None}),
        "admet_constraints": parsed.get(
            "admet_constraints",
            {"oral_bioavailability_min": None, "cyp3a4_ic50_min": None},
        ),
        "synthetic_constraints": parsed.get(
            "synthetic_constraints",
            {"max_synthetic_steps": 10},
        ),
    }


def _map_properties(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    properties: list[dict[str, Any]] = []
    seen: set[str] = set()

    for priority in parsed.get("objectives_priority", []):
        mapped = PROPERTY_MAP.get(priority)
        if mapped is None:
            continue
        name, direction = mapped
        if name in seen:
            continue
        seen.add(name)
        properties.append(
            {
                "name": name,
                "direction": direction,
                "priority": len(properties) + 1,
            }
        )

    activity = parsed.get("activity", {})
    if activity.get("type") and "binding_affinity" not in seen:
        properties.append(
            {
                "name": "binding_affinity",
                "direction": activity.get("direction", "maximize"),
                "priority": len(properties) + 1,
            }
        )

    return properties


def _map_constraints(parsed_constraints: dict[str, Any]) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    mw_range = parsed_constraints.get("molecular_weight")
    if isinstance(mw_range, list) and len(mw_range) == 2:
        low, high = mw_range
        if low is not None:
            constraints["min_mw"] = low
        if high is not None:
            constraints["max_mw"] = high
    if parsed_constraints.get("lipinski_strict"):
        constraints["lipinski_strict"] = True
    return constraints
