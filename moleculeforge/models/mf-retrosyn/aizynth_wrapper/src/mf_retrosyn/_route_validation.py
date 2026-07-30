"""Validation helpers for retrosynthesis route runner outputs."""

from __future__ import annotations

import math
from typing import Any

REQUIRED_STEP_FIELDS = (
    "step_id",
    "reaction",
    "reaction_type",
    "reactants",
    "conditions",
    "yield",
    "building_blocks",
)
ASSESSMENT_ROUTE_TYPES = frozenset({"retrosynthetic_accessibility_score"})


class RetrosynRouteError(Exception):
    """Base error for an invalid planner route payload."""


class RetrosynRouteTypeError(TypeError, RetrosynRouteError):
    """Planner route payload has an invalid JSON type."""


class RetrosynRouteValueError(ValueError, RetrosynRouteError):
    """Planner route payload is missing executable information."""


def validate_retrosyn_routes(routes: Any, runner_name: str) -> list[dict]:
    executable_routes, _assessments = partition_retrosyn_results(routes, runner_name)
    return executable_routes


def partition_retrosyn_results(
    routes: Any,
    runner_name: str,
) -> tuple[list[dict], list[dict]]:
    if not isinstance(routes, list):
        raise RetrosynRouteTypeError(f"{runner_name} must return a list of route dictionaries")
    executable_routes: list[dict] = []
    assessments: list[dict] = []
    for route_index, route in enumerate(routes):
        if not isinstance(route, dict):
            raise RetrosynRouteTypeError(f"{runner_name} route {route_index} is not a dictionary")
        route_id = _required_text(
            route.get("route_id"),
            f"{runner_name} route {route_index} is missing route_id",
        )
        route_type = route.get("route_type")
        if route_type in ASSESSMENT_ROUTE_TYPES:
            _validate_assessment(route, runner_name, route_id)
            assessments.append(route)
            continue
        _validate_executable_route(route, runner_name, route_id)
        executable_routes.append(route)
    return executable_routes, assessments


def _validate_assessment(route: dict, runner_name: str, route_id: str) -> None:
    steps = route.get("steps")
    if steps not in (None, []):
        raise RetrosynRouteValueError(
            f"{runner_name} assessment {route_id} must not contain executable steps"
        )
    score = route.get("score", route.get("predicted_score"))
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise RetrosynRouteValueError(
            f"{runner_name} assessment {route_id} requires a finite score"
        )
    if not math.isfinite(float(score)):
        raise RetrosynRouteValueError(
            f"{runner_name} assessment {route_id} requires a finite score"
        )


def _validate_executable_route(route: dict, runner_name: str, route_id: str) -> None:
    steps = route.get("steps")
    if not isinstance(steps, list) or not steps:
        raise RetrosynRouteValueError(
            f"{runner_name} route {route_id} must contain non-empty steps"
        )
    step_ids: set[str] = set()
    for step_index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise RetrosynRouteTypeError(
                f"{runner_name} route {route_id} step {step_index} is not a dictionary"
            )
        _validate_step(step, runner_name, route_id, step_index)
        step_id = step["step_id"]
        if step_id in step_ids:
            raise RetrosynRouteValueError(
                f"{runner_name} route {route_id} has duplicate step_id {step_id}"
            )
        step_ids.add(step_id)


def _validate_step(
    step: dict,
    runner_name: str,
    route_id: str,
    step_index: int,
) -> None:
    prefix = f"{runner_name} route {route_id} step {step_index}"
    for field in REQUIRED_STEP_FIELDS:
        if _is_missing(step.get(field)):
            raise RetrosynRouteValueError(f"{prefix} is missing {field}")
    for field in ("step_id", "reaction", "reaction_type"):
        _required_text(step[field], f"{prefix} is missing {field}")
    _validate_reaction(step["reaction"], prefix)
    _validate_smiles_records(
        step["reactants"],
        f"{prefix} reactants",
        required_numeric_fields=("amount_mmol",),
    )
    if not isinstance(step["conditions"], dict):
        raise RetrosynRouteTypeError(f"{prefix} conditions must be a dictionary")
    for field in ("temperature_C", "time_h"):
        _required_finite_number(
            step["conditions"].get(field),
            f"{prefix} conditions requires numeric {field}",
        )
    _required_yield(
        step["yield"],
        f"{prefix} requires numeric yield in (0, 1]",
    )
    _validate_smiles_records(step["building_blocks"], f"{prefix} building_blocks")
    if "reagents" in step:
        reagents = step["reagents"]
        if not isinstance(reagents, list) or not all(
            isinstance(reagent, str) and reagent and reagent == reagent.strip()
            for reagent in reagents
        ):
            raise RetrosynRouteTypeError(f"{prefix} reagents must be a list of non-empty strings")
    if "purification" in step:
        _required_text(step["purification"], f"{prefix} purification must be non-empty")
    if "operation" in step:
        _required_text(step["operation"], f"{prefix} operation must be non-empty")


def _validate_reaction(reaction: str, prefix: str) -> None:
    parts = reaction.split(">>")
    if len(parts) != 2 or any(
        not side or not side.strip() or any(not component.strip() for component in side.split("."))
        for side in parts
    ):
        raise RetrosynRouteValueError(
            f"{prefix} reaction must contain non-empty reactants and products separated by one '>>'"
        )


def _validate_smiles_records(
    value: Any,
    field: str,
    *,
    required_numeric_fields: tuple[str, ...] = (),
) -> None:
    if not isinstance(value, list) or not value:
        raise RetrosynRouteValueError(f"{field} must be a non-empty list")
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise RetrosynRouteTypeError(f"{field}[{index}] must be a dictionary")
        _required_text(
            record.get("smiles"),
            f"{field}[{index}] requires smiles",
        )
        for numeric_field in required_numeric_fields:
            _required_finite_number(
                record.get(numeric_field),
                f"{field}[{index}] requires numeric {numeric_field}",
            )


def _required_finite_number(value: Any, message: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise RetrosynRouteValueError(message)
    return float(value)


def _required_yield(value: Any, message: str) -> float:
    parsed = _required_finite_number(value, message)
    if parsed <= 0 or parsed > 1:
        raise RetrosynRouteValueError(message)
    return parsed


def _required_text(value: Any, message: str) -> str:
    if not isinstance(value, str) or not value or not value.strip() or value != value.strip():
        raise RetrosynRouteValueError(message)
    return value


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}
