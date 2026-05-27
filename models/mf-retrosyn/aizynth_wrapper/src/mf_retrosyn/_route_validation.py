"""Validation helpers for retrosynthesis route runner outputs."""

from __future__ import annotations

from typing import Any

REQUIRED_STEP_FIELDS = ("reaction", "reactants", "conditions", "building_blocks")


def validate_retrosyn_routes(routes: Any, runner_name: str) -> list[dict]:
    if not isinstance(routes, list):
        raise TypeError(f"{runner_name} must return a list of route dictionaries")
    for route_index, route in enumerate(routes):
        if not isinstance(route, dict):
            raise TypeError(f"{runner_name} route {route_index} is not a dictionary")
        route_id = route.get("route_id")
        if not route_id:
            raise ValueError(f"{runner_name} route {route_index} is missing route_id")
        steps = route.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"{runner_name} route {route_id} must contain non-empty steps")
        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                raise TypeError(
                    f"{runner_name} route {route_id} step {step_index} is not a dictionary"
                )
            for field in REQUIRED_STEP_FIELDS:
                if _is_missing(step.get(field)):
                    raise ValueError(
                        f"{runner_name} route {route_id} step {step_index} "
                        f"is missing {field}"
                    )
    return routes


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}
