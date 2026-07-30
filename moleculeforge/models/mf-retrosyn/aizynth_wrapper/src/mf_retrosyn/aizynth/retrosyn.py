"""AiZynthFinder retrosynthetic route planner."""

import inspect
import os
from pathlib import Path
from typing import Any

from mf_retrosyn._route_validation import (
    RetrosynRouteError,
    RetrosynRouteValueError,
    validate_retrosyn_routes,
)


class AiZynthFinderRunner:
    """Thin adapter around the public AiZynthFinder Python interface."""

    def __init__(
        self,
        config_path: str,
        stock: str | None = None,
        expansion_policy: str | None = None,
        filter_policy: str | None = None,
    ):
        config = Path(config_path)
        if not config.is_file():
            raise FileNotFoundError(f"AiZynthFinder config file not found: {config}")
        try:
            from aizynthfinder.aizynthfinder import AiZynthFinder
        except ImportError as exc:
            raise RuntimeError(
                "aizynthfinder package is required for AiZynthFinder routes"
            ) from exc

        self.finder = AiZynthFinder(configfile=str(config))
        self.stock = stock
        self.expansion_policy = expansion_policy
        self.filter_policy = filter_policy
        self._select_if_configured(self.finder.stock, stock)
        self._select_if_configured(self.finder.expansion_policy, expansion_policy)
        self._select_if_configured(self.finder.filter_policy, filter_policy)

    def find_routes(self, smiles: str, max_routes: int = 10) -> list[dict]:
        if not isinstance(smiles, str) or not smiles:
            raise ValueError("smiles is required for AiZynthFinder route planning")
        self.finder.target_smiles = smiles
        self.finder.tree_search()
        self.finder.build_routes()
        route_dicts = _route_dicts_from_collection(self.finder.routes)
        routes = [
            _normalise_aizynth_route(route, smiles, idx) for idx, route in enumerate(route_dicts)
        ]
        return _validated_aizynth_routes(routes[:max_routes])

    @staticmethod
    def _select_if_configured(collection, name: str | None) -> None:
        if name:
            collection.select(name)


class AiZynthRetrosyn:
    """Retrosynthetic route planning using AiZynthFinder.

    Uses a Monte Carlo tree search with a learned expansion policy
    and rollout policy to find synthetic routes for target molecules.
    """

    def __init__(self, runner=None):
        self.runner = runner

    @classmethod
    def from_env(cls) -> "AiZynthRetrosyn":
        config_path = os.environ.get("AIZYNTH_CONFIG_PATH")
        if not config_path:
            raise RuntimeError("AIZYNTH_CONFIG_PATH is required for AiZynthFinder routes")
        return cls(
            runner=AiZynthFinderRunner(
                config_path=config_path,
                stock=os.environ.get("AIZYNTH_STOCK"),
                expansion_policy=os.environ.get("AIZYNTH_EXPANSION_POLICY"),
                filter_policy=os.environ.get("AIZYNTH_FILTER_POLICY"),
            )
        )

    async def find_routes(self, smiles: str, max_routes: int = 10) -> list[dict]:
        """Find retrosynthetic routes for a target molecule.

        Args:
            smiles: Target molecule SMILES string.
            max_routes: Maximum number of routes to return.

        Returns:
            List of route dictionaries with keys: 'route_id', 'smiles',
            'steps', 'score', 'reactions', 'intermediates', 'building_blocks'.
        """
        if self.runner is None:
            raise RuntimeError("AIZYNTH_RUNNER is required")
        result = self.runner.find_routes(smiles, max_routes=max_routes)
        if inspect.isawaitable(result):
            result = await result
        return _validated_aizynth_routes(result)


def _route_dicts_from_collection(routes: Any) -> list[dict]:
    if routes is None:
        return []
    dicts = getattr(routes, "dicts", None)
    if dicts is None and hasattr(routes, "make_dicts"):
        routes.make_dicts()
        dicts = getattr(routes, "dicts", None)
    if dicts is None:
        if isinstance(routes, list):
            return routes
        raise TypeError("AiZynthFinder routes must expose RouteCollection.dicts")
    return list(dicts)


def _normalise_aizynth_route(route: dict, smiles: str, index: int) -> dict:
    if not isinstance(route, dict):
        raise TypeError(f"AiZynthFinder route {index} is not a dictionary")
    if "route_id" in route and "steps" in route:
        return route
    steps = _extract_steps(route)
    normalized = {
        "route_id": str(route.get("route_id") or route.get("id") or f"aizynth-{index + 1}"),
        "smiles": str(route.get("smiles") or smiles),
        "score": _first_float(route, "score", "top_score", "predicted_score"),
        "steps": steps,
    }
    predicted_yield = _route_yield(route, steps)
    if predicted_yield is not None:
        normalized["predicted_yield"] = predicted_yield
    return normalized


def _complete_aizynth_route(route: dict) -> dict:
    smiles = str(route.get("smiles") or "")
    steps = route.get("steps")
    if not isinstance(steps, list):
        return route
    completed_steps = []
    for step in steps:
        if not isinstance(step, dict):
            completed_steps.append(step)
            continue
        completed = dict(step)
        reactants = (
            completed.get("reactants") if isinstance(completed.get("reactants"), list) else []
        )
        reactant_smiles = [
            str(item["smiles"])
            for item in reactants
            if isinstance(item, dict) and item.get("smiles")
        ]
        if not completed.get("reaction") and reactant_smiles and smiles:
            completed["reaction"] = f"{'.'.join(reactant_smiles)}>>{smiles}"
        if not completed.get("building_blocks") and reactant_smiles:
            completed["building_blocks"] = [{"smiles": item} for item in reactant_smiles]
        completed_steps.append(completed)
    completed_route = dict(route)
    completed_route["steps"] = completed_steps
    return completed_route


def _extract_steps(route: dict) -> list[dict]:
    steps = route.get("steps")
    if isinstance(steps, list):
        return steps
    extracted: list[dict] = []
    _walk_route_tree(route, extracted)
    return extracted


def _walk_route_tree(
    node: Any,
    steps: list[dict],
    product_smiles: str = "",
) -> None:
    if not isinstance(node, dict):
        return
    node_type = str(node.get("type", "")).lower()
    children = node.get("children") if isinstance(node.get("children"), list) else []
    if node_type in {"mol", "molecule"}:
        current_product = str(node.get("smiles") or product_smiles)
        for child in children:
            _walk_route_tree(child, steps, current_product)
        return
    if node_type == "reaction" or node.get("is_reaction") is True:
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        reactant_smiles = [
            str(child["smiles"])
            for child in children
            if isinstance(child, dict) and child.get("smiles")
        ]
        step = {
            "step_id": str(node.get("step_id") or f"retro-{len(steps) + 1}"),
            "reaction": _forward_reaction_smiles(
                node,
                reactant_smiles,
                product_smiles,
            ),
            "reaction_type": _reaction_type(metadata),
            "reactants": _reaction_metadata_reactants(metadata, reactant_smiles),
            "conditions": _reaction_conditions(metadata),
            "building_blocks": _reaction_building_blocks(metadata, reactant_smiles),
        }
        for field in (
            "yield",
            "yield_uncertainty",
            "reagents",
            "purification",
            "operation",
        ):
            if field in metadata:
                step[field] = metadata[field]
        steps.append(step)
    for child in children:
        _walk_route_tree(child, steps)


def _validated_aizynth_routes(routes: Any) -> list[dict]:
    try:
        return validate_retrosyn_routes(routes, "AiZynthFinder")
    except RetrosynRouteError as exc:
        raise RetrosynRouteValueError(
            f"AiZynthFinder routes unavailable for execution: {exc}"
        ) from exc


def _forward_reaction_smiles(
    node: dict,
    reactant_smiles: list[str],
    product_smiles: str,
) -> str:
    if reactant_smiles and product_smiles:
        return f"{'.'.join(reactant_smiles)}>>{product_smiles}"
    raw_reaction = str(
        node.get("reaction") or node.get("reaction_smiles") or node.get("smiles") or ""
    )
    parts = raw_reaction.split(">>")
    if len(parts) == 2 and parts[0] and parts[1]:
        return f"{parts[1]}>>{parts[0]}"
    return raw_reaction


def _reaction_type(metadata: dict) -> Any:
    for field in ("reaction_type", "reaction_class", "classification"):
        if field in metadata:
            return metadata[field]
    return None


def _reaction_metadata_reactants(
    metadata: dict,
    reactant_smiles: list[str],
) -> list[dict]:
    records = metadata.get("reactants")
    if isinstance(records, list):
        return [dict(record) if isinstance(record, dict) else record for record in records]
    return [{"smiles": smiles} for smiles in reactant_smiles]


def _reaction_conditions(metadata: dict) -> Any:
    conditions = metadata.get("conditions")
    if isinstance(conditions, dict):
        return dict(conditions)
    direct = {field: metadata[field] for field in ("temperature_C", "time_h") if field in metadata}
    return direct or None


def _reaction_building_blocks(
    metadata: dict,
    reactant_smiles: list[str],
) -> list[dict]:
    records = metadata.get("building_blocks")
    if isinstance(records, list):
        return [dict(record) if isinstance(record, dict) else record for record in records]
    return [{"smiles": smiles} for smiles in reactant_smiles]


def _first_float(data: dict, *keys: str) -> float:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _route_yield(route: dict, steps: list[dict]) -> float | None:
    for key in ("predicted_yield", "estimated_yield", "yield"):
        if key in route:
            return float(route[key])
    step_yields = [
        float(step["yield"]) for step in steps if isinstance(step, dict) and "yield" in step
    ]
    if len(step_yields) != len(steps) or not step_yields:
        return None
    total = 1.0
    for step_yield in step_yields:
        total *= step_yield
    return total
