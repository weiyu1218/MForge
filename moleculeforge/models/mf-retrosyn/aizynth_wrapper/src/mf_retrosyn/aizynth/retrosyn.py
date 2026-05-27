"""AiZynthFinder retrosynthetic route planner."""
import inspect
import os
from pathlib import Path
from typing import Any

from mf_retrosyn._route_validation import validate_retrosyn_routes


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
            _normalise_aizynth_route(route, smiles, idx)
            for idx, route in enumerate(route_dicts)
        ]
        return routes[:max_routes]

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
        return validate_retrosyn_routes(result, "AiZynthRetrosyn")


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
    return {
        "route_id": str(route.get("route_id") or route.get("id") or f"aizynth-{index + 1}"),
        "smiles": str(route.get("smiles") or smiles),
        "score": _first_float(route, "score", "top_score", "predicted_score"),
        "predicted_yield": _first_float(route, "predicted_yield", "yield"),
        "steps": steps,
    }


def _extract_steps(route: dict) -> list[dict]:
    steps = route.get("steps")
    if isinstance(steps, list):
        return steps
    extracted: list[dict] = []
    _walk_route_tree(route, extracted)
    return extracted


def _walk_route_tree(node: Any, steps: list[dict]) -> None:
    if not isinstance(node, dict):
        return
    node_type = str(node.get("type", "")).lower()
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    reaction = (
        node.get("reaction")
        or node.get("reaction_smiles")
        or metadata.get("reaction_smiles")
    )
    children = node.get("children") if isinstance(node.get("children"), list) else []
    if reaction or node_type == "reaction":
        reactants = [
            {"smiles": str(child["smiles"])}
            for child in children
            if isinstance(child, dict) and child.get("smiles")
        ]
        steps.append(
            {
                "step_id": str(node.get("step_id") or f"retro-{len(steps) + 1}"),
                "reaction": str(reaction or metadata.get("smiles") or ""),
                "reactants": reactants,
                "conditions": metadata.get("conditions"),
                "building_blocks": metadata.get("building_blocks"),
            }
        )
    for child in children:
        _walk_route_tree(child, steps)


def _first_float(data: dict, *keys: str) -> float:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return float(value)
    return 0.0
