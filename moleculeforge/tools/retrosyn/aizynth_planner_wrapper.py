#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import copy
import inspect
import json
import os
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AIZYNTH_SRC = ROOT / "models" / "mf-retrosyn" / "aizynth_wrapper" / "src"
if str(AIZYNTH_SRC) not in sys.path:
    sys.path.append(str(AIZYNTH_SRC))


def main() -> int:
    try:
        response = asyncio.run(_run(_read_request()))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    json.dump(response, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _read_request() -> dict[str, object]:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise RuntimeError("AiZynth planner wrapper requires JSON stdin") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("AiZynth planner wrapper request must be a JSON object")
    return payload


async def _run(payload: dict[str, object]) -> dict[str, object]:
    smiles = str(payload.get("smiles") or "")
    if not smiles:
        raise RuntimeError("AiZynth planner wrapper requires smiles")
    max_routes = int(payload.get("max_routes") or 10)
    if max_routes <= 0:
        raise RuntimeError("AiZynth planner wrapper requires max_routes > 0")
    engine = str(payload.get("engine") or "aizynth").strip().lower()
    if engine not in {"aizynth", "aizynthfinder"}:
        raise RuntimeError(f"Unsupported AiZynth planner engine: {engine}")

    start = time.perf_counter()
    if os.environ.get("AIZYNTH_STOCK_SMILES", "").strip():
        routes = _find_routes_with_inline_stock(smiles, max_routes=max_routes)
    else:
        from mf_retrosyn.aizynth.retrosyn import AiZynthRetrosyn

        planner = AiZynthRetrosyn.from_env()
        routes = planner.find_routes(smiles, max_routes=max_routes)
        if inspect.isawaitable(routes):
            routes = await routes
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    normalized = _validated_routes(routes)
    return {
        "routes": normalized,
        "total_routes_found": len(normalized),
        "elapsed_ms": elapsed_ms,
    }


def _find_routes_with_inline_stock(smiles: str, max_routes: int) -> list[dict[str, object]]:
    _install_optional_import_shims()

    from aizynthfinder.aizynthfinder import AiZynthFinder
    from mf_retrosyn.aizynth.retrosyn import (
        _complete_aizynth_route,
        _normalise_aizynth_route,
        _route_dicts_from_collection,
    )

    finder = AiZynthFinder(configdict=_inline_stock_config(max_routes))
    _select_if_configured(finder.expansion_policy, os.environ.get("AIZYNTH_EXPANSION_POLICY"))
    _select_if_configured(finder.filter_policy, os.environ.get("AIZYNTH_FILTER_POLICY"))
    stock_name = os.environ.get("AIZYNTH_STOCK", "").strip() or "inline"
    finder.stock.load(_inline_stock_query(), stock_name)
    finder.stock.select(stock_name)
    finder.target_smiles = smiles
    finder.tree_search()
    finder.build_routes()
    return [
        _complete_aizynth_route(
            _normalise_aizynth_route(_preserve_reaction_node_smiles(route), smiles, index)
        )
        for index, route in enumerate(_route_dicts_from_collection(finder.routes))
    ][:max_routes]


def _preserve_reaction_node_smiles(route: dict[str, object]) -> dict[str, object]:
    preserved = dict(route)
    if str(preserved.get("type", "")).lower() == "reaction" and preserved.get("smiles"):
        preserved.setdefault("reaction", preserved["smiles"])
    children = preserved.get("children")
    if isinstance(children, list):
        preserved["children"] = [
            _preserve_reaction_node_smiles(child) if isinstance(child, dict) else child
            for child in children
        ]
    return preserved


def _install_optional_import_shims() -> None:
    sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))
    sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))
    sys.modules.setdefault("tensorflow_serving", types.ModuleType("tensorflow_serving"))

    if "rxnutils.routes.scoring" not in sys.modules:
        scoring = types.ModuleType("rxnutils.routes.scoring")
        scoring.DeepsetModelClient = _UnavailableOptionalRouteFeature
        scoring.deepset_route_score = _unavailable_optional_route_feature
        scoring.reaction_class_rank_score = _unavailable_optional_route_feature
        sys.modules["rxnutils.routes.scoring"] = scoring

    if "rxnutils.routes.comparison" not in sys.modules:
        comparison = types.ModuleType("rxnutils.routes.comparison")
        comparison.simple_route_similarity = _unavailable_optional_route_feature
        sys.modules["rxnutils.routes.comparison"] = comparison

    if "rxnutils.routes.image" not in sys.modules:
        image = types.ModuleType("rxnutils.routes.image")
        image.RouteImageFactory = _UnavailableOptionalRouteFeature
        image.molecule_to_image = _unavailable_optional_route_feature
        image.molecules_to_images = _unavailable_optional_route_feature
        sys.modules["rxnutils.routes.image"] = image

    if "rxnutils.routes.readers" not in sys.modules:
        readers = types.ModuleType("rxnutils.routes.readers")
        readers.read_aizynthfinder_dict = _unavailable_optional_route_feature
        sys.modules["rxnutils.routes.readers"] = readers


class _UnavailableOptionalRouteFeature:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("AiZynth optional route feature is unavailable in command smoke mode")


def _unavailable_optional_route_feature(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError("AiZynth optional route feature is unavailable in command smoke mode")


def _inline_stock_config(max_routes: int) -> dict[str, object]:
    config = _aizynth_config_from_env()
    config.pop("stock", None)
    search_config = _dict_setting(config, "search")
    _set_int_env(search_config, "iteration_limit", "AIZYNTH_SEARCH_ITERATION_LIMIT")
    _set_int_env(search_config, "time_limit", "AIZYNTH_SEARCH_TIME_LIMIT")
    _set_int_env(search_config, "max_transforms", "AIZYNTH_SEARCH_MAX_TRANSFORMS")
    _set_bool_env(search_config, "return_first", "AIZYNTH_SEARCH_RETURN_FIRST")
    if search_config:
        config["search"] = search_config

    post_processing = _dict_setting(config, "post_processing")
    post_processing["min_routes"] = 1
    post_processing["max_routes"] = max_routes
    config["post_processing"] = post_processing

    cutoff_number = os.environ.get("AIZYNTH_EXPANSION_CUTOFF_NUMBER", "").strip()
    if cutoff_number:
        _apply_expansion_cutoff_number(
            config,
            _positive_int(cutoff_number, "AIZYNTH_EXPANSION_CUTOFF_NUMBER"),
        )
    return config


def _aizynth_config_from_env() -> dict[str, object]:
    config_path = os.environ.get("AIZYNTH_CONFIG_PATH", "").strip()
    if not config_path:
        raise RuntimeError("AIZYNTH_CONFIG_PATH is required for AiZynth planner wrapper")
    config = Path(config_path)
    if not config.is_file():
        raise RuntimeError(f"AiZynth config file not found: {config}")
    import yaml

    loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError("AiZynth config must be a YAML object")
    return copy.deepcopy(loaded)


def _dict_setting(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError(f"AiZynth config {key} must be a YAML object")
    return dict(value)


def _set_int_env(config: dict[str, object], setting: str, env_name: str) -> None:
    value = os.environ.get(env_name, "").strip()
    if value:
        config[setting] = _positive_int(value, env_name)


def _set_bool_env(config: dict[str, object], setting: str, env_name: str) -> None:
    value = os.environ.get(env_name, "").strip()
    if not value:
        return
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        config[setting] = True
        return
    if lowered in {"0", "false", "no", "off"}:
        config[setting] = False
        return
    raise RuntimeError(f"{env_name} must be a boolean")


def _positive_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise RuntimeError(f"{name} must be positive")
    return parsed


def _apply_expansion_cutoff_number(config: dict[str, object], cutoff_number: int) -> None:
    expansion = config.get("expansion")
    if not isinstance(expansion, dict):
        raise RuntimeError("AiZynth config expansion must be a YAML object")
    updated = {}
    for name, value in expansion.items():
        if isinstance(value, list):
            if len(value) != 2:
                raise RuntimeError("AiZynth expansion list entries must contain model and template")
            updated[name] = {
                "model": value[0],
                "template": value[1],
                "cutoff_number": cutoff_number,
            }
        elif isinstance(value, dict):
            policy_config = dict(value)
            policy_config["cutoff_number"] = cutoff_number
            updated[name] = policy_config
        else:
            raise RuntimeError("AiZynth expansion entries must be lists or YAML objects")
    config["expansion"] = updated


def _inline_stock_query() -> object:
    from aizynthfinder.chem import Molecule
    from aizynthfinder.context.stock.queries import StockQueryMixin

    class InlineSmilesStock(StockQueryMixin):
        def __init__(self, smiles: list[str]) -> None:
            self._smiles = {Molecule(smiles=item).smiles for item in smiles}

        def __contains__(self, mol: object) -> bool:
            return str(getattr(mol, "smiles", "")) in self._smiles

        def __len__(self) -> int:
            return len(self._smiles)

        def availability_string(self, mol: object) -> str:
            return "inline" if mol in self else "Not in stock"

    return InlineSmilesStock(_stock_smiles_from_env())


def _stock_smiles_from_env() -> list[str]:
    raw = os.environ.get("AIZYNTH_STOCK_SMILES", "").strip()
    if not raw:
        raise RuntimeError("AIZYNTH_STOCK_SMILES is required for inline AiZynth stock")
    if raw.startswith("["):
        values = json.loads(raw)
        if not isinstance(values, list):
            raise RuntimeError("AIZYNTH_STOCK_SMILES JSON must be a list")
        smiles = values
    else:
        smiles = [item.strip() for item in raw.split(",")]
    if not smiles or not all(isinstance(item, str) and item for item in smiles):
        raise RuntimeError("AIZYNTH_STOCK_SMILES must contain non-empty SMILES strings")
    return [str(item) for item in smiles]


def _select_if_configured(collection: object, name: str | None) -> None:
    if name:
        collection.select(name)


def _validated_routes(routes: object) -> list[dict[str, object]]:
    if not isinstance(routes, list):
        raise RuntimeError("AiZynth planner must return a list of route dictionaries")
    if not routes:
        raise RuntimeError("AiZynth planner returned no routes")
    normalized = []
    for index, route in enumerate(routes):
        if not isinstance(route, dict):
            raise RuntimeError("AiZynth planner routes must be JSON objects")
        route_id = str(route.get("route_id") or route.get("id") or f"aizynth-{index + 1}")
        steps = route.get("steps")
        reaction_smiles = route.get("reaction_smiles")
        if not isinstance(steps, list) and not isinstance(reaction_smiles, list):
            raise RuntimeError("AiZynth planner route requires steps or reaction_smiles")
        normalized_route = dict(route)
        normalized_route["route_id"] = route_id
        if isinstance(steps, list):
            normalized_route["steps"] = _srb_ready_steps(route_id, steps)
        normalized.append(normalized_route)
    return normalized


def _srb_ready_steps(route_id: str, steps: list[object]) -> list[dict[str, object]]:
    normalized_steps = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise RuntimeError("AiZynth planner route steps must be JSON objects")
        normalized_step = dict(step)
        normalized_step.setdefault("step_id", f"{route_id}-step-{index + 1}")
        normalized_step.setdefault("operation", "add")
        normalized_step.setdefault("reaction_type", "generic")
        if "reactants" not in normalized_step and isinstance(
            normalized_step.get("building_blocks"), list
        ):
            normalized_step["reactants"] = [
                dict(block)
                for block in normalized_step["building_blocks"]
                if isinstance(block, dict)
            ]
        normalized_steps.append(normalized_step)
    return normalized_steps


if __name__ == "__main__":
    raise SystemExit(main())
