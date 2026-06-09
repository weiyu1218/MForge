"""CLI runner for TAR ProxylessNAS-style architecture search.

The generator-router service can call this module through
``TAR_PROXYLESS_SEARCH_COMMAND="python -m generator_router_svc.tar_proxyless_runner"``.
It intentionally reuses the shared TaskAwareRouter scheduler instead of defining
a second architecture-search implementation.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from mf_core.routing.task_router import (
    GENERATOR_NAMES,
    ProxylessSearchScheduler,
    TaskAwareRouter,
)


def run_proxyless_search(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run the shared TAR Proxyless search scheduler from a JSON-like payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a JSON object")

    reward_batches_by_dataset = _payload_mapping(
        payload,
        "reward_batches_by_dataset",
    )
    generator_costs = _clean_float_mapping(
        _payload_mapping(payload, "generator_costs"),
        "generator_costs",
    )
    cost_weight = _payload_float(payload, "cost_weight", default=0.0)
    learning_rate = _payload_float(payload, "learning_rate")
    temperature = _payload_float(payload, "temperature", default=1.0)

    router = TaskAwareRouter(n_generators=len(GENERATOR_NAMES))
    scheduler = ProxylessSearchScheduler(
        router=router,
        generator_costs=generator_costs,
        cost_weight=cost_weight,
        learning_rate=learning_rate,
        temperature=temperature,
    )
    result = scheduler.run(reward_batches_by_dataset)
    active_logits = router.architecture_logits[: router.n_generators].detach().cpu()
    result.update(
        {
            "architecture_logits": {
                name: float(active_logits[index].item())
                for index, name in enumerate(GENERATOR_NAMES[: router.n_generators])
            },
            "generator_names": list(GENERATOR_NAMES[: router.n_generators]),
            "cost_weight": cost_weight,
            "learning_rate": learning_rate,
            "temperature": temperature,
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Read stdin JSON, write result JSON, and return a process exit code."""

    _ = argv
    try:
        raw_payload = sys.stdin.read()
        payload = json.loads(raw_payload)
        result = run_proxyless_search(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"tar_proxyless_runner: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, sort_keys=True))
    return 0


def _payload_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _payload_float(
    payload: Mapping[str, Any],
    name: str,
    *,
    default: float | None = None,
) -> float:
    if name not in payload or payload.get(name) is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return default
    value = float(payload[name])
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _clean_float_mapping(payload: Mapping[str, Any], name: str) -> dict[str, float]:
    clean: dict[str, float] = {}
    for key, raw_value in payload.items():
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{name} values must be finite")
        clean[str(key)] = value
    return clean


if __name__ == "__main__":
    raise SystemExit(main())
