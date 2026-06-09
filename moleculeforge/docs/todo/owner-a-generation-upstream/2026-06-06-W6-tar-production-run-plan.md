# W6 TAR Production Run Plan

Date: 2026-06-06
Scope: Owner A, W6 TAR ProxylessNAS reward-cost search preparation

## Goal

Prepare a production TAR reward-cost search run using the existing
`ProxylessSearchScheduler` and optional `TAR_PROXYLESS_SEARCH_COMMAND` runner
without changing TAR algorithms, benchmark thresholds, or deployment defaults.
This document is a run plan only. It does not authorize production deployment.

## Current Baseline

Engineering path:

- Shared scheduler:
  `mf_core.routing.task_router.ProxylessSearchScheduler`
- Service method:
  `GeneratorRouterServicer.RunProxylessSearch()`
- Command runner:
  `python -m generator_router_svc.tar_proxyless_runner`
- Runtime env:
  `TAR_PROXYLESS_SEARCH_COMMAND`
  `TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS`

Runner input contract:

```json
{
  "reward_batches_by_dataset": {
    "kras": [
      {"hfm_3d": 0.2, "fragfm": 0.8}
    ]
  },
  "generator_costs": {"hfm_3d": 5.0, "fragfm": 1.0},
  "cost_weight": 0.1,
  "learning_rate": 1.0,
  "temperature": 1.0
}
```

Required runner output fields:

```json
{
  "rounds": [],
  "architecture_probabilities": {}
}
```

The local runner also emits `architecture_logits`, `generator_names`,
`cost_weight`, `learning_rate`, and `temperature`.

## Reward Payload Requirements

Use a new payload path under:

```text
data/processing/generator_artifacts/tar_reward_payload_YYYYMMDD_<run_id>.json
```

Minimum source gates:

- at least two datasets or one approved production dataset with at least five
  rounds;
- every reward batch contains at least two active generator names;
- rewards are finite numeric values on a documented scale;
- `generator_costs` includes every generator expected in the run;
- `cost_weight`, `learning_rate`, and `temperature` are finite positive values,
  except `cost_weight` may be zero for an explicitly cost-agnostic comparison;
- source notes record reward provenance, oracle/validation source, dataset/task
  names, generator names, and cost source.

Run this payload preflight before service or command execution:

```bash
.venv/bin/python - <<'PY'
import json
import math
from pathlib import Path

payload_path = Path(
    "data/processing/generator_artifacts/"
    "tar_reward_payload_YYYYMMDD_<run_id>.json"
)
payload = json.loads(payload_path.read_text(encoding="utf-8"))
reward_batches = payload["reward_batches_by_dataset"]
generator_costs = payload["generator_costs"]

assert isinstance(reward_batches, dict) and reward_batches, "missing rewards"
round_count = 0
for dataset, batches in reward_batches.items():
    assert isinstance(dataset, str) and dataset
    assert isinstance(batches, list) and batches
    for batch in batches:
        assert isinstance(batch, dict)
        assert len(batch) >= 2, batch
        for generator_name, reward in batch.items():
            assert generator_name in generator_costs, generator_name
            assert math.isfinite(float(reward)), reward
        round_count += 1

assert round_count >= 5, round_count
for value in generator_costs.values():
    assert math.isfinite(float(value)), value
assert math.isfinite(float(payload.get("learning_rate", 0.0)))
assert float(payload.get("learning_rate", 0.0)) > 0.0
assert math.isfinite(float(payload.get("temperature", 1.0)))
assert float(payload.get("temperature", 1.0)) > 0.0
assert math.isfinite(float(payload.get("cost_weight", 0.0)))

print("datasets", sorted(reward_batches))
print("round_count", round_count)
print("generators", sorted(generator_costs))
PY
```

## Command Smoke

Do not change deployment defaults for this smoke.

```bash
PYTHONPATH="libs/mf-core/src:services/generator-router-svc/src" \
  .venv/bin/python -m generator_router_svc.tar_proxyless_runner \
    < data/processing/generator_artifacts/tar_reward_payload_YYYYMMDD_<run_id>.json \
    > data/processing/generator_artifacts/tar_proxyless_result_YYYYMMDD_<run_id>.json
```

Validate result:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

result = json.loads(
    Path(
        "data/processing/generator_artifacts/"
        "tar_proxyless_result_YYYYMMDD_<run_id>.json"
    ).read_text(encoding="utf-8")
)
assert isinstance(result["rounds"], list) and result["rounds"]
assert isinstance(result["architecture_probabilities"], dict)
assert isinstance(result.get("architecture_logits"), dict)
assert isinstance(result.get("generator_names"), list)
prob_sum = sum(float(value) for value in result["architecture_probabilities"].values())
assert 0.99 <= prob_sum <= 1.01, prob_sum
print("rounds", len(result["rounds"]))
print("architecture_probabilities", result["architecture_probabilities"])
PY
```

## Service Smoke

Run after command smoke passes. This uses the real service command contract
without changing deployment manifests:

```bash
PYTHONPATH="libs/mf-core/src:services/generator-router-svc/src" \
TAR_PROXYLESS_SEARCH_COMMAND=".venv/bin/python -m generator_router_svc.tar_proxyless_runner" \
TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS=300 \
  .venv/bin/python - <<'PY'
import asyncio
import json
from pathlib import Path

from generator_router_svc.main import GeneratorRouterServicer
from mf_core.proto_gen.moleculeforge.v1.generator import router_pb2

async def main() -> None:
    payload = json.loads(
        Path(
            "data/processing/generator_artifacts/"
            "tar_reward_payload_YYYYMMDD_<run_id>.json"
        ).read_text(encoding="utf-8")
    )
    request = router_pb2.RouterProxylessSearchRequest(
        reward_batches_json=json.dumps(payload["reward_batches_by_dataset"]),
        generator_costs_json=json.dumps(payload["generator_costs"]),
        cost_weight=float(payload.get("cost_weight", 0.0)),
        learning_rate=float(payload["learning_rate"]),
        temperature=float(payload.get("temperature", 1.0)),
    )
    response = await GeneratorRouterServicer().RunProxylessSearch(request, None)
    result = json.loads(response.result_json)
    assert response.acknowledged is True
    assert response.round_count == len(result["rounds"])
    assert result["architecture_probabilities"]
    Path(
        "data/processing/generator_artifacts/"
        "tar_proxyless_service_smoke_YYYYMMDD_<run_id>.json"
    ).write_text(
        json.dumps(
            {
                "round_count": response.round_count,
                "generator_names": list(response.generator_names),
                "architecture_probabilities": dict(result["architecture_probabilities"]),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(response.round_count, dict(result["architecture_probabilities"]))

asyncio.run(main())
PY
```

## Promotion Decision Data

Record these before any W6 promotion decision:

- reward payload path and hash;
- payload preflight output;
- exact command and environment, excluding secrets;
- command smoke output;
- service smoke output;
- chosen production mode: configured command or in-service scheduler;
- downstream generator quality comparison;
- cluster readiness evidence with real ConfigMap/env and service request logs.

## Stop Conditions

Stop and ask before:

- lowering reward payload gates;
- choosing the production `TAR_PROXYLESS_SEARCH_COMMAND` value;
- changing Docker/Kubernetes/Helm defaults;
- changing scheduler/router semantics;
- editing benchmark thresholds or Owner B implementation files;
- launching broad benchmark suites as acceptance evidence.

## Back-Check

- This plan does not run TAR search.
- This plan does not change deployment defaults.
- This plan reuses the existing runner and service contracts.
- This plan treats W6 as production-resource blocked until real reward data,
  downstream quality evidence, and cluster validation exist.
