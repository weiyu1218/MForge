# W6 TAR ProxylessNAS Runner Preflight

Date: 2026-06-03

Owner: A / generation upstream

## Scope

W6 is the local engineering gate for the TAR ProxylessNAS runner command. It should provide a concrete `TAR_PROXYLESS_SEARCH_COMMAND` target that consumes the existing JSON payload contract and returns the existing JSON result contract.

This gate must not redesign TAR, HUMU, HFM, JMCG, or the generator set.

## Current Evidence

Existing code already provides the core TAR path:

- `moleculeforge/libs/mf-core/src/mf_core/routing/task_router.py`
  - `TaskAwareRouter.architecture_logits`
  - `TaskAwareRouter.proxyless_architecture_probabilities()`
  - `TaskAwareRouter.proxyless_expected_cost()`
  - `TaskAwareRouter.proxyless_architecture_optimizer_step()`
  - `ProxylessSearchScheduler.run()`
- `moleculeforge/services/generator-router-svc/src/generator_router_svc/main.py`
  - `GeneratorRouterServicer.RunProxylessSearch()`
  - `_proxyless_search_payload_from_request()`
  - `_proxyless_search_from_command()`
  - `_validate_proxyless_search_result()`
  - `TAR_PROXYLESS_SEARCH_COMMAND`
  - `TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS`
- `moleculeforge/protos/moleculeforge/v1/generator/router.proto`
  - `RouterProxylessSearchRequest`
  - `RouterProxylessSearchResponse`
- `moleculeforge/tests/unit/test_task_router.py`
  - scheduler tests
  - service request tests
  - external command contract test
  - deployment env wiring test

Infrastructure already exposes the env:

- `moleculeforge/infra/docker/docker-compose.dev.yml`
- `moleculeforge/infra/kubernetes/deployments/moleculeforge-services.yaml`
- `moleculeforge/infra/helm/moleculeforge/values.yaml`

Search result:

- No existing TAR runner script was found under project `scripts/`, `tools/`, or `services/generator-router-svc`.

## Existing Command Contract

Input is stdin JSON:

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

Output is stdout JSON. Current service validation requires:

```json
{
  "rounds": [],
  "architecture_probabilities": {}
}
```

Additional fields may be included without breaking the service.

## Gap

The remaining local gap is a real command target that can be configured as:

```bash
TAR_PROXYLESS_SEARCH_COMMAND="python -m generator_router_svc.tar_proxyless_runner"
```

The command should:

- read stdin JSON;
- validate the payload shape through the existing scheduler path;
- run `ProxylessSearchScheduler`;
- return `rounds` and `architecture_probabilities`;
- include useful local artifact metadata such as `architecture_logits` and `generator_names`;
- fail with a non-zero exit code and stderr message on invalid input.

## Non-Goals

- Do not invent a production reward dataset.
- Do not claim production TAR training completion.
- Do not change HUMU pretraining, HUMU encoder architecture, HFM architecture, or existing checkpoints.
- Do not require GPUs for this runner; it is a small control-plane optimizer over reward batches.
- Do not modify `/workspace/SemMol` or `/workspace/Projects`.

## Recommended Implementation

Add one small module:

- `moleculeforge/services/generator-router-svc/src/generator_router_svc/tar_proxyless_runner.py`

Add focused tests to:

- `moleculeforge/tests/unit/test_task_router.py`

Update documentation after implementation:

- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

## Back-Check

- [x] This preflight is based on current code, not only architecture intent.
- [x] The missing item is narrowed to a runner command target.
- [x] Existing service and scheduler contracts are reused.
- [x] HUMU/HFM/JMCG model code remains out of scope.
- [x] No business code was modified by this preflight.
- [x] No tests were run.
