# W6 TAR Production Readiness Gap

Date: 2026-06-06
Scope: Owner A, W6 TAR ProxylessNAS reward-cost search production readiness

## Current Position

The TAR ProxylessNAS local runner path exists and is locally verified. It is not
production reward-search evidence yet.

Current verified engineering evidence:

| Gate | Local Evidence | Status |
|---|---|---|
| Scheduler | `ProxylessSearchScheduler` runs multi-dataset reward-cost architecture updates | Local pass |
| Service path | `GeneratorRouterServicer.RunProxylessSearch()` runs scheduler locally or via command | Local pass |
| Runner command | `python -m generator_router_svc.tar_proxyless_runner` reads stdin JSON and writes service-compatible stdout JSON | Local pass |
| Deployment wiring | Compose, raw Kubernetes, and Helm expose `TAR_PROXYLESS_SEARCH_COMMAND` and timeout env | Local pass |

Current deployment state:

- Docker Compose uses `${TAR_PROXYLESS_SEARCH_COMMAND:-}`.
- Raw Kubernetes and Helm keep `proxyless-search-command` as an empty production
  resource slot.
- When the command is configured, runtime status and command execution validate
  the first executable through the shared command requirement path.

## Latest Source Inventory Check

A read-only inventory check on 2026-06-06 scanned
`data/processing/generator_artifacts/` for files matching the production run
plan reward-payload naming intent.

Observed result:

- No real TAR reward payload was found.
- No `tar_reward_payload_YYYYMMDD_<run_id>.json` production-candidate input was
  found.
- Only W11 FragFM local engineering data and sample-export files were present
  in the matched inventory.

Conclusion:

- W6 remains blocked on approved real reward-cost source data.
- Do not create synthetic or placeholder reward payloads to satisfy the
  production preflight.
- The next W6 action is still to obtain or identify the approved reward source,
  then run the documented preflight in the W6 production run plan.

## Non-Promotion Reasons

W6 is not production-ready because these gates are still missing:

| Missing Gate | Required Evidence | Owner / Resource |
|---|---|---|
| Real reward data | Non-demo reward batches from production oracle/validation feedback with stable provenance | Owner A + data |
| Production command value | Approved `TAR_PROXYLESS_SEARCH_COMMAND` value or decision to use in-service scheduler only | Owner A decision |
| Search run evidence | Service-compatible result with rounds, architecture probabilities, logits, costs, and run record | Owner A |
| Downstream effect | Evidence that selected architecture weights improve or preserve generation quality | Owner A + benchmark resources |
| Cluster runtime | Real generator-router deployment with configured command/value and request/response evidence | Owner A + H10 resources |

## Next Executable Gates

Proceed in this order unless the user reprioritizes:

1. Define and validate real reward batch source requirements.
2. Run a strict local command smoke with the real reward payload.
3. Decide whether production should configure
   `TAR_PROXYLESS_SEARCH_COMMAND="python -m generator_router_svc.tar_proxyless_runner"`
   or rely on the in-service scheduler path.
4. Run service-level `RunProxylessSearch` smoke with the approved payload.
5. Record downstream quality and cluster evidence before promotion.

Run plan:

- `docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-run-plan.md`

## Stop Conditions

Stop and ask before:

- lowering reward-data requirements;
- choosing production command/default values;
- changing Docker/Kubernetes/Helm defaults;
- editing TAR scheduler semantics or generator routing policy;
- editing Owner B benchmark thresholds or implementation files;
- modifying HUMU pretraining or HFM Lorentz flow architecture;
- overwriting protected checkpoints.

## Back-Check

- W6 local runner command target is complete.
- Production W6 still needs real reward data, command/default decision,
  downstream quality evidence, and cluster validation.
- No production TAR search run is authorized by this gap record.
