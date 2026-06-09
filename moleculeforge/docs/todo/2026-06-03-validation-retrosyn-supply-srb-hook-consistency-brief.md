# Validation/RetroSyn/Supply/SRB Hook Consistency Brief

**Date:** 2026-06-03

**Parent plan:** `moleculeforge/docs/todo/2026-06-03-corearchitecture-v2-continuation-governance-plan.md`

**Selected technical gate:** Validation / RetroSyn / Supply / SRB hook consistency.

---

## 1. Why This Gate Is Next

This gate is the next rational step after CRG/Provenance because it checks the concrete full-workflow synthesis chain:

- validation result controls whether the workflow proceeds or refines
- retrosynthesis route planning feeds supply assessment
- supply feasibility controls SRB compilation
- SRB consumes real route steps rather than fixed reaction placeholders

It is locally inspectable and narrower than HUMU/JMCG production completion.

## 2. Exact Files Inspected

Workflow and full clients:

- `moleculeforge/agents/orchestrator/src/orchestrator/workflow/graph_builder.py`
- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`

Agent implementations:

- `moleculeforge/agents/validation_agent/src/validation_agent/agent.py`
- `moleculeforge/agents/retrosyn_agent/src/retrosyn_agent/agent.py`
- `moleculeforge/agents/supply_agent/src/supply_agent/agent.py`
- `moleculeforge/agents/srb_agent/src/srb_agent/agent.py`
- `moleculeforge/agents/srb_agent/src/srb_agent/compiler.py`

Reference tests inspected only:

- `moleculeforge/tests/unit/test_service_artifact_status.py`

## 3. Confirmed Implementation Facts

- `WorkflowGraph._validating()` stores validation output and derives `validation_passed`.
- `WorkflowGraph._retrosyn()` calls optional `plan_routes`, `assess_supply`, and `compile_synthesis` hooks in sequence.
- `FullWorkflowClients.validate_candidates()` uses the default Boltz2 affinity path unless a request-level oracle level is supplied.
- When oracle level is supplied, `_validate_with_oracle_cascade()` delegates each candidate to `ValidationAgent`.
- `FullWorkflowClients.plan_routes()` delegates route planning to `RetroSynAgent`.
- `FullWorkflowClients.assess_supply()` delegates supply assessment to `SupplyAgent`.
- `FullWorkflowClients.compile_synthesis()` delegates SSP compilation to `SRBAgent`, and skips SRB when supply feasibility is unavailable.
- `RetroSynAgent` can persist `retrosyn_routes` and optional `route_humu_embedding` beliefs.
- `SupplyAgent` can persist `supply_feasibility` and read shared CRG route/supply beliefs.
- `SRBAgent` compiles SSPs from `retrosyn_route.steps` through `compile_ssp()`.
- `compile_ssp()` requires route steps and `route_id`, so SRB no longer silently fabricates protocols from fixed reaction types.

## 4. Confirmed Gap

The full workflow hook chain has a no-route handling gap:

- `FullWorkflowClients.assess_supply()` calls `_first_retrosyn_route(state)` before delegating to `SupplyAgent`.
- `_first_retrosyn_route(state)` raises when `retrosyn.routes` is missing or empty.
- `FullWorkflowClients.compile_synthesis()` also calls `_first_retrosyn_route(state)` unless supply has already been marked unavailable.

This means a planner result such as `{"routes": []}` can fail the orchestrator hook chain before supply is marked unavailable and before SRB can return a skipped result.

## 5. Allowed Change Scope

Low-risk implementation scope:

- Modify `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`.
- Add focused unit coverage in `moleculeforge/tests/unit/test_service_artifact_status.py`.
- Update architecture/progress docs.

Do not change model runners, external planner contracts, supply provider adapters, or SRB compiler semantics in this gate.

## 6. Verification Policy

Do not run tests unless explicitly authorized.

Allowed verification:

- static diff inspection
- `git diff --check`
- `rg` section and symbol checks

Recommended tests later, if authorized:

- targeted tests for no-route full workflow supply/SRB handling
- existing orchestrator full-client tests around RetroSyn/Supply/SRB hooks

---

## 7. Minimal Fix Execution: 2026-06-03

### 7.1 Code Change

Changed:

- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`

Implementation:

- `FullWorkflowClients.assess_supply()` now handles empty or missing `retrosyn.routes` by returning a structured supply result with `overall_feasibility=unavailable`.
- `FullWorkflowClients.compile_synthesis()` now returns a structured skipped result when no retrosyn route exists.
- `_first_retrosyn_route_or_none()` was added for safe hook-level branching while preserving `_first_retrosyn_route()` strict behavior for callers that require a route.

Reason:

- Full workflow should degrade from "no route" to "supply unavailable / SRB skipped" instead of raising before the downstream hook can record a state.
- SRB compiler strictness remains unchanged: real compilation still requires `retrosyn_route.route_id` and non-empty `retrosyn_route.steps`.

### 7.2 Test Coverage Added But Not Run

Changed:

- `moleculeforge/tests/unit/test_service_artifact_status.py`

Added focused tests:

- `test_full_workflow_clients_assess_supply_marks_unavailable_without_routes`
- `test_full_workflow_clients_compile_synthesis_skips_without_routes`

Tests were not run because the project instruction says not to run tests unless explicitly requested.

### 7.3 Verification Performed

Allowed static verification only:

- `git diff --check` for touched files
- trailing-whitespace scan for new progress documents
- symbol/section presence check with `rg`
- `python -m py_compile` for the touched Python files

### 7.4 Documentation Change

Changed:

- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`

Documentation now records the no-route full-workflow hook behavior and explicitly says the 2026-06-03 supplement did not run tests.

### 7.5 Verification Not Performed

Tests were not run.

Recommended command later, if explicitly authorized:

- `uv run pytest tests/unit/test_service_artifact_status.py -k "no_routes or full_workflow_clients_assess_supply or full_workflow_clients_compile_synthesis" -q`

### 7.6 Back-Check Result

- [x] The change stayed inside the selected Validation/RetroSyn/Supply/SRB hook gate.
- [x] The code change was limited to orchestrator full-client hook behavior.
- [x] SRB compiler semantics were not weakened.
- [x] Tests were added for the new behavior but not executed.
- [x] Production planner, supply provider, SiLA2, and cluster deployment gates remain unclaimed.
