# CRG/Provenance Subsystem Selection Brief

**Date:** 2026-06-03

**Parent plan:** `moleculeforge/docs/todo/2026-06-03-corearchitecture-v2-continuation-governance-plan.md`

**Selected first technical gate:** CRG/Provenance consistency and read-back chain.

---

## 1. Why This Gate Is First

CRG/Provenance is the most rational first technical gate after governance because it is central to CoreArchitecture v2 and can be inspected locally without pretending that missing production model artifacts are already available.

This gate connects:

- Core Reasoning Graph state (`CRG`)
- cross-agent belief and evidence writes
- provenance service persistence
- graph repository read-back
- optional signature / Rekor integration
- orchestrator workflow state

It is smaller than "finish CoreArchitecture v2" and more locally verifiable than HUMU, HCIV, JMCG, production DKI, supplier integrations, or L4 quantum runners.

## 2. Current Evidence Baseline

The current implementation comparison document claims that:

- workflow state carries a serializable CRG
- provenance metadata includes CRG
- provenance service writes workflow CRG data to the graph repository
- `GraphRepository` supports workflow belief writes, CRG edge writes, and run CRG read-back
- multiple agents write or read CRG beliefs
- `BaseAgent` supports structured AgentMessage envelopes and signature command hooks

These claims must be checked against code before any implementation claim is made.

## 3. Exact Files To Inspect

Core CRG and graph repository:

- `moleculeforge/libs/mf-core/src/mf_core/types/crg.py`
- `moleculeforge/libs/mf-core/src/mf_core/db/repositories/graph_repo.py`
- `moleculeforge/libs/mf-agents/src/mf_agents/crg/graph.py`
- `moleculeforge/protos/moleculeforge/v1/core/crg.proto`
- `moleculeforge/schemas/crg.schema.json`

Agent envelope and orchestration:

- `moleculeforge/libs/mf-agents/src/mf_agents/base/agent.py`
- `moleculeforge/agents/orchestrator/src/orchestrator/agent.py`
- `moleculeforge/agents/orchestrator/src/orchestrator/workflow/graph_builder.py`
- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`

Provenance service:

- `moleculeforge/services/provenance-svc/src/provenance_svc/main.py`
- `moleculeforge/services/provenance-svc/src/provenance_svc/domain/sigstore_integration.py`
- `moleculeforge/services/provenance-svc/src/provenance_svc/signer.py`
- `moleculeforge/services/provenance-svc/src/provenance_svc/models.py`

Agent CRG participants:

- `moleculeforge/agents/nl2obj/src/nl2obj/agent.py`
- `moleculeforge/agents/generator_coord/src/generator_coord/agent.py`
- `moleculeforge/agents/validation_agent/src/validation_agent/agent.py`
- `moleculeforge/agents/retrosyn_agent/src/retrosyn_agent/agent.py`
- `moleculeforge/agents/supply_agent/src/supply_agent/agent.py`
- `moleculeforge/agents/srb_agent/src/srb_agent/agent.py`
- `moleculeforge/agents/critic_agent/src/critic_agent/agent.py`

Reference tests for reading only unless the user explicitly authorizes tests:

- `moleculeforge/tests/unit/test_graph_repo.py`
- `moleculeforge/tests/unit/test_provenance.py`
- `moleculeforge/tests/integration/test_dki_provenance.py`

## 4. Allowed Change Scope

Initial inspection phase:

- no business-code changes
- no test execution
- no writes under `/workspace/SemMol`
- no writes under `/workspace/Projects`

If inspection finds a low-risk inconsistency, the allowed next-step change scope must be declared before editing. Candidate scopes:

- documentation-only correction if implementation and architecture docs disagree
- narrowly scoped CRG serialization/read-back correction
- narrowly scoped provenance metadata persistence correction
- narrowly scoped missing CRG belief write in one agent

Do not change model runners, DKI infrastructure, external supplier integrations, Sigstore/Rekor production identity, or benchmark logic as part of this gate.

## 5. Verification Policy

Per project instruction, do not run tests unless the user explicitly authorizes them.

Allowed verification without test execution:

- `git status --short -- <path>`
- `git diff --check -- <path>`
- `rg` / `sed` / `nl` read-only inspection
- import path and symbol existence inspection by reading files

Potential future verification requiring explicit user authorization:

- unit tests for `GraphRepository`
- unit tests for provenance service
- integration test for DKI provenance
- service startup or workflow smoke test

## 6. Documentation Update Targets

If only inspection is performed:

- update this brief with an inspection result section, or create a dated audit note under `moleculeforge/docs/todo/`

If implementation facts differ from the current comparison document:

- update `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`

If a code change is made:

- update the relevant progress MD with changed files, evidence, unverified items, and remaining blockers

## 7. Back-Check Checklist

After the CRG/Provenance inspection step:

- [ ] Was the selected gate still the right first gate?
- [ ] Were only the listed files inspected?
- [ ] Were no business-code files modified during inspection?
- [ ] Were `/workspace/SemMol` and `/workspace/Projects` kept read-only?
- [ ] Are implementation claims tied to exact file paths?
- [ ] Are unverified production dependencies still marked incomplete?
- [ ] Is the next change scope explicit before any code edit?

---

## 8. Inspection Result: 2026-06-03

### 8.1 Confirmed Implementation Facts

CRG model and local helper exist:

- `moleculeforge/libs/mf-core/src/mf_core/types/crg.py` defines `Belief`, `CRGEdge`, and `CRG`.
- `moleculeforge/libs/mf-agents/src/mf_agents/crg/graph.py` defines `ChemicalReasoningGraph.add_belief()`, `add_edge()`, `update_belief()`, `query()`, and `to_crg()`.

Graph repository supports the documented CRG persistence surface:

- `moleculeforge/libs/mf-core/src/mf_core/db/repositories/graph_repo.py` implements `write_workflow_belief()`.
- The same repository implements `write_crg_edge()`.
- The same repository implements `get_run_crg()`.
- `moleculeforge/libs/mf-core/src/mf_core/db/repositories/__init__.py` builds a shared CRG repository only when `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` are configured.

Base agent shared CRG and signed envelope hooks exist:

- `moleculeforge/libs/mf-agents/src/mf_agents/base/agent.py` supports structured `AgentMessage` parsing/publishing.
- It supports JSON-LD payload fields through `_encode_agent_payload()`.
- It supports signature command hooks through `SIGSTORE_SIGN_COMMAND` and `SIGSTORE_VERIFY_COMMAND`.
- It exposes `read_shared_crg(run_id)`.

Provenance service can write CRG metadata to graph repository in production mode:

- `moleculeforge/services/provenance-svc/src/provenance_svc/main.py` has `ProductionProvenanceStore.record()`.
- That path calls `_write_crg_to_graph()` with `stored["metadata"]["crg"]`.
- `_write_crg_to_graph()` writes beliefs through `write_workflow_belief()` and edges through `write_crg_edge()`.
- Production store construction requires Neo4j, Postgres database URL, and MinIO configuration.

Orchestrator workflow state carries a CRG-like dict:

- `moleculeforge/agents/orchestrator/src/orchestrator/workflow/graph_builder.py` includes `crg` in `WorkflowState`.
- `_with_status()` appends a workflow-stage belief on every state transition.
- `_record_workflow_stage_belief()` creates stage beliefs and `derives_from` edges between adjacent stage beliefs.
- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py` records final workflow provenance only when `workflow_scope == "full"`.
- `_record_workflow_provenance()` stores final-state CRG in provenance metadata and sets `crg["provenance_id"]` to the workflow-state artifact id.

Agent CRG participation exists:

- `nl2obj` writes `parsed_intent` and, when compiler output exists, `compiled_cig`.
- `generator_coord` writes `selected_generators`, reads failure beliefs, and reads `route_humu_embedding`.
- `validation_agent` writes and reads `validation_status`.
- `retrosyn_agent` writes `retrosyn_routes` and `route_humu_embedding`; it also reads failed validation / zero route beliefs.
- `supply_agent` writes and reads `supply_feasibility`; it also reads zero route beliefs.
- `srb_agent` writes `ssp_compiled` and reads unavailable supply beliefs.
- `critic_agent` writes and reads `critic_verdict`; it also reads validation, supply, and retrosyn failure beliefs.

### 8.2 Confirmed Gaps / Risks

1. Local CRG edge versioning is inconsistent.

   `ChemicalReasoningGraph.add_belief()` increments `CRG.version`, and `update_belief()` increments it, but `add_edge()` does not. The class docstring says it provides versioning, and `CRG` includes edges as first-class state, so edge mutation should likely advance the version.

2. Full workflow provenance CRG is not the same as shared repository CRG.

   The final workflow-state provenance record uses `final_state["crg"]`, which is produced by orchestrator workflow stage transitions. Agent-level beliefs written directly to `GraphRepository` are not automatically merged into `final_state["crg"]` before provenance record creation. This means final provenance metadata may contain only orchestrator stage beliefs even when shared CRG has richer agent beliefs.

3. Shared CRG is environment-dependent.

   Agent cross-read/write behavior only happens when Neo4j env vars configure `build_shared_crg_repository_from_env()`. Without those variables, each agent still creates a local `ChemicalReasoningGraph`, but cross-agent sharing is disabled.

4. Production CRG read-back is not fully proven by current tests.

   `moleculeforge/tests/unit/test_graph_repo.py` covers repository query shape with mocks. `moleculeforge/tests/integration/test_dki_provenance.py` covers real artifact chain, audit, child count, and object existence, but the inspected portion does not cover CRG metadata write and `get_run_crg()` read-back from real DKI.

5. Real Sigstore/Rekor remains a production gate.

   `SigstoreIntegration` supports local dev signatures and command-based Sigstore/Rekor hooks. Real Fulcio/Rekor identity and command configuration are still external production dependencies.

### 8.3 Recommended Next Change Scope

The lowest-risk code change is:

- File: `moleculeforge/libs/mf-agents/src/mf_agents/crg/graph.py`
- Change: increment `self.crg.version` inside `ChemicalReasoningGraph.add_edge()`.
- Reason: it makes local CRG versioning consistent with belief creation and belief update, with minimal blast radius.

The next documentation or implementation decision after that should be separate:

- either document the final-state CRG vs shared repository CRG distinction in `current-implementation-vs-corearchitecture-v2.md`
- or design a narrow merge/read-back step before `_record_workflow_provenance()` writes final provenance metadata

Do not combine these two tasks in one unbounded edit.

### 8.4 Back-Check Result

- [x] The selected gate remains the right first gate because it is central and locally inspectable.
- [x] Inspection was limited to CRG, provenance, orchestrator, participating agents, repository construction, and reference tests.
- [x] No business-code file was modified during inspection.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified.
- [x] Implementation claims above are tied to exact file paths.
- [x] Production dependencies remain explicitly marked incomplete.
- [x] The next proposed code scope is explicit and single-file.

---

## 9. Minimal Fix Execution: 2026-06-03

### 9.1 Code Change

Changed:

- `moleculeforge/libs/mf-agents/src/mf_agents/crg/graph.py`

Implementation:

- `ChemicalReasoningGraph.add_edge()` now increments `self.crg.version` after appending a `CRGEdge`.

Reason:

- Edge insertion mutates first-class CRG state.
- `add_belief()` and `update_belief()` already increment `CRG.version`.
- Keeping `add_edge()` aligned makes local CRG versioning internally consistent.

### 9.2 Verification Performed

Allowed static verification only:

- `git diff --check` for changed files
- `git diff --stat` for intended scope
- direct diff review for `graph.py`
- document section presence check with `rg`

### 9.3 Verification Not Performed

Tests were not run because the project instruction says not to run tests unless explicitly requested.

Recommended test later, if authorized:

- add or run a unit check that `ChemicalReasoningGraph.add_edge()` increments `CRG.version`
- run CRG/provenance related unit tests:
  - `moleculeforge/tests/unit/test_graph_repo.py`
  - relevant agent CRG tests in `moleculeforge/tests/unit/test_*agent.py`

### 9.4 Back-Check Result

- [x] The code change matched the selected gate.
- [x] The code change was limited to one CRG helper file.
- [x] Documentation was updated in the subsystem brief.
- [x] No unrelated refactor was introduced.
- [x] Production DKI and Sigstore gates remain unclaimed.
- [x] Test execution remains unperformed and explicitly recorded.

---

## 10. Architecture Comparison Calibration: 2026-06-03

### 10.1 Documentation Change

Changed:

- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`

Implementation facts clarified:

- The final workflow provenance metadata CRG comes mainly from orchestrator workflow-stage beliefs in `final_state["crg"]`.
- Agent-level beliefs written directly to the shared `GraphRepository` are not automatically merged into `final_state["crg"]` before workflow provenance record creation.
- Workflow provenance metadata CRG and Neo4j shared CRG repository should be treated as related but not identical state surfaces.
- `ChemicalReasoningGraph.add_edge()` now increments `CRG.version`.

### 10.2 Back-Check Result

- [x] The architecture comparison document now matches the inspected implementation more precisely.
- [x] The documentation update does not claim full production CRG read-back completion.
- [x] The remaining production blockers are still visible.
- [x] No tests were run.
