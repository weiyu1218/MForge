# GeneratorCoord/HFM Feedback Propagation Brief

**Date:** 2026-06-03

**Parent plan:** `moleculeforge/docs/todo/2026-06-03-corearchitecture-v2-continuation-governance-plan.md`

**Selected technical gate:** GeneratorCoord / HFM feedback propagation.

---

## 1. Why This Gate Is Next

This gate checks the bridge from local workflow feedback toward future joint generation:

- validation / critic failures become `generation_feedback`
- route HUMU embeddings become `route_humu_feedback`
- feedback is passed through HFM and GeneratorCoord generation paths
- HFM-3D can steer latent samples only when feedback contains HUMU embeddings

This is narrower than implementing JMCG and helps prevent overclaiming current feedback semantics.

## 2. Exact Files Inspected

Workflow and full-client propagation:

- `moleculeforge/agents/orchestrator/src/orchestrator/workflow/graph_builder.py`
- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`

Generator coordination:

- `moleculeforge/agents/generator_coord/src/generator_coord/agent.py`

HFM service and generator:

- `moleculeforge/services/hfm-generator-svc/src/hfm_generator_svc/main.py`
- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py`

Reference architecture document:

- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`

## 3. Confirmed Implementation Facts

- `WorkflowGraph._refining()` appends failed critic output or validation output to `generation_feedback`.
- `FullWorkflowClients.generate_candidates()` calls `_attach_generation_feedback()` before both default HFM generation and explicit GeneratorCoord dispatch.
- `_attach_generation_feedback()` serializes non-empty workflow feedback into `generator_params["generation_feedback"]`.
- `_generate_with_generator_coord()` passes the same `generator_params` into `GeneratorCoordAgent.process()`.
- `GeneratorCoordAgent` reads shared CRG context and extracts `route_humu_embedding` beliefs into route HUMU feedback records.
- `GeneratorCoordAgent` injects route HUMU feedback into `generator_params["route_humu_feedback"]` when it dispatches generator clients.
- `_generator_proto_request()` stringifies generator params into `GenerateRequest.generator_params`.
- `HFMGeneratorServicer.Generate()` expands `GenerateRequest.generator_params` into keyword arguments for `HFM3DGenerator.generate()`.
- `HFM3DGenerator._feedback_steering_target()` inspects `route_humu_feedback` and `generation_feedback`.
- `HFM3DGenerator._feedback_embedding_records()` only creates steering records when feedback payloads include `humu_embedding` or `route_humu_embedding`.
- `HFM3DGenerator._apply_feedback_steering()` applies a bounded Lorentz tangent step toward the mean feedback embedding target.

## 4. Current Boundary / Risk

This gate does not prove JMCG.

Current behavior is a local feedback steering channel:

- plain validation / critic feedback is passed through as `generation_feedback`
- HFM-3D only steers when that feedback contains HUMU embedding records
- route HUMU feedback can steer HFM only when RetroSynAgent produced route embeddings, persisted them to shared CRG, and GeneratorCoord dispatches through generator clients
- default HFM generation receives workflow `generation_feedback`, but it does not automatically read shared CRG route HUMU beliefs

Therefore current implementation supports feedback propagation and embedding-based local steering, not molecule/route/property/pocket joint manifold sampling.

## 5. Implementation Decision

No code change was made in this gate.

Reason:

- The inspected propagation chain matches the current architecture document's bounded claim.
- The remaining gap is architectural: define the future JMCG feedback contract before changing HFM generation semantics.

## 6. Verification Performed

Allowed static verification only:

- `rg` location checks for `generation_feedback`, `route_humu_feedback`, and `route_humu_embedding`
- direct code inspection of the listed files
- read-only inspection of existing focused tests:
  - `moleculeforge/tests/unit/test_service_artifact_status.py`
  - `moleculeforge/tests/unit/test_generator_coord_agent.py`
  - `moleculeforge/tests/unit/test_generators.py`

Tests were not run.

Existing test coverage found by inspection:

- full workflow passes `generation_feedback` to default HFM generation
- full workflow passes `generation_feedback` to GeneratorCoord generation
- GeneratorCoord passes shared CRG route HUMU feedback to generator clients
- HFM-3D route HUMU feedback can steer latent samples and record feedback metadata

## 7. Back-Check Result

- [x] The gate stayed focused on feedback propagation.
- [x] No code files were changed for this gate.
- [x] Current behavior is documented as local steering, not JMCG.
- [x] The next step requires an architectural decision before implementation.

## 8. Recommended Next Decision

Before modifying HFM generation semantics, define the JMCG feedback contract:

- what constitutes molecule feedback
- what constitutes route feedback
- what constitutes property or pocket feedback
- how those records are weighted in shared HUMU space
- whether route HUMU feedback should be available to default HFM generation without going through GeneratorCoord
