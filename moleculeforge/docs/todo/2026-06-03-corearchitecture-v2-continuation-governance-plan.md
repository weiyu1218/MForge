# CoreArchitecture v2 Continuation Governance Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a controlled continuation process for MoleculeForge CoreArchitecture v2 work without damaging existing team changes.

**Architecture:** This plan treats the project as an engineering program, not a single coding task. Work proceeds from baseline capture, documentation governance, change isolation, and gated execution toward CoreArchitecture v2 alignment.

**Tech Stack:** Python, uv workspace, FastAPI/gRPC services, protobuf, LangGraph-style agents, Qdrant/Redis DKI decision, Neo4j/PostgreSQL/MinIO/Feast, PyTorch model packages, Kubernetes/Helm manifests.

---

## 0. Current Baseline

**Date:** 2026-06-03

**Branch:** `feature/corearchitecture-v2-completion`

**Working tree state:** `git status --short | wc -l` reported `230` entries. This is a heavily dirty collaborative branch. Existing modifications, deletions, generated artifacts, and untracked files must be treated as team-owned until explicitly assigned.

**Files already reviewed for project context:**

- `/workspace/MForge/MoleculeForge_CoreArchitecture_v2.md`
- `/workspace/MForge/MoleculeForge_CoreArchitecture_v2_完成度评估.md`
- `/workspace/MForge/MoleculeForge_CodeArchitecture.md`
- `/workspace/MForge/MoleculeForge_审核执行Plan.md`
- `/workspace/MForge/moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`

**Read-only external information roots:**

- `/workspace/SemMol`
- `/workspace/Projects`

These two roots may be inspected for context only. Do not modify files there.

**Current project stage judgment:**

MoleculeForge is in a CoreArchitecture v2 engineering alignment stage. It is beyond a pure skeleton, but it is not a completed CoreArchitecture v2 implementation. Many services, agents, schemas, runner adapters, preflight checks, deployment wirings, and local validation hooks exist. The production gates remain open for real model artifacts, external runners, DKI services, Sigstore/Rekor identity, full KRAS pilot, and full benchmark validation.

**No-test baseline rule:**

Per `.claude/CLAUDE.md`, do not run tests unless the user explicitly asks. When a future step needs test evidence, ask first or wait for an explicit test instruction.

### Baseline Back-Check

- [ ] Confirm the active branch before each work session with `git branch --show-current`.
- [ ] Confirm dirty worktree size before edits with `git status --short | wc -l`.
- [ ] Confirm no business code was changed during baseline/documentation-only steps.
- [ ] Confirm `/workspace/SemMol` and `/workspace/Projects` remain read-only.

---

## 1. Engineering Management Rules

### 1.1 Source-of-Truth Rule

Use the following documents as the active project truth hierarchy:

1. `/workspace/MForge/moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
   - Purpose: factual current implementation vs CoreArchitecture v2 alignment.
2. This document:
   - Purpose: continuation governance, work sequencing, and step-by-step review discipline.
3. Existing task documents under `/workspace/MForge/moleculeforge/docs/todo/`
   - Purpose: historical plans and subsystem-specific task trails.

Do not treat older architecture plans as implementation truth when they conflict with current source code or the current implementation comparison document.

### 1.2 Change Isolation Rule

- Do not revert, delete, rename, or overwrite existing team changes unless the user explicitly asks.
- Do not repair broad dirty worktree state as a side task.
- If a future feature requires business-code changes, create a task-specific scope first:
  - exact files to touch
  - expected behavior
  - test or verification evidence required
  - documentation updates required
- If an isolated worktree is requested later, create it only after deciding whether the new work should start from current dirty branch state or from clean HEAD. A clean worktree may miss ongoing team changes, so it is not automatically safe.

### 1.3 Documentation Update Rule

Every completed engineering step must update one of:

- this governance plan, when the sequencing or management rule changes
- `current-implementation-vs-corearchitecture-v2.md`, when implementation facts change
- a subsystem todo file, when detailed execution work is done

Each update must record:

- what changed
- which CoreArchitecture v2 layer it affects
- what was verified
- what was not verified
- what remains blocked by external resource, model artifact, runner command, credentials, DKI, or cluster deployment

### 1.4 Evidence Rule

Do not claim a gate is complete without fresh evidence. Acceptable evidence:

- file path and code location
- command and exit code
- test output, when user explicitly requested tests
- deployment manifest wiring
- explicit external dependency marked as missing

Unacceptable evidence:

- "it should work"
- mock behavior presented as production behavior
- old test output reused as current proof
- architecture document promises treated as implementation facts

### 1.5 Step Back-Check Rule

After each task, perform a local review:

1. Was the task necessary for the current project stage?
2. Did the task touch only the intended files?
3. Did it preserve team changes?
4. Did documentation change match the actual implementation change?
5. Was verification run only when allowed?
6. Are remaining blockers explicitly recorded?

---

## 2. Priority Roadmap

### P0: Stabilize Project Control

**Purpose:** Make the project safe to continue in a collaborative dirty branch.

- [ ] Record branch, dirty worktree size, and current high-level status at session start.
- [ ] Keep a short work log in this document or a linked subsystem todo.
- [ ] Avoid business-code edits until the target subsystem is chosen.
- [ ] Preserve existing team modifications.

**Back-check after P0:**

- [ ] Does the team have a clear current status snapshot?
- [ ] Are unowned changes protected?
- [ ] Is the next task bounded to a subsystem?

### P1: Confirm Active Truth Against Current Code

**Purpose:** Prevent decisions based on outdated architecture documents.

- [ ] Use `current-implementation-vs-corearchitecture-v2.md` as the alignment baseline.
- [ ] Compare each proposed task against current code before modifying anything.
- [ ] Mark conflicts between old documents and current implementation as "document conflict", not as automatic code defects.

**Back-check after P1:**

- [ ] Is the task based on current code rather than target architecture only?
- [ ] Is every claimed gap traceable to a file or explicit missing external resource?

### P2: Choose One Gate at a Time

**Purpose:** Avoid broad, unsafe edits across agents, services, models, and infrastructure.

Recommended first gates, in order:

1. Documentation and status governance.
2. CRG/Provenance read-write consistency, because much of it already exists and can be reasoned about locally.
3. Validation/RetroSyn/Supply/SRB hook consistency, because these are more concrete than JMCG.
4. GeneratorCoord/HFM feedback propagation, because it is the bridge toward joint feedback.
5. HUMU/HCIV/JMCG production gaps, only after the local orchestration chain is stable.

**Back-check after P2:**

- [ ] Was exactly one gate selected?
- [ ] Is the gate small enough to review in one session?
- [ ] Does the gate have a clear documentation update target?

### P3: Execute Subsystem Work With Review Boundaries

**Purpose:** Make engineering progress without losing auditability.

For each subsystem task:

- [ ] Define scope.
- [ ] List exact files to inspect.
- [ ] List exact files allowed to change.
- [ ] Identify whether tests are needed.
- [ ] Get explicit user permission before running tests.
- [ ] Make the smallest change that advances the gate.
- [ ] Update the relevant MD progress entry.
- [ ] Perform the Step Back-Check Rule.

**Back-check after P3:**

- [ ] Did code and docs move together?
- [ ] Are unverified production dependencies still marked as incomplete?
- [ ] Did the step avoid unrelated refactors?

### P4: Production Gate Closure

**Purpose:** Separate local engineering readiness from true production completion.

Open production gates:

- [ ] Production HCIV checkpoint and loader validation.
- [ ] Production HUMU mol/pocket/route encoder artifacts.
- [ ] HFM molecular decoder or `HFM_MOLECULAR_DECODER_COMMAND`.
- [ ] Real FragFM / CReM / MMPT / iCLM artifacts or runner commands.
- [ ] DKI environment: Qdrant, Redis, Neo4j, PostgreSQL, MinIO, Feast.
- [ ] Sigstore/Rekor sign and verify commands, identity token, expected identity.
- [ ] L4 GPU4PySCF/ORCA runner commands and artifacts.
- [ ] Commercial supplier endpoints and credentials.
- [ ] Full KRAS G12C pilot.
- [ ] MOSES/GuacaMol/PMO/CrossDocked benchmark data and thresholds.

**Back-check after P4 work:**

- [ ] Is the gate actually production-backed?
- [ ] Was a local mock or smoke path clearly separated from production evidence?
- [ ] Are missing credentials/artifacts documented instead of hidden?

## 3. Recommended Next Work Sequence

### Task 1: Preserve Baseline and Governance

**Files:**

- Create: `moleculeforge/docs/todo/2026-06-03-corearchitecture-v2-continuation-governance-plan.md`
- Do not modify business code.

**Steps:**

- [ ] Confirm branch with `git branch --show-current`.
- [ ] Confirm dirty worktree scale with `git status --short | wc -l`.
- [ ] Create this governance document.
- [ ] Record current stage judgment.
- [ ] Record no-test rule.
- [ ] Record back-check rules.

**Back-check:**

- [ ] Only this governance document changed.
- [ ] No business code changed.
- [ ] The document covers the user's four required areas:
  - current baseline
  - new documentation
  - engineering rules
  - priority sequence

### Task 2: Produce a Subsystem Selection Brief

**Files:**

- Modify: this governance document or create a sibling subsystem brief under `moleculeforge/docs/todo/`.

**Steps:**

- [ ] Read `current-implementation-vs-corearchitecture-v2.md`.
- [ ] Choose one first technical gate from P2.
- [ ] Explain why it is first.
- [ ] List exact files to inspect.
- [ ] List expected documentation update target.
- [ ] Stop before modifying code.

**Back-check:**

- [ ] The chosen gate is smaller than "finish CoreArchitecture v2".
- [ ] The gate can be reviewed independently.
- [ ] The chosen gate does not require unavailable external production resources unless the task is only to document the blocker.

### Task 3: Execute the First Technical Gate

**Files:**

- To be specified by Task 2.

**Steps:**

- [ ] Inspect only the selected files.
- [ ] Propose exact change.
- [ ] Apply minimal change.
- [ ] Update relevant progress MD.
- [ ] Run verification only if user explicitly authorizes.
- [ ] Record what remains unverified.

**Back-check:**

- [ ] The diff matches the selected gate.
- [ ] No unrelated files changed.
- [ ] Remaining blockers are explicit.

---

## 4. Communication Rules

- Keep updates short and specific.
- Do not repeat skill or hook boilerplate.
- Do not claim complete understanding of every code path without inspecting the relevant files.
- When uncertain, say exactly what is unknown and what file or command would resolve it.
- For `/workspace/SemMol` and `/workspace/Projects`, read only.

---

## 5. Current Decision

The next rational action is not broad implementation. It is controlled subsystem execution through one gate at a time.

Recommended first technical gate:

**CRG/Provenance consistency and read-back chain**, because it is central to CoreArchitecture v2, already partially implemented, and less dependent on unavailable model artifacts than HUMU/JMCG.

Alternative first gate:

**Validation/RetroSyn/Supply/SRB hook consistency**, because it is concrete, service-oriented, and can improve the end-to-end workflow without pretending production model gates are solved.

Do not start HUMU/JMCG production work until the local orchestration and evidence chain are stable.

---

## 6. Execution Log

### 2026-06-03: P0/P1 Completed, P2 Gate Started

Completed:

- Created this governance plan.
- Created `moleculeforge/docs/todo/2026-06-03-crg-provenance-subsystem-selection-brief.md`.
- Selected the first technical gate: CRG/Provenance consistency and read-back chain.
- Performed static inspection of CRG, provenance, orchestrator, graph repository, and participating agent files.
- Applied a minimal CRG helper fix: `ChemicalReasoningGraph.add_edge()` now increments `CRG.version`.
- Calibrated `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md` to distinguish workflow provenance metadata CRG from shared `GraphRepository` CRG.

Verification:

- Static diff checks were used.
- Documentation section presence was checked with `rg`.
- Tests were not run, following the project no-test rule.

Back-check:

- [x] Work stayed on the selected CRG/Provenance gate.
- [x] Business-code change was limited to one local CRG helper.
- [x] Documentation was updated together with implementation facts.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified.
- [x] Production gates remain unclaimed.

Recommended next gate:

**Validation/RetroSyn/Supply/SRB hook consistency** should be next, unless the team first wants to turn the CRG/Provenance read-back gap into an explicit implementation task. That decision should be separated from the already completed local CRG versioning correction.

### 2026-06-03: Validation/RetroSyn/Supply/SRB Gate Started

Created:

- `moleculeforge/docs/todo/2026-06-03-validation-retrosyn-supply-srb-hook-consistency-brief.md`

Completed:

- Inspected orchestrator workflow, `FullWorkflowClients`, ValidationAgent, RetroSynAgent, SupplyAgent, SRBAgent, and SRB compiler.
- Confirmed the main hook chain exists.
- Found a concrete no-route hook gap in the full workflow: empty `retrosyn.routes` could raise before supply became unavailable and before SRB returned skipped.
- Applied a local orchestrator full-client fix for no-route supply/SRB behavior.
- Added focused unit tests for the new no-route behavior, but did not run them.
- Updated the architecture comparison document.

Verification:

- `git diff --check` passed for touched files.
- New progress documents passed trailing-whitespace scan.
- Touched Python files passed `python -m py_compile`.
- Tests were not run.

Back-check:

- [x] The task stayed narrower than HUMU/JMCG production work.
- [x] The fix did not alter external planner, supply provider, or SRB compiler contracts.
- [x] Documentation records what changed and what remains unverified.

### 2026-06-03: GeneratorCoord/HFM Feedback Gate Audited

Created:

- `moleculeforge/docs/todo/2026-06-03-generatorcoord-hfm-feedback-propagation-brief.md`

Completed:

- Inspected workflow `generation_feedback` creation.
- Inspected full-client propagation into default HFM and explicit GeneratorCoord generation paths.
- Inspected GeneratorCoord route HUMU feedback extraction from shared CRG.
- Inspected HFM service parameter forwarding.
- Inspected HFM-3D feedback steering behavior.

Decision:

- No code change was made in this gate.
- Current behavior supports feedback propagation and embedding-based local steering.
- It does not implement JMCG joint molecule/route/property/pocket sampling.

Back-check:

- [x] The gate stayed focused on feedback propagation.
- [x] No production generation semantics were changed without a JMCG contract.
- [x] The next step is an architecture decision, not a blind code patch.

### 2026-06-03: JMCG Feedback Contract Gate Drafted

Created:

- `moleculeforge/docs/todo/2026-06-03-jmcg-feedback-contract-brief.md`

Completed:

- Rechecked the CoreArchitecture v2 JMCG definition.
- Rechecked current HFM / GeneratorCoord / orchestrator feedback behavior.
- Defined a normalized `moleculeforge.jmcg.feedback.v1` envelope for future feedback records.
- Defined initial feedback kinds: molecule, route, property, pocket, and intent.
- Defined required steering-capable record fields, including subject identity, HUMU embedding, curvature, weight, polarity, confidence, and evidence ids.
- Mapped existing `route_humu_feedback` and `generation_feedback` payloads into the future contract.
- Recorded conservative weighting rules and explicit non-goals.

Decision boundary:

- The next hard architecture decision is whether default HFM generation should receive shared CRG route feedback directly, or whether route feedback should continue to flow only through GeneratorCoord.
- Conservative recommendation recorded in the brief: keep shared CRG reads in GeneratorCoord for now and keep HFM as a pure generation component.

Verification:

- Static code/document inspection only.
- Tests were not run.

Back-check:

- [x] The step created a contract before code semantics changed.
- [x] No business code was modified in this gate.
- [x] Legacy fields remain compatible in the proposed contract.
- [x] The document explicitly avoids claiming completed JMCG.
- [x] A hard decision is isolated for the next implementation gate.

### 2026-06-03: JMCG Feedback Contract Parser/Producer Gate Implemented

Decision confirmed:

- Default HFM must not read shared CRG directly.
- Route HUMU feedback continues to flow through GeneratorCoord and into generator params.

Modified:

- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py`
- `moleculeforge/agents/generator_coord/src/generator_coord/agent.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/tests/unit/test_generator_coord_agent.py`
- `moleculeforge/docs/todo/2026-06-03-jmcg-feedback-contract-brief.md`

Completed:

- HFM-3D now accepts `jmcg_feedback` envelope payloads in addition to legacy `route_humu_feedback` and `generation_feedback`.
- HFM-3D now parses `records` lists from contract-shaped feedback payloads.
- HFM-3D records `feedback_steering_kinds` metadata when contract records include `kind`.
- GeneratorCoord now emits `moleculeforge.jmcg.feedback.v1` route feedback envelopes while preserving legacy `route_humu_feedback`.
- Focused test specifications were added but not executed.

Verification:

- `python -m py_compile` passed for touched Python files.
- `git diff --check` passed for touched Python files.
- `rg` confirmed new `jmcg_feedback` paths.
- Pytest was not run due to the project no-test rule.

Back-check:

- [x] The implementation followed the confirmed decision boundary.
- [x] HFM remains a pure generator parameter consumer and does not read CRG directly.
- [x] Existing legacy route feedback remains compatible.
- [x] This is still local HUMU feedback steering, not completed JMCG joint sampling.

### 2026-06-03: JMCG Feedback Semantics Gate Implemented

Created:

- `moleculeforge/docs/todo/2026-06-03-jmcg-feedback-semantics-gate.md`

Modified:

- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py`
- `moleculeforge/tests/unit/test_generators.py`

Completed:

- HFM-3D now validates contract-shaped feedback records before steering.
- Invalid embedding dimensions, invalid weights, zero confidence, and unknown polarity are dropped.
- Effective weight is computed as `weight * confidence`.
- Feedback records are aggregated per kind before global target aggregation.
- Initial kind weights are applied.
- `polarity="repel"` now uses reversed spatial HUMU coordinates before projection.
- Metadata now records accepted count, dropped count, kind count, kinds, sources, and effective weight.
- Focused test specs were added but not executed.

Verification:

- `python -m py_compile` passed for touched Python files.
- Tests were not run due to the project local no-test rule.

Back-check:

- [x] Default HFM still does not read shared CRG.
- [x] GeneratorCoord remains the explicit route feedback injection boundary.
- [x] Legacy feedback fields remain compatible.
- [x] The implementation improves local steering semantics only.
- [x] JMCG joint molecule/route/property/pocket sampling remains incomplete.

### 2026-06-03: Property Feedback Producer Gate Implemented

Created:

- `moleculeforge/docs/todo/2026-06-03-property-feedback-producer-gate.md`

Modified:

- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`
- `moleculeforge/agents/generator_coord/src/generator_coord/agent.py`
- `moleculeforge/tests/unit/test_service_artifact_status.py`
- `moleculeforge/tests/unit/test_generator_coord_agent.py`
- `moleculeforge/tests/unit/test_generators.py`

Completed:

- Full workflow generation now derives non-steering `kind="property"` records from workflow `generation_feedback`.
- Derived property records are wrapped in `moleculeforge.jmcg.feedback.v1`.
- Property records intentionally omit `humu_embedding`, so HFM-3D cannot use them for steering.
- Legacy `generation_feedback` remains attached.
- GeneratorCoord now merges existing `jmcg_feedback.records` with route HUMU records instead of overwriting the envelope.
- Focused test specs were added but not executed.

Verification:

- `python -m py_compile` passed for touched Python files.
- Tests were not run due to the project local no-test rule.

Back-check:

- [x] Property feedback is context/provenance only.
- [x] No property HUMU embedding was invented.
- [x] Default HFM still does not read shared CRG.
- [x] Route feedback remains the only steering-capable record produced by this local route.
- [x] JMCG joint sampling remains incomplete.

### 2026-06-03: Pocket / Intent Feedback Producer Gate Implemented

Created:

- `moleculeforge/docs/todo/2026-06-03-pocket-intent-feedback-producer-gate.md`

Modified:

- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`
- `moleculeforge/tests/unit/test_service_artifact_status.py`
- `moleculeforge/tests/unit/test_generator_coord_agent.py`
- `moleculeforge/tests/unit/test_generators.py`

Completed:

- Full workflow generation now derives non-steering `kind="intent"` records from available HCIV / intent cone context.
- Full workflow generation now derives non-steering `kind="pocket"` records from CIG `target_context` pocket-related fields.
- Intent and pocket records intentionally omit `humu_embedding`.
- Existing property context records are preserved in the same `jmcg_feedback` envelope.
- GeneratorCoord merge behavior preserves intent / pocket / property records and appends route HUMU records.
- Focused test specs were added but not executed.

Verification:

- `python -m py_compile` passed for touched Python files.
- Tests were not run due to the project local no-test rule.

Back-check:

- [x] Pocket and intent feedback are context/provenance only.
- [x] No pocket or intent HUMU embedding was invented.
- [x] Default HFM still does not read shared CRG.
- [x] Route feedback remains the only steering-capable record produced by this local route.
- [x] JMCG joint sampling remains incomplete.

### 2026-06-03: JMCG Feedback Audit / Consolidation Gate Implemented

Created:

- `moleculeforge/docs/todo/2026-06-03-jmcg-feedback-audit-consolidation-gate.md`

Modified:

- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`
- `moleculeforge/agents/generator_coord/src/generator_coord/agent.py`
- `moleculeforge/tests/unit/test_service_artifact_status.py`
- `moleculeforge/tests/unit/test_generator_coord_agent.py`
- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`

Completed:

- Audited the current property / intent / pocket / route `jmcg_feedback` producer and consumer paths.
- Confirmed default HFM remains an explicit generator-parameter consumer and does not read shared CRG directly.
- Confirmed property / intent / pocket records remain non-steering because they omit `humu_embedding`.
- Tightened `evidence_ids` normalization so string ids remain single ids rather than being split into characters.
- Preserved optional route provenance fields from CRG `route_humu_embedding` payloads before JMCG route envelope creation.
- Focused test specs were updated but not executed.

Verification:

- `python -m py_compile` passed for touched Python files.
- `git diff --check` passed.
- Trailing whitespace scan passed for touched docs and code.
- `rg` confirmed the non-steering record tests and code paths still omit `humu_embedding`.
- Tests were not run due to the project local no-test rule.

Back-check:

- [x] The gate stayed focused on schema/provenance consolidation.
- [x] No property / intent / pocket HUMU embedding was invented.
- [x] Route feedback remains the only steering-capable record produced by this local route.
- [x] Existing legacy `route_humu_feedback` remains compatible.
- [x] JMCG joint sampling remains incomplete.

### 2026-06-03: 乙方 W1/W3/W5/W12 完成

**W1 (CRG 最终态合并/读回)**

Verified complete (no code change needed):

- `services/orchestrator-svc/src/orchestrator_svc/main.py` already contains `_merge_agent_beliefs_into_crg()` (line 812-861) called inside `_record_workflow_provenance()` (line 869).
- `libs/mf-core/src/mf_core/db/repositories/__init__.py` exports `build_shared_crg_repository_from_env`.
- `tests/unit/test_graph_repo.py` has 3 focused test cases (lines 280-427) covering merge, deduplication, and no-repo fallback.

Remaining blocker: H1 (DKI Neo4j) for real integration validation.

**W3 (PCBO 参考 candidate provider / oracle evaluator)**

Created:

- `pipelines/pareto_bo/src/pareto_bo/providers.py`: `TangentSpaceNoiseCandidateProvider` (local only), `SmilesCandidateProvider` (SMILES list → HUMU/fingerprint embeddings), `LocalOracleEvaluator` (gRPC oracle or embedding proxy fallback).

Modified:

- `pipelines/pareto_bo/src/pareto_bo/service.py`: `_runtime_from_env` accepts `default_factory`; `_default_candidate_provider` / `_default_oracle_evaluator` wired in.
- `tests/unit/test_mf_eval.py`: 4 focused tests for W3 added.

Remaining blocker: real candidate provider/oracle evaluator runner values and production acceptance (C class).

**W5 (benchmark harness 非 skip 路径补完)**

Modified:

- `tests/benchmark/__init__.py`: added `_open_text(path)` helper for transparent gzip support (`.gz`/`.gzip` suffix); `read_smiles_file`, `read_scored_smiles_table`, `read_jsonl_records` now use `_open_text`.

Remaining blocker: H8 official benchmark data and official thresholds.

**W12 (CReM-pharm-3D 真实 scorer 闭环 — B 类 AI 编码侧)**

Modified:

- `tests/unit/test_phase_b_generators.py`: 3 focused tests for W12 scorer paths added — pharmacophore mock scorer ranking, HUMU scorer ranking + embedding write, DockOracleGrpcScorer batch with mock gRPC stub.

Remaining blocker: real DiffDock-L / pharmacophore / HUMU scorer runner values and cluster acceptance (H5 + H10).

Verification:

- `python -m py_compile` passed for all touched Python files.
- `git diff --check` passed for all touched files.
- Tests were not run due to the project no-test rule.

Back-check:

- [x] W1/W3/W5/W12 stayed within 乙's assigned scope (see tasksplit §4).
- [x] No甲's files were modified (orchestrator generation side, hfm_3d, generator_coord untouched).
- [x] C1/C2/C3 contracts unchanged; no new predicates added to C2 table.
- [x] All remaining production gates explicitly recorded as blockers.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified.

### 2026-06-03: 甲方 W2 Pocket / Intent HUMU Feedback Gate Implemented

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/W2-pocket-intent-embedding-preflight.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/W2-pocket-intent-embedding-implementation-plan.md`

Modified:

- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`
- `moleculeforge/tests/unit/test_service_artifact_status.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`

Completed:

- Established the Owner A generation-upstream progress workspace.
- Confirmed the current HUMU/HFM feedback vector contract: `dim=128` is the spatial dimension, while current steering-capable Lorentz full-coordinate embeddings are 129-dimensional.
- Updated the two-person task split and interface-acceptance documents to reflect read/copy-only context for `/workspace/SemMol` and `/workspace/Projects`.
- Updated W2 scope: plain 128-dimensional HCIV is not inserted as HFM `humu_embedding`; intent feedback becomes steering-capable only from an already-valid 129-dimensional Lorentz axis.
- Added optional pocket feedback enrichment through the existing `HUMU_ENCODER_TARGET` convention, with packed float32 decoding and fail-closed behavior.
- Preserved metadata-only pocket context as non-steering.
- Added focused specs for pocket enrichment, pocket fallback, valid intent axis steering, and 128-dimensional HCIV non-steering.
- Fixed a pre-existing numerically unstable HFM repel-feedback test assertion by checking actual latent displacement rather than float32 Lorentz distance on large coordinates.

Verification:

- User explicitly authorized focused pytest for this gate.
- `uv run pytest tests/unit/test_service_artifact_status.py tests/unit/test_generators.py -q` passed with exit code 0.
- The final focused pytest run reported one existing LangGraph deprecation warning and no failures.
- `python -m py_compile moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py moleculeforge/tests/unit/test_service_artifact_status.py moleculeforge/tests/unit/test_generators.py` passed.
- `git diff --check` passed.
- Trailing whitespace scan passed for touched W2 files and docs.

Back-check:

- [x] W2 stayed within 甲's assigned generation-upstream scope.
- [x] HUMU pretraining remains frozen and unchanged.
- [x] HUMU encoder architecture remains unchanged.
- [x] HFM architecture remains unchanged.
- [x] Metadata-only context is not treated as a HUMU embedding.
- [x] Current steering-capable feedback embeddings are required to be 129-dimensional.
- [x] The architecture docs still state this is local feedback steering, not JMCG joint sampling.

### 2026-06-03: 甲方 W6 TAR ProxylessNAS Runner Gate Implemented

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/W6-tar-runner-preflight.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/W6-tar-runner-implementation-plan.md`
- `moleculeforge/services/generator-router-svc/src/generator_router_svc/tar_proxyless_runner.py`

Modified:

- `moleculeforge/tests/unit/test_task_router.py`
- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

Completed:

- Added a concrete local runner command target: `python -m generator_router_svc.tar_proxyless_runner`.
- The runner reads the existing TAR reward-cost payload from stdin and writes service-compatible JSON to stdout.
- The runner reuses `ProxylessSearchScheduler`; no new routing algorithm was introduced.
- Added focused specs for direct runner execution, CLI subprocess behavior, and service invocation through the real runner command.
- Updated architecture/task/progress docs to distinguish local runner readiness from production data and cluster validation.

Verification:

- `python -m py_compile moleculeforge/services/generator-router-svc/src/generator_router_svc/tar_proxyless_runner.py moleculeforge/tests/unit/test_task_router.py` passed.
- `git diff --check` passed.
- `uv run python -m generator_router_svc.tar_proxyless_runner` command-level smoke passed with a two-round KRAS reward payload.
- Pytest was not run for W6 because this gate has not received separate explicit pytest authorization.

Back-check:

- [x] W6 stayed within 甲's generation-upstream scope.
- [x] HUMU pretraining remains frozen and unchanged.
- [x] HFM architecture and checkpoints remain unchanged.
- [x] Production reward dataset, production `TAR_PROXYLESS_SEARCH_COMMAND` deployment value, and cluster validation remain explicit blockers.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.

### 2026-06-04: 甲方 W8-E JMCG Engineering Skeleton Gate Implemented

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/W8-jmcg-engineering-skeleton-preflight.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/W8-jmcg-engineering-skeleton-implementation-plan.md`
- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/jmcg_sampler.py`

Modified:

- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/__init__.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

Completed:

- Added `JMCGEngineeringSampler` local W8-E skeleton under HFM inference.
- Added JSON-serializable `moleculeforge.jmcg.joint_sample.v1` engineering skeleton output.
- Added parser compatibility for `moleculeforge.jmcg.feedback.v1` records.
- Preserved the 129-dimensional HUMU/HFM steering-capable embedding contract.
- Added focused specs for legal output, invalid dimension non-steering behavior, and parser compatibility.

Verification:

- `python -m py_compile moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/jmcg_sampler.py moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/__init__.py moleculeforge/tests/unit/test_generators.py` passed.
- `git diff --check` passed.
- `uv run python - <<'PY' ... JMCGEngineeringSampler ... PY` command-level smoke passed.
- Pytest was not run for W8-E because this gate has not received separate explicit pytest authorization.

Back-check:

- [x] W8-E stayed within 甲's generation-upstream scope.
- [x] HFM default generation behavior was not changed.
- [x] HUMU pretraining remains frozen and unchanged.
- [x] W8-R research quality, joint training artifact, and production validation remain explicit blockers.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.
