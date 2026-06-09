# Pocket / Intent Feedback Producer Gate

**Date:** 2026-06-03

**Parent plan:** `moleculeforge/docs/todo/2026-06-03-corearchitecture-v2-continuation-governance-plan.md`

**Previous gates:**

- `moleculeforge/docs/todo/2026-06-03-jmcg-feedback-contract-brief.md`
- `moleculeforge/docs/todo/2026-06-03-jmcg-feedback-semantics-gate.md`
- `moleculeforge/docs/todo/2026-06-03-property-feedback-producer-gate.md`

**Selected technical gate:** introduce non-steering pocket and intent feedback records into `jmcg_feedback`.

---

## 1. Scope

This gate adds pocket and intent context records to `jmcg_feedback`. These records must not steer HFM unless a future gate supplies evidence-backed HUMU embeddings.

In scope:

- Derive `kind="intent"` records from available HCIV / intent cone context.
- Derive `kind="pocket"` records from available CIG target context when present.
- Omit `humu_embedding` for both record types.
- Preserve existing `generation_feedback`, property records, and route records.
- Keep default HFM free of direct shared CRG reads.

Out of scope:

- Calling HUMU pocket encoder.
- Creating pocket or intent HUMU embeddings.
- Changing HFM steering behavior.
- Implementing joint sampler.
- Production DKI, Sigstore, benchmark, or KRAS full-pilot validation.

## 2. Target Behavior

When workflow state has HCIV or intent cone:

- Orchestrator generation params should include a non-steering `kind="intent"` record.
- The record should include metadata with available HCIV / intent cone summaries.

When workflow state or request has CIG target context with pocket-related data:

- Orchestrator generation params should include a non-steering `kind="pocket"` record.
- The record should include metadata with available target context keys, not an invented HUMU embedding.

When property and route feedback also exist:

- Existing property context records should be preserved.
- GeneratorCoord should keep merging route HUMU records with existing property / pocket / intent records.

## 3. Risk Boundary

The main risk is pretending pocket or intent records are HUMU-space steering records before a reliable embedding source is selected. This gate avoids that by omitting `humu_embedding`.

## 4. Back-Check Checklist

- [x] Did pocket and intent feedback remain non-steering?
- [x] Did existing property and route feedback remain compatible?
- [x] Did default HFM remain free of direct shared CRG reads?
- [x] Did documentation clearly state this is not completed JMCG?
- [x] Were only static checks run locally?

## 5. Implementation Result

Modified:

- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`
- `moleculeforge/tests/unit/test_service_artifact_status.py`
- `moleculeforge/tests/unit/test_generator_coord_agent.py`
- `moleculeforge/tests/unit/test_generators.py`

Completed:

- Orchestrator generation now derives non-steering `kind="intent"` records from available HCIV / intent cone context.
- Orchestrator generation now derives non-steering `kind="pocket"` records from CIG `target_context` pocket-related fields.
- Intent and pocket records intentionally omit `humu_embedding`.
- Existing property records are preserved and merged into the same `jmcg_feedback` envelope.
- GeneratorCoord already preserves existing records and appends route HUMU records, so property / pocket / intent records survive route feedback injection.
- HFM-3D continues to ignore records without HUMU embeddings.

Focused test specs added:

- default HFM generation receives non-steering intent and pocket records
- GeneratorCoord preserves existing intent and pocket records while appending route records
- HFM does not steer from intent and pocket records without HUMU embeddings

Verification:

- `python -m py_compile` passed for touched Python files.
- Pytest was not run because `.claude/CLAUDE.md` forbids local test execution without explicit instruction.

Remaining boundary:

- This gate does not call HUMU pocket encoder.
- Intent / pocket records are context/provenance only.
- JMCG joint sampling remains incomplete.
