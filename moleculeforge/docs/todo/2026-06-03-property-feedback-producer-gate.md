# Property Feedback Producer Gate

**Date:** 2026-06-03

**Parent plan:** `moleculeforge/docs/todo/2026-06-03-corearchitecture-v2-continuation-governance-plan.md`

**Previous gates:**

- `moleculeforge/docs/todo/2026-06-03-jmcg-feedback-contract-brief.md`
- `moleculeforge/docs/todo/2026-06-03-jmcg-feedback-semantics-gate.md`

**Selected technical gate:** introduce non-steering property feedback records into `jmcg_feedback`.

---

## 1. Scope

This gate adds property feedback as contract-shaped context records. Property feedback must not affect HFM steering unless a future gate provides an evidence-backed HUMU embedding source.

In scope:

- Convert validation / critic property feedback into `kind="property"` `jmcg_feedback` records.
- Keep property records non-steering by omitting `humu_embedding`.
- Preserve existing legacy `generation_feedback`.
- Merge property feedback with route feedback when GeneratorCoord adds route `jmcg_feedback`.
- Keep default HFM free of direct shared CRG reads.

Out of scope:

- Creating property HUMU embeddings.
- Steering from score-only property feedback.
- Pocket feedback producer.
- JMCG joint sampler.
- Production DKI, benchmark, Sigstore, or KRAS full-pilot validation.

## 2. Target Behavior

When workflow `generation_feedback` contains validation / critic feedback:

- Orchestrator full workflow generation params should include a `jmcg_feedback` envelope.
- That envelope should contain `kind="property"` records without `humu_embedding`.
- Records should include `source`, `run_id`, `subject`, `polarity`, `confidence`, `evidence_ids`, and `metadata`.
- HFM-3D should parse the envelope but ignore non-steering property records because they lack HUMU embeddings.

When GeneratorCoord also reads route HUMU feedback:

- Existing `jmcg_feedback` property records should be preserved.
- Route records should be appended to the same envelope.
- Legacy `route_humu_feedback` should remain present.

## 3. Risk Boundary

The main risk is overclaiming score-only property feedback as HUMU steering. This gate explicitly avoids that by omitting `humu_embedding`.

## 4. Back-Check Checklist

- [x] Did property feedback remain non-steering?
- [x] Did existing `generation_feedback` remain compatible?
- [x] Did GeneratorCoord merge rather than overwrite existing `jmcg_feedback`?
- [x] Did default HFM remain free of direct shared CRG reads?
- [x] Did documentation clearly state this is not completed JMCG?

## 5. Implementation Result

Modified:

- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`
- `moleculeforge/agents/generator_coord/src/generator_coord/agent.py`
- `moleculeforge/tests/unit/test_service_artifact_status.py`
- `moleculeforge/tests/unit/test_generator_coord_agent.py`
- `moleculeforge/tests/unit/test_generators.py`

Completed:

- Orchestrator full workflow generation now derives a `moleculeforge.jmcg.feedback.v1` envelope from workflow `generation_feedback`.
- Derived property records use `kind="property"` and intentionally omit `humu_embedding`.
- Existing legacy `generation_feedback` remains attached.
- GeneratorCoord now preserves existing `jmcg_feedback.records` and appends route HUMU records instead of overwriting the envelope.
- HFM-3D continues to ignore property records without HUMU embeddings, so property feedback remains non-steering.

Focused test specs added:

- default HFM generation receives property context records alongside legacy `generation_feedback`
- GeneratorCoord generation receives property context records
- GeneratorCoord merges property records with route records
- HFM does not steer from property records without HUMU embeddings

Verification:

- `python -m py_compile` passed for touched Python files.
- Pytest was not run because `.claude/CLAUDE.md` forbids local test execution without explicit instruction.

Remaining boundary:

- This gate does not create property HUMU embeddings.
- Property records are context/provenance only.
- JMCG joint sampling remains incomplete.
