# 2026-06-03 JMCG Feedback Audit / Consolidation Gate

## Purpose

Audit the recently added `moleculeforge.jmcg.feedback.v1` local feedback path after the property, intent, pocket, and route producer gates.

This gate is intentionally narrow. It checks contract consistency and provenance handling; it does not implement JMCG joint sampling.

## Scope

Code paths audited:

- `services/orchestrator-svc/src/orchestrator_svc/main.py`
  - `_attach_generation_feedback`
  - `_jmcg_context_feedback_from_state`
  - `_property_jmcg_feedback_from_generation_feedback`
  - `_merge_jmcg_feedback`
- `agents/generator_coord/src/generator_coord/agent.py`
  - `_route_humu_feedback_from_crg`
  - `_jmcg_feedback_envelope`
  - `_existing_jmcg_feedback_records`
  - `_jmcg_route_feedback_record`
- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py`
  - `_feedback_embedding_records`
  - `_valid_feedback_records`
  - `_weighted_feedback_embedding`
  - `_feedback_kind_weight`

## Findings

- The core boundary is consistent: default HFM consumes feedback only from explicit generator params and does not read shared CRG directly.
- `kind="property"`, `kind="intent"`, and `kind="pocket"` records remain context-only because they intentionally omit `humu_embedding`.
- HFM-3D still steers only from records that include `humu_embedding` or `route_humu_embedding`, and invalid dimensions / weights / polarity are dropped before steering.
- GeneratorCoord merges existing context records before appending route records, so property / intent / pocket records are not overwritten by route feedback.
- A small provenance consistency issue existed: `evidence_ids` could be shaped incorrectly when supplied as a string, and CRG route HUMU payloads could drop optional provenance fields before route envelope creation.

## Changes

- Orchestrator property feedback now normalizes `evidence_ids` so a string id becomes a single-item list instead of a list of characters.
- GeneratorCoord route feedback now preserves optional `source`, `weight`, `polarity`, `confidence`, `evidence_ids`, and `metadata` from CRG `route_humu_embedding` payloads.
- GeneratorCoord route JMCG records now normalize `evidence_ids` and only accept mapping-shaped `metadata`.
- Focused test specs were updated to cover string `evidence_ids` normalization and route provenance preservation.

## Back-Check

- [x] The gate stayed focused on contract/provenance consistency.
- [x] No non-steering record gained a HUMU embedding.
- [x] No new direct CRG read was added to HFM.
- [x] Route remains the only steering-capable record produced by this local route when a route HUMU embedding exists.
- [x] The implementation still does not claim completed JMCG joint sampling.

## Remaining Gaps

- No property / pocket / intent HUMU embedding producer is selected or implemented.
- No evidence-backed joint `(m,r,p)` model or sampler exists yet.
- HFM steering is still local post-flow latent steering, not joint manifold generation.
- Pytest execution remains pending because the project instruction says not to run local tests unless explicitly requested.

## Verification

- `python -m py_compile` passed for touched Python files.
- `git diff --check` passed.
- Trailing whitespace scan passed for touched docs and code.
- `rg` checks confirmed non-steering record tests and code paths still omit `humu_embedding`.
- Pytest was not run due to the project local no-test rule.
