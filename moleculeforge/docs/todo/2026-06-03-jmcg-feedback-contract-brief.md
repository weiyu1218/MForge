# JMCG Feedback Contract Brief

**Date:** 2026-06-03

**Parent plan:** `moleculeforge/docs/todo/2026-06-03-corearchitecture-v2-continuation-governance-plan.md`

**Selected technical gate:** JMCG feedback contract definition before changing HFM / GeneratorCoord generation semantics.

---

## 1. Why This Gate Is Next

The previous feedback propagation audit confirmed that MoleculeForge currently has a local feedback steering path:

- workflow validation / critic failures can be serialized as `generation_feedback`
- route HUMU embeddings can be read from shared CRG by GeneratorCoord and serialized as `route_humu_feedback`
- HFM-3D can steer latent samples when feedback payloads contain HUMU embeddings

This is not yet JMCG. The CoreArchitecture v2 target defines JMCG as joint modeling of molecule structure `m`, synthesis route `r`, and property profile `p` in shared HUMU space. Before changing generation behavior, the project needs an explicit feedback contract so future code does not confuse local latent steering with true joint molecule / route / property / pocket co-generation.

## 2. Exact Files Inspected

Architecture and current status:

- `MoleculeForge_CoreArchitecture_v2.md`
- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- `moleculeforge/docs/todo/2026-06-03-generatorcoord-hfm-feedback-propagation-brief.md`

Current feedback implementation:

- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`
- `moleculeforge/agents/generator_coord/src/generator_coord/agent.py`
- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py`
- `moleculeforge/services/humu-encoder-svc/src/humu_encoder_svc/main.py`

## 3. Current Implementation Facts

- `FullWorkflowClients._attach_generation_feedback()` writes workflow feedback to `generator_params["generation_feedback"]`.
- GeneratorCoord currently derives route feedback only from CRG beliefs whose predicate is `route_humu_embedding`.
- GeneratorCoord writes parsed route feedback to `generator_params["route_humu_feedback"]`.
- HFM-3D currently scans only `route_humu_feedback` and `generation_feedback`.
- HFM-3D currently extracts steering embeddings only from `humu_embedding` or `route_humu_embedding`.
- HFM-3D currently averages all matching feedback embeddings equally, projects the mean back to the Lorentz manifold, and applies a bounded steering step.
- The current path has no explicit feedback kind, subject identity, polarity, confidence, evidence list, property axis, pocket axis, or per-record weight.

## 4. Proposed Feedback Envelope

Future generation entry points should accept one normalized envelope in addition to the legacy fields:

```json
{
  "schema": "moleculeforge.jmcg.feedback.v1",
  "run_id": "run-...",
  "project_id": "project-...",
  "records": []
}
```

`records` is a list of feedback records. Each record should be independently auditable and should not rely on list position for meaning.

## 5. Feedback Record Types

Supported initial `kind` values:

- `molecule`: feedback about a generated molecule or candidate.
- `route`: feedback about a retrosynthetic route or route family.
- `property`: feedback about a measured or predicted property objective.
- `pocket`: feedback about target pocket compatibility.
- `intent`: feedback about the compiled HCIV / intent cone direction.

Initial implementation may support only records with HUMU embeddings. Plain text or score-only records should pass through for provenance but must not affect HFM steering unless they are transformed into HUMU-space records.

## 6. Required Record Fields

Each steering-capable record should contain:

```json
{
  "kind": "route",
  "source": "retrosyn_agent",
  "run_id": "run-...",
  "subject": {
    "type": "route",
    "id": "route-..."
  },
  "humu_embedding": [0.0],
  "curvature": 1.0,
  "weight": 1.0,
  "polarity": "attract",
  "confidence": 1.0,
  "evidence_ids": ["belief-...", "artifact-..."],
  "metadata": {}
}
```

Field semantics:

- `kind`: molecule, route, property, pocket, or intent.
- `source`: agent, service, oracle, or model that produced the record.
- `run_id`: workflow run identifier used for CRG/provenance alignment.
- `subject`: typed identity of the molecule, route, property, pocket, or intent object.
- `humu_embedding`: Lorentz/HUMU coordinate in the generator latent dimension.
- `curvature`: curvature used when the embedding was produced.
- `weight`: non-negative relative strength before confidence and polarity are applied.
- `polarity`: `attract` moves toward the embedding; `repel` moves away from it.
- `confidence`: bounded in `[0, 1]`; multiplies `weight`.
- `evidence_ids`: CRG belief ids, provenance artifact ids, oracle ids, or route ids backing the record.
- `metadata`: optional details such as oracle level, score name, route score, or property value.

## 7. Legacy Mapping

Current fields should remain valid during migration:

- `route_humu_feedback` list maps to `kind="route"`, `source="generator_coord"` unless a source is present, `subject.type="route"`, `subject.id=route_id`, `polarity="attract"`, `weight=1.0`, and `confidence=1.0`.
- A `generation_feedback` item with `humu_embedding` maps according to its explicit `kind` if present; otherwise it maps to `kind="molecule"` only when a molecule/candidate identifier is present, and to `kind="property"` only when a property or oracle key is present.
- A `generation_feedback` item without HUMU embedding remains non-steering context.
- Existing `humu_embedding` and `route_humu_embedding` keys should be accepted as embedding aliases.

## 8. Weighting Rules

The first implementation should stay conservative:

- Validate that embedding dimension equals the active HFM latent dimension.
- Drop steering effect for records with invalid embedding, negative weight, invalid confidence, or unknown polarity.
- Compute effective weight as `weight * confidence`.
- For `attract`, include the embedding with positive effective weight.
- For `repel`, compute a tangent direction away from the embedding instead of treating it as a negative Euclidean average.
- Aggregate by HUMU kind before global aggregation so route feedback cannot silently dominate molecule or property feedback only because it has more records.
- Use configured kind weights with safe defaults:
  - `molecule`: 1.0
  - `route`: 0.8
  - `property`: 0.8
  - `pocket`: 1.0
  - `intent`: 1.0
- Preserve the existing global clamps: `feedback_steering_weight` and `feedback_steering_max_step`.

This still remains local steering. It should be described as "contract-shaped HUMU feedback steering", not as completed JMCG.

## 9. Non-Goals

This contract does not claim:

- true `p(m,r,p | T,c)` joint model training
- joint sampler correctness
- production-grade HUMU encoder quality
- production route/property/pocket evidence quality
- end-to-end KRAS pilot validation
- external DKI / Sigstore / cluster gate closure

Those remain separate production and research gates.

## 10. Implementation Gates After This Contract

Recommended order:

1. Add a normalized feedback parser helper in HFM-3D that accepts the new envelope while preserving legacy `route_humu_feedback` and `generation_feedback`. Completed locally on 2026-06-03.
2. Add focused unit tests for parser behavior and steering metadata, but do not run tests without explicit user instruction. Test specifications were added on 2026-06-03; pytest was not run.
3. Make GeneratorCoord emit contract-shaped route records while keeping the legacy field during migration. Completed locally on 2026-06-03.
4. Decide whether default HFM generation should receive shared CRG route feedback directly from orchestrator, or whether route feedback must continue to flow only through GeneratorCoord. Decision confirmed on 2026-06-03: default HFM must not directly read shared CRG; route feedback continues through GeneratorCoord.
5. Add property/pocket feedback producers only after there is an evidence-backed HUMU embedding source for those records.

## 11. Hard Decision Boundary

The hard architecture decision is step 4 above:

**Should default HFM generation read shared CRG route feedback without going through GeneratorCoord?**

Conservative recommendation:

- Keep shared CRG reads inside GeneratorCoord for now.
- Let default HFM consume only feedback explicitly attached by orchestrator workflow state.
- Avoid hidden repository reads inside the generator model layer.

Reason:

- It keeps HFM as a pure generation component.
- It avoids coupling model inference to Neo4j / CRG availability.
- It makes provenance and feedback attachment explicit at orchestration boundaries.

This recommendation is implementable, but it should be confirmed before changing default HFM generation semantics.

Confirmed 2026-06-03:

- Default HFM generation will not directly read shared CRG.
- Route HUMU feedback remains attached explicitly by GeneratorCoord.
- HFM-3D accepts `jmcg_feedback` only as a generator parameter payload, not as an implicit CRG repository read.

## 12. Verification Performed

Allowed static verification only:

- `rg` location checks for `JMCG`, `generation_feedback`, `route_humu_feedback`, `route_humu_embedding`, `humu_embedding`, and `pocket`
- direct code inspection of the listed files
- branch and dirty worktree baseline capture

Tests were not run.

## 13. Back-Check Result

- [x] The gate stayed focused on feedback contract semantics.
- [x] Initial contract drafting did not change code behavior; the later local implementation is recorded separately below.
- [x] The contract preserves current legacy fields.
- [x] The document explicitly separates local HUMU feedback steering from true JMCG.
- [x] The hard architecture decision is isolated before implementation.

## 14. 2026-06-03 Local Implementation Update

Implemented after the hard decision was confirmed:

- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py` now scans `jmcg_feedback` alongside legacy `route_humu_feedback` and `generation_feedback`.
- HFM-3D now accepts envelope records under `records` and extracts HUMU embeddings from contract-shaped records.
- HFM-3D steering metadata now records parsed feedback kinds through `feedback_steering_kinds`.
- `agents/generator_coord/src/generator_coord/agent.py` now builds a `moleculeforge.jmcg.feedback.v1` envelope from CRG route HUMU feedback while preserving the legacy `route_humu_feedback` field.
- GeneratorCoord returns the same `jmcg_feedback` envelope in its process result and forwards it to configured generator clients through `generator_params["jmcg_feedback"]`.
- Focused test specifications were added to `tests/unit/test_generators.py` and `tests/unit/test_generator_coord_agent.py`.

Verification:

- `python -m py_compile` passed for the touched Python files.
- `git diff --check` passed for the touched Python files.
- `rg` confirmed the new `jmcg_feedback` contract paths.
- Pytest was not run, following the project local no-test rule.

Back-check:

- [x] Default HFM still does not read shared CRG directly.
- [x] Legacy `route_humu_feedback` remains available.
- [x] The change is a compatible feedback parser/producer update, not a claim of completed JMCG.
- [x] Property and pocket feedback producers remain blocked until evidence-backed HUMU embedding sources are selected.
