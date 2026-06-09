# W8-E JMCG Engineering Skeleton Preflight

Date: 2026-06-04

Owner: A / generation upstream

## Scope

W8-E is the engineering skeleton gate for JMCG. It should introduce a local, testable contract for joint `(molecule, route, property, pocket, intent)` samples in shared HUMU/Lorentz space.

This gate is not W8-R. It must not claim trained joint generation quality.

## Current Evidence

Existing JMCG-adjacent code:

- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`
  - builds `moleculeforge.jmcg.feedback.v1` records for property / intent / pocket context;
  - W2 upgrades eligible pocket / intent records to steering-capable 129-dimensional HUMU embeddings.
- `moleculeforge/agents/generator_coord/src/generator_coord/agent.py`
  - preserves existing `jmcg_feedback.records`;
  - appends route HUMU feedback records from CRG.
- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py`
  - consumes `jmcg_feedback`, `route_humu_feedback`, and `generation_feedback`;
  - validates embedding dimension against active 129-dimensional latent points;
  - performs bounded Lorentz latent steering with per-kind aggregation and dropped-record metadata.
- `moleculeforge/tests/unit/test_humu_training.py`
  - already covers HUMU pretraining-side `joint_source` / `intent_source` data contracts.

Current gap:

- There is no generation-side JMCG sampler object that produces explicit `(m,r,p)` or `(m,r,p,pocket,intent)` joint sample records.
- HFM feedback steering remains a local post-flow latent steering path, not a joint sampler.
- No module currently validates or serializes W8-E joint sample records for downstream engineering use.

## Recommended Local Skeleton

Create:

- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/jmcg_sampler.py`

Export from:

- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/__init__.py`

Focused specs:

- `moleculeforge/tests/unit/test_generators.py`

The sampler should:

- parse `moleculeforge.jmcg.feedback.v1` envelopes and legacy record lists;
- accept molecule candidates with SMILES, optional HUMU embedding, and optional latent metadata;
- accept route candidates and property profiles;
- validate steering-capable embeddings as 129-dimensional Lorentz full coordinates;
- produce serializable joint samples with fields for molecule / route / property / pocket / intent;
- compute deterministic alignment metadata from available HUMU embeddings;
- keep missing embeddings as non-steering context rather than fabricating embeddings.

## Non-Goals

- Do not change HUMU pretraining, loss, encoder architecture, or checkpoint continuation.
- Do not train a new JMCG model.
- Do not change HFM flow, checkpoint loading, decoder behavior, or production generation defaults.
- Do not require real joint training data.
- Do not claim W8-R research completion.
- Do not modify or execute `/workspace/SemMol` or `/workspace/Projects`.

## Acceptance For W8-E Local Skeleton

- A local sampler can construct at least one legal joint sample from a small molecule candidate plus route/property/context feedback.
- A 129-dimensional HUMU embedding is accepted into alignment scoring.
- A 128-dimensional embedding is rejected or treated as non-steering context.
- The output is JSON-serializable and explicitly marked `engineering_skeleton`.
- Documentation keeps production joint training and quality validation as remaining gates.

## Back-Check

- [x] This preflight separates W8-E engineering skeleton from W8-R research quality.
- [x] The proposed module is local to HFM inference and does not alter default production generation.
- [x] Current 129-dimensional HUMU/HFM embedding contract is preserved.
- [x] HUMU pretraining remains frozen.
- [x] No business code was modified by this preflight.
- [x] No tests were run.
