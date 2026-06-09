# W2 Pocket / Intent Embedding Preflight

## Purpose

Prepare W2 before changing code. The risk is turning context-only records into steering records without a reliable HUMU-space embedding source or with the wrong Lorentz vector dimension.

## Current Facts

- Property / intent / pocket records currently exist as `moleculeforge.jmcg.feedback.v1` context records.
- They intentionally omit `humu_embedding`, so HFM ignores them for steering.
- HFM only steers from records containing `humu_embedding` or `route_humu_embedding`.
- HFM drops feedback records whose embedding length does not equal the active latent dimension.
- Current HFM latent dimension is 129.
- Current HUMU mol / pocket / route encoders are constructed with `dim=128` but output Lorentz full coordinates with length 129.
- `humu/encoder.proto` documents `EncodeResponse.humu_embedding` as 129-dim float32.
- `retrosyn/route.proto` documents route HUMU embeddings as 129-dim Lorentz embeddings.
- Existing RetroSyn route encoding already uses `HUMU_ENCODER_TARGET` and decodes float32 embedding bytes with `struct.iter_unpack("<f", payload)`.

## W2 Rules

- Pocket records may become steering-capable only when pocket input contains enough structured pocket geometry for HUMU pocket encoding.
- Required pocket geometry should include at least coordinates plus per-coordinate element and residue type fields matching the current pocket encoder contract.
- Metadata-only pocket context, such as `pdb_id` or `pocket_id` alone, must remain non-steering.
- Intent records may become steering-capable only if the source is already a Lorentz full-coordinate vector matching HFM active latent dimension, or if a specific intent encoder / projection source is selected.
- A plain 128-dimensional HCIV vector must not be inserted directly as HFM `humu_embedding` without a justified Lorentz full-coordinate conversion.
- Any new steering-capable record must preserve `source`, `subject`, `weight`, `polarity`, `confidence`, `evidence_ids`, and `metadata`.

## Proposed W2 Scope

Allowed shared file, after explicit progress registration:

- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`

Preferred implementation shape:

- Add a narrow orchestrator-side helper/client for optional pocket encoding.
- Reuse the existing `HUMU_ENCODER_TARGET` convention used by RetroSynAgent.
- Decode `EncodeResponse.humu_embedding` as packed float32 values.
- Make the helper fail closed: if target is unavailable or pocket geometry is insufficient, keep the record non-steering instead of fabricating an embedding.

Likely test-spec files, not executed without explicit authorization:

- `moleculeforge/tests/unit/test_service_artifact_status.py`
- `moleculeforge/tests/unit/test_generators.py`

Do not modify in W2 preflight:

- HUMU pretraining code.
- HUMU encoder architecture.
- HFM model architecture.
- `/workspace/SemMol`
- `/workspace/Projects`

## Open Implementation Questions

1. Should W2 first support only already-present 129-dimensional intent axes, leaving HCIV conversion for a later gate?
2. Should pocket encoding be synchronous within orchestrator generation params, or delegated through an explicit service client hook?
3. Should the W2 helper support injected test doubles first, then wire gRPC via `HUMU_ENCODER_TARGET` in the production path?

Current recommendation:

- Start with injected helper/test-double support and `HUMU_ENCODER_TARGET` gRPC wiring in the same helper.
- Treat pocket encoding as optional enrichment: successful structured pocket encoding adds `kind="pocket"` `humu_embedding`; failure or insufficient fields preserves the current non-steering record.
- For intent, do not convert HCIV directly in W2. Only accept an already Lorentz full-coordinate 129-dimensional axis/source if present; otherwise keep intent non-steering until an explicit intent encoder/projection gate is selected.

## Back-Check Criteria

- [ ] No metadata-only context is treated as a HUMU embedding.
- [ ] All steering-capable embeddings are 129-dimensional in the current HFM path.
- [ ] W2 does not modify HUMU pretraining or HFM architecture.
- [ ] Existing route feedback behavior is not changed.
