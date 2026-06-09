# W9 HFM Neural Geometry Decoder Preflight

## Goal

Prepare W9 before implementation. The current HFM path already has decoder interfaces, but the default local artifact remains a nearest-neighbor decoder plus stored SDF/runtime conformer fallback. W9 should add a concrete neural geometry decoder training/runner path without changing HUMU pretraining or HFM flow architecture.

## Current Evidence

- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py` already supports:
  - `molecular_decoder` injection.
  - `HFM_MOLECULAR_DECODER_COMMAND` via `ExternalMolecularDecoder`.
  - decoder output as `Molecule` or JSON object.
  - JSON output fields `smiles`, `atom_types`, `coordinates`, `sdf` / `sdf_bytes`, `metadata`, and `decoder_entry_id`.
  - direct SDF construction from decoder `atom_types` + `coordinates`.
- `services/hfm-generator-svc/src/hfm_generator_svc/main.py` already accepts either:
  - `HFM_CHECKPOINT_PATH + HFM_DECODER_PATH`, or
  - `HFM_CHECKPOINT_PATH + HFM_MOLECULAR_DECODER_COMMAND`.
- Deployment wiring already exposes `HFM_MOLECULAR_DECODER_COMMAND`.
- `models/mf-generators/hfm_3d/train.py` currently trains:
  - Lorentz flow model.
  - a fingerprint decoder head.
  - a nearest-neighbor decoder artifact with optional SDF entries.
- The current local HFM checkpoint in `checkpoints/hfm3d_4h200` is smoke/full-flow evidence only, not production geometry quality.

## Gap

The missing local AI-code piece is not another wrapper. The missing piece is a concrete command target and training/export path that can:

1. load a neural geometry decoder artifact,
2. consume a post-flow 129-dimensional HFM latent from stdin JSON,
3. return `smiles`, `atom_types`, and `coordinates` or `sdf`,
4. plug into the existing `HFM_MOLECULAR_DECODER_COMMAND` interface,
5. fail fast when the artifact is missing or incompatible.

## Recommended Scope

Implement a local neural-geometry decoder package under HFM-3D:

- `mf_generators.hfm_3d.decoder.neural_geometry_decoder`
  - artifact dataclasses / load-save helpers,
  - fixed-vocabulary atom symbol mapping,
  - latent-to-coordinate neural module,
  - nearest-entry SMILES selection using existing decoder entries,
  - coordinate prediction for the selected molecule atom count,
  - JSON command runner.
- `models/mf-generators/hfm_3d/train_geometry_decoder.py`
  - trains the geometry decoder from an existing decoder artifact containing latent + SDF entries,
  - writes a torch artifact + JSON manifest.

This scope is an engineering production path, not a claim of W9 research-quality decoder performance.

## Out Of Scope

- Do not modify HUMU pretraining, HUMU checkpoint continuation, HUMU encoder architecture, or HFM Lorentz flow architecture.
- Do not replace the existing `HFM_MOLECULAR_DECODER_COMMAND` contract.
- Do not claim production quality before a real artifact is trained and benchmarked.
- Do not use the one-epoch `checkpoints/hfm3d_4h200` checkpoint as final quality evidence.

## Acceptance

Local engineering acceptance:

- A tiny SDF-backed decoder artifact can train for one CPU epoch in a unit/smoke test.
- The runner can decode one latent JSON request to valid JSON with `smiles`, `atom_types`, and `coordinates`.
- `HFM3DGenerator` with a configured molecular decoder object can consume the runner-compatible decoder output and produce SDF bytes.
- Runtime failures for missing/incompatible artifacts are explicit.

Remaining production gates:

- real training data,
- real trained geometry decoder artifact,
- `HFM_MOLECULAR_DECODER_COMMAND` production env value,
- cluster validation,
- benchmark quality evidence.

## Back-Check

- [x] Existing wrapper and service wiring are recognized as already present.
- [x] The proposed scope is a concrete local command target + train/export path.
- [x] HUMU pretraining remains frozen.
- [x] HFM flow architecture remains frozen.
- [x] The plan separates engineering acceptance from production quality.
