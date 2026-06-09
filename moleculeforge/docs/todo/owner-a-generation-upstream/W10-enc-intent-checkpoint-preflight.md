# W10 Enc_intent Checkpoint Preflight

## Goal

Prepare W10 before implementation. The project already has an `HCIVEncoder` baseline and production checkpoint loading. The missing Owner A local code path is a concrete train/export script that can produce a checkpoint suitable for `HCIV_CHECKPOINT_PATH`.

## Current Evidence

- `services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_encoder.py` already contains:
  - `HCIVEncoder` with objective node, directed edge, directed hyperedge and graph projection components.
  - `cig_to_features()` for deterministic feature extraction.
  - `load_hciv_encoder_checkpoint()` using `torch.load(..., weights_only=True)`.
  - `hash_encode_hciv()` for local demo fallback only.
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/compiler.py` already enforces:
  - `production_real` defaults to learned HCIV encoding.
  - `hash` / `random` modes are local-demo only.
  - `HCIV_CHECKPOINT_PATH` is required in production when no learned encoder is injected.
- `tests/unit/test_cic_compiler.py` already covers:
  - production learned mode fail-fast without `HCIV_CHECKPOINT_PATH`,
  - production learned mode loads a raw `HCIVEncoder.state_dict()`,
  - directed topology sensitivity,
  - CIG JSON-LD / objective edge / hyperedge behavior.
- Deployment wiring already exposes `HCIV_CHECKPOINT_PATH` as an empty production resource slot in the CIG compiler config.

## Gap

There is no local training/export entry point that can:

1. load training records containing `cig` and supervised `target_hciv`,
2. train the existing `HCIVEncoder`,
3. write an artifact compatible with `load_hciv_encoder_checkpoint()`,
4. write a small manifest that records schema, source data, dimension and example count,
5. verify the exported checkpoint can be loaded by `CIGCompiler` in production learned mode.

## Recommended Scope

Implement a small supervised engineering path:

- `cig_compiler_svc.domain.hciv_training`
  - `HCIVTrainingExample`
  - `load_hciv_training_examples()`
  - `train_hciv_encoder_checkpoint()`
- `services/cig-compiler-svc/train_hciv_encoder.py`
  - CLI wrapper around the training helper.

Training records should require explicit target HCIV coordinates. Do not self-distill from `hash_encode_hciv()` because that would make a production checkpoint imitate the local demo path.

## Out Of Scope

- Do not change HUMU pretraining.
- Do not change HFM.
- Do not change the HCIV encoder architecture unless a test proves the current loader/export contract cannot work.
- Do not claim production quality before real CIG/HCIV training data and validation metrics exist.
- Do not inject 128-dimensional HCIV vectors into HFM `humu_embedding`; W2 rules still apply.

## Acceptance

Local engineering acceptance:

- A tiny JSONL supervised dataset can train for one CPU epoch.
- The output checkpoint exists and is loadable through `CIGCompiler` with `HCIV_CHECKPOINT_PATH`.
- The output HCIV has `dim + 1` Lorentz full coordinates.
- Static compile and focused CIC tests pass.

Remaining production gates:

- real supervised CIG/HCIV training data,
- real training run,
- production `HCIV_CHECKPOINT_PATH` value,
- cluster validation,
- downstream quality evidence for intent-conditioned generation.

## Back-Check

- [x] Existing production checkpoint loader is reused.
- [x] The scope is training/export, not a new CIG compiler service path.
- [x] Hash/random encoders remain local-demo only.
- [x] W2 128/129 HUMU steering boundary remains unchanged.
- [x] HUMU pretraining and HFM are out of scope.
