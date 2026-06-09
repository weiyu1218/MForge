# W10 Enc_intent Checkpoint Gate

## Scope

Add a concrete local supervised training/export path for the existing CIG `HCIVEncoder` so it can produce checkpoints compatible with `HCIV_CHECKPOINT_PATH`.

This gate implements engineering readiness only. It does not claim production `Enc_intent` quality.

## Modified

- `services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_encoder.py`
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_training.py`
- `services/cig-compiler-svc/train_hciv_encoder.py`
- `tests/unit/test_cic_compiler.py`

## Implemented

- Added supervised HCIV training dataset loading from JSON or JSONL records.
- Training records require explicit `cig` plus `target_hciv` coordinates.
- Added target validation for `dim + 1` finite Lorentz full-coordinate vectors.
- Added `HCIVEncoder.forward_coordinates(cig)` so training can use a differentiable path while `encode()` keeps the existing `HCIV` / `IntentCone` output contract.
- Added `train_hciv_encoder_checkpoint()` to train tiny CPU artifacts and export a torch checkpoint containing:
  - schema,
  - `state_dict`,
  - dimension,
  - curvature.
- Added optional JSON manifest export with source data, example count, epoch count, batch size and final loss.
- Added `services/cig-compiler-svc/train_hciv_encoder.py` CLI wrapper.

## Verification

RED first:

- `test_hciv_training_examples_load_cig_and_target` failed because `cig_compiler_svc.domain.hciv_training` did not exist.
- `test_train_hciv_encoder_checkpoint_writes_loadable_artifact` failed because `train_hciv_encoder_checkpoint()` did not exist, then exposed the valid design issue that `HCIVEncoder.encode()` is detached and not trainable.
- `test_train_hciv_encoder_cli_writes_checkpoint_and_manifest` failed because `train_hciv_encoder.py` did not exist.

GREEN / regression:

- Focused W10 gate passed:
  - `uv run pytest tests/unit/test_cic_compiler.py::TestCICCompiler::test_hciv_training_examples_load_cig_and_target tests/unit/test_cic_compiler.py::TestCICCompiler::test_train_hciv_encoder_checkpoint_writes_loadable_artifact tests/unit/test_cic_compiler.py::TestCICCompiler::test_train_hciv_encoder_cli_writes_checkpoint_and_manifest tests/unit/test_cic_compiler.py::TestCICCompiler::test_production_learned_loads_checkpoint -q`
  - result: 4 passed, exit code 0.
- Static compile passed:
  - `python -m py_compile moleculeforge/services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_encoder.py moleculeforge/services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_training.py moleculeforge/services/cig-compiler-svc/train_hciv_encoder.py moleculeforge/tests/unit/test_cic_compiler.py`
- File-level CIC regression passed:
  - `uv run pytest tests/unit/test_cic_compiler.py -q`
  - result: 31 items in this shard, exit code 0.
- `git diff --check` passed.

## Remaining Gate

- Real supervised CIG/HCIV training data is still missing.
- A real production-quality `Enc_intent` checkpoint still needs training.
- Production `HCIV_CHECKPOINT_PATH` remains a resource/env deployment gate.
- Cluster validation and downstream intent-conditioned generation quality evidence remain missing.

## Back-Check

- [x] Existing `HCIVEncoder` architecture was reused.
- [x] Existing production `HCIV_CHECKPOINT_PATH` loader was reused.
- [x] Hash/random HCIV encoders remain local-demo only.
- [x] W2 128/129 HUMU steering boundary remains unchanged.
- [x] HUMU pretraining and HFM were not modified.
- [x] Owner B code was not modified.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.
