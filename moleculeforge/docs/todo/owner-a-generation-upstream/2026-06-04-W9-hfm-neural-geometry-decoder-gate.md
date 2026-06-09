# W9 HFM Neural Geometry Decoder Gate

## Scope

Add a concrete local neural geometry decoder train/export/runner path for HFM-3D without changing HUMU pretraining or the HFM Lorentz flow architecture.

This gate implements engineering readiness only. It does not claim production molecular geometry quality.

## Modified

- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/__init__.py`
- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/neural_geometry_decoder.py`
- `models/mf-generators/hfm_3d/train_geometry_decoder.py`
- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py`
- `tests/unit/test_generators.py`

## Implemented

- Added SDF-backed source artifact loading from existing HFM decoder JSON entries.
- Added a bounded fixed-size `NeuralGeometryDecoder` MLP from 129-dimensional HFM latent to padded 3D coordinates.
- Added masked MSE tiny training/export helper and a torch artifact schema containing:
  - model state,
  - latent dimension,
  - max atom count,
  - nearest-entry SMILES / atom types / latent metadata,
  - source decoder artifact path.
- Added `python -m mf_generators.hfm_3d.decoder.neural_geometry_decoder --artifact <artifact.pt>` stdin/stdout runner for the existing HFM molecular decoder JSON contract.
- Added `models/mf-generators/hfm_3d/train_geometry_decoder.py` CLI wrapper.
- Updated HFM generator metadata handling so decoder-supplied `metadata.decoder_mode` is preserved; payloads without a decoder mode still default to `molecular_decoder`.

## Verification

RED first:

- `test_neural_geometry_decoder_loads_sdf_training_examples` failed because `mf_generators.hfm_3d.decoder` did not exist.
- `test_neural_geometry_decoder_trains_tiny_artifact` failed because training/export helpers did not exist.
- `test_neural_geometry_decoder_runner_outputs_hfm_contract` failed because the module runner emitted no JSON.
- `test_hfm_generator_consumes_neural_geometry_decoder_output` failed because generator metadata overwrote `neural_geometry_decoder`.
- `test_geometry_decoder_training_cli_writes_artifact` failed because `train_geometry_decoder.py` did not exist.

GREEN / regression:

- Focused W9 + legacy decoder gate passed:
  - `uv run pytest tests/unit/test_generators.py::TestHFM3DTraining::test_geometry_decoder_training_cli_writes_artifact tests/unit/test_generators.py::TestHFM3DGenerator::test_neural_geometry_decoder_loads_sdf_training_examples tests/unit/test_generators.py::TestHFM3DGenerator::test_neural_geometry_decoder_trains_tiny_artifact tests/unit/test_generators.py::TestHFM3DGenerator::test_neural_geometry_decoder_runner_outputs_hfm_contract tests/unit/test_generators.py::TestHFM3DGenerator::test_hfm_generator_consumes_neural_geometry_decoder_output tests/unit/test_generators.py::TestHFM3DGenerator::test_external_molecular_decoder_command_decodes_latent -q`
  - result: 6 passed, exit code 0.
- Static compile passed:
  - `python -m py_compile moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/__init__.py moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/neural_geometry_decoder.py moleculeforge/models/mf-generators/hfm_3d/train_geometry_decoder.py moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py moleculeforge/tests/unit/test_generators.py`
- File-level generator regression passed:
  - `uv run pytest tests/unit/test_generators.py -q`
  - result: 65 items in this shard, exit code 0.
- `git diff --check` passed.

## Remaining Gate

- Train a real neural geometry decoder artifact on real production data.
- Decide whether production should use `HFM_DECODER_PATH` nearest-neighbor fallback, direct injected decoder, or `HFM_MOLECULAR_DECODER_COMMAND`.
- Deploy the production artifact / command value.
- Run cluster validation and benchmark geometry quality.

## Back-Check

- [x] Existing HFM molecular decoder contract was reused, not replaced.
- [x] HUMU pretraining remained frozen.
- [x] HFM Lorentz flow architecture remained frozen.
- [x] Existing smoke checkpoint `checkpoints/hfm3d_4h200` remains smoke evidence only.
- [x] Owner B code was not modified.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.
