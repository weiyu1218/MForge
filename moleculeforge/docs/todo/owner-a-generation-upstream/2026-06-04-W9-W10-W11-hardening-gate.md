# 2026-06-04 W9/W10/W11 Hardening Gate

## Scope

This gate resolves the stage re-acceptance findings for Owner A local engineering paths:

- W9 HFM neural geometry decoder accepted invalid 129-dimensional latent vectors in source decoder artifacts.
- W10 HCIV supervised training accepted invalid `target_hciv` coordinates.
- W11 FragFM quality report treated checkpoint/rate-matrix artifacts as loadable even when key schema entries were missing.

This gate does not change HUMU pretraining, HUMU encoder architecture, HFM architecture, checkpoints, Owner B code, or `/workspace/SemMol` / `/workspace/Projects`.

## Implementation

- W9 now validates source decoder artifact `latent` values through `mf_core.geometry.normalize_lorentz_embedding()` before creating `GeometryTrainingExample` records.
- W10 now validates `target_hciv` coordinates through the same Lorentz helper and threads `curvature` through HCIV training-data loading.
- W11 now requires FragFM checkpoint artifacts to contain `fragment_encoder.weight` and rate-matrix artifacts to contain `base_rate`; both schema keys must have compatible shapes.
- W9 test fixture latent coordinates were corrected so positive examples are valid Lorentz full-coordinate vectors.
- Shared interface status now marks W2 `_jmcg_context_feedback_from_state` and HFM `_feedback_embedding_records` as completed.

## Verification

RED first:

- `uv run pytest tests/unit/test_generators.py::TestHFM3DGenerator::test_neural_geometry_decoder_rejects_invalid_lorentz_latent tests/unit/test_cic_compiler.py::TestCICCompiler::test_hciv_training_examples_reject_invalid_lorentz_target tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_checkpoint_schema_missing tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_rate_matrix_schema_missing -q`
- Result: 4 failed with expected symptoms: invalid Lorentz inputs did not raise, and missing FragFM schema keys still reported `pass`.

GREEN and adjacent checks:

- Same 4 focused tests passed with exit code 0.
- Adjacent focused pytest passed with exit code 0: 13 items covering W9 valid examples/training/runner/generator consume, W10 valid load/train/CLI, and W11 pass/fail quality reports.
- `python -m py_compile moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/neural_geometry_decoder.py moleculeforge/services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_training.py moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py moleculeforge/tests/unit/test_generators.py moleculeforge/tests/unit/test_cic_compiler.py` passed.
- `git diff --check` passed for touched hardening files and the shared interface document.
- W11 strict quality CLI smoke still passed on the local runtime artifact with threshold 0.0: `pass 50 0 0.0 True True`.

## Back-Check

- The fix addresses root validation gaps, not only the specific all-zero examples.
- Legal Lorentz origin vectors remain valid.
- Existing W9/W10 local training/export paths remain runnable.
- W11 local artifact remains runtime-smoke only because `humu_embedding_coverage=0.0`.
- Production gates remain: real HFM decoder artifact, real supervised HCIV data/checkpoint, real HUMU-labeled FragFM data/artifact, benchmark thresholds, cluster validation.
