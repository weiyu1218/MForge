# 2026-06-04 Embedding Validation Hardening Gate

## Scope

This gate resolves the Owner A stage-gate finding that W2 and W8-E accepted any numeric 129-dimensional list as steering-capable.

Touched code:

- `libs/mf-core/src/mf_core/geometry/lorentz.py`
- `libs/mf-core/src/mf_core/geometry/__init__.py`
- `services/orchestrator-svc/src/orchestrator_svc/main.py`
- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py`
- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/jmcg_sampler.py`

Touched tests:

- `tests/unit/test_generators.py`
- `tests/unit/test_service_artifact_status.py`

## Implementation

- Added `mf_core.geometry.lorentz.normalize_lorentz_embedding()`.
- The helper rejects non-finite values, non-positive time coordinates, wrong dimensions, invalid curvature, and vectors that do not satisfy the Lorentz hyperboloid equation within tolerance.
- W2 intent/pocket embedding validation now uses the shared helper.
- HFM feedback consumer now rejects invalid 129-dimensional feedback records instead of projecting arbitrary vectors into a target.
- W8-E alignment scoring now uses the shared helper and counts invalid present embeddings as ignored.
- W8-E now decodes packed little-endian float32 `Molecule.humu_embedding` bytes, while still accepting UTF-8 JSON bytes.

## Verification

RED first:

- `uv run pytest tests/unit/test_generators.py::TestHFM3DGenerator::test_jmcg_feedback_drops_invalid_lorentz_embeddings tests/unit/test_generators.py::TestHFM3DGenerator::test_jmcg_engineering_sampler_rejects_invalid_lorentz_embeddings tests/unit/test_generators.py::TestHFM3DGenerator::test_jmcg_engineering_sampler_decodes_packed_float32_molecule_embedding tests/unit/test_service_artifact_status.py::test_full_workflow_invalid_lorentz_intent_axis_stays_non_steering -q` failed with the expected four failures.

GREEN:

- The same four focused tests passed.
- `python -m py_compile moleculeforge/libs/mf-core/src/mf_core/geometry/__init__.py moleculeforge/libs/mf-core/src/mf_core/geometry/lorentz.py moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/jmcg_sampler.py moleculeforge/tests/unit/test_generators.py moleculeforge/tests/unit/test_service_artifact_status.py` passed.
- `git diff --check` passed for touched files.
- Adjacent focused HFM/W8-E pytest passed: 6 items.
- Adjacent focused W2 pytest passed: 4 items.
- File-level focused pytest passed: `uv run pytest tests/unit/test_generators.py tests/unit/test_service_artifact_status.py -q` ran 273 items with exit code 0 and one existing LangGraph deprecation warning.

## Back-Check

- The fix addresses the root cause instead of special-casing all-zero vectors.
- Legal Lorentz origin `[1.0, 0.0, ...]` remains valid.
- Projected HUMU/HFM Lorentz points remain valid.
- Invalid 129-dimensional all-zero vectors are now non-steering.
- Packed float32 molecule HUMU bytes are no longer silently ignored in W8-E.
- HUMU pretraining, HUMU encoder architecture, HFM architecture, and checkpoints were not modified.
- Owner B files were not modified.
- `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.

## Remaining Gates

- W8-R true joint sampling training quality remains open.
- W6 production reward data and deployment validation remain open.
- W9 HFM decoder is now the next Owner A engineering scope candidate.
