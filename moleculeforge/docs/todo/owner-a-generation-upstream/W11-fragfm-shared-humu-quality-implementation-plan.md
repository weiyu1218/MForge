# W11 FragFM Shared HUMU Quality Implementation Plan

> For agentic workers: execute task-by-task with TDD. Do not modify HUMU
> pretraining, HUMU encoder architecture, HFM architecture, or Owner B code.

## Goal

Make FragFM's local artifact path preserve shared HUMU conditional evidence and
provide a repeatable quality gate for artifact readiness.

## Architecture

The training CLI remains the artifact producer. It will validate optional
rule-level HUMU embeddings and carry them into `vocab.json` plus manifest
metadata. A new small `mf_generators.fragfm.quality` module will inspect a
vocabulary/checkpoint/rate-matrix artifact set and report whether it is ready
for shared HUMU conditional validation.

## Tasks

### Task 1: Preserve Training HUMU Embeddings

Files:

- Modify: `models/mf-generators/fragfm/train.py`
- Test: `tests/unit/test_generators.py`

Steps:

1. Add a failing test that writes FragFM JSONL records with valid 129-dimensional
   Lorentz `humu_embedding` values, runs the training CLI, and asserts that
   `vocab.json` preserves them and `training_manifest.json` reports full HUMU
   coverage.
2. Run only that test and verify it fails because the embedding is missing.
3. Update `_normalize_record()` to validate optional `humu_embedding` with
   `normalize_lorentz_embedding(expected_dim=129)`.
4. Update `_write_vocab_artifact()` and manifest creation to retain
   `humu_embedding`, `humu_embedding_count`, and `humu_embedding_coverage`.
5. Re-run the focused test and verify it passes.

### Task 2: Add FragFM Artifact Quality Report

Files:

- Create: `models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py`
- Test: `tests/unit/test_generators.py`

Steps:

1. Add failing tests for:
   - a passing report when all rules have valid HUMU embeddings and artifacts
     load;
   - a failing report when HUMU coverage is below threshold.
2. Implement `build_quality_report()` with JSON-serializable output:
   `status`, `rules`, `humu_embedding_count`, `humu_embedding_coverage`,
   `invalid_humu_embeddings`, `checkpoint_loadable`, `rate_matrix_loadable`,
   `messages`.
3. Add `python -m mf_generators.fragfm.quality` CLI options for vocab,
   checkpoint, rate matrix, min coverage, output path, HUMU dim, curvature,
   and strict exit status.
4. Re-run the focused tests and verify they pass.

### Task 3: Verification And Docs

Files:

- Modify: `docs/todo/owner-a-generation-upstream/progress.md`
- Modify: `docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- Modify: `docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
- Modify: `docs/architecture/current-implementation-vs-corearchitecture-v2.md`

Steps:

1. Run focused W11 pytest.
2. Run file-level `tests/unit/test_generators.py` if focused tests pass.
3. Run `python -m py_compile` on touched Python files.
4. Run `git diff --check`.
5. Update progress and architecture docs with completed local engineering gate
   and remaining production gates.

## Back-Check

- [x] The plan keeps W11 limited to local engineering readiness.
- [x] It does not require changing frozen HUMU pretraining.
- [x] It creates a quality gate instead of claiming current artifact quality.
