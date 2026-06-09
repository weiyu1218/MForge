# W13 Cross-Paradigm KD Implementation Plan

> Execute with TDD. Keep the change local to teacher embedding artifact
> export/report. Do not change KD loss semantics or production deployment
> manifests in this gate.

## Goal

Provide a repeatable local preflight/export path for teacher embedding artifacts
used by cross-paradigm KD training.

## Tasks

### Task 1: Teacher Embedding Artifact Export

Files:

- Create: `libs/mf-core/src/mf_core/routing/kd_artifacts.py`
- Test: `tests/unit/test_cross_paradigm_kd.py`

Steps:

1. Add a failing test that writes JSONL records containing
   `teacher_embedding`, calls `export_teacher_embeddings_artifact()`, and
   asserts that the output JSON contains schema, count, dimension, and
   `teacher_embeddings`.
2. Implement JSON/JSONL loading, finite validation, consistent dimension
   validation, optional expected-dimension validation, and canonical output.
3. Re-run the focused test and verify it passes.

### Task 2: Teacher Embedding Artifact Report

Files:

- Modify: `libs/mf-core/src/mf_core/routing/kd_artifacts.py`
- Test: `tests/unit/test_cross_paradigm_kd.py`

Steps:

1. Add a failing test that builds a report for a mismatched artifact and
   asserts `status == "fail"` with a dimension message.
2. Implement `build_teacher_embeddings_report()` and CLI options:
   `--input`, `--output`, `--embedding-field`, `--expected-dim`,
   `--min-embeddings`, `--report`, `--strict`.
3. Re-run focused tests, py_compile, and diff check.

### Task 3: Documentation

Files:

- Modify: `docs/todo/owner-a-generation-upstream/progress.md`
- Modify: `docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- Modify: `docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
- Modify: `docs/architecture/current-implementation-vs-corearchitecture-v2.md`

Steps:

1. Record W13 local artifact gate.
2. Mark remaining production teacher/training/cluster gates as incomplete.
3. Back-check that W13 did not alter HUMU/HFM or Owner B code.
