# Owner A Code-Freeze And Owner B Handoff Checklist

Date: 2026-06-06
Scope: Owner A generation-upstream code-freeze review before Owner B
coordination

## Purpose

This checklist is the current Owner A handoff surface after the W11 code
hardening pass. It separates three statuses that must not be mixed:

- Owner A code-writing readiness.
- Owner B coordination and acceptance.
- Production training, official benchmark, and cluster acceptance.

The current user direction is to finish code engineering first, coordinate with
Owner B second, and run real tests/training only later. This document does not
authorize large-scale training, benchmark threshold changes, Owner B code
edits, production artifact promotion, or cluster deployment changes.

## Code-Freeze Position

Owner A code is close to code-freeze for local engineering handoff, but the
overall client task is not production-complete.

Local engineering work now exists for:

- W2 pocket / intent HUMU feedback producer and HFM feedback consumption.
- W6 TAR ProxylessNAS command target.
- W8-E JMCG engineering skeleton.
- W9 HFM neural geometry decoder train/export/runner path.
- W10 HCIV supervised train/export path.
- W11 FragFM HUMU-labeled quality path, local 5k candidate, deployment-default
  hardening, service/runtime hardening, and sample export hardening.
- W13 KD teacher embedding artifact export/report path.

Production acceptance still lacks:

- Owner B final coordination on W1/W3/W5/W12 and shared contract review.
- Production-quality source data and artifacts for W6, W9, W10, W11, and W13.
- Official benchmark evidence without threshold relaxation.
- Cluster cold-start/readiness/request evidence.
- Permanent artifact promotion decisions.

## Owner A Artifacts To Hand Off

| Gate | Owner A deliverable | Current evidence | Handoff caveat |
|---|---|---|---|
| W2 | Async optional intent/pocket feedback enrichment; 129-dim Lorentz steering guard | Focused W2 and HFM/JMCG tests recorded in `2026-06-05-W4-focused-validation-record.md` | 128-dim payloads remain non-steering; HUMU pretraining remains frozen |
| W6 | `python -m generator_router_svc.tar_proxyless_runner` command target | Runner smoke and service contract recorded in W6 docs | Needs real reward payload, production env value, and cluster evidence |
| W8-E | `JMCGEngineeringSampler` skeleton and C1 feedback contract | W8-E is local engineering-complete in the handoff | W8-R research-quality joint training is not complete |
| W9 | Neural geometry decoder train/export/runner path | W9 focused tests and run plan recorded | Current decoder artifact is smoke-only; needs real latent/SDF data and production artifact |
| W10 | HCIV supervised training/export path | W10 focused tests and run plan recorded | Needs real supervised CIG/HCIV data and production checkpoint |
| W11 | FragFM HUMU-labeled local 5k candidate, quality CLI, deployment defaults, runtime/sample-export hardening | `checkpoints/fragfm_humu_5k/` strict-local candidate, focused W11 tests, sample reports | Local engineering evidence only; not production W11 acceptance |
| W13 | KD teacher embedding artifact utility | W13 focused tests and CLI smoke recorded | Needs real teacher source and distillation evidence |

## Owner B Handoff Items

### W1 CRG Unit Patch Seam

Owner B should receive the W1 unit compatibility issue from
`docs/todo/owner-a-generation-upstream/2026-06-05-W4-focused-validation-record.md`.

Reproducible failing tests from that record:

- `test_merge_agent_beliefs_merges_shared_crg_into_final_state`
- `test_merge_agent_beliefs_deduplicates_existing_beliefs`
- `test_merge_agent_beliefs_falls_through_when_no_repository`

The failing patch target was:

```text
orchestrator_svc.main.build_shared_crg_repository_from_env
```

The implementation imports that symbol inside `_merge_agent_beliefs_into_crg()`.
Owner B should either patch `mf_core.db.repositories.build_shared_crg_repository_from_env`
in the tests or deliberately expose a module-level import seam in
`orchestrator_svc.main`. Owner A should not modify Owner B tests or W1
implementation unless explicitly authorized.

### W5 Benchmark Gate

Owner B should keep W5 blocked until official benchmark data and
production-quality generated samples exist.

Known blocked resources:

- `MOSES_REFERENCE_SMILES_PATH`
- `FRAGFM_MOSES_GENERATED_SMILES_PATH`
- `PMO_SCORE_TABLE_PATH`
- `CROSSDOCKED_BENCHMARK_JSONL`
- production-quality HFM or FragFM generated sample artifacts

Do not relax MOSES, GuacaMol, or PMO thresholds. The local 8/64/256 FragFM SMILES
exports are wiring evidence only.

### Contract Surfaces To Reconfirm

Owner A and Owner B should confirm that no new fields or predicates are being
introduced before the next integration step:

- C1 `generator_params` feedback envelope:
  `moleculeforge.jmcg.feedback.v1`.
- C2 CRG predicate table, especially `route_humu_embedding`.
- C3 HUMU encoder interface, including current 129-dimensional Lorentz
  full-coordinate output.

The active contract source is
`docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`.

## Code-Freeze Review Checklist

Run these before claiming Owner A code is ready for Owner B review. These are
focused checks, not full production acceptance.

### Documentation And Process Safety

```bash
git diff --check -- \
  docs/todo/owner-a-generation-upstream \
  docs/architecture/corearchitecture-v2-completion-interface-acceptance.md \
  docs/architecture/corearchitecture-v2-completion-tasksplit.md
```

```bash
ps -eo pid,etime,stat,cmd | \
  rg -n 'pytest|models/mf-generators/.*/train.py|fragfm_generator_svc|mf_generators.fragfm|hfm_3d/train|train_hciv|tar_proxyless' || true
```

### W11 Focused Engineering Checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/unit/test_service_artifact_status.py::test_fragfm_service_rejects_invalid_intent_cone_as_invalid_argument \
  tests/unit/test_service_artifact_status.py::test_fragfm_service_builds_generator_with_trained_artifacts \
  tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_checkpoint \
  tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_rate_matrix -q
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_writes_smiles_and_report \
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_does_not_leave_smiles_when_report_write_fails \
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_blocked_output_parent_before_generation \
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_same_output_and_report_path -q
```

```bash
python -m py_compile \
  services/fragfm-generator-svc/src/fragfm_generator_svc/main.py \
  models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py \
  tests/unit/test_service_artifact_status.py \
  tests/unit/test_generators.py
```

### Deployment Default Scan

```bash
rg -n "FRAGFM_(VOCAB|CHECKPOINT|RATE_MATRIX)_PATH|checkpoints/fragfm_humu_5k" \
  infra/docker/docker-compose.dev.yml \
  infra/kubernetes/deployments/moleculeforge-services.yaml \
  infra/helm/moleculeforge/values.yaml \
  tests/unit/test_service_artifact_status.py
```

Expected default paths:

```text
checkpoints/fragfm_humu_5k/vocab.json
checkpoints/fragfm_humu_5k/best_model.pt
checkpoints/fragfm_humu_5k/rate_matrix.pt
```

## Stop Conditions

Stop and ask before:

- launching any stronger FragFM, HFM decoder, HCIV, KD, or TAR training run;
- changing benchmark thresholds;
- promoting a local artifact to a permanent production path;
- changing Docker, Kubernetes, Helm, or service defaults again;
- modifying Owner B implementation files;
- changing HUMU pretraining, HUMU encoder architecture, or HFM Lorentz flow
  architecture;
- overwriting or deleting protected artifacts:
  `checkpoints/fragfm`, `checkpoints/humu`, `checkpoints/hfm3d_4h200`.

## Back-Check

- [x] This checklist is documentation-only.
- [x] It preserves Owner A / Owner B boundaries.
- [x] It does not claim W11 or Owner A production acceptance.
- [x] It keeps large-scale training paused.
- [x] It keeps benchmark thresholds unchanged.
- [x] It keeps `checkpoints/fragfm_humu_5k/` classified as strict-local
      engineering evidence only.

## Latest Focused Verification

Recorded after this checklist was created:

- Documentation diff hygiene passed:
  `git diff --check -- moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-code-freeze-owner-b-handoff-checklist.md moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md moleculeforge/docs/todo/owner-a-generation-upstream/README.md moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`
- Placeholder keyword scan across the updated handoff docs produced no output.
- W11 service focused code-freeze shard passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_service_artifact_status.py::test_fragfm_service_rejects_invalid_intent_cone_as_invalid_argument tests/unit/test_service_artifact_status.py::test_fragfm_service_builds_generator_with_trained_artifacts tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_checkpoint tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_rate_matrix -q`
  Result: 4 passed. Warnings were the existing disabled-plugin
  `asyncio_mode` / `pytest.mark.asyncio` warnings.
- W11 sample export focused code-freeze shard passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_writes_smiles_and_report tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_does_not_leave_smiles_when_report_write_fails tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_blocked_output_parent_before_generation tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_same_output_and_report_path -q`
  Result: 4 passed. Warning was the existing disabled-plugin `asyncio_mode`
  warning.
- Compile check passed:
  `python -m py_compile services/fragfm-generator-svc/src/fragfm_generator_svc/main.py models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py tests/unit/test_service_artifact_status.py tests/unit/test_generators.py`
- Deployment default scan confirmed Docker Compose, raw Kubernetes, Helm, and
  the focused deployment regression reference
  `checkpoints/fragfm_humu_5k/{vocab.json,best_model.pt,rate_matrix.pt}`.
- Process scan found no lingering training, pytest, FragFM service, FragFM CLI,
  HFM training, HCIV training, or TAR runner process beyond the scan command
  itself.
