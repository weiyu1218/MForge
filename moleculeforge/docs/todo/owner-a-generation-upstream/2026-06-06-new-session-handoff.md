# 2026-06-06 Owner A New Session Handoff

## Purpose

This is the detailed handoff for a new API conversation or new coding agent. It
captures the current project idea, ownership boundaries, code state, artifact
state, verification evidence, known failures, and the recommended next tasks.

Use this document as the source of truth for Owner A current state. Older dated
preflight and implementation-plan files remain evidence, but many of them record
intermediate states that were later superseded.

## Superseded Historical Guidance

These older statements are intentionally preserved in historical logs, but they
must not be treated as current instructions:

- `2026-06-05-owner-a-production-resource-preflight.md` recommended W11 local
  HUMU-labeled FragFM data enrichment. That work is now complete, including the
  50-record smoke, 5000-record labeled dataset, 5k local candidate artifact, and
  deployment-default hardening.
- Historical `progress.md` entries that say `checkpoints/fragfm_humu_5k/` did
  not exist were true during earlier attempts. They are superseded by the later
  "W11 5000-Record HUMU-Labeled FragFM Local Candidate Completed" and
  "W11 FragFM HUMU 5k Deployment Defaults Hardened" entries.
- Historical quality notes about `checkpoints/fragfm` coverage 0.0 still apply
  only to the old protected smoke artifact, not to `checkpoints/fragfm_humu_5k`.

## Operating Instructions For The Next Agent

- Use Superpowers discipline. At minimum, use relevant skills for planning,
  TDD/debugging, executing work, and verification-before-completion.
- Do not stop after every small step. Make a reasonable plan, execute focused
  steps, back-check each step, and continue unless a hard decision or blocker
  appears.
- Prefer focused commands over broad suites. Some broad suites are slow and
  include expected production-resource failures.
- Before claiming success, run fresh verification in the current session.
- Do not revert unrelated dirty worktree changes. The repository has many
  unrelated existing modifications and untracked resources.

## Workspace And Git Reality

- Workspace root: `/workspace/MForge`
- Project root: `/workspace/MForge/moleculeforge`
- The effective git root is `/workspace/MForge`.
- The worktree is dirty with many pre-existing unrelated changes. Do not clean,
  delete, reset, or revert unrelated files.
- Some Owner A docs under `moleculeforge/docs/todo/owner-a-generation-upstream/`
  may appear untracked from the top-level git view. They are still the active
  local handoff documents for this workspace.
- Use path-limited commands when inspecting diffs, for example:

```bash
git diff -- moleculeforge/infra/docker/docker-compose.dev.yml \
  moleculeforge/infra/kubernetes/deployments/moleculeforge-services.yaml \
  moleculeforge/infra/helm/moleculeforge/values.yaml \
  moleculeforge/tests/unit/test_service_artifact_status.py
```

## Project Idea

MoleculeForge implements CoreArchitecture v2 for molecular design. The intended
loop is:

```text
natural language intent
  -> CIG / HCIV intent compilation
  -> generation upstream
  -> validation / oracle cascade
  -> retrosynthesis / supply / SRB / critic
  -> provenance and CRG belief recording
  -> feedback into the next generation round
```

Key terms:

- CIG: compiled intent graph from natural language.
- HCIV / Enc_intent: hyperbolic intent vector used for generation conditioning.
- HUMU: shared hyperbolic multimodal space for molecule, pocket, route, and
  intent encoders.
- HFM-3D: hyperbolic flow matching 3D generator.
- FragFM: fragment flow generator that should condition on the shared HUMU
  space.
- JMCG: joint molecular context generation. Current state is engineering
  skeleton and local context wiring, not research-complete joint training.
- CRG: causal/reasoning graph used by agents to write/read beliefs.
- W gates: architecture completion work items.
- H gates: resource/data/deployment/cluster gates.

## Ownership Boundaries

Owner A owns generation-upstream:

- W2: pocket / intent HUMU embedding producer and HFM feedback consumer.
- W6: TAR ProxylessNAS runner / command target.
- W8: JMCG engineering skeleton and W8-E acceptance.
- W9: HFM-3D neural geometry decoder path.
- W10: Enc_intent / HCIV checkpoint training/export path.
- W11: FragFM shared HUMU conditional-space quality path.
- W13: cross-paradigm KD teacher embedding artifact path.

Owner B owns downstream validation/supply/provenance:

- W1: CRG final-state merge/readback.
- W3: PCBO / mf-eval provider.
- W5: benchmark harness and official benchmark resources.
- W12: CReM-pharm-3D scorer integration.

Shared / coordination-sensitive files:

- `services/orchestrator-svc/src/orchestrator_svc/main.py`
- `agents/generator_coord/src/generator_coord/agent.py`
- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py`

Before changing any shared file, record the intended function/hunk in
`docs/todo/owner-a-generation-upstream/progress.md`.

## Hard Boundaries

- HUMU pretraining is frozen for this engineering-completion phase.
- Do not change HUMU pretraining config, loss, encoder architecture, or
  checkpoint continuation.
- Use the local HUMU checkpoint only as an embedding source:
  `checkpoints/humu/best_model.pt`.
- Do not overwrite protected artifacts:
  - `checkpoints/fragfm`
  - `checkpoints/humu`
  - `checkpoints/hfm3d_4h200`
- `/workspace/SemMol` and `/workspace/Projects` are read/copy-only context.
  Do not write there and do not execute from there.
- `.env` may be loaded, but do not print secret values.
- Do not relax GuacaMol/PMO/MOSES thresholds to make benchmark tests green.

## HUMU / Lorentz Dimension Contract

This contract is critical:

- HUMU encoder constructor uses `dim=128`.
- Lorentz full-coordinate vectors have `dim + 1 = 129` coordinates.
- HFM active latent and HFM steering-capable HUMU feedback currently require
  129-dimensional Lorentz full-coordinate vectors.
- 128-dimensional payloads must not be silently used for steering.
- Invalid 129-dimensional vectors that do not satisfy finite + Lorentz
  hyperboloid validation are dropped or rejected depending on the code path.
- Current validation utility: `mf_core.geometry.lorentz.normalize_lorentz_embedding()`.

## Completed Owner A Engineering Work

### W2 / W8 Feedback And JMCG Wiring

- Orchestrator can produce `moleculeforge.jmcg.feedback.v1` records for intent,
  pocket, and property feedback.
- Intent feedback becomes steering-capable only when it already has a valid
  129-dimensional Lorentz full-coordinate axis.
- Metadata-only pocket/property feedback remains non-steering.
- Optional pocket HUMU enrichment uses `HUMU_ENCODER_TARGET` when structured
  pocket geometry exists.
- HFM validates feedback embeddings before steering.
- `JMCGEngineeringSampler` emits JSON-serializable
  `moleculeforge.jmcg.joint_sample.v1` engineering skeleton records.
- W8-E is engineering-complete locally; W8-R research-quality joint training is
  not complete.

### W6 TAR

- `generator_router_svc.tar_proxyless_runner` exists as a local command target.
- It reads reward/cost JSON from stdin, reuses `ProxylessSearchScheduler`, and
  writes service-compatible JSON.
- W6 production-readiness gap is recorded in
  `docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-readiness-gap.md`.
- W6 production run plan is recorded in
  `docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-run-plan.md`.
- Remaining W6 gate: real reward data, production
  `TAR_PROXYLESS_SEARCH_COMMAND`, and cluster validation.

### W9 HFM Decoder

- Neural geometry decoder training/export/runner path exists:
  - `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/neural_geometry_decoder.py`
  - `models/mf-generators/hfm_3d/train_geometry_decoder.py`
- Decoder source artifact latent vectors are Lorentz-validated.
- Current `checkpoints/hfm3d_4h200/decoder.json` is smoke-only. It has one
  ethanol entry and references a pytest temp HUMU checkpoint path.
- W9 production-readiness gap is recorded in
  `docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-readiness-gap.md`.
- W9 production training run plan is recorded in
  `docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-training-run-plan.md`.
  It defines real source artifact requirements, a non-protected candidate output
  path, training command template, runner smoke, HFM generator smoke, benchmark
  caveats, and stop conditions.
- Remaining W9 gate: real decoder data/artifact, production command/env,
  geometry benchmark, and cluster validation.

### W10 HCIV

- Supervised CIG + target HCIV training/export path exists:
  - `services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_training.py`
  - `services/cig-compiler-svc/train_hciv_encoder.py`
- `target_hciv` is Lorentz-validated.
- Remaining W10 gate: real supervised CIG/HCIV data, production-quality
  checkpoint, `HCIV_CHECKPOINT_PATH`, downstream validation, and cluster
  validation.
- W10 production-readiness gap is recorded in
  `docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-readiness-gap.md`.
- W10 production training run plan is recorded in
  `docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-training-run-plan.md`.

### W11 FragFM

Implemented local W11 path:

- `models/mf-generators/fragfm/src/mf_generators/fragfm/humu_labeling.py`
  derives 129-dimensional HUMU molecule embeddings from the frozen HUMU
  checkpoint.
- `models/mf-generators/fragfm/train.py` preserves valid 129-dimensional
  `humu_embedding` values in `vocab.json`.
- FragFM training manifest records HUMU embedding count/coverage.
- `models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py` reports:
  - rule count
  - fragment count
  - HUMU embedding count and coverage
  - invalid HUMU embedding count
  - checkpoint loadability
  - rate matrix loadability
  - training manifest consistency when `--manifest` is supplied
  - strict schema status
- Quality hardening requires checkpoint key `fragment_encoder.weight`, rate
  matrix keys `base_rate` and `sa_score_embedding.weight`, and optional
  training manifest consistency with the inspected artifacts.
- Rate loss path was optimized to avoid materializing `[batch, vocab, vocab]`
  full rate tensors.
- SA row-gather optimization avoids calling full
  `sa_score_embedding.forward()` for sparse transition loss.
- Training CLI now supports:
  - `--rate-optimizer {adamw,sgd}`
  - `--disable-rate-grad-clip`
  - `--log-every`
- Training CLI now logs loaded record/fragment counts, actual training device,
  batch progress, and epoch runtime.
- Training manifests now include `requested_device`, `actual_device`, and
  `log_every` for runtime provenance.
- Runtime-control logic is covered by lightweight helper tests so it can be
  verified without launching training subprocesses.
- Defaults remain AdamW + rate gradient clipping.
- `mf_generators.fragfm.sample_export` exports generated SMILES and a JSON
  validity/uniqueness report for benchmark preparation. This is intentionally
  separate from benchmark assertions so W5 thresholds remain unchanged.

W11 artifacts:

- Old protected runtime smoke:
  - `checkpoints/fragfm`
  - 50 records, 1 epoch, HUMU coverage 0.0
  - keep it, but do not treat it as HUMU-conditioned production evidence
- 50-record HUMU-labeled smoke:
  - `data/processing/generator_artifacts/fragfm_records_train_humu_labeled.jsonl`
  - `checkpoints/fragfm_humu_smoke/`
  - HUMU coverage 1.0
  - local smoke only
- 5000-record HUMU-labeled dataset:
  - `data/processing/generator_artifacts/fragfm_records_humu_labeled.jsonl`
  - paired report status pass, 5000/5000 encoded, coverage 1.0
- 5000-record local candidate:
  - `checkpoints/fragfm_humu_5k/`
  - files: `vocab.json`, `best_model.pt`, `rate_matrix.pt`,
    `final_model.pt`, `final_rate_matrix.pt`, `training_manifest.json`,
    `quality_report.json`
  - training config: 1 epoch, batch size 64, hidden dim 8, CPU,
    `--rate-optimizer sgd --disable-rate-grad-clip`
  - manifest: 5000 records, 2860 fragments, HUMU coverage 1.0,
    `rate_optimizer=sgd`, `rate_grad_clip=false`
  - strict quality report: `status=pass`, 5000 rules, 2860 fragments,
    checkpoint/rate-matrix loadable, messages empty
  - read-only manifest-aware quality smoke:
    `/tmp/fragfm_humu_5k_manifest_quality_report.json` reports
    `manifest_consistent=true`, `status=pass`, and `messages=[]`
- Aborted stronger training attempt:
  - `checkpoints/fragfm_humu_candidate_20260606_155439/`
  - CPU run, epochs 5, hidden dim 64, batch size 64, AdamW, rate grad clipping
    enabled
  - stopped after about 46 minutes without checkpoint or manifest
  - contains `training_command.txt`, empty `training.log`, `vocab.json`, and
    `aborted_run_record.md`
  - this is aborted-run evidence only and must not be used for quality,
    benchmark, deployment, or promotion decisions
- 5k sample export smoke:
  - `data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.smi`
  - `data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.report.json`
  - report: 8 requested/generated samples, 8 valid SMILES, validity 1.0,
    8 unique SMILES, uniqueness 1.0
- 5k sample export 64-sample input smoke:
  - `data/processing/generator_artifacts/fragfm_humu_5k_sample_64.smi`
  - `data/processing/generator_artifacts/fragfm_humu_5k_sample_64.report.json`
  - report: 64 requested/generated samples, 64 valid SMILES, validity 1.0,
    64 unique SMILES, uniqueness 1.0
- 5k sample export 256-sample input smoke:
  - `data/processing/generator_artifacts/fragfm_humu_5k_sample_256.smi`
  - `data/processing/generator_artifacts/fragfm_humu_5k_sample_256.report.json`
  - report: 256 requested/generated samples, 256 valid SMILES, validity 1.0,
    256 unique SMILES, uniqueness 1.0

W11 deployment default hardening:

- Docker Compose default env values now point to:
  - `checkpoints/fragfm_humu_5k/vocab.json`
  - `checkpoints/fragfm_humu_5k/best_model.pt`
  - `checkpoints/fragfm_humu_5k/rate_matrix.pt`
- Raw Kubernetes `fragfm-generator-config` points to the same paths.
- Helm values `fragfm-generator-config` points to the same paths.
- Env overrides remain supported.
- Focused deployment test verifies the paths and the quality report coverage.
- Runtime smoke verifies the service `_build_generator()` path can load
  `checkpoints/fragfm_humu_5k/` and generate one RDKit-parseable molecule. Local
  cold-start remains slow because PyTorch import, vocab parsing, and the large
  rate matrix dominate startup, so cluster readiness evidence is still needed.
- Artifact promotion policy is recorded in
  `docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-artifact-promotion-policy.md`.
- Production-readiness gap and stop conditions are recorded in
  `docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-readiness-gap.md`.
- Production training run plan is recorded in
  `docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-training-run-plan.md`.
- The consolidated W6/W9/W10/W11/W13 production execution sequence is recorded
  in
  `docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-production-execution-roadmap.md`.

Important W11 caveat:

- `checkpoints/fragfm_humu_5k/` is a strict-local engineering candidate. It is
  not final production W11 acceptance because it lacks benchmark evidence,
  production training configuration, formal threshold approval, and cluster
  validation.
- The sample export smoke is benchmark preparation evidence only. It does not
  satisfy official W5/MOSES/GuacaMol/PMO acceptance and does not change any
  benchmark threshold.
- The current user direction is to stop training work for now and focus on code
  completion. Do not launch stronger W11 training until explicitly re-approved.

### W13 KD

- `mf_core.routing.kd_artifacts` can export/report canonical
  `cross_paradigm_teacher_embeddings.v1` artifacts from JSON/JSONL teacher
  records.
- W13 production-readiness gap is recorded in
  `docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-readiness-gap.md`.
- W13 production run plan is recorded in
  `docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-run-plan.md`.
- Remaining W13 gate: real teacher source/embeddings, real distillation,
  benchmark evidence, and cluster validation.

## Most Recent Verification Evidence

W11 deployment focused test:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/unit/test_service_artifact_status.py::test_fragfm_deployment_wires_artifact_and_sampler_env -q
```

Result: exit code 0, 1 passed. Warnings are existing disabled-plugin
`asyncio_mode` and unknown `pytest.mark.asyncio` warnings.

W11 runtime smoke:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/unit/test_service_artifact_status.py::test_fragfm_deployment_default_artifact_loads_and_generates -q
```

Result: exit code 0, 1 passed. This is slow on the current workstation because
it imports PyTorch and loads the 5k FragFM vocab/checkpoint/rate-matrix
artifacts.

W11 5k strict quality gate:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python -m mf_generators.fragfm.quality \
    --vocab checkpoints/fragfm_humu_5k/vocab.json \
    --checkpoint checkpoints/fragfm_humu_5k/best_model.pt \
    --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt \
    --manifest checkpoints/fragfm_humu_5k/training_manifest.json \
    --min-humu-coverage 1.0 \
    --strict \
    --output checkpoints/fragfm_humu_5k/quality_report.json
```

W11 sample export smoke:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python -m mf_generators.fragfm.sample_export \
    --vocab checkpoints/fragfm_humu_5k/vocab.json \
    --checkpoint checkpoints/fragfm_humu_5k/best_model.pt \
    --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt \
    --output data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.smi \
    --report data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.report.json \
    --samples 8 \
    --device cpu
```

Result: exit code 0. The report records 8 generated samples, 8 valid SMILES,
validity 1.0, 8 unique SMILES, and uniqueness 1.0.

W11 FragFM MOSES validity wiring smoke:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:libs/mf-eval/src:models/mf-generators/fragfm/src" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
MOSES_REFERENCE_SMILES_PATH=data/benchmarks/moses_reference_smiles.smi \
FRAGFM_MOSES_GENERATED_SMILES_PATH=data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.smi \
  .venv/bin/python -m pytest \
    tests/benchmark/moses_benchmark.py::TestMosesBenchmark::test_fragfm_moses_validity -q
```

Result: exit code 0, 1 passed. Warning is the existing disabled-plugin
`asyncio_mode` warning. This validates local benchmark input wiring for the
8-sample FragFM export only; it is not official W5/MOSES acceptance.

W11 64-sample FragFM export:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python -m mf_generators.fragfm.sample_export \
    --vocab checkpoints/fragfm_humu_5k/vocab.json \
    --checkpoint checkpoints/fragfm_humu_5k/best_model.pt \
    --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt \
    --output data/processing/generator_artifacts/fragfm_humu_5k_sample_64.smi \
    --report data/processing/generator_artifacts/fragfm_humu_5k_sample_64.report.json \
    --samples 64 \
    --device cpu
```

Result: exit code 0. The report records 64 generated samples, 64 valid SMILES,
validity 1.0, 64 unique SMILES, and uniqueness 1.0.

W11 64-sample FragFM MOSES validity wiring smoke:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:libs/mf-eval/src:models/mf-generators/fragfm/src" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
MOSES_REFERENCE_SMILES_PATH=data/benchmarks/moses_reference_smiles.smi \
FRAGFM_MOSES_GENERATED_SMILES_PATH=data/processing/generator_artifacts/fragfm_humu_5k_sample_64.smi \
  .venv/bin/python -m pytest \
    tests/benchmark/moses_benchmark.py::TestMosesBenchmark::test_fragfm_moses_validity -q
```

Result: exit code 0, 1 passed. Warning is the existing disabled-plugin
`asyncio_mode` warning. This is stronger than the 8-sample smoke but still not
official W5/MOSES acceptance because the artifact and sample set are local
engineering evidence.

W11 256-sample FragFM export:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python -m mf_generators.fragfm.sample_export \
    --vocab checkpoints/fragfm_humu_5k/vocab.json \
    --checkpoint checkpoints/fragfm_humu_5k/best_model.pt \
    --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt \
    --output data/processing/generator_artifacts/fragfm_humu_5k_sample_256.smi \
    --report data/processing/generator_artifacts/fragfm_humu_5k_sample_256.report.json \
    --samples 256 \
    --device cpu
```

Result: exit code 0. The report records 256 generated samples, 256 valid
SMILES, validity 1.0, 256 unique SMILES, and uniqueness 1.0.

W11 256-sample FragFM MOSES validity wiring smoke:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:libs/mf-eval/src:models/mf-generators/fragfm/src" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
MOSES_REFERENCE_SMILES_PATH=data/benchmarks/moses_reference_smiles.smi \
FRAGFM_MOSES_GENERATED_SMILES_PATH=data/processing/generator_artifacts/fragfm_humu_5k_sample_256.smi \
  .venv/bin/python -m pytest \
    tests/benchmark/moses_benchmark.py::TestMosesBenchmark::test_fragfm_moses_validity -q
```

Result: exit code 0, 1 passed. Warning is the existing disabled-plugin
`asyncio_mode` warning. This matches the current default benchmark batch size,
but it is still local wiring evidence and not official W5/MOSES acceptance.

W11 training CLI runtime-control helper tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/unit/test_generators.py::TestFragFMGenerator::test_training_batch_log_policy_logs_first_interval_and_final_batches \
  tests/unit/test_generators.py::TestFragFMGenerator::test_training_manifest_records_runtime_controls_without_launching_training -q
```

Result: exit code 0, 2 passed. Warning is the existing disabled-plugin
`asyncio_mode` warning. These tests validate batch log policy, CPU fallback, and
manifest runtime fields without launching training.

FragFM training CLI help / compile checks:

```bash
.venv/bin/python models/mf-generators/fragfm/train.py --help | \
  rg -n -- '--log-every|--rate-optimizer|--kd-teacher-embeddings|--humu-embedding-dim'
python -m py_compile \
  moleculeforge/models/mf-generators/fragfm/train.py \
  moleculeforge/tests/unit/test_generators.py
git diff --check -- \
  moleculeforge/models/mf-generators/fragfm/train.py \
  moleculeforge/tests/unit/test_generators.py
```

Result: all exited 0. The help output includes `--log-every`,
`--rate-optimizer`, `--kd-teacher-embeddings`, and `--humu-embedding-dim`.

Known earlier focused validation from
`2026-06-05-W4-focused-validation-record.md`:

- W2 orchestrator feedback producer: 8 passed.
- W2/W8 HFM JMCG consumer: 12 passed.
- C1 generator coordinator: 20 passed.
- C2 validation/srb downstream regression: 32 passed.
- W3 mf-eval: 24 passed.
- W11 quality focused tests: 6 passed.
- W13 pytest: 18 passed.
- W9/W10/W11 hardening focused regression: 4 passed.

Known non-green / blocked items:

- W1 unit gate had 3 failures in `tests/unit/test_graph_repo.py`; this is Owner
  B scope unless explicitly authorized.
- W5 benchmark has expected production-data/quality failures and skips. Do not
  relax thresholds.
- Broad unit suites may be slow and include unrelated known failures. Prefer
  focused gates first.

## Current Deployment / Config State For FragFM

Relevant files:

- `infra/docker/docker-compose.dev.yml`
- `infra/kubernetes/deployments/moleculeforge-services.yaml`
- `infra/helm/moleculeforge/values.yaml`
- `tests/unit/test_service_artifact_status.py`

Current expected default paths:

```text
FRAGFM_VOCAB_PATH=checkpoints/fragfm_humu_5k/vocab.json
FRAGFM_CHECKPOINT_PATH=checkpoints/fragfm_humu_5k/best_model.pt
FRAGFM_RATE_MATRIX_PATH=checkpoints/fragfm_humu_5k/rate_matrix.pt
FRAGFM_HUMU_CURVATURE=1.0
```

Use this scan to verify:

```bash
rg -n "FRAGFM_(VOCAB|CHECKPOINT|RATE_MATRIX)_PATH|checkpoints/fragfm_humu_5k" \
  infra/docker/docker-compose.dev.yml \
  infra/kubernetes/deployments/moleculeforge-services.yaml \
  infra/helm/moleculeforge/values.yaml \
  tests/unit/test_service_artifact_status.py
```

## Current Recommended Next Plan

Default recommendation: read the consolidated production execution roadmap, then
continue W11 toward production readiness unless the user prioritizes another
Owner A production-resource gate.

Primary sequencing reference:

- `docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-production-execution-roadmap.md`
- `docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-code-freeze-owner-b-handoff-checklist.md`

The roadmap keeps W11 as the near-term main path because it already has local
HUMU-labeled data, a strict candidate, deployment-default hardening, runtime
smoke, sample export, and MOSES validity wiring evidence. It also records the
resource-gated next actions for W6 reward payloads, W10 supervised HCIV data,
W9 decoder source data, and W13 teacher embedding artifacts.

The code-freeze checklist is the current bridge from Owner A local engineering
work to Owner B review. It records the W1 patch-seam handoff, W5 benchmark
blocker, C1/C2/C3 contract surfaces, and focused W11 checks. Use it before
claiming Owner A code is ready for Owner B review.

Step 1: use the artifact promotion policy for any further default change.

- Local engineering candidates become production artifacts only through the
  recorded promotion gates.
- Do not overwrite `checkpoints/fragfm`.
- Prefer a new explicit production path when available, for example
  `checkpoints/fragfm_humu_production/` or a dated artifact directory.
- Read
  `docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-artifact-promotion-policy.md`
  before changing deployment defaults again.
- Read
  `docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-readiness-gap.md`
  before launching longer training, choosing production artifact names, or
  claiming W11 production readiness.
- Read
  `docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-training-run-plan.md`
  before launching any stronger FragFM training run.

Step 2: complete the Owner A code-freeze / Owner B handoff checklist.

- Current user instruction is to finish code engineering first, coordinate with
  Owner B second, and run real tests/training later.
- Use
  `docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-code-freeze-owner-b-handoff-checklist.md`.
- Do not modify Owner B code while preparing the handoff unless explicitly
  authorized.

Step 3: harden W11 code before any further training.

- Current user instruction is no large-scale training now.
- Prefer engineering work that improves observability, artifact validation,
  sample export, runtime safety, and focused tests.
- Keep verification lightweight unless a broader gate is specifically needed.
- Do not treat the aborted
  `checkpoints/fragfm_humu_candidate_20260606_155439/` directory as a candidate.

Step 4: strengthen W11 training evidence later, only after explicit approval.

- Evaluate whether the current 1 epoch, hidden dim 8 candidate is enough for
  local engineering only.
- If training again, write to a new artifact directory.
- Preserve strict HUMU coverage and quality CLI checks.
- Keep `--rate-optimizer` controls explicit in manifests.

Step 5: add benchmark evidence without relaxing thresholds.

- W5 benchmark remains blocked until official benchmark data and
  production-quality generated samples are available.
- Do not make thresholds easier.
- Use `mf_generators.fragfm.sample_export` to prepare production-scale generated
  SMILES files for downstream benchmark runs.
- The local 8-sample FragFM MOSES validity smoke can be repeated with
  `MOSES_REFERENCE_SMILES_PATH=data/benchmarks/moses_reference_smiles.smi` and
  `FRAGFM_MOSES_GENERATED_SMILES_PATH=data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.smi`,
  but do not call it official W5 acceptance.
- The 64-sample local FragFM MOSES validity smoke can be repeated with
  `FRAGFM_MOSES_GENERATED_SMILES_PATH=data/processing/generator_artifacts/fragfm_humu_5k_sample_64.smi`,
  but do not call it official W5 acceptance.
- The 256-sample local FragFM MOSES validity smoke can be repeated with
  `FRAGFM_MOSES_GENERATED_SMILES_PATH=data/processing/generator_artifacts/fragfm_humu_5k_sample_256.smi`.
  This matches the current default `MOSES_BENCHMARK_BATCH_SIZE`, but do not call
  it official W5 acceptance.
- If adding W11-specific benchmark wiring, keep it separate from threshold
  changes and from official W5 acceptance claims.

Step 6: cluster validation.

- Validate the FragFM service with the 5k or production artifact in a real
  cluster only when infrastructure is available.
- Record exact commands, pod/config state, and service smoke evidence.

Alternative next gates if the user reprioritizes:

- W6: configure real reward data and `TAR_PROXYLESS_SEARCH_COMMAND`. Read
  `docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-readiness-gap.md`
  and
  `docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-run-plan.md`
  before preparing reward payloads or changing TAR command deployment.
- W9: produce/deploy a production HFM decoder artifact or command. Read
  `docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-readiness-gap.md`
  and
  `docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-training-run-plan.md`
  before preparing source data, launching decoder training, changing W9
  artifacts, or changing deployment.
- W10: train/deploy a production HCIV checkpoint.
  Read
  `docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-readiness-gap.md`
  and
  `docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-training-run-plan.md`
  before preparing source data, launching HCIV training, or changing
  `HCIV_CHECKPOINT_PATH`.
- W13: provide real teacher embeddings and distillation evidence.
  Read
  `docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-readiness-gap.md`
  and
  `docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-run-plan.md`
  before preparing teacher artifacts, changing teacher deployment, or launching
  distillation.

## Useful Commands

Environment setup:

```bash
cd /workspace/MForge/moleculeforge
set -a; source .env; set +a
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"
```

Focused W11 deployment test:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/unit/test_service_artifact_status.py::test_fragfm_deployment_wires_artifact_and_sampler_env -q
```

W11 strict quality check:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python -m mf_generators.fragfm.quality \
    --vocab checkpoints/fragfm_humu_5k/vocab.json \
    --checkpoint checkpoints/fragfm_humu_5k/best_model.pt \
    --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt \
    --manifest checkpoints/fragfm_humu_5k/training_manifest.json \
    --min-humu-coverage 1.0 \
    --strict \
    --output checkpoints/fragfm_humu_5k/quality_report.json
```

W11 5k sample export:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python -m mf_generators.fragfm.sample_export \
    --vocab checkpoints/fragfm_humu_5k/vocab.json \
    --checkpoint checkpoints/fragfm_humu_5k/best_model.pt \
    --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt \
    --output data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.smi \
    --report data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.report.json \
    --samples 8 \
    --device cpu
```

W11 FragFM MOSES validity wiring smoke:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:libs/mf-eval/src:models/mf-generators/fragfm/src" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
MOSES_REFERENCE_SMILES_PATH=data/benchmarks/moses_reference_smiles.smi \
FRAGFM_MOSES_GENERATED_SMILES_PATH=data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.smi \
  .venv/bin/python -m pytest \
    tests/benchmark/moses_benchmark.py::TestMosesBenchmark::test_fragfm_moses_validity -q
```

W11 64-sample export:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python -m mf_generators.fragfm.sample_export \
    --vocab checkpoints/fragfm_humu_5k/vocab.json \
    --checkpoint checkpoints/fragfm_humu_5k/best_model.pt \
    --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt \
    --output data/processing/generator_artifacts/fragfm_humu_5k_sample_64.smi \
    --report data/processing/generator_artifacts/fragfm_humu_5k_sample_64.report.json \
    --samples 64 \
    --device cpu
```

W11 64-sample FragFM MOSES validity wiring smoke:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:libs/mf-eval/src:models/mf-generators/fragfm/src" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
MOSES_REFERENCE_SMILES_PATH=data/benchmarks/moses_reference_smiles.smi \
FRAGFM_MOSES_GENERATED_SMILES_PATH=data/processing/generator_artifacts/fragfm_humu_5k_sample_64.smi \
  .venv/bin/python -m pytest \
    tests/benchmark/moses_benchmark.py::TestMosesBenchmark::test_fragfm_moses_validity -q
```

W11 256-sample export:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python -m mf_generators.fragfm.sample_export \
    --vocab checkpoints/fragfm_humu_5k/vocab.json \
    --checkpoint checkpoints/fragfm_humu_5k/best_model.pt \
    --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt \
    --output data/processing/generator_artifacts/fragfm_humu_5k_sample_256.smi \
    --report data/processing/generator_artifacts/fragfm_humu_5k_sample_256.report.json \
    --samples 256 \
    --device cpu
```

W11 256-sample FragFM MOSES validity wiring smoke:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:libs/mf-eval/src:models/mf-generators/fragfm/src" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
MOSES_REFERENCE_SMILES_PATH=data/benchmarks/moses_reference_smiles.smi \
FRAGFM_MOSES_GENERATED_SMILES_PATH=data/processing/generator_artifacts/fragfm_humu_5k_sample_256.smi \
  .venv/bin/python -m pytest \
    tests/benchmark/moses_benchmark.py::TestMosesBenchmark::test_fragfm_moses_validity -q
```

W9 source artifact preflight template:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/hfm_3d/src" \
  .venv/bin/python - <<'PY'
from collections import Counter
from pathlib import Path

from mf_generators.hfm_3d.decoder.neural_geometry_decoder import (
    load_geometry_training_examples,
)

source = Path(
    "data/processing/generator_artifacts/"
    "hfm_decoder_source_YYYYMMDD_<run_id>.json"
)
examples = load_geometry_training_examples(source)
unique_smiles = {example.smiles for example in examples}
atom_counts = [len(example.atom_types) for example in examples]
elements = Counter(atom for example in examples for atom in example.atom_types)

assert len(examples) >= 1000, len(examples)
assert len(unique_smiles) >= 1000, len(unique_smiles)
assert max(atom_counts) <= 64, max(atom_counts)
assert all(example.latent.numel() == 129 for example in examples)

print("entries", len(examples))
print("unique_smiles", len(unique_smiles))
print("atom_count_max", max(atom_counts))
print("elements", dict(sorted(elements.items())))
PY
```

W9 decoder training command template. Do not launch without user/resource
approval:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/hfm_3d/src" \
  .venv/bin/python models/mf-generators/hfm_3d/train_geometry_decoder.py \
    --decoder-artifact data/processing/generator_artifacts/hfm_decoder_source_YYYYMMDD_<run_id>.json \
    --output-artifact checkpoints/hfm_geometry_decoder_candidate_YYYYMMDD_<run_id>/neural_geometry_decoder.pt \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --max-atoms 64 \
    --device cpu
```

W9 neural decoder runner smoke template:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/hfm_3d/src" \
  .venv/bin/python -m mf_generators.hfm_3d.decoder.neural_geometry_decoder \
    --artifact checkpoints/hfm_geometry_decoder_candidate_YYYYMMDD_<run_id>/neural_geometry_decoder.pt \
    < /tmp/hfm_decoder_request_YYYYMMDD_<run_id>.json \
    > checkpoints/hfm_geometry_decoder_candidate_YYYYMMDD_<run_id>/runner_smoke.json
```

W10 HCIV source preflight template:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:services/cig-compiler-svc/src" \
  .venv/bin/python - <<'PY'
from collections import Counter
from pathlib import Path

from cig_compiler_svc.domain.hciv_training import load_hciv_training_examples

source = Path(
    "data/processing/generator_artifacts/"
    "hciv_supervised_train_YYYYMMDD_<run_id>.jsonl"
)
examples = load_hciv_training_examples(source, dim=128, curvature=1.0)
objective_types = Counter(
    str(node.type) for example in examples for node in example.cig.objective_nodes
)
assert len(examples) >= 1000, len(examples)
assert objective_types, "missing objective coverage"
assert all(example.target_coordinates.numel() == 129 for example in examples)
print("examples", len(examples))
print("objective_types", dict(sorted(objective_types.items())))
PY
```

W10 HCIV training command template. Do not launch without user/resource
approval:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:services/cig-compiler-svc/src" \
  .venv/bin/python services/cig-compiler-svc/train_hciv_encoder.py \
    --data data/processing/generator_artifacts/hciv_supervised_train_YYYYMMDD_<run_id>.jsonl \
    --output-checkpoint checkpoints/hciv_encoder_candidate_YYYYMMDD_<run_id>/hciv_encoder.pt \
    --manifest checkpoints/hciv_encoder_candidate_YYYYMMDD_<run_id>/hciv_encoder.manifest.json \
    --dim 128 \
    --curvature 1.0 \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --device cpu
```

W13 teacher embedding artifact export template:

```bash
PYTHONPATH="libs/mf-core/src" \
  .venv/bin/python -m mf_core.routing.kd_artifacts \
    --input data/processing/generator_artifacts/kd_teacher_records_YYYYMMDD_<source>.jsonl \
    --output data/processing/generator_artifacts/kd_teacher_embeddings_<consumer>_<dim>_YYYYMMDD_<run_id>.json \
    --embedding-field teacher_embedding \
    --expected-dim <dim> \
    --min-embeddings 1000 \
    --report data/processing/generator_artifacts/kd_teacher_embeddings_<consumer>_<dim>_YYYYMMDD_<run_id>.report.json \
    --strict
```

W6 TAR command smoke template:

```bash
PYTHONPATH="libs/mf-core/src:services/generator-router-svc/src" \
  .venv/bin/python -m generator_router_svc.tar_proxyless_runner \
    < data/processing/generator_artifacts/tar_reward_payload_YYYYMMDD_<run_id>.json \
    > data/processing/generator_artifacts/tar_proxyless_result_YYYYMMDD_<run_id>.json
```

Whitespace / diff check for current W11 deployment docs:

```bash
git diff --check -- \
  moleculeforge/infra/docker/docker-compose.dev.yml \
  moleculeforge/infra/kubernetes/deployments/moleculeforge-services.yaml \
  moleculeforge/infra/helm/moleculeforge/values.yaml \
  moleculeforge/tests/unit/test_service_artifact_status.py
```

Path scan:

```bash
rg -n "checkpoints/fragfm(_humu_5k)?/(vocab|best_model|rate_matrix)" \
  infra/docker/docker-compose.dev.yml \
  infra/kubernetes/deployments/moleculeforge-services.yaml \
  infra/helm/moleculeforge/values.yaml \
  tests/unit/test_service_artifact_status.py
```

## What Not To Do

- Do not restart or modify HUMU pretraining.
- Do not overwrite `checkpoints/fragfm`, `checkpoints/humu`, or
  `checkpoints/hfm3d_4h200`.
- Do not treat `checkpoints/fragfm_humu_5k/` as production acceptance.
- Do not modify Owner B code unless the user explicitly authorizes it.
- Do not relax benchmark thresholds.
- Do not run broad slow suites as the first move.
- Do not print secrets from `.env`.
- Do not edit or execute from `/workspace/SemMol` or `/workspace/Projects`.

## Essential Read Order For New Agent

1. `docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
2. This file.
3. `docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-production-execution-roadmap.md`
4. `docs/todo/owner-a-generation-upstream/README.md`
5. `docs/todo/owner-a-generation-upstream/progress.md`
6. `docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
7. `docs/architecture/corearchitecture-v2-completion-tasksplit.md`
8. `docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-readiness-gap.md`
9. `docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-training-run-plan.md`
10. `docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-readiness-gap.md`
11. `docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-run-plan.md`
12. `docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-readiness-gap.md`
13. `docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-training-run-plan.md`
14. `docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-readiness-gap.md`
15. `docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-training-run-plan.md`
16. `docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-readiness-gap.md`
17. `docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-run-plan.md`
18. `docs/todo/owner-a-generation-upstream/2026-06-05-W4-focused-validation-record.md`
19. Root architecture docs only if deeper context is needed:
    - `MoleculeForge_CoreArchitecture_v2.md`
    - `MoleculeForge_CodeArchitecture.md`
    - `MoleculeForge_CoreArchitecture_v2_完成度评估.md`

Only read older W-specific preflight/implementation-plan files when you need
evidence for that specific W gate. Do not treat older recommendations as the
current plan if they conflict with this handoff.

Supporting evidence only:

- `docs/todo/owner-a-generation-upstream/2026-06-05-owner-a-production-resource-preflight.md`
  is useful for resource observations, but its W11 recommendation is superseded.
