# MoleculeForge Owner A New Session Entry

Date: 2026-06-06
Audience: a new coding agent continuing Owner A work
Workspace root: `/workspace/MForge`
Project root: `/workspace/MForge/moleculeforge`

## Read This First

This file is the short entry point for a new conversation. It replaces older
copy-paste prompts and old preflight recommendations. The detailed handoff is:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`

Read that handoff before making code changes. Older dated files remain useful as
evidence, but they are historical unless the new handoff explicitly promotes
them as current state.

## Copy-Paste Prompt For The New Conversation

Use this as the first user message in the new conversation:

```text
请你接手 MoleculeForge 项目，当前工作区是 /workspace/MForge，项目根目录是 /workspace/MForge/moleculeforge。

你必须先完整阅读并理解这个最新入口文档：
/workspace/MForge/moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md

然后完整阅读这个最新详细交接文档：
/workspace/MForge/moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md

再按 handoff 里的 Essential Read Order 继续阅读相关文档和代码。注意：我们是 Owner A（generation upstream）。不要修改 Owner B 代码，除非我明确授权。/workspace/SemMol 和 /workspace/Projects 只能读或复制，不能写入或执行。HUMU 预训练冻结，不要改 HUMU 预训练配置、loss、encoder 架构或 checkpoint continuation。不要覆盖 checkpoints/fragfm、checkpoints/humu、checkpoints/hfm3d_4h200。不要打印 .env secret。

请先复述你对项目思想、当前阶段、Owner A/Owner B 边界、W11 最新状态、已通过/未通过 gate、当前代码/部署/artifact 状态、下一步推荐计划的理解，然后结合 Superpowers skills，按合理步骤继续执行；不要频繁停止，只有遇到重大决策或 blocker 才问我。
```

## Current Snapshot

- Owner A scope: generation-upstream work items W2, W6, W8, W9, W10, W11, W13.
- Owner B scope: W1, W3, W5, W12 and downstream validation / retrosynthesis /
  supply / SRB / critic / provenance implementation.
- HUMU pretraining is frozen.
- Steering-capable HUMU/HFM embeddings are 129-dimensional Lorentz full
  coordinates. Plain 128-dimensional payloads must not be used for steering.
- W11 now has a strict-local 5000-record HUMU-labeled FragFM candidate at
  `checkpoints/fragfm_humu_5k/`.
- A stronger W11 CPU training attempt at
  `checkpoints/fragfm_humu_candidate_20260606_155439/` was intentionally
  stopped on 2026-06-06 after about 46 minutes without any checkpoint or
  manifest. That directory is aborted-run evidence only, not a candidate.
- Current user direction: do not run large-scale training now. Focus on
  engineering code completeness, observability, and lightweight verification.
- Docker Compose, raw Kubernetes, and Helm FragFM defaults now point to
  `checkpoints/fragfm_humu_5k/{vocab.json,best_model.pt,rate_matrix.pt}`.
- The W11 5k candidate and deployment default hardening are local engineering
  evidence only. They are not production benchmark or cluster acceptance.
- W11 now also has a FragFM sample export/report CLI for benchmark preparation.
  The 5k smoke output is
  `data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.smi` with
  report
  `data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.report.json`.
  That report has 8 generated samples, validity 1.0, and uniqueness 1.0. This
  is benchmark preparation only, not official W5/MOSES/GuacaMol/PMO acceptance.
- The FragFM MOSES validity benchmark path has been smoke-tested locally with
  that 8-sample file and `data/benchmarks/moses_reference_smiles.smi`, but this
  remains wiring evidence only because the sample set is not production-scale or
  official acceptance evidence.
- A stronger 64-sample W11 input smoke is also available at
  `data/processing/generator_artifacts/fragfm_humu_5k_sample_64.smi` with
  report
  `data/processing/generator_artifacts/fragfm_humu_5k_sample_64.report.json`.
  It has 64 generated samples, validity 1.0, uniqueness 1.0, and the existing
  FragFM MOSES validity benchmark path passes on that file. This is still local
  wiring evidence, not official W5 acceptance.
- A 256-sample W11 input smoke now matches the current default MOSES benchmark
  batch size:
  `data/processing/generator_artifacts/fragfm_humu_5k_sample_256.smi` and
  `data/processing/generator_artifacts/fragfm_humu_5k_sample_256.report.json`.
  It has 256 generated samples, validity 1.0, uniqueness 1.0, and the FragFM
  MOSES validity path passes with thresholds unchanged. It remains local wiring
  evidence, not production benchmark acceptance.
- W11 production-readiness gap and stop conditions are recorded in
  `docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-readiness-gap.md`.
- W11 production training run plan is recorded in
  `docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-training-run-plan.md`.
- FragFM training CLI now records runtime controls in its manifest
  (`requested_device`, `actual_device`, `log_every`) and supports
  `--log-every` batch progress logging. This is code observability hardening,
  not production training evidence.
- FragFM quality CLI now supports `--manifest` and reports
  `manifest_consistent`; the 5k local candidate passed a read-only
  manifest-aware quality smoke, but this remains local engineering evidence.
- W9 production-readiness gap and production training run plan are recorded in:
  - `docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-readiness-gap.md`
  - `docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-training-run-plan.md`
- W10 production-readiness gap and production training run plan are recorded in:
  - `docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-readiness-gap.md`
  - `docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-training-run-plan.md`
- W13 production-readiness gap and production run plan are recorded in:
  - `docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-readiness-gap.md`
  - `docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-run-plan.md`
- W6 production-readiness gap and production run plan are recorded in:
  - `docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-readiness-gap.md`
  - `docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-run-plan.md`
- The consolidated Owner A production execution roadmap is recorded in
  `docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-production-execution-roadmap.md`.
  Use it as the cross-gate sequencing guide for W6, W9, W10, W11, and W13.
- The current code-freeze and Owner B coordination checklist is recorded in
  `docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-code-freeze-owner-b-handoff-checklist.md`.
  Use it before claiming Owner A code is ready for Owner B review.

## Essential Read Order

1. `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
2. `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-production-execution-roadmap.md`
3. `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-code-freeze-owner-b-handoff-checklist.md`
4. `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
5. `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`
6. `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
7. `moleculeforge/docs/architecture/corearchitecture-v2-completion-tasksplit.md`
8. `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-readiness-gap.md`
9. `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-training-run-plan.md`
10. `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-readiness-gap.md`
11. `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-run-plan.md`
12. `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-readiness-gap.md`
13. `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-training-run-plan.md`
14. `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-readiness-gap.md`
15. `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-training-run-plan.md`
16. `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-readiness-gap.md`
17. `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-run-plan.md`
18. `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-05-W4-focused-validation-record.md`
19. Root architecture docs only if deeper context is needed:
   - `MoleculeForge_CoreArchitecture_v2.md`
   - `MoleculeForge_CodeArchitecture.md`
   - `MoleculeForge_CoreArchitecture_v2_完成度评估.md`

Supporting evidence only:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-05-owner-a-production-resource-preflight.md`
  contains useful resource observations, but its W11 recommended next step is
  superseded because W11 HUMU labeling, 5k local candidate training, and
  deployment-default hardening are now complete.

## Non-Negotiable Boundaries

- Do not modify Owner B implementation files unless explicitly authorized.
- Do not modify HUMU pretraining config, loss, encoder architecture, or
  checkpoint continuation.
- Do not overwrite protected artifacts:
  - `moleculeforge/checkpoints/fragfm`
  - `moleculeforge/checkpoints/humu`
  - `moleculeforge/checkpoints/hfm3d_4h200`
- `/workspace/SemMol` and `/workspace/Projects` are read/copy-only context.
- `.env` may be loaded, but do not print secrets.
- Use focused validation first. Broad pytest suites are slow and may include
  known production-resource failures.

## Current Recommended Next Work

Read
`docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-production-execution-roadmap.md`
before choosing the next production-resource gate.

Recommended sequence:

1. Keep W11 as the near-term main path because it already has local HUMU-labeled
   data, strict quality, deployment defaults, runtime smoke, sample export, and
   MOSES validity wiring evidence, but do not launch stronger training under
   the current user instruction.
2. Run the code-freeze / Owner B handoff checklist before claiming Owner A code
   is ready for review:
   `docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-code-freeze-owner-b-handoff-checklist.md`.
3. Keep W11 engineering hardening focused on observability, artifact
   validation, sample export, runtime safety, and focused tests.
4. Prepare W6 reward payload evidence when real reward data exists.
5. Prepare W10 supervised HCIV checkpoint evidence when real CIG/HCIV target
   data exists.
6. Prepare W9 HFM decoder evidence when real latent/SDF decoder source data
   exists.
7. Prepare W13 teacher embedding artifacts before any distillation run.

Stop before long training, production artifact naming, deployment default
changes, threshold changes, Owner B edits, protected artifact writes, or
external process changes.

Alternative Owner A gates still needing production resources:

- W6: real reward data and production `TAR_PROXYLESS_SEARCH_COMMAND`.
  Read
  `docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-run-plan.md`
  before preparing reward payloads or changing TAR command deployment.
- W9: production HFM decoder artifact or `HFM_MOLECULAR_DECODER_COMMAND`.
  Read
  `docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-training-run-plan.md`
  before preparing or launching decoder training.
- W10: real supervised CIG/HCIV data and production `HCIV_CHECKPOINT_PATH`.
  Read
  `docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-training-run-plan.md`
  before preparing or launching HCIV training.
- W13: real teacher embeddings and distillation evidence.
  Read
  `docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-run-plan.md`
  before preparing teacher artifacts or launching distillation.

W5 remains blocked on official benchmark data and production-quality generated
samples. Do not relax thresholds to make W5 green.
