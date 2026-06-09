# Owner A Generation Upstream Workspace

## New Session Entry

For any new conversation or new API session, read this first:

- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`

The start file is the short entry point. The 2026-06-06 handoff is the detailed
current-state package. Older dated files remain evidence, but they should not be
treated as the current plan unless the start file or handoff points to them.

The consolidated cross-gate production execution roadmap is:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-production-execution-roadmap.md`

Use that roadmap to sequence W6, W9, W10, W11, and W13 after reading the
handoff.

The current code-freeze and Owner B coordination checklist is:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-code-freeze-owner-b-handoff-checklist.md`

Use that checklist before claiming Owner A code is ready for Owner B review.

## Role

Owner A owns the generation-upstream side of the CoreArchitecture v2 completion split.

Primary work items:

- W2: pocket / intent HUMU embedding producer.
- W6: TAR ProxylessNAS training runner.
- W8: JMCG engineering skeleton and W8-E engineering acceptance.
- W9: HFM-3D neural geometry decoder path.
- W10: Enc_intent / HCIV production checkpoint pipeline.
- W11: FragFM shared HUMU conditional-space quality path.
- W13: Cross-paradigm KD production distillation path.

Owner B owns W1, W3, W5, and W12. Owner A should not modify Owner B owned implementation areas unless an explicit handoff is recorded.

## Frozen Boundaries

- HUMU pretraining is frozen for the current engineering-completion phase.
- Use the existing local HUMU checkpoint at `moleculeforge/checkpoints/humu/best_model.pt`.
- Do not change HUMU pretraining configuration, loss, encoder architecture, or checkpoint continuation logic in this phase.
- HFM checkpoint `moleculeforge/checkpoints/hfm3d_4h200/best_model.pt` and decoder `moleculeforge/checkpoints/hfm3d_4h200/decoder.json` are smoke/full-flow artifacts, not production-quality generation evidence.
- Do not claim JMCG research completion before W8-R has real joint-training quality evidence.

## Read-Only Context

The following directories may be read or copied from as context, but must not be modified, executed, or used as write targets from this workspace:

- `/workspace/SemMol`
- `/workspace/Projects`

## Shared Files

These files are shared or coordination-sensitive:

- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`
- `moleculeforge/agents/generator_coord/src/generator_coord/agent.py`
- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py`

Before changing any shared file, record the intended function / hunk in `progress.md`.

## Current Dimension Contract

The current HUMU / HFM path uses Lorentz full coordinates:

- `dim=128` means the hyperboloid has 128 spatial dimensions.
- The actual Lorentz vector has `dim + 1 = 129` coordinates, including the time coordinate.
- `LorentzManifold._project()` preserves the last dimension and computes `x_0` from the spatial coordinates.
- HUMU molecule / pocket / route encoders project to `dim + 1`.
- HFM-3D active latent points are currently 129-dimensional.

Therefore, any steering-capable `jmcg_feedback.records[*].humu_embedding` consumed by HFM must currently be 129-dimensional. A 128-dimensional payload will be dropped by HFM feedback validation.

## Gate Workflow

Each Owner A gate should follow:

1. Define scope and allowed files.
2. Check upstream / downstream contracts.
3. Make the smallest useful change.
4. Update this workspace and the relevant architecture / todo docs.
5. Run only allowed static checks unless explicit pytest authorization is given.
6. Add a back-check entry to `progress.md`.

## Current Gate

Current gate after the 2026-06-06 W11 5000-record HUMU-labeled candidate and
deployment-default hardening:

- Hand W1 unit patch-seam failure back to Owner B with the exact failing patch
  target.
- Keep Owner A local engineering gates green through focused validation while avoiding broad slow suites unless needed for final release.
- W11 50-record local HUMU-labeled FragFM smoke is complete in new paths only:
  `data/processing/generator_artifacts/fragfm_records_train_humu_labeled.jsonl`
  and `checkpoints/fragfm_humu_smoke/`. It has HUMU coverage 1.0, but remains
  local engineering smoke, not production-quality evidence.
- W11 5000-record local HUMU-labeled input data is complete:
  `data/processing/generator_artifacts/fragfm_records_humu_labeled.jsonl`
  and its report show 5000/5000 encoded, HUMU coverage 1.0.
- W11 5000-record local HUMU-labeled FragFM candidate is complete in
  `checkpoints/fragfm_humu_5k/`: 1 epoch, batch size 64, hidden dim 8,
  `--rate-optimizer sgd --disable-rate-grad-clip`. The manifest records
  5000 records, 2860 fragments, HUMU coverage 1.0, and the strict
  `mf_generators.fragfm.quality --min-humu-coverage 1.0 --strict` gate passes.
  This is a local engineering candidate, not final production W11 acceptance.
- A stronger CPU training attempt at
  `checkpoints/fragfm_humu_candidate_20260606_155439/` was stopped after about
  46 minutes without checkpoint or manifest. It is aborted-run evidence only and
  must not be used as a candidate.
- Current user direction is to pause large-scale training and complete code
  engineering first.
- The current Owner A code-freeze / Owner B handoff checklist is recorded in
  `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-code-freeze-owner-b-handoff-checklist.md`.
  It records the exact W1 patch-seam handoff, W5 benchmark blocker, contract
  surfaces C1/C2/C3, and focused checks to run before Owner B review.
- W11 FragFM deployment defaults now point Docker Compose, raw Kubernetes, and
  Helm to `checkpoints/fragfm_humu_5k/{vocab.json,best_model.pt,rate_matrix.pt}`
  instead of the old `checkpoints/fragfm` smoke artifact. The focused deployment
  regression verifies the artifact exists and its quality report has coverage
  1.0. This is deployment-default hardening, not cluster acceptance.
- W11 runtime smoke now verifies the service `_build_generator()` path can load
  `checkpoints/fragfm_humu_5k/` and generate one RDKit-parseable molecule.
  Cold-start remains slow locally because PyTorch import, vocab parsing, and the
  large rate matrix dominate startup; cluster readiness still needs evidence.
- W11 FragFM sample export/report tooling now exists at
  `models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py`.
  It writes one generated SMILES per line plus a JSON report with requested,
  generated, valid, and unique counts. The 5k smoke files are
  `data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.smi` and
  `data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.report.json`;
  the report has 8 generated samples, validity 1.0, and uniqueness 1.0. This is
  benchmark preparation only, not official W5/MOSES/GuacaMol/PMO acceptance.
- The existing FragFM MOSES validity benchmark path was smoke-tested with
  `MOSES_REFERENCE_SMILES_PATH=data/benchmarks/moses_reference_smiles.smi` and
  `FRAGFM_MOSES_GENERATED_SMILES_PATH=data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.smi`.
  That proves local wiring for this input path only; it is not official W5
  benchmark acceptance.
- A 64-sample FragFM export is also available at
  `data/processing/generator_artifacts/fragfm_humu_5k_sample_64.smi` with report
  `data/processing/generator_artifacts/fragfm_humu_5k_sample_64.report.json`.
  The report has 64 generated samples, validity 1.0, and uniqueness 1.0. The
  same FragFM MOSES validity path passes on this file with thresholds unchanged.
  This remains local benchmark-input wiring evidence only.
- A 256-sample FragFM export is now available at
  `data/processing/generator_artifacts/fragfm_humu_5k_sample_256.smi` with
  report
  `data/processing/generator_artifacts/fragfm_humu_5k_sample_256.report.json`.
  The report has 256 generated samples, validity 1.0, and uniqueness 1.0. The
  FragFM MOSES validity path passes on this file with thresholds unchanged.
  This is default-batch-size wiring evidence, not production benchmark
  acceptance.
- W11 artifact promotion policy is recorded in
  `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-artifact-promotion-policy.md`.
- W11 production-readiness gap and stop conditions are recorded in
  `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-readiness-gap.md`.
- W11 production training run plan is recorded in
  `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-training-run-plan.md`.
- W11 FragFM training CLI now has runtime observability controls:
  `--log-every`, actual/requested device recording, and manifest `log_every`.
  These are code hardening changes, not production training evidence.
- Consolidated production execution roadmap is recorded in
  `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-production-execution-roadmap.md`.
  Recommended sequencing is W11 near-term hardening first, W6 reward payload
  evidence when real reward data is available, W10 supervised HCIV evidence
  when real target data is available, W9 decoder evidence when real latent/SDF
  source data is available, and W13 teacher artifacts before distillation.
- Prepare production resources/artifacts for other Owner A gates: W6 reward data, W9 decoder artifact, W10 HCIV checkpoint, and W13 teacher embeddings/distillation evidence.
- W6 TAR production-readiness gap is recorded in
  `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-readiness-gap.md`.
- W6 TAR production run plan is recorded in
  `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-run-plan.md`.
- W9 HFM decoder production-readiness gap is recorded in
  `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-readiness-gap.md`.
- W9 HFM decoder production training run plan is recorded in
  `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-training-run-plan.md`.
- W10 HCIV production-readiness gap is recorded in
  `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-readiness-gap.md`.
- W10 HCIV production training run plan is recorded in
  `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-training-run-plan.md`.
- W13 KD production-readiness gap is recorded in
  `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-readiness-gap.md`.
- W13 KD production run plan is recorded in
  `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-run-plan.md`.
- Keep W5 benchmark blocked until official benchmark data and production-quality
  generated samples are available; do not relax thresholds.
