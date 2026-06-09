# W11 FragFM Production Training Run Plan

Date: 2026-06-06
Scope: Owner A, W11 FragFM production-training preparation

## Goal

Prepare the next FragFM training run so it can produce a stronger candidate than
`checkpoints/fragfm_humu_5k/` without overwriting protected or historical
artifacts. This document is a run plan only. It does not authorize launching a
long training job.

## Current Baseline

Input data:

- `data/processing/generator_artifacts/fragfm_records_humu_labeled.jsonl`
- Size: about 15 MB
- Records: 5000
- HUMU coverage: 1.0
- HUMU dimension: 129 Lorentz full coordinates

Current local candidate:

- `checkpoints/fragfm_humu_5k/`
- Training config: 1 epoch, batch size 64, hidden dim 8, CPU,
  `--rate-optimizer sgd --disable-rate-grad-clip`
- Manifest best loss: 8.700536715833447
- Strict quality report: pass, 5000 rules, 2860 fragments, coverage 1.0; a
  read-only manifest-aware quality smoke reports `manifest_consistent=true`

## Candidate Output Directory

Use a new non-protected directory. Recommended pattern:

```text
checkpoints/fragfm_humu_candidate_YYYYMMDD_<run_id>/
```

Do not write to:

- `checkpoints/fragfm`
- `checkpoints/humu`
- `checkpoints/hfm3d_4h200`
- `checkpoints/fragfm_humu_5k`

## Training Command Template

The next stronger local candidate should increase model capacity and epochs while
keeping HUMU validation strict:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python models/mf-generators/fragfm/train.py \
    --data data/processing/generator_artifacts/fragfm_records_humu_labeled.jsonl \
    --output-dir checkpoints/fragfm_humu_candidate_YYYYMMDD_<run_id> \
    --epochs 5 \
    --batch-size 64 \
    --hidden-dim 64 \
    --lr 0.0003 \
    --rate-loss-weight 0.1 \
    --rate-optimizer adamw \
    --save-every 1 \
    --humu-embedding-dim 129 \
    --humu-curvature 1.0 \
    --device cpu
```

Notes:

- `--hidden-dim` must be divisible by 8.
- `--device cpu` is conservative for the current workstation. Use a GPU only
  after confirming resource availability and that it will not interfere with
  external `/workspace/SemMol` work.
- Keep rate gradient clipping enabled unless a measured run shows the previous
  memory/runtime issue recurs.
- Do not use `--resume checkpoints/fragfm_humu_5k/best_model.pt` with a
  different hidden dimension.

## Required Post-Training Checks

Run strict quality:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python -m mf_generators.fragfm.quality \
    --vocab checkpoints/fragfm_humu_candidate_YYYYMMDD_<run_id>/vocab.json \
    --checkpoint checkpoints/fragfm_humu_candidate_YYYYMMDD_<run_id>/best_model.pt \
    --rate-matrix checkpoints/fragfm_humu_candidate_YYYYMMDD_<run_id>/rate_matrix.pt \
    --manifest checkpoints/fragfm_humu_candidate_YYYYMMDD_<run_id>/training_manifest.json \
    --min-humu-coverage 1.0 \
    --strict \
    --output checkpoints/fragfm_humu_candidate_YYYYMMDD_<run_id>/quality_report.json
```

Run service/runtime smoke by temporarily overriding `FRAGFM_*` env values to the
candidate paths. Do not change Docker/Kubernetes/Helm defaults until promotion
gates are complete.

Export benchmark input after the candidate passes strict quality:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python -m mf_generators.fragfm.sample_export \
    --vocab checkpoints/fragfm_humu_candidate_YYYYMMDD_<run_id>/vocab.json \
    --checkpoint checkpoints/fragfm_humu_candidate_YYYYMMDD_<run_id>/best_model.pt \
    --rate-matrix checkpoints/fragfm_humu_candidate_YYYYMMDD_<run_id>/rate_matrix.pt \
    --output data/processing/generator_artifacts/fragfm_humu_candidate_YYYYMMDD_<run_id>_sample_256.smi \
    --report data/processing/generator_artifacts/fragfm_humu_candidate_YYYYMMDD_<run_id>_sample_256.report.json \
    --samples 256 \
    --device cpu
```

## Promotion Decision Data

Record these before any promotion decision:

- exact command and environment;
- wall-clock training time;
- manifest values;
- strict quality report with `manifest_consistent=true`;
- runtime smoke result;
- 256-sample export report;
- MOSES validity wiring result with thresholds unchanged;
- cluster readiness evidence when available.

## Stop Conditions

Stop and ask before:

- launching the training command above;
- changing `--epochs`, `--hidden-dim`, optimizer, or output path to a long or
  permanent production run;
- changing deployment defaults to the new candidate;
- deleting old candidates;
- editing benchmark thresholds;
- touching protected checkpoints.

## Back-Check

- This plan does not start training.
- This plan does not choose a permanent production artifact path.
- This plan keeps `checkpoints/fragfm_humu_5k/` as the current deployment
  default until promotion gates pass.
