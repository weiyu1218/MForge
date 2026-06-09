# W13 Cross-Paradigm KD Production Run Plan

Date: 2026-06-06
Scope: Owner A, W13 teacher embedding artifact and distillation preparation

## Goal

Prepare production-candidate teacher embedding artifacts and generator
distillation runs without changing KD loss semantics, benchmark thresholds, or
deployment defaults. This document is a run plan only. It does not authorize
long training jobs or production promotion.

## Current Baseline

Engineering path:

- Artifact utility:
  `libs/mf-core/src/mf_core/routing/kd_artifacts.py`
- CLI:
  `python -m mf_core.routing.kd_artifacts`
- Canonical artifact schema:
  `cross_paradigm_teacher_embeddings.v1`
- Consumers:
  - `models/mf-generators/hfm_3d/train.py`
  - `models/mf-generators/fragfm/train.py`
  - `models/mf-generators/uas/train.py`
  - `models/mf-generators/crem_3d/train.py`
  - `models/mf-generators/mmpt_rag/train.py`
  - `models/mf-generators/incremental_clm/.../online_learner.py`

## Teacher Record Requirements

Use source paths under:

```text
data/processing/generator_artifacts/kd_teacher_records_YYYYMMDD_<source>.jsonl
```

Required JSONL shape for embedding export:

```json
{"id":"teacher-record-1","teacher_embedding":[0.1,0.2],"teacher_source":"approved-source","generator_target":"hfm_3d"}
```

Minimum source gates:

- records come from an approved non-demo teacher source;
- every record has a stable non-empty `id`;
- every record has finite numeric `teacher_embedding`;
- source notes record teacher model/service, data provenance, target generator,
  intended embedding dimension, and generation task domain;
- no benchmark threshold is changed to justify the teacher source.

## Per-Consumer Artifact Outputs

Use separate output files per consumer and dimension:

```text
data/processing/generator_artifacts/kd_teacher_embeddings_<consumer>_<dim>_YYYYMMDD_<run_id>.json
data/processing/generator_artifacts/kd_teacher_embeddings_<consumer>_<dim>_YYYYMMDD_<run_id>.report.json
```

Recommended first target dimensions:

| Consumer | Expected Dimension | Notes |
|---|---:|---|
| HFM-3D | 129 | Current Lorentz full-coordinate latent dimension |
| FragFM | selected `--hidden-dim` | Must equal training hidden dim |
| UAS | selected data `input_dim // 2` | Must equal autoencoder latent dimension |
| CReM | structural feature dimension from `train.py` | Validate with strict report before non-zero KD |
| MMPT | structural feature dimension from `train.py` | Validate with strict report before non-zero KD |
| iCLM | model student embedding dimension | Requires model/update runner evidence |

Export a canonical artifact:

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

Inspect the report before training:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

report = json.loads(
    Path(
        "data/processing/generator_artifacts/"
        "kd_teacher_embeddings_<consumer>_<dim>_YYYYMMDD_<run_id>.report.json"
    ).read_text(encoding="utf-8")
)
assert report["status"] == "pass", report["messages"]
assert report["embedding_count"] >= 1000, report
assert report["embedding_dim"] == <dim>, report
print(report["status"], report["embedding_count"], report["embedding_dim"])
PY
```

If an approved source has fewer than 1000 teacher embeddings, stop and record the
reason before lowering `--min-embeddings`.

## Distillation Run Templates

Do not launch these commands without user/resource approval.

HFM-3D:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-encoders/humu_mol_encoder/src:models/mf-generators/hfm_3d/src" \
  .venv/bin/python models/mf-generators/hfm_3d/train.py \
    --data data/processing/generator_artifacts/hfm_training_data_YYYYMMDD_<run_id> \
    --output-dir checkpoints/hfm3d_kd_candidate_YYYYMMDD_<run_id> \
    --humu-checkpoint checkpoints/humu/best_model.pt \
    --decoder-artifact checkpoints/hfm3d_kd_candidate_YYYYMMDD_<run_id>/decoder.json \
    --kd-teacher-embeddings data/processing/generator_artifacts/kd_teacher_embeddings_hfm_3d_129_YYYYMMDD_<run_id>.json \
    --kd-weight 0.1 \
    --kd-generator-idx 0 \
    --epochs 20 \
    --batch-size 64 \
    --device cpu
```

FragFM:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python models/mf-generators/fragfm/train.py \
    --data data/processing/generator_artifacts/fragfm_records_humu_labeled.jsonl \
    --output-dir checkpoints/fragfm_humu_kd_candidate_YYYYMMDD_<run_id> \
    --epochs 5 \
    --batch-size 64 \
    --hidden-dim 64 \
    --lr 0.0003 \
    --rate-loss-weight 0.1 \
    --kd-teacher-embeddings data/processing/generator_artifacts/kd_teacher_embeddings_fragfm_64_YYYYMMDD_<run_id>.json \
    --kd-weight 0.1 \
    --kd-generator-idx 1 \
    --humu-embedding-dim 129 \
    --humu-curvature 1.0 \
    --device cpu
```

UAS:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/uas/src" \
  .venv/bin/python models/mf-generators/uas/train.py \
    --data data/processing/generator_artifacts/uas_embeddings_YYYYMMDD_<run_id>.jsonl \
    --output-dir checkpoints/uas_kd_candidate_YYYYMMDD_<run_id> \
    --kd-teacher-embeddings data/processing/generator_artifacts/kd_teacher_embeddings_uas_<latent_dim>_YYYYMMDD_<run_id>.json \
    --kd-weight 0.1 \
    --kd-generator-idx 2 \
    --epochs 20 \
    --batch-size 128 \
    --device cpu
```

CReM and MMPT:

- Use the existing `--kd-teacher-embeddings`, `--kd-weight`, and
  `--kd-generator-idx` flags on their training CLIs.
- Run a strict artifact report with the expected structural feature dimension
  before setting `--kd-weight > 0`.
- Write outputs to new non-protected candidate artifact paths under
  `models/artifacts/crem/` or `models/artifacts/mmpt/` only after promotion
  naming is approved; use `data/processing/generator_artifacts/` or
  `checkpoints/<consumer>_kd_candidate_YYYYMMDD_<run_id>/` for preliminary runs.

iCLM:

- Use `ICLM_UPDATE_COMMAND` or an injected online learner only after the target
  model exposes student embeddings with a known shape.
- The update payload must include `kd_teacher_embeddings` and `kd_weight`.
- Verify the response records task and KD metrics before promotion.

## Promotion Decision Data

Record these before any W13 promotion decision:

- teacher source, source hash, and target consumer;
- artifact export command and strict report;
- expected dimension and actual dimension;
- generator training/update command and environment, excluding secrets;
- manifest values showing non-empty `kd_teacher_embeddings` and non-zero
  `kd_weight`;
- baseline-vs-KD quality comparison;
- official benchmark evidence without threshold relaxation;
- cluster readiness evidence when deployment is involved.

## Stop Conditions

Stop and ask before:

- launching any generator training or iCLM update command above;
- lowering `--min-embeddings` or changing expected dimensions;
- choosing permanent teacher artifact names;
- changing `HYPSEEK_TEACHER_COMMAND`, `HYPSEEK_TEACHER_URL`, or deployment
  defaults;
- editing KD loss semantics or `CrossParadigmKDLayer`;
- editing Owner B benchmark implementation or thresholds;
- overwriting protected checkpoints.

## Back-Check

- This plan does not launch distillation training.
- This plan does not modify code.
- This plan keeps per-consumer dimension validation explicit.
- This plan treats W13 as production-resource blocked until real teacher data,
  distillation runs, benchmark evidence, and cluster evidence exist.
