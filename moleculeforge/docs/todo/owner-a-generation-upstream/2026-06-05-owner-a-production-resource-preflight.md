# Owner A Production Resource Preflight

Date: 2026-06-05
Role: Owner A, generation-upstream

## Superseded Notice

This file is historical evidence only. Its "Recommended Next Step" section
recommended W11 local HUMU-labeled FragFM data enrichment, but that work was
completed on 2026-06-06:

- 50-record HUMU-labeled smoke: `checkpoints/fragfm_humu_smoke/`
- 5000-record HUMU-labeled dataset:
  `data/processing/generator_artifacts/fragfm_records_humu_labeled.jsonl`
- 5000-record local FragFM candidate: `checkpoints/fragfm_humu_5k/`
- Docker Compose, raw Kubernetes, and Helm defaults now point to
  `checkpoints/fragfm_humu_5k/{vocab.json,best_model.pt,rate_matrix.pt}`

Use `START_HERE_NEW_SESSION.md` and
`2026-06-06-new-session-handoff.md` for current next steps.

## Purpose

After W4 focused validation, this preflight checks whether the remaining Owner A
production gates have enough local resources to move from engineering skeletons
to data/artifact work.

No code, `.env`, or generated artifact was changed during this preflight.

## Environment Readiness

| Gate | Key env/resource | Current state | Meaning |
|---|---|---|---|
| W6 TAR | `TAR_PROXYLESS_SEARCH_COMMAND` | unset | TAR local runner exists, but production command/reward run is not configured. |
| W9 HFM decoder | `HFM_CHECKPOINT_PATH`, `HFM_DECODER_PATH` | set, files exist | Current files are smoke artifacts, not production geometry evidence. |
| W9 decoder command | `HFM_MOLECULAR_DECODER_COMMAND` | unset | Production command target is not deployed. |
| W10 HCIV | `HCIV_CHECKPOINT_PATH` | unset | No production HCIV checkpoint deployed. |
| HUMU service | `HUMU_CHECKPOINT_PATH` | set, file exists | Frozen HUMU checkpoint is available for embedding generation. |
| HUMU service target | `HUMU_ENCODER_TARGET` | unset | No running remote HUMU encoder target is configured. Local router can still load checkpoint in-process. |
| W11 FragFM | `FRAGFM_VOCAB_PATH`, `FRAGFM_CHECKPOINT_PATH`, `FRAGFM_RATE_MATRIX_PATH` | set, files exist | Current FragFM artifact is loadable but HUMU coverage is 0.0. |
| W11 benchmark data | `FRAGFM_MOSES_GENERATED_SMILES_PATH` | unset | MOSES/benchmark validation is not configured. |
| W13 KD | `CROSS_PARADIGM_TEACHER_RECORDS`, `CROSS_PARADIGM_TEACHER_EMBEDDINGS` | unset | No production teacher embedding source configured. |
| W13 HypSeek | `HYPSEEK_TEACHER_COMMAND`, `HYPSEEK_TEACHER_URL` | unset | No external teacher service configured. |
| W5 benchmark | `MOSES_REFERENCE_SMILES_PATH`, `PMO_SCORE_TABLE_PATH`, `CROSSDOCKED_BENCHMARK_JSONL` | unset | Official benchmark data is missing. |

## Local Artifact Findings

### HUMU

- `checkpoints/humu/best_model.pt` is loadable and contains
  `encoder_mol`, `encoder_pocket`, `encoder_route`, and `encoder_intent`.
- The checkpoint is epoch 50 with `loss=1.207609370008754`.
- `checkpoints/humu/validation_metrics.jsonl` has 10 validation entries; the
  latest epoch 50 row reports `retrieval_top1=0.7916278540250958`,
  `val_loss=1.0582103152037337`, `collapse_ratio=0.0`, and small Lorentz norm
  deviation.
- HUMU pretraining remains frozen. This checkpoint can be used as an embedding
  source, but this preflight does not change HUMU training.

### FragFM

- `data/processing/generator_artifacts/fragfm_records.jsonl` has 5000 records.
- `data/processing/generator_artifacts/fragfm_records_train.jsonl` has 50 records.
- Both files have fields `id`, `product`, `fragments`, and `sa_score_bin`.
- Both files currently have `humu_embedding_count=0`.
- `checkpoints/fragfm/training_manifest.json` shows the current local artifact
  was trained for 1 epoch on 50 records. It is a runtime smoke artifact.

### HFM Decoder

- `checkpoints/hfm3d_4h200/decoder.json` has one entry, ethanol/`CCO`.
- Its `humu_checkpoint` points to a pytest temp path:
  `/tmp/pytest-of-root/pytest-54/test_training_cli_writes_kd_em0/humu.pt`.
- Therefore this decoder is not production evidence. It is a smoke/full-flow
  artifact only.

## Recommended Next Step

The most executable Owner A gate is W11 local HUMU-labeled FragFM data
enrichment:

1. Add a small, tested CLI that reads FragFM JSONL records, loads the frozen
   HUMU molecule encoder from `HUMU_CHECKPOINT_PATH`, encodes each `product`
   SMILES, validates the 129-dimensional Lorentz embedding, and writes a new
   derived JSONL with `humu_embedding`.
2. First run it on `fragfm_records_train.jsonl` for a 50-record smoke artifact.
3. If the smoke passes, run it on `fragfm_records.jsonl` for the 5000-record
   local artifact candidate.
4. Train a separate FragFM artifact directory, for example
   `checkpoints/fragfm_humu_smoke/`, without overwriting the current
   `checkpoints/fragfm` artifact.
5. Run the existing `mf_generators.fragfm.quality --strict` gate with a non-zero
   `--min-humu-coverage`.

This would be a local engineering/acceptance improvement, not final production
quality. Production W11 would still need benchmark evidence, formal thresholds,
and cluster validation.

## Alternatives

| Option | Description | Trade-off |
|---|---|---|
| A, recommended | Generate HUMU-labeled FragFM data from the frozen local HUMU checkpoint | Moves W11 from 0 coverage smoke to a real local HUMU-conditioned artifact path; still not final production quality. |
| B | Wait for externally curated HUMU-labeled FragFM data | Cleaner provenance, but blocks Owner A progress. |
| C | Train FragFM again without HUMU labels | Fast, but does not advance the W11 shared-HUMU requirement. |

## Back-Check

- [x] No code or data artifact was written in this preflight.
- [x] Existing HUMU pretraining remains frozen.
- [x] Current HFM and FragFM checkpoints remain classified as smoke artifacts.
- [x] W11 is the only remaining Owner A gate with enough local input to advance
      immediately.
- [x] W6, W9, W10, W13, and W5 still need external data, production artifacts,
      or configured service targets before production acceptance.
