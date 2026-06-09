# W13 Cross-Paradigm KD Production Readiness Gap

Date: 2026-06-06
Scope: Owner A, W13 cross-paradigm teacher embedding and distillation readiness

## Current Position

The cross-paradigm KD artifact handoff path exists and is locally verified. It is
not production distillation evidence yet.

Current verified engineering evidence:

| Gate | Local Evidence | Status |
|---|---|---|
| KD layer | `CrossParadigmKDLayer` consumes normalized teacher distributions and teacher embedding targets | Local pass |
| Teacher adapters | Boltz2 and HypSeek records can become teacher distributions | Local pass |
| Teacher app/runner | Generator router can call `HYPSEEK_TEACHER_COMMAND` or `HYPSEEK_TEACHER_URL` | Local pass |
| Embedding artifact utility | `mf_core.routing.kd_artifacts` exports/reports canonical teacher embedding artifacts | Local pass |
| Generator consumers | HFM-3D, FragFM, UAS, CReM, MMPT, and iCLM paths can consume KD teacher embeddings | Local pass |

Current local artifact utility:

- CLI: `python -m mf_core.routing.kd_artifacts`
- Canonical schema: `cross_paradigm_teacher_embeddings.v1`
- Report schema: `cross_paradigm_teacher_embeddings_report.v1`
- Checks: finite values, consistent dimension, optional expected dimension, and
  minimum embedding count.

## Latest Source Inventory Check

A read-only inventory check on 2026-06-06 scanned
`data/processing/generator_artifacts/` for files matching the production run
plan teacher-record and teacher-embedding naming intent.

Observed result:

- No real KD teacher-record source was found.
- No `kd_teacher_records_YYYYMMDD_<source>.jsonl` production-candidate input was
  found.
- No per-consumer `kd_teacher_embeddings_<consumer>_<dim>_*` artifact or report
  was found.
- No non-protected KD candidate checkpoint directory was found.
- Only W11 FragFM local engineering data and sample-export files were present
  in the matched inventory.

Conclusion:

- W13 remains blocked on approved real teacher records.
- Do not lower teacher-record or embedding-count requirements to advance without
  an approved source.
- The next W13 action is still to obtain or identify the approved teacher
  source, then run strict per-consumer artifact export and report checks from
  the W13 production run plan.

## Non-Promotion Reasons

W13 is not production-ready because these gates are still missing:

| Missing Gate | Required Evidence | Owner / Resource |
|---|---|---|
| Real teacher records | Non-demo teacher records from approved teacher sources with stable provenance | Owner A + data |
| Per-consumer embedding artifacts | Canonical teacher embedding artifacts with dimensions matching the target generator training path | Owner A |
| Distillation runs | Real generator training or update runs with non-zero KD weight and manifest evidence | Owner A + compute/data |
| Quality benchmark | Evidence that KD improves or preserves generation quality without threshold relaxation | Owner A + benchmark resources |
| Teacher deployment | Production `HYPSEEK_TEACHER_COMMAND` or `HYPSEEK_TEACHER_URL` resource, if score-teacher path is used | Owner A + H10 resources |
| Cluster runtime | Real service deployment and request/response evidence | Owner A + H10 resources |

## Dimension Caution

Do not assume one teacher embedding dimension works for every consumer:

- HFM-3D KD teacher embeddings should match the HFM latent full-coordinate
  dimension, currently 129.
- FragFM KD teacher embeddings must match `--hidden-dim`.
- UAS KD teacher embeddings must match the autoencoder latent dimension,
  currently `input_dim // 2` for the chosen training data.
- CReM and MMPT KD metrics use their local structural feature embedding
  dimensions.
- iCLM online KD requires teacher embeddings to match the model-provided student
  embeddings.

Create separate artifacts per consumer or per dimension when needed.

## Next Executable Gates

Proceed in this order unless the user reprioritizes:

1. Define the approved teacher-record source and target consumer list.
2. Export canonical teacher embedding artifacts per target dimension.
3. Run strict preflight reports before any generator training.
4. Launch distillation only after user/resource approval.
5. Record generator manifests, quality metrics, benchmark evidence, and cluster
   evidence before promotion.

Run plan:

- `docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-run-plan.md`

## Stop Conditions

Stop and ask before:

- launching any generator distillation training run;
- changing KD loss semantics or `CrossParadigmKDLayer`;
- choosing permanent teacher artifact names;
- changing production teacher deployment env values;
- editing Owner B benchmark thresholds or implementation files;
- modifying HUMU pretraining or HFM Lorentz flow architecture;
- overwriting protected checkpoints.

## Back-Check

- W13 local artifact handoff path is complete.
- Production W13 still needs real teacher data, per-consumer artifacts,
  distillation runs, benchmark evidence, and cluster validation.
- No production distillation run is authorized by this gap record.
