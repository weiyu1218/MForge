# W11 FragFM Production Readiness Gap

Date: 2026-06-06
Scope: Owner A, W11 FragFM shared HUMU conditional-space production readiness

## Current Position

`checkpoints/fragfm_humu_5k/` is the active strict-local engineering candidate.
It is suitable for local service smoke, deployment-default hardening, sample
export, and benchmark-input wiring evidence. It is not a production artifact.

Current verified local evidence:

| Gate | Local Evidence | Status |
|---|---|---|
| HUMU coverage and manifest consistency | `checkpoints/fragfm_humu_5k/quality_report.json` reports `status=pass`, `rules=5000`, `fragments=2860`, HUMU coverage 1.0, and `invalid_humu_embeddings=0`; a read-only manifest-aware quality smoke also reports `manifest_consistent=true` | Local pass |
| Runtime load | `fragfm_generator_svc._build_generator()` loads the 5k vocab/checkpoint/rate matrix and generates one RDKit-parseable molecule | Local pass |
| Deployment defaults | Docker Compose, raw Kubernetes, and Helm default FragFM paths point to `checkpoints/fragfm_humu_5k/{vocab.json,best_model.pt,rate_matrix.pt}` | Local config pass |
| Sample export | `mf_generators.fragfm.sample_export` writes SMILES plus JSON report | Local pass |
| 8-sample input | `fragfm_humu_5k_sample_smoke.smi`, validity 1.0, uniqueness 1.0 | Local smoke |
| 64-sample input | `fragfm_humu_5k_sample_64.smi`, validity 1.0, uniqueness 1.0 | Local smoke |
| 256-sample input | `fragfm_humu_5k_sample_256.smi`, validity 1.0, uniqueness 1.0 | Local default-batch-size smoke |
| MOSES validity wiring | `tests/benchmark/moses_benchmark.py::TestMosesBenchmark::test_fragfm_moses_validity` passes on the 8/64/256 local files with thresholds unchanged | Wiring pass |
| Training observability | `train.py` supports `--log-every`, records requested/actual device and `log_every` in manifest, and has lightweight helper tests for batch log policy and runtime manifest fields | Code hardening pass |

## Latest Evidence Field Audit

A read-only audit on 2026-06-06 parsed the existing W11 JSON reports and counted
non-empty SMILES lines. It did not rerun generation or rewrite artifacts.

Observed artifact fields:

- `training_manifest.json` uses `records=5000`, `fragments=2860`,
  `humu_embedding_count=5000`, `humu_embedding_coverage=1.0`, `epochs=1`,
  `rate_optimizer=sgd`, and `rate_grad_clip=false`.
- `quality_report.json` uses `rules=5000`, `fragments=2860`,
  `humu_embedding_count=5000`, `humu_embedding_coverage=1.0`,
  `invalid_humu_embeddings=0`, `checkpoint_loadable=true`,
  `rate_matrix_loadable=true`, `messages=[]`, and `status=pass`.
- A read-only manifest-aware quality smoke against the same 5k candidate wrote
  only `/tmp/fragfm_humu_5k_manifest_quality_report.json` and reported
  `status=pass`, `checkpoint_loadable=true`, `rate_matrix_loadable=true`,
  `manifest_consistent=true`, and `messages=[]`.
- `fragfm_humu_5k_sample_smoke.report.json` uses `requested_samples=8`,
  `generated_samples=8`, `valid_smiles=8`, `unique_smiles=8`,
  `validity=1.0`, and `uniqueness=1.0`; the `.smi` file has 8 non-empty
  lines.
- `fragfm_humu_5k_sample_64.report.json` uses `requested_samples=64`,
  `generated_samples=64`, `valid_smiles=64`, `unique_smiles=64`,
  `validity=1.0`, and `uniqueness=1.0`; the `.smi` file has 64 non-empty
  lines.
- `fragfm_humu_5k_sample_256.report.json` uses `requested_samples=256`,
  `generated_samples=256`, `valid_smiles=256`, `unique_smiles=256`,
  `validity=1.0`, and `uniqueness=1.0`; the `.smi` file has 256 non-empty
  lines.

Conclusion:

- The W11 local evidence quantities remain internally consistent.
- The current reports use `records`, `fragments`, `rules`,
  `invalid_humu_embeddings`, `valid_smiles`, `unique_smiles`, and
  `manifest_consistent` field names. Future documentation should prefer those
  exact names when referring to JSON fields.
- This audit is local evidence verification only. It does not change production
  acceptance status.

## Latest Training Attempt Status

After the production run plan was recorded, a stronger W11 CPU training attempt
was started at `checkpoints/fragfm_humu_candidate_20260606_155439/` and then
intentionally stopped on 2026-06-06 after about 46 minutes. It produced
`training_command.txt`, an empty `training.log`, `vocab.json`, and
`aborted_run_record.md`, but no checkpoint, rate matrix, final model, or
training manifest.

This directory is aborted-run evidence only. It is not a candidate artifact and
must not be used for quality, benchmark, deployment, or promotion decisions.

Current user direction is to pause large-scale training and finish engineering
code work first.

## Non-Promotion Reasons

The current 5k candidate cannot be promoted because these gates are still
missing:

| Missing Gate | Required Evidence | Owner / Resource |
|---|---|---|
| Production training | A new explicit artifact directory with production-scale data, justified epochs/hidden dim/optimizer choices, manifest, and strict quality report with manifest consistency | Owner A + compute/data |
| Formal benchmark set | Production-scale generated SMILES, official MOSES reference, GuacaMol/PMO resources, and unchanged thresholds | Owner A + H8 resources |
| Cluster runtime | Real cluster pod/config state, service startup time, readiness/liveness, and at least one generation request/response | Owner A + H10 resources |
| Release naming | Immutable production artifact path such as `checkpoints/fragfm_humu_production/` or dated run directory | Owner A decision |
| Ownership boundary | No Owner B implementation edits unless explicitly authorized | Process gate |

## Next Executable Gates

Proceed in this order unless the user reprioritizes:

1. Continue engineering code hardening and focused validation for W11
   training/export/runtime paths. Do not launch stronger training under the
   current user direction.
2. When stronger training is explicitly re-approved, use a fresh output
   directory. Do not overwrite `checkpoints/fragfm`, `checkpoints/humu`,
   `checkpoints/hfm3d_4h200`, `checkpoints/fragfm_humu_5k`, or the aborted
   `checkpoints/fragfm_humu_candidate_20260606_155439/` directory.
   Current plan:
   `docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-training-run-plan.md`.
3. Define the production sample export target size and artifact path after a
   stronger trained artifact exists.
4. Run formal benchmark commands only with official resources and unchanged
   thresholds.
5. Validate the promoted artifact in a real cluster and record startup/readiness
   evidence.

## Stop Conditions

Stop and ask for user decision before:

- launching a long production training run;
- choosing a permanent production artifact name;
- changing deployment defaults away from `checkpoints/fragfm_humu_5k/`;
- modifying benchmark thresholds;
- editing Owner B implementation files;
- killing or modifying external `/workspace/SemMol` processes.

## Back-Check

- The current local 256-sample smoke improves benchmark-input confidence but does
  not change production acceptance status.
- `checkpoints/fragfm_humu_5k/` remains strict-local engineering evidence.
- W5 official benchmark acceptance remains blocked on official resources and
  production-quality generated samples.
