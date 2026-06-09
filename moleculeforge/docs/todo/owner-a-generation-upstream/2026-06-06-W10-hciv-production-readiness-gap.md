# W10 HCIV Production Readiness Gap

Date: 2026-06-06
Scope: Owner A, W10 Enc_intent / HCIV checkpoint production readiness

## Current Position

The supervised HCIV encoder training/export path exists and is locally verified.
It is not production `Enc_intent` evidence yet.

Current verified engineering evidence:

| Gate | Local Evidence | Status |
|---|---|---|
| Data loading | `load_hciv_training_examples()` reads JSON/JSONL records with `cig` and explicit `target_hciv` | Local pass |
| Target validation | `target_hciv` is normalized and validated as `dim + 1` Lorentz full-coordinate data | Local pass |
| Training/export | `train_hciv_encoder_checkpoint()` trains a tiny CPU checkpoint and writes optional manifest | Local pass |
| CLI wrapper | `services/cig-compiler-svc/train_hciv_encoder.py` wraps training/export | Local pass |
| Production loading | `CIGCompiler` in `production_real` learned mode loads `HCIV_CHECKPOINT_PATH` | Local pass |

Current deployment state:

- Docker Compose exposes `HCIV_CHECKPOINT_PATH` as an env override.
- Raw Kubernetes and Helm `cig-compiler-config` keep `hciv-checkpoint-path` as
  an empty production resource slot.
- Production learned mode fails fast when `HCIV_CHECKPOINT_PATH` is missing.
- `hash` and `random` encoders remain local-demo only.

## Latest Source Inventory Check

A read-only inventory check on 2026-06-06 scanned
`data/processing/generator_artifacts/` for files matching the production run
plan supervised HCIV naming intent.

Observed result:

- No real supervised CIG/HCIV source JSONL was found.
- No `hciv_supervised_train_YYYYMMDD_<run_id>.jsonl` production-candidate input
  was found.
- No non-protected `checkpoints/hciv_encoder_candidate_*` directory was found.
- Only W11 FragFM local engineering data and sample-export files were present
  in the matched inventory.

Conclusion:

- W10 remains blocked on approved real supervised CIG plus target-HCIV data.
- Do not use hash/random demo targets as production supervised data.
- The next W10 action is still to obtain or identify the approved supervised
  source, then run the documented source preflight in the W10 production
  training plan.

## Non-Promotion Reasons

W10 is not production-ready because these gates are still missing:

| Missing Gate | Required Evidence | Owner / Resource |
|---|---|---|
| Real supervised data | A non-demo CIG plus target HCIV dataset with stable provenance, Lorentz-valid 129-dimensional targets, and domain coverage | Owner A + data |
| Production checkpoint | A trained HCIV checkpoint from real supervised data with manifest and source hash | Owner A + compute/data |
| Deployment value | Production `HCIV_CHECKPOINT_PATH` set to the candidate or promoted checkpoint | Owner A decision |
| Downstream quality | Evidence that intent-conditioned generation responds correctly to learned HCIV outputs | Owner A + downstream coordination |
| Cluster runtime | Real CIG compiler deployment with production checkpoint and request/response evidence | Owner A + H10 resources |

## Next Executable Gates

Proceed in this order unless the user reprioritizes:

1. Define and validate real supervised CIG/HCIV source data requirements.
2. Use the production training run plan to prepare a new non-protected candidate
   output directory.
3. Train only after user/resource approval.
4. Run checkpoint load, CIG compiler smoke, and downstream intent-conditioned
   generation smoke against the new checkpoint.
5. Record quality and cluster evidence before any production promotion.

Run plan:

- `docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-training-run-plan.md`

## Stop Conditions

Stop and ask before:

- launching a long HCIV training run;
- lowering source-data requirements;
- choosing a permanent production checkpoint path;
- changing Docker/Kubernetes/Helm defaults;
- editing Owner B implementation files;
- changing W2 HFM steering rules or injecting HCIV as `humu_embedding`;
- modifying HUMU pretraining or HFM Lorentz flow architecture;
- overwriting protected checkpoints.

## Back-Check

- W10 engineering path is complete locally.
- Production W10 still needs real supervised data, production checkpoint,
  downstream quality evidence, deployment mode, and cluster validation.
- The training run plan exists, but it has not been executed.
