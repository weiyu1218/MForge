# W11 FragFM Shared HUMU Quality Preflight

Date: 2026-06-04
Owner: A / generation upstream

## Scope

W11 covers FragFM's production path in the shared HUMU conditional space. This
gate is limited to local engineering readiness:

- preserve valid rule-level HUMU embeddings from FragFM training data into the
  vocabulary artifact;
- add a repeatable local quality report/gate for FragFM artifacts;
- keep current HUMU pretraining, HUMU encoder architecture, HFM architecture,
  and checkpoints unchanged.

This gate does not train or certify a production-quality FragFM model. Real
quality remains blocked on production training data, production training runs,
official benchmarks, and cluster validation.

## Existing Evidence

Already implemented:

- `models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py`
  loads vocabulary artifacts, assembly rules, optional checkpoint and
  SA-aware rate matrix artifacts.
- `FragFMGenerator` ranks rules by SA-aware transition score plus
  rule-level HUMU / intent-cone alignment when `humu_embedding` is present.
- `FragFMGenerator` accepts an injected `humu_latent_sampler`.
- `services/fragfm-generator-svc/src/fragfm_generator_svc/main.py` injects
  `SharedHUMULatentSampler`, which samples from an intent cone in the HUMU
  Lorentz manifold.
- `models/mf-generators/fragfm/train.py` writes vocabulary, model checkpoint,
  rate matrix, manifest, and already supports teacher embedding KD loss.
- Deployment wiring already exposes `FRAGFM_VOCAB_PATH`,
  `FRAGFM_CHECKPOINT_PATH`, `FRAGFM_RATE_MATRIX_PATH`, and
  `FRAGFM_HUMU_CURVATURE`.

Local artifact check:

- `checkpoints/fragfm/vocab.json` exists with 50 assembly rules.
- Current local artifact has 0 rules with `humu_embedding`. That proves the
  service can run but does not prove shared HUMU conditional quality.

## Gap

The remaining local code gap is not basic service wiring. The concrete gap is
artifact quality evidence:

1. `train.py` normalizes training records but drops any input
   `humu_embedding` before writing `vocab.json`.
2. There is no local command/report that checks whether a FragFM artifact set
   has HUMU embedding coverage, valid Lorentz full-coordinate embeddings,
   loadable checkpoint, and loadable rate matrix.
3. Without this gate, a FragFM artifact can look deployable while still having
   no shared HUMU conditional signal.

## Contract

For production shared HUMU conditioning, rule-level `humu_embedding` values
must be finite Lorentz full-coordinate vectors. Current HUMU/HFM space uses
129 coordinates (`dim=128` spatial dimension plus time coordinate). Validation
must use `mf_core.geometry.normalize_lorentz_embedding()` rather than checking
length alone.

Records without `humu_embedding` are still valid local FragFM records, but they
must reduce HUMU coverage in quality reporting.

## Allowed Files

Expected code scope:

- `models/mf-generators/fragfm/train.py`
- new `models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py`
- `tests/unit/test_generators.py`
- Owner A progress / architecture documentation

Out of scope:

- HUMU pretraining pipeline/config/loss/checkpoints
- HUMU encoder architecture
- HFM architecture/checkpoints
- Owner B code
- `/workspace/SemMol` and `/workspace/Projects` writes or execution

## Acceptance

Local engineering acceptance for W11:

- training records with valid 129-dimensional Lorentz `humu_embedding` preserve
  that vector in `vocab.json`;
- invalid HUMU embeddings in training data fail fast;
- quality report emits JSON with rule count, HUMU coverage, invalid embedding
  count, checkpoint/rate-matrix load status, and pass/fail status;
- quality report fails when HUMU coverage is below the configured threshold;
- focused pytest, py_compile, and `git diff --check` pass.

Remaining production gate after W11:

- train a real production FragFM artifact set with shared HUMU labels;
- set a project threshold for HUMU coverage and benchmark quality;
- deploy artifact paths in the target environment;
- run official benchmark / cluster validation.

## Back-Check

- [x] W11 is scoped to FragFM local engineering readiness.
- [x] Existing service/sampler/deployment wiring is reused, not replaced.
- [x] Current local FragFM artifact is treated as smoke/runtime evidence only.
- [x] HUMU pretraining remains frozen.
- [x] Owner B code remains read-only.
