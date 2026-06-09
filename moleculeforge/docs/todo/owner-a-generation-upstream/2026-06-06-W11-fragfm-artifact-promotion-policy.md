# W11 FragFM Artifact Promotion Policy

Date: 2026-06-06
Scope: Owner A, W11 FragFM shared HUMU conditional-space artifacts

## Current Candidate

`checkpoints/fragfm_humu_5k/` is a strict-local engineering candidate. It can be
used for local service smoke, deployment-default hardening, and follow-on
benchmark preparation. It is not final production W11 acceptance.

Current local candidate facts:

- Source data: `data/processing/generator_artifacts/fragfm_records_humu_labeled.jsonl`
- Manifest fields: `records=5000`, `fragments=2860`,
  `humu_embedding_count=5000`, `humu_embedding_coverage=1.0`
- HUMU embedding dimension: 129 Lorentz full coordinates
- Training config: 1 epoch, batch size 64, hidden dim 8, CPU,
  `--rate-optimizer sgd --disable-rate-grad-clip`
- Strict quality report fields: `status=pass`, `rules=5000`,
  `fragments=2860`, `invalid_humu_embeddings=0`,
  `checkpoint_loadable=true`, and `rate_matrix_loadable=true`

The runtime smoke regression now proves the service path can load this artifact
and generate one RDKit-parseable molecule. Cold-start loading is slow on the
current workstation because PyTorch import, `vocab.json` parsing, and the large
`rate_matrix.pt` artifact dominate startup. Production promotion must therefore
include cluster cold-start and service readiness evidence, not only local
artifact schema checks.

## Protected Paths

Do not overwrite:

- `checkpoints/fragfm`
- `checkpoints/humu`
- `checkpoints/hfm3d_4h200`

`checkpoints/fragfm` remains the old protected runtime smoke artifact and must
not be silently replaced by a promoted HUMU-conditioned artifact.

## Promotion Rule

Future production FragFM artifacts must be written to a new explicit directory,
for example:

- `checkpoints/fragfm_humu_production/`
- `checkpoints/fragfm_humu_YYYYMMDD_<run_id>/`

Deployment defaults must not be moved to the new path until all promotion gates
below have recorded evidence. If a production path replaces
`checkpoints/fragfm_humu_5k/`, keep the 5k directory as historical local
engineering evidence unless the user explicitly authorizes cleanup.

## Promotion Gates

- HUMU coverage gate: strict quality report status `pass`, 129-dimensional HUMU
  coverage at the declared threshold, and zero invalid HUMU embeddings.
- Runtime gate: `fragfm_generator_svc._build_generator()` loads vocab,
  checkpoint, and rate matrix; `FragFMGenerator.generate(batch_size=1)` returns
  at least one RDKit-parseable molecule with the configured artifact paths.
- Training gate: manifest records data source, record count, fragment count,
  epochs, hidden dimension, optimizer choices, HUMU coverage, and checkpoint /
  rate-matrix paths. In the current manifest schema these are represented by
  fields including `records`, `fragments`, `epochs`, `rate_optimizer`,
  `rate_grad_clip`, `humu_embedding_count`, `humu_embedding_coverage`,
  `vocab_path`, `checkpoint_path`, and `rate_matrix_path`.
- Benchmark gate: GuacaMol, PMO, and MOSES thresholds are not relaxed. W5
  remains blocked until official benchmark resources and production-quality
  generated samples exist.
- Deployment gate: Docker Compose, raw Kubernetes, Helm, and real cluster smoke
  evidence point to the promoted artifact. Record pod readiness, service
  startup time, and at least one generation request/response.
- Ownership gate: Owner A changes remain in generation-upstream scope. Owner B
  implementation files are not modified unless the user explicitly authorizes
  that handoff.

## Current Non-Promotion Reasons

`checkpoints/fragfm_humu_5k/` is not promoted to production because it still
lacks:

- production-scale training configuration beyond the 1-epoch hidden-dim-8 CPU
  candidate;
- official benchmark evidence;
- formal release threshold approval;
- cluster cold-start and runtime service validation;
- immutable production artifact naming and release record.
