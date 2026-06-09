# W9 HFM Decoder Production Readiness Gap

Date: 2026-06-06
Scope: Owner A, W9 HFM-3D neural geometry decoder production readiness

## Current Position

The HFM neural geometry decoder engineering path exists and is locally verified.
It is not production geometry evidence yet.

Current verified engineering evidence:

| Gate | Local Evidence | Status |
|---|---|---|
| Source loading | `load_geometry_training_examples()` reads SDF-backed decoder JSON entries and validates 129-dimensional Lorentz latents | Local pass |
| Tiny training/export | `train_geometry_decoder_artifact()` trains a small torch artifact with latent dimension, max atom count, entries, and source artifact metadata | Local pass |
| Runner contract | `python -m mf_generators.hfm_3d.decoder.neural_geometry_decoder --artifact <artifact.pt>` speaks the existing HFM molecular decoder JSON stdin/stdout contract | Local pass |
| HFM consumption | HFM generator preserves `metadata.decoder_mode=neural_geometry_decoder` when decoder output provides it | Local pass |
| CLI wrapper | `models/mf-generators/hfm_3d/train_geometry_decoder.py` wraps training/export | Local pass |

Current smoke artifact:

- `checkpoints/hfm3d_4h200/decoder.json`
- one entry: ethanol / `CCO`
- `humu_checkpoint` points to a pytest temp path
- status: smoke/full-flow only, not production geometry quality

Current run planning:

- Production training run plan:
  `docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-training-run-plan.md`
- The plan defines source artifact requirements, a non-protected candidate
  output directory, training command template, runner smoke, HFM generator smoke,
  benchmark caveats, and stop conditions.

## Latest Source Inventory Check

A read-only inventory check on 2026-06-06 scanned
`data/processing/generator_artifacts/` for files matching the production run
plan decoder-source naming intent.

Observed result:

- No real HFM decoder source artifact was found.
- No `hfm_decoder_source_YYYYMMDD_<run_id>.json` production-candidate input was
  found.
- No non-protected `checkpoints/hfm_geometry_decoder_candidate_*` directory was
  found.
- Only W11 FragFM local engineering data and sample-export files were present
  in the matched inventory.

Conclusion:

- W9 remains blocked on approved real latent/SDF decoder source data.
- Do not train from the current one-entry
  `checkpoints/hfm3d_4h200/decoder.json` smoke artifact as production evidence.
- The next W9 action is still to obtain or identify the approved decoder source,
  then run the documented source preflight in the W9 production training plan.

## Non-Promotion Reasons

W9 is not production-ready because these gates are still missing:

| Missing Gate | Required Evidence | Owner / Resource |
|---|---|---|
| Real decoder source data | Multi-molecule decoder JSON with valid SDF geometries, 129-dimensional Lorentz latents, meaningful chemical coverage, and stable HUMU checkpoint provenance | Owner A + data |
| Production decoder artifact | Trained neural geometry decoder artifact from real source data, with manifest or release notes | Owner A + compute/data |
| Deployment mode decision | Explicit choice between `HFM_DECODER_PATH`, injected decoder, or `HFM_MOLECULAR_DECODER_COMMAND` | Owner A decision |
| Geometry benchmark | Quantitative geometry quality evidence on real molecules, not only parser/runtime smoke | Owner A + benchmark data |
| Cluster runtime | Real service deployment with production decoder artifact or command, readiness, and generation request/response evidence | Owner A + H10 resources |

## Next Executable Gates

Proceed in this order unless the user reprioritizes:

1. Define the real HFM decoder source artifact requirements: minimum molecule
   count, SDF validity, latent validation, HUMU checkpoint provenance, and
   chemical coverage.
2. Use the production decoder training run plan to prepare a new non-protected
   candidate output directory.
3. Train only after user/resource approval.
4. Run the decoder runner smoke and HFM generator smoke from the run plan
   against the new artifact.
5. Record geometry benchmark and cluster evidence before promotion.

## Stop Conditions

Stop and ask before:

- overwriting `checkpoints/hfm3d_4h200`;
- launching a long decoder training run;
- choosing the production deployment mode for W9;
- changing HFM deployment defaults;
- modifying HUMU pretraining or HFM Lorentz flow architecture;
- editing Owner B implementation files;
- killing or modifying external `/workspace/SemMol` processes.

## Back-Check

- W9 engineering path is complete locally.
- The existing `checkpoints/hfm3d_4h200/decoder.json` remains smoke evidence
  only.
- A production training run plan now exists, but it has not been executed.
- Production W9 still needs real data, artifact, benchmark, deployment mode, and
  cluster validation.
