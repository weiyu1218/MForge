# W13 Cross-Paradigm KD Preflight

Date: 2026-06-04
Owner: A / generation upstream

## Scope

W13 covers production teacher-student distillation across generator paradigms.
This local gate is limited to the teacher embedding artifact path used by the
generator training CLIs.

Out of scope:

- changing `CrossParadigmKDLayer` scoring/loss semantics;
- changing HypSeek teacher service semantics;
- training a production teacher;
- cluster publishing or official benchmark validation;
- HUMU pretraining, HUMU encoder architecture, HFM architecture/checkpoints.

## Existing Evidence

Already implemented:

- `mf_core.routing.cross_paradigm_kd.CrossParadigmKDLayer` accepts oracle
  `normalized_score`, external `teacher_distribution`, and teacher embedding
  targets.
- Boltz2 and HypSeek score adapters convert oracle records into normalized
  teacher distributions.
- `generator_router_svc.main` can call `HYPSEEK_TEACHER_COMMAND` or
  `HYPSEEK_TEACHER_URL`, and exposes a local `hypseek_app`.
- HFM-3D, FragFM, UAS, CReM, and MMPT training paths can consume
  `--kd-teacher-embeddings`.
- iCLM update path can receive `kd_teacher_embeddings` through service update
  requests or an external runner.
- Compose, Kubernetes, and Helm wiring for HypSeek teacher and iCLM update env
  already exists.

## Gap

The score-teacher path is covered, and embedding KD consumers exist. The local
missing piece is the production handoff artifact:

- each generator training CLI expects a finite teacher embedding artifact;
- `load_teacher_embeddings_artifact()` validates at load time, but there is no
  standalone preflight/export tool that turns teacher records into a canonical
  artifact and reports count/dimension/readiness before training;
- without this tool, production training can fail late or silently use a
  mismatched teacher embedding dimension.

## Acceptance

Local engineering acceptance:

- add a small `mf_core.routing.kd_artifacts` utility module;
- export canonical `teacher_embeddings` JSON from JSON/JSONL teacher records;
- validate finite values, consistent dimension, optional expected dimension,
  and minimum embedding count;
- provide `python -m mf_core.routing.kd_artifacts` for export/report;
- add focused tests and update Owner A progress/docs.

Remaining production gate:

- real teacher records / teacher embedding source;
- production teacher service or runner deployment;
- real generator distillation runs;
- benchmark quality and cluster validation.

## Back-Check

- [x] This gate is artifact engineering, not KD algorithm research.
- [x] Existing generator KD consumers remain unchanged.
- [x] Existing HypSeek score teacher path remains unchanged.
