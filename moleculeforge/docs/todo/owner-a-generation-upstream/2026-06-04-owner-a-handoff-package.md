# 2026-06-04 Owner A Handoff Package

## Purpose

This document answers Owner B's P0 request for a complete Owner A handoff list before W4 validation.

Owner A scope covered here:

- W2 pocket / intent HUMU embedding producer.
- W6 TAR ProxylessNAS runner.
- W8-E JMCG engineering skeleton.
- Embedding validation hardening.
- W9 HFM neural geometry decoder path.
- W10 Enc_intent / HCIV checkpoint train/export path.
- W11 FragFM shared HUMU quality gate.
- W13 Cross-Paradigm KD teacher embedding artifact gate.
- W9/W10/W11 stage re-acceptance hardening.

Owner B implementation files remain read-only from Owner A. C1/C2/C3 contract fields were not changed in this handoff; validation was tightened inside existing fields and artifact gates.

## Shared Contract Status

- C1 `moleculeforge.jmcg.feedback.v1`: unchanged. W2/HFM now fail closed on invalid 129-dimensional embeddings through shared Lorentz validation.
- C2 CRG predicates: unchanged. No new predicate names were introduced by Owner A.
- C3 HUMU encoder: unchanged. Current steering-capable embeddings remain 129-dimensional Lorentz full-coordinate vectors from the existing HUMU checkpoint.
- Shared file occupancy table in `docs/architecture/corearchitecture-v2-completion-interface-acceptance.md` now marks W2 `_jmcg_context_feedback_from_state` and HFM `_feedback_embedding_records` as completed.

## Handoff Matrix

| Work item | Owner A delivery | Key files | Verification evidence | Remaining gate |
|---|---|---|---|---|
| W2 | Optional pocket HUMU enrichment and conservative intent-axis steering | `services/orchestrator-svc/src/orchestrator_svc/main.py`, `tests/unit/test_service_artifact_status.py`, `tests/unit/test_generators.py` | `uv run pytest tests/unit/test_service_artifact_status.py tests/unit/test_generators.py -q` passed after authorization; hardening later passed 273 items | Production use still depends on real `HUMU_ENCODER_TARGET` and compatible HUMU checkpoint |
| W6 | Local TAR ProxylessNAS command target | `services/generator-router-svc/src/generator_router_svc/tar_proxyless_runner.py`, `tests/unit/test_task_router.py` | W6 focused pytest and `tests/unit/test_task_router.py` file-level 30 items passed; command smoke passed | Real reward data, production `TAR_PROXYLESS_SEARCH_COMMAND`, cluster validation |
| W8-E | JMCG engineering skeleton outputting joint sample records | `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/jmcg_sampler.py`, `tests/unit/test_generators.py` | Focused JMCG/HFM tests passed; later embedding hardening file-level focused pytest passed 273 items | W8-R real joint training quality, artifact, cluster/e2e validation |
| Embedding hardening | Shared Lorentz validation for W2/HFM/W8-E | `libs/mf-core/src/mf_core/geometry/lorentz.py`, HFM generator, JMCG sampler, orchestrator service | 4 tests failed RED first, then passed; `uv run pytest tests/unit/test_generators.py tests/unit/test_service_artifact_status.py -q` passed 273 items | None for local validation; production data still required |
| W9 | HFM neural geometry decoder train/export/runner path | `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/neural_geometry_decoder.py`, `models/mf-generators/hfm_3d/train_geometry_decoder.py`, `tests/unit/test_generators.py` | W9 focused + legacy decoder 6 items passed; `tests/unit/test_generators.py -q` passed 65 items; current hardening adjacent shard passed 13 items | Real production-quality decoder artifact, env/command deployment, geometry benchmark, cluster validation |
| W10 | Supervised HCIV encoder checkpoint train/export path | `services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_training.py`, `services/cig-compiler-svc/train_hciv_encoder.py`, HCIV encoder tests | W10 focused 4 items passed; `tests/unit/test_cic_compiler.py -q` passed 31 items; current hardening adjacent shard passed 13 items | Real supervised CIG/HCIV data, production checkpoint, `HCIV_CHECKPOINT_PATH`, cluster/downstream validation |
| W11 | FragFM shared HUMU training evidence preservation and quality report | `models/mf-generators/fragfm/train.py`, `models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py`, `tests/unit/test_generators.py` | W11 focused 4 items passed; FragFM subset 9 items passed; current hardening adjacent shard passed 13 items; strict CLI smoke `pass 50 0 0.0 True True` | Real HUMU-labeled FragFM data, production artifact, formal coverage/benchmark thresholds, cluster validation |
| W13 | KD teacher embedding artifact export/report gate | `libs/mf-core/src/mf_core/routing/kd_artifacts.py`, `tests/unit/test_cross_paradigm_kd.py` | W13 focused 2 items passed; `tests/unit/test_cross_paradigm_kd.py -q` passed 18 items; CLI smoke `pass 2 2 cross_paradigm_teacher_embeddings.v1` | Real production teacher source, real distillation training, benchmark quality, cluster validation |

## Current Hardening Evidence

The latest Owner A hardening gate is recorded in:

- `docs/todo/owner-a-generation-upstream/2026-06-04-W9-W10-W11-hardening-gate.md`

Latest focused verification:

- 4 new hardening tests failed RED first and then passed.
- Adjacent focused pytest passed: 13 items, exit code 0.
- `python -m py_compile` passed for touched W9/W10/W11 code and tests.
- `git diff --check` passed for touched hardening files and docs.
- W11 strict CLI smoke passed on the local runtime artifact with coverage threshold 0.0: `pass 50 0 0.0 True True`.

## W4 Notes For Owner B

- Do not treat the local HFM, FragFM, HCIV, or KD smoke artifacts as production quality.
- Do not report official benchmark quality until H8 official data and thresholds exist.
- If W4 full validation fails inside Owner A files, hand back the exact command, exit code, failing test, file, line and stderr.
- If failures involve W1/W3/W5/W12 implementation or C-class resources, keep them in Owner B/resource scope unless the failing stack enters Owner A code.
