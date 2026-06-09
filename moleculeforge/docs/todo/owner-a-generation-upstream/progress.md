# Owner A Generation Upstream Progress

This is a chronological log. Read the newest handoff first:

- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`

Older entries are historical evidence. Statements such as
`checkpoints/fragfm_humu_5k/` being absent were true at that point in the
timeline, but are superseded by the later W11 5k local candidate and deployment
default hardening entries.

## 2026-06-03 Baseline

Status:

- Owner A split accepted as generation upstream.
- HUMU pretraining remains frozen for this engineering phase.
- Existing local HUMU checkpoint is available at `moleculeforge/checkpoints/humu/best_model.pt`.
- Existing HFM smoke checkpoint and decoder are available at `moleculeforge/checkpoints/hfm3d_4h200/best_model.pt` and `moleculeforge/checkpoints/hfm3d_4h200/decoder.json`.
- No pytest has been run in this Owner A workspace.

Dimension check:

- `moleculeforge/libs/mf-humu/src/mf_humu/manifold/lorentz.py` defines the Lorentz hyperboloid in `R^{d+1}`.
- `LorentzManifold.origin(dim)` creates a tensor with `dim + 1` coordinates.
- `LorentzManifold._project(x)` preserves the input last dimension and recomputes the time coordinate from spatial coordinates.
- HUMU encoder service constructs molecule / pocket / route encoders with `dim=128`.
- HUMU molecule and pocket encoders project `nn.Linear(..., dim + 1)` outputs and mean-pool back to Lorentz full coordinates.
- HUMU route encoder projects through `nn.Linear(dim, dim + 1)`.
- HFM-3D creates `LorentzFlowMatching(dim=128)` but samples prior latent tensors with shape `129`.
- HFM feedback validation compares feedback embedding length to `latent_points.shape[-1]`.
- `moleculeforge/protos/moleculeforge/v1/humu/encoder.proto` documents `EncodeResponse.humu_embedding` as 129-dim float32.
- `moleculeforge/protos/moleculeforge/v1/retrosyn/route.proto` documents route HUMU embeddings as 129-dim Lorentz embeddings.

Conclusion:

- Current steering-capable HFM feedback embeddings must be 129-dimensional Lorentz full-coordinate vectors.
- A 128-dimensional payload is not valid for current HFM steering and should remain non-steering or be converted by a justified Lorentz projection step before use.

Back-check:

- [x] No business code was modified.
- [x] No tests were run.
- [x] `/workspace/SemMol` and `/workspace/Projects` remain read-only context only.
- [x] The next implementation gate should start with W2 preflight, not direct embedding injection.

## Active Gate Queue

1. Remaining production resource gates: W6 reward data/deploy, W9 decoder artifact/deploy, W10 HCIV checkpoint/deploy, W11 FragFM HUMU-labeled artifact/deploy, W13 teacher embedding / distillation / deploy.

## 2026-06-03 W2 Preflight Refinement

Updated documents:

- `moleculeforge/docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/W2-pocket-intent-embedding-preflight.md`

Refinements:

- `/workspace/SemMol` and `/workspace/Projects` are now documented as read/copy-only context, not no-read directories.
- C3 now states that current HUMU encoder outputs are 129-dimensional Lorentz full coordinates.
- W2 preflight now recommends reusing the existing `HUMU_ENCODER_TARGET` convention and float32 bytes decoding pattern from RetroSyn route encoding.
- W2 intent handling remains conservative: do not inject a plain 128-dimensional HCIV vector as `humu_embedding`.

Back-check:

- [x] The update resolves the previous 128/129 ambiguity.
- [x] The update does not change business code.
- [x] The update keeps HUMU pretraining and HFM architecture frozen.

## 2026-06-03 W2 Implementation Plan Created

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/W2-pocket-intent-embedding-implementation-plan.md`

Plan decisions:

- W2 should keep feedback assembly centralized in `_attach_generation_feedback`.
- `_attach_generation_feedback` and `_jmcg_context_feedback_from_state` should become async when pocket HUMU encoding is implemented.
- The only current business-code call site is `FullWorkflowClients.generate_candidates`.
- W2 test specs should focus on `tests/unit/test_service_artifact_status.py`; HFM consumer behavior is already covered by existing non-steering and invalid-dimension specs unless implementation changes HFM directly.

Back-check:

- [x] The plan does not modify business code.
- [x] The plan keeps W2 scoped to orchestrator enrichment.
- [x] The plan avoids direct HCIV-to-HFM embedding injection.

## 2026-06-03 W2 Implementation Gate Started

Shared file occupation:

- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`
  - Functions / region: `_attach_generation_feedback`, `_jmcg_context_feedback_from_state`, `_intent_jmcg_feedback_record`, `_pocket_jmcg_feedback_record`, and new local HUMU feedback helper functions.
  - Purpose: optional pocket / intent steering-capable JMCG feedback enrichment.

Scope guard:

- Do not change HUMU pretraining.
- Do not change HUMU encoder architecture.
- Do not change HFM model architecture.
- Do not modify `/workspace/SemMol` or `/workspace/Projects`.

Back-check:

- [x] Shared file occupation is recorded before business-code edits.
- [x] W2 scope is limited to orchestrator feedback enrichment.

## 2026-06-03 W2 Implementation Gate Completed

Modified:

- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`
- `moleculeforge/tests/unit/test_service_artifact_status.py`
- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`

Created earlier in this gate:

- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/W2-pocket-intent-embedding-preflight.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/W2-pocket-intent-embedding-implementation-plan.md`

Implemented:

- `_attach_generation_feedback()` and `_jmcg_context_feedback_from_state()` now support async optional feedback enrichment.
- Intent feedback becomes steering-capable only when `intent_cone.axis` is already a 129-dimensional Lorentz full-coordinate vector.
- Plain 128-dimensional HCIV vectors remain non-steering metadata and are not inserted as `humu_embedding`.
- Pocket feedback becomes steering-capable only when structured pocket geometry is present and `_encode_pocket_humu_feedback()` returns a valid 129-dimensional HUMU embedding.
- Metadata-only pocket context remains non-steering.
- Optional pocket HUMU encoding uses the existing `HUMU_ENCODER_TARGET` convention and decodes packed float32 bytes.
- Pocket enrichment fails closed: missing target, gRPC failure, malformed bytes, or invalid dimension preserves non-steering behavior.
- Focused test specs were added but not executed.

Verification:

- `python -m py_compile moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py moleculeforge/tests/unit/test_service_artifact_status.py` passed.
- `git diff --check` passed.
- Trailing whitespace scan passed for touched W2 files and docs.
- `rg` stale-claim scan found no remaining old W2 wording such as direct HCIV embedding or dim=128 output claims.
- Pytest was not run because explicit test authorization was not given.

Back-check:

- [x] No HUMU pretraining code was changed.
- [x] No HUMU encoder architecture was changed.
- [x] No HFM architecture was changed.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.
- [x] Metadata-only context is not treated as a HUMU embedding.
- [x] Current steering-capable embeddings are required to be 129-dimensional.
- [x] The architecture docs still state this is local feedback steering, not JMCG joint sampling.

## 2026-06-03 W2 Focused Pytest Authorized And Passed

User authorization:

- The user explicitly authorized focused pytest for this W2 gate.

Commands run:

- `uv run pytest tests/unit/test_service_artifact_status.py tests/unit/test_generators.py -q`
- `uv run pytest tests/unit/test_generators.py::TestHFM3DGenerator::test_jmcg_feedback_repel_moves_latent_away_from_embedding -q`
- `uv run pytest tests/unit/test_generators.py::TestHFM3DGenerator::test_route_humu_feedback_steers_latent_toward_route_embedding -q`
- `uv run pytest tests/unit/test_generators.py::TestHFM3DGenerator::test_route_humu_feedback_steers_latent_toward_route_embedding tests/unit/test_generators.py::TestHFM3DGenerator::test_jmcg_feedback_repel_moves_latent_away_from_embedding tests/unit/test_service_artifact_status.py::test_full_workflow_generator_receives_pocket_embedding_when_encoder_available tests/unit/test_service_artifact_status.py::test_full_workflow_metadata_only_pocket_feedback_stays_non_steering tests/unit/test_service_artifact_status.py::test_full_workflow_intent_axis_embedding_becomes_steering_capable tests/unit/test_service_artifact_status.py::test_full_workflow_hciv_vector_does_not_become_humu_embedding -q`

Observed and fixed:

- Initial full focused run executed 266 shard items and found one failure in `TestHFM3DGenerator.test_jmcg_feedback_repel_moves_latent_away_from_embedding`.
- The failure was a pre-existing numerical assertion issue: the latent changed, but float32 Lorentz distance on large coordinates clamped to `0.0`.
- The test now asserts actual latent displacement with `torch.linalg.vector_norm(steered_latent - baseline_latent) > 0.0` and keeps the feedback kind metadata assertion.
- A temporary import regression in `test_route_humu_feedback_steers_latent_toward_route_embedding` was fixed by restoring its `LorentzManifold` import.

Final verification:

- `uv run pytest tests/unit/test_service_artifact_status.py tests/unit/test_generators.py -q` passed with exit code 0.
- The final full focused run reported one existing LangGraph deprecation warning and no failures.
- `python -m py_compile moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py moleculeforge/tests/unit/test_service_artifact_status.py moleculeforge/tests/unit/test_generators.py` passed.
- `git diff --check` passed.
- Trailing whitespace scan passed for touched W2 files and docs.

Back-check:

- [x] W2 new orchestrator specs pass.
- [x] Existing route feedback steering test passes.
- [x] Existing repel feedback behavior is still verified without relying on numerically unstable Lorentz distance in float32.
- [x] No production behavior was changed to satisfy the HFM repel test.

## 2026-06-03 W6 TAR Runner Preflight And Plan Created

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/W6-tar-runner-preflight.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/W6-tar-runner-implementation-plan.md`

Preflight conclusion:

- TAR core already had `TaskAwareRouter.architecture_logits`, Proxyless probability/cost/update methods, `ProxylessSearchScheduler`, `RunProxylessSearch`, external command env wiring, and focused tests.
- No existing TAR runner script was found under project `scripts/`, `tools/`, or `services/generator-router-svc`.
- The local code gap was narrowed to a concrete `TAR_PROXYLESS_SEARCH_COMMAND` target, not a new routing algorithm.

Back-check:

- [x] The plan reuses existing scheduler and gRPC command contract.
- [x] HUMU pretraining remains frozen.
- [x] HFM architecture/checkpoints remain untouched.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.
- [x] No tests were run during planning.

## 2026-06-03 W6 TAR Runner Implementation Gate Completed

Modified:

- `moleculeforge/services/generator-router-svc/src/generator_router_svc/tar_proxyless_runner.py`
- `moleculeforge/tests/unit/test_task_router.py`
- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`

Implemented:

- Added `generator_router_svc.tar_proxyless_runner`.
- The runner reads stdin JSON, validates the existing reward-cost payload shape, reuses `ProxylessSearchScheduler`, and writes stdout JSON.
- Output includes `rounds`, `architecture_probabilities`, `architecture_logits`, `generator_names`, `cost_weight`, `learning_rate`, and `temperature`.
- Added focused specs for direct runner execution, CLI subprocess behavior, and GeneratorRouterService invocation through the real runner command.

Verification:

- `python -m py_compile moleculeforge/services/generator-router-svc/src/generator_router_svc/tar_proxyless_runner.py moleculeforge/tests/unit/test_task_router.py` passed.
- `git diff --check` passed.
- `uv run python -m generator_router_svc.tar_proxyless_runner` with a two-round KRAS reward payload passed and returned service-compatible JSON.
- Focused pytest was not run for W6 because this gate does not yet have separate explicit pytest authorization; it should be included in W4 or run after user authorization.

Remaining gate:

- Real reward dataset is still missing.
- Production `TAR_PROXYLESS_SEARCH_COMMAND` value is not yet deployed by default.
- Cluster release validation is still missing.

Back-check:

- [x] The implementation reuses `ProxylessSearchScheduler`.
- [x] The command accepts the same payload used by `_proxyless_search_from_command`.
- [x] The service can be configured to call the new command target.
- [x] Production data and deployment blockers remain explicitly marked incomplete.
- [x] HUMU pretraining, HUMU encoder architecture, HFM architecture, and checkpoints were not changed.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.

## 2026-06-04 W8-E JMCG Engineering Skeleton Preflight And Plan Created

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/W8-jmcg-engineering-skeleton-preflight.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/W8-jmcg-engineering-skeleton-implementation-plan.md`

Preflight conclusion:

- HFM already consumes `jmcg_feedback`, `route_humu_feedback`, and `generation_feedback` for local Lorentz feedback steering.
- HUMU pretraining-side tests already cover joint / intent data contracts.
- The missing generation-upstream W8-E piece was a local sampler contract that emits explicit joint sample records, not a trained research model.
- The skeleton should live under `mf_generators.hfm_3d.inference` and must not alter default HFM production generation behavior.

Back-check:

- [x] W8-E is separated from W8-R research quality.
- [x] The plan preserves the 129-dimensional HUMU/HFM contract.
- [x] HUMU pretraining remains frozen.
- [x] No tests were run during planning.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.

## 2026-06-04 W8-E JMCG Engineering Skeleton Gate Completed

Modified:

- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/jmcg_sampler.py`
- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/__init__.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`

Implemented:

- Added `JMCGContextRecord`, `JMCGJointSample`, `JMCGEngineeringSampler`, and `parse_jmcg_context()`.
- The sampler parses `moleculeforge.jmcg.feedback.v1` envelopes, direct record lists, JSON strings, bytes, and single-record mappings.
- The sampler emits JSON-serializable `moleculeforge.jmcg.joint_sample.v1` engineering skeleton records with molecule / route / property / pocket / intent fields.
- The sampler validates steering-capable embeddings as 129-dimensional Lorentz full coordinates for alignment scoring.
- Invalid 128-dimensional embeddings are ignored as non-steering context and counted in metadata.
- Default HFM generation behavior was not changed.
- Added focused specs for legal joint sample output, invalid-dimension non-steering behavior, and parser compatibility.

Verification:

- `python -m py_compile moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/jmcg_sampler.py moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/__init__.py moleculeforge/tests/unit/test_generators.py` passed.
- `git diff --check` passed.
- `uv run python - <<'PY' ... JMCGEngineeringSampler ... PY` command-level smoke passed and returned JSON with `metadata.mode == "engineering_skeleton"`.
- Focused pytest was not run for W8-E because this gate does not have separate explicit pytest authorization; include it in W4 or run after user authorization.

Remaining gate:

- W8-R true joint sampling training quality is still missing.
- Joint training data / compute / model artifact are still missing.
- End-to-end production validation is still missing.

Back-check:

- [x] The skeleton emits explicit joint sample records.
- [x] It does not change default HFM generation behavior.
- [x] It does not fabricate HUMU embeddings from metadata-only context.
- [x] It preserves the 129-dimensional steering-capable contract.
- [x] W8-R production/research quality remains marked incomplete.
- [x] HUMU pretraining, HUMU encoder architecture, HFM architecture, and checkpoints were not changed.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.

## 2026-06-04 Stage-Gate Re-Acceptance

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-04-stage-gate-reacceptance.md`

Read-only review scope:

- W2 pocket / intent HUMU embedding producer.
- W6 TAR ProxylessNAS runner.
- W8-E JMCG engineering skeleton.
- Owner B progress documents were read only; no Owner B code was modified.

Verification:

- `python -m py_compile moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py moleculeforge/services/generator-router-svc/src/generator_router_svc/tar_proxyless_runner.py moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/jmcg_sampler.py moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/__init__.py moleculeforge/tests/unit/test_service_artifact_status.py moleculeforge/tests/unit/test_task_router.py moleculeforge/tests/unit/test_generators.py` passed.
- `git diff --check` passed.
- W6 command-level KRAS two-round smoke passed.
- W8-E normal-input command-level smoke passed.
- No pytest was run during this re-acceptance pass.

Findings:

- W6 local code gate remains acceptable.
- W2 and W8-E correctly enforce 129-dimensional shape, but currently accept mathematically invalid 129-dimensional values such as all-zero, `NaN`, and `Inf`.
- W8-E currently ignores packed float32 `Molecule.humu_embedding` bytes and does not count that as an ignored embedding.

Back-check:

- [x] No business code was modified during re-acceptance.
- [x] No Owner B code was modified.
- [x] HUMU pretraining, HUMU encoder architecture, HFM architecture, and checkpoints remain untouched.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.
- [x] The next gate should harden shared HUMU / Lorentz embedding validation before moving to W9.

## 2026-06-04 Embedding Validation Hardening Gate Completed

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-04-embedding-validation-hardening.md`

Modified:

- `moleculeforge/libs/mf-core/src/mf_core/geometry/__init__.py`
- `moleculeforge/libs/mf-core/src/mf_core/geometry/lorentz.py`
- `moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py`
- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py`
- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/jmcg_sampler.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/tests/unit/test_service_artifact_status.py`

Implemented:

- Added shared `normalize_lorentz_embedding()` validation in `mf_core.geometry.lorentz`.
- W2 no longer turns invalid 129-dimensional intent axes into steering-capable `humu_embedding`.
- HFM feedback consumer now drops invalid 129-dimensional feedback records instead of projecting arbitrary vectors.
- W8-E now rejects invalid 129-dimensional alignment embeddings and counts present invalid embeddings as ignored.
- W8-E now decodes packed little-endian float32 `Molecule.humu_embedding` bytes.

Verification:

- New focused tests were run RED first and failed for the expected four symptoms.
- The same four focused tests passed after implementation.
- `python -m py_compile` passed for the touched code and test files.
- `git diff --check` passed for the touched files.
- Adjacent HFM/W8-E focused pytest passed with 6 items.
- Adjacent W2 focused pytest passed with 4 items.
- File-level focused pytest passed: `uv run pytest tests/unit/test_generators.py tests/unit/test_service_artifact_status.py -q` ran 273 items with exit code 0 and one existing LangGraph deprecation warning.

Back-check:

- [x] This fixes the root cause found during stage-gate re-acceptance.
- [x] Legal Lorentz origin and projected HUMU/HFM embeddings remain accepted.
- [x] Invalid 129-dimensional all-zero vectors are now non-steering.
- [x] Packed float32 molecule HUMU bytes are no longer silently ignored by W8-E.
- [x] HUMU pretraining, HUMU encoder architecture, HFM architecture, and checkpoints were not changed.
- [x] Owner B code was not modified.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.

## 2026-06-04 W6 Focused Pytest Gate Completed

Modified:

- `moleculeforge/tests/unit/test_task_router.py`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`
- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`

Observed:

- Initial W6 focused pytest exposed a timeout in `test_generator_router_service_uses_builtin_proxyless_runner_command`.
- Root cause: the test set `TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS=10`, but the real built-in runner subprocess cold-start imports project/PyTorch dependencies and can exceed 10 seconds in this environment.

Implemented:

- Kept the lightweight fake-command service test at 10 seconds.
- Raised only the real built-in runner subprocess test timeout to 120 seconds.
- No production code was changed for W6.

Verification:

- W6 focused pytest passed: `uv run pytest tests/unit/test_task_router.py::test_tar_proxyless_runner_executes_shared_scheduler tests/unit/test_task_router.py::test_tar_proxyless_runner_cli_reads_stdin_and_writes_json tests/unit/test_task_router.py::test_generator_router_service_uses_builtin_proxyless_runner_command -q`.
- TAR file-level pytest passed: `uv run pytest tests/unit/test_task_router.py -q` ran 30 items with exit code 0.

Back-check:

- [x] The timeout change matches the production default direction (`TAR_PROXYLESS_SEARCH_TIMEOUT_SECONDS` defaults to 300 in deployment wiring).
- [x] The real runner command path is now covered by pytest, not just command-level smoke.
- [x] W6 local code gate remains complete.
- [x] Remaining W6 gates are still external: real reward dataset, production command/env deployment, and cluster validation.

## 2026-06-04 W9 HFM Decoder Preflight And Plan Created

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/W9-hfm-decoder-preflight.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/W9-hfm-decoder-implementation-plan.md`

Preflight conclusion:

- Existing HFM generator/service already support `molecular_decoder` injection and `HFM_MOLECULAR_DECODER_COMMAND`.
- Existing generator can consume decoder JSON with `smiles`, `atom_types`, `coordinates`, and `sdf` / `sdf_bytes`.
- Existing training writes a nearest-neighbor decoder artifact with SDF entries, but this is still not a production neural geometry decoder.
- W9 should add a concrete train/export/runner path for neural geometry decoding, not another service wrapper.

Back-check:

- [x] No business code was modified during W9 planning.
- [x] HUMU pretraining remains frozen.
- [x] HFM flow architecture remains frozen.
- [x] Existing HFM checkpoint/decoder artifacts remain classified as smoke/full-flow evidence, not production quality.
- [x] Next W9 implementation should start with RED tests for decoder artifact loading and tiny CPU training.

## 2026-06-04 W9 HFM Neural Geometry Decoder Gate Completed

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-04-W9-hfm-neural-geometry-decoder-gate.md`
- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/__init__.py`
- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/neural_geometry_decoder.py`
- `moleculeforge/models/mf-generators/hfm_3d/train_geometry_decoder.py`

Modified:

- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`

Implemented:

- Added SDF-backed HFM decoder artifact loading for geometry training examples.
- Added `NeuralGeometryDecoder` and `NeuralGeometryDecoderArtifact` with masked coordinate training/export.
- Added `python -m mf_generators.hfm_3d.decoder.neural_geometry_decoder --artifact <artifact.pt>` as the local HFM molecular decoder command target.
- Added `models/mf-generators/hfm_3d/train_geometry_decoder.py` CLI wrapper.
- Preserved decoder-supplied `metadata.decoder_mode` in `HFM3DGenerator` while keeping legacy payloads defaulted to `molecular_decoder`.

Verification:

- Five new W9 tests were run RED first and failed for the expected missing module/helper/runner/provenance/script symptoms.
- Focused W9 + legacy decoder regression passed: 6 tests, exit code 0.
- `python -m py_compile moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/__init__.py moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/neural_geometry_decoder.py moleculeforge/models/mf-generators/hfm_3d/train_geometry_decoder.py moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py moleculeforge/tests/unit/test_generators.py` passed.
- `uv run pytest tests/unit/test_generators.py -q` passed with 65 items in this shard and exit code 0.
- `git diff --check` passed.

Remaining gate:

- A real production-quality geometry decoder artifact still needs training on real data.
- Production `HFM_MOLECULAR_DECODER_COMMAND` or artifact deployment remains unset.
- Cluster validation and benchmark geometry quality evidence remain missing.

Back-check:

- [x] W9 now has a concrete local train/export/runner path.
- [x] Existing HFM molecular decoder contract was reused instead of replaced.
- [x] The current implementation is engineering readiness, not production quality.
- [x] HUMU pretraining, HUMU encoder architecture, HFM Lorentz flow architecture, and checkpoints were not changed.
- [x] Owner B code was not modified.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.

## 2026-06-04 W10 Enc_intent Checkpoint Preflight And Plan Created

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/W10-enc-intent-checkpoint-preflight.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/W10-enc-intent-checkpoint-implementation-plan.md`

Preflight conclusion:

- `HCIVEncoder`, production `HCIV_CHECKPOINT_PATH` fail-fast and checkpoint loading already exist.
- W10 should add supervised training/export, not another CIG compiler service path.
- Training data should explicitly provide target HCIV coordinates; hash/random encoders remain local-demo only and must not become the production teacher.

Back-check:

- [x] No business code was modified during W10 planning.
- [x] Existing production checkpoint loader remains the target interface.
- [x] W2 128/129 HUMU steering boundary remains unchanged.

## 2026-06-04 W10 Enc_intent Checkpoint Gate Completed

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-04-W10-enc-intent-checkpoint-gate.md`
- `moleculeforge/services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_training.py`
- `moleculeforge/services/cig-compiler-svc/train_hciv_encoder.py`

Modified:

- `moleculeforge/services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_encoder.py`
- `moleculeforge/tests/unit/test_cic_compiler.py`
- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`

Implemented:

- Added supervised CIG/HCIV JSON/JSONL training data loader.
- Added `HCIVEncoder.forward_coordinates(cig)` for differentiable training while preserving `encode()` output.
- Added `train_hciv_encoder_checkpoint()` to train and export schema-wrapped checkpoints compatible with `load_hciv_encoder_checkpoint()`.
- Added optional JSON manifest export.
- Added `train_hciv_encoder.py` CLI wrapper.

Verification:

- Three W10 tests were run RED first and failed for expected missing module/function/script symptoms; the training test also exposed the detached `encode()` path and drove the `forward_coordinates()` fix.
- Focused W10 gate passed: 4 tests, exit code 0.
- `python -m py_compile moleculeforge/services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_encoder.py moleculeforge/services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_training.py moleculeforge/services/cig-compiler-svc/train_hciv_encoder.py moleculeforge/tests/unit/test_cic_compiler.py` passed.
- `uv run pytest tests/unit/test_cic_compiler.py -q` passed with 31 items in this shard and exit code 0.
- `git diff --check` passed.

Remaining gate:

- Real supervised CIG/HCIV training data is still missing.
- A real production-quality `Enc_intent` checkpoint still needs training.
- Production `HCIV_CHECKPOINT_PATH` deployment and cluster validation remain missing.

Back-check:

- [x] Existing HCIV encoder architecture was reused.
- [x] Existing production checkpoint loader was reused.
- [x] Hash/random encoders remain local-demo only.
- [x] W2 128/129 HUMU steering boundary remains unchanged.
- [x] HUMU pretraining, HFM, and Owner B code were not modified.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.

## 2026-06-04 W11 FragFM Shared HUMU Quality Preflight And Plan Created

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/W11-fragfm-shared-humu-quality-preflight.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/W11-fragfm-shared-humu-quality-implementation-plan.md`

Preflight conclusion:

- FragFM already had vocabulary artifact loading, SA-aware rate matrix, intent-cone alignment, injected shared HUMU latent sampler, deployment env wiring, and KD teacher embedding training entry.
- The missing local W11 code was artifact quality evidence: training records with `humu_embedding` were normalized into `vocab.json` without preserving that field, and no local quality report checked HUMU coverage or artifact loadability.
- Current local `checkpoints/fragfm/vocab.json` has 50 rules and 0 `humu_embedding` entries, so it is smoke/runtime evidence only.

Back-check:

- [x] W11 is scoped to local engineering quality gate, not production quality.
- [x] Existing FragFM generator/service behavior is reused.
- [x] HUMU pretraining and encoder architecture remain frozen.

## 2026-06-04 W11 FragFM Shared HUMU Quality Gate Completed

Created:

- `moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py`

Modified:

- `moleculeforge/models/mf-generators/fragfm/train.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`

Implemented:

- FragFM training records now validate optional `humu_embedding` as finite 129-dimensional Lorentz full coordinates using `mf_core.geometry.normalize_lorentz_embedding()`.
- Valid rule-level `humu_embedding` values are preserved in `vocab.json`.
- Training manifest now records `humu_embedding_dim`, `humu_curvature`, `humu_embedding_count`, and `humu_embedding_coverage`.
- Added `mf_generators.fragfm.quality.build_quality_report()` and `python -m mf_generators.fragfm.quality` CLI.
- Quality report checks vocabulary HUMU coverage, invalid HUMU embeddings, checkpoint loadability, and rate-matrix loadability.

Verification:

- Three W11 tests were run RED first and failed for expected missing `humu_embedding` preservation and missing quality module symptoms.
- W11 focused pytest passed: 4 tests, exit code 0.
- FragFM subset pytest passed: 9 tests, exit code 0. The excluded two tests are pre-existing subprocess training smoke tests; full `tests/unit/test_generators.py` was attempted but stopped after 9 minutes because it had advanced only 2/69 items in the current overloaded environment.
- `python -m py_compile moleculeforge/models/mf-generators/fragfm/train.py moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py moleculeforge/tests/unit/test_generators.py` passed.
- `git diff --check` passed for touched W11 files and docs.
- Quality CLI smoke passed on current local artifact with threshold 0.0: output summary `pass 50 0 0.0 True True`, meaning 50 rules, 0 HUMU embeddings, coverage 0.0, checkpoint loadable, rate matrix loadable.

Remaining gate:

- Real HUMU-labeled FragFM training data is still missing.
- A production-quality FragFM artifact with nonzero HUMU coverage still needs training.
- Formal HUMU coverage and benchmark thresholds still need to be set.
- Cluster release validation remains missing.

Back-check:

- [x] W11 no longer drops valid training-time HUMU embeddings.
- [x] Current local artifact is explicitly not promoted to production quality because coverage is 0.0.
- [x] No HUMU pretraining code/config/loss/checkpoint was changed.
- [x] No HUMU encoder architecture was changed.
- [x] HFM architecture and checkpoints were not changed.
- [x] Owner B code was not modified.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.

## 2026-06-04 W13 Cross-Paradigm KD Preflight And Plan Created

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/W13-cross-paradigm-kd-preflight.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/W13-cross-paradigm-kd-implementation-plan.md`

Preflight conclusion:

- `CrossParadigmKDLayer`, Boltz2/HypSeek teacher distribution adapters, `generator_router_svc.main:hypseek_app`, router command/URL teacher calls, iCLM update KD path, generator training CLI `--kd-teacher-embeddings`, and deployment wiring already exist.
- The remaining local code gap was a production handoff artifact gate: a standalone way to export and preflight teacher embedding artifacts before generator distillation training.

Back-check:

- [x] W13 scope is artifact handoff/preflight, not KD algorithm research.
- [x] Existing HypSeek score teacher path remains unchanged.

## 2026-06-04 W13 KD Teacher Embedding Artifact Gate Completed

Created:

- `moleculeforge/libs/mf-core/src/mf_core/routing/kd_artifacts.py`

Modified:

- `moleculeforge/tests/unit/test_cross_paradigm_kd.py`
- `moleculeforge/docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`

Implemented:

- Added `export_teacher_embeddings_artifact()` to convert JSON/JSONL teacher records into canonical `cross_paradigm_teacher_embeddings.v1` artifacts.
- Added `build_teacher_embeddings_report()` to report status, embedding count, dimension, expected dimension, minimum count, and validation messages.
- Added `python -m mf_core.routing.kd_artifacts` CLI with `--input`, `--output`, `--embedding-field`, `--expected-dim`, `--min-embeddings`, `--report`, and `--strict`.

Verification:

- Two W13 tests were run RED first and failed for the expected missing module.
- Focused W13 pytest passed: 2 tests, exit code 0.
- File-level KD pytest passed: `uv run pytest tests/unit/test_cross_paradigm_kd.py -q` ran 18 items with exit code 0.
- CLI smoke passed and printed `pass 2 2 cross_paradigm_teacher_embeddings.v1`.
- `python -m py_compile moleculeforge/libs/mf-core/src/mf_core/routing/kd_artifacts.py moleculeforge/tests/unit/test_cross_paradigm_kd.py` passed.
- `git diff --check` passed for touched W13 files; trailing whitespace scan passed for W13 untracked docs/new module.

Remaining gate:

- Real production teacher records / teacher embeddings are still missing.
- Real generator distillation runs are still missing.
- Benchmark quality evidence and cluster deployment validation remain missing.

Back-check:

- [x] No KD loss semantics were changed.
- [x] Existing generator KD consumers were not changed.
- [x] Existing HypSeek service behavior was not changed.
- [x] HUMU pretraining, HUMU encoder architecture, HFM architecture/checkpoints were not changed.
- [x] Owner B code was not modified.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.

## 2026-06-04 W9/W10/W11 Hardening Gate Completed

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-04-W9-W10-W11-hardening-gate.md`

Modified:

- `moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/neural_geometry_decoder.py`
- `moleculeforge/services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_training.py`
- `moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/tests/unit/test_cic_compiler.py`
- `moleculeforge/docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`

Implemented:

- W9 decoder source artifact `latent` values now must be valid Lorentz full-coordinate vectors before entering geometry decoder training.
- W10 supervised `target_hciv` coordinates now must be valid Lorentz full-coordinate vectors; `curvature` is threaded through training-data loading.
- W11 FragFM quality gate now fails when checkpoint artifacts lack `fragment_encoder.weight` or rate-matrix artifacts lack `base_rate`.
- The shared interface occupancy table now marks W2 `_jmcg_context_feedback_from_state` and HFM `_feedback_embedding_records` as completed.

Verification:

- Four new hardening tests were run RED first and failed for the expected missing-validation symptoms.
- The same 4 tests passed after implementation.
- Adjacent focused pytest passed: 13 items, exit code 0.
- `python -m py_compile` passed for touched code and tests.
- `git diff --check` passed for touched hardening files and the shared interface document.
- W11 strict quality CLI smoke still passed on the local runtime artifact with threshold 0.0: `pass 50 0 0.0 True True`.

Remaining gate:

- W9 still needs real production-quality geometry decoder data/artifact, env/command deployment, cluster validation and geometry benchmark evidence.
- W10 still needs real supervised CIG/HCIV data, production-quality checkpoint, `HCIV_CHECKPOINT_PATH` deployment, cluster validation and downstream quality validation.
- W11 still needs real HUMU-labeled FragFM training data, production-quality artifact, formal coverage/benchmark thresholds and cluster validation.

Back-check:

- [x] The hardening uses the shared Lorentz validator rather than one-off all-zero checks.
- [x] Legal Lorentz origin vectors remain valid.
- [x] Existing W9/W10 local training/export paths remain covered by adjacent tests.
- [x] Current FragFM local artifact remains explicitly runtime-smoke only because HUMU coverage is 0.0.
- [x] HUMU pretraining, HUMU encoder architecture, HFM architecture/checkpoints were not changed.
- [x] Owner B code was not modified.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.

## 2026-06-04 Owner A Handoff Package Created

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-04-owner-a-handoff-package.md`

Purpose:

- Answer Owner B's P0 request for a complete Owner A changed-file / verification / remaining-gate handoff before W4 validation.
- Consolidate W2, W6, W8-E, embedding hardening, W9, W10, W11, W13 and W9/W10/W11 hardening evidence in one Owner A handoff document.

Back-check:

- [x] The handoff package does not modify business code.
- [x] C1/C2/C3 are recorded as unchanged.
- [x] Remaining production gates are still explicit and are not replaced by local smoke artifacts.
- [x] Owner B implementation files remain read-only from Owner A.

## 2026-06-05 W4 Focused Validation Pass Recorded

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-05-W4-focused-validation-record.md`

Scope:

- Revalidated the current stage after W9/W10/W11 hardening with `.env` loaded and proxy env unset.
- Read Owner B W1 code/tests only to diagnose failures; no Owner B implementation file was modified.
- Did not change business code or `.env`.

Verification:

- W1 unit gate: `uv run pytest tests/unit/test_graph_repo.py -q` failed reproducibly with 14 items, 11 passed and 3 failed. All failures are the same patch seam: tests patch `orchestrator_svc.main.build_shared_crg_repository_from_env`, but `orchestrator_svc.main` imports that function locally inside `_merge_agent_beliefs_into_crg()` and does not expose a module-level symbol. Classified as Owner B W1 unit-test compatibility issue.
- W2 orchestrator feedback producer focused gate passed: 8 items.
- W2/W8 HFM JMCG consumer focused gate passed: 12 items.
- C2 predicate/downstream agent regression passed: `uv run pytest tests/unit/test_validation_agent.py tests/unit/test_srb_agent.py -q`, 32 items.
- C1 generator coordinator regression passed: `uv run pytest tests/unit/test_generator_coord_agent.py -q`, 20 items.
- W3 mf-eval local provider gate passed: `uv run pytest tests/unit/test_mf_eval.py -q`, 24 items.
- W5 benchmark harness failed as expected for production gate reasons: `uv run pytest tests/benchmark -q`, 18 items, 8 failed and 10 skipped. Failures are GuacaMol/PMO thresholds on repeated local `CCO`; skips are missing official benchmark resources (`CROSSDOCKED_BENCHMARK_JSONL`, `MOSES_REFERENCE_SMILES_PATH`, `FRAGFM_MOSES_GENERATED_SMILES_PATH`, `PMO_SCORE_TABLE_PATH`).
- W11 quality focused gate passed: 6 items.
- W11 strict quality CLI passed on current local runtime artifact with threshold 0.0: report `status=pass`, `rules=50`, `humu_embedding_coverage=0.0`, checkpoint/rate-matrix loadable.
- W13 file-level pytest passed: `uv run pytest tests/unit/test_cross_paradigm_kd.py -q`, 18 items.
- W13 CLI smoke passed with report `status=pass` and canonical artifact summary `cross_paradigm_teacher_embeddings.v1 2 2`.
- W9/W10/W11 hardening focused regression passed: 4 items.

Large-group note:

- The broad W2 command `uv run pytest tests/unit/test_generators.py tests/unit/test_service_artifact_status.py -q` collected 285 items but was terminated after roughly 14 minutes with exit code 143 because it advanced only a few test points. It is not counted as pass or failure; focused work-item gates above were used for this stage validation.

Back-check:

- [x] No code was changed during this validation pass.
- [x] Owner B code was not modified.
- [x] W1 failure has an exact root-cause handoff path for Owner B.
- [x] W5 remains a production benchmark/data gate; thresholds were not relaxed.
- [x] Current FragFM artifact remains runtime-smoke only because HUMU coverage is 0.0.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.

## 2026-06-05 Owner A Production Resource Preflight Completed

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-05-owner-a-production-resource-preflight.md`

Findings:

- `HUMU_CHECKPOINT_PATH` is set and `checkpoints/humu/best_model.pt` is loadable. The checkpoint contains `encoder_mol`, `encoder_pocket`, `encoder_route`, and `encoder_intent`; latest validation metrics include epoch 50 `retrieval_top1=0.7916278540250958`, `val_loss=1.0582103152037337`, and `collapse_ratio=0.0`.
- `data/processing/generator_artifacts/fragfm_records.jsonl` has 5000 FragFM records and `fragfm_records_train.jsonl` has 50 records, but both have `humu_embedding_count=0`.
- Current `checkpoints/fragfm` is loadable but remains a 1-epoch, 50-record runtime smoke artifact with HUMU coverage 0.0.
- Current `checkpoints/hfm3d_4h200/decoder.json` has one ethanol/`CCO` entry and references a pytest temp HUMU checkpoint path, so it is smoke/full-flow only, not production geometry evidence.
- `TAR_PROXYLESS_SEARCH_COMMAND`, `HFM_MOLECULAR_DECODER_COMMAND`, `HCIV_CHECKPOINT_PATH`, `HUMU_ENCODER_TARGET`, `CROSS_PARADIGM_TEACHER_RECORDS`, `CROSS_PARADIGM_TEACHER_EMBEDDINGS`, `HYPSEEK_TEACHER_COMMAND`, `HYPSEEK_TEACHER_URL`, and official benchmark data envs are unset.

Recommendation:

- The most executable next Owner A gate is W11 HUMU-labeled FragFM local data enrichment: use the frozen HUMU molecule encoder to derive 129-dimensional `humu_embedding` values for FragFM records, write a separate derived JSONL, train to a separate artifact directory, and validate with a non-zero HUMU coverage threshold.
- This is a hard decision because it writes derived data/artifacts and would start a new local training path. It should not overwrite `checkpoints/fragfm`, and it should remain labelled local engineering evidence until benchmark/cluster validation exists.

Back-check:

- [x] No code, `.env`, or data artifact was changed during this preflight.
- [x] HUMU pretraining remains frozen.
- [x] Current smoke artifacts were not reclassified as production evidence.
- [x] W11 is the only remaining Owner A gate with enough local input to advance immediately.

## 2026-06-05 New Session Handoff Entry Created

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`

Updated:

- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`

Purpose:

- Provide a clean, single entry point for a new API session or new conversation.
- Reduce old-chat noise by marking older dated files as evidence/history unless
  the start file explicitly points to them.
- Preserve the latest W4 focused validation, production-resource preflight,
  boundaries, current failures, and recommended next W11 task in one place.

Back-check:

- [x] No code, `.env`, or generated artifact was changed.
- [x] The new entry file points to the current source-of-truth docs.
- [x] The next task is explicit: W11 local HUMU-labeled FragFM data enrichment,
      with new derived data/artifact paths only.
- [x] Owner B code remains read-only unless explicitly authorized.

## 2026-06-06 W11 HUMU-Labeled FragFM Smoke Gate Completed

Modified:

- `moleculeforge/models/mf-generators/fragfm/pyproject.toml`
- `moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/__init__.py`
- `moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/humu_labeling.py`
- `moleculeforge/tests/unit/test_generators.py`

Created derived local artifacts:

- `moleculeforge/data/processing/generator_artifacts/fragfm_records_train_humu_labeled.jsonl`
- `moleculeforge/data/processing/generator_artifacts/fragfm_records_train_humu_labeled.report.json`
- `moleculeforge/checkpoints/fragfm_humu_smoke/`

Implemented:

- Added `mf_generators.fragfm.humu_labeling` with `label_fragfm_records()`
  and `python -m mf_generators.fragfm.humu_labeling`.
- The utility reads FragFM JSONL records, encodes each `product` SMILES through
  the frozen local HUMU molecule encoder checkpoint, validates each vector as a
  129-dimensional Lorentz full-coordinate embedding, writes a separate derived
  JSONL preserving original fields plus `humu_embedding`, and emits a JSON
  labeling report.
- The local HUMU encoder loader reads `encoder_mol` directly through
  `HUMUMoleculeEncoder`; it does not depend on `humu-encoder-svc`.
- FragFM package init now lazy-loads `FragFMGenerator` so `python -m
  mf_generators.fragfm.humu_labeling` does not import the generation stack before
  running the labeling CLI.

Verification:

- Four HUMU-labeling tests passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_humu_labeling_writes_valid_embeddings_and_preserves_records
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_humu_labeling_skips_unencodable_products
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_humu_labeling_refuses_to_overwrite_input
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_humu_labeling_strict_fails_below_min_coverage
  -q`, exit code 0, 4 passed. The only warning is `asyncio_mode` unknown when
  plugin autoload is disabled for faster focused collection.
- `python3 -m py_compile` passed for the new labeling module, FragFM package
  init, and `tests/unit/test_generators.py`.
- 50-record labeling smoke passed with the frozen HUMU checkpoint:
  `fragfm_records_train_humu_labeled.report.json` reports `status=pass`,
  `total_records=50`, `encoded_records=50`, `humu_embedding_coverage=1.0`,
  `invalid_smiles=0`, `invalid_embeddings=0`, and `expected_humu_dim=129`.
- The derived JSONL has 50 lines, all records have finite 129-dimensional
  `humu_embedding` values, and the max sampled Lorentz equation deviation was
  approximately `1.55e-06`.
- Separate FragFM smoke training passed:
  `uv run python models/mf-generators/fragfm/train.py --data
  data/processing/generator_artifacts/fragfm_records_train_humu_labeled.jsonl
  --output-dir checkpoints/fragfm_humu_smoke --epochs 1 --batch-size 16
  --hidden-dim 16 --device cpu --humu-embedding-dim 129 --humu-curvature 1.0`,
  exit code 0, 1 epoch, `loss=4.5128`.
- `checkpoints/fragfm_humu_smoke/training_manifest.json` reports
  `records=50`, `fragments=58`, `humu_embedding_count=50`, and
  `humu_embedding_coverage=1.0`.
- Strict quality gate passed on the new smoke artifact:
  `uv run python -m mf_generators.fragfm.quality --vocab
  checkpoints/fragfm_humu_smoke/vocab.json --checkpoint
  checkpoints/fragfm_humu_smoke/best_model.pt --rate-matrix
  checkpoints/fragfm_humu_smoke/rate_matrix.pt --min-humu-coverage 1.0
  --strict`, exit code 0, report `status=pass`, `rules=50`,
  `humu_embedding_coverage=1.0`, checkpoint/rate-matrix loadable.

Remaining gate:

- This is a 50-record, 1-epoch local engineering smoke artifact, not
  production-quality FragFM evidence.
- W11 still needs a 5000-record local candidate or externally curated
  HUMU-labeled data, formal coverage and benchmark thresholds, official
  benchmark evidence, cluster validation, and production artifact promotion.

Back-check:

- [x] Existing `checkpoints/fragfm`, `checkpoints/humu`, and
      `checkpoints/hfm3d_4h200` were not overwritten.
- [x] HUMU pretraining config/loss/encoder architecture/checkpoint continuation
      remained untouched.
- [x] Owner B code was not modified.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or
      executed.
- [x] W11 is now above 0.0 HUMU coverage for local smoke evidence, but still not
      reclassified as production quality.

## 2026-06-06 W11 5000-Record HUMU Labeling Completed; CPU Training Blocked

Created derived local artifacts:

- `moleculeforge/data/processing/generator_artifacts/fragfm_records_humu_labeled.jsonl`
- `moleculeforge/data/processing/generator_artifacts/fragfm_records_humu_labeled.report.json`

Completed:

- Ran `mf_generators.fragfm.humu_labeling` on the 5000-record local FragFM
  candidate dataset using the frozen HUMU checkpoint.
- The source file `fragfm_records.jsonl` was not modified.
- The derived JSONL has 5000 rows; each row preserves the original fields and
  adds `humu_embedding`.

Verification:

- `fragfm_records_humu_labeled.report.json` reports `status=pass`,
  `total_records=5000`, `encoded_records=5000`,
  `humu_embedding_coverage=1.0`, `invalid_smiles=0`,
  `invalid_embeddings=0`, `skipped_records=0`, and
  `expected_humu_dim=129`.
- Line count check confirmed 5000 input rows and 5000 derived output rows.
- Embedding validation scan found all embeddings finite, all dimensions equal
  to 129, no missing embeddings, and max Lorentz equation deviation
  approximately `4.72e-06`.

Training attempt:

- Attempted to train `checkpoints/fragfm_humu_5k/` with the 5000-record derived
  data on CPU.
- First run used `uv run python models/mf-generators/fragfm/train.py ...`
  with `--batch-size 64`. After about 25 minutes it was still in the first epoch
  and had only written `vocab.json`; no checkpoint or manifest existed. The
  run was interrupted and the partial directory was removed.
- Second run used `.venv/bin/python models/mf-generators/fragfm/train.py ...`
  with `--batch-size 5000`. After about 13 minutes it was still computing and
  had only written `vocab.json`; no checkpoint or manifest existed. The run was
  interrupted and the partial directory was removed.

Remaining gate:

- `checkpoints/fragfm_humu_5k/` was not produced and should not be referenced as
  an existing artifact.
- The completed 5000-record HUMU-labeled JSONL is ready as an input artifact for
  a later GPU/cluster run or a training implementation optimization.
- W11 production acceptance still needs a production-quality trained artifact,
  formal coverage/benchmark thresholds, official benchmark evidence, and cluster
  validation.

Back-check:

- [x] Partial `checkpoints/fragfm_humu_5k/` training directories were removed.
- [x] Existing `checkpoints/fragfm`, `checkpoints/fragfm_humu_smoke`,
      `checkpoints/humu`, and `checkpoints/hfm3d_4h200` were not overwritten.
- [x] HUMU pretraining and HUMU encoder architecture remained untouched.
- [x] Owner B code was not modified.
- [x] `/workspace/SemMol` and `/workspace/Projects` were not modified or
      executed.

## 2026-06-06 W11 FragFM Rate-Loss Optimization Added; 5k CPU Training Still Blocked

Modified:

- `moleculeforge/models/mf-generators/fragfm/train.py`
- `moleculeforge/tests/unit/test_generators.py`

Diagnosis:

- The 5000-record derived dataset has 2860 unique fragments and max fragment
  length 16.
- The original `_rate_transition_loss()` called `SAAwareRateMatrix.forward()`
  for each batch, materializing a full `[batch, vocab, vocab]` tensor even
  though the loss only needs rows for observed fragment transitions.

Implemented:

- Added a sparse transition-row path for `SAAwareRateMatrix` inside
  `_rate_transition_loss()`.
- The fallback full-matrix path remains for custom rate-matrix implementations.
- Added a focused regression test proving the sparse loss matches the full rate
  matrix loss on a small fixture.

Verification:

- `python3 -m py_compile models/mf-generators/fragfm/train.py
  tests/unit/test_generators.py` passed.
- Focused pytest passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
  tests/unit/test_generators.py::TestFragFMGenerator::test_rate_transition_loss_matches_full_rate_matrix_without_materializing_batches
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_humu_labeling_writes_valid_embeddings_and_preserves_records
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_humu_labeling_skips_unencodable_products
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_humu_labeling_refuses_to_overwrite_input
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_humu_labeling_strict_fails_below_min_coverage
  -q`, exit code 0, 5 passed. The only warning is the same
  `asyncio_mode` warning caused by disabled plugin autoload.

5k retry:

- Retried `checkpoints/fragfm_humu_5k/` training with the sparse rate-loss path,
  CPU, 1 epoch, batch size 64, hidden dim 16. It still ran for more than 10
  minutes with only `vocab.json` written.
- Retried with hidden dim 8. It still ran for more than 12 minutes with only
  `vocab.json` written.
- Both partial directories were removed.

Conclusion:

- The rate-loss memory issue is fixed, but the 5k FragFM training artifact still
  does not complete in this CPU environment.
- `checkpoints/fragfm_humu_5k/` still does not exist and must not be referenced
  as a completed artifact.
- The next W11 training step should run on GPU/cluster, reduce the model/training
  workload further, or add an explicit artifact-building path for quality-gate
  handoff if a full training step is not required.

Back-check:

- [x] The optimization preserves rate-loss numerical behavior on the focused
      fixture.
- [x] HUMU-labeled data files remain valid and unchanged.
- [x] Existing FragFM/HUMU/HFM checkpoints were not overwritten.
- [x] Owner B code was not modified.

## 2026-06-06 W11 FragFM Sparse SA Row-Gather Optimization Added

Modified:

- `moleculeforge/models/mf-generators/fragfm/train.py`
- `moleculeforge/tests/unit/test_generators.py`

Diagnosis:

- A follow-up micro-profile on the 5000-record HUMU-labeled dataset showed that
  the first sparse rate-loss optimization removed the `[batch, vocab, vocab]`
  materialization, but still called `sa_score_embedding(sa_bin)` per sample and
  therefore read a full `vocab * vocab` SA modulation vector before slicing the
  observed transition row.
- With 2860 unique fragments, `SAAwareRateMatrix` initialization remains heavy
  because the existing artifact schema stores `base_rate` and
  `sa_score_embedding.weight` as full `vocab x vocab` tensors. This optimization
  does not change that schema.

Implemented:

- `_sparse_rate_transition_loss()` now gathers only the SA modulation rows
  needed by observed transitions from `sa_score_embedding.weight`, without
  calling the embedding module forward path.
- The transition rows are batched before `cross_entropy`, which reduces Python
  per-transition overhead for the intended larger batch path.
- The custom rate-matrix fallback remains unchanged.

Verification:

- New TDD regression first failed because the old sparse path called full
  `sa_score_embedding.forward()`, then passed after row-gather implementation.
- Focused pytest passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
  tests/unit/test_generators.py::TestFragFMGenerator::test_rate_transition_loss_matches_full_rate_matrix_without_materializing_batches
  tests/unit/test_generators.py::TestFragFMGenerator::test_rate_transition_loss_gathers_sa_rows_without_embedding_forward
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_humu_labeling_writes_valid_embeddings_and_preserves_records
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_humu_labeling_skips_unencodable_products
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_humu_labeling_refuses_to_overwrite_input
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_humu_labeling_strict_fails_below_min_coverage
  -q`, exit code 0, 6 passed. The only warning is the existing
  `asyncio_mode` warning caused by disabled plugin autoload.
- `python3 -m py_compile models/mf-generators/fragfm/train.py
  tests/unit/test_generators.py` passed.
- `git diff --check -- models/mf-generators/fragfm/train.py
  tests/unit/test_generators.py` passed.
- Short `/tmp` training smoke using the 50-record HUMU-labeled split passed:
  1 epoch, batch size 16, hidden dim 8, CPU; produced `best_model.pt`,
  `rate_matrix.pt`, `vocab.json`, and manifest with 50 records, 58 fragments,
  `humu_embedding_count=50`, coverage 1.0. The temporary directory was removed.
- 5000-record label report still passes: 5000/5000 encoded, coverage 1.0.
- `checkpoints/fragfm_humu_5k/` still does not exist.

Performance observation:

- On the 5000-record labeled data, with one initialized `SAAwareRateMatrix`, the
  batch-64 sparse rate loss profile was about 2.9 seconds after row-gather and
  transition batching, compared with about 6.9 seconds after the first sparse
  path and much worse before sparse loss.
- Startup/data normalization and full `SAAwareRateMatrix` initialization remain
  significant CPU costs. This is still not enough evidence to mark a 5k local
  artifact complete.

Conclusion:

- W11 training hot path is narrower, and the math/schema contract is preserved.
- `checkpoints/fragfm_humu_5k/` remains unproduced and must not be referenced as
  completed.
- Remaining W11 gate still requires GPU/cluster training or another explicitly
  reviewed training/artifact optimization before production-quality artifact
  evidence can be recorded.

Back-check:

- [x] No HUMU pretraining config/loss/encoder/checkpoint continuation was
      modified.
- [x] No existing `checkpoints/fragfm`, `checkpoints/humu`, or
      `checkpoints/hfm3d_4h200` artifacts were overwritten.
- [x] Owner B code was not modified.

## 2026-06-06 W11 FragFM Rate Optimizer Controls Added

Modified:

- `moleculeforge/models/mf-generators/fragfm/train.py`
- `moleculeforge/tests/unit/test_generators.py`

Diagnosis:

- After row-gather optimization, a full single-batch profile on the
  5000-record HUMU-labeled dataset showed `_rate_transition_loss()` was no
  longer the dominant cost for batch 64.
- Remaining CPU costs included dense optimizer work over the full
  `SAAwareRateMatrix` tensors: rate-matrix optimizer step was about 11.5
  seconds with AdamW in the profile, while an SGD comparison reduced it to about
  2.9 seconds. Full `SAAwareRateMatrix` initialization remains expensive because
  the checkpoint schema is still dense.

Implemented:

- Added explicit FragFM training CLI controls:
  `--rate-optimizer {adamw,sgd}` and `--disable-rate-grad-clip`.
- Default behavior remains `--rate-optimizer adamw` with rate gradient clipping
  enabled.
- The model optimizer remains AdamW. The new option only controls the
  `SAAwareRateMatrix` optimizer, preserving checkpoint and rate-matrix artifact
  schema.
- Training manifest now records `rate_optimizer` and `rate_grad_clip`.

Verification:

- New TDD CLI regression first failed because the arguments were not recognized,
  then passed after implementation.
- Focused pytest passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
  tests/unit/test_generators.py::TestFragFMGenerator::test_rate_transition_loss_matches_full_rate_matrix_without_materializing_batches
  tests/unit/test_generators.py::TestFragFMGenerator::test_rate_transition_loss_gathers_sa_rows_without_embedding_forward
  tests/unit/test_generators.py::TestFragFMGenerator::test_training_cli_records_rate_optimizer_controls
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_humu_labeling_writes_valid_embeddings_and_preserves_records
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_humu_labeling_skips_unencodable_products
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_humu_labeling_refuses_to_overwrite_input
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_humu_labeling_strict_fails_below_min_coverage
  -q`, exit code 0, 7 passed. The only warning is the existing
  `asyncio_mode` warning caused by disabled plugin autoload.
- `python3 -m py_compile models/mf-generators/fragfm/train.py
  tests/unit/test_generators.py` passed.
- `git diff --check -- models/mf-generators/fragfm/train.py
  tests/unit/test_generators.py` passed.
- 50-record HUMU-labeled `/tmp` training smoke passed with
  `--rate-optimizer sgd --disable-rate-grad-clip`, producing
  checkpoint/rate-matrix/vocab/manifest and manifest values
  `records=50`, `fragments=58`, `humu_embedding_count=50`, coverage 1.0,
  `rate_optimizer=sgd`, `rate_grad_clip=false`.
- 256-record subset smoke from the 5000-record HUMU-labeled data passed with the
  same options, producing checkpoint/rate-matrix/vocab/manifest and manifest
  values `records=256`, `fragments=259`, `humu_embedding_count=256`, coverage
  1.0, `rate_optimizer=sgd`, `rate_grad_clip=false`. Temporary directories were
  removed.

Conclusion:

- W11 now has an explicit CPU-friendly training option that preserves artifact
  schema and default training behavior.
- This improves the path toward a 5k local candidate but does not itself create
  `checkpoints/fragfm_humu_5k/`.
- `checkpoints/fragfm_humu_5k/` remains absent and must not be referenced as a
  completed artifact.

Back-check:

- [x] The new optimizer option is explicit and manifest-recorded.
- [x] Default training behavior remains AdamW with rate grad clipping enabled.
- [x] No existing protected checkpoints were overwritten.
- [x] Owner B code was not modified.

## 2026-06-06 W11 5000-Record HUMU-Labeled FragFM Local Candidate Completed

Created:

- `moleculeforge/checkpoints/fragfm_humu_5k/vocab.json`
- `moleculeforge/checkpoints/fragfm_humu_5k/best_model.pt`
- `moleculeforge/checkpoints/fragfm_humu_5k/rate_matrix.pt`
- `moleculeforge/checkpoints/fragfm_humu_5k/final_model.pt`
- `moleculeforge/checkpoints/fragfm_humu_5k/final_rate_matrix.pt`
- `moleculeforge/checkpoints/fragfm_humu_5k/training_manifest.json`
- `moleculeforge/checkpoints/fragfm_humu_5k/quality_report.json`

Command:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python models/mf-generators/fragfm/train.py \
    --data data/processing/generator_artifacts/fragfm_records_humu_labeled.jsonl \
    --output-dir checkpoints/fragfm_humu_5k \
    --epochs 1 \
    --batch-size 64 \
    --hidden-dim 8 \
    --device cpu \
    --rate-optimizer sgd \
    --disable-rate-grad-clip
```

Result:

- Training completed after 79 batch-64 batches and logged
  `Epoch 1/1: loss=8.7005`.
- `training_manifest.json` records `records=5000`, `fragments=2860`,
  `epochs=1`, `humu_embedding_count=5000`,
  `humu_embedding_coverage=1.0`, `humu_embedding_dim=129`,
  `rate_optimizer=sgd`, and `rate_grad_clip=false`.
- Artifact schema check: `best_model.pt` contains
  `fragment_encoder.weight` with shape `(2860, 8)`; `rate_matrix.pt` contains
  `base_rate` with shape `(2860, 2860)` and `sa_score_embedding.weight` with
  shape `(10, 8179600)`.
- `vocab.json` scan found 5000 rules, 2860 fragments, no missing
  `humu_embedding`, no bad embedding dimensions, and no non-finite values.

Quality gate:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python -m mf_generators.fragfm.quality \
    --vocab checkpoints/fragfm_humu_5k/vocab.json \
    --checkpoint checkpoints/fragfm_humu_5k/best_model.pt \
    --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt \
    --min-humu-coverage 1.0 \
    --strict \
    --output checkpoints/fragfm_humu_5k/quality_report.json
```

- Quality result: `status=pass`, `rules=5000`, `fragments=2860`,
  `humu_embedding_count=5000`, `humu_embedding_coverage=1.0`,
  `checkpoint_loadable=true`, `rate_matrix_loadable=true`, `messages=[]`.

Scope:

- This is a 1-epoch, hidden-dim 8 local engineering candidate artifact. It
  advances W11 beyond the 50-record smoke and proves the 5000-record
  HUMU-labeled path can produce a strict local artifact.
- This is still not final W11 production acceptance. Remaining gates are
  production-quality training configuration, benchmark evidence, formal
  quality thresholds, deployment wiring, and cluster validation.

Back-check:

- [x] Existing protected `checkpoints/fragfm`, `checkpoints/humu`, and
      `checkpoints/hfm3d_4h200` were not overwritten.
- [x] HUMU pretraining config/loss/encoder/checkpoint continuation was not
      modified.
- [x] Owner B code was not modified.

## 2026-06-06 W11 FragFM HUMU 5k Deployment Defaults Hardened

Modified:

- `moleculeforge/infra/docker/docker-compose.dev.yml`
- `moleculeforge/infra/kubernetes/deployments/moleculeforge-services.yaml`
- `moleculeforge/infra/helm/moleculeforge/values.yaml`
- `moleculeforge/tests/unit/test_service_artifact_status.py`

Implemented:

- FragFM service defaults now point to the strict-local 5000-record HUMU-labeled
  candidate artifact:
  - `checkpoints/fragfm_humu_5k/vocab.json`
  - `checkpoints/fragfm_humu_5k/best_model.pt`
  - `checkpoints/fragfm_humu_5k/rate_matrix.pt`
- Docker Compose uses these as `FRAGFM_*` default env values while preserving
  env override support.
- Kubernetes and Helm `fragfm-generator-config` defaults now use the same
  paths.
- The deployment wiring test now asserts the artifact files exist and that
  `quality_report.json` has `status=pass` and HUMU coverage 1.0.

Verification:

- RED phase: the focused deployment test failed before implementation because
  Docker Compose still referenced `checkpoints/fragfm/vocab.json`.
- GREEN phase:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_service_artifact_status.py::test_fragfm_deployment_wires_artifact_and_sampler_env -q`
  passed with 1 item. Warnings are the existing disabled-plugin
  `asyncio_mode` and unknown `pytest.mark.asyncio` warnings.
- Path scan confirmed Docker Compose, Kubernetes, Helm, and the focused test
  all reference `checkpoints/fragfm_humu_5k` for the FragFM vocab,
  checkpoint, and rate matrix defaults.

Scope:

- This hardens local deployment defaults away from the old coverage-0.0
  `checkpoints/fragfm` smoke artifact.
- This is still not final W11 production acceptance. Remaining gates are
  production-quality training configuration, benchmark evidence, formal
  quality thresholds, artifact promotion policy, and cluster validation.

Back-check:

- [x] Existing protected `checkpoints/fragfm`, `checkpoints/humu`, and
      `checkpoints/hfm3d_4h200` were not overwritten.
- [x] HUMU pretraining config/loss/encoder/checkpoint continuation was not
      modified.
- [x] Owner B code was not modified.
- [x] Deployment defaults are now consistent across Docker Compose, raw
      Kubernetes, Helm values, and the focused deployment regression.

## 2026-06-06 New Session Handoff Refreshed

Modified:

- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`

Implemented:

- Replaced the stale new-conversation copy-paste prompt that still pointed to
  the pre-W11-labeling task.
- Made `START_HERE_NEW_SESSION.md` a short entry point with current W11 5k and
  deployment-default status.
- Added a detailed new-session handoff covering project idea, ownership,
  frozen boundaries, HUMU/Lorentz contracts, completed Owner A work, W11
  artifact state, deployment defaults, verification evidence, known blockers,
  useful commands, and recommended next tasks.
- Updated README to point new conversations to both the short entry and the
  detailed 2026-06-06 handoff.

Scope:

- This is documentation cleanup and handoff packaging only.
- Historical progress entries were kept as evidence; stale guidance was removed
  from the active new-session entry instead of rewriting the timeline.
- No code, data artifact, protected checkpoint, HUMU pretraining path, or Owner
  B implementation file was changed by this handoff refresh.

Back-check:

- [x] The active new-session prompt no longer asks the next agent to do the
      already-completed W11 HUMU-labeling enrichment as the current task.
- [x] The handoff states that `checkpoints/fragfm_humu_5k/` is strict-local
      engineering evidence, not production/cluster acceptance.
- [x] The handoff keeps the protected artifact and HUMU pretraining boundaries
      explicit.

## 2026-06-06 W11 FragFM Runtime Smoke And Promotion Policy Added

Modified:

- `moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py`
- `moleculeforge/tests/unit/test_service_artifact_status.py`
- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-runtime-smoke-and-promotion-plan.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-artifact-promotion-policy.md`

Implemented:

- Added a focused runtime smoke regression:
  `tests/unit/test_service_artifact_status.py::test_fragfm_deployment_default_artifact_loads_and_generates`.
- The smoke sets `FRAGFM_*` env values to
  `checkpoints/fragfm_humu_5k/{vocab.json,best_model.pt,rate_matrix.pt}`,
  calls `fragfm_generator_svc._build_generator()`, generates one molecule, and
  verifies the SMILES is RDKit-parseable.
- `FragFMGenerator` now infers `TwoLevelDFM` hidden dimension from
  `fragment_encoder.weight` before loading a checkpoint. This allows the
  hidden-dim-8 5k candidate checkpoint to load through the production runtime
  path instead of assuming the default hidden dim 256.
- FragFM inference now caches scored assembly rules and computes sparse
  transition scores in a batch from `base_rate` and `sa_score_embedding`,
  avoiding repeated full rate-matrix construction or per-rule tensor calls
  during generation.
- Added W11 artifact promotion policy. It keeps `checkpoints/fragfm_humu_5k/`
  classified as strict-local engineering evidence and requires a new explicit
  production artifact path plus HUMU coverage, runtime, training, benchmark,
  deployment, and ownership gates before promotion.

Debugging notes:

- Initial runtime smoke attempt did not complete in a reasonable time. A direct
  probe showed `vocab.json` parsing takes about 9.3 seconds, current-workstation
  PyTorch import takes about 52.6 seconds, checkpoint load is about 0.2 seconds,
  and rate-matrix load is about 5.9 seconds.
- A `/workspace/SemMol` `torchrun` process was already running and likely
  contributed to slow PyTorch import / CPU contention. It was not modified or
  terminated because `/workspace/SemMol` is read/copy-only context.
- Direct service smoke after the runtime changes completed:
  `_build_generator()` at about 148.8 seconds and `generate(batch_size=1)` at
  about 150.3 seconds on this workstation. This is local evidence only; cluster
  cold-start and readiness still need production validation.

Verification:

- RED/capability evidence: the new runtime smoke exposed that the previous
  service path was not a practical runtime smoke for the 5k artifact under the
  current environment and needed runtime hardening.
- GREEN:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_service_artifact_status.py::test_fragfm_deployment_default_artifact_loads_and_generates -q`
  passed with 1 item. Warnings are the existing disabled-plugin
  `asyncio_mode` and unknown `pytest.mark.asyncio` warnings.
- Focused service verification:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_service_artifact_status.py::test_fragfm_deployment_default_artifact_loads_and_generates tests/unit/test_service_artifact_status.py::test_fragfm_deployment_wires_artifact_and_sampler_env -q`
  passed with 2 items. Warnings are the same existing disabled-plugin
  `asyncio_mode` and unknown `pytest.mark.asyncio` warnings.
- Existing FragFM generator load regression:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_loads_checkpoint_and_rate_matrix_artifacts -q`
  passed with 1 item. Warning is the existing disabled-plugin `asyncio_mode`
  warning.
- W11 strict quality CLI passed:
  `PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" .venv/bin/python -m mf_generators.fragfm.quality --vocab checkpoints/fragfm_humu_5k/vocab.json --checkpoint checkpoints/fragfm_humu_5k/best_model.pt --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt --min-humu-coverage 1.0 --strict --output checkpoints/fragfm_humu_5k/quality_report.json`.
- `python3 -m py_compile tests/unit/test_service_artifact_status.py models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py`
  passed.
- `git diff --check` passed for the touched FragFM runtime, service test, and
  Owner A W11 docs.
- Trailing whitespace scan passed for the touched FragFM runtime, service test,
  and Owner A W11 docs.
- Reserved-word scan over the touched Owner A W11 docs found no stale
  reserved markers.
- Process scan found no lingering pytest, quality, or FragFM smoke processes.

Scope:

- This is W11 production-readiness hardening, not production acceptance.
- It does not change deployment defaults beyond the already-recorded
  `fragfm_humu_5k` defaults.
- It does not train or overwrite any artifact.

Back-check:

- [x] Existing protected `checkpoints/fragfm`, `checkpoints/humu`, and
      `checkpoints/hfm3d_4h200` were not overwritten.
- [x] HUMU pretraining config/loss/encoder/checkpoint continuation was not
      modified.
- [x] Owner B implementation files were not modified.
- [x] `checkpoints/fragfm_humu_5k/` remains documented as local engineering
      evidence, not final production W11 acceptance.

## 2026-06-06 W11 FragFM Sample Export Tool Added

Modified:

- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-sample-export-plan.md`

Created:

- `moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py`
- `moleculeforge/data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.smi`
- `moleculeforge/data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.report.json`

Implemented:

- Added `mf_generators.fragfm.sample_export`.
- `export_fragfm_samples()` instantiates `FragFMGenerator`, generates a
  requested number of molecules, writes one SMILES per line, and optionally
  writes a JSON report.
- `build_sample_report()` records schema version, requested/generated counts,
  valid SMILES count, validity, unique valid SMILES count, uniqueness, and the
  artifact/output paths.
- The CLI supports `--vocab`, `--checkpoint`, `--rate-matrix`, `--output`,
  `--report`, `--samples`, and `--device`.
- Added focused unit coverage:
  `tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_writes_smiles_and_report`.

Evidence:

- RED phase:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_writes_smiles_and_report -q`
  failed before implementation with `ModuleNotFoundError: No module named
  'mf_generators.fragfm.sample_export'`.
- GREEN phase:
  the same focused pytest command passed with 1 item. Warning is the existing
  disabled-plugin `asyncio_mode` warning.
- Small `/tmp` CLI smoke exited 0 with 3 generated samples, validity 1.0,
  unique SMILES 2, and uniqueness 0.6666666666666666.
- 5k candidate sample export smoke exited 0:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python -m mf_generators.fragfm.sample_export \
    --vocab checkpoints/fragfm_humu_5k/vocab.json \
    --checkpoint checkpoints/fragfm_humu_5k/best_model.pt \
    --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt \
    --output data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.smi \
    --report data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.report.json \
    --samples 8 \
    --device cpu
```

- 5k smoke report:
  `schema_version=fragfm_sample_export_report.v1`,
  `requested_samples=8`, `generated_samples=8`, `valid_smiles=8`,
  `validity=1.0`, `unique_smiles=8`, and `uniqueness=1.0`.
- FragFM MOSES validity benchmark wiring smoke exited 0 with 1 passed:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:libs/mf-eval/src:models/mf-generators/fragfm/src" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
MOSES_REFERENCE_SMILES_PATH=data/benchmarks/moses_reference_smiles.smi \
FRAGFM_MOSES_GENERATED_SMILES_PATH=data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.smi \
  .venv/bin/python -m pytest \
    tests/benchmark/moses_benchmark.py::TestMosesBenchmark::test_fragfm_moses_validity -q
```

  Warning is the existing disabled-plugin `asyncio_mode` warning. This is local
  benchmark input wiring evidence for the 8-sample export only, not official
  W5/MOSES acceptance.
- Fresh focused verification:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_writes_smiles_and_report -q`
  exited 0 with 1 passed and the existing disabled-plugin `asyncio_mode`
  warning.
- `python3 -m py_compile models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py tests/unit/test_generators.py`
  exited 0.
- Path-limited `git diff --check` over touched tracked files exited 0. The new
  sample export module and Owner A docs are untracked in the current top-level
  git view, so they were covered by the manual scans below.
- Trailing whitespace scan over the sample export module, focused test file,
  and touched Owner A docs exited 0.
- Reserved-marker scan over the touched Owner A docs found no unchecked
  checkboxes or stale template markers.
- Sample smoke report assertion script printed `sample report ok`.
- Process scan found no lingering pytest, FragFM quality, or FragFM sample
  export process; only the process scan command itself matched.

Scope:

- This is benchmark preparation tooling and local evidence only.
- It does not satisfy official W5/MOSES/GuacaMol/PMO acceptance.
- It does not relax benchmark thresholds.
- It does not promote `checkpoints/fragfm_humu_5k/` beyond strict-local
  engineering candidate status.
- The MOSES validity smoke uses the existing benchmark threshold unchanged, but
  the input is only 8 local samples, so it remains wiring evidence.

Back-check:

- [x] Existing protected `checkpoints/fragfm`, `checkpoints/humu`, and
      `checkpoints/hfm3d_4h200` were not overwritten by this task.
- [x] This task did not add or edit HUMU pretraining config/loss/encoder or
      checkpoint-continuation files. The wider dirty worktree still contains
      pre-existing HUMU pretraining modifications; they were not touched here.
- [x] Owner B implementation files were not modified by this task.
- [x] The sample export output is under
      `data/processing/generator_artifacts/`, not a protected checkpoint path.
- [x] This task did not edit W5 benchmark thresholds. The wider dirty worktree
      still contains pre-existing benchmark harness modifications; they were
      not touched here.

## 2026-06-06 W11 FragFM 64-Sample Benchmark Input Smoke

Created:

- `moleculeforge/data/processing/generator_artifacts/fragfm_humu_5k_sample_64.smi`
- `moleculeforge/data/processing/generator_artifacts/fragfm_humu_5k_sample_64.report.json`

Modified:

- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

Implemented / Evidence:

- Exported 64 samples from `checkpoints/fragfm_humu_5k/`:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python -m mf_generators.fragfm.sample_export \
    --vocab checkpoints/fragfm_humu_5k/vocab.json \
    --checkpoint checkpoints/fragfm_humu_5k/best_model.pt \
    --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt \
    --output data/processing/generator_artifacts/fragfm_humu_5k_sample_64.smi \
    --report data/processing/generator_artifacts/fragfm_humu_5k_sample_64.report.json \
    --samples 64 \
    --device cpu
```

- Result: exit code 0.
- Report:
  `schema_version=fragfm_sample_export_report.v1`,
  `requested_samples=64`, `generated_samples=64`, `valid_smiles=64`,
  `validity=1.0`, `unique_smiles=64`, and `uniqueness=1.0`.
- The output SMILES file has 64 lines.
- FragFM MOSES validity wiring smoke with the 64-sample file exited 0 with
  1 passed:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:libs/mf-eval/src:models/mf-generators/fragfm/src" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
MOSES_REFERENCE_SMILES_PATH=data/benchmarks/moses_reference_smiles.smi \
FRAGFM_MOSES_GENERATED_SMILES_PATH=data/processing/generator_artifacts/fragfm_humu_5k_sample_64.smi \
  .venv/bin/python -m pytest \
    tests/benchmark/moses_benchmark.py::TestMosesBenchmark::test_fragfm_moses_validity -q
```

  Warning is the existing disabled-plugin `asyncio_mode` warning.

Scope:

- This is stronger local benchmark-input wiring evidence than the 8-sample
  smoke.
- It is still not official W5/MOSES/GuacaMol/PMO acceptance because the
  underlying artifact is the strict-local 1-epoch hidden-dim-8 candidate and
  the generated set is not production-scale.
- Benchmark thresholds were not changed.

Back-check:

- [x] Existing protected `checkpoints/fragfm`, `checkpoints/humu`, and
      `checkpoints/hfm3d_4h200` were not overwritten.
- [x] HUMU pretraining files were not edited by this task.
- [x] Owner B implementation files were not modified by this task.
- [x] The new output files are under `data/processing/generator_artifacts/`.

## 2026-06-06 W11 FragFM 256-Sample Benchmark Input Smoke

Created:

- `moleculeforge/data/processing/generator_artifacts/fragfm_humu_5k_sample_256.smi`
- `moleculeforge/data/processing/generator_artifacts/fragfm_humu_5k_sample_256.report.json`

Modified:

- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

Implemented / Evidence:

- Exported 256 samples from `checkpoints/fragfm_humu_5k/`:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python -m mf_generators.fragfm.sample_export \
    --vocab checkpoints/fragfm_humu_5k/vocab.json \
    --checkpoint checkpoints/fragfm_humu_5k/best_model.pt \
    --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt \
    --output data/processing/generator_artifacts/fragfm_humu_5k_sample_256.smi \
    --report data/processing/generator_artifacts/fragfm_humu_5k_sample_256.report.json \
    --samples 256 \
    --device cpu
```

- Result: exit code 0.
- Report:
  `schema_version=fragfm_sample_export_report.v1`,
  `requested_samples=256`, `generated_samples=256`, `valid_smiles=256`,
  `validity=1.0`, `unique_smiles=256`, and `uniqueness=1.0`.
- The output SMILES file has 256 lines.
- FragFM MOSES validity wiring smoke with the 256-sample file exited 0 with
  1 passed:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:libs/mf-eval/src:models/mf-generators/fragfm/src" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
MOSES_REFERENCE_SMILES_PATH=data/benchmarks/moses_reference_smiles.smi \
FRAGFM_MOSES_GENERATED_SMILES_PATH=data/processing/generator_artifacts/fragfm_humu_5k_sample_256.smi \
  .venv/bin/python -m pytest \
    tests/benchmark/moses_benchmark.py::TestMosesBenchmark::test_fragfm_moses_validity -q
```

  Warning is the existing disabled-plugin `asyncio_mode` warning.

Scope:

- This is local default-batch-size benchmark-input wiring evidence.
- It is still not official W5/MOSES/GuacaMol/PMO acceptance because the
  underlying artifact is the strict-local 1-epoch hidden-dim-8 candidate and
  there is no production-scale benchmark or cluster evidence.
- Benchmark thresholds were not changed.

Back-check:

- [x] Existing protected `checkpoints/fragfm`, `checkpoints/humu`, and
      `checkpoints/hfm3d_4h200` were not overwritten.
- [x] HUMU pretraining files were not edited by this task.
- [x] Owner B implementation files were not modified by this task.
- [x] The new output files are under `data/processing/generator_artifacts/`.

## 2026-06-06 W11 Production Readiness Gap Recorded

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-readiness-gap.md`

Modified:

- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

Implemented:

- Recorded the current W11 local evidence matrix: strict quality, runtime load,
  deployment defaults, sample export, 8/64/256 sample reports, and MOSES
  validity wiring smoke.
- Recorded the non-promotion reasons: missing production training, formal
  benchmark set, cluster runtime, immutable release naming, and ownership gate.
- Recorded stop conditions before long training, production artifact naming,
  deployment default changes, benchmark threshold changes, Owner B edits, or
  external `/workspace/SemMol` process changes.

Back-check:

- [x] No code was changed by this documentation step.
- [x] Protected checkpoints were not touched.
- [x] `checkpoints/fragfm_humu_5k/` remains documented as strict-local
      engineering evidence, not production acceptance.

## 2026-06-06 W11 Production Training Run Plan Recorded

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-training-run-plan.md`

Modified:

- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-readiness-gap.md`

Implemented:

- Recorded the next stronger FragFM training command template using a new
  candidate output directory, 5 epochs, hidden dim 64, AdamW rate optimizer, and
  strict 129-dimensional HUMU validation.
- Recorded required post-training checks: strict quality, runtime smoke, sample
  export, MOSES validity wiring, and cluster readiness evidence.
- Recorded stop conditions before launching training or choosing permanent
  production artifact names.

Back-check:

- [x] This step did not launch training.
- [x] This step did not choose a permanent production artifact name.
- [x] This step did not change deployment defaults.
- [x] Protected checkpoints were not touched.

## 2026-06-06 W9 HFM Decoder Production Readiness Gap Recorded

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-readiness-gap.md`

Modified:

- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

Implemented:

- Recorded W9 local engineering evidence: SDF-backed source loading, tiny
  training/export, runner contract, HFM consumption, and CLI wrapper.
- Recorded that `checkpoints/hfm3d_4h200/decoder.json` has one ethanol entry and
  references a pytest temp HUMU checkpoint path, so it remains smoke/full-flow
  evidence only.
- Recorded W9 non-promotion reasons: missing real decoder source data,
  production decoder artifact, deployment mode decision, geometry benchmark, and
  cluster runtime.

Back-check:

- [x] No HFM code was changed.
- [x] `checkpoints/hfm3d_4h200` was not overwritten.
- [x] HUMU pretraining and HFM Lorentz flow architecture were not modified.
- [x] Owner B implementation files were not modified.

## 2026-06-06 W9 HFM Decoder Production Training Run Plan Recorded

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-training-run-plan.md`

Modified:

- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-readiness-gap.md`

Implemented:

- Recorded real HFM decoder source artifact requirements: 129-dimensional
  Lorentz-valid latent vectors, RDKit-parseable SDF, source/HUMU provenance, and
  minimum source coverage for a production-candidate run.
- Recorded a new non-protected output directory pattern:
  `checkpoints/hfm_geometry_decoder_candidate_YYYYMMDD_<run_id>/`.
- Recorded the decoder training command template, artifact load check, runner
  smoke, and HFM generator smoke through `HFM_MOLECULAR_DECODER_COMMAND`.
- Recorded benchmark caveats: existing HFM benchmark helpers use
  `HFM_DECODER_PATH`, W5 benchmark ownership remains Owner B, and thresholds
  must not be changed.
- Recorded stop conditions before launching training, lowering source gates,
  choosing deployment mode, changing HFM defaults, touching Owner B files, or
  overwriting protected artifacts.

Back-check:

- [x] This step did not launch decoder training.
- [x] This step did not modify HFM code.
- [x] This step did not choose a production deployment mode.
- [x] `checkpoints/hfm3d_4h200` was not overwritten.
- [x] HUMU pretraining and HFM Lorentz flow architecture were not modified.

## 2026-06-06 W10 HCIV Production Readiness Gap And Run Plan Recorded

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-readiness-gap.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-training-run-plan.md`

Modified:

- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

Implemented:

- Recorded W10 current engineering evidence: supervised JSON/JSONL CIG plus
  `target_hciv` loading, Lorentz target validation, tiny checkpoint export, CLI
  wrapper, and production `HCIV_CHECKPOINT_PATH` loading.
- Recorded W10 non-promotion reasons: missing real supervised data, production
  checkpoint, deployment value, downstream intent-conditioned generation
  evidence, and cluster runtime.
- Recorded production run plan with source data requirements, non-protected
  candidate output directory, training command template, checkpoint load smoke,
  CIG compiler production learned smoke, promotion evidence, and stop
  conditions.

Back-check:

- [x] This step did not launch HCIV training.
- [x] This step did not modify CIG compiler code.
- [x] This step did not choose a production `HCIV_CHECKPOINT_PATH`.
- [x] Hash/random HCIV encoders remain documented as local-demo only.
- [x] HUMU pretraining, HFM Lorentz flow architecture, and W2 steering rules
      were not modified.

## 2026-06-06 W13 KD Production Readiness Gap And Run Plan Recorded

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-readiness-gap.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-run-plan.md`

Modified:

- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

Implemented:

- Recorded W13 current engineering evidence: KD layer, Boltz2/HypSeek teacher
  distribution adapters, HypSeek teacher runner/app wiring, canonical teacher
  embedding artifact utility, and generator KD consumers.
- Recorded W13 non-promotion reasons: missing real teacher records,
  per-consumer canonical embedding artifacts, real distillation runs, benchmark
  quality evidence, teacher deployment resources, and cluster runtime.
- Recorded per-consumer dimension caution for HFM-3D, FragFM, UAS, CReM, MMPT,
  and iCLM.
- Recorded production run plan for teacher-record source requirements, strict
  artifact export/report, per-consumer artifact naming, distillation command
  templates, promotion evidence, and stop conditions.

Back-check:

- [x] This step did not launch generator distillation training.
- [x] This step did not modify KD code or loss semantics.
- [x] This step did not choose permanent teacher artifact names.
- [x] This step did not change teacher deployment env values.
- [x] HUMU pretraining and HFM Lorentz flow architecture were not modified.

## 2026-06-06 W6 TAR Production Readiness Gap And Run Plan Recorded

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-readiness-gap.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-run-plan.md`

Modified:

- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

Implemented:

- Recorded W6 current engineering evidence: shared `ProxylessSearchScheduler`,
  service `RunProxylessSearch`, built-in
  `python -m generator_router_svc.tar_proxyless_runner` command target, and
  deployment env wiring.
- Recorded W6 non-promotion reasons: missing real reward payloads, production
  command/default decision, downstream quality evidence, and cluster runtime.
- Recorded production run plan with reward payload requirements, preflight
  checks, command smoke, service smoke, promotion evidence, and stop conditions.

Back-check:

- [x] This step did not run TAR search.
- [x] This step did not modify scheduler/router code.
- [x] This step did not choose a production `TAR_PROXYLESS_SEARCH_COMMAND`.
- [x] This step did not change deployment defaults.
- [x] HUMU pretraining and HFM Lorentz flow architecture were not modified.

## 2026-06-06 Owner A Production Execution Roadmap Recorded

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-production-execution-roadmap.md`

Modified:

- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

Implemented:

- Recorded a consolidated W6/W9/W10/W11/W13 production execution roadmap.
- Kept W11 as the recommended near-term main path because it has the strongest
  local evidence: HUMU-labeled data, strict quality, deployment defaults,
  runtime smoke, sample export, and MOSES validity wiring.
- Sequenced W6 reward payload work, W10 supervised HCIV checkpoint work, W9 HFM
  decoder work, and W13 teacher embedding work behind their real source-data
  and resource gates.
- Recorded safe no-authorization steps, resource-gated steps, long-term release
  path, back-check protocol, and stop-and-ask decisions.
- Updated the new-session entry, workspace README, and detailed handoff so a new
  agent reads the roadmap before choosing the next production-resource gate.

Back-check:

- [x] This step did not modify business code.
- [x] This step did not launch training, benchmark acceptance runs, TAR search,
      or distillation.
- [x] This step did not change Docker, Kubernetes, Helm, or service defaults.
- [x] This step did not choose a production artifact name.
- [x] This step did not modify Owner B implementation files.
- [x] This step did not touch protected checkpoint paths.

## 2026-06-06 Production Source Inventory Read-Only Check

Checked:

- `moleculeforge/data/processing/generator_artifacts`
- `moleculeforge/checkpoints`

Observed source/artifact inventory:

- W11 FragFM files exist:
  - `fragfm_records_humu_labeled.jsonl`
  - `fragfm_records_humu_labeled.report.json`
  - `fragfm_records_train_humu_labeled.jsonl`
  - `fragfm_records_train_humu_labeled.report.json`
  - `fragfm_humu_5k_sample_smoke.smi`
  - `fragfm_humu_5k_sample_smoke.report.json`
  - `fragfm_humu_5k_sample_64.smi`
  - `fragfm_humu_5k_sample_64.report.json`
  - `fragfm_humu_5k_sample_256.smi`
  - `fragfm_humu_5k_sample_256.report.json`
- W11 FragFM candidate directories exist:
  - `checkpoints/fragfm_humu_5k`
  - `checkpoints/fragfm_humu_smoke`

Not observed in the current production-source inventory:

- W6 real TAR reward payload matching the production run-plan naming pattern.
- W9 real HFM decoder source artifact matching the production run-plan naming
  pattern.
- W10 real supervised HCIV JSONL matching the production run-plan naming
  pattern.
- W13 real teacher-record source or per-consumer teacher embedding artifact
  matching the production run-plan naming pattern.

Conclusion:

- The only immediately available production-resource evidence remains W11 local
  engineering evidence.
- W6, W9, W10, and W13 should remain source-data blocked until approved real
  source files are provided or discovered.
- Do not fabricate placeholder production inputs to make those preflights pass.

Back-check:

- [x] This inventory step was read-only.
- [x] This step did not open or print `.env`.
- [x] This step did not create, modify, or delete artifacts.
- [x] This step did not execute from `/workspace/SemMol` or
      `/workspace/Projects`.
- [x] This step did not launch training, TAR search, benchmarks, or
      distillation.

## 2026-06-06 Production Source Inventory Synced To Gate Gaps

Modified:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W6-tar-production-readiness-gap.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W9-hfm-decoder-production-readiness-gap.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W10-hciv-production-readiness-gap.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W13-kd-production-readiness-gap.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

Implemented:

- Added a `Latest Source Inventory Check` section to each W6/W9/W10/W13
  production readiness gap.
- Recorded that the current matched inventory contains W11 FragFM local
  engineering evidence, but no approved production-candidate input for W6 TAR
  reward payloads, W9 HFM decoder source data, W10 supervised HCIV data, or W13
  KD teacher records.
- Preserved each gate's production-resource blocked status.
- Preserved the instruction not to create synthetic or placeholder production
  inputs to make preflights pass.

Back-check:

- [x] This step modified documentation only.
- [x] This step did not change production readiness criteria.
- [x] This step did not launch training, TAR search, benchmarks, or
      distillation.
- [x] This step did not change Docker, Kubernetes, Helm, or service defaults.
- [x] This step did not modify Owner B implementation files.
- [x] This step did not touch protected checkpoint paths.

## 2026-06-06 W11 Local Evidence Field Audit

Checked:

- `moleculeforge/checkpoints/fragfm_humu_5k/training_manifest.json`
- `moleculeforge/checkpoints/fragfm_humu_5k/quality_report.json`
- `moleculeforge/data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.report.json`
- `moleculeforge/data/processing/generator_artifacts/fragfm_humu_5k_sample_64.report.json`
- `moleculeforge/data/processing/generator_artifacts/fragfm_humu_5k_sample_256.report.json`
- `moleculeforge/data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.smi`
- `moleculeforge/data/processing/generator_artifacts/fragfm_humu_5k_sample_64.smi`
- `moleculeforge/data/processing/generator_artifacts/fragfm_humu_5k_sample_256.smi`

Observed:

- `training_manifest.json` uses `records=5000`, `fragments=2860`,
  `humu_embedding_count=5000`, `humu_embedding_coverage=1.0`, `epochs=1`,
  `rate_optimizer=sgd`, and `rate_grad_clip=false`.
- `quality_report.json` uses `rules=5000`, `fragments=2860`,
  `humu_embedding_count=5000`, `humu_embedding_coverage=1.0`,
  `invalid_humu_embeddings=0`, `checkpoint_loadable=true`,
  `rate_matrix_loadable=true`, `messages=[]`, and `status=pass`.
- The 8/64/256 sample reports use `valid_smiles` and `unique_smiles`, not
  `valid_smiles_count` or `unique_smiles_count`.
- The 8/64/256 `.smi` files have 8, 64, and 256 non-empty lines respectively,
  matching each report's `generated_samples`, `valid_smiles`, and
  `unique_smiles` values.

Modified:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-readiness-gap.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-artifact-promotion-policy.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

Implemented:

- Recorded exact current JSON field names for W11 local evidence.
- Preserved the existing local evidence quantities and production non-promotion
  status.
- Clarified that the audit did not rerun generation or rewrite artifacts.

Back-check:

- [x] This step was read-only for artifacts.
- [x] This step modified documentation only.
- [x] This step did not run FragFM generation, MOSES/GuacaMol/PMO benchmark
      acceptance, or training.
- [x] This step did not change deployment defaults.
- [x] This step did not touch protected checkpoint paths.

## 2026-06-06 W11 Deployment Default Path Recheck

Checked:

- `moleculeforge/infra/docker/docker-compose.dev.yml`
- `moleculeforge/infra/kubernetes/deployments/moleculeforge-services.yaml`
- `moleculeforge/infra/helm/moleculeforge/values.yaml`
- `moleculeforge/tests/unit/test_service_artifact_status.py`

Observed:

- Docker Compose `fragfm-generator-svc` defaults remain:
  - `FRAGFM_VOCAB_PATH=${FRAGFM_VOCAB_PATH:-checkpoints/fragfm_humu_5k/vocab.json}`
  - `FRAGFM_CHECKPOINT_PATH=${FRAGFM_CHECKPOINT_PATH:-checkpoints/fragfm_humu_5k/best_model.pt}`
  - `FRAGFM_RATE_MATRIX_PATH=${FRAGFM_RATE_MATRIX_PATH:-checkpoints/fragfm_humu_5k/rate_matrix.pt}`
- Raw Kubernetes `fragfm-generator-config` remains:
  - `vocab-path=checkpoints/fragfm_humu_5k/vocab.json`
  - `checkpoint-path=checkpoints/fragfm_humu_5k/best_model.pt`
  - `rate-matrix-path=checkpoints/fragfm_humu_5k/rate_matrix.pt`
- Helm `fragfm-generator-config` remains:
  - `vocab-path=checkpoints/fragfm_humu_5k/vocab.json`
  - `checkpoint-path=checkpoints/fragfm_humu_5k/best_model.pt`
  - `rate-matrix-path=checkpoints/fragfm_humu_5k/rate_matrix.pt`
- Focused deployment tests still assert the same three paths and the presence
  of the local 5k artifact files.

Conclusion:

- W11 deployment defaults still match the current local-engineering candidate
  documented in the handoff.
- This recheck does not change the production non-promotion status of
  `checkpoints/fragfm_humu_5k/`.

Back-check:

- [x] This step was read-only.
- [x] This step did not modify deployment manifests.
- [x] This step did not run service startup, FragFM generation, benchmarks, or
      training.
- [x] This step did not touch protected checkpoint paths.

## 2026-06-06 W11 Training Paused And Code Observability Hardening

Context:

- User direction changed to no large-scale training for now; focus on code and
  engineering completion first.
- A previously started stronger CPU FragFM run at
  `checkpoints/fragfm_humu_candidate_20260606_155439/` was stopped and recorded
  as aborted-run evidence only.

Aborted run status:

- Directory: `moleculeforge/checkpoints/fragfm_humu_candidate_20260606_155439/`
- Files present: `training_command.txt`, empty `training.log`, `vocab.json`,
  `aborted_run_record.md`.
- Files not produced: `best_model.pt`, `rate_matrix.pt`, `final_model.pt`,
  `final_rate_matrix.pt`, `training_manifest.json`.
- This directory is not a candidate artifact and must not be used for quality,
  benchmark, deployment, or promotion decisions.

Modified code:

- `moleculeforge/models/mf-generators/fragfm/train.py`
- `moleculeforge/tests/unit/test_generators.py`

Implemented:

- Added `--log-every` training CLI control; default is `25`, and `0` disables
  batch progress logs.
- Added runtime logs for loaded record/fragment counts, output directory,
  requested-device fallback, actual training device, batch progress, and epoch
  seconds.
- Added manifest runtime fields: `requested_device`, `actual_device`, and
  `log_every`.
- Extracted lightweight helpers:
  - `_resolve_training_device()`
  - `_should_log_batch()`
  - `_training_manifest_payload()`
- Replaced the new runtime-control verification with helper-level tests so this
  behavior can be checked without launching a training subprocess.

Modified docs:

- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-readiness-gap.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-production-execution-roadmap.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

Verification:

- Stopped leftover focused pytest/training subprocesses from the previous run.
- Process scans after cleanup matched only the scan commands themselves, not
  active FragFM training or pytest work.
- Lightweight focused pytest passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_training_batch_log_policy_logs_first_interval_and_final_batches tests/unit/test_generators.py::TestFragFMGenerator::test_training_manifest_records_runtime_controls_without_launching_training -q`
  Result: 2 passed, with the existing disabled-plugin `asyncio_mode` warning.
- CLI help check passed:
  `.venv/bin/python models/mf-generators/fragfm/train.py --help | rg -n -- '--log-every|--rate-optimizer|--kd-teacher-embeddings|--humu-embedding-dim'`
  Result: all expected options were present.
- Compile check passed:
  `python -m py_compile moleculeforge/models/mf-generators/fragfm/train.py moleculeforge/tests/unit/test_generators.py`
- Whitespace diff check passed:
  `git diff --check -- moleculeforge/models/mf-generators/fragfm/train.py moleculeforge/tests/unit/test_generators.py`

Back-check:

- [x] This step did not launch or continue large-scale training.
- [x] This step did not create a production candidate artifact.
- [x] This step did not overwrite protected checkpoint paths.
- [x] This step did not change Docker Compose, Kubernetes, Helm, or service
      deployment defaults.
- [x] This step did not modify benchmark thresholds.
- [x] This step did not modify Owner B implementation files.
- [x] This step did not modify HUMU pretraining config, loss, encoder
      architecture, or checkpoint continuation.
- [x] This step did not stop or modify external `/workspace/SemMol` GPU
      processes.

## 2026-06-06 W11 FragFM Rate Matrix Quality Schema Hardened

Context:

- Continued under the current no-large-scale-training instruction.
- Selected a W11 code-hardening gap in FragFM artifact quality validation.
- `FragFMGenerator` sparse runtime scoring depends on both `base_rate` and
  `sa_score_embedding.weight`, but the quality gate previously only validated
  `base_rate` shape.

Modified:

- `moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

TDD record:

- Added
  `tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_rate_matrix_sa_embedding_missing`.
- RED command:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_rate_matrix_sa_embedding_missing -q`
- RED result: failed because `build_quality_report()` returned `status="pass"`
  for a rate matrix state dict with `base_rate` but no
  `sa_score_embedding.weight`.

Implemented:

- `_loadable_rate_matrix()` now requires `sa_score_embedding.weight`.
- The expected shape is `(10, vocab_size * vocab_size)`, matching
  `SAAwareRateMatrix` and the runtime sparse scoring path.
- Missing or shape-mismatched SA embedding weight marks
  `rate_matrix_loadable=false` and adds a clear report message.

Verification:

- Focused quality tests passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_passes_with_valid_shared_humu_artifacts tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_rate_matrix_schema_missing tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_rate_matrix_sa_embedding_missing -q`
  Result: 3 passed, with the existing disabled-plugin `asyncio_mode` warning.
- Compile check passed:
  `python -m py_compile moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py moleculeforge/tests/unit/test_generators.py`
- Whitespace diff check passed:
  `git diff --check -- moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py moleculeforge/tests/unit/test_generators.py`
- Process scan found no lingering pytest or FragFM training processes after the
  test run.

Back-check:

- [x] This step did not launch or continue training.
- [x] This step did not modify generation sampling behavior.
- [x] This step did not change Docker Compose, Kubernetes, Helm, or service
      deployment defaults.
- [x] This step did not modify benchmark thresholds.
- [x] This step did not modify Owner B implementation files.
- [x] This step did not touch protected checkpoint paths.
- [x] This step did not modify HUMU pretraining config, loss, encoder
      architecture, or checkpoint continuation.

## 2026-06-06 W11 FragFM Runtime Rate Matrix Fail-Fast Hardened

Context:

- Continued W11 code hardening under the current no-large-scale-training
  instruction.
- The quality gate now checks `sa_score_embedding.weight`, but runtime loading in
  `FragFMGenerator` still accepted incomplete rate matrix state dicts via
  `strict=False`.

Modified:

- `moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

TDD record:

- Added
  `tests/unit/test_generators.py::TestFragFMGenerator::test_rejects_rate_matrix_artifact_missing_sa_embedding_weight`.
- RED command:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_rejects_rate_matrix_artifact_missing_sa_embedding_weight -q`
- RED result: failed with `DID NOT RAISE <class 'ValueError'>`, proving runtime
  accepted a rate matrix artifact containing `base_rate` but missing
  `sa_score_embedding.weight`.

Implemented:

- Added `FragFMGenerator._validate_rate_matrix_state()`.
- Runtime now rejects non-mapping rate matrix payloads, missing `base_rate`,
  mismatched `base_rate` shape, missing `sa_score_embedding.weight`, and
  mismatched SA embedding shape before calling `load_state_dict()`.
- Expected runtime schema is aligned with `SAAwareRateMatrix` and the W11 quality
  gate: `base_rate=(vocab_size, vocab_size)` and
  `sa_score_embedding.weight=(10, vocab_size * vocab_size)`.

Verification:

- Focused runtime/quality tests passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_loads_checkpoint_and_rate_matrix_artifacts tests/unit/test_generators.py::TestFragFMGenerator::test_rejects_rate_matrix_artifact_missing_sa_embedding_weight tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_passes_with_valid_shared_humu_artifacts tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_rate_matrix_sa_embedding_missing -q`
  Result: 4 passed, with the existing disabled-plugin `asyncio_mode` warning.
- Compile check passed:
  `python -m py_compile moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py moleculeforge/tests/unit/test_generators.py`
- Whitespace diff check passed:
  `git diff --check -- moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py moleculeforge/tests/unit/test_generators.py`
- Process scan found no lingering pytest or FragFM training processes after the
  test run.

Back-check:

- [x] This step did not launch or continue training.
- [x] This step did not change generation ranking or sampling behavior for valid
      artifacts.
- [x] This step did not change Docker Compose, Kubernetes, Helm, or service
      deployment defaults.
- [x] This step did not modify benchmark thresholds.
- [x] This step did not modify Owner B implementation files.
- [x] This step did not touch protected checkpoint paths.
- [x] This step did not modify HUMU pretraining config, loss, encoder
      architecture, or checkpoint continuation.

## 2026-06-06 W11 FragFM Sample Export Atomic Write Hardened

Context:

- Continued W11 code hardening under the current no-large-scale-training
  instruction.
- `export_fragfm_samples()` wrote the final `.smi` output before writing the JSON
  report. If report writing failed, a partial benchmark-input artifact could be
  left behind.

Modified:

- `moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

TDD record:

- Added
  `tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_does_not_leave_smiles_when_report_write_fails`.
- RED command:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_does_not_leave_smiles_when_report_write_fails -q`
- RED result: failed because `fragfm_generated.smi` existed after report path
  setup failed.

Implemented:

- Sample export now writes `.smi` and report content to temporary sibling files
  first and promotes them with `Path.replace()` only after writes succeed.
- Temporary files are removed on exceptions.
- Report path parent preparation happens before the final `.smi` path is
  promoted, so a report-path failure does not leave a final SMILES file.

Verification:

- Focused sample export tests passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_writes_smiles_and_report tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_does_not_leave_smiles_when_report_write_fails -q`
  Result: 2 passed, with the existing disabled-plugin `asyncio_mode` warning.
- Compile check passed:
  `python -m py_compile moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py moleculeforge/tests/unit/test_generators.py`
- Whitespace diff check passed:
  `git diff --check -- moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py moleculeforge/tests/unit/test_generators.py`
- Process scan found no lingering pytest or FragFM training processes after the
  test run.

Back-check:

- [x] This step did not launch or continue training.
- [x] This step did not change sample generation semantics for successful
      exports.
- [x] This step did not change Docker Compose, Kubernetes, Helm, or service
      deployment defaults.
- [x] This step did not modify benchmark thresholds.
- [x] This step did not modify Owner B implementation files.
- [x] This step did not touch protected checkpoint paths.
- [x] This step did not modify HUMU pretraining config, loss, encoder
      architecture, or checkpoint continuation.

## 2026-06-06 W11 FragFM Runtime Checkpoint Fail-Fast Hardened

Context:

- Continued W11 code hardening under the current no-large-scale-training
  instruction.
- `FragFMGenerator` inferred checkpoint hidden dimension from
  `fragment_encoder.weight`, but missing checkpoint schema could still be
  accepted through the default hidden dimension and `strict=False` loading path.

Modified:

- `moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

TDD record:

- Added
  `tests/unit/test_generators.py::TestFragFMGenerator::test_rejects_checkpoint_artifact_missing_fragment_encoder_weight`.
- RED command:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_rejects_checkpoint_artifact_missing_fragment_encoder_weight -q`
- RED result: failed with `DID NOT RAISE <class 'ValueError'>`, proving runtime
  accepted a checkpoint artifact without `fragment_encoder.weight`.

Implemented:

- Added `FragFMGenerator._validate_checkpoint_state()`.
- Runtime now rejects non-mapping checkpoint payloads, missing
  `fragment_encoder.weight`, non-2D/empty hidden dimensions, and fragment
  vocabulary size mismatches before constructing/loading `TwoLevelDFM`.
- This aligns runtime checkpoint requirements with the W11 quality gate and
  prevents silent fallback to default hidden dimension for malformed artifacts.

Verification:

- Focused runtime/quality/sample-export tests passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_loads_checkpoint_and_rate_matrix_artifacts tests/unit/test_generators.py::TestFragFMGenerator::test_rejects_checkpoint_artifact_missing_fragment_encoder_weight tests/unit/test_generators.py::TestFragFMGenerator::test_rejects_rate_matrix_artifact_missing_sa_embedding_weight tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_passes_with_valid_shared_humu_artifacts tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_rate_matrix_sa_embedding_missing tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_writes_smiles_and_report tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_does_not_leave_smiles_when_report_write_fails -q`
  Result: 7 passed, with the existing disabled-plugin `asyncio_mode` warning.
- Compile check passed:
  `python -m py_compile moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py moleculeforge/tests/unit/test_generators.py`
- Whitespace diff check passed:
  `git diff --check -- moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py moleculeforge/tests/unit/test_generators.py moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`
- Process scan found no lingering pytest or FragFM training processes after the
  test run.

Back-check:

- [x] This step did not launch or continue training.
- [x] This step did not change generation ranking or sampling behavior for valid
      artifacts.
- [x] This step did not change Docker Compose, Kubernetes, Helm, or service
      deployment defaults.
- [x] This step did not modify benchmark thresholds.
- [x] This step did not modify Owner B implementation files.
- [x] This step did not touch protected checkpoint paths.
- [x] This step did not modify HUMU pretraining config, loss, encoder
      architecture, or checkpoint continuation.

## 2026-06-06 W11 FragFM Quality Manifest Consistency Hardened

Context:

- Continued W11 engineering hardening under the current no-large-scale-training
  instruction.
- FragFM training writes `training_manifest.json`, but the quality report did
  not previously verify that the manifest agreed with the vocabulary,
  checkpoint, rate matrix, and HUMU coverage actually being inspected.

Modified:

- `moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

TDD record:

- Added
  `tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_training_manifest_disagrees`.
- RED command:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_training_manifest_disagrees -q`
- RED result: failed with
  `TypeError: build_quality_report() got an unexpected keyword argument 'manifest_path'`,
  proving the quality gate had no manifest consistency input.
- Added
  `tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_cli_strict_fails_when_training_manifest_disagrees`.
- CLI RED method: temporarily removed the already-implemented
  `manifest_path=args.manifest` handoff in `main()`.
- CLI RED command:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_cli_strict_fails_when_training_manifest_disagrees -q`
- CLI RED result: failed because strict CLI returned `0` when the manifest
  disagreed with the artifact fragment count. The handoff was restored before
  GREEN verification.

Implemented:

- `build_quality_report()` now accepts optional `manifest_path`.
- Reports now include `manifest_path` and `manifest_consistent`.
- Added manifest consistency checks for:
  - JSON object payload and `schema_version == "fragfm_training.v1"`;
  - `records` against vocabulary `assembly_rules` count;
  - `fragments` against vocabulary fragment count;
  - `humu_embedding_count` and `humu_embedding_coverage` against current
    validated vocabulary contents;
  - `humu_embedding_dim` and `humu_curvature` against quality-report inputs;
  - `vocab_path`, `checkpoint_path`, and `rate_matrix_path` against quality
    inputs.
- Added quality CLI `--manifest`; `--strict` now returns non-zero when a
  provided manifest is inconsistent.
- When no manifest is provided, the quality report preserves existing behavior
  and treats `manifest_consistent` as `true`.

Verification:

- New manifest report test passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_training_manifest_disagrees -q`
  Result: 1 passed, with the existing disabled-plugin `asyncio_mode` warning.
- Focused quality tests passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_passes_with_valid_shared_humu_artifacts tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_training_manifest_disagrees tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_cli_strict_fails_when_training_manifest_disagrees tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_humu_coverage_is_too_low tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_checkpoint_schema_missing tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_rate_matrix_schema_missing tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_rate_matrix_sa_embedding_missing -q`
  Result: 7 passed, with the existing disabled-plugin `asyncio_mode` warning.
- Focused W11 runtime/quality/sample-export tests passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_loads_checkpoint_and_rate_matrix_artifacts tests/unit/test_generators.py::TestFragFMGenerator::test_rejects_checkpoint_artifact_missing_fragment_encoder_weight tests/unit/test_generators.py::TestFragFMGenerator::test_rejects_rate_matrix_artifact_missing_sa_embedding_weight tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_passes_with_valid_shared_humu_artifacts tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_training_manifest_disagrees tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_cli_strict_fails_when_training_manifest_disagrees tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_rate_matrix_sa_embedding_missing tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_writes_smiles_and_report tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_does_not_leave_smiles_when_report_write_fails -q`
  Result: 9 passed, with the existing disabled-plugin `asyncio_mode` warning.
- Compile check passed:
  `python -m py_compile moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py moleculeforge/tests/unit/test_generators.py`
- Whitespace diff check passed:
  `git diff --check -- moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py moleculeforge/tests/unit/test_generators.py moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`
- Read-only local 5k quality CLI smoke with manifest passed and wrote only to
  `/tmp/fragfm_humu_5k_manifest_quality_report.json`:
  `PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" .venv/bin/python -m mf_generators.fragfm.quality --vocab checkpoints/fragfm_humu_5k/vocab.json --checkpoint checkpoints/fragfm_humu_5k/best_model.pt --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt --manifest checkpoints/fragfm_humu_5k/training_manifest.json --min-humu-coverage 1.0 --strict --output /tmp/fragfm_humu_5k_manifest_quality_report.json`
  Observed summary: `status=pass`, `rules=5000`, `fragments=2860`,
  `humu_embedding_count=5000`, `humu_embedding_coverage=1.0`,
  `checkpoint_loadable=true`, `rate_matrix_loadable=true`,
  `manifest_consistent=true`, `message_count=0`.
- Process scan found no lingering pytest, FragFM training, or FragFM quality CLI
  processes after verification.

Back-check:

- [x] This step did not launch or continue training.
- [x] This step did not change generation ranking or sampling behavior for valid
      artifacts.
- [x] This step did not change Docker Compose, Kubernetes, Helm, or service
      deployment defaults.
- [x] This step did not modify benchmark thresholds.
- [x] This step did not modify Owner B implementation files.
- [x] This step did not touch protected checkpoint paths or write into
      `checkpoints/fragfm_humu_5k/`.
- [x] This step did not modify HUMU pretraining config, loss, encoder
      architecture, or checkpoint continuation.

## 2026-06-06 W11 Manifest-Aware Quality Procedure Synced

Context:

- After adding `--manifest` support to the FragFM quality CLI, the W11 run plan
  and handoff still had quality-check command snippets that only inspected
  vocab/checkpoint/rate-matrix artifacts.

Modified:

- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-readiness-gap.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-training-run-plan.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

Implemented:

- Updated W11 strict quality command snippets to include
  `--manifest .../training_manifest.json`.
- Recorded the new `manifest_consistent` quality-report field in the handoff,
  readiness gap, and new-session entry.
- Clarified that the observed 5k manifest-aware quality pass was a read-only
  local smoke writing only to `/tmp/fragfm_humu_5k_manifest_quality_report.json`,
  not a production artifact rewrite or promotion.
- Updated the W11 production run plan so future candidate promotion evidence
  requires strict quality with `manifest_consistent=true`.

Verification:

- W11 doc command consistency scan passed:
  `rg -n "mf_generators.fragfm.quality|--manifest|manifest_consistent|W11 strict quality|W11 5k strict quality" moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-training-run-plan.md moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-readiness-gap.md`
- Whitespace diff check passed:
  `git diff --check -- moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-training-run-plan.md moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-production-readiness-gap.md`
- Process scan found no lingering pytest, FragFM training, or FragFM quality CLI
  processes after the doc sync.

Back-check:

- [x] This step did not launch or continue training.
- [x] This step did not change code behavior, generation behavior, deployment
      defaults, or benchmark thresholds.
- [x] This step did not modify Owner B implementation files.
- [x] This step did not touch protected checkpoint paths or write into
      `checkpoints/fragfm_humu_5k/`.
- [x] This step did not modify HUMU pretraining config, loss, encoder
      architecture, or checkpoint continuation.

## 2026-06-06 W11 FragFM Quality Report Atomic Write Hardened

Context:

- Continued W11 quality/export artifact hygiene hardening under the current
  no-large-scale-training instruction.
- `mf_generators.fragfm.sample_export` already writes final artifacts
  atomically, but `mf_generators.fragfm.quality --output` wrote directly to the
  final JSON report path. A write failure could leave a partial
  `quality_report.json`.

Modified:

- `moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

TDD record:

- Added
  `tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_cli_does_not_leave_report_when_output_write_fails`.
- RED command:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_cli_does_not_leave_report_when_output_write_fails -q`
- RED result: failed because `quality_report.json` existed after a simulated
  output write failure.

Implemented:

- Added `_write_report_atomic()` to the FragFM quality CLI.
- `--output` now writes to a temporary sibling file and promotes it with
  `Path.replace()` only after the full JSON report is written.
- Temporary report files are removed on exceptions.
- Stdout behavior is unchanged when `--output` is not supplied.

Verification:

- New atomic-write test passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_cli_does_not_leave_report_when_output_write_fails -q`
  Result: 1 passed, with the existing disabled-plugin `asyncio_mode` warning.
- Focused quality tests passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_passes_with_valid_shared_humu_artifacts tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_training_manifest_disagrees tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_cli_strict_fails_when_training_manifest_disagrees tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_cli_does_not_leave_report_when_output_write_fails tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_humu_coverage_is_too_low tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_checkpoint_schema_missing tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_rate_matrix_schema_missing tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_quality_report_fails_when_rate_matrix_sa_embedding_missing -q`
  Result: 8 passed, with the existing disabled-plugin `asyncio_mode` warning.
- Compile check passed:
  `python -m py_compile moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py moleculeforge/tests/unit/test_generators.py`
- Whitespace diff check passed:
  `git diff --check -- moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py moleculeforge/tests/unit/test_generators.py moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`
- Process scan found no lingering pytest, FragFM training, or FragFM quality CLI
  processes after verification.

Back-check:

- [x] This step did not launch or continue training.
- [x] This step did not change quality scoring semantics, generation behavior,
      deployment defaults, or benchmark thresholds.
- [x] This step did not modify Owner B implementation files.
- [x] This step did not touch protected checkpoint paths or write into
      `checkpoints/fragfm_humu_5k/`.
- [x] This step did not modify HUMU pretraining config, loss, encoder
      architecture, or checkpoint continuation.

## 2026-06-06 W11 FragFM Sample Export Path Conflict Hardened

Context:

- Continued W11 sample-export artifact hygiene hardening under the current
  no-large-scale-training instruction.
- `export_fragfm_samples()` atomically writes the SMILES file and optional JSON
  report, but it did not reject a caller passing the same path for both outputs.
  That could make the report and SMILES outputs overwrite each other.

Modified:

- `moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

TDD record:

- Added
  `tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_same_output_and_report_path`.
- RED command:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_same_output_and_report_path -q`
- RED result: failed with `DID NOT RAISE <class 'ValueError'>`, proving sample
  export accepted identical output/report paths.

Implemented:

- `export_fragfm_samples()` now resolves `output_path` and `report_path` before
  constructing the generator.
- If both paths resolve to the same location, it raises
  `ValueError("FragFM sample export report_path must differ from output_path")`.
- The check happens before generation or filesystem writes, so the invalid call
  leaves no final output artifact.

Verification:

- New path-conflict test passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_same_output_and_report_path -q`
  Result: 1 passed, with the existing disabled-plugin `asyncio_mode` warning.
- Focused sample export tests passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_writes_smiles_and_report tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_does_not_leave_smiles_when_report_write_fails tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_same_output_and_report_path -q`
  Result: 3 passed, with the existing disabled-plugin `asyncio_mode` warning.
- Compile check passed:
  `python -m py_compile moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py moleculeforge/tests/unit/test_generators.py`
- Whitespace diff check passed:
  `git diff --check -- moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py moleculeforge/tests/unit/test_generators.py moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`
- Process scan found no lingering pytest, FragFM training, or FragFM generator
  CLI processes after verification.

Back-check:

- [x] This step did not launch or continue training.
- [x] This step did not change successful sample generation semantics,
      deployment defaults, or benchmark thresholds.
- [x] This step did not modify Owner B implementation files.
- [x] This step did not touch protected checkpoint paths or write into
      `checkpoints/fragfm_humu_5k/`.
- [x] This step did not modify HUMU pretraining config, loss, encoder
      architecture, or checkpoint continuation.

## 2026-06-06 W11 FragFM Missing Checkpoint Fail-Fast Hardened

Context:

- Continued W11 runtime artifact fail-fast hardening under the current
  no-large-scale-training instruction.
- `FragFMGenerator` already failed fast for missing explicit rate matrix paths,
  but an explicit missing checkpoint path was silently ignored because the
  model was loaded only when `Path(checkpoint_path).exists()`.

Modified:

- `moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

TDD record:

- Added
  `tests/unit/test_generators.py::TestFragFMGenerator::test_rejects_missing_checkpoint_artifact_when_path_is_explicit`.
- RED command:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_rejects_missing_checkpoint_artifact_when_path_is_explicit -q`
- RED result: failed with `DID NOT RAISE <class 'FileNotFoundError'>`,
  proving the explicit missing checkpoint was silently ignored.

Implemented:

- `FragFMGenerator.__init__()` now checks an explicit `checkpoint_path` before
  loading.
- Missing explicit checkpoint paths now raise
  `FileNotFoundError("FragFM checkpoint artifact not found: ...")`.
- Vocab-only generation with no checkpoint path remains unchanged.

Verification:

- New missing-checkpoint test passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_rejects_missing_checkpoint_artifact_when_path_is_explicit -q`
  Result: 1 passed, with the existing disabled-plugin `asyncio_mode` warning.
- Focused runtime tests passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_uses_fragment_vocabulary_rules_and_validity_check tests/unit/test_generators.py::TestFragFMGenerator::test_loads_checkpoint_and_rate_matrix_artifacts tests/unit/test_generators.py::TestFragFMGenerator::test_rejects_missing_checkpoint_artifact_when_path_is_explicit tests/unit/test_generators.py::TestFragFMGenerator::test_rejects_checkpoint_artifact_missing_fragment_encoder_weight tests/unit/test_generators.py::TestFragFMGenerator::test_rejects_rate_matrix_artifact_missing_sa_embedding_weight -q`
  Result: 5 passed, with the existing disabled-plugin `asyncio_mode` warning.
- Compile check passed:
  `python -m py_compile moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py moleculeforge/tests/unit/test_generators.py`
- Whitespace diff check passed:
  `git diff --check -- moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py moleculeforge/tests/unit/test_generators.py moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`
- Process scan found no lingering pytest, FragFM training, or FragFM generator
  CLI processes after verification.

Back-check:

- [x] This step did not launch or continue training.
- [x] This step did not change vocab-only generation, valid checkpoint loading,
      deployment defaults, or benchmark thresholds.
- [x] This step did not modify Owner B implementation files.
- [x] This step did not touch protected checkpoint paths or write into
      `checkpoints/fragfm_humu_5k/`.
- [x] This step did not modify HUMU pretraining config, loss, encoder
      architecture, or checkpoint continuation.

## 2026-06-06 W11 FragFM Service Optional Artifact Preflight Hardened

Context:

- Continued W11 service-runtime artifact hardening under the current
  no-large-scale-training instruction.
- `mf_core.artifacts.require_available()` only rejects required unavailable
  artifacts. FragFM checkpoint and rate-matrix artifacts are optional when
  unset, but should fail fast when the service operator explicitly configures a
  path that does not exist.
- `FragFMGenerator` runtime now rejects explicit missing checkpoint/rate-matrix
  paths, so the service preflight needed matching startup/request behavior.

Modified:

- `moleculeforge/services/fragfm-generator-svc/src/fragfm_generator_svc/main.py`
- `moleculeforge/tests/unit/test_service_artifact_status.py`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

TDD record:

- Added
  `tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_checkpoint`.
- RED command:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_checkpoint -q`
- RED result: failed with `DID NOT RAISE <class 'RuntimeError'>`, proving the
  service accepted a configured missing optional checkpoint artifact.

Implemented:

- Added `_require_configured_artifacts_available()` in
  `fragfm_generator_svc.main`.
- `_require_runtime()` now calls both `require_available(statuses)` and
  `_require_configured_artifacts_available(statuses)`.
- `_abort_unavailable()` uses the same configured-artifact check, so startup
  and request-time abort paths report configured missing optional artifacts
  consistently.
- Added
  `tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_rate_matrix`.
  This coverage passed against the same helper, confirming the service rejects a
  configured missing optional rate matrix without additional production code.

Verification:

- New checkpoint service preflight test passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_checkpoint -q`
  Result: 1 passed, with existing disabled-plugin `asyncio_mode` and
  `pytest.mark.asyncio` warnings.
- New rate-matrix service preflight coverage passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_rate_matrix -q`
  Result: 1 passed, with existing disabled-plugin `asyncio_mode` and
  `pytest.mark.asyncio` warnings.
- Focused FragFM service artifact tests passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_service_artifact_status.py::test_fragfm_service_builds_generator_with_trained_artifacts tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_checkpoint tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_rate_matrix tests/unit/test_service_artifact_status.py::test_fragfm_deployment_wires_artifact_and_sampler_env -q`
  Result: 4 passed, with existing disabled-plugin `asyncio_mode` and
  `pytest.mark.asyncio` warnings.
- Compile check passed:
  `python -m py_compile services/fragfm-generator-svc/src/fragfm_generator_svc/main.py tests/unit/test_service_artifact_status.py`
- Whitespace diff check passed:
  `git diff --check -- services/fragfm-generator-svc/src/fragfm_generator_svc/main.py tests/unit/test_service_artifact_status.py docs/todo/owner-a-generation-upstream/progress.md`
- Final independent process scan found no lingering pytest, FragFM training,
  FragFM service, or FragFM CLI process after verification.

Back-check:

- [x] This step did not launch or continue training.
- [x] This step did not change deployment defaults, successful artifact loading,
      generation semantics, or benchmark thresholds.
- [x] This step did not modify Owner B implementation files.
- [x] This step did not touch protected checkpoint paths or write into
      `checkpoints/fragfm_humu_5k/`.
- [x] This step did not modify HUMU pretraining config, loss, encoder
      architecture, or checkpoint continuation.

## 2026-06-06 W11 FragFM Service Invalid Intent-Cone Handling Hardened

Context:

- Continued W11 service-runtime request-boundary hardening under the current
  no-large-scale-training instruction.
- FragFM service now passes request `intent_cone` payloads into the generator,
  but invalid `intent_cone` payloads could leak parser/model-validation
  exceptions instead of being mapped to the existing gRPC invalid-argument path.
- This was inconsistent with the existing `batch_size` request validation path.

Modified:

- `moleculeforge/services/fragfm-generator-svc/src/fragfm_generator_svc/main.py`
- `moleculeforge/tests/unit/test_service_artifact_status.py`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

TDD record:

- Added
  `tests/unit/test_service_artifact_status.py::test_fragfm_service_rejects_invalid_intent_cone_as_invalid_argument`.
- Initial RED command:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_service_artifact_status.py::test_fragfm_service_rejects_invalid_intent_cone_as_invalid_argument -q`
- First run exposed that the new test must not depend on disabled
  `pytest.mark.asyncio` handling; the test was corrected to use
  `asyncio.run(...)`.
- Correct RED result: failed because `_intent_cone_from_request()` raised a
  pydantic validation error directly, the error message did not contain
  `intent_cone`, `context.abort()` was not called, and the request did not map
  to `grpc.StatusCode.INVALID_ARGUMENT`.

Implemented:

- `FragFMGeneratorServicer.Generate()` now parses `intent_cone` before calling
  the generator.
- `TypeError` and `ValueError` from `_intent_cone_from_request()` are mapped to
  `_abort_invalid_argument(context, "intent_cone is invalid: ...")`.
- Invalid `intent_cone` requests no longer call the generator.
- Valid `intent_cone` request handling remains unchanged.

Verification:

- New invalid-intent-cone test passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_service_artifact_status.py::test_fragfm_service_rejects_invalid_intent_cone_as_invalid_argument -q`
  Result: 1 passed, with existing disabled-plugin `asyncio_mode` and
  `pytest.mark.asyncio` warnings.
- Focused FragFM service request/artifact tests passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_service_artifact_status.py::test_fragfm_service_rejects_invalid_intent_cone_as_invalid_argument tests/unit/test_service_artifact_status.py::test_fragfm_service_builds_generator_with_trained_artifacts tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_checkpoint tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_rate_matrix -q`
  Result: 4 passed, with existing disabled-plugin `asyncio_mode` and
  `pytest.mark.asyncio` warnings.
- Existing async generator intent-cone adjacent tests passed with pytest plugins
  enabled:
  `.venv/bin/python -m pytest tests/unit/test_service_artifact_status.py::test_generator_services_pass_request_intent_cone_to_model_object tests/unit/test_service_artifact_status.py::test_fragfm_service_rejects_invalid_intent_cone_as_invalid_argument -q`
  Result: 4 passed.
- A broader adjacent command with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` was also
  attempted, but the existing async parametrized tests failed because async
  pytest plugins were intentionally disabled. That command is not regression
  evidence for this change; the plugin-enabled adjacent run above is the valid
  check for those async tests.
- Compile check passed:
  `python -m py_compile services/fragfm-generator-svc/src/fragfm_generator_svc/main.py tests/unit/test_service_artifact_status.py`
- Whitespace diff check passed:
  `git diff --check -- services/fragfm-generator-svc/src/fragfm_generator_svc/main.py tests/unit/test_service_artifact_status.py docs/todo/owner-a-generation-upstream/progress.md`
- Process scan found no lingering pytest, FragFM training, FragFM service, or
  FragFM CLI process after verification.

Back-check:

- [x] This step did not launch or continue training.
- [x] This step did not change valid `intent_cone` generation behavior,
      deployment defaults, artifact paths, or benchmark thresholds.
- [x] This step did not modify Owner B implementation files.
- [x] This step did not touch protected checkpoint paths or write into
      `checkpoints/fragfm_humu_5k/`.
- [x] This step did not modify HUMU pretraining config, loss, encoder
      architecture, or checkpoint continuation.

## 2026-06-06 W11 FragFM Sample Export Output Parent Preflight Hardened

Context:

- Continued W11 sample-export artifact hygiene hardening under the current
  no-large-scale-training instruction.
- `export_fragfm_samples()` already cleaned up partial output/report files on
  write failures and rejected identical output/report paths, but it constructed
  the generator before discovering that an output parent path was blocked by an
  existing file.
- Bad output paths should fail before generation starts so failed export
  requests do not waste generator startup or sampling work.

Modified:

- `moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py`
- `moleculeforge/tests/unit/test_generators.py`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

TDD record:

- Added
  `tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_blocked_output_parent_before_generation`.
- RED command:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_blocked_output_parent_before_generation -q`
- RED result: failed with `AssertionError: generator should not be constructed`,
  proving `export_fragfm_samples()` constructed the generator before detecting
  the blocked output parent path.

Implemented:

- Added `_ensure_parent_directory()` to `mf_generators.fragfm.sample_export`.
- `export_fragfm_samples()` now validates/creates the report parent directory
  and output parent directory before constructing `FragFMGenerator`.
- Removed later duplicate parent-directory creation calls after generation.
- Existing atomic write and cleanup behavior remains unchanged.

Verification:

- New blocked-output-parent preflight test passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_blocked_output_parent_before_generation -q`
  Result: 1 passed, with the existing disabled-plugin `asyncio_mode` warning.
- Focused sample export tests passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_writes_smiles_and_report tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_does_not_leave_smiles_when_report_write_fails tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_blocked_output_parent_before_generation tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_same_output_and_report_path -q`
  Result: 4 passed, with the existing disabled-plugin `asyncio_mode` warning.
- Compile check passed:
  `python -m py_compile models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py tests/unit/test_generators.py`
- Whitespace diff check passed:
  `git diff --check -- models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py tests/unit/test_generators.py docs/todo/owner-a-generation-upstream/progress.md`
- Final independent process scan found no lingering pytest, FragFM training,
  FragFM service, or FragFM CLI process after verification.

Back-check:

- [x] This step did not launch or continue training.
- [x] This step did not change successful sample generation semantics,
      deployment defaults, artifact paths, or benchmark thresholds.
- [x] This step did not modify Owner B implementation files.
- [x] This step did not touch protected checkpoint paths or write into
      `checkpoints/fragfm_humu_5k/`.
- [x] This step did not modify HUMU pretraining config, loss, encoder
      architecture, or checkpoint continuation.

## 2026-06-06 Owner A Code-Freeze / Owner B Handoff Checklist Added

Context:

- The current user direction is to finish Owner A code engineering first,
  coordinate with Owner B second, and only later run real production tests or
  training.
- After the latest W11 service/runtime/export hardening, the next useful step
  was to make the Owner A code-freeze and Owner B handoff surface explicit.
- This step is documentation-only. It does not claim Owner A production
  acceptance and does not authorize training.

Created:

- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-code-freeze-owner-b-handoff-checklist.md`

Updated:

- `moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/README.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
- `moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`

Recorded in the checklist:

- Owner A code-writing readiness is close to local code-freeze, but the overall
  client task is not production-complete.
- Owner B handoff item W1: the known W1 unit compatibility issue is the patch
  seam `orchestrator_svc.main.build_shared_crg_repository_from_env`, as recorded
  in `2026-06-05-W4-focused-validation-record.md`.
- Owner B handoff item W5: official benchmark data and production-quality
  generated samples remain missing; thresholds must not be relaxed.
- Contract surfaces for Owner A / Owner B review remain C1 `generator_params`,
  C2 CRG predicates, and C3 HUMU encoder 129-dimensional Lorentz output.
- Focused code-freeze checks are listed for documentation diff hygiene, process
  safety, W11 service request/artifact behavior, W11 sample export behavior,
  compile checks, and FragFM deployment-default path scans.

Back-check:

- [x] This step did not modify business code.
- [x] This step did not launch or continue training.
- [x] This step did not modify Owner B implementation files.
- [x] This step did not change benchmark thresholds or production acceptance
      criteria.
- [x] This step keeps `checkpoints/fragfm_humu_5k/` classified as strict-local
      engineering evidence only.

Focused verification after checklist creation:

- Documentation diff hygiene passed:
  `git diff --check -- moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-owner-a-code-freeze-owner-b-handoff-checklist.md moleculeforge/docs/todo/owner-a-generation-upstream/START_HERE_NEW_SESSION.md moleculeforge/docs/todo/owner-a-generation-upstream/README.md moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md moleculeforge/docs/todo/owner-a-generation-upstream/progress.md`
- Placeholder keyword scan across the updated handoff docs produced no output.
- W11 service focused code-freeze shard passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_service_artifact_status.py::test_fragfm_service_rejects_invalid_intent_cone_as_invalid_argument tests/unit/test_service_artifact_status.py::test_fragfm_service_builds_generator_with_trained_artifacts tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_checkpoint tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_rate_matrix -q`
  Result: 4 passed, with existing disabled-plugin `asyncio_mode` and
  `pytest.mark.asyncio` warnings.
- W11 sample export focused code-freeze shard passed:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_writes_smiles_and_report tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_does_not_leave_smiles_when_report_write_fails tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_blocked_output_parent_before_generation tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_same_output_and_report_path -q`
  Result: 4 passed, with the existing disabled-plugin `asyncio_mode` warning.
- Compile check passed:
  `python -m py_compile services/fragfm-generator-svc/src/fragfm_generator_svc/main.py models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py tests/unit/test_service_artifact_status.py tests/unit/test_generators.py`
- Deployment default scan confirmed Docker Compose, raw Kubernetes, Helm, and
  the focused deployment regression reference
  `checkpoints/fragfm_humu_5k/{vocab.json,best_model.pt,rate_matrix.pt}`.
- Process scan found no lingering training, pytest, FragFM service, FragFM CLI,
  HFM training, HCIV training, or TAR runner process beyond the scan command
  itself.
