# CoreArchitecture v2 Owner B Follow-up Todo

## Source

- `docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
- `docs/architecture/corearchitecture-v2-completion-tasksplit.md`

## Current State

Owner B is the validation, retrosynthesis, supply, SRB, critic, and provenance downstream role.

Completed Owner B AI coding tasks from the two source documents:

- W1: CRG final-state merge/readback. Local code is complete. Remaining gate: H1 DKI Neo4j real acceptance.
- W3: PCBO reference candidate provider / oracle evaluator. Local code is complete. Remaining gate: real candidate provider/oracle runner values and production acceptance.
- W5: benchmark harness non-skip path. Local code is complete. Remaining gate: H8 official benchmark data and thresholds.
- W12: CReM-pharm-3D scorer integration tests. Local code is complete. Remaining gate: real DiffDock-L/pharmacophore/HUMU scorer runners and H10 cluster validation.

Owner B has no new AI coding item listed in the two source documents after W1/W3/W5/W12. Remaining Owner B work is W4 validation, C-class resource coordination, production acceptance, and execution-log maintenance.

Do not claim JMCG is complete before W8-R has evidence. The documents only allow declaring W8-E engineering alignment when C1 is green and `JMCGEngineeringSampler` produces `moleculeforge.jmcg.joint_sample.v1`.

## Todo

### P0: Sync With Owner A Before W4

- [ ] Ask Owner A for the complete changed-file list for all Owner A work items already landed or ready for validation: W2, W6, W8, W9, W10, W11, W13.
- [ ] Require Owner A to register each completed handoff in the execution log with: date, ID, changed files, verification command, actual result, and remaining gate.
- [ ] Specifically confirm W2 status. The task split still lists W2 as an Owner A A-class item, and the interface document shared-file table still shows `_jmcg_context_feedback_from_state` as not started. Without a W2 handoff log, Owner B should not treat pocket/intent steering as accepted.
- [ ] Confirm whether Owner A changed any C1/C2/C3 schema or predicate. If yes, require entry in the interface document contract-change table before validation.
- [ ] Confirm Owner A did not require Owner B to modify `generator_coord/agent.py` or `hfm_3d/generator.py`; those are Owner A primary files and Owner B is read-only there.

### P1: Run W4 Only After Explicit Test Authorization

- [ ] Request explicit user authorization before running pytest.
- [ ] After authorization, run the acceptance commands listed in the interface document:

```bash
uv run pytest tests/unit/test_graph_repo.py -q
uv run pytest tests/unit/test_mf_eval.py -q
uv run pytest tests/benchmark -q
uv run pytest tests/unit/test_generator_coord_agent.py tests/unit/test_generators.py -q
uv run pytest tests/unit/test_validation_agent.py tests/unit/test_srb_agent.py -q
uv run pytest tests/unit -q
uv run pytest -q
```

- [ ] Include Owner A W2 validation commands once Owner A hands off W2:

```bash
uv run pytest tests/unit/test_generators.py tests/unit/test_service_artifact_status.py -q
```

- [ ] For every failure, keep the original command, exit code, failing test, file, line, and stderr. Do not change business logic to bypass a failure.
- [ ] If the failure belongs to Owner A scope, hand it back to Owner A with the exact failing command and output. If the failure belongs to Owner B scope, define scope before editing and register any shared-file occupancy first.
- [ ] After each successful validation group, append execution logs to both source documents' execution-log sections.

### P2: Finish Owner B Production Gates

- Resource preflight 2026-06-04:
  - H1 passed and is logged in both source documents.
  - Blocked by missing source-document env/resources: H5 oracle commands, H8 official benchmark data, H2 Sigstore/Rekor production chain, H4 L4 quantum command, H6 retrosynthesis runner JSON, and H11 `KRAS_E2E_SCOPE` plus H2/H5/H6 prerequisites.
  - Set but not sufficient for completion: `HFM_CHECKPOINT_PATH`, `HFM_DECODER_PATH`, `CRITIC_AGENT_READY`, `ORCHESTRATOR_E2E_READY`, `PROVENANCE_STORE_MODE`, `RUN_KRAS_G12C_E2E`.

- [x] H1 -> W1 real acceptance:
  - Resource owner: Owner B.
  - Required env: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `MINIO_ENDPOINT_URL`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`, `QDRANT_HOST` or `QDRANT_URL`, `REDIS_HOST` or `REDIS_URL`, `TEST_DATABASE_URL`, `PROVENANCE_DATABASE_URL`, `FEAST_REPO_PATH`.
  - Expected evidence: integration tests that were skipped without DKI resources turn pass after resource deployment.
  - Verified 2026-06-04: `bash -lc 'set -a; source .env; set +a; unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"; uv run pytest tests/integration/test_dki_*.py -q'` exited 0 with 10 passed, 0 skipped. Warning retained: Qdrant client 1.18.0 is outside the recommended minor-version range for server 1.12.4.

- [ ] H5 -> W3 production oracle acceptance:
  - Resource owner: Owner B.
  - Required env: `DOCK_ORACLE_COMMAND`, `BOLTZ2_ORACLE_COMMAND`, `FEP_ORACLE_COMMAND`, `ADMET_ORACLE_COMMAND`.
  - Expected evidence: PCBO provider/evaluator can call local L0-L3 oracle path rather than only embedding proxy fallback.

- [ ] H8 -> W5 official benchmark acceptance:
  - Resource owner in task split: Owner A for generation artifact validation; Owner B consumes it for W5.
  - Required env/data: `MOSES_REFERENCE_SMILES_PATH`, `PMO_SCORE_TABLE_PATH`, `CROSSDOCKED_BENCHMARK_JSONL`, `HFM_CHECKPOINT_PATH`, `HFM_DECODER_PATH`, formal thresholds.
  - Expected evidence: official data path runs without code changes. Benchmark result must not be reported as official before these resources exist.

- [ ] H5 + H10 -> W12 real scorer and cluster acceptance:
  - Resource owner: Owner B.
  - Required resources: real DiffDock-L runner, pharmacophore scorer runner, HUMU scorer runner, cluster deployment.
  - Expected evidence: W12 scorer path uses real runners and passes cluster validation.

- [ ] H2 production audit chain:
  - Resource owner: Owner B.
  - Required env: `SIGSTORE_SIGN_COMMAND`, `SIGSTORE_VERIFY_COMMAND`, `SIGSTORE_IDENTITY_TOKEN`, `SIGSTORE_EXPECTED_IDENTITY`, `SIGSTORE_REKOR_URL`.
  - Acceptance command after resources exist:

```bash
env RUN_AUDIT_E2E=1 uv run pytest tests/e2e/test_audit_completeness.py
```

- [ ] H4 L4 quantum correction:
  - Resource owner: Owner B.
  - Required env: `L4_QUANTUM_ORACLE_COMMAND` or `L4_GPU4PYSCF_COMMAND` / `L4_ORCA_COMMAND`.

- [ ] H6 retrosynthesis production runners:
  - Resource owner: Owner B.
  - Required env: `RETROSYN_PLANNER_COMMANDS_JSON` or the documented named runner env values.

- [ ] H11 KRAS G12C full pilot:
  - Resource owner: Owner B.
  - Prerequisites: H1, H2, H5, H6, and service readiness.
  - Required env includes `CRITIC_AGENT_READY`, `ORCHESTRATOR_E2E_READY`, `PROVENANCE_STORE_MODE=production_real`.
  - The two source documents specify the run flags `RUN_KRAS_G12C_E2E=1` and `KRAS_E2E_SCOPE=full`, but do not specify the exact pytest file. Verify the repository test target before execution and log the verified command.

### P3: Contract Checks With Owner A

- [ ] C1 generator feedback:
  - Owner B side must keep property/intent/pocket records compatible with `moleculeforge.jmcg.feedback.v1`.
  - Non-steering property records must not change HFM latent.
  - Owner A W2 must provide 129-dimensional HUMU embeddings for steering-capable pocket/intent records. Otherwise HFM drops the record.

- [ ] C2 CRG predicates:
  - Owner B must keep `route_humu_embedding` payload parseable by Owner A: `humu_embedding`, `route_id`, optional `source`, `weight`, `polarity`, `confidence`, `evidence_ids`, `metadata`.
  - New predicate names require contract-change registration and Owner A confirmation before implementation.

- [ ] C3 HUMU encoder:
  - Owner B RetroSynAgent route encoding must continue to write `route_humu_embedding`.
  - Owner A W2/W8 and Owner B route encoding must use the same `HUMU_CHECKPOINT_PATH`, currently documented as `checkpoints/humu/best_model.pt`.

### P4: Shared-File Discipline

- [ ] Before any Owner B edit to `services/orchestrator-svc/src/orchestrator_svc/main.py`, register the exact function/section in the interface document shared-file table.
- [ ] Do not edit `agents/generator_coord/src/generator_coord/agent.py` or `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py` unless the scope is re-approved and registered.
- [ ] Do not alter C1/C2/C3 fields silently. Add a contract-change row first and wait for Owner A confirmation.

### P5: Execution Log Fields

Append after every key operation with these fields:

- Date and role: use `2026-06-04（乙）` for today's entries.
- ID and operation: W4 validation, H1 deployment, H5 oracle acceptance, H8 benchmark data acceptance, H10 cluster validation, H11 pilot, or the exact work item being handled.
- Changed files: list concrete paths, or write `未改业务代码`.
- Verification command: list the exact command that was run, or write `未跑 pytest，未获授权`.
- Actual result: include exit code and key evidence.
- Remaining gate: list the exact external resource, Owner A handoff, or `无`.
- Contract change: write `C1/C2/C3 无变更`, or reference the registered contract-change row.

Use this in both:

- `docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
- `docs/architecture/corearchitecture-v2-completion-tasksplit.md`

## Not In Scope

- D1 HUMU SE(3)/E(3) tri-tower upgrade.
- D2 NATS JetStream migration.
- D3 MMPT patent RAG and Seq2Seq production upgrade.
- D4 SRB to SiLA2 real wet-lab hardware loop.

These four items are explicitly frozen or excluded by the source documents.
