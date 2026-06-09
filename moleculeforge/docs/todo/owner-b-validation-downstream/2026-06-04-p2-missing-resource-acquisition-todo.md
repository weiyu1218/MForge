# P2 Missing Resource Acquisition Todo

## Source

- `docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
- `docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- `docs/todo/owner-b-validation-downstream/2026-06-04-owner-b-followup-todo.md`

## Current Status

P2 is no longer blocked by H1. H1 -> W1 real DKI acceptance passed on 2026-06-04 with `tests/integration/test_dki_*.py`: 10 passed, 0 skipped.

The remaining P2 blockers are external resources and production configuration. These are not new AI coding tasks. Do not replace them with mock commands, local fallback paths, synthetic datasets, or placeholder credentials.

Current preflight shows:

- Missing: H5 oracle commands, H8 official benchmark data, H2 Sigstore/Rekor production chain, H4 L4 quantum command, H6 retrosynthesis planner commands, H10/W12 CReM scorer targets, and H11 `KRAS_E2E_SCOPE`.
- Present but not sufficient by itself: `HFM_CHECKPOINT_PATH`, `HFM_DECODER_PATH`, KRAS model/artifact flags, DKI env, `CRITIC_AGENT_READY`, `ORCHESTRATOR_E2E_READY`, `PROVENANCE_STORE_MODE`, `RUN_KRAS_G12C_E2E`, and `CREM_MMP_DB_PATH`.

## Acquisition Rules

- Store secrets in `.env` or the production secret manager. Do not commit credential values.
- Every command env must point to a real executable command. The first executable token must pass preflight.
- Every dataset path must point to a real file with the schema consumed by the existing tests.
- After each resource is acquired, rerun the smallest matching preflight first, then run the source-document acceptance command only when resources are complete.
- After each key operation, append an execution-log entry to both source documents with command, exit code, result, remaining gate, and `C1/C2/C3 无变更`.

## H5: L1-L3 Oracle Runner Commands

### What Is Missing

- `DOCK_ORACLE_COMMAND`
- `BOLTZ2_ORACLE_COMMAND`
- `FEP_ORACLE_COMMAND`
- `ADMET_ORACLE_COMMAND`

### What These Resources Do

These commands let validation and PCBO call real L1-L3 oracle paths instead of only local fallback or embedding-proxy scoring:

- Docking / DiffDock / GNINA style docking score.
- Boltz-2 affinity inference.
- FEP / OpenFE free-energy estimation.
- ADMET prediction runner.

H5 unlocks W3 production oracle acceptance and is also a prerequisite for H11 full KRAS pilot in the task split.

### How To Get Them

- [ ] Ask the oracle/platform owner for production JSON command wrappers for docking, Boltz-2, FEP/OpenFE, and ADMET.
- [ ] Confirm each wrapper uses the schema expected by the corresponding service implementation:
  - `services/dock-svc/src/dock_svc/main.py`
  - `services/boltz2-svc/src/boltz2_svc/main.py`
  - `services/fep-svc/src/fep_svc/main.py`
  - `services/admet-svc/src/admet_svc/main.py`
- [ ] Confirm required local artifacts and tools are installed before setting the command env. Current `.env` already has related artifact/tool keys such as Boltz, GNINA/DiffDock, OpenFE, and ADMET paths, but H5 is not complete until command env values are present and executable.
- [ ] Put the real command strings into `.env` or production secrets for these env keys: `DOCK_ORACLE_COMMAND`, `BOLTZ2_ORACLE_COMMAND`, `FEP_ORACLE_COMMAND`, and `ADMET_ORACLE_COMMAND`.
- [ ] Run command availability preflight before any production acceptance.
- [ ] Then run W3/H5 acceptance and append execution logs to both source documents.

## H8: Official Benchmark Data

### What Is Missing

- `MOSES_REFERENCE_SMILES_PATH`
- `PMO_SCORE_TABLE_PATH`
- `CROSSDOCKED_BENCHMARK_JSONL`
- Formal benchmark thresholds.

Already set but not sufficient:

- `HFM_CHECKPOINT_PATH`
- `HFM_DECODER_PATH`

### What These Resources Do

These files move W5 benchmark from resource-gated skip/smoke behavior to official benchmark execution:

- MOSES reference SMILES: distribution-learning reference set.
- PMO score table: objective scores for PMO tasks such as DRD2, JNK3, and GSK3B.
- CrossDocked JSONL: pocket-ligand benchmark records for pocket-conditioned validation.
- Formal thresholds: the pass/fail criteria that make reported benchmark numbers official.

### How To Get Them

- [ ] Coordinate with Owner A because tasksplit assigns H8 generation artifact validation to Owner A while Owner B consumes it for W5.
- [ ] Ask the benchmark/data owner to place official benchmark files in a stable project data path, not a temporary download directory.
- [ ] Validate file shape before running benchmark:
  - `MOSES_REFERENCE_SMILES_PATH`: text file readable as SMILES rows.
  - `PMO_SCORE_TABLE_PATH`: scored table containing the columns consumed by `tests/benchmark/pmo_benchmark.py`, including `drd2`, `jnk3`, and `gsk3b`.
  - `CROSSDOCKED_BENCHMARK_JSONL`: JSONL records with `pocket_id`, `ligand_smiles`, `split`, and optional `docking_score`.
- [ ] Agree formal threshold env values with Owner A and the benchmark owner before reporting official metrics.
- [ ] Put the real benchmark file paths into `.env` or production config for these env keys: `MOSES_REFERENCE_SMILES_PATH`, `PMO_SCORE_TABLE_PATH`, and `CROSSDOCKED_BENCHMARK_JSONL`.
- [ ] Run `uv run pytest tests/benchmark -q` only after data and thresholds are present.

## H10/W12: CReM-Pharm-3D Real Scorer And Cluster Acceptance

### What Is Missing

- `CREM_DOCK_ORACLE_TARGET`
- `CREM_PHARMACOPHORE_SCORER_COMMAND`
- `CREM_HUMU_SCORER_COMMAND`
- H10 cluster deployment validation.

Already set and file exists:

- `CREM_MMP_DB_PATH`

Optional timeout env:

- `CREM_SCORER_COMMAND_TIMEOUT_SECONDS`

### What These Resources Do

These resources make W12 use real scoring paths when CReM-pharm-3D ranks generated mutations:

- `CREM_DOCK_ORACLE_TARGET`: gRPC target for real docking oracle scoring.
- `CREM_PHARMACOPHORE_SCORER_COMMAND`: command wrapper for pharmacophore scoring.
- `CREM_HUMU_SCORER_COMMAND`: command wrapper for HUMU alignment/embedding scoring.
- H10 cluster validation proves the configured service starts and sees mounted artifacts/secrets in the real deployment environment.

### How To Get Them

- [ ] Ask the oracle/platform owner for the real Dock Oracle gRPC endpoint and set `CREM_DOCK_ORACLE_TARGET`.
- [ ] Ask the pharmacophore owner for a real JSON scorer command and set `CREM_PHARMACOPHORE_SCORER_COMMAND`.
- [ ] Ask the HUMU owner for a real HUMU scorer command and set `CREM_HUMU_SCORER_COMMAND`.
- [ ] Confirm command behavior against `services/crem-generator-svc/src/crem_generator_svc/main.py`; do not invent a new schema.
- [ ] Ask the cluster/platform owner to deploy the CReM generator service with `CREM_MMP_DB_PATH` mounted and scorer secrets injected.
- [ ] Run W12 scorer path validation only after all three scorer inputs and H10 cluster readiness are available.

## H2: Sigstore/Rekor Production Audit Chain

### What Is Missing

Source-document H2 env:

- `SIGSTORE_SIGN_COMMAND`
- `SIGSTORE_VERIFY_COMMAND`
- `SIGSTORE_IDENTITY_TOKEN`
- `SIGSTORE_EXPECTED_IDENTITY`
- `SIGSTORE_REKOR_URL`

Audit E2E additionally checks:

- `PROVENANCE_SVC_URL`
- `SIGSTORE_E2E_READY`
- `OTEL_EXPORTER_OTLP_ENDPOINT`

### What These Resources Do

H2 turns local development signatures into verifiable production audit records. It is required for audit E2E and is also part of the H11 full KRAS pilot prerequisite chain.

### How To Get Them

- [ ] Ask the security/platform owner whether the project will use deployed Fulcio/Rekor or a managed Sigstore/Rekor service.
- [ ] Obtain the production Rekor URL and set `SIGSTORE_REKOR_URL`.
- [ ] Obtain the signing identity token and expected identity from the identity provider/security owner.
- [ ] Obtain or package real sign/verify command wrappers compatible with:
  - `services/provenance-svc/src/provenance_svc/domain/sigstore_integration.py`
  - `libs/mf-agents/src/mf_agents/lineage/sigstore_signer.py`
- [ ] Put the real Sigstore/Rekor values into `.env` or production secrets for these env keys: `SIGSTORE_SIGN_COMMAND`, `SIGSTORE_VERIFY_COMMAND`, `SIGSTORE_IDENTITY_TOKEN`, `SIGSTORE_EXPECTED_IDENTITY`, and `SIGSTORE_REKOR_URL`.
- [ ] For audit E2E, also set service and telemetry readiness env required by `tests/e2e/test_audit_completeness.py`.
- [ ] Then run:

```bash
env RUN_AUDIT_E2E=1 uv run pytest tests/e2e/test_audit_completeness.py
```

## H4: L4 Quantum Correction

### What Is Missing

At least one of:

- `L4_QUANTUM_ORACLE_COMMAND`
- `L4_GPU4PYSCF_COMMAND`
- `L4_ORCA_COMMAND`

### What These Resources Do

H4 provides the high-fidelity L4 quantum correction path used by validation when L4 oracle execution is requested.

### How To Get Them

- [ ] Ask the quantum/oracle owner which backend will be used: generic L4 command, GPU4PySCF, or ORCA.
- [ ] Install and validate the selected backend on the target machine or cluster image.
- [ ] Package the selected backend as a command wrapper accepted by `agents/validation_agent/src/validation_agent/agent.py`.
- [ ] Set only the real selected command env. Do not set fake commands to bypass preflight.
- [ ] Run L4 validation smoke after the executable and required artifacts are present.

## H6: Retrosynthesis Production Runners

### What Is Missing

Preferred source-document env:

- `RETROSYN_PLANNER_COMMANDS_JSON`

Implementation also supports:

- `RETROSYN_PLANNER_COMMAND`
- `RASCORE_PLANNER_COMMAND`
- `RSGPT_PLANNER_COMMAND`
- `UALIGN_PLANNER_COMMAND`
- `AIZYNTH_PLANNER_COMMAND`

Already set but not sufficient:

- `AIZYNTH_CONFIG_PATH`
- `AIZYNTH_STOCK`
- `AIZYNTH_EXPANSION_POLICY`
- `AIZYNTH_FILTER_POLICY`

### What These Resources Do

H6 lets RetroSynAgent and retrosyn-svc call real route planners instead of only local configuration artifacts. It is a required prerequisite for H11 full KRAS pilot.

### How To Get Them

- [ ] Ask retrosynthesis/model owner which production engines are available: RAscore, RSGPT, UAlign, AiZynth, or a smaller approved subset.
- [ ] For each selected engine, obtain a real command wrapper compatible with:
  - `agents/retrosyn_agent/src/retrosyn_agent/agent.py`
  - `services/retrosyn-svc/src/retrosyn_svc/main.py`
- [ ] Prefer `RETROSYN_PLANNER_COMMANDS_JSON` for multi-engine production configuration.
- [ ] If production chooses named envs instead, set the specific `RASCORE_PLANNER_COMMAND`, `RSGPT_PLANNER_COMMAND`, `UALIGN_PLANNER_COMMAND`, or `AIZYNTH_PLANNER_COMMAND` values.
- [ ] Confirm route planner command output contains real route dictionaries; RAscore command output must contain a real retrosynthetic accessibility score. Do not accept empty or synthetic output.
- [ ] Run production retrosynthesis inference smoke after commands are available.

## H11: KRAS G12C Full Pilot

### What Is Missing

- `KRAS_E2E_SCOPE`
- H2 Sigstore/Rekor production chain.
- H5 oracle commands.
- H6 retrosynthesis production runners.

Already set in current preflight:

- `RUN_KRAS_G12C_E2E`
- `CRITIC_AGENT_READY`
- `ORCHESTRATOR_E2E_READY`
- `PROVENANCE_STORE_MODE`
- HFM/Boltz/AiZynth artifact env checked by `tests/e2e/test_kras_g12c_pilot.py`.
- DKI env checked by `tests/e2e/test_kras_g12c_pilot.py`.

### What These Resources Do

H11 is the full end-to-end pilot gate. It proves the production-style KRAS G12C workflow can run with real DKI, audit chain, oracle runners, retrosynthesis runners, and service readiness.

### How To Get Them

- [ ] Do not attempt H11 before H2, H5, and H6 are acquired and logged.
- [ ] Set `KRAS_E2E_SCOPE=full` only when full pilot resources are actually present.
- [ ] Verify the repository target before execution. The current repository contains `tests/e2e/test_kras_g12c_pilot.py`, but the two source documents only specify run flags, so log the verified command explicitly.
- [ ] Run the full pilot with:

```bash
env RUN_KRAS_G12C_E2E=1 KRAS_E2E_SCOPE=full uv run pytest tests/e2e/test_kras_g12c_pilot.py -q
```

## Suggested Acquisition Order

- [ ] H5 first: it unlocks W3 production oracle acceptance and is an H11 prerequisite.
- [ ] H6 second: it unlocks real retrosynthesis and is an H11 prerequisite.
- [ ] H2 third: it unlocks audit E2E and is an H11 prerequisite.
- [ ] H10/W12 next: finish CReM scorer and cluster validation once oracle/scorer services exist.
- [ ] H8 in parallel with Owner A: official benchmark data and thresholds depend on generation artifact validation.
- [ ] H4 after runner/service owners are available unless a pilot path explicitly needs it first.
- [ ] H11 last: run only after H2, H5, H6, H1, and service readiness are logged.

## Handoff Template

Use this template after each resource domain is acquired:

```text
日期/角色: 2026-06-04（乙）
资源域: Hn 名称
投放内容: env 名称列表，不写密钥值
改动文件: 未改业务代码，或写实际变更的配置文件路径
验证命令: 实际执行的完整命令
实际结果: exit code、pass/fail/skip 数、关键 warning
剩余 gate: 具体剩余资源，或写无
契约变更: C1/C2/C3 无变更
```

Append the filled log to:

- `docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
- `docs/architecture/corearchitecture-v2-completion-tasksplit.md`
