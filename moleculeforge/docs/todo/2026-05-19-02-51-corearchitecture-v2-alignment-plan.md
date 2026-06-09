# CoreArchitecture v2 实现对齐方案

## 目标

根据 `docs/architecture/current-implementation-vs-corearchitecture-v2.md`，继续补齐当前实现与 `MoleculeForge_CoreArchitecture_v2.md` 的差距。当前正在运行的 HUMU baseline 预训练不停止、不重启、不复用为新联合目标证据。

## 当前证据

- HUMU 训练进程仍在运行：主 PID `285291`，命令为 `python -u -m torch.distributed.run --standalone --nproc_per_node=4 pipelines/humu_pretrain/train.py --config /tmp/humu_4h200.yaml --resume checkpoints/humu_4h200/checkpoint_epoch_0003.pt`。
- 当前运行日志仍只包含 `l_mol_pocket` 和 `l_mol_route`，不包含 `l_pocket_route`、`l_intent`、`l_curvature_reg`。
- `configs/models/humu_pretrain.yaml` 已声明 `data.joint_source` 和 `data.intent_source`，但 `data/processing/humu_pretrain/` 下当前只有 `mol`、`pocket`、`route`、`route_eval`，没有 `joint` 和 `intent` 目录。
- CrossDocked pocket 原始 ligand SMILES 与 USPTO route root SMILES 的 raw exact match 为 0：`pocket_unique_raw=5471`、`route_total=409035`、`exact_raw_matches=0`。
- 前 1000 条 pocket canonical 与前 10000 条 route canonical 抽样匹配为 0：`sample_pocket_unique_canon=303`、`sample_route_valid=10000`、`sample_canon_matches=0`。
- DKI/E2E 所需环境变量当前均未设置：`TEST_DATABASE_URL`、`NEO4J_URI`、`QDRANT_HOST` 或 `QDRANT_URL`、`MINIO_ENDPOINT_URL`、`REDIS_HOST` 或 `REDIS_URL`、`RUN_KRAS_G12C_E2E`、`RUN_AUDIT_E2E` 等。
- `docker compose -f infra/docker/docker-compose.test.yml config` 可解析，但未在本轮启动 stack；既有记录显示当前环境曾阻塞于 Docker layer 注册权限。
- Provenance service 已区分 `local_demo` 和 `production_real` store；默认 `local_demo` 仍为 in-memory，`production_real` 缺 Neo4j/Postgres/MinIO 配置时 fail-fast。
- Orchestrator LangGraph state 已贯穿 `run_id`、`trace_id`、`events`。
- `pipelines/humu_pretrain/train.py --preflight-only` 已用于 HUMU 数据契约检查；当前 `/tmp/humu_4h200.yaml` 预检会失败，错误为 `FileNotFoundError: data.joint_source is required for HUMU pretraining`，符合当前旧 baseline 配置不能标记为新联合目标训练的事实。

## 架构差异

### 1. CIC / CIG / HCIV

设想要求真实 SRM/LLM parser、grounding evidence 和 learned HCIV。当前源码已有 production/local_demo 边界，缺口主要是外部 semantic parser URI、HCIV checkpoint 和生产运行证据。

### 2. HUMU / JMCG


### 3. HUMU Encoders

设想要求 SE(3)/E(3)/reaction graph foundation 级编码器。当前是 RDKit graph features、pocket point features、reaction string features 和 CIG feature MLP，属于可运行轻量实现，不是 foundation encoder。

### 4. AMGE / TAR / KD

设想要求多类生成器均有真实权重和采样能力。当前保留 HFM、FragFM、CReM、MMPT、ICLM、UAS 六类生成器；缺口是真实 runner、真实权重和跨生成器反馈闭环。

### 5. MARB / CRG / Provenance

设想要求 agent 共享 CRG，并将 graph、event、artifact 分别写入 Neo4j、Postgres、MinIO。当前 Orchestrator 已接 LangGraph 状态机，但 Provenance 仍是进程内记录，CRG 和 trace 还没有生产级持久化闭环。

### 6. Oracle / DKI / E2E

设想要求 L0-L4 oracle、DKI、KRAS pilot、audit E2E 形成证据。当前 wrapper 和 fail-fast 边界存在，但真实后端、模型、runner、benchmark/E2E 证据缺失。

## 调用链路分析

目标链路：

```text
自然语言目标
  -> CIC semantic parser + grounding
  -> CIG
  -> learned HCIV
  -> HUMU mol/pocket/route/intent joint manifold
  -> TAR 选择生成器
  -> AMGE 真实生成
  -> Oracle cascade
  -> CRG + Provenance + trace
  -> DKI Postgres/Neo4j/Qdrant/MinIO/Redis
  -> KRAS/Audit E2E 和 benchmark 证据
```

当前链路：

```text
HUMU baseline training
  -> paired mol-pocket / mol-route loss
  -> checkpoint_epoch_0015.pt / best_model.pt

服务和 agent
  -> 部分 production path fail-fast
  -> Orchestrator LangGraph state
  -> Provenance in-memory chain
  -> DKI integration tests 因环境变量缺失 skip
```

关键断点：

- 当前本地 pocket 与 route 数据没有可直接证明的三元配对交集，不能拼接伪造 `mol-pocket-route`。
- Provenance 只在进程内保存记录，重启即丢失，不能满足 CRG/DKI 审计链。
- E2E 当前依赖环境标记和外部服务，未形成真实完成证据。

## 实施方案

### 阶段 1：HUMU 新联合目标数据契约

目标：不停止当前 baseline run，新增独立的数据契约和 smoke 入口，确保没有真实三元配对时 fail-fast。

涉及文件：

- `pipelines/humu_pretrain/src/humu_pretrain/data_loader.py`
- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
- `configs/models/humu_pretrain.yaml`
- `tests/unit/test_humu_training.py`
- 需要新建或更新的 HUMU 数据准备入口，路径需在实施前确认。

执行内容：

- 已增加 `preflight_humu_data_contract()` 和 `train.py --preflight-only`，验证非空、schema、SMILES 合法性、route reactions、pocket coordinates。
- 只允许由真实匹配或用户提供的三元配对文件生成 `mol-pocket-route`。当前 raw exact match 为 0，因此不能自动把 CrossDocked 和 USPTO 组合成 joint 数据。
- 增加单卡 smoke 配置，启用 `pocket_route`、`intent`、`curvature_reg`，日志必须包含对应 loss。
- 当前 baseline 继续运行，新 smoke 使用独立 output_dir 和日志路径。

验收：

- 缺 `joint_source` 或 `intent_source` 时明确失败。
- 有真实 joint/intent fixture 时，训练日志包含 `l_pocket_route`、`l_intent`、`l_curvature_reg`。
- 不产生伪 joint 数据。

执行记录：

- 已新增 HUMU 数据契约预检，函数入口为 `pipelines/humu_pretrain/src/humu_pretrain/data_loader.py::preflight_humu_data_contract`。
- 已新增 CLI 入口：`uv run python pipelines/humu_pretrain/train.py --config <config> --preflight-only`。
- 已验证当前 `/tmp/humu_4h200.yaml` 会 fail-fast：启用 `pocket_route` 但未提供 `joint_source`。
- 已新增单元测试覆盖有效 joint/intent fixture 和缺 joint records 的失败路径。

未完成：

- 尚未生成真实 `data/processing/humu_pretrain/joint`。
- 尚未生成真实 `data/processing/humu_pretrain/intent`。

### joint_source 数据获取规范

`joint_source` 的目标是提供真实的 `mol-pocket-route` 三元正样本。每条记录必须证明同一个 ligand 同时有口袋坐标和合成路线，不能把无关 pocket 与 route 按索引拼接。

当前代码接受 JSONL 文件。每条 joint 记录至少需要：

```json
{
  "id": "joint-000001",
  "pdb_id": "1ABC_A",
  "pocket_path": "pocket_000001.json",
  "ligand_smiles": "CCO",
  "route_id": "route-000001",
  "reactions": ["CCBr>>CCO"],
  "target_id": "KRAS_G12C",
  "source_dataset": "verified_joint_source",
  "split": "train"
}
```

其中：

- `ligand_smiles` 必须是同一个 ligand 的真实结构，且 RDKit 可解析。
- `pocket_path` 指向同目录下的 pocket sidecar，sidecar 必须包含 `pocket_atoms`，每个 atom 需要 `x`、`y`、`z`、`element`、`residue`。
- `reactions` 必须是非空 reaction list，每条 reaction 必须是 `reactants>>products` 格式。
- `route_id` 必须指向该 ligand 的真实或可追溯 retrosynthesis route。
- `target_id` 用于和 intent 数据匹配。

获取路径按可信度排序：

1. 真实三元数据源：从同一个项目或数据库导出 target-pocket、ligand、route 三者已经绑定的记录。该路径最好，能直接作为正式 joint 训练数据。
2. 对 CrossDocked ligand 运行真实 retrosyn runner：以 `data/processing/humu_pretrain/pocket/index.jsonl` 中的 `ligand_smiles` 为输入，用 AiZynthFinder/RSGPT/UAlign 等真实 runner 生成 route；只有 runner 返回含 `route_id`、`steps`、`reaction`、`reactants`、`conditions`、`building_blocks` 的结果时，才可写入 joint。
3. 从 route 数据反查 pocket：以 USPTO route 的 `root_smiles` 为 ligand identity，匹配真实 pocket 数据。当前 raw exact match 为 0，前 1000 pocket 与前 10000 route canonical 抽样 match 也为 0，因此当前本地 CrossDocked + USPTO 不能直接生成有效 joint。
4. 小样本 smoke fixture：允许人工整理少量真实 pocket sidecar 和同 ligand route，用于验证训练链路；不得标记为正式训练数据。

禁止路径：

- 按 batch index、文件顺序或随机方式把 pocket 与 route 拼接。
- 用固定 reaction、空 route、hash、random、模板字符串补 route。
- 用当前 CrossDocked ligand 和 USPTO route 强行组合成 `mol-pocket-route`。

### intent_source 数据获取规范

`intent_source` 的目标是给 joint 或 paired 样本提供 CIG/objective/constraint features。当前代码按 `mol_id`、`target_id`、`ligand_smiles`、`source_dataset` 依次匹配 intent 记录。

当前代码接受 JSONL 文件。每条 intent 记录至少需要 `targets` 或 `objective_nodes`：

```json
{
  "intent_id": "intent-kras-g12c-000001",
  "target_id": "KRAS_G12C",
  "targets": {
    "binding_affinity": -9.5,
    "selectivity": 100.0
  },
  "weights": {
    "binding_affinity": 1.0,
    "selectivity": 0.5,
    "sa_score": 0.2
  },
  "constraints": {
    "logp": [1.0, 4.0],
    "mw": [350.0, 550.0]
  },
  "objective_nodes": [],
  "edges": []
}
```

获取路径：

1. 从 CIC production 输出导出：通过 semantic parser + grounding 生成 CIG，再把 CIG 中的 objective nodes、weights、constraints、target_id 写成 intent JSONL。这是正式路径。
2. 从已有 CIG/HCIV 训练样本导出：如果已有 CIG JSON-LD 或内部 CIG 对象，提取 `target_id`、目标值、权重、约束和边。
3. 从任务配置人工整理：只适合小样本 smoke。必须来自真实任务定义，不允许为了让训练运行而编造目标。

匹配规则：

- joint 记录有 `target_id` 时，intent 记录应优先使用相同 `target_id`。
- 如果一个 target 有多套意图，应增加 `mol_id`、`ligand_smiles` 或 `id` 做唯一匹配，避免 `IntentDataset` 发现重复 key 后失败。
- 如果启用 `loss_weights.intent > 0`，没有匹配 intent 的训练样本会失败。

### 阶段 2：Provenance 持久化边界

目标：将 Provenance 从纯 in-memory 改为可配置 backend，生产模式写 Neo4j/Postgres/MinIO，缺配置直接失败。

涉及文件：

- `services/provenance-svc/src/provenance_svc/main.py`
- `services/provenance-svc/src/provenance_svc/models.py`
- `libs/mf-core/src/mf_core/db/repositories/graph_repo.py`
- `libs/mf-core/src/mf_core/db/orm/__init__.py`
- `libs/mf-core/src/mf_core/db/minio_client.py`
- `tests/unit/test_provenance.py`
- `tests/unit/test_service_artifact_status.py`

执行内容：

- 已增加 provenance store adapter，区分 `local_demo` 和 `production_real`。
- `production_real` 要求 Neo4j/Postgres/MinIO 配置；缺失时服务健康检查不能返回伪健康。
- 生产 store 写入 artifact graph、audit event 和 object store。
- `local_demo` 保持 in-memory 行为用于单元测试，响应标明 `provenance_store: in_memory`。

验收：

- 单元测试覆盖 in-memory adapter 和 production adapter 的 fake client 写入。
- 无真实 DKI 环境时不声称生产持久化通过。

执行记录：

- 已更新 `services/provenance-svc/src/provenance_svc/main.py`。
- 已新增测试覆盖 production 缺 DKI 配置时 503，以及自定义 store 委托写入。

未完成：

- 当前环境仍缺真实 Neo4j/Postgres/MinIO 配置，未完成生产 DKI 写入验收。

### 阶段 3：CRG 与 trace 贯穿

目标：让 Orchestrator workflow state 至少携带 run_id、trace_id、artifact_ids，并将这些字段传给 Provenance 和后续服务边界。

涉及文件：

- `agents/orchestrator/src/orchestrator/workflow/graph_builder.py`
- `services/orchestrator-svc/src/orchestrator_svc/main.py`
- `libs/mf-telemetry/src/mf_telemetry/tracing/opentelemetry.py`
- `tests/test_mvp_pipeline.py`
- `tests/e2e/test_audit_completeness.py`

执行内容：

- 已在 workflow 初始 state 中生成或接收 `run_id` 和 `trace_id`。
- 每个状态迁移写入 structured event。
- E2E 继续受环境标记控制，但 preflight 必须列出缺失依赖，不能只用空 assert。

验收：

- Orchestrator status 返回 `run_id`、`trace_id`、`history`、`events`。
- Audit E2E 在未配置环境时输出明确缺失项；配置齐全时走真实服务。

执行记录：

- 已更新 `agents/orchestrator/src/orchestrator/workflow/graph_builder.py`，state 包含 `run_id`、`trace_id`、`artifact_ids`、`events`。
- 已更新 `services/orchestrator-svc/src/orchestrator_svc/main.py`，REST/gRPC 响应透出 `run_id` 和 `trace_id`。
- 修改 `graph_builder.py` 前因文件属主为 `FL`，已将该文件属主调整为 `FWY`。

### 阶段 4：外部依赖补证

目标：把无法由当前代码生成的真实资产列为硬依赖，避免伪造完成。

需要用户或环境提供：

- `joint_source`：真实 `mol-pocket-route` 三元配对数据。
- `intent_source`：真实 CIG/objective/constraint features。
- DKI：Postgres、Neo4j、Qdrant、MinIO、Redis 可用连接。
- 模型/runner：HUMU 新 checkpoint、HFM decoder、ADMET、Boltz、DiffDock/GNINA、OpenFE、Retrosyn runner。
- E2E 标记：`RUN_KRAS_G12C_E2E=1`、`RUN_AUDIT_E2E=1` 以及对应服务地址。

## KISS 四问

1. 这是现实问题还是想象问题？
   - 是现实问题。缺口有文件、配置、日志和环境变量证据支撑。
2. 有没有更简单的做法？
   - 有。先补数据契约、持久化边界和 trace 字段，不同时重写全部保留生成器或启动新长训。
3. 会破坏什么？
   - 风险是 production 默认 fail-fast 后旧 demo 不能作为生产成功返回。缓解方式是保留显式 `local_demo`。
4. 当前项目真的需要这个功能吗？
   - 需要。用户目标是实现与 CoreArchitecture v2 一致，而不是继续保留只能局部运行的 baseline。

## 风险

- 当前本地数据不能证明存在真实 `mol-pocket-route` 三元配对。若用户不提供 joint 数据，HUMU 新联合目标只能完成代码和 smoke fixture，不能完成真实训练证据。
- 当前 DKI 环境变量缺失。没有真实 backend 时，只能验证 adapter 合约，不能声称 DKI 生产闭环完成。
- 当前外部模型和 runner 缺失。不能生成真实 AMGE/Oracle/E2E 结果。
- 当前 HUMU baseline 正在占用 GPU。新 smoke 需要等用户授权使用空闲资源或明确另行安排。

## 验证记录

- `uv run pytest tests/unit/test_humu_training.py tests/unit/test_service_artifact_status.py tests/test_mvp_pipeline.py tests/unit/test_provenance.py -q`
  - 结果：69 passed
- `uv run ruff check --select F,I,E501 pipelines/humu_pretrain/src/humu_pretrain/data_loader.py pipelines/humu_pretrain/train.py services/provenance-svc/src/provenance_svc/main.py agents/orchestrator/src/orchestrator/workflow/graph_builder.py services/orchestrator-svc/src/orchestrator_svc/main.py tests/unit/test_humu_training.py tests/unit/test_service_artifact_status.py tests/test_mvp_pipeline.py`
  - 结果：All checks passed
- `uv run python pipelines/humu_pretrain/train.py --config /tmp/humu_4h200.yaml --preflight-only`
  - 结果：按预期失败，`FileNotFoundError: data.joint_source is required for HUMU pretraining`

## 下一步

- 准备真实 `joint_source` 和 `intent_source`。
- 使用 `--preflight-only` 检查数据契约。
- 在不停止当前 HUMU baseline 的前提下，另开独立 output_dir 运行单卡新联合目标 smoke。
- 补齐 DKI 配置后，再验证 Provenance production store。
