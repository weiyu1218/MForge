# CoreArchitecture v2 Completion 执行方案

## 目标

完成 `MForge/MoleculeForge_CoreArchitecture_v2_完成度评估.md` 中两类缺口：

- `部分完成但未达到架构要求`
- `未完成或证据显示尚未落地`

本方案只覆盖核心架构落地，不新增前端、商业化、湿实验室硬件集成之外的功能。任何缺少真实数据、真实模型权重、真实外部服务配置的路径，实施时必须直接失败并输出可操作错误，不允许用随机数、固定池、hash、mock 或 placeholder 冒充完成。

## 当前证据

### 环境与资源

- 当前工作目录：`/workspace`
- 当前代码目录：`MForge/moleculeforge`
- `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader` 显示 4 张 H200 均空闲：
  - GPU 0：`NVIDIA H200`，`143771 MiB`，`0 MiB` used，`0%`
  - GPU 1：`NVIDIA H200`，`143771 MiB`，`0 MiB` used，`0%`
  - GPU 2：`NVIDIA H200`，`143771 MiB`，`0 MiB` used，`0%`
  - GPU 3：`NVIDIA H200`，`143771 MiB`，`0 MiB` used，`0%`
- `git status --short --branch` 在 `/workspace`、`MForge`、`MForge/moleculeforge` 均返回 `fatal: not a git repository`。执行前需要确认真实 git 仓库根目录，否则无法满足 `feature/<task-name>` 分支规范。
- `du -sh MForge/moleculeforge/data` 显示本地数据目录约 `1.3G`，未看到架构要求的大规模 ChEMBL/PDB/CrossDocked/PaRoutes/SureChEMBL/Enamine 数据闭环。

### 已确认的核心缺口

- CIC 主链路仍调用 `_heuristic_extract`，没有调用 `ground_knowledge`：`services/cig-compiler-svc/src/cig_compiler_svc/domain/compiler.py:36`
- `ground_knowledge` 独立存在但未进入主编译链路：`services/cig-compiler-svc/src/cig_compiler_svc/domain/stages/stage1b_grounding.py:7`
- learned HCIV 无权重时回退 hash：`services/cig-compiler-svc/src/cig_compiler_svc/domain/compiler.py:43`
- HUMU molecule encoder 从 SMILES hash 生成初始 embedding：`models/mf-encoders/humu_mol_encoder/src/mf_encoders/humu_mol/encoder.py:29`
- HUMU pocket encoder 对缺省坐标使用 `torch.randn`：`models/mf-encoders/humu_pocket_encoder/src/mf_encoders/humu_pocket/encoder.py:28`
- HUMU route encoder 基于 route 内容 hash seed：`models/mf-encoders/humu_route_encoder/src/mf_encoders/humu_route/encoder.py:31`
- HUMU pretrain 有训练循环，但 encoder 内部多为冻结/弱特征，且 stub wrapper 仍返回 `"status": "trained"`：`pipelines/humu_pretrain/src/humu_pretrain/pipeline.py:192`
- HFM-3D 解码仍从固定 SMILES 池选择：`models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py:76`
- FragFM、LaMGen、HFM service、EvoMol service、MMPT service、ICLM service 等仍存在随机或 hash 生成路径。
- Orchestrator graph builder 只返回节点/边描述，不是 LangGraph `StateGraph`：`agents/orchestrator/src/orchestrator/workflow/graph_builder.py:28`
- Orchestrator service 返回固定状态统计：`services/orchestrator-svc/src/orchestrator_svc/main.py:42`
- ADMET/Dock/Boltz2/FEP 服务仍随机或模拟返回：`services/admet-svc/src/admet_svc/main.py:14`、`services/dock-svc/src/dock_svc/main.py:16`、`services/boltz2-svc/src/boltz2_svc/main.py:13`、`services/fep-svc/src/fep_svc/main.py:17`
- HUMU index service 仍模拟 ANN 搜索：`services/humu-index-svc/src/humu_index_svc/main.py:52`
- Feature Store service 仍用 hash 生成 feature：`services/feature-store-svc/src/feature_store_svc/main.py:31`
- Patent indexing pipeline 返回 `molecules_indexed: 0`、`status: staged`：`pipelines/patent_indexing/src/patent_indexing/pipeline.py:25`
- Reaction indexing pipeline 返回 `total_templates: 0`、`status: staged`：`pipelines/reaction_indexing/src/reaction_indexing/pipeline.py:24`
- SRB compiler 按固定反应类型构造步骤，不消费真实 retrosyn route：`agents/srb_agent/src/srb_agent/compiler.py:59`
- KRAS Pilot E2E 全部 skip：`tests/e2e/test_kras_g12c_pilot.py:44`
- Audit completeness E2E 全部 skip：`tests/e2e/test_audit_completeness.py:16`

## 完整调用链路分析

目标架构调用链：

```text
用户自然语言意图
  -> API Gateway / Orchestrator
  -> CIC: LLM/tool-call semantic parse + UniProt/PDB/ChEMBL/SureChEMBL grounding
  -> CIG: typed objective graph
  -> HCIV: learned intent encoder
  -> HUMU: molecule/pocket/route/intent joint manifold
  -> TAR: task-aware generator routing
  -> AMGE: HFM-3D / FragFM / CReM / selected generators
  -> Oracle Cascade: L0 ADMET/RDKit -> L1 Dock -> L2 Boltz -> L3 OpenFE -> L4 quantum
  -> FTO / supply / retrosyn
  -> SRB: SSP/XDL
  -> CRG + provenance + OpenTelemetry/Sigstore
  -> DKI: Postgres/Neo4j/Milvus/MinIO/Feature Store
  -> E2E benchmark and audit evidence
```

现有实现调用链：

```text
用户自然语言意图
  -> API Gateway local reasoning demo
  -> heuristic NL parse
  -> RDKit-random candidate pool
  -> RDKit/启发式 scoring
  -> SQLite persistence
```

缺失的关键断点：

- CIC 没有真实 grounding 主链路。
- HCIV 默认 hash，不是 learned encoder 权重。
- HUMU 三塔/四塔没有真实可训练输入图与权重产物。
- 生成器没有真实 3D 分子生成、权重加载和并行生成闭环。
- Oracle cascade 没有真实外部引擎 job runner。
- DKI 服务 wrapper 存在，但 HTTP service 没接真实 Milvus/Feast/Neo4j/Postgres。
- MARB 没有真实状态机、共享 CRG 并发控制和审计链。
- E2E 测试通过 skip 保持绿色，不构成完成证据。

## KISS 四问

1. 这是现实问题还是想象问题？
   - 是现实问题。评估文档和源码行号均显示核心路径仍是启发式、随机、hash、固定池或 skip。

2. 有没有更简单的做法？
   - 有。先完成最小真实闭环，不同时实现所有 8 个生成器。优先落地 DKI、数据、CIC、HUMU、HFM/FragFM/CReM、L0-L2 Oracle，再扩展到全量架构。

3. 会破坏什么？
   - 最大风险是把原有本地 MVP demo 直接替换成依赖完整外部栈的路径，导致基础测试不可运行。实施时必须保留 demo 能力，但明确区分 `local_demo` 与 `production_real` 模式，生产模式不得回退假实现。

4. 当前项目真的需要这个功能吗？
   - 需要。用户目标是完成 CoreArchitecture v2 中已评估为未落地的核心架构，不是继续维护局部 demo。

## 执行原则

- 每个阶段必须先补测试或验收脚本，再改实现。
- 任何模型、数据、外部工具缺失时，必须抛出明确错误，不允许 fallback 到随机、固定池、hash 或 mock。
- 对已有接口保持向后兼容；旧 demo 可保留为显式 `local_demo`，不得作为生产路径默认值。
- 4 张 H200 优先用于 HUMU pretrain、HFM/FragFM 训练、Boltz/OpenFE 批处理，不用于单元测试。
- 实施前必须确认真实 git 仓库根目录并创建 `feature/corearchitecture-v2-completion` 分支。

## 工作包 0：仓库与质量门准备

### 涉及文件

- 只检查：项目真实 git 根目录、`pyproject.toml`、`.venv`、`uv.lock`、`Makefile`
- 可能修改：无，除非用户确认仓库根目录与分支策略

### 任务

1. 确认真实 git 仓库位置。
2. 创建 `feature/corearchitecture-v2-completion` 分支。
3. 固化基础验证命令：
   - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/unit tests/test_mvp_pipeline.py -q -p no:cacheprovider`
   - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/e2e -q -p no:cacheprovider`
4. 先处理 SQLite 权限问题：当前 `data/moleculeforge.db` 属主为 `FL`，当前用户 `FWY` 无写权限。优先方案是测试环境使用 `MF_DB_PATH` 指向用户可写临时数据库，不修改用户持有的主库文件。

### 验收

- 能在 feature 分支上执行。
- 单元与 MVP 测试仍通过。
- E2E 中 `sqlite3.OperationalError: attempt to write a readonly database` 不再出现。

## 工作包 1：真实 DKI 与 integration test

### 涉及文件

- `infra/docker/docker-compose.test.yml`
- `infra/docker/docker-compose.dki.yaml`
- `configs/services/*.yaml`
- `libs/mf-core/src/mf_core/db/*`
- `services/humu-index-svc/src/humu_index_svc/main.py`
- `services/feature-store-svc/src/feature_store_svc/main.py`
- `tests/integration/test_dki_postgres.py`
- `tests/integration/test_dki_neo4j.py`
- `tests/integration/test_dki_milvus.py`

### 任务

1. 启动 test stack 前检查端口 `5433`、`7475`、`7688`、`19531`、`9002`、`4223`。
2. 启动 `docker compose -f infra/docker/docker-compose.test.yml up -d`。
3. 将 Postgres/Neo4j/Milvus tests 从无条件 skip 改为读取 `TEST_DATABASE_URL`、`NEO4J_URI`、`MILVUS_HOST`、`MILVUS_PORT`。
4. `humu-index-svc` 接入 `MilvusCollectionClient`，删除模拟 ANN 结果。
5. `feature-store-svc` 接入真实离线/在线 store；如果 Feast 配置缺失，直接返回配置错误。

### 验收

- DKI integration tests 真实读写 Postgres、Neo4j、Milvus。
- `humu-index-svc` 搜索结果来自 Milvus collection。
- Feature Store 不再使用 hash 生成特征。

## 工作包 2：真实数据 ingestion 与 manifest

### 涉及文件

- `data/ingestion/chembl_ingestion.py`
- `data/ingestion/pdbbind_ingestion.py`
- `data/ingestion/crossdocked_ingestion.py`
- `data/ingestion/reaction_ingestion.py`
- `data/ingestion/surechembl/daily_sync.py`
- `data/ingestion/enamine_real/faiss_indexer.py`
- `pipelines/patent_indexing/src/patent_indexing/pipeline.py`
- `pipelines/reaction_indexing/src/reaction_indexing/pipeline.py`
- `configs/models/humu_pretrain.yaml`
- `data/dvc/pipelines/*.yaml`

### 任务

1. 确认真实数据集路径。当前配置指向 `zzzzz/Chembl/chembl_36/`、`zzzzz/CrossDocked2020/`、`zzzzz/USPTO-MIT/`，本工作区未验证存在。
2. 为每类数据生成 manifest：source、version、record_count、schema_hash、created_at、shard paths。
3. Patent indexing 必须真实读取 SureChEMBL/USPTO 数据，输出非零 `molecules_indexed` 或明确报错。
4. Reaction indexing 必须真实提取 reaction SMARTS，输出非零 `total_templates` 或明确报错。
5. Enamine REAL 建立真实 building block index，不允许返回固定供应链数据。

### 验收

- HUMU pretrain 三类输入均有非零样本。
- Patent/reaction pipeline 不再返回 `status: staged` 冒充完成。
- 所有训练和索引 pipeline 都能根据 manifest 复现输入。

## 工作包 3：CIC 与 learned HCIV

### 涉及文件

- `services/cig-compiler-svc/src/cig_compiler_svc/domain/compiler.py`
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/stages/stage1_semantic.py`
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/stages/stage1b_grounding.py`
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/stages/stage2_cig_build.py`
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_encoder.py`
- `models/mf-encoders/humu_intent_encoder/src/mf_encoders/humu_intent/encoder.py`
- `tests/unit/test_cic_compiler.py`
- `tests/integration/cic/test_cic_end_to_end.py`

### 任务

1. 将 `ground_knowledge` 接入 `CIGCompiler.compile`。
2. 新增明确的 semantic parser adapter 边界：LLM/tool-call adapter 与 heuristic demo 分离。
3. learned mode 缺少 encoder 或 checkpoint 时直接错误退出，不再 fallback hash。
4. `HCIVEncoder.encode` 返回契约与 `CIGCompiler._encode_hciv` 对齐。
5. CIG 中保留 grounding evidence：UniProt ID、PDB ID、ChEMBL/SureChEMBL evidence、confidence、source timestamp。

### 验收

- KRAS G12C 输入能生成含 target、activity、ADMET、FTO、synthetic constraints 的 CIG。
- learned HCIV 来自真实 encoder checkpoint。
- hash/random 只能作为显式 demo/test mode。

## 工作包 4：HUMU 联合预训练

### 涉及文件

- `models/mf-encoders/humu_mol_encoder/src/mf_encoders/humu_mol/encoder.py`
- `models/mf-encoders/humu_pocket_encoder/src/mf_encoders/humu_pocket/encoder.py`
- `models/mf-encoders/humu_route_encoder/src/mf_encoders/humu_route/encoder.py`
- `models/mf-encoders/humu_intent_encoder/src/mf_encoders/humu_intent/encoder.py`
- `libs/mf-humu/src/mf_humu/*`
- `pipelines/humu_pretrain/src/humu_pretrain/data_loader.py`
- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
- `configs/models/humu_pretrain.yaml`
- `tests/unit/test_humu_training.py`

### 任务

1. molecule encoder 改为真实 molecular graph 输入，不再由 SMILES hash 产生 embedding。
2. pocket encoder 改为真实 pocket residue/atom coordinate 输入，缺坐标直接报错。
3. route encoder 改为真实 route tree / reaction graph 输入，不再 hash seed。
4. intent encoder 接入 CIG graph features。
5. pretrain pipeline 改为 DDP 训练入口，支持 `CUDA_VISIBLE_DEVICES=0,1,2,3`。
6. 删除或改造 `pretrain_*_encoder` 中直接返回 `"status": "trained"` 的 stub wrapper。
7. 加入训练进度显示：单行 `\r` 进度条、loss、LR、batch、epoch time。

### H200 使用

- GPU 0-3：HUMU pretrain DDP bf16 主任务。
- 单卡 smoke：GPU 0，1 个 epoch，小样本。
- 四卡正式：GPU 0-3，完整 manifest 数据。

### 验收

- 产出 `best_model.pt`、`final_model.pt`、训练日志、manifest hash。
- 指标包含 HUMU distortion、activity cliff AUROC、mol-pocket retrieval、route consistency。
- 训练失败时保留原始 stderr 和 checkpoint 状态，不伪造完成。

## 工作包 5：AMGE 最小真实生成闭环

### 涉及文件

- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/*`
- `models/mf-generators/fragfm/src/mf_generators/fragfm/*`
- `models/mf-generators/crem_3d/src/mf_generators/crem_3d/*`
- `libs/mf-core/src/mf_core/routing/task_router.py`
- `libs/mf-core/src/mf_core/routing/cross_paradigm_kd.py`
- `services/hfm-generator-svc/src/hfm_generator_svc/main.py`
- `services/fragfm-generator-svc/src/fragfm_generator_svc/main.py`
- `services/crem-generator-svc/src/crem_generator_svc/main.py`
- `services/generator-router-svc/src/generator_router_svc/main.py`
- `tests/unit/test_generators.py`
- `tests/unit/test_task_router.py`
- `tests/unit/test_cross_paradigm_kd.py`

### 任务

1. 第一批只落地 HFM-3D、FragFM、CReM-3D 和 TAR/KD。
2. HFM-3D 删除固定 SMILES pool 解码，改为真实 decoder 或直接要求 checkpoint/decoder artifact。
3. FragFM 使用真实 fragment vocabulary 与 assembly validity check。
4. CReM-3D 接真实 matched molecular pair / pharmacophore mutation。
5. TAR 训练或加载真实 routing weights；随机初始化只能用于测试。
6. KD teacher 改为 Oracle cascade feedback，不再用 `WeakTeacher` 的字符串启发式作为生产 teacher。

### 验收

- 生成候选包含合法 SMILES、3D conformer、HUMU latent、generator provenance。
- 不存在固定池、随机 score、hash decoding 作为生产路径。
- route allocation 可被 Oracle feedback 更新。

## 工作包 6：真实 Oracle Cascade

### 涉及文件

- `services/admet-svc/src/admet_svc/main.py`
- `services/dock-svc/src/dock_svc/main.py`
- `services/boltz2-svc/src/boltz2_svc/main.py`
- `services/fep-svc/src/fep_svc/main.py`
- `models/mf-oracles/admet_ai/src/mf_oracles/admet_ai/oracle.py`
- `models/mf-oracles/gnina/src`
- `models/mf-oracles/diffdock_l/src/mf_oracles/diffdock_l/oracle.py`
- `models/mf-oracles/boltz2/src/mf_oracles/boltz2/oracle.py`
- `models/mf-oracles/openfe/src/mf_oracles/openfe/oracle.py`
- `pipelines/boltz2_eval`
- `tests/unit/services`
- `tests/e2e/test_kras_g12c_pilot.py`

### 任务

1. L0：ADMET service 接入真实 RDKit/ADMET 模型，删除随机输出。
2. L1：Dock service 接 GNINA/DiffDock job runner。
3. L2：Boltz2 service 接真实 Boltz model runner。
4. L3：FEP service 接 OpenFE RBFE job runner。
5. L4：GPU4PySCF/ORCA 作为高成本精修路径，只有当配置和许可可验证时启用。
6. 每个 Oracle 输出：input artifact hash、model version、runtime、uncertainty、stderr path、provenance event。

### 验收

- 随机模拟 score 全部从生产路径删除。
- Oracle cascade 支持自适应升级：低成本筛选到高成本验证。
- KRAS Pilot 至少能跑通 L0-L2；L3/L4 可作为显式 slow/gpu 测试。

## 工作包 7：MARB、CRG 与审计链

### 涉及文件

- `agents/orchestrator/src/orchestrator/workflow/graph_builder.py`
- `agents/orchestrator/src/orchestrator/agent.py`
- `agents/*/src/*`
- `libs/mf-agents/src/mf_agents/*`
- `services/orchestrator-svc/src/orchestrator_svc/main.py`
- `services/provenance-svc/src/provenance_svc/*`
- `libs/mf-telemetry/src/mf_telemetry/tracing/opentelemetry.py`
- `tests/e2e/test_audit_completeness.py`

### 任务

1. `graph_builder` 改为真实 LangGraph 状态机；实施前按当前锁定版本查官方 API。
2. 实现 PLANNING、GENERATING、VALIDATING、REFINING、ESCALATING 状态迁移。
3. CRG 写入 Neo4j，belief/event 写入 Postgres，artifact 写入 MinIO。
4. provenance-svc 使用真实签名/验证路径；无法使用 Sigstore 时必须明确配置为本地开发签名，不能伪称 Sigstore。
5. trace_id 从 API Gateway 贯穿到 generator、Oracle、FTO、SRB。

### 验收

- Audit completeness E2E 不再 skip。
- 每个 pipeline step 至少有一个可验证 AuditEvent。
- OpenTelemetry trace 可关联 run_id、candidate_id、oracle_call_id。

## 工作包 8：FTO、供应链、Retrosyn、SRB

### 涉及文件

- `services/fto-patent-svc/src/fto_patent_svc/*`
- `pipelines/patent_indexing/src/patent_indexing/pipeline.py`
- `services/supply-oracle-svc/src/supply_oracle_svc/main.py`
- `services/retrosyn-svc/src/retrosyn_svc`
- `models/mf-retrosyn/aizynth_wrapper/src/mf_retrosyn/aizynth/retrosyn.py`
- `models/mf-retrosyn/rsgpt/src/mf_retrosyn/rsgpt/retrosyn.py`
- `models/mf-retrosyn/ualign/src/mf_retrosyn/ualign/retrosyn.py`
- `agents/srb_agent/src/srb_agent/compiler.py`
- `wetlab/xdl-compiler/src/xdl_compiler/*`

### 任务

1. FTO service 查询真实 patent index，不再返回空 hits 和随机 dead zone。
2. Supply oracle 查询真实 catalog/index，不再用 hash 生成价格、库存、交期。
3. Retrosyn 接真实 AiZynthFinder/RSGPT/UAlign 输出；缺模型或配置直接错误。
4. SRB 从 retrosyn route 的 reaction、reactants、conditions、building blocks 编译 SSP，不再轮询固定 reaction types。
5. XDL 输出必须可追溯到 SSP step id。

### 验收

- FTO verdict 有 patent evidence。
- Supply result 有 catalog source 和 timestamp。
- SSP/XDL 每一步都能追溯到 retrosyn route。

## 工作包 9：E2E 与 Benchmark 解锁

### 涉及文件

- `tests/e2e/test_kras_g12c_pilot.py`
- `tests/e2e/test_audit_completeness.py`
- `tests/benchmark/*`
- `libs/mf-eval/src/mf_eval/*`
- `tools/benchmarks/*`

### 任务

1. 解开 KRAS G12C Pilot skip，改为真实服务栈测试。
2. 解开 Audit completeness skip，验证 provenance/trace/signature。
3. 跑 MOSES/GuacaMol/PMO/CrossDocked benchmark。
4. 记录 HUMU distortion、activity cliff、EF1%、合成-分子嵌入一致性。

### 验收

- E2E 不再依赖 skip 表示通过。
- benchmark 结果有命令、数据 manifest、模型 checkpoint、硬件信息、日志路径。

## 实施顺序

```text
0 仓库与质量门
  -> 1 DKI
  -> 2 数据 ingestion
  -> 3 CIC + learned HCIV
  -> 4 HUMU pretrain
  -> 5 AMGE 最小真实闭环
  -> 6 Oracle L0-L3
  -> 7 MARB/CRG/Audit
  -> 8 FTO/Supply/Retrosyn/SRB
  -> 9 E2E/Benchmark
```

不建议先同时实现所有生成器或所有 Oracle。当前最短真实闭环是：

```text
CIC -> learned HCIV -> HUMU -> HFM/FragFM/CReM -> L0/L1/L2 Oracle -> FTO -> Retrosyn -> SRB -> Audit
```

## 风险与缓解

- 风险：真实数据未在工作区存在。
  - 缓解：先做 manifest 校验；数据缺失时停止实施并列出缺失路径。

- 风险：外部工具版本/API 不确定。
  - 缓解：涉及 LangGraph、Feast、Boltz、DiffDock、GNINA、OpenFE、Sigstore、OpenTelemetry 的具体配置前，必须按当前锁文件和官方文档核验，不凭自然语言猜配置。

- 风险：完整真实 Oracle 成本高、耗时长。
  - 缓解：分层 L0-L3，先确保 L0-L2 可跑通，L3/L4 只对少量 Pareto 候选运行。

- 风险：替换 demo 路径破坏现有基础测试。
  - 缓解：保留显式 `local_demo`，生产路径默认 `production_real`，测试分别覆盖。

- 风险：4 卡训练产生长任务。
  - 缓解：先单卡 smoke，再四卡 DDP；训练日志必须实时写入，异常保留 stderr。


## 工作包 0 前置验证记录

验证时间：2026-05-13。

### Git

- 用户已允许初始化 git 仓库。
- 已在 `/workspace/MForge/moleculeforge` 执行 `git init -b feature/corearchitecture-v2-completion`。
- 当前分支：`feature/corearchitecture-v2-completion`
- 因工作区目录属主与当前用户不一致，已执行 `git config --global --add safe.directory /workspace/MForge/moleculeforge` 解除 Git safe directory 阻塞。
- 当前仓库尚无初始提交，所有项目文件处于 untracked 状态。

### 数据集

- `zzzzz` 实际路径：`/workspace/MForge/zzzzz`
- 已确认存在：
  - `/workspace/MForge/zzzzz/Chembl/chembl_36/chembl_36_sqlite/chembl_36.db`
  - `/workspace/MForge/zzzzz/CrossDocked2020/CrossDocked2020_v1.3`
  - `/workspace/MForge/zzzzz/PDBBind/P-L.tar.gz`
  - `/workspace/MForge/zzzzz/USPTO-MIT/USPTO-MIT.zip`
  - `/workspace/MForge/zzzzz/USPTO-MIT/USPTO50K.zip`
  - `/workspace/MForge/zzzzz/RetroPath/*/templates.*.gz`
- 顶层未看到 SureChEMBL、Enamine REAL、Google Patents、Reaxys 离线数据目录。

### 模型权重

- 项目内只找到：
  - `/workspace/MForge/moleculeforge/models/esm2/esm2_t33_650M_UR50D.pt`
  - `/workspace/MForge/moleculeforge/models/esm2/esm2_alphabet.pkl`
- HUMU/HFM/FragFM/Boltz/OpenFE 等 checkpoint 目录只包含 `.gitkeep` 或 `.gitignore`，未找到可用权重。
- 结论：HUMU/HFM/FragFM/Boltz/OpenFE 权重需要补充或先训练产出。

### 外部工具与 Python 包

- 命令行工具未找到：`gnina`、`boltz`、`openfe`、`openfecli`、`pmx`、`gmx`、`obabel`、`rdkit`
- `.venv` Python import 检查：
  - 可用：`rdkit`、`pymilvus`、`neo4j`、`feast`、`opentelemetry`
  - 不可用：`openfe`、`boltz`、`sigstore`、`gnina`、`diffdock`
- 环境变量只检测到 Anthropic 相关变量，未检测到 ChEMBL/SureChEMBL/USPTO/Enamine/Boltz/OpenFE/GNINA/Feast/Milvus/Neo4j/Postgres/MinIO/NATS/Sigstore/OpenTelemetry 的连接配置变量。

### Docker 与端口

- Docker CLI 可用：`Docker version 29.1.3`
- Docker Compose 可用：`Docker Compose version 2.40.3`
- 已将 `FWY` 加入 `docker` 组；当前 Codex 会话未继承新组时，可用 `sg docker -c "..."` 访问 Docker daemon。
- Docker daemon 已配置代理：`HTTP_PROXY=http://127.0.0.1:7890`、`HTTPS_PROXY=http://127.0.0.1:7890`。
- 目标端口 `5433`、`7475`、`7688`、`19531`、`9002`、`4223` 均空闲。
- 当前环境缺少 `CAP_SYS_ADMIN`/namespace 权限；`sudo unshare -m /bin/true` 返回 `Operation not permitted`。
- 因上述权限限制，Docker 可以下载镜像 layer，但注册 layer 失败：`failed to register layer: unshare: operation not permitted`。
- 结论：Docker test stack 是 DKI 集成验收的环境阻塞项，不是当前非 Docker 工作包的硬阻塞。后续先推进数据、CIC、HUMU、生成器与本地测试；DKI 的 Postgres/Neo4j/Milvus/MinIO/NATS 真实验收需要换到具备 `CAP_SYS_ADMIN` 或 privileged Docker 的环境执行。

### GPU

- 4 张 H200 仍空闲，GPU 0-3 均为 `0 MiB` 显存使用、`0%` 利用率。

## 下一步

从不依赖 Docker test stack 的工作包继续执行；涉及 DKI 集成服务的验收项单独标记为环境阻塞，待具备 `CAP_SYS_ADMIN` 或 privileged Docker 的环境后再验证。

## 2026-05-13 非 Docker 工作包执行记录

### 工作包 3：CIC 与 learned HCIV 部分落地

已更新文件：

- `services/cig-compiler-svc/src/cig_compiler_svc/domain/compiler.py`
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/stages/stage1b_grounding.py`
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/stages/stage2_cig_build.py`
- `tests/unit/test_cic_compiler.py`
- `tests/integration/cic/test_cic_end_to_end.py`

当前行为：

- `CIGCompiler.compile` 主链路改为 `semantic_parser -> ground_knowledge -> build_cig -> HCIV encode`。
- `build_cig` 优先使用 grounding 后的 UniProt accession、PDB ID 和 grounding evidence；无 grounding 结果时保留原 target name 兼容旧输入。
- `encoding_mode="learned"` 缺少 `learned_encoder` 时直接抛出 `RuntimeError`，不再 fallback 到 hash HCIV。
- `HCIVEncoder.encode` 返回 `(HCIV, IntentCone)` 时，`CIGCompiler` 直接使用 encoder 输出的 cone。

未完成项：

- semantic parser adapter 目前只提供可注入边界，尚未接真实 LLM/tool-call adapter。
- learned HCIV 仍缺真实 checkpoint；当前只阻断无 encoder fallback，不构成真实权重验收。
- KRAS Pilot E2E 仍因服务栈、权重和 oracle 缺失保持 skip。

验证记录：

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/unit/test_cic_compiler.py tests/integration/cic/test_cic_end_to_end.py -q -p no:cacheprovider`
  - 结果：35 items，退出码 0。

### 工作包 5：HFM-3D 生产路径固定池隔离

已更新文件：

- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py`
- `tests/unit/test_humu_training.py`

当前行为：

- `HFM3DGenerator` 默认模式为 `production_real`。
- `production_real` 生成路径要求 checkpoint 已加载且注入 `smiles_decoder`，缺任一项直接抛出 `RuntimeError`，不再使用固定 SMILES pool。
- 固定 SMILES pool 仅保留在显式 `mode="local_demo"` 中，用于本地 smoke/demo 测试。
- `checkpoint_path` 被显式提供但文件不存在时直接抛出 `FileNotFoundError`。

未完成项：

- HFM-3D 尚无真实 decoder artifact/checkpoint。
- 该更新只阻断生产路径伪生成，不构成真实 HFM-3D 生成闭环验收。

验证记录：

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/unit/test_humu_training.py tests/unit/test_generators.py -q -p no:cacheprovider`
  - 结果：17 items，退出码 0。

### 工作包 2 / 工作包 8：Patent 与 Reaction indexing fail-fast

已更新文件：

- `pipelines/patent_indexing/src/patent_indexing/pipeline.py`
- `pipelines/reaction_indexing/src/reaction_indexing/pipeline.py`
- `tests/unit/test_indexing_pipelines.py`

当前行为：

- `patent_indexing.index_surechembl_to_milvus` 要求 `surechembl_path` 指向真实存在的数据路径，缺失或空数据直接失败。
- `patent_indexing.index_uspto_patents` 要求 `uspto_path` 指向真实存在的数据路径，缺失或空数据直接失败。
- Patent indexing 不再返回 `molecules_indexed: 0` 和 `status: staged` 作为完成状态。
- Patent indexing 需要显式传入 `milvus_client.insert(collection, records)`；未传入 client 时直接失败，不伪造 Milvus 写入。
- Dead zone 更新需要显式传入 `dead_zone_updater.refresh(config)`；未传入 updater 时直接失败，不再导入不存在的 `DeadZoneUpdater`。
- `reaction_indexing.extract_reaction_templates` 要求每个 source 配置 `source_paths[source]`，并从真实文件中提取含 `>>` 的 reaction SMARTS。
- Reaction indexing 不再返回 `total_templates: 0` 和 `status: staged` 作为完成状态。
- `reaction_indexing.index_reactions` 对零模板直接失败；非零模板返回 `status: completed` 和实际模板数量。

未完成项：

- SureChEMBL、Google Patents、Enamine REAL、Reaxys 离线数据目录仍未在 `/workspace/MForge/zzzzz` 顶层发现。
- Patent indexing 当前支持本地 `.smi`、`.smiles`、`.txt`、`.csv`、`.tsv`、`.gz` 文本记录解析；尚未实现 SureChEMBL/USPTO XML、Markush 结构解析。
- Reaction indexing 当前支持本地文本模板文件解析；尚未从 USPTO-MIT zip 或 RetroPath 全量压缩包构建完整 manifest。
- Milvus 真实写入仍受 Docker test stack 环境阻塞，当前单元测试只验证显式 client 合约。

验证记录：

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/unit/test_indexing_pipelines.py -q -p no:cacheprovider`
  - 结果：4 items，退出码 0。
- `.venv/bin/ruff check pipelines/patent_indexing/src/patent_indexing/pipeline.py pipelines/reaction_indexing/src/reaction_indexing/pipeline.py`
  - 结果：退出码 0。
- `.venv/bin/ruff check --select I,F tests/unit/test_indexing_pipelines.py`
  - 结果：退出码 0。

### 基线验证记录

- `PYTHONDONTWRITEBYTECODE=1 MF_DB_PATH=/tmp/moleculeforge-codex-baseline.db .venv/bin/python -m pytest tests/unit tests/test_mvp_pipeline.py -q -p no:cacheprovider`
  - 结果：149 items，退出码 0。
- `PYTHONDONTWRITEBYTECODE=1 MF_DB_PATH=/tmp/moleculeforge-codex-e2e.db .venv/bin/python -m pytest tests/e2e -q -p no:cacheprovider`
  - 结果：25 items，退出码 0；其中 11 个既有 skip 仍存在。

### 剩余阻塞

- DKI integration test 仍需要具备 `CAP_SYS_ADMIN` 或 privileged Docker 的环境。
- HUMU/HFM/FragFM/Boltz/OpenFE 真实 checkpoint 仍未找到。
- `tests/e2e/test_kras_g12c_pilot.py` 和 `tests/e2e/test_audit_completeness.py` 仍未解锁，原因是完整服务栈、真实模型权重、oracle、FTO、retrosyn、provenance/trace/signature 依赖未满足。
