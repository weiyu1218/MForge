# MoleculeForge CoreArchitecture v2 完成度评估

## 范围

- 核心代码目录：`MForge/moleculeforge`
- 对照文档：`MForge/MoleculeForge_CoreArchitecture_v2.md`
- 本次操作：只读分析与测试验证，未修改核心代码

## 验证记录

### 单元与 MVP 测试

命令：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/unit tests/test_mvp_pipeline.py -q -p no:cacheprovider
```

结果：

- 142 项通过
- 警告包括 Pydantic V2 `Config` 弃用提示，以及 `torch.load(weights_only=False)` 的 FutureWarning

### E2E 测试

命令：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/e2e -q -p no:cacheprovider
```

结果：

- 25 项中 12 项通过，11 项跳过，2 项失败
- 失败项：
  - `tests/e2e/test_reason_workbench.py::test_reason_run_completes`
  - `tests/e2e/test_reason_workbench.py::test_reason_run_recognises_known_molecule`
- 失败原因：

```text
sqlite3.OperationalError: attempt to write a readonly database
```

写入点：

- `MForge/moleculeforge/libs/mf-core/src/mf_core/db/store.py:143`

数据库状态：

- `MForge/moleculeforge/data/moleculeforge.db` 权限为 `rw-r--r--`
- 主库文件属主为 `FL`
- 测试运行时生成了 `moleculeforge.db-shm` 与 `moleculeforge.db-wal`
- 当前用户无法写入主库，导致 `/v1/reason/runs` 写 `runs` 表失败

## 总体结论

当前 `MForge/moleculeforge` 不是完整落地的 `MoleculeForge_CoreArchitecture_v2.md`。更准确的状态是：

```text
架构骨架 + 局部可运行原型 + 大量模拟服务
```

现阶段已经完成了工程目录、workspace 配置、基础类型、局部 HUMU 数学算子、API demo、本地 RDKit 流程、部分生成器和 Agent 的接口骨架，并且基础单元测试能够通过。

尚未完成的是架构文档中的核心创新主体：联合 HUMU 学习、真实多范式生成、真实 Agent 协同、真实 Oracle 级联、真实 FTO/逆合成/供应链闭环、生产级 DKI 与审计链。

## 已完成或基本可用

| 模块 | 状态 | 证据 |
|---|---|---|
| Monorepo 工作区 | 已搭建 | 根 `pyproject.toml` 已登记 libs、models、services、agents、pipelines |
| 协议目录 | 已搭建 | `protos/moleculeforge/v1/...` 下存在 18 个 `.proto` 文件 |
| 基础类型 | 已实现基础模型 | `mf_core.types.cig`、`mf_core.types.humu`、`mf_core.types.crg`、`mf_core.types.ssp` |
| HUMU 基础数学 | 局部可用 | `mf_humu.manifold.lorentz` 实现 `distance`、`expmap`、`logmap` |
| API Gateway | 可运行原型 | `services/api-gateway/src/api_gateway/main.py` |
| 本地属性预测 | 局部可用 | RDKit 描述符、QED、SA、ADMET 启发式结果可通过 API 返回 |
| 本地 reasoning demo | 局部可用 | NL parse、候选生成、scoring、过滤、novelty、ranking 单进程流程存在 |
| 单元测试 | 当前通过 | 142 项通过 |

## 部分完成但未达到架构要求

| 模块 | 已有内容 | 缺口 |
|---|---|---|
| CIC | 有启发式 NL 抽取和 CIG 构建 | 不是 LLM + 工具调用编译；`ground_knowledge` 存在但主编译链未调用；HCIV 默认 hash 编码 |
| HUMU 编码器 | 有 molecule、pocket、route、intent encoder 类 | 主要是 hash、随机或简单几何特征，不是 SE(3)-GNN、E(3)-GNN、TreeLSTM 三塔联合训练 |
| AMGE 生成器 | HFM、FragFM、LaMGen、MMPT、EvoMol、ICLM、CReM、UAS 目录和类基本齐 | 多数返回固定 SMILES 池或随机/hash 结果，没有真实 3D 分子生成和训练权重 |
| TAR + KD | 有 TaskAwareRouter 和 CrossParadigmKDLayer | 路由器是小型随机初始化网络 + hard rules；KD teacher 是弱启发式，不是 Boltz/HypSeek 教师蒸馏 |
| MARB Agent | Agent 类、CRG 容器、消息接口存在 | 没有真实 LangGraph 状态机；多个 Agent 直接返回固定结果 |
| Oracle Cascade | 有 ADMET/Dock/Boltz/FEP 服务文件 | 多数服务使用随机数或模拟返回，不是真实 Boltz-2、GNINA/DiffDock、OpenFE、GPU4PySCF |
| DKI | SQLite store、Milvus/Neo4j/MinIO wrapper、Docker/K8s 配置骨架存在 | Milvus、Feature Store、HUMU index 多为模拟结果；未验证真实数据服务部署链 |
| SRB | SSP 编译器和 XDL compiler 存在 | SSP 步骤按固定反应类型生成，不是从真实 retrosyn route 编译 |

## 未完成或证据显示尚未落地

- JMCG：没有真实 `(m, r, p)` 联合分布训练闭环。
- HUMU 联合预训练：有 pipeline，但未看到真实 ChEMBL/PDB/PaRoutes 大规模训练产物或权重闭环。
- 真实 CIG 编译：缺少 LLM 工具调用、UniProt/PDB/SureChEMBL/ChEMBL grounding 的主链路。
- 真实 HFM-3D、FragFM-HUMU、LaMGen-3D-Pro、MMPT-RAG、EvoMol-RL、ICLM、UAS：只有原型或固定池生成，不是架构中的算法级实现。
- L1-L4 Oracle：Boltz-2、DiffDock/GNINA、OpenFE、GPU4PySCF 都未真实接入。
- FTO/Patent Dead Zone：有接口和 updater，但没有真实 SureChEMBL/USPTO/Reaxys/Google Patents 数据链路。
- 端到端 KRAS G12C Pilot：测试文件中明确 `skip`，原因是需要服务、模型权重或完整栈。
- 审计完整性：测试全部 `skip`，需要完整 pipeline、provenance-svc、CRG + Sigstore、OpenTelemetry。

## 分层对照

### 第一层：CIC

状态：部分完成。

已完成：

- 有 `CIGCompiler`。
- 有 `_heuristic_extract`。
- 有 `build_cig`。
- 有 hash/learned/random 三种 HCIV encoding mode。

未完成：

- 默认不是 learned encoder。
- 自然语言解析不是 SRM/LLM 工具调用链。
- UniProt/PDB grounding 没有进入主 `compile` 链路。
- CIG 结构比架构文档中的 JSON-LD 目标图简化很多。

### 第二层：HUMU

状态：基础数学完成，核心学习未完成。

已完成：

- Lorentz manifold 基础运算。
- IntentCone 类型和采样函数。
- Patent dead zone potential 函数。
- Unfamiliarity 函数。
- 分子、口袋、路径、意图 encoder 类。

未完成：

- 没有真实 SE(3)-Equivariant Message Passing。
- 没有 EquiBind-style E(3)-GNN。
- 没有双向图 Transformer + 双曲 TreeLSTM。
- 没有真实联合对比训练产物。
- Patent dead zone 使用的是欧氏差值势能，不是完整 HUMU 障碍势。

### 第三层：AMGE

状态：目录与接口基本齐，真实生成能力未完成。

已完成：

- 8 类生成器名称基本齐全。
- HFM 有 LorentzFlowMatching 类和 checkpoint save/load。
- FragFM 有 fragment vocabulary 与 SA-aware rate matrix 相关文件。
- EvoMol 有 Pareto archive、sleeping bandit、hypervolume reward。
- ICLM 有 EWC、PackNet、online learner 文件。
- UAS 有 autoencoder 原型。

未完成：

- 多数生成器输出固定小 SMILES 池。
- 没有真实 3D 坐标生成。
- 没有真实模型权重。
- 没有真实跨范式并行生成和蒸馏闭环。
- 没有真实 FTO-aware RAG 负样本对比解码。

### 第四层：MARB

状态：Agent 骨架存在，真实多 Agent 推理未完成。

已完成：

- Orchestrator、NL2Obj、GeneratorCoord、RetroSyn、Validation、FTO、Supply、Critic、SRB agent 目录存在。
- `ChemicalReasoningGraph` 容器存在。
- Critic 有 100 条规则文件。

未完成：

- `graph_builder` 不是 LangGraph `StateGraph` 实现，只是节点/边描述。
- Agent 返回结果多为固定结构。
- 没有真实共享 CRG 的并发控制、签名、冲突处理。
- Orchestrator 没有执行文档中的 PLANNING/GANERATING/VALIDATING/REFINING/ESCALATING 状态机。

### 第五层：Oracle Cascade

状态：服务文件存在，真实 Oracle 未接入。

已完成：

- 有 ADMET、Dock、Boltz2、FEP 服务骨架。
- 本地 API demo 能用 RDKit 启发式做 L0 类筛选。

未完成：

- Boltz-2 返回随机/模拟 score。
- Dock 返回随机/模拟 docking score。
- FEP 服务未真实 OpenFE。
- L4 GPU4PySCF/ORCA DFTB3 未接入。
- 不确定度传播和自适应升级逻辑未完成。

### 第六层：SRB

状态：SSP/XDL 原型存在，真实路径编译未完成。

已完成：

- `SSP` 类型存在。
- `compile_ssp` 能生成结构化 SSP。
- `xdl-compiler` 能输出 XDL procedure/XML。

未完成：

- SSP 步骤不是由真实逆合成路线推导。
- yield/cost 是规则估计。
- 没有真实 wet-lab bridge 或 SiLA2 执行。

### 第七层：DKI

状态：配置和 wrapper 存在，真实数据基础设施未完成。

已完成：

- SQLite 本地 store。
- Alembic migration。
- Milvus/Neo4j/MinIO client wrapper。
- docker-compose 和 K8s namespace/Helm 骨架。

未完成：

- HUMU index 服务返回模拟 ANN 结果。
- Feature Store 返回 hash 生成特征，不是真实 Feast。
- SureChEMBL、Enamine REAL ingestion 标明 placeholder。
- Integration tests 需要外部栈，未在本次通过完整验证。

### 第八层：工程实施蓝图

状态：目录和服务数量基本覆盖，生产实现未完成。

已完成：

- 文档中的核心服务大多有对应目录。
- FastAPI/gRPC server 文件存在。
- Docker/K8s/Helm 配置骨架存在。

未完成：

- gRPC 服务没有注册真实 protobuf servicer。
- 多数服务使用 `return type(...)()` 构造响应。
- 端口和服务配置没有经真实集群启动验证。
- GPU/H100/A100/H200 资源规划未实际落地。

### 第九层：评估体系

状态：测试目录存在，核心 benchmark 未完成。

已完成：

- 有 `tests/benchmark` 目录。
- 有 MOSES/GuacaMol/PMO benchmark 文件。
- 有 e2e smoke tests。

未完成：

- KRAS G12C Pilot 全部 skip。
- Audit completeness 全部 skip。
- 未看到完整 MOSES/GuacaMol/PMO/CrossDocked benchmark 实测结果。
- 未看到 HUMU distortion、activity cliff、EF1%、合成-分子嵌入一致性实测结果。

## 最终判断

现阶段可以认定为：

```text
基础工程骨架：完成度较高
局部可运行 demo：已完成
架构文档核心创新：大部分未完成
真实端到端药物设计平台：未完成
```

如果按 `MoleculeForge_CoreArchitecture_v2.md` 的 Phase 0/1/2 路线图衡量：

- Phase 0：目录、配置、基础数据处理、基础 HUMU 算子有一部分完成；真实 K8s/DKI/数据集/HUMU 预训练未完成。
- Phase 1：生成器与 Agent 的代码骨架存在；真实 HFM/FragFM/CReM/TAR、真实 LangGraph Agent、端到端集成未完成。
- Phase 2：高级生成器、供应链 Oracle、高精度 Oracle、完整基准评估基本未完成。

