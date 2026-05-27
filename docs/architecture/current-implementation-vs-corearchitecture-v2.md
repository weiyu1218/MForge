# 当前实现与 CoreArchitecture v2 设想对比

## 0. 元信息

- 对照架构文档：`/workspace/MForge/MoleculeForge_CoreArchitecture_v2.md`
- 代码审阅范围：`/workspace/MForge/moleculeforge`
- DKI 运行环境：`/workspace/mf-dki-bare`
- 更新时间：2026-05-19（替换 05:40 旧版，本次更新补充逐文件 file:line 级证据 + Claude 能力边界声明）
- 方法：六路只读 subagent 并行勘察 + pyproject/uv.lock 依赖审查 + pytest 实跑统计；本文只记录有代码、配置、命令输出或测试结果支撑的事实。

---

## 1. 总体结论

当前项目已经不是「只有骨架」的状态，但仍不是完整落地的 CoreArchitecture v2。准确状态：

```text
工程骨架       :  非常完整（22+ 服务、9 个 agent、8 类生成器目录、协议/schema 齐备）
DKI 基础设施   :  真实可用（Postgres / Neo4j / Qdrant / MinIO / Redis 集成测试 12 项通过）
真实模型权重    :  缺失（除 HUMU 三塔 ~20.6 万参数 baseline ckpt 外，8 类生成器 checkpoint 目录均为空）
真实推理路径    :  绝大部分为「诚实占位」——缺资源时 raise RuntimeError 而非伪造结果
端到端业务流程  :  KRAS / Audit pilot 均为占位测试，未真实跑通
评估体系       :  benchmark 文件因命名错误未被 pytest 收集；MOSES 4 项指标本地复刻，其它（GuacaMol/PMO/DUD-E）0 实现
```

与设想的差距集中在三个层面：

1. **算法层**：设想里的 SE(3)/E(3)-GNN、双向图 Transformer + 双曲 TreeLSTM、ProxylessNAS Router、REINFORCE 在线学习、Lorentz 等变注意力均未实现，相关模块要么是 Linear+MLP 简化版，要么是孤立未被调用的占位类。
2. **资源层**：8 类生成器 checkpoint 目录全空；ADMET-AI / Boltz-2 / GNINA / DiffDock-L / OpenFE / GPU4PySCF / AiZynthFinder / RSGPT 等 11 个外部模型/工具均未在 `uv.lock` 中声明；SureChEMBL / USPTO / Reaxys / Google Patents 四个客户端均硬编码假数据。
3. **集成层**：22 个 gRPC 服务里**没有一个**完成 `add_*Servicer_to_server` 注册；buf 未生成 Python pb2；Orchestrator 存在 3 条互不调用的执行路径；critic-svc 与 critic_agent 完全无关联；CIC 三套 NL 解析互不复用。

---

## 2. 已验证的基础事实

### 2.1 DKI 服务状态（mf-dki-bare）

```text
postgres         :15432
neo4j-http       :17474
neo4j-bolt       :17687
qdrant-http      :16333
qdrant-grpc      :16334
minio-api        :19000
minio-console    :19001
redis            :16379
```

### 2.2 测试通过率（本次实跑）

| Suite | Collected | Pass | Fail | Skip | Errors |
|---|---|---|---|---|---|
| `tests/unit` | 231 + 1 collection error | 231 | 0 | 0 | 1 |
| `tests/integration` | 36 | 24 (cic) | 0 | 12 (DKI 集群依赖) | 0 |
| `tests/e2e` | 26 | 10 (predict_api) | 2 (reason_workbench DB readonly) | 11 (KRAS+audit) | 0 |
| `tests/benchmark` | **0**（文件命名 `*_benchmark.py` 不匹配 pytest 默认 `test_*.py`） | 0 | 0 | 0 | 0 |
| `tests/anti_degradation` | 8 | 5 | 3 | 0 | 0 |
| `tests/test_mvp_pipeline` | 9 | 9 | 0 | 0 | 0 |

#### 关键事故性发现

- **unit collection error**：`tests/unit/test_ssp_compiler.py:16` 导入不存在的 `_build_steps`（真实签名 `_build_steps_from_route`，见 `agents/srb_agent/src/srb_agent/compiler.py:61`），11 个测试因此从 collect 阶段就丢失。
- **benchmark 整体 collect=0**：`tests/benchmark/` 下三个文件命名为 `moses_benchmark.py / guacamol_benchmark.py / pmo_benchmark.py`，违反 pytest 默认 `python_files = test_*.py`，**未被任何 CI 执行**。
- **e2e reason_workbench 2 项失败**：`sqlite3.OperationalError: attempt to write a readonly database`（`libs/mf-core/src/mf_core/db/store.py:145`）。这是工作目录权限问题，不是代码缺陷。
- **anti_degradation 3 项失败**：生产代码污染——`agents/orchestrator/.../pipeline.py:208`、`agents/nl2obj/.../parser.py:314` 含 `except Exception: pass`；`pipelines/humu_pretrain/.../pipeline.py:562` 触发 mock/dummy 检测。

### 2.3 已通过的 DKI 集成测试

```text
tests/integration/test_dki_postgres.py / test_dki_neo4j.py / test_dki_qdrant.py /
test_dki_minio.py / test_dki_redis.py / test_dki_provenance.py /
test_dki_patent_indexing.py
→ 12 passed
```

### 2.4 仓库内唯一真实的模型权重产物

```text
checkpoints/humu_4h200/best_model.pt      2,513,014 B  epoch=16  loss=3.357
  ├ encoder_mol     85,524 params  (Linear(16→129) + LorentzAttention + 129×129 proj)
  ├ encoder_pocket  85,008 params
  └ encoder_route   35,843 params
  (encoder_intent: NOT PRESENT — 4 塔里 intent 塔状态未保存)
  total: 206,375 params

models/esm2/esm2_t33_650M_UR50D.pt        (ESM2 蛋白编码器；但 HUMU pocket encoder 代码未引用)

models/mf-generators/*/checkpoints/       (8 个目录，全部仅含 .gitignore + .gitkeep)
```

### 2.5 配置一致性问题

- `infra/kubernetes/namespaces/mf-data-ns.yaml` netpol 允许 **19530**（Milvus 默认端口），但 `configs/services/qdrant.yaml` 与 docker-compose 实际暴露 **16333/16334**（Qdrant）。
- `infra/kubernetes/namespaces/mf-oracles-ns.yaml` netpol 允许 **50061–50067**，与各 oracle 服务 main.py 实际监听端口（50053/50056/50058）**不一致**。
- `infra/docker/base/Dockerfile.oracle:GNINA_SHA256` 是 `PLACEHOLDER_UPDATE_WITH_ACTUAL_SHA256_FROM_RELEASE`，构建必然失败。

---

## 3. 完成度矩阵

| # | 层 | 设想关键件 | 当前状态 | 完成度 |
|--|---|---|---|---|
| 0 | 元架构 / JMCG | 联合分布 P(m,r,p\|T,c) 共生成 | 三塔对比训练存在，无 FTO/性质轮廓闭环 | 30% |
| 1 | CIC / CIG / HCIV | LLM 工具调用编译 → JSON-LD 超图 → 双曲 128 维 | regex 启发式 + 真实 HTTP grounding；CIG 非超图；HCIV 默认 hash | 25% |
| 2 | HUMU 流形 + 编码器 | Lorentz manifold + 三塔 SE(3)/E(3) + TreeLSTM | Lorentz 算子真实；四塔均启发式（RDKit 2D / 字符计数） | 20% |
| 3 | AMGE 八生成器 | 8 类共享 HUMU + TAR ProxylessNAS + KD | 8 类目录齐；3 类（HFM/FragFM/CReM）有 artifact 路径但 decode = JSON 字典；4 类（LaMGen/MMPT/EvoMol/UAS）必须外部 runner，无 runner 即 raise；iCLM 是 MD5 hash 固定池 | 15% |
| 3b | TAR + KD | ProxylessNAS + REINFORCE + Boltz 蒸馏 | MLP + softmax + running mean；KD teacher = SMILES 字符启发式；蒸馏 loss = MSE-to-zero | 10% |
| 4 | MARB 多智能体 | 9 agent + CRG + LangGraph + Sigstore | LangGraph StateGraph 真存在但节点空操作；6 个 agent 是占位；Critic 100 规则真实；critic-svc 完全无关联硬编码 | 25% |
| 5 | Oracle 级联 | L0–L4 五级 + 不确定度门控 | L0 RDKit 真；L1–L3 全是 runner 转发器，自身无模型；L4 完全缺失；不确定度门控 0 行业务代码 | 15% |
| 6 | FTO / Patent Dead Zone | 4 数据源 + Markush 解析 + 双层 FTO + HUMU 反馈 | 4 客户端硬编码假数据；patent index 真接 Qdrant；FTO agent 是常量返回；Dead Zone 不写 Neo4j | 15% |
| 7 | RetroSyn / SRB | RetroGNN + RSGPT + UAlign + AiZynth + XDL + SiLA2 | RetroGNN 缺失；3 wrapper 均 ~33 行 thin runner；retrosyn_agent 是常量；SRB compiler 真实但 agent.process 不调用；XDL 仓内最小实现；SiLA2 仅字段占位 | 20% |
| 8 | DKI 数据基础设施 | Milvus / Neo4j / Postgres / MinIO / Feast | **Milvus 整体替换为 Qdrant**；其它 4 项真实可用；Feast repo 目录不存在；默认运行路径退到 SQLite | 50% |
| 9 | 工程实施 / 服务化 | 22 服务 + K8s + Helm + Terraform | 24 服务目录；**0 gRPC 服务注册 servicer**；buf 未生成 pb2；Helm chart 空壳；Terraform 仅 tfvars；docker-compose 只编排 3 个服务 | 25% |
| 10 | Provenance / Audit | Sigstore + Rekor + OpenTelemetry | sigstore SDK 探测后立即 fallback SHA256；无 Rekor 调用；OTel 未接入 | 30% |
| 11 | 评估体系 / Pilot | MOSES / GuacaMol / PMO / CrossDocked / KRAS pilot | MOSES 4 项本地复刻；GuacaMol/PMO 全 stub 且 collect=0；KRAS pilot 占位断言；audit E2E 占位 | 10% |

加权完成度约 22–25%。

---

## 4. 分层深度对比

### 4.0 元架构与 JMCG

**设想**：分子 m、合成路径 r、性质轮廓 p 作为联合随机对象 (m,r,p)，在双曲流形 ℍ^d 上学习 P(m,r,p|T,c)；生成、验证、合成规划在统一空间共生成。

**实际**：
- `pipelines/humu_pretrain/.../pipeline.py:612-622`：真实跑 `mol-pocket / mol-route / pocket-route / mol-intent` 四组 in-batch 对比损失 + curvature regularization。
- **FTO 损失（设想要点）完全缺失**：`L_fto(z_mol, z_patent)` 在 pipeline.py 中无对应实现，patent 嵌入未进入训练目标。
- **性质轮廓 p 未进入分布**：oracle 反馈（亲和力、ADMET）从未写回 HUMU；retrosyn 找到路径后不调任何 HUMU 编码器。
- **真实 checkpoint 实测**（`checkpoints/humu_4h200/best_model.pt`）：epoch=16, loss=3.357, total 206,375 params，**intent 塔状态未保存**。
- 闭环训练（oracle/FTO → HUMU → generator 反馈）：0 行实现。

### 4.1 CIC / CIG / HCIV

**设想**：
- 三阶段编译：LLM (SRM) + 工具调用 (UniProt/PDB/SureChEMBL/ChEMBL) → CIG（JSON-LD 有向超图）→ HCIV（双曲 128 维 Lorentz）+ Intent Cone。

**实际**：
- **NL 解析无 LLM**：`agents/nl2obj/parser.py:3` 注释明言 "Pure-Python parser (regex + heuristics), no LLM call required"。grep `LLM|openai|anthropic|tool_call|tool_use` 在 CIC 范围内仅命中 parser.py 的"no LLM"注释。
- **三套并存且互不调用**：
  1. `services/nl2obj-svc/src/nl2obj_svc/main.py:12-37` — gRPC Parse() 返回**硬编码** objectives JSON。
  2. `agents/nl2obj/src/nl2obj/parser.py` — regex + 词典 410 行（含 ~13 个药物 SMILES + ~25 个靶点正则 + ~25 个 SMARTS 模式）。
  3. `services/cig-compiler-svc/.../stages/stage1_semantic.py:5-11` — 又一份 regex（仅 5 个属性关键字）。
- **Grounding 真实**：`stage1b_grounding.ground_knowledge()` 真用 `urlopen` 调 UniProt / RCSB PDB / ChEMBL / SureChEMBL 四源（`stage1b_grounding.py:1-132` 与四个 `tools/*_tool.py`）。无缓存层，触发条件是 regex 命中已知 gene 名。
- **CIG 非超图**：`libs/mf-core/.../types/cig.py:91` 行 Pydantic 模型，`ObjectiveNode` 平铺列表，`ObjectiveEdge` 类型存在但 `stage2_cig_build` 不生成任何边；无 `@context`/`@id` JSON-LD 字段。
- **HCIV 三模并存**：
  - HASH（`hciv_encoder.py:92-103`）：SHA-256 字节铺到 Lorentz 球面，确定但与化学无关。
  - RANDOM（`hciv_generator.py:7-18`）：`sin(基向量 * 12.9898 + seed * 78.233) * 0.5`。
  - LEARNED（`hciv_encoder.py:24-29`）：`nn.Linear(64, dim*2)` 单层；输入 64 维**只用 4 位**（位置 0/28/29/30 对 obj 数量与 fto/admet/affinity 标志），其余 60 位恒为 0。
- **HCIV checkpoint 缺失**：`production_real + learned` 路径强制要求 `HCIV_CHECKPOINT_PATH` 否则 `RuntimeError`；仓库内**无该 ckpt 文件**。
- **维度宣称不一致**：`types/humu.py` 默认 `coordinates=[0.0]*129, dim=128`；`HCIVEncoder` 默认 `dim=32`；compiler 默认 `hciv_dim=128`。
- **CIC Refine 端点未实现**：`services/cig-compiler-svc/.../main.py` `Refine` 直接 `abort_unavailable("CIG refinement runner is not configured")`。

### 4.2 HUMU 流形与编码器

**设想**：Lorentz ℍ^128；三塔编码器（SE(3)-GNN 分子 + EquiBind E(3)-GNN 口袋 + 双向图 Transformer + 双曲 TreeLSTM 路径）；联合对比损失；可学习曲率；Patent Dead Zone 障碍势；OOD 不熟悉度门控；EHVI-PoF 双曲 GP。

**实际**：

| 件 | 设想 | 实际 | 证据 |
|---|---|---|---|
| Lorentz manifold 算子 | 完整 | 真实完整 | `libs/mf-humu/.../manifold/lorentz.py:72-94` exp/log/distance 闭式解 + 数值稳定 |
| 可学习曲率 | nn.Parameter | **常量 float** `self.k = curvature` | `lorentz.py:15-25`；grep `nn.Parameter.*curv` 全仓 0 命中 |
| 分子编码器（SE(3)-GNN） | 设想 | RDKit 16 维原子特征 + 邻接归一化 2 次平均 + Linear + 普通点积 attention；**无 3D 坐标**；**非 SE(3) 等变** | `models/mf-encoders/humu_mol_encoder/.../encoder.py:27-106` |
| 口袋编码器（EquiBind E(3)-GNN + ESM-2） | 设想 | 12 维点特征（坐标 + 元素 + 残基 one-hot）+ Linear + 普通 attention；**ESM-2 未引用**（虽 yaml 写了 `esm2_checkpoint`） | `humu_pocket_encoder/.../encoder.py:15-96` |
| 路径编码器（双向图 Transformer + 双曲 TreeLSTM） | 设想 | 反应 SMILES 字符串字符计数（18 维）→ MLP；**grep `TreeLSTM` 全仓 0 命中** | `humu_route_encoder/.../encoder.py:64-75` |
| LorentzAttention | "Lorentz 等变" | 内部是 ambient space Linear + 普通点积 softmax | `libs/mf-humu/.../encoders/lorentz_attention.py:29-57` |
| IntentCone 采样 | 在意图锥内采样 | 维度**硬编码 129**，不读 cone.axis 实际 shape | `libs/mf-humu/.../operations/intent_cone.py:24-39` |
| Patent Dead Zone 势能 | Lorentz | **欧氏距离 + Python 双重 for-loop**（O(B·N)） | `libs/mf-humu/.../operations/dead_zone.py:22-33` |
| Unfamiliarity | autoencoder reconstruction | 基础 topk-mean 欧氏距离；无训练过的 autoencoder 产物 | `libs/mf-humu/.../operations/unfamiliarity.py` |
| 双曲 GP / EHVI | SVGP + 双曲 Matern + EHVI-PoF | EHVI 实现真实存在 | `libs/mf-humu/.../gp/{svgp.py:215,ehvi.py:135,kernels.py:91}` |
| 联合训练 pipeline | 设想 | 真实存在 1268 行训练循环 + DDP + AMP + checkpoint rotate + 验证 | `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py` |
| FTO loss 接入 | 设想 | **缺失**：`L_fto(z_mol, z_patent)` 在 pipeline.py 中无实现 | — |

### 4.3 AMGE 八生成器 + TAR + 跨范式 KD

**设想 vs 实际逐项对比**：

| 生成器 | 设想 | 实际 generate 实现 | 真实 3D | HUMU z 驱动 | checkpoint | 关键代码 |
|---|---|---|---|---|---|---|
| **HFM-3D** | Lorentz Flow Matching 20 步 + SE(3) 等变切空间向量场 + 学习 decoder | 20 步 ODE + Lorentz manifold 真实；**decoder = JSON 字典 L2 最近邻**；**3D 坐标来自 RDKit UFF** 而非 flow model；`LorentzEquivariantLayer` 文件存在但未被 generator 引用 | 是（RDKit UFF，与 flow 无关） | 部分（IntentCone 传入但 fallback 高斯+expmap） | 空 | `hfm_3d/generator.py:174-212` |
| **FragFM** | 双层 DFM (scaffold + R) + SA-aware rate matrix | `assembly_rules[i % N]` **JSON 列表轮询**；`TwoLevelDFM` / `SAAwareRateMatrix` / `FragmentVocabulary` 类齐备但**未被 generator 调用** | 否 | 否 | 空 | `fragfm/generator.py:111-122` |
| **LaMGen-3D-Pro** | 多靶点 LLM 0.3s/mol k≤10 | generator 61 行，无 runner 即 `raise RuntimeError("LAMGEN_RUNNER is required")`；`MultiTargetAttention` 硬编 `n_targets=4` ≠ 设想 k≤10；服务端无条件抛错 | 否 | 否（透传 runner） | 空 | `lamgen_3d/generator.py:22-24`；`lamgen-generator-svc/main.py:46` |
| **iCLM** | 在线持续学习 EWC + PackNet | **MD5 hash 索引 10-SMILES 固定池**；EWC（35 行）/PackNet（24 行）/OnlineLearner（18 行）类齐备但**与 generator 完全断开** | 否 | 否 | 空 | `incremental_clm/generator.py:17-23` |
| **MMPT-RAG** | ChEMBL 正样本 + SureChEMBL 负样本 + FTO-aware 解码 | 3 硬编码 seeds × 3 硬编码 MMP 规则（F↔Cl↔Br、OC→OCC），**纯字符串 `.replace`** 未经 RDKit 校验；FTO-aware 解码 0 行实现 | 否 | 否 | 空 | `mmpt_rag/generator.py:27-43` |
| **EvoMol-RL** | HVI reward + sleeping bandit + Pareto 存档 | HVI（72）/HV reward（57）/Sleeping bandit（29）/ParetoArchive（25）**算法真实可用**；**但 generator.generate 委托 runner**，算法模块成孤岛 | 否 | 否 | 空 | `evomol_rl/generator.py:24-25` |
| **CReM-pharm-3D** | DiffDock-L 2s/pose 实时打分 + 片段替换 | JSON `mutations` 列表轮询；`fragment_replacement.py` 真实（RDKit `CombineMols`+`AddBond`）但**未被 generator 引用**；**DiffDock 集成 0 行**（grep 0 结果） | 否（无 sdf_bytes） | 否 | 空 | `crem_3d/generator.py:99-118` |
| **UAS** | autoencoder reconstruction OOD-aware 采样 | runner 桩，generate 不调任何 AE/OOD 组件，直接转发 runner；无 uas-svc | 否 | 否 | 空 | `uas/generator.py:55-56` |

**TAR 路由器（设想 ProxylessNAS + REINFORCE）**：

```python
# libs/mf-core/src/mf_core/routing/task_router.py:94-104
def forward(self, hciv, profile):
    h_proj = self.projection(hciv)
    t_proj = self.task_projection(task_vec)
    combined = h_proj + t_proj
    logits = torch.matmul(self.gen_embeddings, combined)
    weights = F.softmax(logits, dim=0)
```

- **不是 ProxylessNAS**：`nn.Linear(128→32) + nn.Linear(8→32) + nn.Parameter(8×32)` MLP + softmax。
- **不是 REINFORCE**：`update_with_feedback` 仅 running mean（`task_router.py:194-197`）。
- Hard rules（4 条）真实存在：`low_data` / `high_fto` / `lead_opt` / `scaffold_hop`（`task_router.py:60-77`）。
- **两套互不连通的路由器并存**：
  - `mf_core.routing.TaskAwareRouter`（库内）
  - `services/generator-router-svc/.../online_learner.py` 是**完全另一套 Thompson Sampling**（Gamma 采样 8 个匿名 `gen-0..gen-7` 槽位），与 8 个真实生成器名无映射，且**不引用** `TaskAwareRouter`。
- **Generator Coordinator Agent 名称错位**：`agents/generator_coord/.../agent.py:16-22` 硬编 5 个**不存在的名字** `["template_based", "llm_direct", "evolutionary", "fragment_growing", "scaffold_hopping"]`，与 mf-generators 下 8 个真实模块全不对齐。

**跨范式 KD**：

```python
# libs/mf-core/src/mf_core/routing/cross_paradigm_kd.py:144-150
for emb, idx in zip(embeddings, indices):
    teacher_target = torch.zeros_like(emb).detach()  # <-- 全零教师目标
    mse = torch.nn.functional.mse_loss(emb, teacher_target)
```

- **教师 = SMILES 字符特征启发式**：`WeakTeacher`（cross_paradigm_kd.py:26-44），按字符串长度、环字符、白名单字符算 0.5±x 标量。
- **蒸馏 loss = MSE-to-zero**：把 embedding 压向零向量，并非真正的 student-teacher 蒸馏。
- 无 Boltz-2 / HypSeek 教师引用。

### 4.4 MARB 多智能体 + CRG + Sigstore

**设想**：9 Agent + CRG（JSON-LD beliefs+edges）+ LangGraph 状态机 + Sigstore/Rekor + JSON-LD 消息协议 + NATS JetStream + OCC + Critic 独立 LLM 族。

**实际**：

| 件 | 设想 | 实际 |
|---|---|---|
| Orchestrator | LangGraph StateGraph 真实 | LangGraph StateGraph 真存在（`workflow/graph_builder.py:27-47`），5 节点 + 条件路由真实；**但节点只追加 status 字符串，不调任何 agent**。**同包内有 3 条独立执行路径**：StateGraph、`ReasoningPipeline`（pipeline.py 585 行，真业务逻辑）、`agent.py` for-loop 9 字符串，三者互不共享状态 |
| NL2Obj | LLM 工具调用 | regex（见 §4.1）；`agent.py:20-36` 仅回显输入，**不调 parser** |
| GeneratorCoord | 真分发到多生成器并行 | 仅返回固定字符串列表（5 个不存在的策略名） |
| RetroSyn agent | 3 层逆合成 + HUMU 回写 | 37 行常量返回 `{"layers":..., "pathways":[]}` |
| Validation agent | L0–L4 cascade + 不确定度门控 | 87 行真实 cascade 逻辑，但 oracle 通过构造函数注入，无连接代码；**仅 L0 做 `admet_score >= l0_threshold` 判断**，L1/L2/L3 一律 `return True`；从未调用 `predict_with_uncertainty` |
| FTO agent | SureChEMBL + Markush + 双层 | 37 行硬编码 `{"patent_matches":0, "ip_risk":"low"}` |
| Supply agent | Enamine REAL 49B + 4 数据源 | 40 行硬编码常量 |
| Critic | 100 规则 + 不同 LLM 族 (Gemini/DeepSeek) | **100 规则文件真实存在 2165 行**（`agents/critic_agent/.../rules/rule_001..rule_100`），importlib 自动加载；**但多数规则仅从 `properties` dict 读预计算字段做阈值比较，不重算分子性质**；**0 LLM 调用**（grep Gemini/DeepSeek 全仓 0 命中） |
| SRB | SSP + XDL + SiLA2 | `compile_ssp` (`agents/srb_agent/.../compiler.py:22-58`) 真实可用；**但 `agent.py:process` 不调 compile_ssp**，返回占位 |
| critic-svc | 微服务 | **完全独立于 critic_agent**，74 行 gRPC servicer 返回硬编码 JSON `drug_likeness=0.78, novelty=0.65, "promising"`，与 100 规则 lib 无任何连接 |

**CRG 实现**：
- `libs/mf-core/.../types/crg.py:5-29` 与 `libs/mf-agents/.../crg/graph.py:48-49,94` 是真实 Pydantic 数据类，含 add_belief/add_edge/update_belief/query/to_crg。
- **三处 schema 不一致**：
  - `protos/.../core/crg.proto:32` 用 `beliefs / source_belief_id`。
  - `libs/mf-core/.../types/crg.py` 用 `beliefs / source_belief_id`。
  - `schemas/crg.schema.json:11-24` 用 `nodes / source_id / target_id`。
- **无 OCC / 向量时钟**：仅 `self.crg.version += 1` 单调递增，无写入冲突检测；grep `OCC|optimistic|vector_clock|cas_` 在 mf-agents/mf-core 0 命中。
- **9 个 agent 各自实例化本地 CRG**（无共享/同步代码）。

**Sigstore 签名**：

```python
# libs/mf-agents/src/mf_agents/lineage/sigstore_signer.py:22-30
try:
    from sigstore.sign import SigningContext
    SigningContext.production()
except ImportError:
    pass  # fallback to HMAC-SHA256
h = hashlib.sha256()
h.update(self.identity.encode()); h.update(payload)
return h.digest()
```

- 即便 import 成功也**不调用** sigstore 签名 API，立即落入 sha256 stub。
- `services/provenance-svc/.../sigstore_integration.py:62-72` 永远走 `local_dev_signature` 分支；检测到 sigstore 类型时直接 `raise RuntimeError("Sigstore signing backend is not configured")`。
- 无 Fulcio / Rekor / OIDC token 任何 HTTP 调用代码。`rekor_url` 仅作字符串保存。

**消息总线**：
- `protos/.../agent/message.proto:5` 注释 "universal envelope for all inter-agent NATS communication"。
- **实际实现是 Redis**：`libs/mf-agents/.../messaging/redis_bus.py:29-37` 用 `redis>=5.0` pub/sub + in-process FallbackBus。
- grep NATS/jetstream 在业务代码 0 命中。

### 4.5 Oracle 级联

**设想 L0–L4**：QED/SA/Lipinski/PAINS → Boltz-2+ADMET-AI+Chemprop → DiffDock-L+GNINA → OpenFE RBFE → GPU4PySCF+ORCA；不确定度门控 + 不确定度传播。

**实际**：

| 级 | 设想 | 实际 | 证据 |
|---|---|---|---|
| L0 | QED+SA+Lipinski+PAINS | QED+SA+Lipinski **真实**；**PAINS 缺失** | `models/mf-oracles/rdkit-oracle/.../scorer.py:30-62` |
| L1 | Boltz-2+ADMET-AI+Chemprop | 5 个 oracle wrapper 均「runner 转发器」；`evaluate` 在 runner=None 时 raise；pyproject 仅依赖 `mf-core, numpy`，**未列出 admet-ai/boltz/torch** | `models/mf-oracles/admet_ai/.../oracle.py:78` 等 |
| L2 | DiffDock-L + GNINA 重排 | 两 oracle 都是 wrapper；服务 `dock-svc/main.py:67` 通过环境检查后**仍主动抛** `RuntimeError("{docking_engine} docking runner is not configured")` | — |
| L3 | OpenFE RBFE | wrapper；`openfe` 包**未在 uv.lock**；`skip_when_unavailable` 可全部跳过 | — |
| L4 | GPU4PySCF DFT + ORCA | **完全缺失**：`ValidationAgent.oracle_levels[4]` 仅是字符串 `"experimental_assay"`，无对应 oracle 类、pyproject、依赖 | `agents/validation_agent/.../agent.py:14-20` |
| 不确定度门控 | 设想 | **0 行**：cascade 调度器从未调 `predict_with_uncertainty`；rdkit_oracle 不确定度恒为 0 | `agent.py:79-87` 行只在 L0 上做 admet_score 阈值，L1/L2/L3 一律 `return True` |
| L1/L2/L3/L4 服务端 | 真实推理 | `admet-svc/main.py:62`、`dock-svc:67`、`boltz2-svc:63`、`fep-svc:59` **通过环境检查后仍 raise** "not configured" | — |

**mf-chem 预测引擎（与 L1 重叠）**：
- `libs/mf-chem/src/mf_chem/predict/engine.py:472` 行，包含真实 RDKit 描述符 + 一组**未训练 weights** 的 `_PropertyHead`（`nn.Sequential(Linear→GELU→...)` ，注释明言"Without trained weights the head is a fixed-seed init that yields deterministic outputs"）。
- `_predict_admet`（行 397-427）和 `_cpu_admet_from_props`（429-441）从 logp/MW/tpsa 经**线性公式手算** logD/logS/clearance/half_life/PPB/hERG/Caco2——这是当前 ADMET 输出的真正来源，不是设想的 ML 模型集成。

### 4.6 FTO / Patent Dead Zone

**设想**：SureChEMBL/USPTO/Reaxys/Google Patents 实时；Markush 解析引擎；双层 FTO（结构相似性 + 权利要求语义）；Patent Dead Zone 写回 HUMU。

**实际**：
- **fto_agent**（37 行）：`process()` 不论输入返回常量 `{"patent_matches": 0, "structure_novel": True, "ip_risk": "low", "blocking_patents": []}`（`agents/fto_agent/.../agent.py:28-36`）。**不调** fto-patent-svc。
- **fto-patent-svc** `main.py` 187 行：`SearchPatents` 真实工作——读 `PATENT_INDEX_URI=file://...` JSON 文件 → **SMILES 字符串相等比较**过滤（非结构相似性）→ 返回。`CheckDeadZone` 与 `AnalyzeMolecule` 直接 raise "not configured"（行 166、174）。
- **4 个数据源客户端全部硬编码假数据**：
  - `surechembl.py:30` 注释 "In production: httpx call to SureChEMBL API"，下方硬编码返回 `WO-2024-123456` 等假记录。
  - `uspto.py:30-39`：硬编码 `12123456 / Biotech Inc. / Smith, John, Doe, Jane`。
  - `google_patents_bq.py:38-49`：硬编码 jurisdictions 循环假记录。
  - `reaxys.py:49-70`：硬编码"Markush 结构"返回 R1/R2/R3 替代基。
- **uv.lock 中无任何数据源 SDK**：`surechembl / patentsview / google-cloud-bigquery / reaxys` 全缺。
- **DeadZoneUpdater**：`dead_zone_updater.py:56-79` 真实 Tanimoto；但 `sync_to_graph` 行 81-89 注释 `# Cypher MERGE would go here in production` 实际只对 zone 累加计数返回，**未写 Neo4j**。
- **patent_indexing pipeline**：真实可用（355 行），`tests/integration/test_dki_patent_indexing.py` 已通过，能写入真实 Qdrant `patents_embedding` 并 search。引用的 `pipelines/patent_indexing/build_faiss.py` / `build_milvus.py` / `markush_extractor.py` 在 DVC stage 中提及但**实际不存在**。

### 4.7 RetroSyn / SRB

**设想**：3 层 retrosyn（RetroGNN → RSGPT+UAlign → AiZynth 4.0 MCTS + Enamine/REAL）→ HUMU 回写；SRB → SSP → XDL 2.0 → SiLA2。

**实际**：

| 件 | 设想 | 实际 | 证据 |
|---|---|---|---|
| RetroGNN 快速过滤 | Layer A | **完全缺失**（grep 全仓 0 命中） | — |
| RSGPT wrapper | 真实模型 | 33 行 thin runner，`find_routes` 若无 runner `raise RuntimeError("RSGPT_RUNNER is required")`；pyproject 仅依赖 mf-core | `models/mf-retrosyn/rsgpt/.../retrosyn.py` |
| UAlign wrapper | 真实模型 | 同上 33 行 thin runner | — |
| AiZynthFinder MCTS | Layer C | wrapper 33 行，**不依赖 `aizynthfinder` 包**；`uv.lock` 中无 `aizynthfinder` | — |
| retrosyn_agent | 调度三层 | 37 行常量返回 `{"layers":{"strategy":..., "pathways":[], "reactions":[]}}`，**无任何层调度逻辑** | `agents/retrosyn_agent/.../agent.py:28-36` |
| retrosyn-svc | gRPC | `PlanRoutes` / `ScoreRoute` 直接 raise "not configured" | `retrosyn-svc/main.py:51,59` |
| HUMU 回写 | 路径找到后更新 z_mol | **0 行**（grep 0 命中） | — |
| SRB compiler | SSP 编译 | **真实可用** 158 行：将 `retrosyn_route.steps` 编译为 SSP/SSPStep/SSPMaterial；估算 total_yield/cost；对路由字段严格校验 | `agents/srb_agent/.../compiler.py:22-58` |
| SRB agent.process | 调用 compiler | **常量返回** `{"protocols":[{"pathway_id":i, "steps":[], ...}]}`，**未调用 `compile_ssp`** | `agents/srb_agent/.../agent.py:26-38` |
| XDL 2.0 编译 | 兼容 ChemputerXDL | **仓内最小子集自实现**（`wetlab/xdl-compiler/.../compiler.py:6-69`）；不依赖外部 `xdl/ChemputerXDL` 包；硬件调度退化为统一 `reactor` | — |
| SiLA2 endpoint | 预留 | 仅 `SSP.sila2_endpoint: str \| None = None` 字段占位；grep `SiLA2` 全仓无客户端代码 | `libs/mf-core/.../types/ssp.py:44` |

**Supply Oracle**（核心架构内）：
- **supply_agent**（40 行）：常量返回 0。
- **supply-oracle-svc**：`FileSupplyCatalog` 用 `smiles == record["smiles"]` 字符串相等查找本地 JSON；所有真实外部源（Enamine REAL 49B / Enamine in-stock / Mcule / eMolecules / Chemspace）**全部缺失**；**无 faiss 任何变体在 uv.lock**。

### 4.8 DKI 数据基础设施

**设想 → 实际**：

| 设想 | 实际 |
|---|---|
| Milvus 2.5 + 3 collection (molecules/pockets/patents) | **Milvus 整体替换为 Qdrant 1.12.4**；仅 `molecules_humu` 单 collection；patents 通过 patent_indexing pipeline 写入 |
| Neo4j 5 + 关系图 | Neo4j 5 community；`GraphRepository` 真实覆盖设想中 3 种关系 + 5 种额外关系；**仅 `PROVENANCE_STORE_MODE=production_real` 启用**，默认 `local_demo` 走 InMemory |
| PostgreSQL 16 + TimescaleDB | timescaledb-ha:pg16；3 个 alembic migration 真实；hypertables 在扩展可用时创建；**但 `MoleculeRepository` 是空壳**：`upsert` 调用 `self.session.execute(None)` 永不命中真实 SQL（`libs/mf-core/.../repositories/molecule_repo.py:15-42`） |
| MinIO/S3 | aiobotocore 真实；`object_store/paths.py` 命名规范完整；仅 provenance-svc 在 production_real 模式真实使用 |
| Feast (Redis online + Postgres offline) | 服务 + 依赖 + alembic schema 齐备；**`FEAST_REPO_PATH` 指向的目录在仓库中不存在**；无 `feature_store.yaml` / FeatureView / entity 定义 |
| 默认数据存储 | **api-gateway 默认 SQLite**（`/workspace/MForge/moleculeforge/data/moleculeforge.db` 已存在 ~504KB），`mf_core.db.store` 启动时 init_db 真实可用；80+ seed 已知药物分子 |

### 4.9 工程实施 / 微服务

**设想 22 服务 + K8s + Helm + Terraform + GPU 节点池**：

| 件 | 设想 | 实际 |
|---|---|---|
| 微服务数 | 22 | services/ 下 24 个目录 |
| gRPC servicer 注册 | 22/22 | **0/22**——`grep -rEn "_pb2_grpc\|add_.*Servicer_to_server" services/` **无任何匹配**；`find -name "*_pb2*.py"` 也无任何生成文件 |
| buf 生成的 Python pb2 | 必须 | `protos/buf.gen.yaml` 指定输出到 `libs/mf-core/src/mf_core/proto_gen`，**该目录不存在**；buf gen 从未执行 |
| K8s namespace | 5 (mf-data/mf-oracles/mf-generators/mf-agents/mf-mlops) | 5 个 ns 真实存在 + ResourceQuota + LimitRange + NetworkPolicy |
| K8s Deployment | 必须 | **0 个 Deployment/Service/HPA/DaemonSet/Ingress** |
| GPU 节点池声明 | H100/A100/A40 | ResourceQuota 给了 `nvidia.com/gpu` 配额（mf-oracles 15、mf-generators 22），**无 Deployment 消费**；无 GPU 节点 selector / tolerations / `nvidia.com/gpu.product=H100` 型号过滤 |
| docker-compose 服务编排 | 22 服务 | dev.yaml 仅声明 **3 个**：`humu-encoder-svc / generator-router-svc / api-gateway`；其余 19 个 in 任何 compose 中均未定义 |
| Helm | 完整 chart | `Chart.yaml` 声明 7 个 subchart 依赖；**`charts/` 与 `templates/` 目录均不存在**；空壳无法 `helm install` |
| Terraform | 完整 module | 仅 `environments/{dev,staging,prod}/terraform.tfvars` 3 行；**无任何 .tf 文件、provider、module** |
| 推理优化 | ONNX + TensorRT-LLM | onnxruntime-gpu 在 lock 但无业务 import；**TensorRT 0 命中** |
| MLOps | MLflow + W&B + DVC + LakeFS | mlflow / wandb 包在 lock；**业务代码无 `import mlflow` / `import wandb`**；`dvc` 未在 pyproject 声明；LakeFS 0 命中 |

### 4.10 Provenance / Sigstore / Audit

| 件 | 设想 | 实际 |
|---|---|---|
| Provenance store | Neo4j + Postgres + MinIO | `ProductionProvenanceStore` 真实存在，`tests/integration/test_dki_provenance.py` 真实跑通；默认 `local_demo` 走 InMemoryProvenanceStore |
| Sigstore | Fulcio + Rekor 真实签名 | 永远 fallback SHA256（`sigstore_integration.py:62-72`）；探测到 sigstore 类型时反而 raise；无 cosign/fulcio/OIDC/Rekor HTTP 调用 |
| Rekor 透明日志 | 设想 | `get_rekor_entry` 用 `int(hashlib.sha256(...)[:12], 16)` **伪造** log_index 字段 |
| OpenTelemetry trace 传播 | 设想 | 仅 graph_builder 中 `trace_id` 写入 events；**未传播到下游消息或子 agent**；无 OTLP exporter 配置 |
| Audit E2E | 真实跑通 | `tests/e2e/test_audit_completeness.py` 需 `RUN_AUDIT_E2E=1`，**测试体仅** `assert os.environ.get("PROVENANCE_SVC_URL")` 等环境检查，无真实链路调用 |

### 4.11 评估体系 / Benchmark / Pilot

| 设想指标 | 实际 |
|---|---|
| MOSES 2.0（FCD/SNN/Frag/Scaff） | `libs/mf-eval/.../moses.py` 仅 4 项本地复刻（validity/uniqueness/novelty/diversity）；**FCD/SNN/Frag/Scaff 缺失**；validity 在 rdkit 不可用时 fallback 硬编码 `0.95` |
| GuacaMol v3（22 任务 MPO/rediscovery） | `tests/benchmark/guacamol_benchmark.py` 5 个 stub 全 `pytest.skip`；无 `guacamol` 库引用；**collect=0**（文件命名错） |
| PMO 23 任务 | 同上全 stub；无 mol_opt 库引用；**collect=0** |
| CrossDocked 2020 v2 对接评估 | 仅作为 HUMU 预训练 24,242 pocket 数据源；**无评估代码** |
| DUD-E EF1% HypSeek | grep `dud/dude/EF1/hypseek` 全空 |
| Pareto HV 评估 | `libs/mf-humu/.../gp/ehvi.py` 是 BO 采集函数；**无离线 HV 评估器** |
| Unfamiliarity AUROC | grep `auroc/calibration` 全空 |
| HUMU 双曲嵌入 distortion | grep `tree.distortion` 全空；仅有距离正确性测试 |
| Activity cliff Mann-Whitney | grep `mann.whitney/cliff.separation` 全空 |
| 分子-路径嵌入一致性 | mf-eval/humu/ 子模块**根本不存在** |
| KRAS G12C E2E | `tests/e2e/test_kras_g12c_pilot.py` 154 行 7 测试，全 skipif `RUN_KRAS_G12C_E2E=1`；**测试体仅 `assert nl_input` / `assert os.environ.get("HFM_CHECKPOINT_PATH")` 等占位**，无真实 pipeline 调用 |
| Audit E2E | 占位（见 §4.10） |
| 唯一可真跑的 e2e | `tests/e2e/test_predict_api.py` 10/10 通过；`tests/e2e/test_reason_workbench.py` 5 项（2 项因 SQLite readonly 失败）；只走 api-gateway，不涉及 HFM-3D/HUMU/RetroSyn 真实模型 |

`libs/mf-eval/` 仅 `molecule/moses.py` 1 个模块，README 声称的 `humu/ pareto/ agent/` 三套子模块 0 实现；`mf_eval/__init__.py` 空文件；生产代码 0 处 `import mf_eval`。

---

## 5. 跨层全局发现

### 5.1 互不连通的并行实现（"双层并存"）

| # | 现象 | 影响 |
|---|---|---|
| 1 | NL 解析三套（nl2obj-svc 硬编码 / nl2obj/parser.py regex / cig-compiler-svc stage1_semantic.py regex），互不调用 | 改一处不影响其他两处；不知道哪条是"主路径" |
| 2 | TaskAwareRouter（mf_core 库）vs generator-router-svc/OnlineLearner（Thompson Sampling），互不引用 | 服务路由器只有 8 个匿名 `gen-0..gen-7` 槽位，无生成器名映射 |
| 3 | critic_agent（100 规则 lib）vs critic-svc（硬编码 gRPC），互不关联 | 100 规则在仓内是死代码（无 critic-svc 调用方） |
| 4 | Orchestrator 3 条执行路径（StateGraph + ReasoningPipeline + agent.py for-loop），互不共享状态 | LangGraph 是空操作，真业务在 pipeline.py；外部用户不知道入口 |
| 5 | Generator Coordinator Agent 硬编 5 个不存在的策略名 vs TAR.GENERATOR_NAMES 8 个真实名 vs 8 个实际生成器 | 三层名字不对齐，agent 无法实际调度 |
| 6 | Schema 三处不一致：proto crg `beliefs/source_belief_id` ↔ Python crg.py `beliefs/source_belief_id` ↔ JSON Schema `nodes/source_id` | 协议层与 schema 层不可互译 |
| 7 | api-gateway 默认 SQLite（store.py）vs 设想 PostgreSQL+TimescaleDB；MoleculeRepository 是空壳（execute(None)） | 默认运行路径与设想数据流脱节 |
| 8 | 8 类生成器：算法模块（DFM/EWC/PackNet/HVI/Bandit/AE/OOD）齐备但**全部未被 generator.generate 引用** | 生成器目录里有大量"博物馆代码" |

### 5.2 缺失但应该存在的依赖（pyproject + uv.lock 审查）

**应该有但缺失**（设想需要）：
- `pymilvus`（Milvus 已被 Qdrant 替换，可视为已决定）
- `dvc`（DVC 流程文件存在但无依赖）
- `lakefs-client`
- `nats-py`（设想 NATS JetStream 编排，已被 Redis 替换）
- `sigstore`（provenance-svc 中 import 探测，但 lock 未声明）
- `tensorrt / tensorrt-llm`
- `hvac`（Vault）
- `aizynthfinder / rdchiral`
- `gpu4pyscf / pyscf / psi4`
- `mace`（MACE-OFF24 力场）
- `faiss / faiss-cpu / faiss-gpu`（Enamine REAL ANN 索引）
- `botorch / gpytorch`（Pareto BO）
- `prefect / airflow / ray`

**存在但 0 引用**：
- `chemprop`（在 lock，无 import）
- `wandb`（在 lock，无 import）
- `mlflow`（在 lock，仅 pareto_bo pyproject 声明，无 src 代码）
- `onnxruntime-gpu`（在 lock + Dockerfile.oracle 安装，无 ONNX 模型加载/导出代码）

---

## 6. Claude 能力边界声明

> 本节回答用户问题："你是否能够完成，哪些又是你不能完成的"。
>
> 判断维度：
> - Claude（本次对话）能在 `/workspace/MForge/moleculeforge` 内独立修改/补全代码、构造确定性 stub、对接已有依赖。
> - 不能：拉取真实大规模数据、训练大型模型、对接需要付费 SDK 或需要 OAuth/OIDC 的外部服务、获取需要购买的专利数据集 SDK。

### 6.1 Claude 能独立完成（无需额外资源）

按设想缺口可单独 PR 解决的项：

**算法/库代码层**（纯本地代码修补）
1. **修 `tests/unit/test_ssp_compiler.py:16` 导入错误** → 改 `_build_steps` 为 `_build_steps_from_route`，重启 11 项 unit 测试。
2. **修 `tests/benchmark/*.py` 命名** → 全部重命名为 `test_*_benchmark.py` 或在 `pyproject.toml`/`pytest.ini` 添加 `python_files = ['test_*.py', '*_benchmark.py']`。让 benchmark 至少能被 collect。
3. **修 schema/proto/python crg 字段一致性** → 选定一套规范（建议沿用 proto/python 的 `beliefs/source_belief_id`），重写 `schemas/crg.schema.json`。
4. **修配置不一致**：
   - `infra/kubernetes/namespaces/mf-data-ns.yaml` netpol 19530 → 16333/16334。
   - `mf-oracles-ns.yaml` netpol 50061-50067 → 服务 main.py 实际端口。
   - `Dockerfile.oracle:GNINA_SHA256` 占位 → 改成构建时从环境变量读取，或暂时注释掉强制校验。
5. **修 anti_degradation 3 项失败** → 替换 `except Exception: pass` 为窄异常 + log；改名 `dummy_emb` → 业务名。
6. **CIG 三套 NL 解析合并** → 让 `services/nl2obj-svc/main.py` 委托 `agents/nl2obj/parser.py`，淘汰硬编码 JSON 分支；或反向让 cig-compiler-svc 调用 parser.py。
7. **Generator Coordinator Agent 重命名 5 策略 → 对齐 TAR.GENERATOR_NAMES 8 个真实生成器名**。
8. **TAR 与 generator-router-svc 整合** → 让 svc 调用 `mf_core.routing.TaskAwareRouter` 而非自实现 Thompson Sampling，或反向把 hard rules 迁入 svc。
9. **PAINS 过滤补全** → RDKit 内置 `FilterCatalog` 可直接接入 `models/mf-oracles/rdkit-oracle/.../scorer.py`，不需要外部资源。
10. **iCLM/UAS/CReM/FragFM 把孤立算法模块接入 generator.generate** → 这些类齐备，只需 4 行胶水代码即可不再走 JSON 字典轮询。
11. **SRBAgent.process 接 compile_ssp** → 1 函数调用替换占位。
12. **MoleculeRepository 真实实现** → ORM 已定义，把 `self.session.execute(None)` 替换为正确的 SQLAlchemy select/insert。
13. **生成 buf protobuf Python stub + 22 服务 servicer 注册** → 仅需正确配置 `protos/buf.gen.yaml` + 改 `tools/dev/generate_protos.py` 用本地 protoc + 在每个服务 main.py 添加 `xxx_pb2_grpc.add_XxxServicer_to_server(self, server)` 一行。
14. **`mf-eval` 模块补全** → 实现 `humu/distortion.py`（基于已有 `gp/kernels.py` 的双曲距离）、`humu/cliff_analysis.py`（Mann-Whitney U scipy 内置）、`pareto/hv_evaluator.py`（基于已有 `evomol_rl/hypervolume.py`）。
15. **HUMU 联合训练加入 FTO loss** → pipeline.py 已有 mol/pocket/route/intent 四损失结构，加 `L_fto` 只需在 patent 嵌入可用时增加一个 `_in_batch_contrastive_loss(z_mol, z_patent)`。
16. **可学习曲率改造** → `LorentzManifold.__init__` 把 `self.k = curvature` 改为 `self.k = nn.Parameter(torch.tensor(curvature))` 并在训练 pipeline 加入。
17. **Patent Dead Zone 改用 Lorentz 距离 + 批量矩阵化** → 已有 `manifold.distance` 闭式解，替换欧氏 + 双重 for-loop。
18. **不确定度门控写入 ValidationAgent** → 改 `agent.py:79-87`，让 L1/L2/L3 真正调 `predict_with_uncertainty` 并按方差决定升级阈值。
19. **mocking/print/bare-except 清理** → 工程债务清扫。
20. **Sigstore 真接入（无需外部凭据时）** → 仅在 `OIDC_TOKEN` 存在时调用 `sigstore.sign.SigningContext`，否则保留 SHA256 fallback 但显式标记 `signature_type="local_dev"`。

**工程层**：
- 写 22 个服务的 K8s Deployment manifest（基于 ResourceQuota 已声明的 GPU 配额）。
- 写 Helm subchart templates。
- 写 docker-compose 余下 19 个服务的编排。
- 写 Terraform module（如果有目标云）。
- 写 Feast `feature_store.yaml` + `feature_views.py`（数据已在 alembic v003 表中）。

### 6.2 Claude 能完成但需用户提供资源

| 缺口 | Claude 能做的 | 需用户提供 |
|---|---|---|
| HFM-3D 真实训练 | 写训练脚本（HFM 已有 train.py 174 行）、写 ChEMBL → SDF → flow matching 数据 pipeline | ChEMBL 34 完整下载 + GPU 算力（设想 4×H100/A100）+ 数月训练时间 |
| FragFM SA-aware DFM 真实训练 | 写训练循环 + SA 惩罚 rate matrix 接入 | USPTO/Pistachio 反应数据 + GPU |
| LaMGen-3D-Pro multi-target | 写 multi-target attention + speculative decoding 集成 | 基础 LLM 权重（Llama-3.3-70B + LoRA 或 Qwen2.5）+ 多靶点 ChEMBL 配对数据 + GPU |
| MMPT-RAG FTO-aware 解码 | 写 RAG 检索 + 负样本对比解码 | Qdrant 中真实 ChEMBL MMP pairs index + SureChEMBL Markush 展开样本 |
| EvoMol-RL 端到端 | 把 HVI/UCB/ParetoArchive 接入 generator.generate | 真实多目标 oracle 反馈（Boltz-2 + ADMET ML 模型） |
| CReM-3D DiffDock-L 实时打分 | 接入 DiffDock-L 推理 | DiffDock-L 权重（~1 GB）+ GPU |
| Boltz-2 ADMET 真实预测 | 写 oracle wrapper 真实推理代码 | Boltz-2 权重 + GPU |
| GNINA / DiffDock-L docking | 写 docking 调度 | GNINA 二进制 + DiffDock-L 权重 |
| OpenFE FEP | 写 FEP 调度 | OpenFE 包 + HPC（8×A100，多小时单算） |
| GPU4PySCF DFT L4 | 写 L4 oracle | GPU4PySCF + PSCF/ORCA + GPU |
| SureChEMBL / USPTO / Reaxys / Google Patents 真接入 | 写真实 httpx 客户端 + Markush 解析 | API 凭据（部分免费、部分付费）+ 可能的 BigQuery 账户 |
| AiZynthFinder MCTS | 接入 aizynthfinder 包 | aizynthfinder 安装包 + USPTO 反应模板（公开） |
| RSGPT / UAlign 真实模型 | 写 transformer 推理代码 | 训练好的权重（RSGPT 论文未公开权重需自训）+ GPU |
| Enamine REAL 49B Faiss 索引 | 写 Faiss IVF-PQ 索引构建 | REAL Space 数据集（需 Enamine 授权下载）+ 大磁盘（~500GB） |
| ESM-2 真实集成到 pocket encoder | 写集成代码 | ESM-2 权重已在 `models/esm2/esm2_t33_650M_UR50D.pt` ✓ 但需用户确认使用 |
| Sigstore Fulcio/Rekor 真签名 | 写 OIDC + Fulcio 流程 | Sigstore OIDC 凭据 + 网络可访问 Rekor 公共日志（公开免费但需出网） |
| Vault | 写 hvac 集成 | Vault server 部署 |
| OpenTelemetry trace 端到端 | 写 OTel SDK 集成 | OTLP collector 部署（Jaeger/Tempo） |
| MOSES / GuacaMol / PMO benchmark 实跑 | 写 benchmark runner + 调用现有 generator | benchmark 数据集（公开）+ 生成器权重 + GPU |
| KRAS G12C pilot 真实跑通 | 串接 NL→CIG→generate→validate→FTO→retrosyn→critic | 所有上述生成器+oracle+retrosyn runner 真实可用为前提 |

### 6.3 Claude 不能完成

| 不能做的事 | 原因 |
|---|---|
| 自行训练大规模 HUMU foundation model | 需 GPU 集群（设想 4×H200 141GB HBM3e）和数周到数月时间；Claude 无算力 |
| 自行训练 8 类生成器（HFM/FragFM/LaMGen/iCLM/MMPT/EvoMol/CReM/UAS）大模型权重 | 同上，每个生成器都需要数千 GPU 小时 |
| 自行训练 Boltz-2 / DiffDock-L / GNINA / ADMET-AI / Chemprop | 同上 |
| 自行采购/下载 Enamine REAL Space 49B 数据集 | 需 Enamine 授权 + 商业协议 |
| 自行订阅 Reaxys / SureChEMBL Premium / PatSnap | 需付费订阅 |
| 自行运行真实湿实验室（XDL/SiLA2 → 真实合成机器人） | 需物理硬件 |
| 自行真实跑通 KRAS G12C pilot 输出真实 Pareto 前沿 | 依赖上述资源全部到位 |
| 自行做 GxP / 21 CFR Part 11 合规审计认证 | 需第三方审计机构 + 法务团队 |
| 自行获取 OpenFE 商业级 FEP 校准数据 | 需领域专家 + 实验数据 |
| 自行获取 Mirati / Amgen 等公司真实专利全文 + 法务级 claim 解析 | 需法律团队 + 付费数据库 |
| 自行做 K8s 真实多节点 GPU 部署验证 | 需物理或云上的 H100/A100 集群 |
| 自行决定项目商业战略 / 商业化路线（Multi-tenant 等） | 设想文档第 3 阶段，超出技术范围 |

### 6.4 Claude 的能力误区（容易被误期望）

- **Claude 不能保证生成代码"无 bug"**：任何超过 50 行的新代码都需要用户在 CI / 真实环境验证。
- **Claude 不能预测大模型训练效果**：可写训练脚本，不能保证收敛到设想的 SOTA 指标。
- **Claude 不应被授权直接对外部服务（云、SDK 凭据、生产数据库）做修改**：所有破坏性操作必须用户审批。
- **Claude 的代码改动必须经用户确认才能合并**：本对话仅产出方案/代码，不应直接 commit/push（CLAUDE.md 已规定）。

---

## 7. 当前不能声称完成的事项

- 不能声称 CoreArchitecture v2 已完整实现。
- 不能声称 JMCG 联合分布 `P(m,r,p|T,c)` 已训练完成（FTO loss 未接入，性质轮廓 p 未进入闭环）。
- 不能声称 HUMU foundation model 已训练完成（当前 206,375 参数 baseline，远小于设想规模）。
- 不能声称 8 类生成器中任何一个有真实可发布的模型权重。
- 不能声称任何 L1–L4 oracle 真实运行过（仅 L0 RDKit 真实）。
- 不能声称 FTO 真实接入 SureChEMBL / USPTO / Reaxys / Google Patents。
- 不能声称 Patent Dead Zone 真实写回 HUMU 训练或生成器采样约束。
- 不能声称 retrosyn 真实跑通任何路径规划（3 个 wrapper 均 thin runner）。
- 不能声称 XDL 2.0 与官方 ChemputerXDL 兼容。
- 不能声称 SiLA2 / Wetlab 已对接任何真实设备。
- 不能声称任何 gRPC 服务可被外部 protobuf 客户端调用。
- 不能声称 K8s 全栈已部署（无 Deployment manifest）。
- 不能声称 Helm chart 可 `helm install`（templates 缺失）。
- 不能声称 Sigstore / Rekor 真实签名链已打通（永远 SHA256 fallback）。
- 不能声称 OpenTelemetry trace 端到端传播（仅 graph_builder 局部）。
- 不能声称 KRAS G12C pilot 已端到端跑通（占位测试）。
- 不能声称 Audit E2E 已通过（占位测试）。
- 不能声称 MOSES / GuacaMol / PMO / CrossDocked / DUD-E 任何 benchmark 有真实结果。
- 不能声称 Critic 100 规则在生产 pipeline 中被消费（critic-svc 与 critic_agent 互不连通）。
- 不能声称 NATS JetStream 已接入（实际是 Redis）。

---

## 8. 当前可以声称完成的事项

- 工程目录、协议层骨架、JSON Schema、Pydantic 类型系统、Lorentz manifold 数学算子、CIG 数据结构、SSP / XDL 编译器内部实现：基本完成且可读。
- DKI 5 大组件（Postgres / Neo4j / Qdrant / MinIO / Redis）已实际启动并通过 12 项集成测试。
- MoleculeForge 已从 Milvus / NATS 替换为 Qdrant / Redis。
- Provenance production store 已能将 graph / event / object 写入真实 DKI 并 readback（`test_dki_provenance.py` 通过）。
- Patent indexing pipeline 已能将真实输入文件中的 patent records 写入 Qdrant `patents_embedding` 并搜索回 evidence（`test_dki_patent_indexing.py` 通过）。
- HUMU 三塔 baseline（mol / pocket / route）联合对比损失训练真实跑过 17 epoch，产出 ~20.6 万参数 checkpoint。
- L0 RDKit 真实可用（QED / SA / Lipinski / 描述符）。
- KRAS / Audit E2E preflight 严格化：缺真实资源时明确列出阻塞项，**不会用空流程或假环境变量伪造 E2E 通过**。
- `tests/unit` 231 项通过、`tests/integration` 24 项 cic 通过、`tests/test_mvp_pipeline` 9 项通过；预测 API 真实可访问（FastAPI 8000 端口 + `/v1/predict` 真实调用 RDKit 描述符 + 启发式 ADMET 公式）。

---

## 9. 路线图建议（按依赖关系排序）

### 阶段 A — 工程债务清扫（Claude 独立可做，~1 周）
1. 修 unit collection error（`test_ssp_compiler.py:16`）。
2. 修 benchmark 文件命名 → pyproject.toml 加 `python_files`。
3. 修 schema/proto/crg 字段一致性。
4. 修 K8s netpol 端口；修 Dockerfile.oracle GNINA_SHA256 占位。
5. 修 anti_degradation 3 项失败。
6. 修 MoleculeRepository 空壳。
7. 整合 3 套 NL 解析；统一 TAR 与 generator-router-svc 路由器。
8. Generator Coordinator Agent 5 策略名 → 对齐 TAR 8 真名。
9. 生成 buf pb2 + 22 服务注册 servicer。
10. 写 22 服务 K8s Deployment manifest + docker-compose 余下 19 服务编排。
11. 补全 Helm subchart templates。

### 阶段 B — 算法补全（Claude 独立可做，~2 周）
1. 把 iCLM / UAS / CReM / FragFM 孤立算法模块接入 generator.generate。
2. SRBAgent.process 接 compile_ssp。
3. 把 ValidationAgent L1–L3 的 `return True` 替换为真实 `predict_with_uncertainty` + 不确定度阈值。
4. Patent Dead Zone 改用 Lorentz 距离 + 批量矩阵化。
5. HUMU 加入可学习曲率（nn.Parameter）。
6. PAINS 过滤接入 L0 oracle。
7. `mf-eval` 补全：distortion / cliff_analysis / hv_evaluator。

### 阶段 C — 真实模型集成（需用户提供资源）
1. 接入 ESM-2 到 pocket encoder（权重已在仓库）。
2. 接入 AiZynthFinder（公开包 + USPTO 模板）。
3. 接入 chemprop ADMET 真实模型。
4. 接入 sigstore（OIDC token 可用时）。
5. 接入 OpenTelemetry collector。
6. 真实跑 MOSES / GuacaMol / PMO（在算法补全完成后，对所有可生成的 generator）。

### 阶段 D — 需要算力和数据的工作
1. HFM-3D 真实预训练（ChEMBL 34 + 4×H200）。
2. FragFM SA-aware DFM 真实预训练。
3. HUMU foundation model 大规模联合预训练（含 FTO loss 接入）。
4. KRAS G12C pilot 端到端真实跑通。
5. SureChEMBL / USPTO / Reaxys 真实接入（数据 API 与法务确认后）。
6. Enamine REAL 49B Faiss 索引（数据获取后）。

### 阶段 E — 商业/合规（超出 Claude 范围）
1. GxP / 21 CFR Part 11 审计认证。
2. Multi-tenant K8s 部署。
3. 湿实验室设备对接（XDL / SiLA2）。
4. 商业化与定价。

---

*文档版本：v3（替换 v2 的 2026-05-19 05:40 版）*
*基于 6 路只读 subagent 勘察 + pyproject/uv.lock 审查 + pytest 实跑统计*
*所有差距判断均给出 file:line 证据，未给出证据的不写入*
