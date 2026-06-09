# CoreArchitecture v2 补齐工作量与两人任务划分

## 0. 元信息

- 来源真值（不可与之冲突）：`docs/architecture/current-implementation-vs-corearchitecture-v2.md`（2026-06-03，709 行，第 16 节列 18 项未完成事项 + 逐层偏差）。
- 治理规则：`docs/todo/2026-06-03-corearchitecture-v2-continuation-governance-plan.md`（一次只推进一个 gate、改代码前先定 scope、未授权不跑测试、保护他人改动）。
- 本文性质：工作划分规划。不修改业务代码，不跑测试。所有"当前证据/剩余 gate"均引自上述对照文档，未新增臆测。
- 冻结边界（对照文档 §14.4）：HUMU 预训练 pipeline、配置、loss、mol/pocket/route baseline 编码器、checkpoint 续训路径**保持现状**，不纳入本次两人补齐范围。
- 选型边界（对照文档 §14.1）：DKI 采用 Qdrant/Redis，不回迁 Milvus/NATS。NATS JetStream（§16 第 12 项）不补。

---

## 1. 结论速览

把"可实现但未实现"的事项按可执行主体分四类：

| 类别 | 含义 | 项数 | 谁主导 |
|---|---|---|---|
| **A 类：AI 可直接完成** | 本地可写、可本地验证，不依赖外部生产资源/凭据/集群 | 6 | AI Coding 主写，人工 review + 授权跑测试 |
| **B 类：AI 编码 + 人工投放资源（协作）** | AI 写模型/训练脚本/适配层，人工提供数据/算力/训练产物或外部服务 | 6 | 双人协作，AI 先交可运行骨架，人工补资源 |
| **C 类：纯人工投放** | 适配层 AI 已写完，只缺真实凭据/artifact/集群/官方数据 | 6 | 人工为主，AI 仅按报错微调 |
| **D 类：本阶段冻结/不补** | HUMU 等变三塔、NATS 回迁、MMPT 专利 RAG、SiLA2 湿实验 | 4 | 不做 |

> 范围调整（2026-06-03）：按用户决定，**专利**（MMPT-RAG 专利 RAG 检索，§16-8）与**湿实验**（SRB→SiLA2 硬件闭环，§16-6）不再补齐，移入 D 类。MMPT-RAG 保留本地内置 MMP 变换路径，SRB 保留结构化 SSP/XDL/SiLA2 plan 输出，二者均不再做生产升级。

核心判断：**当前缺的大多不是"可写的 wrapper"，而是真实生产 artifact / runner command / 凭据 / 外部 DKI / 集群环境 / 联合训练证据**（对照文档 §17 第 633 行原话）。因此两人能在本地"无外部资源"前提下推进的，集中在 A 类与 B 类的编码侧；C 类必须人工先投放资源，AI 才能闭环验收。

对齐边界（关键）：两人完成 A+B+C 后，设想的工程闭环、服务化、评估、审计、端到端 pilot 均可对齐（HUMU 等变三塔、DKI 回迁、专利、湿实验四项已豁免）。**唯一不由工程能力决定的是 W8（JMCG 联合采样）**——它拆为 W8-E 工程验收（两人范围终点，可对齐）与 W8-R 研究验收（联合训练质量，单列里程碑，非两人范围）。详见 §2.2 的 W8 边界说明。在 W8-R 拿到证据前，工程对齐 ≠ JMCG 研究完成。

---

## 2. 补齐项总清单（逐项三类标注）

> 编号 `Wn` = work item。"剩余 gate"列与对照文档 §16 表一致。

### 2.1 A 类 —— AI 可直接完成（本地可写可测，无需外部生产资源）

> 6 项：W1–W6。（原 W7「MMPT Seq2Seq decoder」已随专利 RAG 移入 D 类。）

| ID | 事项 | 涉及层 | 当前证据（已存在） | 要补的本地代码 | 依赖 |
|---|---|---|---|---|---|
| **W1** | CRG 最终态合并/读回 | MARB（§16-11，CRG brief §8.2 gap#2） | `final_state["crg"]` 仅含 orchestrator stage belief；各 agent 写入 shared `GraphRepository` 的 belief 不会在 provenance 记录前自动合并 | 在 `services/orchestrator-svc/.../main.py` 的 `_record_workflow_provenance()` 前，按 `run_id` 调 `GraphRepository.get_run_crg()` 读回 agent belief 并合并进 `final_state["crg"]`，再写 provenance metadata | 无（接口已存在） |
| **W2** | pocket/intent HUMU embedding producer | JMCG（§16-1） | property/intent/pocket records 已产出；property 保持 non-steering，intent/pocket 只能在有可靠、finite 且满足 Lorentz hyperboloid 方程的 129 维 Lorentz full-coordinate 来源时变为 steering-capable；humu-encoder-svc 已支持 mol/route/pocket 编码，pocket encoder 已存在 | 让 orchestrator 在结构化 pocket geometry 存在且 `HUMU_ENCODER_TARGET` 可用时调用 `humu-encoder-svc` 产出 pocket HUMU embedding；intent record 仅接受已有合法 129 维 Lorentz full-coordinate axis，不能直接把 128 维 HCIV 向量写入 `humu_embedding`；HCIV→Lorentz 转换另列后续 gate | 本地 HUMU checkpoint（已存在 `checkpoints/humu/best_model.pt`） |
| **W3** | PCBO 参考 candidate provider / oracle evaluator | Oracle/PCBO（§16-15） | service/runner/FastAPI/scheduler/部署 wiring 全有，但只暴露 `module:attribute` / command 注入点 | 写一组接到现有 L0-L3 oracle 的具体 candidate provider 与 oracle evaluator 参考实现（callable path），让 `pareto-bo-svc` 默认能跑本地闭环 | W partial: 需 oracle service 本地可达 |
| **W4** | 跑通并修复 6.3 已写未跑的测试 | 全链路 | 6.3 多个 gate 写了 focused test spec 但按 no-test 规则**未执行**（contract/semantics/property/pocket/intent producer 等） | 经用户授权后 `uv run pytest tests/unit -q` 等，修复任何因新代码暴露的失败 | **需用户显式授权跑测试** |
| **W5** | benchmark harness 非 skip 路径补完 | 评估（§16-17 的代码侧） | MOSES/GuacaMol/PMO/CrossDocked 已资源门控，临时资源 smoke 可跑 | 补任何缺失的资源门控分支与断言，使官方数据就位后零代码改动即可跑 | 官方数据由 C 类 H8 投放 |
| **W6** | TAR ProxylessNAS 训练 runner 脚本 | AMGE（§16-9） | scheduler / architecture gate / reward-cost update / `RunProxylessSearch` gRPC 全有；Owner A 已新增 `python -m generator_router_svc.tar_proxyless_runner` 本地 command target | 本地 AI 代码侧已完成；剩余是把真实 reward 数据集接入生产训练/部署配置并验收 | 训练数据集由 B 类人工提供；集群发布仍需验证 |

### 2.2 B 类 —— AI 编码 + 人工投放资源（协作）

| ID | 事项 | 涉及层 | AI 负责（本地可交付） | 人工负责（投放） | 剩余 gate |
|---|---|---|---|---|---|
| **W8** | JMCG `(m,r,p)` 联合采样模型 | JMCG（§16-1） | W8-E 本地工程骨架已新增 `JMCGEngineeringSampler`，可从 HFM candidate + route/property/pocket/intent feedback 构造 JSON-serializable joint sample，并校验 finite 且满足 Lorentz hyperboloid 方程的 129 维 HUMU embedding；下一步是训练脚本/真实模型 | 联合训练数据、算力、训练产物 artifact、端到端验收 | W8-E 骨架已落地；W8-R 真实联合采样模型质量 + 生产验证仍缺 |
| **W9** | HFM-3D 生产神经几何 decoder | AMGE（§16-5） | Owner A 已新增本地 neural geometry decoder 训练/export/runner 路径：`mf_generators.hfm_3d.decoder.neural_geometry_decoder` 可从 SDF-backed HFM decoder artifact 训练 tiny MLP artifact，并可作为 `python -m ... --artifact` stdin/stdout command target 输出现有 HFM molecular decoder JSON schema；`train_geometry_decoder.py` CLI 已存在；接口 `molecular_decoder` / `HFM_MOLECULAR_DECODER_COMMAND` 继续复用 | 训练真实 decoder artifact、投放 `HFM_MOLECULAR_DECODER_COMMAND` 或生产 artifact 值、集群验收、几何质量 benchmark | 本地工程路径已落地；默认生产路径仍未投放真实 neural geometry decoder |
| **W10** | Enc_intent HCIV 生产 checkpoint | CIC/HCIV（§16-2,3） | Owner A 已补本地 supervised train/export 工程路径：`cig_compiler_svc.domain.hciv_training` 可加载 `cig + target_hciv` JSON/JSONL 数据、训练现有 `HCIVEncoder` 并导出兼容 `HCIV_CHECKPOINT_PATH` 的 checkpoint；`services/cig-compiler-svc/train_hciv_encoder.py` CLI 已存在 | 真实训练数据、真实训练运行、投放 `HCIV_CHECKPOINT_PATH`、集群验收、下游质量验证 | 本地工程路径已落地；缺训练好的 production-quality checkpoint |
| **W11** | FragFM 共享 HUMU 条件空间生产闭环 | AMGE（§16-6） | Owner A 已新增本地质量门：FragFM training CLI 会校验并保留 valid 129 维 Lorentz full-coordinate `humu_embedding` 到 vocab artifact 和 manifest；新增 `mf_generators.fragfm.quality` 可生成 vocabulary/checkpoint/rate-matrix 的 HUMU coverage/loadability JSON report；`checkpoints/fragfm_humu_5k/` 现为 5000-record HUMU coverage 1.0 的 strict-local candidate，Docker Compose/Kubernetes/Helm 默认值已接到该 candidate；旧 `checkpoints/fragfm` 仍只作 coverage=0 smoke/runtime 证据 | 生产质量验收、集群发布验证 | 本地工程质量门、5k local candidate、deployment-default hardening 已落地；production-quality 训练配置、正式阈值、benchmark/集群验证仍缺 |
| **W12** | CReM-pharm-3D 真实 scorer 闭环 | AMGE（§16-7） | 本地真实 scorer runner 闭环已完成：GNINA docking 通过 `DOCK_ORACLE_COMMAND` + `DOCK_ORACLE_RECEPTOR_PDB`，pharmacophore 通过 6OIM/MOV reference SDF + RDKit shape/color scorer，HUMU 通过 `HUMU_CHECKPOINT_PATH` wrapper；DiffDock-L 按用户决定移出 W12 本地 gate | 集群验收 | H10 集群发布验证 |
| **W13** | Cross-Paradigm KD 生产蒸馏 | AMGE（§16-10） | Owner A 已新增 teacher embedding artifact export/report 本地质量门：`mf_core.routing.kd_artifacts` 可从 JSON/JSONL teacher records 导出 canonical `cross_paradigm_teacher_embeddings.v1` artifact，并报告 embedding count/dim/finite/expected-dim/min-count readiness；KD layer / HypSeek app / iCLM update runner / 各 generator CLI KD loss 入口已存在 | 生产 teacher 服务、真实蒸馏训练、真实集群发布验证 | 本地 artifact handoff gate 已落地；仍缺 production teacher source、真实 teacher embeddings、蒸馏训练质量、benchmark/集群验证 |

#### W8 边界说明：工程验收 vs 研究验收（关键）

W8（JMCG 联合采样）是最终设想的第〇层、最顶层目标，也是两人协作完成后**唯一可能仍对不齐设想的项**。它的特殊性在于：能否达成不由工程能力决定，而由联合采样模型训练能否拿到有效结果决定（研究风险）。因此把 W8 拆成两道独立验收线，两人的工程范围终点 = 工程验收，研究质量单列：

| 验收线 | 验收对象 | 判定标准 | 归属 | 是否阻塞"工程对齐" |
|---|---|---|---|---|
| **W8-E 工程验收** | 契约化 HUMU steering + 可运行联合采样骨架 | (1) `jmcg_feedback` envelope 的 molecule/route/property/pocket/intent 五类 record 全部可被 HFM 按契约 C1 消费；(2) W2 的 pocket/intent steering-capable embedding 生效、不回归 route；(3) 联合采样器 `p(m,r,p\|z,T,c)` 骨架本地可运行（接口、前向、采样路径打通），喂占位/小规模数据即可产出结构合法的 `(m,r,p)` 三元组；(4) 授权后相关 unit 测试通过 | 两人（甲主） | **是 —— 这是两人工程范围的终点** |
| **W8-R 研究验收** | 联合采样质量达标 | 联合训练后 `(m,r,p)` 在共享 HUMU 上的联合一致性、mol-route 一致性、性质命中等指标达到设想阈值；需真实联合训练数据、算力、训练产物 | 后续研究里程碑（非两人工程范围） | 否 —— 单列，不阻塞工程对齐声明 |

判定规则：
- 两人交付 **W8-E** 即视为"工程范围内 JMCG 已对齐"，可与设想其余工程项一并声明对齐。
- **W8-R** 是研究里程碑，依赖资源投放与训练效果，存在训不出预期质量的可能；在 W8-R 拿到证据前，**不得把 JMCG 描述为"完成"**（对照文档 §17 第 633/693 行红线）。
- 文档/对外口径必须区分二者：工程对齐 ≠ JMCG 研究完成。

### 2.3 C 类 —— 纯人工投放（适配层已写完，只缺资源；附具体做法）

| ID | 事项 | §16 | 具体做法（人工执行） |
|---|---|---|---|
| **H1** | DKI 环境部署 | 7,11 | 起 `infra/docker/docker-compose.dki.yaml`（Postgres/Neo4j/Qdrant/MinIO）+ Redis/Feast；投放 env：`NEO4J_URI/USER/PASSWORD`、`MINIO_ENDPOINT_URL/ACCESS_KEY/SECRET_KEY/BUCKET`、`QDRANT_HOST` 或 `QDRANT_URL`、`REDIS_HOST` 或 `REDIS_URL`、`TEST_DATABASE_URL`、`PROVENANCE_DATABASE_URL`、`FEAST_REPO_PATH`。投放后 `tests/integration` 10 项 skip 应转 pass |
| **H2** | Sigstore/Rekor 生产审计链 | 13 | 部署 Fulcio/Rekor 或采购托管；投放 `SIGSTORE_SIGN_COMMAND`、`SIGSTORE_VERIFY_COMMAND`、`SIGSTORE_IDENTITY_TOKEN`、`SIGSTORE_EXPECTED_IDENTITY`、`SIGSTORE_REKOR_URL`。验收：`env RUN_AUDIT_E2E=1 uv run pytest tests/e2e/test_audit_completeness.py` |
| **H4** | L4 GPU4PySCF/ORCA 量子校正 | 14 | 安装 GPU4PySCF / ORCA；投放 `L4_QUANTUM_ORACLE_COMMAND` 或 `L4_GPU4PYSCF_COMMAND` / `L4_ORCA_COMMAND` 及 artifact。当前 `find_spec` 探测 gpu4pyscf/pyscf=False、`command -v orca` 无输出 |
| **H5** | L1-L3 Oracle runner 真实接入 | 5 | 本地 command wrapper gate 已完成：`.env` 已投放 `DOCK_ORACLE_COMMAND`/`BOLTZ2_ORACLE_COMMAND`/`FEP_ORACLE_COMMAND`/`ADMET_ORACLE_COMMAND`；focused wrapper 与 service command 合同回归通过；OpenADMET 主预测 smoke 通过；Boltz GPU affinity smoke 通过；FEP 已投放 TYK2 OpenFE transformation/result registry，wrapper 与 service background job smoke 通过。真实 OpenFE 长程 MD 和 KRAS covalent-FEP full pilot 仍不作为本次 H5 完成证据 |
| **H6** | RetroSyn 多引擎真实 runner | 6 | 本地 command runner 已投放 RAscore/RSGPT/UAlign/AiZynth，并完成真实 service command path smoke；剩余集群发布验证和 KRAS full pilot 归 H10/H11 |
| **H8** | 官方 benchmark 数据 | 17 | 投放 `MOSES_REFERENCE_SMILES_PATH`、`PMO_SCORE_TABLE_PATH`、`CROSSDOCKED_BENCHMARK_JSONL`、GuacaMol/PMO 所需 `HFM_CHECKPOINT_PATH`+`HFM_DECODER_PATH`，设定正式阈值 |
| **H9** | CIG LLM/SRM parser 接入 | 2 | 提供真实 LLM/SRM：`CIG_SEMANTIC_PARSER_URI`（python/http）或 `CIG_SEMANTIC_PARSER_COMMAND`，及 `CIG_REFINEMENT_COMMAND`。Python/HTTP/command 三种 adapter 已写好 |
| **H10** | 集群发布验证 | 8,11 | K8s/Helm 实际发布所有 service，逐个验证启动、readiness/liveness、artifact 挂载、ConfigMap/Secret。manifest 已就绪 |
| **H11** | KRAS G12C full pilot | 18 | 依赖 H1+H2+H5+H6 + service ready；投放 `CRITIC_AGENT_READY`、`ORCHESTRATOR_E2E_READY`、`PROVENANCE_STORE_MODE=production_real` 等，`env RUN_KRAS_G12C_E2E=1 KRAS_E2E_SCOPE=full` 跑通 |

> C 类共 6 个独立资源域（H1/H2/H4/H8/H9 相对独立；H5/H6 属 runner 类；H10/H11 是集成验收）。原 H7（SiLA2 湿实验）已移入 D 类。

### 2.4 D 类 —— 本阶段冻结/不补

| ID | 事项 | 原因 |
|---|---|---|
| D1 | SE(3)/E(3) HUMU 三塔等变编码器、双曲 TreeLSTM、Lorentz-equivariant attention（§16-4） | 对照文档 §14.4：属"后续 HUMU 升级"，不阻塞当前补齐；HUMU 预训练冻结 |
| D2 | NATS JetStream 回迁（§16-12） | 已确认 Redis 替代 |
| D3 | MMPT-RAG 专利 RAG 检索 + Seq2Seq decoder 生产升级（§16-8） | 用户决定不做专利部分。MMPT-RAG 保留本地内置 MMP 变换 + contrastive ranking 路径，`MMPT_PATENT_RAG_COMMAND` / `MMPT_SEQ2SEQ_DECODER_COMMAND` 外接点保留但不投放 |
| D4 | SRB → SiLA2 真实硬件湿实验闭环（§16-6 硬件侧） | 用户决定不做湿实验。SRB 保留结构化 SSP/XDL/SiLA2 plan 输出，`SILA2_PLAN_COMMAND` 外接点保留但不接真实硬件 |

---

## 3. 依赖关系图

```text
                         ┌──────────────────────────────────────────┐
A 类（本地可写，互相基本独立）：                                       │
   W1 CRG 合并 ─────────────────────────────► 受益于 H1(真实 Neo4j 验收)│
   W2 pocket/intent embedding ──► W8 JMCG 联合采样（B）                 │
   W3 PCBO 参考 provider ──────► 受益于 H5(真实 oracle)                 │
   W4 跑测试 ◄── 依赖 W1/W2/W5/W6 任一落地后回归                        │
   W5 benchmark harness ───────► 数据就位 H8 ──► 才能出正式指标          │
   W6 TAR runner ──────────────► 数据/算力(B) ──► 训练闭环              │
                                                                       │
B 类（AI 骨架 → 人工投放）：                                            │
   W8 JMCG ◄── W2；需联合训练数据+算力                                  │
   W9 HFM decoder ─► 训练 artifact ─► H8 GuacaMol/PMO 正式指标          │
   W10 Enc_intent ─► 训练 checkpoint ─► H9/HCIV 生产链路               │
   W11 FragFM / W12 CReM / W13 KD ─► 质量验收 + H10 集群                │
                                                                       │
C 类（资源投放，决定能否最终验收）：                                     │
   H1 DKI ──┬─► W1 真实验收 / 集成测试转 pass                          │
            ├─► H11 KRAS full                                          │
   H2 Sigstore ─► audit E2E / H11                                      │
   H5 oracle + H6 retrosyn ─► H11                                      │
   H10 集群 ─► 全部生产验收的总闸                                       │
   H11 KRAS full = H1+H2+H5+H6 + service ready 的汇聚点                 │
                         └──────────────────────────────────────────┘
```

关键路径结论：
- **A 类彼此独立**，可并行起步，无外部依赖，是两人立即能动手的部分。
- **B 类**的 AI 骨架可立即写，但"闭环验收"被人工投放卡住。
- **C 类是最终验收总闸**，H10/H11 在最后；H1（DKI）解锁面最广，建议人工最先投放。

---

## 4. 两人分工候选方案

前提：两人背景相近、不分算法/工程专长；目标是对接面少、可并行、互不阻塞。下面给 2 个候选，附各自的对接点与优缺点，供你选定。

### 候选方案一：按"生成上游 / 验证下游"垂直切分（推荐）

以 orchestrator 主链路 `CIG→HCIV→generate→validate→retrosyn→supply→srb→critic→provenance` 为界，前半段归甲、后半段归乙，CRG 作为唯一共享面。

| | 甲（生成上游） | 乙（验证-供应-存证下游） |
|---|---|---|
| A 类 | W2、W6 | W1、W3、W5 |
| B 类 | W8、W9、W10、W11、W13 | W12 |
| C 类配合 | H9（LLM parser）、H8（生成 artifact 验收） | H1、H2、H4、H5、H6、H11 |
| 公共 | — | W4（跑测试）由乙统一执行，甲提供改动清单 |

- **对接点（仅 3 个，契约化）**：
  1. `generator_params` 透传契约：`generation_feedback` / `route_humu_feedback` / `jmcg_feedback`（envelope `moleculeforge.jmcg.feedback.v1`）。甲改 producer/consumer，乙改 orchestrator 派生侧，双方只约定字段，不互改对方文件。
  2. CRG belief 谓词表：`parsed_intent`/`compiled_cig`/`selected_generators`/`validation_status`/`retrosyn_routes`/`route_humu_embedding`/`supply_feasibility`/`ssp_compiled`/`critic_verdict`/`workflow_status`。新增谓词需双方在本文档登记。
  3. HUMU encoder 接口：W2/W8（甲）与 W3/W12（乙）都调 `humu-encoder-svc`，约定 entity_type 与 input_data schema。
- **优点**：对接面最小，主链路前后清晰；CRG 谓词表是天然的契约清单。
- **缺点**：甲的 B 类（W8-W13）偏重、偏研究，工作量大于乙；需在 C 类资源投放上让乙多分担以平衡。

### 候选方案二：按"AI 本地编码 / 人工资源投放与验收"并行切分

一人专注 A+B 的本地编码（写代码、本地验证），另一人专注 C 的资源投放与端到端验收。

| | 甲（编码） | 乙（资源+验收） |
|---|---|---|
| 主体 | W1-W13 全部本地代码骨架 | H1-H11 全部资源投放 |
| 配合 | 为乙的每次投放提供"投放后应跑的命令 + 期望证据" | 投放后回报报错，甲按报错微调 wrapper |

- **对接点**：每个资源域一张"投放卡"（env 清单 + 验收命令 + 期望退出码/指标），见 §5.4。
- **优点**：技能要求单一，甲只写代码、乙只配环境，互不读对方代码。
- **缺点**：甲串行承担全部编码，是瓶颈；乙在甲交付前空转；不利于并行提速。

### 推荐

选 **候选方案一**，并把 C 类资源投放按 §3 关键路径在两人间均摊（乙主导 DKI/Sigstore/oracle，甲主导 LLM parser/生成 artifact）。理由：两人都在写代码，工作量更均衡、可真正并行；对接收敛到 3 个契约面，符合"对接方便"。

---

## 5. 对接规范（无论选哪个方案都执行）

### 5.1 工作流（遵循 governance plan）

每个 work item 按：定 scope（列出允许改的文件）→ 最小改动 → 更新对照文档对应层 → 经授权跑相关测试 → 在本文档"执行日志"登记。一次只推进一个 gate。

### 5.2 防冲突规则

- 当前分支 `feature/corearchitecture-v2-completion` 工作树脏（governance plan 记录 230 项）。**不回滚、不重命名、不覆盖他人改动**。
- 两人各自只改自己 work item 的 scope 文件；跨 scope 文件（如 `orchestrator-svc/main.py`、`generator_coord/agent.py`）改动前在本文档登记意图，避免双写。
- `/workspace/SemMol`、`/workspace/Projects` 仅可作为只读上下文：允许读取和复制参考信息，但不得写入、执行、修改或作为输出目录。

### 5.3 验收证据标准（governance plan §1.4）

可接受：文件路径+代码位置、命令+退出码、（授权后的）测试输出、manifest wiring、明确标注的外部缺口。不可接受："应该能跑"、把 mock 当生产、复用旧测试输出、把架构文档承诺当实现。

### 5.4 投放卡模板（C 类每项一张）

```text
资源域: <Hn 名称>
负责人: <甲/乙>
env 清单: <KEY=...>
前置: <依赖的其他 Hn>
验收命令: <uv run pytest ... 或 服务启动命令>
期望证据: <退出码 / pass 数 / 指标阈值>
当前状态: 未投放 / 投放中 / 已验收
```

### 5.5 三个契约面（方案一）登记表（初始）

| 契约 | 字段/谓词 | 当前持有方 | 变更需双签 |
|---|---|---|---|
| generator_params | `generation_feedback`,`route_humu_feedback`,`jmcg_feedback.records[*]`(kind/source/subject/humu_embedding/weight/polarity/confidence/evidence_ids) | 甲 producer / 乙 orchestrator | 是 |
| CRG 谓词 | 见 §4 候选一对接点 2 | 谁写 belief 谁持有 | 新增谓词需登记 |
| HUMU encoder | entity_type∈{mol,pocket,route}, input_data schema | humu-encoder-svc | 是 |

---

## 6. 里程碑顺序建议

| 阶段 | 内容 | 卡点 |
|---|---|---|
| M0（即刻并行） | A 类 W1/W2/W3/W5 编码；乙投放 H1（DKI）| 无外部依赖 |
| M1 | W4 跑通全量测试回归（授权后）；W1 借 H1 做真实 Neo4j 验收 | 需测试授权 + H1 |
| M2 | B 类骨架 W9/W10/W6/W8 落本地可运行（W8 达 **W8-E 工程验收**）；人工备训练数据/算力 | 数据/算力 |
| M3 | C 类 H2/H4/H5/H6/H9 投放；对应 service 真实验收 | 凭据/artifact |
| M4 | H8 官方 benchmark 数据 → W5 出正式指标；W11/W12/W13 质量验收 | 官方数据 |
| M5 | H10 集群发布验证 → H11 KRAS G12C full pilot 跑通 | 集群+前置全绿 |
| M6（研究里程碑，非两人工程范围） | **W8-R 研究验收**：联合采样质量达标 | 联合训练数据+算力+训练效果 |

> M0-M5 全绿 = 工程范围与设想对齐（四项豁免除外）。M6（W8-R）单列，不阻塞工程对齐声明；未拿到 W8-R 证据前不得声称 JMCG 完成。

> 7 天后本文档若仍在用，需复核：A 类是否已落地、C 类各投放卡状态、对照文档对应层是否同步更新。

---

## 7. 执行日志

（每完成一个 work item 在此追加：日期 / ID / 改动文件 / 验证 / 剩余 gate）

- 2026-06-03：创建本任务划分文档。未改业务代码，未跑测试。
- 2026-06-03：按用户决定移除专利（W7→D3）与湿实验（H7→D4）；为 W8 增设 W8-E 工程验收 / W8-R 研究验收边界，并同步 §1 对齐边界与 §6 里程碑（新增 M6）。
- 2026-06-03（乙）：

  **W1 验收**：`services/orchestrator-svc/src/orchestrator_svc/main.py` 的 `_merge_agent_beliefs_into_crg` 函数已存在（行 812-861），在 `_record_workflow_provenance()` 内第一步调用（行 869）。`build_shared_crg_repository_from_env` 已在 `libs/mf-core/src/mf_core/db/repositories/__init__.py` 导出。`tests/unit/test_graph_repo.py` 已有 3 条专项测试（行 280-427）：合并行为、去重、无仓库时降级。W1 本地代码已完整，剩余 gate：H1（DKI Neo4j）就位后方可做真实 Neo4j 验收。

  **W3 已实现**：
  - 新建 `pipelines/pareto_bo/src/pareto_bo/providers.py`：`TangentSpaceNoiseCandidateProvider`（纯本地，随机噪声扰动 observed embeddings）、`SmilesCandidateProvider`（SMILES 列表→HUMU/指纹 embedding 编码）、`LocalOracleEvaluator`（gRPC oracle 可用时调用 L0-L3，否则回退 embedding proxy 分数）。
  - 修改 `pipelines/pareto_bo/src/pareto_bo/service.py`：`_runtime_from_env` 添加 `default_factory` 参数，无 env var 时回退到默认实现，`_default_candidate_provider` / `_default_oracle_evaluator` 两个工厂函数。
  - 新增 4 条聚焦测试（`tests/unit/test_mf_eval.py` 行 588+）：provider 生成形状验证、env var 控制计数、oracle 回退 embedding proxy、from_env 默认可用。`python -m py_compile` 通过，`git diff --check` 通过。剩余 gate：真实 candidate provider/oracle evaluator runner 值与生产验收（C 类）仍未完成。

  **W5 已实现**：
  - 修改 `tests/benchmark/__init__.py`：添加 `_open_text(path)` 辅助函数，对 `.gz`/`.gzip` 后缀文件透明 gzip 解压；`read_smiles_file`、`read_scored_smiles_table`、`read_jsonl_records` 均改用 `_open_text`。官方 GZIP 格式数据就位后零代码改动即可跑。基础 18 项资源门控路径已验证（对照文档 §16 行 649）。剩余 gate：H8 官方数据投放与正式阈值。

  **W12 已实现（B 类 AI 编码侧）**：
  - 新增 3 条聚焦测试（`tests/unit/test_phase_b_generators.py` 行 207+）：`test_crem_pharmacophore_scorer_ranks_molecules_by_score`（mock pharmacophore scorer，验证按 pharmacophore_score 倒序排列）、`test_crem_humu_scorer_ranks_by_alignment_and_stores_embedding`（mock HUMU scorer，验证 humu_embedding 写入 Molecule、按 humu_alignment_score 倒序排列）、`test_crem_dock_oracle_grpc_scorer_batch_calls_oracle_service`（mock gRPC stub，验证 DockOracleGrpcScorer.score_batch() 正确解析 OracleEvaluation）。`python -m py_compile` 通过，`git diff --check` 通过。未跑 pytest（遵守 no-test 规则）。2026-06-07 已补真实 scorer runner 闭环，剩余 gate：H10 集群发布验证。

- 2026-06-03（甲）：

  **W6 本地 runner 已实现**：
  - 新增 `services/generator-router-svc/src/generator_router_svc/tar_proxyless_runner.py`：可作为 `TAR_PROXYLESS_SEARCH_COMMAND="python -m generator_router_svc.tar_proxyless_runner"` 的本地命令目标，从 stdin 读取 reward-cost payload，复用 `ProxylessSearchScheduler`，输出 `rounds`、`architecture_probabilities`、`architecture_logits`、`generator_names` 和参数回显。
  - 修改 `tests/unit/test_task_router.py`：新增直接 runner、CLI subprocess、GeneratorRouterService 调用真实 runner command 的 3 条 focused 规格。
  - 验证：`python -m py_compile moleculeforge/services/generator-router-svc/src/generator_router_svc/tar_proxyless_runner.py moleculeforge/tests/unit/test_task_router.py` 通过；`git diff --check` 通过；`uv run python -m generator_router_svc.tar_proxyless_runner` 命令级 smoke 通过；2026-06-04 已补 W6 focused pytest 和 `tests/unit/test_task_router.py` 文件级 pytest（30 项）通过。
  - 剩余 gate：真实 reward 数据集、生产环境 `TAR_PROXYLESS_SEARCH_COMMAND` 值投放和集群发布验证。

- 2026-06-04（甲）：

  **W8-E 本地工程骨架已实现**：
  - 新增 `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/jmcg_sampler.py`：提供 `JMCGEngineeringSampler`、`JMCGContextRecord`、`JMCGJointSample` 和 `parse_jmcg_context()`，可从候选 molecule、route/property profile 与 `moleculeforge.jmcg.feedback.v1` 记录构造 `moleculeforge.jmcg.joint_sample.v1` JSON output。
  - 修改 `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/__init__.py` 导出 W8-E skeleton API。
  - 修改 `tests/unit/test_generators.py`：新增合法 joint sample、128 维 invalid embedding non-steering、context parser 兼容 3 条 focused 规格。
  - 验证：`python -m py_compile moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/jmcg_sampler.py moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/__init__.py moleculeforge/tests/unit/test_generators.py` 通过；`git diff --check` 通过；`uv run python - <<'PY' ... JMCGEngineeringSampler ... PY` 命令级 smoke 通过。未跑 pytest（遵守 no-test 规则，等待 W4/用户授权）。
  - 剩余 gate：W8-R 真实联合采样训练质量、联合训练数据/算力/artifact、端到端生产验证。

  **Embedding validation hardening 已实现**：
  - 新增 `libs/mf-core/src/mf_core/geometry/lorentz.py`：`normalize_lorentz_embedding()` 统一校验 finite、time coordinate、维度和 Lorentz hyperboloid 方程。
  - 修改 W2 producer、HFM feedback consumer 和 W8-E sampler 共用该校验；非法 129 维向量 fail closed，packed float32 `Molecule.humu_embedding` bytes 可被 W8-E 解码。
  - 验证：新增 4 条 focused 规格先 RED 后 GREEN；`uv run pytest tests/unit/test_generators.py tests/unit/test_service_artifact_status.py -q` 通过 273 项。

  **W9 本地 neural geometry decoder 工程路径已实现**：
  - 新增 `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/neural_geometry_decoder.py` 与 `decoder/__init__.py`：加载 SDF-backed decoder entries、训练固定 `max_atoms` 的 coordinate MLP、保存/加载 torch artifact、nearest-entry 选择 SMILES/atom types、输出 runner-compatible JSON。
  - 新增 `models/mf-generators/hfm_3d/train_geometry_decoder.py` CLI wrapper。
  - 修改 `HFM3DGenerator` 保留 decoder payload 自带的 `metadata.decoder_mode`，旧 payload 未声明时仍默认 `molecular_decoder`。
  - 验证：5 条新 W9 规格先 RED 后 GREEN；focused W9 + legacy decoder gate 6 项通过；`uv run pytest tests/unit/test_generators.py -q` 通过 65 项；`python -m py_compile` 与 `git diff --check` 通过。
  - 剩余 gate：真实 production-quality decoder artifact、生产 env/command 投放、集群发布和几何质量 benchmark。

  **W10 本地 Enc_intent checkpoint 训练/export 工程路径已实现**：
  - 新增 `services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_training.py`：加载 supervised `cig + target_hciv` JSON/JSONL 记录，训练现有 `HCIVEncoder`，导出 schema-wrapped torch checkpoint 和可选 manifest。
  - 新增 `services/cig-compiler-svc/train_hciv_encoder.py` CLI wrapper。
  - 修改 `HCIVEncoder` 增加可微 `forward_coordinates(cig)`，`encode()` 输出契约保持不变。
  - 验证：3 条新 W10 规格先 RED 后 GREEN；focused W10 gate 4 项通过；`uv run pytest tests/unit/test_cic_compiler.py -q` 通过 31 项；`python -m py_compile` 与 `git diff --check` 通过。
  - 剩余 gate：真实 supervised CIG/HCIV 训练数据、production-quality checkpoint、`HCIV_CHECKPOINT_PATH` 投放、集群验收和下游质量验证。

  **W11 FragFM shared HUMU 本地质量门已实现**：
  - 新增 `models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py`：提供 `build_quality_report()` 和 `python -m mf_generators.fragfm.quality` CLI，可报告 vocab HUMU coverage、invalid embedding、checkpoint loadability、rate-matrix loadability 和 pass/fail status。
  - 修改 `models/mf-generators/fragfm/train.py`：训练记录中的可选 `humu_embedding` 现在必须是 valid 129 维 Lorentz full-coordinate；合法 embedding 会保留进 `vocab.json`，manifest 会记录 `humu_embedding_count` 与 `humu_embedding_coverage`。
  - 验证：3 条新 W11 规格先 RED 后 GREEN；W11 focused 4 项通过；FragFM 子集 9 项通过；`python -m py_compile`、quality CLI smoke、`git diff --check` 通过。
  - 当前本地 `checkpoints/fragfm` quality CLI smoke 在 `--min-humu-coverage 0.0` 下输出 `pass 50 0 0.0 True True`，说明 artifact 可加载但 HUMU coverage=0，只能作为 runtime smoke。
  - 剩余 gate：真实 HUMU-labeled FragFM 训练数据、production-quality artifact、正式 coverage/benchmark 阈值和集群发布验证。

  **W13 Cross-Paradigm KD teacher embedding artifact gate 已实现**：
  - 新增 `libs/mf-core/src/mf_core/routing/kd_artifacts.py`：提供 `export_teacher_embeddings_artifact()`、`build_teacher_embeddings_report()` 和 `python -m mf_core.routing.kd_artifacts` CLI。
  - 可从 JSON/JSONL teacher records 的 `teacher_embedding` 字段导出 canonical `cross_paradigm_teacher_embeddings.v1` artifact，并检查 finite、consistent dimension、`expected_dim` 和 `min_embeddings`。
  - 验证：2 条新 W13 规格先 RED 后 GREEN；`uv run pytest tests/unit/test_cross_paradigm_kd.py -q` 通过 18 项；CLI smoke 输出 `pass 2 2 cross_paradigm_teacher_embeddings.v1`；`python -m py_compile`、`git diff --check` 通过。
  - 剩余 gate：真实 production teacher source / teacher embeddings、真实蒸馏训练、benchmark 质量证据和集群发布验证。

  **W9/W10/W11 阶段复验 hardening 已实现**：
  - W9 decoder source artifact `latent`、W10 supervised `target_hciv` 现在都复用 shared Lorentz validator。
  - W11 FragFM quality gate 现在严格要求 checkpoint `fragment_encoder.weight` 和 rate-matrix `base_rate` schema。
  - 验证：4 条新 hardening 规格先 RED 后 GREEN；相邻 focused pytest 13 项通过；`python -m py_compile`、`git diff --check` 通过；W11 strict CLI smoke 对当前本地 artifact 仍输出 `pass 50 0 0.0 True True`（coverage=0，仅 runtime smoke）。
  - 剩余 gate：真实 W9/W10/W11 production artifacts/data、benchmark 阈值和集群验证。

- 2026-06-04（乙）：

  **W1 真实 DKI 验收已完成**：
  - 改动文件：未改业务代码。
  - 资源状态：`.env` 中 DKI 必需 env 均已投放，`FEAST_REPO_PATH` 存在；Postgres/Neo4j/Qdrant/MinIO/Redis 端口可达。
  - 验证：`bash -lc 'set -a; source .env; set +a; unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"; uv run pytest tests/integration/test_dki_*.py -q'`。
  - 实际结果：exit code 0，10 passed，0 skipped；有 1 条 Qdrant client 1.18.0 与 server 1.12.4 minor version compatibility warning。
  - 验收结果：W1 真实 Neo4j/DKI gate 已验收；C1/C2/C3 无变更。

- 2026-06-05（H2）：

  **H2 Sigstore/Rekor 生产审计链验收已完成**：
  - 配置投放：`.env` 已投放 `SIGSTORE_SIGN_COMMAND`、`SIGSTORE_VERIFY_COMMAND`、`SIGSTORE_REKOR_URL`、`PROVENANCE_SVC_URL`；GitHub Actions self-hosted runner 运行时投放 `SIGSTORE_IDENTITY_TOKEN`、`SIGSTORE_EXPECTED_IDENTITY`、`SIGSTORE_E2E_READY=1`，未写入或泄露 token。
  - 真实链路：`H2 Audit Sigstore E2E` workflow run `27016836066`（commit `d54f536`）在 self-hosted runner 上启动 `production_real` provenance service，真实 `cosign sign-blob` 写入 Rekor bundle，`cosign verify-blob` 按 GitHub Actions expected identity 验证通过，日志输出 `sigstore_rekor_smoke=pass`。
  - 验收：`RUN_AUDIT_E2E=1 PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/e2e/test_audit_completeness.py -q`；实际结果：GitHub Actions job success，4 items `.... [100%]`，exit code 0，4 passed，0 skipped，0 failed。

- 2026-06-05（甲）：

  **W4 focused validation 阶段复跑已记录**：
  - 新增记录：`docs/todo/owner-a-generation-upstream/2026-06-05-W4-focused-validation-record.md`。
  - 通过项：W2 orchestrator feedback producer 8 项；HFM/JMCG consumer 12 项；C1 generator_coord 20 项；C2 validation/srb 32 项；W3 mf_eval 24 项；W11 quality 6 项 + strict CLI smoke；W13 cross-paradigm KD 18 项 + CLI smoke；W9/W10/W11 hardening 4 项。
  - 未通过项：W1 unit gate 14 项中 3 项失败，根因是乙侧测试 patch `orchestrator_svc.main.build_shared_crg_repository_from_env`，但当前实现只在 `_merge_agent_beliefs_into_crg()` 内局部 import `mf_core.db.repositories.build_shared_crg_repository_from_env`；甲未改乙侧代码，需乙侧决定调整 patch 路径或导出模块级 seam。W5 benchmark 18 项中 8 failed/10 skipped，失败来自本地 `CCO` baseline 达不到 GuacaMol/PMO 生产阈值，skip 来自官方 benchmark 数据 env 缺失。
  - 大组说明：`uv run pytest tests/unit/test_generators.py tests/unit/test_service_artifact_status.py -q` 收集 285 项后运行过慢，约 14 分钟后终止，退出码 143；本次不计为 pass/fail，采用上述 work-item focused gates。
  - 验收结果：Owner A 本地工程 gate 保持绿色；W1 unit seam 需 Owner B 处理；W5 仍等待 H8 官方 benchmark 数据和 production-quality generated samples；C1/C2/C3 字段无变更。

- 2026-06-05（H8）：

  **H8 官方 benchmark 数据对接阻塞已记录**：
  - 已投放 MOSES 官方 test split：`MOSES_REFERENCE_SMILES_PATH=data/benchmarks/moses_reference_smiles.smi`；来源为 `molecularsets/moses` 官方 `data/dataset_v1.csv`，筛选 `SPLIT=test` 得到 176074 条 SMILES。
  - 当前资源状态：`PMO_SCORE_TABLE_PATH` missing；`CROSSDOCKED_BENCHMARK_JSONL` missing；`.env` 中 `HFM_CHECKPOINT_PATH`、`HFM_DECODER_PATH` 已设置且路径存在，但当前 decoder JSON 带本地 pytest 临时产物痕迹，不能作为 production-quality HFM artifact 完成证据。
  - 验证：`PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark -q -ra` 首次 exit code 0，18 skipped；加载 `.env` 后 `timeout 180s bash -lc 'set -a; source .env; set +a; PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark -q -ra'` exit code 124，无通过证据；最小 GuacaMol smoke `timeout 120s bash -lc 'set -a; source .env; set +a; export GUACAMOL_BENCHMARK_BATCH_SIZE=1; PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark/guacamol_benchmark.py::TestGuacaMolBenchmark::test_celecoxib_rediscovery -q -ra'` exit code 124。
  - 剩余 gate：甲方/人工提供可信 `PMO_SCORE_TABLE_PATH`（需 `smiles,drd2,jnk3,gsk3b`）、可信 `CROSSDOCKED_BENCHMARK_JSONL`（需 `pocket_id,ligand_smiles,split`，正式 gate 还需真实 `docking_score`）、production-quality `HFM_CHECKPOINT_PATH`+`HFM_DECODER_PATH` 或生产 decoder command，并设定正式阈值。H8 未完成，不登记完成验收。

- 2026-06-08（H8）：

  **H8 smoke 资源推进已记录**：
  - CrossDocked：基于本地 `/workspace/MForge/zzzzz/types/it2_tt_v1.3_completeset_test0.types` 与 `data/processing/crossdocked_full_extract.tmp` 中真实 SDF/GNINA 数据生成 `data/benchmarks/crossdocked_benchmark.jsonl`，共 1000 条 `split=test` 记录，字段含 `pocket_id`、`ligand_smiles`、`docking_score`；`.env` 新增 `CROSSDOCKED_BENCHMARK_JSONL=data/benchmarks/crossdocked_benchmark.jsonl`。
  - 验证：`timeout 180s bash -lc 'set -a; source .env; set +a; PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark/crossdocked_benchmark.py -q -ra'` exit code 0，4 passed，0 skipped。
  - PMO：已按官方 `wenhao-gao/mol_opt` 的 `data/zinc.csv.gz` 下载到 `data/benchmarks/pmo_zinc.csv.gz`；当前环境缺 `tdc`，`timeout 600s uv pip install PyTDC` exit code 124，`timeout 60s env PYTHONDONTWRITEBYTECODE=1 uv run python - <<'PY' ... from tdc import Oracle ... PY` exit code 1，错误 `ModuleNotFoundError: No module named 'tdc'`，因此 `PMO_SCORE_TABLE_PATH` 仍未生成。
  - HFM：`checkpoints/hfm3d_4h200` 仅作为 smoke artifact 使用。
  - 剩余 gate：完成 PyTDC/PMO oracle 环境安装并生成可信 `PMO_SCORE_TABLE_PATH`，再跑 W5 benchmark 非 skip gate；H8 未完成。

- 2026-06-08（H8）：

  **PMO blocker 已解除；H8 仍未完成**：
  - PMO：新建专用 PMO 环境 `.venv-h8-pmo`；基于 PMO 官方 `wenhao-gao/mol_opt` 的 `data/zinc.csv.gz` 与 PyTDC/TDC oracle 模型 `drd2_current`、`jnk3_current`、`gsk3b_current` 生成 `data/benchmarks/pmo_score_table.csv`；`.env` 新增 `PMO_SCORE_TABLE_PATH=data/benchmarks/pmo_score_table.csv`。表字段为 `smiles,drd2,jnk3,gsk3b`，2 条真实 scored ZINC rows；预检 `max_drd2=0.952971043107`、`max_pair_jnk3_gsk3b=0.635`。
  - focused 验证：PMO score table gate `timeout 180s bash -lc 'set -a; source .env; set +a; PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark/pmo_benchmark.py::TestPMOBenchmark::test_drd2_optimization tests/benchmark/pmo_benchmark.py::TestPMOBenchmark::test_multi_objective_jnk3_gsk3b -q -ra'` exit code 0，2 passed，0 skipped；CrossDocked 回归 `timeout 180s bash -lc 'set -a; source .env; set +a; PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark/crossdocked_benchmark.py -q -ra'` exit code 0，4 passed，0 skipped。
  - W5 总 gate：`timeout 300s bash -lc 'set -a; source .env; set +a; PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark -q -ra'` exit code 1，8 passed，9 failed，1 skipped。失败原因是当前 HFM smoke artifact 只生成重复 `CCO`，未达到 GuacaMol/MOSES/PMO 生成质量阈值（Celecoxib similarity 0.028571 < 0.75、MOSES uniqueness 0.00390625 < 0.95、PMO LogP/QED/LogP_QED 低于默认阈值）。
  - 剩余 gate：production-quality HFM checkpoint/decoder 或生产 decoder command、正式生成质量阈值、`uv run pytest tests/benchmark -q` 全量通过；H8 未完成，不登记完成验收。

- 2026-06-05（H6）：

  **H6 AiZynth 真实 command path smoke gate 已完成**：
  - 改动文件：`tools/retrosyn/aizynth_planner_wrapper.py`、`tests/unit/test_h6_retrosyn_wrapper.py`、`.env`。service main/schema 未改。
  - 交付：wrapper 支持真实 ONNX AiZynthFinder + 显式 inline stock，避免 fixed-format 3500 万条 HDF5 stock 全量加载；`.env` 配置 H6 search limits 与 `RETROSYN_PLANNER_COMMANDS_JSON` AiZynth command。
  - 验证：`uv run pytest tests/unit/test_h6_retrosyn_wrapper.py -q` exit code 0，2 passed；retrosyn command focused pytest exit code 0，3 passed；`uv run ruff check tools/retrosyn/aizynth_planner_wrapper.py tests/unit/test_h6_retrosyn_wrapper.py` exit code 0；`python -m py_compile tools/retrosyn/aizynth_planner_wrapper.py tests/unit/test_h6_retrosyn_wrapper.py` exit code 0；source `.env` 后 `runtime_status` 显示 `retrosyn_aizynth_planner_command configured=True available=True`；真实 wrapper smoke `timeout 360 uv run python tools/retrosyn/aizynth_planner_wrapper.py` 对 `CCO/max_routes=1/engine=aizynth` exit code 0，`total_routes_found=1`；真实 service command path smoke `RetrosynServicer().FindRoutes(... engine="ensemble")` exit code 0，`total_routes_found=1`。
  - 剩余 gate：RetroGNN/RSGPT/UAlign 生产 runner、集群发布与生产多引擎验收仍未完成；C1/C2/C3 无变更。

- 2026-06-05（H4）：

  **H4 L4 PySCF quantum command path smoke gate 已完成**：
  - 资源投放：`.env` 新增 H4 key：`L4_QUANTUM_ORACLE_COMMAND` 指向 `tools/oracles/pyscf_quantum_oracle_wrapper.py`，`L4_QUANTUM_ENGINE=pyscf`，`L4_PYSCF_METHOD=RHF`，`L4_PYSCF_BASIS=sto-3g`；当前 `.venv` 已安装 `pyscf==2.13.1` 与 `gpu4pyscf-cuda12x==1.7.1`。
  - 真实 smoke：`printf ... | L4_PYSCF_METHOD=RHF L4_PYSCF_BASIS=sto-3g timeout 600 uv run python tools/oracles/pyscf_quantum_oracle_wrapper.py` exit code 0，返回非空 `scores.quantum_correction=-39.72460094981219`。
  - command path 验证：从 `.env` 读取 H4 key 后构造 `QuantumCommandOracle`，`timeout 700 uv run python ...` exit code 0，返回 `{"C": {"engine": "pyscf", "quantum_correction": -39.724600949812164}}`。
  - focused gate：`uv run pytest tests/unit/test_validation_agent.py -k "quantum_command or l4_quantum" -q` exit code 0，6 passed。
  - 剩余 gate：GPU4PySCF `from gpu4pyscf import scf` 本机超过 5 分钟未返回，未登记 GPU4PySCF 真实计算完成；`command -v orca` 仍 missing，ORCA 未投放；集群发布验证仍未完成。C1/C2/C3 无变更。

- 2026-06-06（H4）：

  **H4 L4 GPU4PySCF quantum command path smoke gate 已完成**：
  - 资源投放：`.env` H4 默认 engine 切到 `L4_QUANTUM_ENGINE=gpu4pyscf`，`L4_QUANTUM_ORACLE_COMMAND="uv run python tools/oracles/pyscf_quantum_oracle_wrapper.py"`、`L4_PYSCF_METHOD=RHF`、`L4_PYSCF_BASIS=sto-3g` 保持 H4 wrapper 合同；`.venv` 中 `pyscf==2.13.1`、`gpu4pyscf-cuda12x==1.7.1`、`cupy-cuda12x==13.6.0` 可探测。
  - native artifact：投放 H200/sm_90 可执行 GPU4PySCF 库 `libgvhf_rys.so`、`libgvhf_md.so`、`libcupy_helper.so`，并投放 H2/STO-3G RHF 所需真实 `libgint.so`（含 `cart2sph_*` + `GINTinit_*`/basis cache 符号；原 wheel 备份在 `h4_sm90_backup_20260606/`）。
  - 真实 smoke：wrapper smoke `timeout 760 bash -lc 'printf ... engine=gpu4pyscf | L4_PYSCF_METHOD=RHF L4_PYSCF_BASIS=sto-3g L4_QUANTUM_ENGINE=gpu4pyscf CUDA_VISIBLE_DEVICES=2 uv run python tools/oracles/pyscf_quantum_oracle_wrapper.py'` exit code 0，返回非空 `scores.quantum_correction=-1.1174874250696716`。
  - command path 验证：从 `.env` 读取 H4 key 后构造 `QuantumCommandOracle`，`timeout 820 bash -lc 'set -a; source .env; set +a; export CUDA_VISIBLE_DEVICES=2; uv run python ...'` exit code 0，返回 `{"[H][H]": {"engine": "gpu4pyscf", "quantum_correction": -1.1174874250696716}}`。
  - focused gate：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -p pytest_asyncio.plugin tests/unit/test_h4_quantum_wrapper.py tests/unit/test_validation_agent.py -k "quantum_command or l4_quantum or gpu4pyscf_rhf" -q` exit code 0，7 passed。
  - 剩余 gate：`command -v orca` exit code 1，ORCA 未投放；完整 upstream `gint` 全目标 sm_90 构建此前 timeout 124，当前验收仅覆盖 H4 wrapper H2/STO-3G RHF 所需真实 GPU4PySCF path；集群发布验证仍未完成。C1/C2/C3 无变更。

- 2026-06-07（H4）：

  **H4 L4 GPU4PySCF wrapper path 修复与复验已完成**：
  - 修复：`tools/oracles/pyscf_quantum_oracle_wrapper.py` GPU 分支从 PySCF `.to_gpu()` 宽转换改为直接构造真实 `gpu4pyscf.scf.hf.RHF(mol).kernel()`；保留窄加载 `_patch_pyscf.py` 和 H4 stdin/stdout command 合同。
  - 真实 smoke：wrapper smoke `timeout 760 bash -lc 'printf ... engine=gpu4pyscf | L4_PYSCF_METHOD=RHF L4_PYSCF_BASIS=sto-3g L4_QUANTUM_ENGINE=gpu4pyscf CUDA_VISIBLE_DEVICES=2 uv run python tools/oracles/pyscf_quantum_oracle_wrapper.py'` exit code 0，返回 `engine=gpu4pyscf`、`scores.quantum_correction=-1.1174874250696716`、`elapsed_ms=177998`。
  - command path 验证：从 `.env` 读取 H4 key 后构造 `QuantumCommandOracle`，`timeout 820 bash -lc 'set -a; source .env; set +a; export CUDA_VISIBLE_DEVICES=2; uv run python ...'` exit code 0，返回 `{"[H][H]": {"engine": "gpu4pyscf", "quantum_correction": -1.1174874250696716}}`。
  - focused gate：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -p pytest_asyncio.plugin tests/unit/test_h4_quantum_wrapper.py tests/unit/test_validation_agent.py -k "quantum_command or l4_quantum or gpu4pyscf_rhf" -q` exit code 0，8 passed；`uv run python -m py_compile tools/oracles/pyscf_quantum_oracle_wrapper.py tests/unit/test_h4_quantum_wrapper.py` exit code 0。
  - 剩余 gate：ORCA 未登记完成；集群发布验证仍未完成。C1/C2/C3 无变更。

- 2026-06-06（甲）：

  **W11 HUMU-labeled FragFM 50-record smoke gate 已完成**：
  - 新增 `mf_generators.fragfm.humu_labeling` CLI/API，直接加载 frozen HUMU molecule encoder checkpoint 派生 129 维 Lorentz `humu_embedding`。
  - 派生数据：`data/processing/generator_artifacts/fragfm_records_train_humu_labeled.jsonl`，report `status=pass`、50/50 encoded、coverage 1.0。
  - 训练 artifact：`checkpoints/fragfm_humu_smoke/`，未覆盖 `checkpoints/fragfm`；manifest `humu_embedding_count=50`、coverage 1.0。
  - 验证：4 条 focused HUMU-labeling tests 通过；`python3 -m py_compile` 通过；strict quality gate `--min-humu-coverage 1.0` 通过，checkpoint/rate-matrix loadable。
  - 剩余 gate：5000-record local candidate / production-quality artifact、正式 benchmark/coverage 阈值、集群发布验证；C1/C2/C3 无变更。

- 2026-06-06（甲）：

  **W11 5000-record HUMU labeling input gate 已完成；5k training artifact 未完成**：
  - 派生数据：`data/processing/generator_artifacts/fragfm_records_humu_labeled.jsonl`，paired report `status=pass`、5000/5000 encoded、coverage 1.0、invalid counts 0。
  - 验证：行数 5000 input / 5000 output；全部 embedding finite，维度 129，无缺失；max Lorentz equation deviation 约 `4.72e-06`。
  - 训练尝试：`checkpoints/fragfm_humu_5k/` 的 CPU training attempt 使用 batch 64 和 batch 5000 均长时间运行后只写出 `vocab.json`，无 checkpoint/manifest；partial 目录已清理。
  - 剩余 gate：`checkpoints/fragfm_humu_5k/` 不存在；需要 GPU/cluster 训练或训练路径优化后再产出 artifact，并继续完成正式 benchmark/coverage 阈值和集群发布验证。C1/C2/C3 无变更。

- 2026-06-06（甲）：

  **W11 FragFM 5k CPU training bottleneck 诊断与窄优化已完成**：
  - 诊断：5000-record dataset 有 2860 个 unique fragments；原 `_rate_transition_loss()` 会物化 `[batch, vocab, vocab]` full rate tensor；第一版 sparse path 仍会为每个样本读取完整 `vocab*vocab` SA modulation 后切行。
  - 改动：新增 sparse transition-row + SA row-gather path，custom rate matrix 仍走 full-matrix fallback。
  - 验证：focused regression 证明 sparse loss 与 full matrix loss 等价，且不再调用 full `sa_score_embedding.forward()`；`python3 -m py_compile` 通过；focused pytest 6 项通过；50-record `/tmp` training smoke 产出 checkpoint/rate-matrix/vocab/manifest 且 coverage 1.0。
  - 重试/状态：5k CPU training 仍未完成；`checkpoints/fragfm_humu_5k/` 不存在。
  - 剩余 gate：rate-loss memory issue 已修复且 hot path 进一步收窄，但仍需 GPU/cluster 或进一步训练优化后再完成 production-quality artifact。C1/C2/C3 无变更。

- 2026-06-06（甲）：

  **W11 FragFM rate optimizer controls 已完成**：
  - 改动：新增显式 `--rate-optimizer {adamw,sgd}` 与 `--disable-rate-grad-clip`；默认仍为 AdamW + rate grad clipping；manifest 记录 `rate_optimizer` 与 `rate_grad_clip`。
  - 验证：新 CLI regression 先 RED 后 GREEN；focused pytest 7 项通过；50-record HUMU-labeled `/tmp` smoke 和 256-record 5k-subset smoke 均用 `--rate-optimizer sgd --disable-rate-grad-clip` 产出 checkpoint/rate-matrix/vocab/manifest，coverage 1.0。
  - 状态：checkpoint/rate-matrix schema 不变；该时点 `checkpoints/fragfm_humu_5k/` 仍未产出，后续 5000-record local candidate 记录见下一条。C1/C2/C3 无变更。

- 2026-06-06（甲）：

  **W11 5000-record HUMU-labeled FragFM local candidate 已完成**：
  - 产物：`checkpoints/fragfm_humu_5k/` 包含 `vocab.json`、`best_model.pt`、`rate_matrix.pt`、`final_model.pt`、`final_rate_matrix.pt`、`training_manifest.json`、`quality_report.json`。
  - 训练：1 epoch、batch 64、hidden dim 8、CPU、`--rate-optimizer sgd --disable-rate-grad-clip`；日志 `Epoch 1/1: loss=8.7005`。
  - manifest：`records=5000`、`fragments=2860`、`humu_embedding_count=5000`、coverage 1.0、`rate_optimizer=sgd`、`rate_grad_clip=false`。
  - quality：strict `mf_generators.fragfm.quality --min-humu-coverage 1.0` pass；checkpoint/rate-matrix loadable；messages empty。
  - 状态：该 artifact 是 local engineering candidate，不是 final production W11 acceptance；剩余 benchmark/production training/deployment/cluster gate 未完成。C1/C2/C3 无变更。

- 2026-06-06（甲）：

  **W11 FragFM HUMU 5k deployment defaults 已硬化**：
  - 改动：Docker Compose、raw Kubernetes、Helm `fragfm-generator-config` 默认值从旧 `checkpoints/fragfm/{vocab.json,best_model.pt,rate_matrix.pt}` 切到 `checkpoints/fragfm_humu_5k/{vocab.json,best_model.pt,rate_matrix.pt}`；仍保留 env override。
  - 验证：focused deployment regression 先 RED 后 GREEN；当前 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_service_artifact_status.py::test_fragfm_deployment_wires_artifact_and_sampler_env -q` 通过 1 项，并校验 artifact 存在且 `quality_report.json` coverage 1.0。
  - 状态：这是 deployment-default hardening，不是 cluster acceptance；剩余 production-quality training、benchmark、正式阈值、artifact promotion policy、cluster validation。C1/C2/C3 无变更。

- 2026-06-07（H5）：

  **H5 L1-L3 oracle command wrapper 本地验收已完成**：
  - `.env` 状态：已投放 `DOCK_ORACLE_COMMAND`、`BOLTZ2_ORACLE_COMMAND`、`FEP_ORACLE_COMMAND`、`ADMET_ORACLE_COMMAND` 及 Boltz/OpenADMET/OpenFE registry 运行 key；日志只登记 key 状态，不记录任何 secret/token/key 具体值。本轮仅补充修改 H5 key `OPENFE_TRANSFORMATION_REGISTRY`、`OPENFE_RESULT_REGISTRY`。
  - 配置验证：`set -a; source .env; set +a; python ...` exit code 0；四个 command env 均为 set，首个可执行均可解析；`OPENFE_RUNNER_PATH` 首个可执行可解析；`OPENFE_CLI_PATH`、`OPENFE_TRANSFORMATION_REGISTRY`、`OPENFE_RESULT_REGISTRY`、`OPENFE_WORK_DIR`、`FEP_JOB_DIR` 均存在。
  - focused gate：`uv run pytest tests/unit/test_h5_oracle_wrappers.py -q` exit code 0，13 passed；`uv run pytest tests/unit/test_service_artifact_status.py -k 'dock_service_runs_configured_json_command or boltz2_service_runs_configured_json_command or fep_service_runs_configured_json_command or admet_service_runs_configured_json_command' -q` exit code 0，4 passed；FEP service focused 回归 `uv run pytest tests/unit/test_service_artifact_status.py -k 'fep_service_submits_background_json_command_job or fep_service_runs_configured_json_command or fep_oracle_service_maps_evaluations_to_rbfe_scores' -q` exit code 0，3 passed；`PYTHONPYCACHEPREFIX=/tmp/mforge-pycache-h5 uv run python -m py_compile ...` exit code 0；`git diff --check -- ...` exit code 0。
  - 真实 smoke：OpenADMET 主预测 smoke exit code 0，JSON parse OK，1 条 clearance float prediction；Boltz GPU affinity smoke 使用 6OIM/CCO、GPU、结构采样 10、affinity 采样 10，exit code 0，stdout 278 bytes，stderr 0 bytes，JSON parse OK，affinity_count=1，生成 CIF、confidence JSON 和 affinity JSON。
  - FEP/OpenFE：`openfe fetch rbfe-tutorial`、`openfe fetch rbfe-tutorial-results`、`openfe gather ... --report dg --tsv`、`openfe gather ... --report ddg --tsv` 均 exit code 0；`openfe plan-rbfe-network ... --n-protocol-repeats 1 -s settings.yaml` exit code 0，持续约 17 分 58 秒，stderr 摘要为 multiprocessing fork DeprecationWarning 与 element-change UserWarning，无 fatal error。已生成 `models/artifacts/openfe/tyk2/transformation_registry.json`（9 条完整 complex/solvent 边）和 `models/artifacts/openfe/tyk2/result_registry.json`（18 条正反向 ddG 边）；官方 TYK2 结果 `final_results_ddg.tsv` 为 9 rows，DDG range 2.29 kcal/mol；`final_results_dg.tsv` 为 10 rows，DG(MLE) range 3.25 kcal/mol。FEP wrapper registry smoke exit code 0，stderr 空，返回 `ddg_kcal_mol=0.8`；FEP service background job smoke exit code 0，最终 `state=completed`、`results=1`、`ddg_kcal_mol=0.8`。
  - Boltz 卡点处理：标准 CLI 在 checkpoint load 前随机初始化 2GB 级模型导致超时，新增 fast CLI 入口跳过会被 checkpoint 覆盖的初始化；runner 固化 `--num_workers`。`sampling_steps=1` / `sampling_steps_affinity=1` 在真实 smoke 中触发 SVD 数值失败，已改为通过 smoke 的采样配置。
  - 状态：H5 本地 command wrapper / ADMET / Boltz / FEP TYK2 registry gate 完成。TYK2 教程输入和 SDF 未包含实验 Ki/IC50 标签，因此该 registry 证明真实 OpenFE 模拟结果具有区分分布，不登记实验相关性；本地仅发现 KRAS G12C 6OIM 共价复合物 PDB 与 Boltz template，未发现 KRAS OpenFE 配体系列、实验 ddG 或 covalent-FEP registry。真实 OpenFE 长程 MD、集群发布与 KRAS full pilot 仍归 H10/H11。C1/C2/C3 无变更。

- 2026-06-07（H6）：

  **H6 多引擎 retrosynthesis 本地真实 command path gate 已完成**：
  - 资源投放：`.env` 接入 AiZynth、UAlign、RSGPT 三个 `RETROSYN_PLANNER_COMMANDS_JSON` runner；RSGPT 使用真实 `finetune_50k.pth`、官方 `rxngpt_llama1B.json` 和匹配 1000 vocab `vocab.json`；RetroGNN 按用户决定舍弃。
  - 交付：`tools/retrosyn/rsgpt_planner_wrapper.py` 使用真实 RSGPT checkpoint 快路径加载 `LlamaForCausalLM`，保留 JSON stdin/stdout command 合同；`tests/unit/test_h6_retrosyn_wrapper.py` 覆盖 AiZynth/RSGPT/UAlign wrapper 合同。
  - 验证：`uv run pytest tests/unit/test_h6_retrosyn_wrapper.py -q` exit code 0，7 passed；retrosyn command focused pytest exit code 0，3 passed；`uv run ruff check tools/retrosyn/aizynth_planner_wrapper.py tools/retrosyn/rsgpt_planner_wrapper.py tools/retrosyn/ualign_planner_wrapper.py tests/unit/test_h6_retrosyn_wrapper.py` exit code 0；`python -m py_compile ...` exit code 0；source `.env` 后三路 retrosyn command 均 configured=True available=True。
  - 真实 smoke：RSGPT wrapper 对 `CCO/max_routes=1/engine=rsgpt` exit code 0，`total_routes_found=1`，route `CCOC(=O)CBr>>CCO`；UAlign wrapper exit code 0，`total_routes_found=1`；service ensemble command path `RetrosynServicer().FindRoutes(... engine="ensemble")` exit code 0，`total_routes_found=3`，返回 AiZynth/RSGPT/UAlign 非空 routes。
  - 状态：H6 本地多引擎真实 command path gate 完成；剩余 gate 为集群发布验证与 KRAS full pilot（H10/H11）。C1/C2/C3 无变更。

- 2026-06-07（H6）：

  **H6 RAscore 快筛替换与四引擎真实 command path gate 已完成**：
  - 资源投放：`models/artifacts/rascore_source/RAscore` 使用官方 RAscore 源码；官方旧 XGB pickle 转换为 `models/artifacts/rascore/XGB_chembl_ecfp_counts/model.json`，与官方 `predict_proba` 分数对齐；`.env` 接入 `rascore/aizynth/ualign/rsgpt` 四个 `RETROSYN_PLANNER_COMMANDS_JSON` runner。
  - 交付：`tools/retrosyn/rascore_planner_wrapper.py` 输出真实 `retrosynthetic_accessibility_score`；service/agent 命名 runner env 从 `RETROGNN_PLANNER_COMMAND` 改为 `RASCORE_PLANNER_COMMAND`；route 排序保护确保 RAscore 分数不会在 `max_routes` 截断时挤掉真实 reaction route。
  - 验证：`uv run pytest tests/unit/test_h6_retrosyn_wrapper.py -q` exit code 0，9 passed；原 H6 retrosyn command focused pytest exit code 0，3 passed；RAscore named/ranking focused pytest exit code 0，4 passed；`uv run ruff check tools/retrosyn/rascore_planner_wrapper.py tools/retrosyn/aizynth_planner_wrapper.py tools/retrosyn/rsgpt_planner_wrapper.py tools/retrosyn/ualign_planner_wrapper.py tests/unit/test_h6_retrosyn_wrapper.py` exit code 0；`python -m py_compile tools/retrosyn/rascore_planner_wrapper.py tools/retrosyn/aizynth_planner_wrapper.py tools/retrosyn/rsgpt_planner_wrapper.py tools/retrosyn/ualign_planner_wrapper.py tests/unit/test_h6_retrosyn_wrapper.py` exit code 0；source `.env` 后四路 retrosyn command 均 configured=True available=True。
  - 真实 smoke：RAscore wrapper 对 `CCO/max_routes=1/engine=rascore` exit code 0，`total_routes_found=1`，`accessibility_score=0.990022599697113`；service ensemble command path `RetrosynServicer().FindRoutes(... engine="ensemble", max_routes=4)` exit code 0，`total_routes_found=4`，返回 AiZynth/RSGPT/UAlign 非空真实 routes + RAscore 可及性评分，`elapsed_ms=1990311`。
  - 状态：H6 本地真实 command path gate 完成；剩余 gate 为集群发布验证与 KRAS full pilot（H10/H11）。C1/C2/C3 无变更。

- 2026-06-07（H3）：

  **H3 商业供应商真实 API 设想已按用户决定舍弃，不登记完成验收**：
  - 范围调整：H3 从 C 类资源域移除；不再采购或投放 Enamine/Mcule/eMolecules/Chemspace 四组真实 API / sandbox；不运行四家商业供应商真实 smoke。
  - 代码/配置清理：`supply-oracle-svc` 删除商业 HTTP provider / aggregator / retry-backoff provider wiring，只保留 `SUPPLY_CATALOG_URI=file://...` 本地 JSON catalog 与 AiZynth HDF5 stock 路径；Docker Compose、raw Kubernetes、Helm 删除 `SUPPLY_COMMERCIAL_*`、四家 `SUPPLY_*_API_*`、`commercial-supply-config` 和 `commercial-supply-credentials` wiring。
  - 占位清理：删除 `data/ingestion/enamine_real/` 下 Enamine REAL FAISS placeholder；删除相关待办/架构/测试文案中的供应商 API、Enamine REAL 占位和供应商特定样例源名。
  - 验证：`.env` H3 key 搜索无匹配；`PYTHONDONTWRITEBYTECODE=1 uv run python ... compile(...)` exit code 0，`syntax_ok`；Supply focused gate `timeout 180s env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 uv run pytest -p pytest_asyncio.plugin ... -q --tb=short` exit code 0，7 passed；`services/supply-oracle-svc/src/supply_oracle_svc/__init__.py` 包级旧描述已清理并恢复 mode `664`。
  - 状态：H3 无剩余 blocker；C1/C2/C3 无变更。

- 2026-06-07（W12）：

  **W12 CReM-pharm-3D 本地真实 scorer runner 闭环已完成**：
  - runner 投放：`.env` 已投放 `CREM_DOCK_ORACLE_TARGET`、`DOCK_ORACLE_RECEPTOR_PDB`、`CREM_PHARMACOPHORE_REFERENCE_SDF`、`CREM_PHARMACOPHORE_SCORER_COMMAND`、`CREM_HUMU_SCORER_COMMAND`、`CREM_SCORER_COMMAND_TIMEOUT_SECONDS`；未记录任何 secret/token/key。DiffDock-L 按用户决定移出 W12 本地 gate。
  - 真实 artifact：`models/artifacts/crem/6OIM.pdb` 与 `models/artifacts/crem/MOV_ideal.sdf` 来自 RCSB 6OIM/MOV；HUMU scorer 使用既有 `HUMU_CHECKPOINT_PATH`；docking scorer 复用 H5 的真实 `DOCK_ORACLE_COMMAND` 与 GNINA binary。
  - 验证：`PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/unit/test_phase_b_generators.py -q -k "crem"` exit code 0，6 passed；`PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/unit/test_service_artifact_status.py -q -k "crem_service_builds_configured_external_scorers or crem_external_json_score_provider_returns_smiles_records or crem_external_json_score_provider_preflight_rejects_missing_executable or crem_deployment_wires_mmp_and_external_scorer_env or crem_runtime_rejects_missing_external_scorer_command or dock_oracle_uses_default_receptor_for_oracle_requests"` exit code 0，6 passed；`PYTHONPYCACHEPREFIX=/tmp/... uv run python -m py_compile ...` exit code 0；`git diff --check -- .env tools/scorers/crem_humu_scorer.py tools/scorers/crem_pharmacophore_scorer.py services/dock-svc/src/dock_svc/main.py tests/unit/test_phase_b_generators.py tests/unit/test_service_artifact_status.py` exit code 0。
  - 真实 smoke：pharmacophore command 对 CReM 最小分子返回非空 `pharmacophore_score=0.5055679672060487`；HUMU command 返回 129 维 `humu_embedding` 与 `humu_alignment_score=-0.12193988789716237`；dock gRPC scorer 返回 `oracle_name=gnina`、`docking_score=2.11969`；CReM generator 全链路 smoke exit code 0，1 条 molecule 同时包含 `docking_score=2.37787`、`pharmacophore_score=0.6224669558183266`、`humu_alignment_score=-0.07110921506371735` 且 `humu_embedding` 非空。
  - 剩余 gate：H10 集群发布验证；生产级 pharmacophore reference 策略和正式 benchmark 仍归后续生产验收，不阻塞 W12 本地真实 scorer runner 闭环。C1/C2/C3 无变更。

- 2026-06-08（H9）：

  **H9 CIG LLM/SRM parser/refiner 真实接入本地 smoke gate 已完成**：
  - 资源投放：`.env` 已投放 `CIG_DEEPSEEK_MODEL=deepseek-v4-flash`、`CIG_SEMANTIC_PARSER_COMMAND`、`CIG_SEMANTIC_PARSER_TIMEOUT_SECONDS`、`CIG_REFINEMENT_COMMAND`、`CIG_REFINEMENT_TIMEOUT_SECONDS` 和 `HCIV_CHECKPOINT_PATH=checkpoints/hciv_encoder/h9_sklearn_hashing_smoke.pt`；DeepSeek API key 仅验证为 set，未记录任何 secret/token/key。
  - 真实 artifact：DeepSeek semantic parser 处理 8 条 CIG intent 生成 `data/processing/cig_hciv/h9_teacher.jsonl`，每条含 129 维 Lorentz target；`services/cig-compiler-svc/train_hciv_encoder.py` 训练导出 `checkpoints/hciv_encoder/h9_sklearn_hashing_smoke.pt` 与 manifest，manifest `example_count=8`、`epochs=20`、`dim=128`。
  - 真实 smoke：parser command smoke exit code 0，返回 KRAS G12C target 与 `max_mw` 约束；`ProductionSemanticParserAdapter()` command path smoke exit code 0；`CIGCompilerServicer(compiler=CIGCompiler(enable_grounding=False)).Compile(...)` exit code 0，输出 CIG、129 维 HCIV 和 129 维 cone；refiner direct command smoke exit code 0；`CIGCompilerServicer().Refine(...)` service command path smoke exit code 0。
  - 验证命令：teacher builder exit code 0；HCIV training exit code 0；`runtime_status()` 显示 `cig_semantic_parser_command` 与 `cig_refinement_command` configured=true、available=true；`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/unit/test_h9_cig_llm_wrappers.py -q` exit code 0，8 passed；`py_compile` 与 `git diff --check` 通过。
  - 剩余 gate：当前 HCIV checkpoint 是 H9 smoke artifact，不登记为 production-quality W10；仍需集群发布验证、外部 grounding 打开后的端到端验收和下游质量验证。C1/C2/C3 无变更。
