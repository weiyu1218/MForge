# 当前实现与 CoreArchitecture v2 架构偏差说明

## 0. 元信息

- 对照架构文档：`/workspace/MForge/MoleculeForge_CoreArchitecture_v2.md`
- 实际实现范围：`/workspace/MForge/moleculeforge`
- 更新时间：2026-06-03
- 核对方式：静态核对源码、配置、schema、proto、测试入口和部署文件；本地关联单元测试与 lint 通过；2026-06-03 补充 CRG/Provenance 静态核对未运行测试；未连接外部模型、数据库或服务。
- 结论边界：本文只记录能在当前仓库文件中找到证据的事实，不把路线图、旧审查文档或注释描述当作已实现能力。

## 1. 总体结论

当前实现不是空壳，已经有较完整的工程目录、部分真实组件和较多 fail-fast 的服务外壳。但它与 `MoleculeForge_CoreArchitecture_v2.md` 描述的目标架构仍有明显偏差。

准确状态如下：

```text
工程结构       :  较完整。当前有 21 个 services、8 个 agents、7 个 generator 目录。
HUMU 基础      :  Lorentz 流形算子、可学习曲率、mol/pocket/route 预训练 pipeline 存在；预训练阶段不依赖用户意图和意图锥。
DKI 基础       :  Postgres / Neo4j / Qdrant / MinIO / Redis / Feast 相关代码和配置存在。
SRB            :  SSP 编译逻辑存在，能从 retrosyn route 生成 SSP。
Oracle         :  L0 RDKit oracle 真实存在；L1-L3 已具备本地 command wrapper gate：admet-svc 支持 `ADMET_ORACLE_COMMAND` 外部 JSON command wrapper，dock-svc 支持 `DOCK_ORACLE_COMMAND` 外部 JSON command wrapper，boltz2-svc 支持通用 `OracleService` adapter 和 `BOLTZ2_ORACLE_COMMAND` 外部 JSON command wrapper，fep-svc 支持通用 `OracleService` adapter 和 `FEP_ORACLE_COMMAND` 外部 JSON command wrapper；配置 ADMET/Dock/Boltz2/FEP command 时会校验首个可执行文件可用，ADMET 和 Dock runtime preflight 已拒绝显式配置但不可用的 command，不会被可用本机 artifact 掩盖；这些 oracle 服务的 Compose、静态 Kubernetes 清单和 Helm values 已暴露 command/timeout env、本机 artifact/tool runtime env 与 `oracle-runner-config` wiring，Compose、静态 Kubernetes 清单和 Helm values/template 已默认指向当前本地 GNINA、DiffDock-L、Boltz-2 和 Boltz input template artifact；Boltz2/FEP adapter 所需的 protein、ensemble、reference ligand、method 和 repeat 参数 env 也已接入；H5 已完成 command env 生效验证、focused wrapper/服务合同回归、OpenADMET 主预测 smoke、Boltz GPU affinity smoke 和 FEP TYK2 OpenFE registry wrapper/service smoke；ValidationAgent 已支持 `L0_ADMET_ORACLE_TARGET`、`L1_DOCKING_ORACLE_TARGET`、`L2_AFFINITY_ORACLE_TARGET`、`L3_FEP_ORACLE_TARGET` 和 `L4_QUANTUM_ORACLE_TARGET` 外部 OracleService wiring，也支持 `L4_QUANTUM_ORACLE_COMMAND`、`L4_GPU4PYSCF_COMMAND` 和 `L4_ORCA_COMMAND` 本地 JSON command wrapper，并在实际执行本地 L4 command 前校验首个可执行文件可用；`orchestrator-svc` 的 Compose、静态 Kubernetes 清单和 Helm values 已暴露 L4 quantum target、command、engine 与 GPU4PySCF/ORCA 命名 command env；FEP TYK2 registry 不代表真实 OpenFE 长程 MD 或 KRAS covalent-FEP full pilot；ORCA、集群发布验证和 KRAS full pilot 仍未完成。
AMGE           :  生成器目录和服务存在，入口已传递 intent cone，但底层多为简化路径或 runner wrapper，未形成共享 HUMU 协同生成。
MARB           :  LangGraph/CRG/Agent 骨架存在；orchestrator-svc 主链路已串起 CIC、HCIV、generation、validation、retrosyn、critic、provenance，RETROSYN 阶段已委托 RetroSynAgent 规划路线，并支持可选 `assess_supply` 与 `compile_synthesis` client hook；FullWorkflowClients 已把路线规划委托给 RetroSynAgent，并把后续 hook 委托给 SupplyAgent 与 SRBAgent；workflow state、provenance metadata 和 Neo4j belief/edge 写入链路中记录 workflow CRG；GraphRepository 已支持按 run_id 读回 workflow CRG；LangGraph refinement 已能把 validation/critic 失败反馈写入 `generation_feedback` 并回到下一轮 generation，FullWorkflowClients 已把该反馈透传给默认 HFM 和显式 GeneratorCoordAgent generation 路径；OrchestratorAgent 可写入 workflow_status belief，NL2ObjAgent 可写入 parsed_intent 与完整 compiled_cig JSON belief，GeneratorCoordAgent 可写入 selected_generators belief，ValidationAgent 可写入 validation_status belief，RetroSynAgent 可写入 retrosyn_routes 与 route_humu_embedding belief，SupplyAgent 可写入 supply_feasibility belief，SRBAgent 可写入 ssp_compiled belief，CriticAgent 可写入 critic_verdict belief；这些 agent 会在存在完整 Neo4j 环境变量时默认使用同一 shared CRG repository factory，并继承 BaseAgent 的 shared CRG 读回接口；OrchestratorAgent 已能读取同一 run 的 completed workflow_status 并返回 cached 工作流结果，NL2ObjAgent 已能读取同一 run 同一 intent 的 compiled_cig belief 并跳过重复 CIG compiler 调用，GeneratorCoordAgent 的 auto 路由已能读取同一 run 的既有 selected_generators、失败类 CRG belief 与 route_humu_embedding 并驱动 generator 选择和反馈，ValidationAgent 已能读取同一 run 同一分子的既有 validation_status 并跳过重复 oracle cascade，RetroSynAgent 已能读取同一 run 同一分子的 failed validation belief 或 `retrosyn_routes=0` 并跳过路线规划，SupplyAgent 已能读取同一 run 同一分子的既有 supply_feasibility 并跳过重复 supply oracle，也能读取同一 run 的 `retrosyn_routes=0` 并直接标记供应不可用，SRBAgent 已能读取同一 run 同一分子的 unavailable supply_feasibility 并跳过 SSP 编译，CriticAgent 已能读取同一 run 同一分子的既有 critic_verdict 并跳过重复规则评估，也能读取 validation/supply 失败 belief 以及 `retrosyn_routes=0` 并纳入 fail verdict，重复执行与失败反馈级 CRG 读回已覆盖上述 agent，BaseAgent 已支持将 mapping payload 编码为 JSON-LD，发布带 signature 的 AgentMessage envelope，并默认生成 UUIDv7 message_id，订阅接收端可自动解包、校验 recipient、message_type、payload_type_url 与 ttl 防循环、按 sender identity 验签，可验证篡改，并可通过 SIGSTORE_SIGN_COMMAND/SIGSTORE_VERIFY_COMMAND 外接 Sigstore/Rekor 命令，签名命令可接收 SIGSTORE_IDENTITY_TOKEN，验证命令可接收 sender、recipient、message_type 与 expected_identity；后续仍缺更深层跨 agent 联合优化、真实 Fulcio/Rekor 命令、生产身份令牌投放和外部系统闭环。
Agent gRPC 客户端:  NL2Obj、Supply、GeneratorCoord、RetroSyn 和 Validation 的 gRPC client 在同步构造 `grpc.aio` channel 前会保证默认 event loop 存在，避免无默认 event loop 的线程环境直接崩溃。
Sigstore 命令执行:  BaseAgent、provenance-svc 和 lineage `SigstoreSigner` 执行外部 Sigstore sign/verify command 前会校验首个可执行文件可用；缺失可执行文件时 fail-fast，不再落到裸 `FileNotFoundError`。
评估体系       :  MOSES / GuacaMol / PMO / CrossDocked benchmark 已有资源门控入口；默认环境缺正式数据和模型 artifact 时会 skip。
```

核心偏差与边界集中在三点：

1. 架构闭环未成立：文档要求的 `NL -> CIG -> HCIV -> HUMU -> AMGE -> Oracle -> RetroSyn -> CRG -> Provenance` 闭环，在主业务路径中没有完整跑通证据。
2. 算法实现降级：文档要求的 SEGNN、E(3)-GNN、双曲 TreeLSTM、Lorentz-equivariant Transformer、ProxylessNAS、REINFORCE、Hyperbolic GP 等，在当前实现中多为 MLP、规则、启发式或未接入模块。
3. DKI 技术选型已确认：CoreArchitecture v2 原文写的是 Milvus 和 NATS JetStream；当前项目正式采用 Qdrant 和 Redis，不再把该差异作为本阶段待补齐项。

## 2. 分层对照矩阵

| 层 | 文档要求 | 当前实现 | 偏差判断 |
|---|---|---|---|
| 0 JMCG | 在共享双曲流形建模 `(m,r,p)` 联合分布 | 主流程仍是分段生成、局部打分、简单排序；orchestrator refinement 已能把 validation/critic feedback 回注下一轮 generation，FullWorkflowClients 已把 `generation_feedback` 序列化透传给默认 HFM 路径和显式 GeneratorCoordAgent dispatch，并能派生 `kind="property"` / `kind="intent"` / `kind="pocket"` `jmcg_feedback` context records；GeneratorCoordAgent 已能从 CRG 读取 route HUMU feedback 并透传到下一轮 generator dispatch，也能把已有 property / intent / pocket context records 与 route records 合并；HFM-3D 已能消费 `route_humu_feedback`、含 HUMU embedding 的 `generation_feedback` 和 `jmcg_feedback` envelope，在 Lorentz 流形上对 post-flow latent 做有界 feedback steering，并已补初始 weight/confidence、polarity、per-kind aggregation 和 dropped-record metadata；GeneratorCoordAgent 已能把 route HUMU feedback 同步封装为 `moleculeforge.jmcg.feedback.v1` envelope，并保留 CRG route payload 中可用的 evidence / metadata provenance；property records 仍不含 HUMU embedding，intent records 仅在已有 finite 且满足 Lorentz hyperboloid 方程的 129 维 full-coordinate intent axis 时才变为 steering-capable，pocket records 仅在结构化 pocket geometry 可由 HUMU encoder 编码出 finite 且满足 Lorentz hyperboloid 方程的 129 维 embedding 且 `HUMU_ENCODER_TARGET` 可用时才变为 steering-capable；HFM-3D inference 已新增 W8-E `JMCGEngineeringSampler`，可从候选 molecule、route/property context 和 `moleculeforge.jmcg.feedback.v1` 构造 JSON-serializable `engineering_skeleton` joint sample，并用通过 Lorentz 合法性校验的 129 维 HUMU embedding 计算 alignment metadata；因此当前已有工程骨架和局部 feedback steering，但仍不是训练完成的 `(m,r,p)` 联合流形采样模型 | 严重偏差 |
| 1 CIC | LLM/SRM 解析，JSON-LD 有向超图 CIG，节点和边编码为 HCIV | 本地路径仍是规则/正则解析；production semantic parser adapter 已支持 `python://module:function`、HTTP/HTTPS JSON endpoint 和 `CIG_SEMANTIC_PARSER_COMMAND` stdin/stdout JSON command 外接 LLM/SRM parser，并在 command 配置后校验首个可执行文件可用；CIG compiler 已有 UniProt/PDB/ChEMBL grounding stage，并支持 `CIG_REFINEMENT_COMMAND` 外部 JSON refinement runner，refinement command 配置后同样进入 runtime executable preflight；`ChemicalIntentGraph` 和 schema 已有 JSON-LD `@context`、objective edges 与 directed hyperedges；HCIV learned encoder 已有 node/edge/hyperedge directed message-passing baseline，Owner A 已补 supervised train/export 工程路径，可从 `cig + target_hciv` 数据训练并导出兼容 `HCIV_CHECKPOINT_PATH` 的 checkpoint；`cig-compiler-svc` 的 Compose、静态 Kubernetes 清单和 Helm values/template 已暴露 semantic parser、refinement runner、`HCIV_CHECKPOINT_PATH` 和 UniProt/PDB/ChEMBL grounding endpoint ConfigMap 数据，真实外部值默认仍为空；hash encoder 仍保留本地 demo 路径 | 明显偏差 |
| 2 HUMU | SE(3) 分子编码、E(3) 口袋编码、图 Transformer + TreeLSTM 路线编码 | Lorentz 基础真实；预训练聚焦 mol/pocket/route 对齐，不接用户意图是合理边界；分子编码器已支持显式 3D conformer 坐标的 E(3)-invariant 距离统计增强，口袋编码器已用 E(3)-invariant 局部距离统计替代方向坐标，路线编码器已从 reaction token 统计增强到 route tree topology 统计；仍不是 SE(3) message passing、E(3)-GNN 或图 Transformer + 双曲 TreeLSTM | 部分符合，模型结构仍有差距 |
| 3 AMGE | 6 个生成范式共享 HUMU，TAR + KD 协同 | 生成器存在但多为简化路径；生成入口已接 intent cone；TAR 已接 HCIV/task/history 输入、REINFORCE-style 在线 policy 更新、ProxylessNAS-style architecture gate、期望资源代价接口、reward-cost architecture optimizer step 和多 dataset/多轮 `ProxylessSearchScheduler`，GeneratorRouterService 已新增 `RunProxylessSearch` gRPC 入口，可用请求内 reward batch 本地运行 scheduler，或通过 `TAR_PROXYLESS_SEARCH_COMMAND` 外接 JSON 训练 runner；同包已新增 `generator_router_svc.tar_proxyless_runner`，可作为 `python -m generator_router_svc.tar_proxyless_runner` 本地 command target 消费同一 payload 并输出 `rounds`、`architecture_probabilities` 和 `architecture_logits`；Compose、静态 Kubernetes 清单和 Helm values 已暴露 TAR search command/timeout env，GeneratorRouter runtime status 会在 TAR search command 配置时校验首个可执行文件可用；GeneratorRouterService feedback 已接入 CrossParadigmKDLayer oracle teacher score，并可通过 `HYPSEEK_TEACHER_COMMAND` 或 `HYPSEEK_TEACHER_URL` 调用外部 HypSeek teacher，配置 HypSeek command 时同样先校验首个可执行文件可用；同包已暴露可独立运行的 `generator_router_svc.main:hypseek_app` FastAPI teacher app；Compose、静态 Kubernetes 清单和 Helm values 已新增 `hypseek-teacher-svc`，把 `generator-router-svc` 的 `HYPSEEK_TEACHER_URL` 指向 `http://hypseek-teacher-svc:8012/teacher`，并暴露 `HYPSEEK_TEACHER_COMMAND` 与 `HYPSEEK_TEACHER_TIMEOUT_SECONDS`；Compose healthcheck 与 Kubernetes/Helm `/healthz` readiness/liveness probes 已接入，真实集群发布验证仍未执行；KD 层已能消费归一化 teacher distribution 和 teacher embedding target 计算 embedding distillation loss，并已有 Boltz2 ΔG/per-member ΔG 与 HypSeek 显式 score-field adapter 到 teacher_distribution 的转换；HFM-3D、FragFM 与 UAS training CLI 已可把该 loss 接入训练；CReM 和 MMPT training CLI 已可基于结构特征 embedding 计算 teacher embedding KD loss；iCLM service `UpdateModel` 已支持 `ICLM_UPDATE_COMMAND` 外部 JSON runner 承接 EWC/KD update 请求，并在 runtime status 和实际执行该 command 前校验首个可执行文件可用，未配置 command 时也可委托注入的 `generator.online_learner.update()` 并传递训练样本和 KD 参数；默认 OnlineLearner 本体已能在模型返回 student embedding 且 batch 带 `kd_teacher_embeddings` / `kd_weight` 时直接计算 teacher embedding MSE KD loss，service 会读回 learner 记录的 task/KD 指标 | 明显偏差 |
| 4 MARB | 多 Agent 通过 CRG 共享状态，消息协议 + Sigstore | LangGraph/CRG 骨架存在；orchestrator-svc 主链路已有 workflow state CRG，并随 provenance metadata 写入 Neo4j belief/edge；RETROSYN 阶段已委托 RetroSynAgent 规划路线，并支持 `assess_supply` 与 `compile_synthesis` client hook，FullWorkflowClients 已委托 RetroSynAgent、SupplyAgent 和 SRBAgent 生成 retrosyn/supply/srb state；GraphRepository 已支持 run_id 级 CRG 读回；orchestrator refinement 已能把 validation/critic 失败结果追加到 `generation_feedback` 并进入下一轮 generation，FullWorkflowClients 已把该反馈透传给默认 HFM 和显式 GeneratorCoordAgent generation 路径；OrchestratorAgent 已支持将 workflow_status belief 写入注入的 CRG repository，并读取同 run 的 completed workflow_status 返回 cached 工作流结果；NL2ObjAgent 已支持将 parsed_intent 与完整 compiled_cig JSON belief 写入注入的 CRG repository，并读取同 run 同 intent 的 compiled_cig belief 跳过重复 CIG compiler 调用；GeneratorCoordAgent 已支持将 selected_generators belief 写入注入的 CRG repository，并在 auto 路由时优先读取既有 selected_generators，也可读取 validation/critic/supply 失败 belief 触发探索型生成器组合，还可读取 route_humu_embedding belief 并写入下一轮 generator dispatch 的 `generator_params.route_humu_feedback`；ValidationAgent 已支持将 validation_status belief 写入注入的 CRG repository，并读取同 run 同分子既有 validation_status 跳过重复 oracle cascade；RetroSynAgent 已支持将 retrosyn_routes 与 route_humu_embedding belief 写入注入的 CRG repository，并读取同 run 同分子的 failed validation belief 或 `retrosyn_routes=0` 跳过路线规划；SupplyAgent 已支持将 supply_feasibility belief 写入注入的 CRG repository，并读取同 run 同分子的既有 supply_feasibility 跳过重复 supply oracle，也可读取同 run 的 `retrosyn_routes=0` 直接输出 unavailable；SRBAgent 已支持将 ssp_compiled belief 写入注入的 CRG repository，并读取同 run 同分子的 unavailable supply_feasibility 跳过 SSP 编译；CriticAgent 已支持将 critic_verdict belief 写入注入的 CRG repository，并读取同 run 同分子的既有 critic_verdict 跳过重复规则评估，也可读取 validation/supply 失败 belief 与 `retrosyn_routes=0` 影响 verdict；这些 agent 默认通过 `build_shared_crg_repository_from_env()` 复用同一 Neo4j 环境配置，并继承 `BaseAgent.read_shared_crg()`；重复执行与失败反馈级 CRG 读回已覆盖上述 agent，BaseAgent 已支持将 mapping payload 编码为 JSON-LD，发布带 signature 的 AgentMessage envelope，并默认生成 UUIDv7 message_id，订阅接收端可自动解包、校验 recipient、message_type、payload_type_url 与 ttl 防循环、按 sender identity 验签，可验证篡改，并可通过 SIGSTORE_SIGN_COMMAND/SIGSTORE_VERIFY_COMMAND 外接 Sigstore/Rekor 命令，签名命令可接收 SIGSTORE_IDENTITY_TOKEN，验证命令可接收 sender、recipient、message_type 与 expected_identity；后续仍缺更深层跨 agent 联合优化、真实 Fulcio/Rekor 命令、生产身份令牌投放和外部系统闭环；当前消息总线采用 Redis；provenance-svc 已支持 `SIGSTORE_SIGN_COMMAND` 外部 Sigstore/Rekor bundle 签名入口和 `SIGSTORE_VERIFY_COMMAND` 外部验证入口，签名命令可接收 `SIGSTORE_IDENTITY_TOKEN`，验证命令可接收 artifact_type、payload_hash 与 `SIGSTORE_EXPECTED_IDENTITY`，签名和验证命令均使用 `SIGSTORE_REKOR_URL`；Compose、静态 Kubernetes 清单和 Helm values/templates 已为 `provenance-svc` 暴露 Sigstore env、ConfigMap 与 Secret wiring；真实 command/identity 值和集群发布验证仍未完成；本地默认仍是 local dev 签名 | 明显偏差 |
| 5 Oracle/PCBO | L0-L4 自适应级联，HUMU 切空间 GP + EHVI/PoF | L0 真实；L1-L3 runner wrapper；admet-svc、dock-svc、boltz2-svc 和 fep-svc 已具备外部 JSON command runner，boltz2-svc 已可通过通用 `OracleService` 返回 L2 affinity score/uncertainty，fep-svc 已可通过通用 `OracleService` 返回 L3 RBFE score/uncertainty；Dock/Boltz2/FEP command 配置后会进入 executable preflight；L1-L3 oracle 服务部署清单已补 command/timeout env、本机 artifact/tool runtime env、Boltz2/FEP adapter 参数 env 与 ConfigMap wiring，静态 Kubernetes 清单和 Helm values/template 已在 `mf-oracles` 与 `mf-agents` 两个 namespace 声明 `oracle-runner-config` 数据；ValidationAgent 可通过外部 `L0_ADMET_ORACLE_TARGET`、`L1_DOCKING_ORACLE_TARGET`、`L2_AFFINITY_ORACLE_TARGET`、`L3_FEP_ORACLE_TARGET` 和 `L4_QUANTUM_ORACLE_TARGET` 接 L0-L4 OracleService，也可通过 `L4_QUANTUM_ORACLE_COMMAND`、`L4_GPU4PYSCF_COMMAND` 或 `L4_ORCA_COMMAND` 调用本地 JSON quantum wrapper，执行本地 L4 command 前会校验首个可执行文件可用；FullWorkflowClients 在请求显式提供 `oracle_level` / `max_oracle_level` / `validation_oracle_level` 时会委托 ValidationAgent 执行自适应 oracle cascade，默认仍保留 Boltz2 affinity gate；`orchestrator-svc` 部署清单已补 L4 quantum env wiring；PCBO 已有 HV、PoF、约束 HVI、批量 constrained HVI 候选排序、EHVI、HUMU log-map 切空间映射、tangent-space RBF GP constrained HVI/EHVI/PoF ranking、库级异步 oracle 采样循环、多轮 optimization scheduler、独立 `pareto_bo` package、callable-path CLI 入口、stdin/stdout JSON runner 接入点、FastAPI optimize endpoint 和 `pareto-bo-svc` Compose/Kubernetes/Helm wiring；真实 provider/oracle evaluator command/env 与生产验收仍需投放 | 明显偏差 |
| 6 SRB | 分子、路线、SSP、XDL、SiLA2 逻辑链路 | SSP 编译存在；SRBAgent protocol 输出已包含 XDL XML 和结构化 SiLA2 step plan；配置 `SILA2_PLAN_COMMAND` 时可把 SSP、XDL 和 SiLA2 plan 交给外部 SiLA2 adapter 并记录返回的 execution metadata；orchestrator-svc 的 Compose、静态 Kubernetes 清单和 Helm values 已暴露 `SILA2_PLAN_COMMAND` 与 `SILA2_PLAN_TIMEOUT_SECONDS`；真实 SiLA2 硬件 endpoint、adapter command 值和集群发布验证仍未完成 | 部分符合 |
| 7 DKI | CoreArchitecture v2 原文为 Milvus + Neo4j + PostgreSQL + MinIO + Feast + NATS | 当前正式采用 Qdrant + Neo4j + PostgreSQL + MinIO + Feast + Redis | 工程选型已确认，非补齐项 |
| 8 工程实施 | 核心服务、K8s、Helm、GPU/数据服务完整部署 | 目录较全；P0 核心 gRPC 服务已注册 servicer；部署配置存在但需实际验证 | 部分符合 |
| 9 评估体系 | MOSES / GuacaMol / PMO / CrossDocked / KRAS pilot | benchmark 已有资源门控入口；缺正式数据和模型时 skip；KRAS E2E 依赖 service ready、外部 DKI、provenance 和 Sigstore 生产环境 | 明显偏差 |

## 3. 第〇层：JMCG 联合流形共生成

### 文档要求

`MoleculeForge_CoreArchitecture_v2.md:30-46` 要求把分子结构 `m`、合成路径 `r`、性质轮廓 `p` 视为联合随机对象，在共享双曲流形 `H^d` 中建模：

```text
p(m,r,p | T,c) = integral p(m,r,p | z,T,c) q(z | T,c) dz
```

这意味着生成器、验证器和逆合成规划不能只是串行传递 SMILES，而要在 HUMU 坐标和反馈中相互约束。

### 当前实现

- `agents/orchestrator/src/orchestrator/pipeline.py:327` 的 generation 阶段使用 `_seed_pool` / RDKit 随机生成逻辑。
- `agents/orchestrator/src/orchestrator/pipeline.py:359` 的 scoring 阶段使用 `mf_chem.predict.get_default_engine()` 的局部预测。
- `agents/orchestrator/src/orchestrator/pipeline.py:450` 后的 ranking 主要基于 QED、SA、logP 等局部性质。
- `agents/orchestrator/src/orchestrator/workflow/graph_builder.py` 的 `REFINING` 节点已把失败的 validation 或 critic 结果追加到 `generation_feedback`；`services/orchestrator-svc/src/orchestrator_svc/main.py` 的 FullWorkflowClients 已把该反馈序列化为 `generator_params.generation_feedback`，并透传给默认 HFM 生成路径和显式 GeneratorCoordAgent dispatch。
- RetroSynAgent 已可通过 route encoder 将 retrosyn route 写回 HUMU embedding 并持久化到 CRG；`retrosyn-svc` 与 `orchestrator-svc` 的 Compose、静态 Kubernetes 清单和 Helm values 已暴露 `HUMU_ENCODER_TARGET`，可把独立 RetroSyn 服务和 orchestrator 内联 RetroSynAgent 指向 `humu-encoder-svc`；GeneratorCoordAgent 已能读取同 run 的 `route_humu_embedding` belief，并把解析后的 route HUMU feedback 作为 `generator_params.route_humu_feedback` 透传给下一轮 generator dispatch。
- `services/hfm-generator-svc/src/hfm_generator_svc/main.py` 会把 `GenerateRequest.generator_params` 原样展开传给 HFM-3D generator；`services/orchestrator-svc/src/orchestrator_svc/main.py` 目前只主动写入 `generation_feedback`，`agents/generator_coord/src/generator_coord/agent.py` 目前只主动写入 `route_humu_feedback`。
- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py` 的 `_feedback_steering_target()` 已遍历 `jmcg_feedback`、`route_humu_feedback` 和 `generation_feedback`；`_feedback_embedding_records()` 识别 `humu_embedding` / `route_humu_embedding`，并按 kind 聚合后投影为单个 Lorentz target。该路径仍是本地 post-flow latent steering，不是训练完成的 `(m,r,p)` 联合采样模型。
- 2026-06-03 已起草并部分落地 `moleculeforge/docs/todo/2026-06-03-jmcg-feedback-contract-brief.md`，定义未来 `moleculeforge.jmcg.feedback.v1` envelope、molecule/route/property/pocket/intent 反馈类型、legacy mapping、权重规则和非目标；默认 HFM 不直接接收 shared CRG route feedback 的架构决策已确认；HFM-3D 已兼容解析 `jmcg_feedback` envelope，GeneratorCoordAgent 已把 route HUMU feedback 同步封装为 contract-shaped records，同时保留 legacy `route_humu_feedback`。
- 2026-06-03 已补充 `moleculeforge/docs/todo/2026-06-03-jmcg-feedback-semantics-gate.md`，HFM-3D 本地 steering 已加入有效 embedding 维度检查、`weight * confidence`、`polarity`、per-kind aggregation、初始 kind weights 和 accepted/dropped metadata；该实现仍只属于本地 HUMU steering，不是 `(m,r,p)` 联合采样模型。
- 2026-06-03 已补充 `moleculeforge/docs/todo/2026-06-03-property-feedback-producer-gate.md`，FullWorkflowClients 已能把 workflow `generation_feedback` 派生为 non-steering `kind="property"` `jmcg_feedback` records，GeneratorCoordAgent 已能合并已有 property records 与 route HUMU records；这些 property records 不含 `humu_embedding`，因此不会触发 HFM-3D steering。
- 2026-06-03 已补充 `moleculeforge/docs/todo/2026-06-03-pocket-intent-feedback-producer-gate.md`，FullWorkflowClients 已能从 HCIV / intent cone context 派生 non-steering `kind="intent"` records，并从 CIG `target_context` pocket-related fields 派生 non-steering `kind="pocket"` records；这些 intent / pocket records 不含 `humu_embedding`，因此不会触发 HFM-3D steering。
- 2026-06-03 Owner A W2 已把 eligible pocket / intent records 升级为 evidence-backed steering records：intent 仅接受已有 finite 且满足 Lorentz hyperboloid 方程的 129 维 full-coordinate axis，pocket 仅在结构化 pocket geometry 可由 `HUMU_ENCODER_TARGET` 编码为 finite 且满足 Lorentz hyperboloid 方程的 129 维 HUMU embedding 时带 `humu_embedding`；metadata-only context 仍保持 non-steering。
- 2026-06-04 Owner A W8-E 已新增 `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/jmcg_sampler.py`，提供 `JMCGEngineeringSampler`、`JMCGJointSample` 和 `parse_jmcg_context()`；该模块可构造 JSON-serializable `moleculeforge.jmcg.joint_sample.v1` engineering skeleton output，校验 finite 且满足 Lorentz hyperboloid 方程的 129 维 HUMU embedding，支持 packed float32 `Molecule.humu_embedding` bytes，记录 alignment distances / ignored embedding counts，并保持默认 HFM generation behavior 不变。

### 偏差说明

当前主路径仍是线性流程，接近：

```text
NL intent -> seed molecules -> local scoring -> ranking
```

不是文档要求的联合流形共生成。当前已补到 validation 失败反馈向下一轮生成回注，并补到 route HUMU feedback 经 CRG 进入下一轮 generator dispatch；服务入口具备继续透传联合 feedback 的参数通道，但 HFM-3D 生成语义仍是局部 steering，不是 molecule / route / property 或 pocket 的联合流形采样。分子、路线和性质轮廓在共享 HUMU 坐标中的联合采样仍未实现。

## 4. 第一层：CIC / CIG / HCIV

### 文档要求

`MoleculeForge_CoreArchitecture_v2.md:50-176` 要求：

- 自然语言经 LLM/SRM 做科学实体抽取和知识锚定。
- CIG 是 JSON-LD 格式的有向超图，包含 objective nodes 和 constraint/preference edges。
- CIG 中节点和边经 `Enc_intent` 编码为 `HCIV in H^128_Lorentz`，并形成意图锥。

### 当前实现

- `agents/nl2obj/src/nl2obj/parser.py:1` 明确是纯 Python parser，使用 regex + heuristics，不需要 LLM 调用。
- `agents/nl2obj/src/nl2obj/parser.py:423` 的 `parse()` 返回固定结构，本地 parser 自身不包含 LLM 多轮澄清。
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/compiler.py:69` 的 production semantic parser adapter 已支持 `python://module:function`、HTTP/HTTPS JSON endpoint 与 `CIG_SEMANTIC_PARSER_COMMAND` stdin/stdout JSON command；HTTP/command 模式均传递 `{"text": ...}` 并要求返回 JSON object，可外接真实 LLM/SRM parser；command 模式会在 subprocess 前校验首个可执行文件可用；Compose、静态 Kubernetes 清单和 Helm values 已暴露 `CIG_SEMANTIC_PARSER_URI` / `CIG_SEMANTIC_PARSER_COMMAND` / timeout ConfigMap wiring。
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/stages/stage1b_grounding.py` 已接入 UniProt、PDB 和 ChEMBL grounding sources，并把 grounding evidence 写入 CIG target context；`cig-compiler-svc` 的 Compose、静态 Kubernetes 清单和 Helm values 已暴露 `UNIPROT_SEARCH_URL`、`RCSB_SEARCH_URL`、`CHEMBL_TARGET_URL` 和 `CHEMBL_TARGET_SEARCH_URL`。
- `services/cig-compiler-svc/src/cig_compiler_svc/main.py:112` 的 `Refine` RPC 已支持 `CIG_REFINEMENT_COMMAND` 外部 JSON command runner；`services/cig-compiler-svc/src/cig_compiler_svc/main.py:180` 请求向 runner 传入 `cig`、`feedback` 和 `context`，响应要求返回 refined `cig`、`hciv` 和 `intent_cone`；`runtime_status()` 会在 semantic parser / refinement command 配置时报告 executable 状态，并在执行 refinement subprocess 前校验首个可执行文件可用；部署清单已暴露 `CIG_REFINEMENT_COMMAND`、timeout 和 `HCIV_CHECKPOINT_PATH` ConfigMap wiring。
- `agents/nl2obj/src/nl2obj/agent.py` 已复用 `nl2obj.parser.parse()`，返回 parsed intent、confidence、targets、activity、constraints、ADMET 和 synthetic constraints；空 intent 会 fail-fast；注入 `cig_compiler_client` 或配置 `CIG_COMPILER_TARGET` 时会调用 `cig-compiler-svc` 产出 CIG、HCIV 和 intent cone。
- `libs/mf-core/src/mf_core/types/cig.py:100` 的 `ChemicalIntentGraph` 已有 JSON-LD `@context`、`edges: list[ObjectiveEdge]` 和 `hyperedges: list[ObjectiveHyperedge]` 字段。
- `schemas/cig.schema.json` 已声明 JSON-LD `@context`、objective `edges` 和 directed `hyperedges`。
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/stages/stage2_cig_build.py:82` 会在 affinity 与 ADMET bundle 同时存在时生成 `trade_off` objective edge。
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_encoder.py:11` 的 `cig_to_features()` 已消费 objective edge 数量、平均强度和 trade-off 数量，作为全局 feature 输入。
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_encoder.py:35` 的 `HCIVEncoder` 已包含 objective node encoder、directed edge encoder、directed hyperedge encoder 和 message-passing 聚合，能区分聚合统计相同但方向相反的 CIG 拓扑。
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/compiler.py:252` 在 production 模式要求 `HCIV_CHECKPOINT_PATH`，缺失时直接 RuntimeError。
- `uv run pytest tests/unit/test_cic_compiler.py -q` 退出码为 0，28 项通过，覆盖 CIG JSON-LD context、objective edges/hyperedges、schema、grounding、refinement command、production HCIV checkpoint fail-fast、HCIV feature generation 和 directed topology sensitivity。

### 偏差说明

CIC 有编译器骨架和 production fail-fast 机制，并已补齐 JSON-LD `@context`、objective edges 与 directed hyperedges 的最小链路。CIG 已具备有向超图数据结构，production adapter 已能外接 Python、HTTP 或 stdin/stdout command LLM/SRM parser，grounding stage 已能写入 UniProt/PDB/ChEMBL evidence，`Refine` RPC 已能通过外部 JSON runner 执行澄清后的 CIG refinement，semantic parser / refinement command 配置后均会做 executable preflight，HCIV learned encoder 已具备 node/edge/hyperedge directed message-passing baseline，并已补本地 supervised train/export 工程路径；`cig-compiler-svc` 部署清单已暴露 semantic parser、refinement runner、grounding endpoint 和 `HCIV_CHECKPOINT_PATH` 配置入口；但仓库内置 LLM/SRM 模型、真实 refinement runner command/env、真实 supervised CIG/HCIV 数据和训练好的 `Enc_intent` checkpoint 值仍未完成。

## 5. 第二层：HUMU 双曲统一分子宇宙

### 文档要求

`MoleculeForge_CoreArchitecture_v2.md:180-256` 要求：

- 使用 `H^128` Lorentz 双曲空间。
- 分子编码器：3D 分子图 + SE(3)-Equivariant Message Passing。
- 口袋编码器：点云 + EquiBind-style E(3)-GNN。
- 路线编码器：反应树 AND-OR 图 + 双向图 Transformer + 双曲 TreeLSTM。
- 联合对比损失训练 mol-pocket、mol-route，并带可学习曲率正则。

该层的预训练目标是建立统一嵌入空间。预训练数据本身没有用户自然语言输入，也没有任务级 CIG/HCIV，因此预训练阶段不引入意图锥是合理设计，不应作为需要修复的偏差。意图锥属于后续生成阶段：用户输入经 CIC 得到 HCIV 后，再用 HCIV 约束 AMGE 的采样和路由。

### 当前实现中的符合项

- `libs/mf-humu/src/mf_humu/manifold/lorentz.py:14` 有 Lorentz manifold 基础实现。
- `libs/mf-humu/src/mf_humu/manifold/learnable_lorentz.py:10` 有可学习曲率参数。
- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py:872` 后的 `_compute_losses()` 包含 mol-pocket、mol-route、pocket-route、intent、curvature regularization 等训练 loss。
- `configs/models/humu_pretrain.yaml:63` 当前实际训练权重为 `mol_pocket: 1.0`、`mol_route: 0.5`、`pocket_route: 0.3`、`intent: 0.0`、`curvature_reg: 0.01`，符合“预训练先学 mol/pocket/route 统一空间”的阶段目标。
- `/workspace/MForge/moleculeforge/checkpoints/humu/best_model.pt` 存在 HUMU checkpoint 文件；`humu-encoder-svc` 的 Compose、静态 Kubernetes 清单和 Helm values/template 已暴露 `HUMU_CHECKPOINT_PATH` 与 `HUMU_DEVICE` 配置，Compose 与 ConfigMap 默认指向该本地 checkpoint 并使用 `cpu` device。

### 当前实现与最终 HUMU 结构的差距

以下差距是“最终 HUMU 设想结构”的后续提升项，不作为当前阶段对 HUMU 预训练的修改要求。当前 HUMU 预训练已经在进行，应保持现有 pipeline、配置、loss 和 checkpoint 产出路径稳定。

- `models/mf-encoders/humu_mol_encoder/src/mf_encoders/humu_mol/encoder.py:60` 从 SMILES/RDKit 构造图特征；`models/mf-encoders/humu_mol_encoder/src/mf_encoders/humu_mol/encoder.py:118` 在输入 dict 携带 `coords` / `coordinates` 时加入平移/旋转不变的 3D 距离统计，并且 `services/humu-encoder-svc/src/humu_encoder_svc/main.py:121` 会保留 molecule `input_data` 中的 3D 坐标；这仍不是 3D SE(3) 等变 message passing。
- `models/mf-encoders/humu_mol_encoder/src/mf_encoders/humu_mol/encoder.py:112` 使用归一化邻接传播，不是 SEGNN。
- `models/mf-encoders/humu_pocket_encoder/src/mf_encoders/humu_pocket/encoder.py:304` 使用 centered point cloud 的 pairwise distance、radial distance、mean/max/min neighbor distance 以及 element/residue 特征；该特征对平移和旋转不变，但仍不是 EquiBind-style E(3)-GNN。
- `models/mf-encoders/humu_route_encoder/src/mf_encoders/humu_route/encoder.py:67` 从 reaction SMILES token、反应物/产物计数和 route tree topology 构造路线特征；`models/mf-encoders/humu_route_encoder/src/mf_encoders/humu_route/encoder.py:125` 解析 `steps` 中的 parent/children 关系得到 step count、branching edges、leaf count 和 max depth，但仍不是反应树图 Transformer + 双曲 TreeLSTM。
- `libs/mf-humu/src/mf_humu/encoders/lorentz_attention.py:29` 使用普通 `nn.Linear` projection，`libs/mf-humu/src/mf_humu/encoders/lorentz_attention.py:59` 使用普通 dot-product softmax，不是严格 Lorentz-equivariant attention。

### 偏差说明

HUMU 的数学基座和预训练 pipeline 已经存在。当前预训练不接用户意图、HCIV 或意图锥不是问题；这是预训练阶段的数据边界。`humu-encoder-svc` 已具备 checkpoint/device 部署配置入口；molecule encoder 已能消费显式 3D conformer 坐标的距离统计，route encoder 已能消费路线树拓扑统计，但最终 HUMU 的 SE(3)/E(3) 等变网络、反应树图 Transformer 和双曲 TreeLSTM 结构仍保留为后续提升。

## 6. 第三层：AMGE 自适应多范式生成引擎

### 文档要求

`MoleculeForge_CoreArchitecture_v2.md:260-500` 要求：

- HFM-3D、FragFM、CReM、MMPT-RAG、iCLM、UAS 等生成范式共享 HUMU 潜空间。
- 从 HCIV 意图锥采样。
- TAR 根据 HCIV、任务画像和 oracle 历史动态路由。
- Cross-Paradigm KD 用 Boltz-2 / HypSeek 等 teacher 信号给生成器提供跨范式蒸馏。

### 当前实现

- 当前 generator 目录有 7 个：`hfm_3d`、`fragfm`、`crem_3d`、`incremental_clm`、`mmpt_rag`、`rdkit_random`、`uas`。
- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py:115` 的 `generate()` 从 latent 采样，在 flow 后调用 `_decode_molecule()`。
- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py:429` 的 decoder artifact 默认仍按 latent 最近邻检索 decoder entries；`models/mf-generators/hfm_3d/train.py:546` 写出的新 decoder artifact 已可为每个 entry 持久化 RDKit SDF 几何，生成器载入 artifact 时会校验 `sdf` 可被 RDKit 解析且与 entry SMILES 匹配，生成阶段会优先复用 entry 内的 `sdf`，缺失时才回退到运行时 conformer。
- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py:482` 已支持注入 `molecular_decoder`，decoder 可消费 post-flow latent 并返回 `smiles`、`atom_types`、`coordinates` 或 `sdf_bytes`；配置 `HFM_MOLECULAR_DECODER_COMMAND` 或注入 `ExternalMolecularDecoder` 时可通过 stdin/stdout JSON command 调用外部分子几何 decoder，并在执行前校验首个可执行文件可用；`models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py:566` 会把 decoder 返回的 atom coordinates/types 直接写入 SDF，未提供几何时才回退到 RDKit conformer。
- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/model/lorentz_flow_matching.py:7` 有 Lorentz flow matching 模块；`models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py:159` 的推理路径已在采样 latent 后调用 `compute_vector_field()` 并通过 Lorentz expmap 做 flow steps，再通过 `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py:262` 消费 `route_humu_feedback` 或含 HUMU embedding 的 `generation_feedback`，并在 `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py:334` 用有界 Lorentz tangent step 把 post-flow latent steering 到更接近反馈 HUMU embedding 的位置，最后进入 decoder。
- `models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py:140` 进入 fragment assembly 生成路径，`models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py:187` 对规则排序，`models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py:194` 使用 fragment vocabulary，`models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py:196` 使用 SA-aware rate matrix，`models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py:200` 计算 rule-level `humu_embedding` / `intent_cone` 对齐分数；已支持注入共享 HUMU latent sampler，并用采样 latent 对齐规则排序和记录 `humu_latent` metadata；`services/fragfm-generator-svc/src/fragfm_generator_svc/main.py:115` 已在服务构造时注入基于 intent cone 的共享 HUMU latent sampler；training CLI 已能产出 vocabulary/model/rate-matrix artifact 并接入 teacher embedding KD loss；`fragfm-generator-svc` 的 Compose 已默认指向当前本地 `checkpoints/fragfm/vocab.json`、`checkpoints/fragfm/best_model.pt` 与 `checkpoints/fragfm/rate_matrix.pt`，静态 Kubernetes 清单和 Helm values/template 已声明同一组 vocabulary、checkpoint、rate-matrix artifact path 与 HUMU curvature ConfigMap 数据；当前本地 FragFM vocabulary/checkpoint/rate-matrix artifact 已存在，生产质量验收和集群发布验证仍未完成。
- `models/mf-generators/crem_3d/src/mf_generators/crem_3d/generator.py:118` 基于 MMP database 记录循环突变，已支持可注入 docking scorer 对突变候选做实时 docking score 记录和重排；已补 `DockOracleGrpcScorer` 对接通用 `OracleService` L2 docking/DiffDock-L 批量评估并按 docking score 重排；同时已支持注入式 3D pharmacophore scorer 按 `pharmacophore_score` 重排，以及 HUMU embedding scorer 按 intent cone alignment 重排并写入 `Molecule.humu_embedding`；`services/crem-generator-svc/src/crem_generator_svc/main.py` 已可通过 `CREM_DOCK_ORACLE_TARGET` 装配 docking scorer，并通过 `CREM_PHARMACOPHORE_SCORER_COMMAND` / `CREM_HUMU_SCORER_COMMAND` 调用 stdin/stdout JSON score provider；runtime status 和实际 scorer subprocess 执行前都会校验首个可执行文件可用；`models/mf-generators/crem_3d/train.py` 已支持 `--kd-teacher-embeddings` / `--kd-weight`，基于归一化 mutation 的结构特征 embedding 计算 KD loss 并写入 manifest；`crem-generator-svc` 的 Compose 已默认指向当前本地 `models/artifacts/crem/crem_mmp_database.json`，静态 Kubernetes 清单和 Helm values/template 已声明同一 MMP DB path、dock oracle target、pharmacophore/HUMU scorer command 与 timeout ConfigMap 数据；真实 DiffDock-L / pharmacophore / HUMU scorer runner 值和集群发布验证仍未完成。
- `models/mf-generators/mmpt_rag/src/mf_generators/mmpt_rag/generator.py:106` 默认 MMP 数据是很小的内置变换；加载 MMPT index artifact 时会保留 retrieved `seed_smiles` / `product_smiles` evidence，并优先用该 evidence 生成候选；`models/mf-generators/mmpt_rag/src/mf_generators/mmpt_rag/generator.py:198` 已支持 `patent_negative_smiles` / transform `negative_smiles` exact 过滤，`models/mf-generators/mmpt_rag/src/mf_generators/mmpt_rag/generator.py:202` 按 transform `positive_smiles`、`negative_smiles`、`retrieval_score` 做 contrastive ranking；配置 `MMPT_PATENT_RAG_COMMAND` 或注入 `patent_retriever` 时可调用外部 stdin/stdout JSON patent RAG retriever，并把返回 transforms 纳入同一 contrastive ranking 链路；配置 `MMPT_SEQ2SEQ_DECODER_COMMAND` 或注入 `seq2seq_decoder` 时可调用外部 stdin/stdout JSON Seq2Seq decoder，用 decoder 返回的 SMILES 覆盖默认字符串替换；两个模型级 external command wrapper 均会在执行前校验首个可执行文件可用；`models/mf-generators/mmpt_rag/train.py` 已支持 `--kd-teacher-embeddings` / `--kd-weight`，基于 MMP transform 结构特征 embedding 计算 KD loss 并写入 manifest；`mmpt-generator-svc` 的 Compose 已默认指向当前本地 `file:///workspace/models/artifacts/mmpt/mmpt_index.json`，静态 Kubernetes 清单和 Helm values/template 已声明同一 MMPT index URI、patent RAG command、Seq2Seq decoder command 及 timeout ConfigMap 数据；artifact preflight 对本地 `file://` URI 已校验实际文件存在，避免缺文件时只因 URI 语法正确而误报可用；显式 `MMPT_INDEX_URI` 只接受当前实际支持的本地 `file://` URI，非 `file://` 或非本地 file URI 会 fail-fast；runtime status 会在 patent RAG / Seq2Seq decoder command 被配置时校验首个可执行文件可用；真实 Seq2Seq Transformer artifact、真实专利 RAG 检索服务、生产 command 值和集群发布验证仍未完成。
- `models/mf-generators/uas/src/mf_generators/uas/generator.py:87` 调用 OOD sampler，`models/mf-generators/uas/src/mf_generators/uas/generator.py:94` 计算 autoencoder reconstruction loss，`models/mf-generators/uas/src/mf_generators/uas/sampler/ood_aware_sampling.py:33` 后把 unfamiliarity 转换为 `p_safe(z)` 并按最小安全概率筛选，生成结果会在 `models/mf-generators/uas/src/mf_generators/uas/generator.py:101` 记录 `uas_safety_probability`。
- `uv run pytest tests/unit/test_phase_b_generators.py -q` 退出码为 0，7 项通过，覆盖 iCLM 缺 model/runner fail-fast、iCLM online learner EWC/PackNet/KD、UAS unfamiliarity filter、CReM fragment replacement 和 FragFM vocabulary + SA-aware rate matrix 本地生成路径。
- `hfm-generator-svc`、`fragfm-generator-svc`、`crem-generator-svc`、`iclm-svc`、`mmpt-generator-svc` 已把请求中的 `intent_cone` 传给底层 generator；`orchestrator-svc` 的 FullWorkflowClients 默认仍直连 HFM，但请求显式提供 `generation_strategy` 且不等于 `hfm_3d` 时会委托 GeneratorCoordAgent，由其按 discovery/target env 调度多 generator；`hfm-generator-svc` runtime gate 已支持 `HFM_CHECKPOINT_PATH + HFM_DECODER_PATH` 或 `HFM_CHECKPOINT_PATH + HFM_MOLECULAR_DECODER_COMMAND` 两种生产输入组合，并会校验 external decoder command 的首个可执行文件可用；Compose 已默认指向现有本地 `checkpoints/hfm3d_4h200/best_model.pt` 与 `checkpoints/hfm3d_4h200/decoder.json`，静态 Kubernetes 清单和 Helm values/template 已声明同一 checkpoint、decoder artifact 和 molecular decoder command ConfigMap 数据；Compose、静态 Kubernetes 清单和 Helm values/template 也已为 `fragfm-generator-svc` 声明 vocabulary/checkpoint/rate-matrix artifact 与 HUMU sampler ConfigMap 数据，为 `crem-generator-svc` 声明 MMP DB、dock oracle target、pharmacophore scorer 和 HUMU scorer ConfigMap 数据，为 `mmpt-generator-svc` 声明 MMPT index、patent RAG runner 与 Seq2Seq decoder runner ConfigMap 数据，并为 `iclm-svc` 声明本地 iCLM model path、device 与 update command/timeout ConfigMap 数据；`orchestrator-svc` 部署清单已暴露 GeneratorCoordAgent 所需的 discovery、client-target JSON、已存在 generator 服务 target env 和 UAS Python client target，UAS 会通过 `python://generator_coord.agent:create_uas_generator_client` 调用本地 client factory，并在缺 `UAS_RUNNER_COMMAND` 或 runner executable 不可用时 fail-fast。
- `protos/moleculeforge/v1/generator/router.proto` 的 `RouterRequest` 已补齐 `hciv`、`target_family`、`stage`、`data_richness`、`novelty_demand`、`multi_target`、`sa_constraint`、`n_samples`。
- `services/generator-router-svc/src/generator_router_svc/main.py` 的 `Route()` 已使用请求中的 HCIV、task profile 和 `generator_performance`。

### TAR 和 KD 偏差

- `libs/mf-core/src/mf_core/routing/task_router.py:21` 的 `GENERATOR_NAMES` 是 6 个生成器，不含 `rdkit_random`。
- `libs/mf-core/src/mf_core/routing/task_router.py:98` 后的 route 逻辑已使用 HCIV、task profile feature、hard rules、running history、policy logits 和 `architecture_logits`。
- `libs/mf-core/src/mf_core/routing/task_router.py` 已提供 ProxylessNAS-style architecture probability gate、`proxyless_expected_cost()` 期望资源代价接口、`proxyless_architecture_optimizer_step()` reward-cost 更新步骤和 `ProxylessSearchScheduler` 多 dataset/多轮搜索调度器，architecture logits 会直接参与 forward routing 权重；`protos/moleculeforge/v1/generator/router.proto` 已新增 `RunProxylessSearch` gRPC，`services/generator-router-svc/src/generator_router_svc/main.py` 会用请求内 reward batch 本地运行 scheduler，或在配置 `TAR_PROXYLESS_SEARCH_COMMAND` 时把同一 payload 交给外部 JSON training runner；`services/generator-router-svc/src/generator_router_svc/tar_proxyless_runner.py` 已提供本地 command target，可用 `python -m generator_router_svc.tar_proxyless_runner` 从 stdin JSON 运行同一 scheduler 并输出 service-compatible JSON；Compose、静态 Kubernetes 清单和 Helm values 已暴露 TAR search command/timeout env。
- `libs/mf-core/src/mf_core/routing/task_router.py:191` 的 `update_with_feedback()` 已在 running mean 之外加入 REINFORCE-style advantage policy logit 更新。
- `libs/mf-core/src/mf_core/routing/cross_paradigm_kd.py:47` 有 KD layer。
- `libs/mf-core/src/mf_core/routing/cross_paradigm_kd.py` 已支持 `teacher_distribution`、teacher embedding target、Boltz2 ΔG/per-member ΔG adapter 和 HypSeek 显式 score-field adapter；`services/generator-router-svc/src/generator_router_svc/main.py` 已暴露可独立运行的 `hypseek_app` FastAPI teacher app，`/teacher` 会把 `oracle_feedback` score records 转换为 HypSeek `teacher_distribution`，`/healthz` 可用于运行时探测；`infra/docker/docker-compose.dev.yml`、`infra/kubernetes/deployments/moleculeforge-services.yaml`、`infra/helm/moleculeforge/values.yaml` 和 Helm service template 已配置 `hypseek-teacher-svc`、router 侧 `HYPSEEK_TEACHER_URL`、`HYPSEEK_TEACHER_COMMAND`、`HYPSEEK_TEACHER_TIMEOUT_SECONDS`、Compose healthcheck 和 Kubernetes/Helm readiness/liveness probes。
- `uv run pytest tests/unit/test_cross_paradigm_kd.py -q` 退出码为 0，16 项通过，覆盖 CrossParadigmKDLayer oracle feedback、Boltz2/HypSeek teacher distribution adapter、teacher embedding target distillation loss 和 generator quality ranking。
- `services/generator-router-svc/src/generator_router_svc/main.py` 的 `GeneratorRouterServicer` 已持有 `CrossParadigmKDLayer`，`SubmitFeedback()` 可消费 oracle teacher records，更新 KD teacher score，并把 teacher score 写入 TAR feedback；配置 `HYPSEEK_TEACHER_COMMAND` 时会把 generator、reward 和已有 oracle feedback 通过 stdin JSON 交给外部 HypSeek teacher runner，配置 `HYPSEEK_TEACHER_URL` 时会通过 HTTP JSON endpoint 调用外部 HypSeek teacher service，并消费其返回的 `teacher_distribution` 或 `normalized_score`；`HYPSEEK_TEACHER_COMMAND` 与 `TAR_PROXYLESS_SEARCH_COMMAND` 配置后会纳入 GeneratorRouter runtime status，并在 subprocess 执行前校验首个可执行文件可用；`models/mf-generators/hfm_3d/train.py`、`models/mf-generators/fragfm/train.py`、`models/mf-generators/uas/train.py`、`models/mf-generators/crem_3d/train.py` 与 `models/mf-generators/mmpt_rag/train.py` 已支持 `--kd-teacher-embeddings` / `--kd-weight` 把 teacher embedding distillation loss 接入训练或 artifact 构建链路；`services/iclm-svc/src/iclm_svc/main.py` 的 `UpdateModel` 已支持 `ICLM_UPDATE_COMMAND` 外部 JSON runner，向 runner 传递 `ICLM_MODEL_PATH`、`ICLM_DEVICE`、训练样本和 KD 参数，并消费返回的 checkpoint/EWC/KD 指标；`ICLM_UPDATE_COMMAND` 配置后会纳入 runtime status，并在更新路径执行前校验首个可执行文件可用；未配置 command 时，`UpdateModel` 可调用注入 generator 的 `online_learner.update()` 并传递同一训练/KD payload；默认 iCLM OnlineLearner 本体已能直接计算 teacher embedding KD loss，并记录 `last_task_loss` / `last_kd_loss` 供 service response 使用；`iclm-svc` 的 Compose 已默认指向当前本地 `models/artifacts/iclm/novomolgen_157m_smiles_bpe`，静态 Kubernetes 清单和 Helm values 仍暴露 model path、device、update command 与 timeout ConfigMap wiring。

### 偏差说明

AMGE 的目录、服务和部分算法模块存在，生成服务入口已传递 intent cone，TAR 输入链路已接入 HCIV、任务画像、历史表现、ProxylessNAS-style architecture gate、期望资源代价、architecture optimizer step、多 dataset/多轮 search scheduler、gRPC search 入口和外部 training runner env；但还没有形成文档要求的共享 HUMU 多范式协同生成。当前实现更接近“多个独立 generator wrapper + 可学习 router”。

## 7. 第四层：MARB 多智能体推理大脑

### 文档要求

`MoleculeForge_CoreArchitecture_v2.md:504-822` 要求：

- Orchestrator、NL2Obj、Generator Coordinator、RetroSyn、Validation、Supply、Scientific Critic 等 agent 通过 CRG 共享状态。
- Agent 消息是 JSON-LD + Sigstore 签名。
- LangGraph 状态机串起 `nl2obj -> humu_encode -> generate -> validate -> retrosyn -> critic -> orchestrate -> refine`。
- 消息总线使用 NATS JetStream。

### 当前实现中的符合项

- `agents/orchestrator/src/orchestrator/workflow/graph_builder.py:42` 在 `WorkflowGraph.build()` 中通过 `_langgraph_symbols()` 懒加载 LangGraph `StateGraph`。
- `agents/orchestrator/src/orchestrator/workflow/graph_builder.py` 已在 workflow state 中维护可序列化 CRG；每个主阶段写入 `workflow_stage` belief，并用 `derives_from` edge 串联阶段顺序。
- `services/orchestrator-svc/src/orchestrator_svc/main.py` 已选定为 CoreArchitecture v2 主编排入口；`engineering` 和 `full` scope 会绑定默认 workflow clients。
- `services/orchestrator-svc/src/orchestrator_svc/main.py:282` 的 `FullWorkflowClients` 串起 CIG compile、generation、validation、RetroSynAgent retrosyn、SupplyAgent supply assessment、SRBAgent SSP compilation、critic 和 provenance 记录；`services/orchestrator-svc/src/orchestrator_svc/main.py:283` 的 generation 默认路径保留 HFM，并会把 workflow state 的 `generation_feedback` 序列化为 `generator_params.generation_feedback` 传给 HFM，请求显式提供 `generation_strategy` 且不等于 `hfm_3d` 时会委托 GeneratorCoordAgent 执行多 generator dispatch 并透传同一反馈参数；默认 validation 路径保留 Boltz2 affinity gate，请求显式提供 `oracle_level` / `max_oracle_level` / `validation_oracle_level` 时会委托 ValidationAgent 执行 L0-L4 自适应 oracle cascade；route planning 由 RetroSynAgent 执行，可复用 `RETROSYN_PLANNER_COMMAND`、`RETROSYN_PLANNER_COMMANDS_JSON` 和 RAscore/RSGPT/UAlign/AiZynth 命名 planner env；`services/orchestrator-svc/src/orchestrator_svc/main.py:379` 和 `services/orchestrator-svc/src/orchestrator_svc/main.py:392` 分别接入 SupplyAgent supply assessment 与 SRBAgent SSP compilation；`retrosyn-svc` 对 `RETROSYN_PLANNER_COMMAND`、`RETROSYN_PLANNER_COMMANDS_JSON` 内的 planner command 和命名 planner command 已做 runtime/startup executable preflight，并通过 `HUMU_ENCODER_TARGET` 调用 route HUMU encoder；当 supply assessment 明确返回 `overall_feasibility=unavailable` 时会跳过 SRB 编译；由于这些 client 在 orchestrator 进程内直接构造，`orchestrator-svc` 的 Compose、静态 Kubernetes 清单和 Helm values 已同步暴露内联 GeneratorCoordAgent target env、Boltz2 runner/artifact env、ValidationAgent L4 quantum env、RetroSyn planner env、route HUMU encoder target、`SUPPLY_ORACLE_TARGET` 和 SiLA2 adapter env。
- 2026-06-03 补充核对与修正：`FullWorkflowClients.assess_supply()` 在 `retrosyn.routes` 为空时会返回 `overall_feasibility=unavailable` 的 supply 结果，`FullWorkflowClients.compile_synthesis()` 在无 route 时会返回 skipped，避免 no-route retrosyn 结果在 full workflow hook 链中直接异常；本次补充未运行测试。
- `services/orchestrator-svc/src/orchestrator_svc/main.py:518` 的 workflow provenance metadata 已包含 `crg`、`crg_belief_count`、`crg_edge_count`。
- `services/provenance-svc/src/provenance_svc/main.py:476` 会把 provenance metadata 内的 workflow CRG 写入图仓库。
- `libs/mf-core/src/mf_core/db/repositories/graph_repo.py:95`、`libs/mf-core/src/mf_core/db/repositories/graph_repo.py:137` 和 `libs/mf-core/src/mf_core/db/repositories/graph_repo.py:160` 已提供 `write_workflow_belief()`、`write_crg_edge()` 和 `get_run_crg()`，可将 workflow belief 与 `derives_from` edge 持久化为 Neo4j 节点和关系，并按 run_id 读回 workflow CRG。
- 2026-06-03 补充核对确认：`services/orchestrator-svc/src/orchestrator_svc/main.py:518` 写入 provenance metadata 的 `final_state["crg"]` 当前主要来自 orchestrator workflow stage belief；各 agent 直接写入 shared `GraphRepository` 的 belief 不会在 provenance record 创建前自动合并回 `final_state["crg"]`。因此 workflow provenance metadata CRG 和 Neo4j shared CRG repository 需要按两个相关但不完全等同的状态面理解。
- `uv run pytest tests/unit/test_graph_repo.py -q` 退出码为 0，11 项通过，覆盖 GraphRepository 本地 repository 行为；真实 Neo4j 环境仍由 DKI 集成测试在外部环境变量投放后验证。
- `libs/mf-agents/src/mf_agents/crg/graph.py:14` 有本地 CRG 图结构；2026-06-03 已让 `add_edge()` 与 `add_belief()` / `update_belief()` 一样递增 `CRG.version`，保持本地 CRG 版本语义一致。
- `libs/mf-core/src/mf_core/types/crg.py:23` 有 CRG 类型定义。
- `libs/mf-agents/src/mf_agents/base/agent.py` 已支持 `publish_agent_message()` 生成带 `signature` 的 `AgentMessage` envelope，并支持 `verify_agent_message()` 对 payload 篡改返回失败。
- `libs/mf-agents/src/mf_agents/base/agent.py` 已提供 `ensure_default_event_loop()`；`agents/nl2obj/src/nl2obj/agent.py`、`agents/supply_agent/src/supply_agent/agent.py`、`agents/generator_coord/src/generator_coord/agent.py`、`agents/retrosyn_agent/src/retrosyn_agent/agent.py` 和 `agents/validation_agent/src/validation_agent/agent.py` 的同步 gRPC client 构造路径会先保证默认 event loop 存在。
- `agents/critic_agent/src/critic_agent/agent.py:20` 后初始化 CRG repository，仓库中存在规则文件，`services/critic-svc/src/critic_svc/main.py:57` 会懒加载 `ScientificCriticAgent`。
- `agents/srb_agent/src/srb_agent/agent.py:24` 后定义 `SRBAgent` 并在处理流程中调用 SSP 编译。

### 当前实现中的偏差

- `agents/orchestrator/src/orchestrator/workflow/graph_builder.py:64` 后的节点仍主要通过 `clients` hook 连接外部服务；RETROSYN 阶段在 `plan_routes` 后支持可选 `assess_supply` 和 `compile_synthesis` hook，并把返回值写入 workflow state 的 `supply` 与 `srb`；workflow CRG 已写入 provenance metadata，可持久化到 Neo4j belief/edge，并可通过 `GraphRepository.get_run_crg()` 按 run_id 读回；LangGraph refinement 已能把 validation/critic 失败反馈写入 `generation_feedback` 并回到下一轮 generation，FullWorkflowClients 已把该反馈作为 `generator_params.generation_feedback` 交给默认 HFM 路径或显式 GeneratorCoordAgent dispatch；`OrchestratorAgent` 已能把 workflow_status belief 写入注入的 CRG repository，并读取同一 run 的 `workflow_status=completed` belief 直接返回 cached 工作流结果；`NL2ObjAgent` 已能把 parsed_intent 与完整 compiled_cig JSON belief 写入注入的 CRG repository，并读取同一 run 同一 intent 的 compiled_cig belief 直接复用 CIG/HCIV/intent_cone；`GeneratorCoordAgent` 已能把 selected_generators belief 写入注入的 CRG repository，auto 策略会优先读取同一 run 的既有 `selected_generators`，也会在无显式 complexity 时读取 `validation_status=failed`、`critic_verdict=fail` 或 `supply_feasibility=unavailable` belief 选择 `mmpt_rag,fragfm` 探索型生成器组合；`ValidationAgent` 已能把 validation_status belief 写入注入的 CRG repository，并读取同一 run 中同一分子的既有 `validation_status` belief 跳过重复 oracle cascade；`RetroSynAgent` 已能把 retrosyn_routes 与 route_humu_embedding belief 写入注入的 CRG repository，并读取同一 run 中同一分子的 `validation_status=failed` 或 `retrosyn_routes=0` belief 跳过路线规划；`SupplyAgent` 已能把 supply_feasibility belief 写入注入的 CRG repository，并读取同一 run 中同一分子的既有 `supply_feasibility` belief 跳过重复 supply oracle，也可读取同一 run 的 `retrosyn_routes=0` belief 直接输出 unavailable；`SRBAgent` 已能把 ssp_compiled belief 写入注入的 CRG repository，并读取同一 run 中同一分子的 `supply_feasibility=unavailable` belief 跳过 SSP 编译；`CriticAgent` 已能把 critic_verdict belief 写入注入的 CRG repository，并读取同一 run 中同一分子的既有 `critic_verdict` belief 跳过重复规则评估，也可读取 `validation_status=failed`、`supply_feasibility=unavailable` 或 `retrosyn_routes=0` belief 纳入 fail verdict；上述 agent 默认通过 `build_shared_crg_repository_from_env()` 复用 `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` 配置，并继承 `BaseAgent.read_shared_crg()` 读回接口，重复执行与失败反馈级 CRG 读回已覆盖上述 agent，BaseAgent 已支持将 mapping payload 编码为 JSON-LD，发布带 signature 的 AgentMessage envelope，并默认生成 UUIDv7 message_id，订阅接收端可自动解包、校验 recipient、message_type、payload_type_url 与 ttl 防循环、按 sender identity 验签，可验证篡改，并可通过 SIGSTORE_SIGN_COMMAND/SIGSTORE_VERIFY_COMMAND 外接 Sigstore/Rekor 命令，签名命令可接收 SIGSTORE_IDENTITY_TOKEN，验证命令可接收 sender、recipient、message_type 与 expected_identity；后续仍缺更深层跨 agent 联合优化、真实 Fulcio/Rekor 命令、生产身份令牌投放和外部系统闭环。
- `agents/orchestrator/src/orchestrator/pipeline.py:171` 的 `ReasoningPipeline` 是 `/v1/reason/*` UI workbench 兼容路径，阶段为 parse、generation、scoring、filter、novelty、ranking、summary，不再作为 CoreArchitecture v2 主编排路径。
- `agents/generator_coord/src/generator_coord/agent.py` 已支持注入 generator client、显式 `generator_targets`、`GENERATOR_DISCOVERY_URI` HTTP/Python discovery provider、`GENERATOR_CLIENT_TARGETS` / `<GENERATOR>_GENERATOR_TARGET` 环境变量覆盖，并可通过通用 `GeneratorService` gRPC client 或 `python://module:function` client factory 调用选中的 generator backend；UAS 已有 `create_uas_generator_client()` 本地 client factory，可通过 `UAS_RUNNER_COMMAND` 外部 JSON runner 调用现有 UASGenerator，缺 command 或首个可执行文件不可用时 fail-fast；FullWorkflowClients 在显式 `generation_strategy` 请求下会委托该 agent 执行多 generator 选择和 dispatch；`orchestrator-svc` Compose/Kubernetes/Helm 已暴露 discovery/target env，调度前会通过 gRPC `Info()` 或注入/Python client 的 `health_check()` 做健康探测；真实集群注册中心部署仍未完成。
- `uv run pytest tests/unit/test_srb_agent.py tests/unit/test_generator_coord_agent.py tests/unit/test_validation_agent.py -q` 退出码为 0，50 项通过，覆盖 SRB SSP/XDL/SiLA2 adapter、GeneratorCoord target/discovery/health/CRG routing 和 ValidationAgent L0-L4 cascade/wiring/CRG skip 边界。
- `agents/nl2obj/src/nl2obj/agent.py` 已接入本地 parser，不再只回显输入字段；已支持注入 CIG compiler client 或通过 `CIG_COMPILER_TARGET` 直连 `cig-compiler-svc` 生成 CIG/HCIV/intent cone。
- `agents/validation_agent/src/validation_agent/agent.py` 默认 wiring 到 L0 RDKit、L1 GNINA、L2 Boltz2、L3 OpenFE；配置 `L0_ADMET_ORACLE_TARGET`、`L1_DOCKING_ORACLE_TARGET`、`L2_AFFINITY_ORACLE_TARGET`、`L3_FEP_ORACLE_TARGET` 或 `L4_QUANTUM_ORACLE_TARGET` 时会将对应层级接到通用 `OracleService` gRPC client；配置 `L4_QUANTUM_ORACLE_COMMAND` 时会调用本地 JSON quantum command wrapper，并在执行前校验首个可执行文件可用；未配置外部 target 且本地 runner 缺失时继续 fail-fast。
- `agents/supply_agent/src/supply_agent/agent.py` 已通过可注入 supply client 或 `SUPPLY_ORACLE_TARGET` 聚合 building block availability、价格、交期和供应来源；`supply-oracle-svc` 支持 `SUPPLY_CATALOG_URI=file://...` 的本地 JSON catalog 与 AiZynth HDF5 stock InChIKey 查询；runtime status、启动 preflight 和服务入口 env client 构建前 preflight 均以实际可用的 file catalog 为供应源，显式 `SUPPLY_CATALOG_URI` 只接受当前实际支持的 `file://` catalog URI，非 `file://` URI 会 fail-fast。
- `libs/mf-agents/src/mf_agents/messaging/redis_bus.py:12` 当前是 Redis-backed bus，不是 NATS JetStream。
- `libs/mf-agents/src/mf_agents/lineage/sigstore_signer.py` 已支持 `SIGSTORE_SIGN_COMMAND` / `SIGSTORE_VERIFY_COMMAND` 外部 Sigstore/Rekor command，向命令传递 payload hash、identity、identity token 与 Rekor URL；未配置命令时使用本地 HMAC-SHA256 fallback。
- `services/provenance-svc/src/provenance_svc/domain/sigstore_integration.py` 默认 local 模式使用 hash 签名；配置 `SIGSTORE_SIGN_COMMAND` 后会把 artifact id、artifact type、artifact payload、payload hash 和 `SIGSTORE_IDENTITY_TOKEN` 和 `SIGSTORE_REKOR_URL` 通过 stdin 交给外部签名命令，并缓存返回的 signature、certificate、bundle、identity 和 Rekor entry；配置 `SIGSTORE_VERIFY_COMMAND` 后会把 artifact id、artifact type、payload hash、signature、bundle、`SIGSTORE_REKOR_URL` 和 `SIGSTORE_EXPECTED_IDENTITY` 交给外部验证命令，并以其返回的 `valid` / `signature_valid` 为准；签名和验证命令执行前会校验首个可执行文件可用；`infra/docker/docker-compose.dev.yml`、`infra/kubernetes/deployments/moleculeforge-services.yaml` 和 `infra/helm/moleculeforge` 已为 provenance-svc 配置 Sigstore env / ConfigMap / Secret wiring。
- `uv run pytest tests/unit/test_provenance.py -q` 退出码为 0，19 项通过，覆盖 provenance model/schema、本地 dev signature、Sigstore sign/verify command preflight、identity token、expected identity、Rekor URL 传递、Rekor bundle cache 和 provenance-svc 部署 env wiring；lineage `SigstoreSigner` 的 executable preflight 由 `tests/test_mvp_pipeline.py` 中的签名/验证 command 缺失测试覆盖。

### 偏差说明

MARB 的主 orchestrator 路径已具备 LangGraph 状态机、workflow state CRG、provenance metadata 记录和 Neo4j belief/edge 写入链路，且 validation/critic 失败反馈可进入下一轮 generation；FullWorkflowClients 默认直连 HFM 做 generation，并已把 `generation_feedback` 序列化到 HFM `generator_params`，在请求显式提供 `generation_strategy` 时会委托 GeneratorCoordAgent 执行多 generator dispatch 并透传同一反馈；FullWorkflowClients 默认用 Boltz2 affinity gate 做 validation，在请求显式提供 oracle level 时会委托 ValidationAgent 执行 L0-L4 自适应 cascade；RETROSYN 阶段已委托 RetroSynAgent 规划路线，并可在路线规划后接入 Supply/SRB hook，FullWorkflowClients 已委托 RetroSynAgent、SupplyAgent 和 SRBAgent 产出 `retrosyn`、`supply` 与 `srb` state，`orchestrator-svc` 部署清单也已暴露这些内联 client 所需的 GeneratorCoordAgent target、Boltz2、ValidationAgent L4、RetroSyn、Supply 和 SiLA2 env；各 agent 类也已具备注入式 CRG repository 写入点，并可通过统一 env factory 默认接入同一 Neo4j-backed repository；BaseAgent 已提供 shared CRG 读回接口，OrchestratorAgent 已能消费 completed workflow_status 返回 cached 工作流结果，GeneratorCoordAgent auto 路由已能消费既有 selected_generators、失败类 CRG belief 和 route_humu_embedding，ValidationAgent 已能消费既有 validation_status 跳过重复 oracle cascade，RetroSynAgent 已能消费 failed validation belief 或 `retrosyn_routes=0` 跳过路线规划，并可把 route HUMU embedding 写回 route 结果与 CRG，SupplyAgent 已能消费既有 supply_feasibility 跳过重复 supply oracle，也能消费 `retrosyn_routes=0` belief 输出 unavailable，SRBAgent 已能消费 unavailable supply_feasibility 跳过 SSP 编译，CriticAgent 已能消费既有 critic_verdict 跳过重复规则评估，也能消费 validation/supply 失败 belief 和 `retrosyn_routes=0`。上述 agent 的重复执行与失败反馈级 CRG 读回已接入；BaseAgent JSON-LD payload 编码、signed AgentMessage envelope 发布、UUIDv7 message_id 默认生成、订阅接收端解包、recipient、message_type、payload_type_url 与 ttl 防循环校验和跨 agent 验证 helper 已接入，并可外接 SIGSTORE_SIGN_COMMAND/SIGSTORE_VERIFY_COMMAND；provenance-svc 的 Compose/Kubernetes/Helm Sigstore 部署 wiring 已补齐；更深层跨 agent 联合优化、真实 Fulcio/Rekor command/identity 值、集群发布验证和外部系统闭环仍不完整，消息总线和签名机制也与文档选型不同。

## 8. 第五层：Oracle 级联与 PCBO

### 文档要求

`MoleculeForge_CoreArchitecture_v2.md:619-650` 和 `MoleculeForge_CoreArchitecture_v2.md:826-853` 要求：

- L0：QED、SA、Lipinski、PAINS。
- L1：Boltz-2、ADMET-AI、Chemprop。
- L2：DiffDock-L + GNINA。
- L3：OpenFE RBFE。
- L4：GPU4PySCF / ORCA。
- 每层返回 `value +/- uncertainty` 并传播不确定度。
- PCBO 使用 HUMU 切空间 Gaussian Process、EHVI、PoF。

### 当前实现中的符合项

- `models/mf-oracles/rdkit-oracle/src/mf_oracles/rdkit_oracle/scorer.py:7` 有 SA score。
- `models/mf-oracles/rdkit-oracle/src/mf_oracles/rdkit_oracle/scorer.py:32` 有 QED。
- `models/mf-oracles/rdkit-oracle/src/mf_oracles/rdkit_oracle/scorer.py:45` 有 Lipinski。
- `models/mf-oracles/rdkit-oracle/src/mf_oracles/rdkit_oracle/scorer.py:67` 有 PAINS。
- `agents/validation_agent/src/validation_agent/agent.py:129` 定义了 L0-L4 层级名称。
- `uv run pytest tests/unit/test_l0_oracle.py -q` 退出码为 0，19 项通过，覆盖 RDKit L0 score/uncertainty、PAINS metadata、ADMET uncertainty wrapper、GNINA/Boltz provenance enforcement 和 OpenFE runner 缺失时的显式 skip 边界。

### 当前实现中的偏差

- `models/mf-oracles/boltz2/src/mf_oracles/boltz2/oracle.py:16` 要求外部 runner；runner 缺失时不可执行；`services/boltz2-svc/src/boltz2_svc/main.py` 已提供 `Boltz2OracleServicer` 通用 `OracleService` adapter，在配置 `BOLTZ2_PROTEIN_PDB_ID` 后可将 Oracle request 映射为 Boltz2 affinity batch 并返回 `affinity` score/uncertainty；配置 `BOLTZ2_ORACLE_COMMAND` 时可通过 stdin/stdout JSON command runner 返回 affinity rows，先校验首个可执行文件可用，并绕过本机 boltz binary/checkpoint runtime gate；部署清单已暴露 `BOLTZ2_ORACLE_COMMAND`、`BOLTZ2_ORACLE_TIMEOUT_SECONDS`、`BOLTZ2_PROTEIN_PDB_ID`、`BOLTZ2_ENSEMBLE_SIZE`、`BOLTZ_MODEL_PATH`、`BOLTZ_INPUT_TEMPLATE_DIR`、`BOLTZ_WORK_DIR` 和 `BOLTZ_BINARY`，Compose 已默认指向当前本地 `models/artifacts/boltz-2` 与 `models/artifacts/boltz-input-templates`。
- `services/dock-svc/src/dock_svc/main.py` 有 `DockServicer` 和通用 `OracleService` adapter；配置 `DOCK_ORACLE_COMMAND` 时会调用外部 JSON command runner 并解析 `scores` / `uncertainties` / `elapsed_ms`，runtime preflight 和实际执行 runner 前都会校验首个可执行文件可用，显式配置的坏 command 不会被 GNINA/DiffDock 可用状态掩盖，service abort 路径也会保留该 command 错误；未配置 runner 时继续 fail-fast；部署清单已暴露 `DOCK_ORACLE_COMMAND`、`DOCK_ORACLE_TIMEOUT_SECONDS`、`GNINA_BINARY` 和 `DIFFDOCK_MODEL_PATH`，Compose 已默认指向当前本地 `models/artifacts/gnina/gnina.1.3.2.cuda12.8` 与 `models/artifacts/diffdock`。
- `services/fep-svc/src/fep_svc/main.py` 的 `FEPServicer` 支持 `FEP_ORACLE_COMMAND` 外部 JSON command runner，能把 `FEPBatchRequest` 序列化给 runner，并解析 `FEPBatchResponse`；配置 command 时 runtime status 会报告 command executable 状态，并在 subprocess 前校验首个可执行文件可用；`FEPOracleServicer` 已接入通用 `OracleService`，在配置 `FEP_REFERENCE_LIGAND_SMILES` 后可把请求分子映射为 test ligands 并返回 `rbfe` score/uncertainty；未配置 runner 时继续 fail-fast；部署清单已暴露 `FEP_ORACLE_COMMAND`、`FEP_ORACLE_TIMEOUT_SECONDS`、`FEP_REFERENCE_LIGAND_SMILES`、`FEP_METHOD`、`FEP_N_REPEATS` 和 `OPENFE_RUNNER_PATH`。
- `models/mf-oracles/admet_ai/src/mf_oracles/admet_ai/oracle.py:43` 是 HTTP runner；`predict_with_uncertainty()` 已通过同一 `/predict` endpoint 携带 `return_uncertainty` 请求并解析外部服务返回的 predictions 和 uncertainties；`services/admet-svc/src/admet_svc/main.py` 已支持 `ADMET_ORACLE_COMMAND` 外部 JSON command runner，能将 `smiles`、`properties` 和 `return_uncertainty` 传给 runner，并解析 predictions、uncertainties 与 elapsed_ms，配置 command 时可绕过本地 `ADMET_MODEL_PATH` runtime gate；显式配置的坏 command 会在 runtime preflight 中 fail-fast，不会被可用的本地 model artifact 掩盖；部署清单已暴露 `ADMET_ORACLE_COMMAND`、`ADMET_ORACLE_TIMEOUT_SECONDS`、`ADMET_MODEL_PATH`、`ADMET_SERVICE_URL`、`ADMET_TARGETS` 和 `ADMET_BATCH_SIZE`。
- `agents/validation_agent/src/validation_agent/agent.py:400` 已支持 L0-L3 默认 oracle wiring，`agents/validation_agent/src/validation_agent/agent.py:405` 到 `agents/validation_agent/src/validation_agent/agent.py:420` 支持外部 `L0_ADMET_ORACLE_TARGET` / `L1_DOCKING_ORACLE_TARGET` / `L2_AFFINITY_ORACLE_TARGET` / `L3_FEP_ORACLE_TARGET` / `L4_QUANTUM_ORACLE_TARGET` wiring，`agents/validation_agent/src/validation_agent/agent.py:433` 支持通用 `L4_QUANTUM_ORACLE_COMMAND` 本地 JSON command wrapper 和命名的 `L4_GPU4PYSCF_COMMAND` / `L4_ORCA_COMMAND` JSON command wiring；`orchestrator-svc` 的 Compose、静态 Kubernetes 清单和 Helm values 已暴露这些 L4 quantum env；具体 GPU4PySCF、ORCA、DFT/DFTB3 runner command 值、artifact 值与集群发布验证仍未完成。
- `libs/mf-eval/src/mf_eval/hv_evaluator.py:30` 有 2D hypervolume / HVI，另有 PoF、constrained HVI、EHVI、批量 constrained HVI acquisition 工具、`humu_logmap_tangent_features()`、`libs/mf-eval/src/mf_eval/hv_evaluator.py:227` 的 `rank_tangent_gp_constrained_hvi_candidates()`、`libs/mf-eval/src/mf_eval/hv_evaluator.py:285` 的 `rank_tangent_gp_constrained_ehvi_candidates()`、`libs/mf-eval/src/mf_eval/hv_evaluator.py:356` 的 `rank_humu_logmap_gp_constrained_ehvi_candidates()`、`libs/mf-eval/src/mf_eval/hv_evaluator.py:404` 的 `async_pcbo_oracle_loop()` 和 `libs/mf-eval/src/mf_eval/hv_evaluator.py:533` 的 `PCBOOptimizationScheduler`；可用 HUMU log-map 或已在切空间的 embeddings 训练 exact RBF GP，按 predicted objective + PoF 或 EHVI + PoF 排序候选，并按多轮 candidate provider -> oracle -> observation update 调度优化。
- `pipelines/pareto_bo/src/pareto_bo/service.py:58` 的 `ParetoBOService.from_env()` 支持两类运行时配置：`PARETO_BO_CANDIDATE_PROVIDER` / `PARETO_BO_ORACLE_EVALUATE` 的 `module:attribute` callable path，以及 `PARETO_BO_CANDIDATE_PROVIDER_COMMAND` / `PARETO_BO_ORACLE_EVALUATE_COMMAND` 的 stdin/stdout JSON command runner。command provider 接收包含 round index 和历史 observation 的 JSON state，返回 `candidate_embeddings`；command oracle 接收候选请求和 acquisition metadata，返回 `objectives` 与 `constraints`；`pipelines/pareto_bo/src/pareto_bo/service.py:178` 的 command runner 会在 subprocess 前校验首个可执行文件可用；`pipelines/pareto_bo/src/pareto_bo/service.py:33` 已暴露 FastAPI `rest_app`，`POST /v1/pareto-bo/optimize` 会用当前 env 构造 `ParetoBOService` 并执行 optimize。
- `uv run pytest tests/unit/test_mf_eval.py -q` 退出码为 0，20 项通过，覆盖 distortion metrics、activity cliff、hypervolume/HVI、PoF、constrained HVI、EHVI、HUMU log-map tangent features、tangent GP constrained acquisition、`async_pcbo_oracle_loop()`、`PCBOOptimizationScheduler`、`ParetoBOService` JSON command provider/oracle、FastAPI endpoint 和 `pareto-bo-svc` deployment wiring。

### 偏差说明

L0 oracle 是真实可用组件；L1-L3 更像外部系统适配层，其中 admet-svc 已具备 `ADMET_ORACLE_COMMAND` JSON runner 接入点，ValidationAgent 已可通过 `L0_ADMET_ORACLE_TARGET` 把 L0 ADMET filter 接到通用 ADMET `OracleService`，L2 dock-svc 已具备 `DOCK_ORACLE_COMMAND` JSON runner 接入点，boltz2-svc 已具备通用 OracleService adapter、`BOLTZ2_ORACLE_COMMAND` JSON runner 接入点和 protein/ensemble adapter 参数 env，L3 fep-svc 已具备 `FEP_ORACLE_COMMAND` JSON runner 接入点、通用 OracleService adapter 和 reference/method/repeat adapter 参数 env；ADMET/Dock/Boltz2/FEP command runner 已执行 executable preflight，ADMET 和 Dock runtime preflight 已拒绝显式配置但不可用的 command；这些 L1-L3 服务的 Compose/Kubernetes/Helm runner env、本机 artifact/tool runtime env 和 adapter 参数 env wiring 已补齐，Compose、静态 Kubernetes 清单和 Helm values/template 已接入当前本地 GNINA、DiffDock-L、Boltz-2 和 Boltz input template artifact 默认路径；真实 runner command 值、Boltz full inference smoke、OpenFE 可执行环境和集群发布验证仍未完成；L4 已具备外部 OracleService 接入点、通用本地 JSON command wrapper、command executable preflight 和 GPU4PySCF/ORCA 命名 command wiring，`orchestrator-svc` 部署清单也已暴露对应 L4 env；真实 GPU4PySCF/ORCA runner command/artifact 值与集群发布验证仍未完成；PCBO 已补局部 PoF/约束 HVI/EHVI acquisition 工具、HUMU log-map 切空间映射、tangent-space RBF GP 候选排序、库级异步 oracle 采样循环、多轮 optimization scheduler、独立 `pareto_bo` service package、env-configured callable CLI 入口、外部 JSON command runner 接入点、command executable preflight、FastAPI optimize endpoint 和 `pareto-bo-svc` 部署 wiring；真实 candidate provider/oracle evaluator command/env 与生产验收仍未完成。

## 9. 第六层：SRB、RetroSyn、Supply

### 文档要求

`MoleculeForge_CoreArchitecture_v2.md:857-904` 要求 SRB 输出 SSP，并把 NL、CIG、HCIV、分子、路径、SSP 全链路写入 Provenance Graph。`MoleculeForge_CoreArchitecture_v2.md:593-617` 要求 RetroSyn 三层引擎：RAscore、RSGPT/UAlign、AiZynthFinder，并将 route embedding 反馈 HUMU。

### 当前实现中的符合项

- `agents/srb_agent/src/srb_agent/compiler.py:22` 的 `compile_ssp()` 能从 molecule 和 retrosyn_route 构建 SSP。
- `agents/srb_agent/src/srb_agent/compiler.py:47` 返回 `SSP`，包含 materials、steps、total yield、cost、xdl_version、sila2_endpoint。
- `agents/srb_agent/src/srb_agent/agent.py:79` 的 agent 已接入 `compile_ssp()`，并在 `agents/srb_agent/src/srb_agent/agent.py:145` 的 protocol 输出层通过现有 XDL bridge 生成 `xdl_xml`，同时从 SSP steps 构造结构化 `sila2_plan`，每个 step 保留 `ssp_step_id`、`retrosyn_route_step_id`、operation、reaction_type、reactants、reagents、temperature、duration 和 purification；`agents/srb_agent/src/srb_agent/agent.py:162` 在配置 `SILA2_PLAN_COMMAND` 时会先校验首个可执行文件可用，再将 SSP、XDL 和 SiLA2 plan 作为 JSON 交给外部 adapter，并把返回的 `sila2_execution` 与 endpoint 写回 protocol；orchestrator-svc 部署清单、静态 Kubernetes ConfigMap 和 Helm values/template 已为该命令和超时参数提供空默认数据。
- `models/mf-retrosyn` 下存在 `aizynth_wrapper`、`rsgpt`、`ualign` 目录。
- `agents/retrosyn_agent/src/retrosyn_agent/agent.py` 已支持注入式多 planner ensemble，通过 `route_planners={"rascore": ..., "aizynth": ..., "rsgpt": ..., "ualign": ...}` 合并多引擎 routes，按 score / predicted_score / route_score / yield 排序并保留 `source_engine`；RAscore 的 `route_type=retrosynthetic_accessibility_score` 会排在真实 reaction route 后，避免快筛分数在 `max_routes` 截断时挤掉真实路线；配置 `RETROSYN_PLANNER_COMMAND` 时也可通过 stdin/stdout JSON 调用外部 planner，并在执行前校验首个可执行文件可用；配置 `RETROSYN_PLANNER_COMMANDS_JSON` 时可从环境构建多 planner command ensemble；同时支持 `RASCORE_PLANNER_COMMAND`、`RSGPT_PLANNER_COMMAND`、`UALIGN_PLANNER_COMMAND` 和 `AIZYNTH_PLANNER_COMMAND` 命名 runner env 自动组成 ensemble。
- `services/retrosyn-svc/src/retrosyn_svc/main.py` 的 `RetrosynServicer` 已支持注入式 `route_planners`，`engine="ensemble"` 时会合并多 planner routes、去重、按 route type 与 score / predicted_score / route_score / yield 排序并返回 top routes；指定单个注入 engine 时可只调用该 planner；配置 `RETROSYN_PLANNER_COMMAND` 时可通过 stdin/stdout JSON 调用外部 RAscore/RSGPT/UAlign/AiZynth runner；配置 `RETROSYN_PLANNER_COMMANDS_JSON` 或 `RASCORE_PLANNER_COMMAND` / `RSGPT_PLANNER_COMMAND` / `UALIGN_PLANNER_COMMAND` / `AIZYNTH_PLANNER_COMMAND` 时可通过环境变量部署多 planner command ensemble；runtime status 和启动 preflight 已把单 planner command、JSON ensemble command 与命名 planner command 从 URI artifact 检查改为 executable command 检查，无 `AIZYNTH_CONFIG_PATH` 但已配置的全部外部 planner command 都可用时可启动，缺失 executable、非法 JSON、空 engine、空 command 或部分 command 不可用会 fail-fast；实际执行 planner subprocess 前同样校验首个可执行文件可用；`retrosyn-svc` 和 `orchestrator-svc` 的 Compose、静态 Kubernetes 清单和 Helm values 已暴露 planner env、`HUMU_ENCODER_TARGET` 和 `AIZYNTH_CONFIG_PATH`，其中 Compose 本地默认指向 `models/artifacts/aizynthfinder/config.yml`，静态 Kubernetes 清单已在 `mf-agents` 与 `mf-oracles` 两个 namespace 声明 `retrosyn-planner-config` ConfigMap，Helm values/template 也已提供同名 ConfigMap 数据渲染入口。
- `services/humu-encoder-svc/src/humu_encoder_svc/main.py` 已支持 proto `entity_type=route` + JSON `input_data` 的 route encoding 请求。
- `agents/retrosyn_agent/src/retrosyn_agent/agent.py` 已支持注入 `route_encoder_client` 或通过 `HUMU_ENCODER_TARGET` 调用 HUMU route encoder，并把路线 `humu_embedding` / `humu_curvature` 写回 route 结果和 `route_humu_embedding` CRG belief；`retrosyn-svc` 和 `orchestrator-svc` 部署配置均已暴露该 target。
- `agents/supply_agent/src/supply_agent/agent.py:37` 已接入 supply client，可对 building block availability、价格、交期和供应商多样性做聚合评估；`agents/supply_agent/src/supply_agent/agent.py:113` 在缺 `SUPPLY_ORACLE_TARGET` 或注入 client 时 fail-fast。
- `services/supply-oracle-svc/src/supply_oracle_svc/main.py:189` 支持本地 JSON catalog。
- `services/supply-oracle-svc/src/supply_oracle_svc/main.py:223` 支持 AiZynth stock HDF5 InChIKey 查询。
- `services/supply-oracle-svc/src/supply_oracle_svc/main.py` 已注册 protobuf servicer。

### 当前实现中的偏差

- `agents/retrosyn_agent/src/retrosyn_agent/agent.py` 和 `services/retrosyn-svc/src/retrosyn_svc/main.py` 默认仍是 `AiZynthRetrosyn.from_env()`；注入式 ensemble、`RETROSYN_PLANNER_COMMANDS_JSON` env ensemble 以及 RAscore/RSGPT/UAlign/AiZynth 命名 command env 已可合并多 planner 输出，agent 与 retrosyn-svc 均已有 `RETROSYN_PLANNER_COMMAND` 外部 JSON runner 入口，并已对实际执行的 planner command 做 executable preflight；retrosyn-svc runtime status 和启动 preflight 已接受 `RETROSYN_PLANNER_COMMAND`、`RETROSYN_PLANNER_COMMANDS_JSON` 内可执行 command 或命名 planner command 作为 AiZynth config 之外的 planner runtime 来源；retrosyn-svc 和 orchestrator-svc 部署清单已提供 planner command 与 `AIZYNTH_CONFIG_PATH` wiring，Compose 本地默认指向 `models/artifacts/aizynthfinder/config.yml`，静态 Kubernetes 与 Helm 已为 AiZynth config path 声明 `retrosyn-planner-config` 数据；仓库内已具备 RAscore 快筛 runner 与 RSGPT/UAlign/AiZynth 本地真实 runner command；仍缺集群发布验证和 KRAS full pilot。
- `services/supply-oracle-svc/src/supply_oracle_svc/main.py:501` 支持 `file://` catalog URI 和 AiZynth HDF5 stock；runtime status、启动 preflight 和服务入口 env client 构建前 preflight 只接受当前实际支持的 `file://` catalog URI，非 `file://` URI 会 fail-fast。

### 偏差说明

SRB 的 SSP 编译和 agent protocol 输出是当前较真实的部分，已经覆盖 SSP、XDL XML 和结构化 SiLA2 step plan，并具备 `SILA2_PLAN_COMMAND` 外部 SiLA2 adapter runner 接入点和 executable preflight；orchestrator-svc 部署 wiring 已补齐，静态 Kubernetes 清单和 Helm values/template 已声明空 adapter command 默认数据；真实 SiLA2 硬件 endpoint、adapter command 值与集群发布验证仍未完成。RetroSyn agent 和 retrosyn-svc 已具备注入式多 planner ensemble，并均支持 `RETROSYN_PLANNER_COMMAND` 单 runner、`RETROSYN_PLANNER_COMMANDS_JSON` 多 runner env ensemble，以及 RAscore/RSGPT/UAlign/AiZynth 命名 command env ensemble，agent 与 retrosyn-svc 均已对 planner command 做 executable preflight；retrosyn-svc 启动 preflight 已不再只依赖 `AIZYNTH_CONFIG_PATH`，可接受单 runner、JSON ensemble command 或命名 planner command 作为 planner runtime 来源，且所有显式配置的 planner command 必须可用；agent 侧还具备 route HUMU embedding 写回路径，retrosyn-svc 与 orchestrator-svc 均已暴露 `HUMU_ENCODER_TARGET` 和 `AIZYNTH_CONFIG_PATH`；Compose 本地默认已有 AiZynth config artifact 路径，RAscore/RSGPT/UAlign/AiZynth 本地真实 runner command 已完成；集群 ConfigMap 值投放、生产推理复验与集群发布验证仍未完成。Supply Agent 已接入 supply client 聚合本地 catalog 或 supply oracle 返回值，Supply Oracle 保留本地 JSON catalog 与 AiZynth HDF5 stock 路径；orchestrator-svc 已暴露 `SUPPLY_ORACLE_TARGET` 指向 supply-oracle-svc。

## 10. 第七层：DKI 数据与知识基础设施

### 文档要求

`MoleculeForge_CoreArchitecture_v2.md:909-985` 原文要求：

- Vector Store：Milvus 2.5 + Faiss IVF-PQ，支持 HUMU 双曲距离和 10^9 级集合。
- Graph Store：Neo4j。
- Relational Store：PostgreSQL 16 + TimescaleDB。
- Object Store：MinIO。
- Feature Store：Feast。
- Agent 消息基础设施：NATS JetStream。

当前项目实施决策：DKI 不回迁 Milvus/NATS，正式采用 Qdrant/Redis，并继续使用 Neo4j、PostgreSQL、MinIO、Feast。

### 当前实现中的符合项

- `infra/docker/docker-compose.dki.yaml:1` 定义了 DKI stack。
- `infra/docker/docker-compose.dki.yaml:2` 是 Postgres。
- `infra/docker/docker-compose.dki.yaml:17` 是 Neo4j。
- `infra/docker/docker-compose.dki.yaml:40` 是 MinIO。
- `services/feature-store-svc/src/feature_store_svc/main.py:1` 是 Feast-based service；Compose、静态 Kubernetes 清单和 Helm values/template 已暴露 `FEAST_REPO_PATH` ConfigMap 数据，默认指向 `feature_repo`。
- `services/provenance-svc/src/provenance_svc/main.py:163` 有 Postgres audit writer。
- `services/provenance-svc/src/provenance_svc/main.py:573` 后会构建 MinIO、Neo4j、Postgres 相关 client。

### 当前实现与决策

- `infra/docker/docker-compose.dki.yaml:28` 使用 Qdrant，符合当前实施决策。
- `services/humu-index-svc/src/humu_index_svc/main.py:1` 明确写的是 Qdrant vector store。
- `services/humu-index-svc/src/humu_index_svc/main.py:37` health 返回 `backend: qdrant`。
- `infra/docker/docker-compose.dki.yaml:56` 使用 Redis，符合当前实施决策。
- `services/feature-store-svc/src/feature_store_svc/main.py:53` batch features 依赖 Feast offline store；如果 store 没有 `get_historical_features` 会返回 501。
- `services/feature-store-svc/src/feature_store_svc/main.py:88` materialize 依赖 Feast `materialize_incremental`，未配置时返回 501。

### 说明

DKI 是当前实现中比较落地的一层。虽然 CoreArchitecture v2 原文写的是 Milvus + NATS，但当前项目明确采用 Qdrant + Redis，因此后续补齐不应再把 DKI 回迁 Milvus/NATS 作为目标。`uv run pytest tests/integration -q` 在当前环境下退出码为 0，32 项中 22 项通过、10 项 skip；skip 直接原因是缺 `MINIO_ENDPOINT_URL` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` / `MINIO_BUCKET`、`NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`、`TEST_DATABASE_URL`、`PROVENANCE_DATABASE_URL`、`QDRANT_HOST` 或 `QDRANT_URL`、`REDIS_HOST` 或 `REDIS_URL` 等外部 DKI 环境变量。当前可确认本地 CIC 等集成路径通过；Qdrant/Redis 与 Postgres、Neo4j、MinIO、Feast 的真实环境可用性和端到端业务接入仍需在投放外部 DKI 环境后验证。

## 11. 第八层：工程实施与服务化

### 文档要求

`MoleculeForge_CoreArchitecture_v2.md:989-1115` 要求核心服务、GPU/CPU/数据服务、K8s namespace、HPA、存储类等工程部署方案。

### 当前实现

当前 `services` 下有 21 个服务目录：

```text
admet-svc
api-gateway
boltz2-svc
cig-compiler-svc
crem-generator-svc
critic-svc
dock-svc
feature-store-svc
fep-svc
fragfm-generator-svc
generator-router-svc
hfm-generator-svc
humu-encoder-svc
humu-index-svc
iclm-svc
mmpt-generator-svc
nl2obj-svc
orchestrator-svc
provenance-svc
retrosyn-svc
supply-oracle-svc
```

有 protobuf servicer 注册的例子：

- `services/humu-encoder-svc/src/humu_encoder_svc/main.py:243`
- `services/hfm-generator-svc/src/hfm_generator_svc/main.py:169`
- `services/fragfm-generator-svc/src/fragfm_generator_svc/main.py:188`
- `services/boltz2-svc/src/boltz2_svc/main.py:535`
- `services/retrosyn-svc/src/retrosyn_svc/main.py:530`
- `services/critic-svc/src/critic_svc/main.py:105`
- `services/cig-compiler-svc/src/cig_compiler_svc/main.py:375` 注册 `CIGCompilerService`。
- `services/nl2obj-svc/src/nl2obj_svc/main.py:173` 注册 `NL2ObjService`。
- `services/admet-svc/src/admet_svc/main.py:421` 注册通用 `OracleService` adapter。
- `services/boltz2-svc/src/boltz2_svc/main.py:529` 注册通用 `OracleService` adapter。
- `services/dock-svc/src/dock_svc/main.py:265` 注册通用 `OracleService` adapter。
- `services/fep-svc/src/fep_svc/main.py:227` 注册通用 `OracleService` adapter。
- `services/supply-oracle-svc/src/supply_oracle_svc/main.py:719` 注册 `SupplyOracleService`。
- `services/orchestrator-svc/src/orchestrator_svc/main.py:747` 注册 `OrchestratorService`。

### 偏差说明

工程目录和服务数量接近文档目标，主要 gRPC 服务已经具备 protobuf service contract 和 servicer 注册。HFM、FragFM、CReM、iCLM、MMPT 生成服务的 `Info()` 已返回非空 `GeneratorInfo`，可被 GeneratorCoordAgent 调度前健康探测消费。实际部署是否可用不能只看目录存在，还需要逐个验证服务启动、artifact 配置和外部依赖。

## 12. 第九层：评估体系与 benchmark

### 文档要求

`MoleculeForge_CoreArchitecture_v2.md:1225-1268` 要求：

- MOSES 2.0 / GuacaMol v3。
- CrossDocked 2020 v2。
- PMO 23 任务。
- KRAS G12C docking、Pareto HV、OOD calibration。
- Agent 任务完成率、Sigstore 审计完整性、D2L。
- HUMU distortion、activity cliff、EF1%、mol-route 一致性。

### 当前实现中的符合项

- `pyproject.toml:137` 的 pytest 配置存在。
- `pyproject.toml:141` 已把 `*_benchmark.py` 纳入 pytest 收集。
- `libs/mf-eval/src/mf_eval/molecule/moses.py:5` 的 `evaluate_moses()` 聚合 MOSES 风格指标，`libs/mf-eval/src/mf_eval/molecule/moses.py:29`、`:49`、`:62`、`:79` 分别实现 validity、uniqueness、novelty、diversity。
- `libs/mf-eval/src/mf_eval/distortion.py`、`libs/mf-eval/src/mf_eval/cliff_analysis.py`、`libs/mf-eval/src/mf_eval/hv_evaluator.py` 存在。
- `tests/e2e/test_kras_g12c_pilot.py:121` 有 KRAS G12C E2E 的显式开关，`tests/e2e/test_kras_g12c_pilot.py:38` 后的 preflight 已要求 `SIGSTORE_IDENTITY_TOKEN`、`SIGSTORE_EXPECTED_IDENTITY`、`SIGSTORE_SIGN_COMMAND`、`SIGSTORE_VERIFY_COMMAND` 与 `SIGSTORE_REKOR_URL` Sigstore 配置。
- `tests/e2e/test_audit_completeness.py` 有 audit completeness E2E 的显式开关和 preflight，已要求 `SIGSTORE_IDENTITY_TOKEN`、`SIGSTORE_EXPECTED_IDENTITY`、`SIGSTORE_SIGN_COMMAND`、`SIGSTORE_VERIFY_COMMAND` 与 `SIGSTORE_REKOR_URL` Sigstore 配置。
- `uv run pytest tests/e2e/test_audit_completeness.py -q` 在当前默认环境下退出码为 0，但 4 个 audit E2E 全部 skip，直接原因是未设置 `RUN_AUDIT_E2E=1`。
- `tests/conftest.py` 已在未显式设置 `MF_DB_PATH` 的 pytest 会话中创建临时 SQLite DB，并在退出时清理，避免 Reason workbench E2E 写入仓库内 `FL:FL 644` 的默认 `data/moleculeforge.db`。
- `uv run pytest tests/e2e -q` 在当前默认环境下退出码为 0，25 个 E2E 中 15 个通过、10 个 skip；skip 全部来自 `RUN_AUDIT_E2E=1` 和 `RUN_KRAS_G12C_E2E=1` 显式开关未启用。

### 当前实现中的偏差

- `tests/benchmark/moses_benchmark.py` 已改为资源门控：缺 `MOSES_REFERENCE_SMILES_PATH` 或生成器 artifact 时 skip，资源齐全时执行 HFM-3D 生成和 MOSES 指标断言。
- `tests/benchmark/guacamol_benchmark.py` 已改为资源门控：缺 HFM-3D artifact 时 skip，资源齐全时执行 rediscovery / median / MPO 评分断言。
- `tests/benchmark/pmo_benchmark.py` 已改为资源门控：缺 HFM-3D artifact 或 `PMO_SCORE_TABLE_PATH` 时 skip，资源齐全时执行 LogP/QED/DRD2/JNK3/GSK3B 断言。
- `tests/benchmark/crossdocked_benchmark.py` 已补齐独立 CrossDocked JSONL benchmark 入口，缺 `CROSSDOCKED_BENCHMARK_JSONL` 时 skip，资源齐全时验证 pocket/ligand、SMILES validity、test split 和 docking score gate。
- `uv run pytest tests/benchmark -q` 在当前默认环境下退出码为 0，但 18 个 benchmark 全部 skip：缺 `CROSSDOCKED_BENCHMARK_JSONL`、`MOSES_REFERENCE_SMILES_PATH`、`HFM_CHECKPOINT_PATH`、`HFM_DECODER_PATH`、`FRAGFM_MOSES_GENERATED_SMILES_PATH` 和 `PMO_SCORE_TABLE_PATH`。
- 使用临时资源文件运行 `tests/benchmark/moses_benchmark.py::TestMosesBenchmark::test_fragfm_moses_validity`、`tests/benchmark/pmo_benchmark.py::TestPMOBenchmark::test_drd2_optimization`、`tests/benchmark/pmo_benchmark.py::TestPMOBenchmark::test_multi_objective_jnk3_gsk3b` 和 `tests/benchmark/crossdocked_benchmark.py`，退出码为 0，7 项通过，证明 FragFM MOSES、PMO 表格和 CrossDocked JSONL 的资源齐全路径可非 skip 执行。
- 使用当前本地 `checkpoints/hfm3d_4h200/best_model.pt`、`checkpoints/hfm3d_4h200/decoder.json` 和临时 MOSES reference 运行 `TestMosesBenchmark` 的 HFM validity、uniqueness、novelty 三项小批量 smoke，退出码为 0，3 项通过，证明 HFM benchmark 入口可加载当前 checkpoint/decoder 进入生成路径；这不是 GuacaMol/PMO 全量质量达标证明。
- `tests/e2e/test_kras_g12c_pilot.py:24` 后要求 HFM、Boltz、AiZynth 等 artifact 和环境变量；默认不等于真实 pilot 已跑通。
- `uv run pytest tests/e2e/test_kras_g12c_pilot.py -q` 在当前默认环境下退出码为 0，但 6 个 KRAS pilot E2E 全部 skip，直接原因是未设置 `RUN_KRAS_G12C_E2E=1`。

### 偏差说明

评估体系已有入口和部分工具函数。MOSES/GuacaMol/PMO/CrossDocked 已具备资源门控 benchmark 入口；临时资源与当前本地 HFM artifact 已能覆盖部分非 skip 执行路径；当前默认环境缺正式 benchmark 数据和生产训练 artifact，默认 benchmark 运行结果仍是全 skip，不能声称已经完成 MOSES/GuacaMol/PMO/CrossDocked/KRAS 全面评估。

## 13. 已经可以确认的符合项

以下内容有当前源码或文件证据支撑：

1. Lorentz manifold 基础算子存在。
2. Learnable curvature 存在。
3. HUMU pretrain pipeline 包含多路 contrastive loss 和 curvature regularization。
4. HFM、FragFM、CReM、MMPT、iCLM、UAS 等 generator 目录存在；FragFM 已有 `TwoLevelDFM`、`SAAwareRateMatrix`、vocabulary artifact、rule-level HUMU intent alignment、可注入共享 HUMU latent sampler、训练产物测试和 teacher embedding KD loss 训练入口；CReM 与 MMPT artifact builder 已能计算结构特征 embedding KD loss；UAS 已有 autoencoder 训练、teacher embedding KD loss 训练入口和 `p_safe(z)` OOD sampling 输出。
5. TaskAwareRouter 和 GeneratorRouterService 存在。
6. CrossParadigmKDLayer 存在，GeneratorRouterService feedback 已能接入 oracle teacher score，并可通过 `HYPSEEK_TEACHER_COMMAND` 或 `HYPSEEK_TEACHER_URL` 调用外部 HypSeek teacher；`generator_router_svc.main:hypseek_app` 已提供可独立运行的 `/teacher` HTTP endpoint 和 `/healthz` health endpoint；`HYPSEEK_TEACHER_COMMAND` 与 `TAR_PROXYLESS_SEARCH_COMMAND` 配置后会做 runtime executable preflight；KD 层已能消费归一化 teacher distribution 和 teacher embedding target 计算 embedding distillation loss，并已有 Boltz2 ΔG/per-member ΔG adapter 与 HypSeek 显式 score-field adapter；HFM-3D、FragFM、UAS、CReM 与 MMPT CLI 已接入该 loss；iCLM service update 路径已能通过 `ICLM_UPDATE_COMMAND` 把训练样本和 KD 参数交给外部 runner，并在 runtime status 和实际执行该 command 前校验首个可执行文件可用，也可在注入 generator online learner 时直接调用 `online_learner.update()`；默认 iCLM OnlineLearner 本体已能直接计算 teacher embedding KD loss；Compose/Kubernetes/Helm 已配置 `hypseek-teacher-svc`、`HYPSEEK_TEACHER_URL=http://hypseek-teacher-svc:8012/teacher`、`HYPSEEK_TEACHER_COMMAND`、`HYPSEEK_TEACHER_TIMEOUT_SECONDS`、Compose healthcheck、Kubernetes/Helm readiness/liveness probes、HypSeek/TAR 空 command ConfigMap 数据和 iCLM model/update runner env wiring，真实集群发布仍未验证。
7. LangGraph StateGraph 存在。
8. CRG 类型和本地图结构存在。
9. Critic rule agent 与 critic-svc 存在。
10. L0 RDKit oracle 存在。
11. SRB SSP compiler 存在，并已由 SRB agent 调用。
12. DKI 的 Postgres、Neo4j、Qdrant、MinIO、Redis、Feast 相关配置和服务代码存在。
13. Provenance service 有 Neo4j、Postgres、MinIO 集成代码。
14. pytest benchmark 收集配置已包含 `*_benchmark.py`。

## 14. 需要优先判定的架构分歧

### 14.1 Qdrant/Redis 已正式替代 Milvus/NATS

当前实现已经明显走向：

```text
Milvus -> Qdrant
NATS JetStream -> Redis
```

该选择已经确认：DKI 就用 Qdrant/Redis。后续不再实施 Milvus/NATS 回迁，相关配置、测试和文档应围绕 Qdrant/Redis 保持一致。

### 14.2 v2 目标是“研究原型”还是“工程 MVP”

当前实现中很多组件采用 fail-fast 和 runner wrapper，这适合工程 MVP，但与文档中要求的算法创新完整实现不同。两种目标对应完全不同的后续工作：

```text
工程 MVP    :  优先补齐服务注册、artifact wiring、端到端 pipeline、真实 benchmark。
研究 v2     :  在不改现有 HUMU 预训练主线的前提下，优先补齐生成阶段的 HCIV/意图锥接入、HFM 生产级神经几何 decoder、FragFM-HUMU 条件耦合、PCBO、KD、JMCG 闭环；HUMU 等变编码器属于后续研究升级，不作为当前修改前置条件。
```

### 14.3 主业务路径应以哪个 orchestrator 为准

当前至少存在两类编排路径：

- `agents/orchestrator/src/orchestrator/workflow/graph_builder.py` 的 LangGraph 路径。
- `agents/orchestrator/src/orchestrator/pipeline.py` 的 ReasoningPipeline 路径。

已选定 `services/orchestrator-svc` 调用的 LangGraph workflow 作为 CoreArchitecture v2 主编排路径。`agents/orchestrator/src/orchestrator/pipeline.py` 保留为 `/v1/reason/*` 前端 reasoning workbench 的兼容路径，不再作为架构主链路。后续补齐 CIC、HUMU、AMGE、Oracle、RetroSyn、CRG 闭环时，应进入 `orchestrator-svc` 的 LangGraph workflow。

### 14.4 当前修改边界

当前阶段只有 HUMU 预训练部分可以不修改，原因是预训练已经在进行，且该阶段本身不依赖用户输入、CIG、HCIV 或意图锥。冻结范围包括：

- HUMU 预训练 pipeline。
- HUMU 预训练配置。
- HUMU 预训练 loss 组合。
- 当前 mol/pocket/route baseline 编码器作为预训练输入侧实现。
- 当前 checkpoint 续训和产出路径。

最终 HUMU 设想结构仍然保留，包括 3D SE(3) 分子编码器、E(3) 口袋编码器、反应树图 Transformer + 双曲 TreeLSTM 路线编码器和严格 Lorentz-equivariant attention。当前 molecule encoder 只补到显式 conformer 的 E(3)-invariant 距离统计，pocket encoder 只补到 E(3)-invariant 局部距离统计，route encoder 只补到路线树拓扑统计特征，这些最终结构仍属于后续 HUMU 升级，不阻塞当前架构补齐。

除上述 HUMU 预训练冻结范围以及已确认的 Qdrant/Redis DKI 选型以外，其余偏差均应按 CoreArchitecture v2 补齐，包括 CIC/HCIV、生成阶段意图锥接入、AMGE、TAR/KD、MARB、Oracle、SRB/Supply、服务注册和评估体系。

## 15. 修复优先级建议

执行原则：当前阶段不修改正在进行的 HUMU 预训练；除 HUMU 预训练冻结范围外，其余架构偏差均列入补齐范围。

### P0：先修工程可运行性

1. 已补齐 `cig-compiler-svc`、`nl2obj-svc`、`admet-svc`、`dock-svc`、`supply-oracle-svc`、`orchestrator-svc` 的 protobuf service contract 和 gRPC servicer 注册。
2. 保持 Qdrant/Redis 作为正式 DKI 架构选型，并同步所有架构文档、配置、netpol、部署说明。
3. 已选定 `orchestrator-svc` LangGraph workflow 为 CoreArchitecture v2 主 orchestrator 路径；`/v1/reason/*` ReasoningPipeline 降级为 UI workbench 兼容路径。
4. 已补齐 `ValidationAgent` 默认 L0-L4 oracle wiring，并支持 `L0_ADMET_ORACLE_TARGET`、`L1_DOCKING_ORACLE_TARGET`、`L2_AFFINITY_ORACLE_TARGET`、`L3_FEP_ORACLE_TARGET` 和 `L4_QUANTUM_ORACLE_TARGET` 外接 OracleService；L1-L4 缺 runner 时保持 fail-fast。
5. 已让 MOSES、GuacaMol、PMO、CrossDocked benchmark 在缺数据时 skip，并验证资源齐全时能真实执行。

### P1：补齐 CoreArchitecture v2 的最小闭环

1. 已补齐 CIC 输出 JSON-LD `@context`、objective edges 和 directed hyperedges 的最小链路，并同步 schema、proto 转换、Pydantic 类型。
2. 已让 HCIV learned encoder 消费 CIG 节点、边和 hyperedges，并补齐 directed hypergraph message-passing baseline；Owner A 已补 supervised train/export 工程路径，仍需真实数据训练并部署生产级 `Enc_intent` checkpoint。
3. 已在生成服务入口传递 CIC 输出的 intent cone，并让 GeneratorRouter 接收 HCIV/task profile；底层 AMGE 算法仍需后续升级为共享 HUMU 协同生成。
4. 已补齐 GeneratorRouter 的 HCIV、task profile、oracle history 输入链路。
5. 已在 `orchestrator-svc` LangGraph 主链路串起 `CIG -> HCIV -> generate -> validate -> retrosyn -> critic -> provenance`，generation 默认保留 HFM 路径，并在请求显式提供 `generation_strategy` 时委托 GeneratorCoordAgent 执行多 generator dispatch；FullWorkflowClients 已把 workflow state 的 `generation_feedback` 序列化到 `generator_params.generation_feedback`，覆盖默认 HFM 和显式 GeneratorCoordAgent 两条 generation 路径；RETROSYN 阶段已委托 RetroSynAgent 规划路线并支持 Supply/SRB hook，FullWorkflowClients 已委托 RetroSynAgent、SupplyAgent 和 SRBAgent 写入 `retrosyn` / `supply` / `srb` workflow state，validation 默认保留 Boltz2 affinity gate，并在请求显式提供 oracle level 时委托 ValidationAgent 执行 L0-L4 adaptive oracle cascade；`orchestrator-svc` 的 Compose/Kubernetes/Helm env 也已覆盖内联 GeneratorCoordAgent target、Boltz2、ValidationAgent L4 quantum、RetroSyn planner、route HUMU encoder、Supply 和 SiLA2 依赖，并在 workflow state 与 provenance metadata 中记录 CRG；workflow CRG 的 Neo4j belief/edge 持久化和 run_id 读回已接入，validation/critic 失败反馈可写入 `generation_feedback` 并回到下一轮 generation，OrchestratorAgent 可把 workflow_status belief 写入注入的 CRG repository，并读取 completed workflow_status 返回 cached 工作流结果，NL2ObjAgent 可把 parsed_intent 与完整 compiled_cig JSON belief 写入注入的 CRG repository，并读取同 run 同 intent 的 compiled_cig 复用 CIG/HCIV/intent_cone，GeneratorCoordAgent 可把 selected_generators belief 写入注入的 CRG repository，并在 auto 路由中读取同一 run 的既有 selected_generators、失败类 CRG belief 或 route_humu_embedding 影响 generator 选择和反馈，ValidationAgent 可把 validation_status belief 写入注入的 CRG repository，并读取同 run 同分子的既有 validation_status 跳过重复 oracle cascade，RetroSynAgent 可把 retrosyn_routes 与 route_humu_embedding belief 写入注入的 CRG repository，并读取 failed validation belief 或 `retrosyn_routes=0` 跳过路线规划，SupplyAgent 可把 supply_feasibility belief 写入注入的 CRG repository，并读取同 run 同分子的既有 supply_feasibility 跳过重复 supply oracle，也可读取 `retrosyn_routes=0` belief 直接输出 unavailable，SRBAgent 可把 ssp_compiled belief 写入注入的 CRG repository，并读取 unavailable supply_feasibility 跳过 SSP 编译，CriticAgent 可把 critic_verdict belief 写入注入的 CRG repository，并读取同 run 同分子的既有 critic_verdict 跳过重复规则评估，也可读取 validation/supply 失败 belief 与 `retrosyn_routes=0` 影响 verdict；这些 agent 已默认使用 shared CRG repository env factory，并继承 shared CRG 读回接口，重复执行与失败反馈级 CRG 读回已覆盖上述 agent，BaseAgent 已支持将 mapping payload 编码为 JSON-LD，发布带 signature 的 AgentMessage envelope，并默认生成 UUIDv7 message_id，订阅接收端可自动解包、校验 recipient、message_type、payload_type_url 与 ttl 防循环、按 sender identity 验签，可验证篡改，并可通过 SIGSTORE_SIGN_COMMAND/SIGSTORE_VERIFY_COMMAND 外接 Sigstore/Rekor 命令，签名命令可接收 SIGSTORE_IDENTITY_TOKEN，验证命令可接收 sender、recipient、message_type 与 expected_identity；后续仍缺更深层跨 agent 联合优化、真实 Fulcio/Rekor 命令、生产身份令牌投放和外部系统闭环。
6. 已让 SupplyAgent 聚合 supply client 返回的 building block availability、价格、交期和供应来源，并可把 supply_feasibility belief 写入注入的 CRG repository；Supply Oracle 支持本地 JSON catalog 和 AiZynth HDF5 stock，显式 `SUPPLY_CATALOG_URI` 会限制为当前真实支持的 `file://` catalog URI；`orchestrator-svc` 已为内联 SupplyAgent 配置 `SUPPLY_ORACLE_TARGET`；缺 supply client 或 `SUPPLY_ORACLE_TARGET` 时 fail-fast。
7. 已让 NL2ObjAgent 复用本地 parser 输出 parsed intent、targets、activity 和 constraints，支持把 parsed_intent belief 写入注入的 CRG repository，并在 CIG compiler 返回结果时把 compiled_cig belief 写入同一 repository；同时支持通过注入 client 或 `CIG_COMPILER_TARGET` 直连 CIG compiler 产出 CIG、HCIV 和 intent cone；CIG compiler production semantic parser adapter 已支持外接 Python、HTTP 或 stdin/stdout command LLM/SRM parser，grounding stage 已写入 UniProt/PDB/ChEMBL evidence，部署清单已暴露对应 grounding endpoint env，`Refine` RPC 已支持 `CIG_REFINEMENT_COMMAND` JSON runner，配置 semantic parser / refinement command 时会先校验首个可执行文件可用，仓库内置完整 LLM/SRM 模型与真实 refinement runner command/env 仍未接入。
8. 已让 GeneratorCoordAgent 支持注入 generator clients、target map、HTTP/Python discovery provider 和环境变量 registry，并 dispatch 到选中的 generator backend；FullWorkflowClients 在显式 `generation_strategy` 请求下已可委托该 agent，并会把 `generation_feedback` 作为 `generator_params.generation_feedback` 透传；同时可把 selected_generators belief 写入注入的 CRG repository；auto 路由已能优先读取既有 selected_generators，读取失败类 CRG belief 影响 generator 选择，并能读取 route_humu_embedding belief 后把 route HUMU feedback 写入 `generator_params.route_humu_feedback`；调度前健康探测已接入，`orchestrator-svc` 部署清单已暴露 discovery/target env，真实集群注册中心部署仍未接入。

### P2：补齐研究型算法能力

1. 分子编码器升级到 3D SE(3)/E(3) 等变网络；当前已支持显式 3D conformer 坐标的 E(3)-invariant 距离统计，但不是最终 SE(3)/E(3) message passing。
2. Route encoder 升级到反应树图网络或 TreeLSTM；当前已加入 route tree topology 统计特征，但不是最终反应树图 Transformer 或双曲 TreeLSTM。
3. HFM 推理路径已接入 Lorentz flow ODE solver、可注入分子几何 decoder 接口和 `HFM_MOLECULAR_DECODER_COMMAND` 外部 JSON decoder runner；模型级 external decoder wrapper 和 hfm-generator-svc runtime status 均会校验 external decoder command 的首个可执行文件可用；默认 decoder artifact 训练写入路径已可持久化 entry-level SDF 几何，推理路径会先校验 artifact SDF 可解析且匹配 SMILES，再优先复用 artifact SDF，缺失时回退到运行时 conformer；Owner A 已新增本地 `mf_generators.hfm_3d.decoder.neural_geometry_decoder` 与 `train_geometry_decoder.py`，可从 SDF-backed decoder entries 训练/export torch geometry decoder artifact，并可作为 `python -m ... --artifact` stdin/stdout command target 返回现有 HFM molecular decoder JSON schema；hfm-generator-svc 部署 wiring 已补齐，Compose、静态 Kubernetes 清单和 Helm values/template 已默认指向当前仓库内的 `checkpoints/hfm3d_4h200/best_model.pt` 与 `checkpoints/hfm3d_4h200/decoder.json`，当前已检入 decoder artifact 也已包含 entry-level `sdf` 字段；仍需训练并投放真实生产质量 neural geometry decoder artifact 或 external decoder command 值，并完成集群发布和几何质量验证。
4. FragFM 已接入两层 DFM、SA-aware transition matrix、rule-level HUMU/intent_cone alignment ranking、可注入共享 HUMU latent sampler，并在 `fragfm-generator-svc` 构造时接入 intent-cone HUMU latent sampler；training CLI 已可消费显式 teacher embedding artifact 并加入 KD loss；Compose、静态 Kubernetes 清单和 Helm values/template 已默认接入当前本地 FragFM vocabulary/checkpoint/rate-matrix artifact，且这些本地 artifact 文件已存在；仍需生产质量验收和集群发布验证。
5. PCBO 已补齐批量 constrained HVI 候选排序工具、EHVI、HUMU log-map 切空间映射、tangent-space RBF GP constrained HVI/EHVI/PoF ranking、库级异步 oracle 采样循环、多轮 optimization scheduler、独立 `pareto_bo` service package、env-configured callable CLI 入口、`PARETO_BO_CANDIDATE_PROVIDER_COMMAND` / `PARETO_BO_ORACLE_EVALUATE_COMMAND` JSON runner 接入点、command executable preflight、FastAPI optimize endpoint、`pareto-bo-svc` Compose/Kubernetes/Helm env wiring 和空 provider/oracle command ConfigMap 数据；仍需投放真实 candidate provider/oracle evaluator command/env 并完成生产验收。
6. Cross-Paradigm KD 已在 GeneratorRouterService feedback 中接入 oracle teacher records，并支持 `HYPSEEK_TEACHER_COMMAND` 外部 HypSeek teacher JSON runner 与 `HYPSEEK_TEACHER_URL` 外部 HypSeek HTTP endpoint；`generator_router_svc.main:hypseek_app` 已提供可独立运行的 HypSeek `/teacher` HTTP endpoint 和 `/healthz` health endpoint；GeneratorRouter 已能对配置的 HypSeek teacher command 与 TAR Proxyless search command 执行 executable preflight；KD 层已支持归一化 teacher distribution、Boltz2 ΔG/per-member ΔG adapter、HypSeek 显式 score-field adapter 和 teacher embedding target 蒸馏损失，HFM-3D、FragFM、UAS、CReM 与 MMPT CLI 已可使用该 loss；iCLM service update 路径已能通过 `ICLM_UPDATE_COMMAND` 承接外部 EWC/KD runner，且会在 runtime status 和实际执行该 command 前校验首个可执行文件可用，也可在注入 generator online learner 时本地执行 update；默认 iCLM OnlineLearner 本体已直接接入 teacher embedding KD loss；Compose、静态 Kubernetes 清单和 Helm values/template 已默认接入当前本地 iCLM model path，并补齐 `hypseek-teacher-svc`、router URL、router command/timeout、health probes 和 iCLM update runner env wiring，仍需真实集群发布验证。

## 16. 当前不能声称完成的事项

基于本次核对，以下事项不能在当前状态下声称已完成：

1. JMCG 联合流形共生成；当前只补到了 HFM-3D 对 route/generation HUMU embedding feedback 的局部 Lorentz latent steering，还不是 `(m,r,p)` 联合生成模型。
2. 生产级 CIG 编译链路；当前已有 JSON-LD `@context`、objective edges、directed hyperedges、外部 Python/HTTP/command semantic parser 接入点、command executable preflight、UniProt/PDB/ChEMBL grounding stage、grounding endpoint env wiring、`CIG_REFINEMENT_COMMAND` JSON refinement runner 接入点、HCIV supervised train/export 工程路径和 Compose/Kubernetes/Helm env wiring，但仍缺仓库内置 LLM/SRM 模型、真实 refinement runner command/env 和训练好的 `Enc_intent` checkpoint 值。
3. 生产级训练好的 `Enc_intent` HCIV checkpoint；当前已有基于 CIG 节点、边和 hyperedge 的 directed message-passing encoder baseline，并已补 `cig_compiler_svc.domain.hciv_training` 与 `train_hciv_encoder.py` 本地 supervised train/export 路径。
4. SE(3)/E(3) HUMU 三塔编码器；当前 mol tower 已支持显式 conformer 距离统计，pocket tower 已支持 E(3)-invariant 局部距离统计，route tower 已支持路线树拓扑统计，三者可以继续作为现阶段训练实现，但不是最终等变三塔。
5. HFM-3D 端到端训练好的生产 decoder artifact；当前推理 latent 已经过 Lorentz flow steps，可消费 route/generation HUMU embedding feedback 做有界 latent steering，并可通过注入式 molecular decoder 或 `HFM_MOLECULAR_DECODER_COMMAND` 外部 JSON runner 消费 post-flow latent、返回 atom types/coordinates 并直接写入 SDF，执行 external decoder command 前会校验首个可执行文件可用；默认 artifact 写入和读取路径已支持 entry-level SDF 几何复用，并在载入时校验 SDF 可解析且匹配 SMILES，缺失时回退到运行时 conformer，当前已检入默认 artifact 已包含 entry-level `sdf` 字段；Owner A 已补本地 neural geometry decoder train/export/runner 工程路径和 focused pytest；hfm-generator-svc 部署 wiring 已补齐，Compose、静态 Kubernetes 清单和 Helm values/template 已接入当前本地 HFM checkpoint 和 decoder 默认路径；但当前仍未训练/投放真实生产质量 neural geometry decoder artifact，`HFM_MOLECULAR_DECODER_COMMAND` 生产值和集群发布验证尚未完成。
6. FragFM 与共享 HUMU 条件空间的完整生产闭环；两层 DFM、SA-aware transition matrix、rule-level HUMU/intent_cone alignment ranking、可注入共享 HUMU latent sampler、`fragfm-generator-svc` intent-cone sampler wiring、vocabulary/checkpoint/rate-matrix/HUMU-curvature 部署 env wiring 和 teacher embedding KD loss 训练入口已存在，Compose、静态 Kubernetes 清单和 Helm values/template 已默认指向当前本地 FragFM artifact，且本地 vocabulary/checkpoint/rate-matrix 文件已存在；但生产质量验收和集群发布验证尚未完成。
7. CReM-pharm-3D 生产 DiffDock-L/pharmacophore/HUMU scorer 闭环；当前已有可注入 docking scorer、`OracleService`/DiffDock-L gRPC scorer、批量 docking score 重排、注入式 3D pharmacophore scorer 重排、HUMU embedding intent alignment 重排能力、service env 装配入口、pharmacophore/HUMU scorer command executable preflight 和 Compose/Kubernetes/Helm env wiring，Compose、静态 Kubernetes 清单和 Helm values/template 已默认接入当前本地 MMP JSON artifact；但真实 DiffDock-L、pharmacophore 与 HUMU scorer runner 值和集群发布验证尚未完成。
8. MMPT-RAG 完整专利 RAG 对比解码；当前已完成 index retrieved seed/product evidence 生成、本地 patent negative SMILES exact 过滤、positive/negative evidence contrastive ranking、`MMPT_PATENT_RAG_COMMAND` 外部 JSON retriever runner 接入点、`MMPT_SEQ2SEQ_DECODER_COMMAND` 外部 JSON decoder runner 接入点、模型级 command executable preflight、`mmpt-generator-svc` 的本地 `file://` index URI runtime fail-fast，以及 Compose/Kubernetes/Helm env wiring，Compose、静态 Kubernetes 清单和 Helm values/template 已默认接入当前本地 MMPT index；但仍未投放真实 Seq2Seq Transformer artifact、真实专利 RAG 检索服务 artifact、生产 command 值和集群发布验证。
9. TAR 完整 ProxylessNAS 搜索训练闭环；当前已接入 REINFORCE-style 在线 policy 更新、ProxylessNAS-style architecture gate、期望资源代价接口、reward-cost architecture optimizer step、多 dataset/多轮 `ProxylessSearchScheduler`、`RunProxylessSearch` gRPC 入口、`TAR_PROXYLESS_SEARCH_COMMAND` 外部 JSON training runner、`python -m generator_router_svc.tar_proxyless_runner` 本地 runner target 和 Compose/Kubernetes/Helm env wiring；但真实训练数据集、生产环境 command 值投放和集群发布验证仍未完成。
10. Cross-Paradigm KD 完整生产级 teacher-student 蒸馏；当前 GeneratorRouterService feedback 已消费 oracle teacher score，也可通过 `HYPSEEK_TEACHER_COMMAND` 或 `HYPSEEK_TEACHER_URL` 消费外部 HypSeek teacher，`generator_router_svc.main:hypseek_app` 已提供可独立运行的 HypSeek `/teacher` HTTP endpoint 和 `/healthz` health endpoint，GeneratorRouter 已对配置的 HypSeek/TAR 外部 command 执行 executable preflight，KD 层已能消费归一化 teacher distribution、把 Boltz2 ΔG/per-member ΔG 与 HypSeek 显式 score-field 适配为 teacher_distribution，并对 teacher embedding target 计算 distillation loss，HFM-3D、FragFM、UAS、CReM 与 MMPT CLI 已接入该 loss，iCLM service update 路径已支持外部 EWC/KD runner、runtime status 与实际执行前 command executable preflight 和注入式 online learner update，默认 iCLM OnlineLearner 已能直接计算 teacher embedding KD loss；Owner A 已新增 `mf_core.routing.kd_artifacts`，可从 JSON/JSONL teacher records 导出 canonical `cross_paradigm_teacher_embeddings.v1` 并做 finite/dimension/min-count preflight；HypSeek teacher app 的 Compose/Kubernetes/Helm 部署配置、router URL、router command/timeout、health probes 和 iCLM model/update runner env wiring 已补齐，但仍需真实 production teacher source、蒸馏训练质量、benchmark 和真实集群发布验证。
11. CRG 作为所有 agent 的持久化共享一致性状态空间；当前完成了 orchestrator workflow state、provenance metadata、workflow CRG 的 Neo4j belief/edge 写入和按 run_id 读回，并补齐了 OrchestratorAgent 的 workflow_status belief、NL2ObjAgent 的 parsed_intent 与完整 compiled_cig JSON belief、GeneratorCoordAgent 的 selected_generators belief、ValidationAgent 的 validation_status belief、RetroSynAgent 的 retrosyn_routes 与 route_humu_embedding belief、SupplyAgent 的 supply_feasibility belief、SRBAgent 的 ssp_compiled belief 与 CriticAgent 的 critic_verdict belief repository 写入点；这些 agent 已默认使用 shared CRG repository env factory，并继承 shared CRG 读回接口，OrchestratorAgent 已能消费 completed workflow_status，NL2ObjAgent 已能消费同 run 同 intent 的 compiled_cig belief，GeneratorCoordAgent auto 路由已能消费既有 selected_generators、失败类 CRG belief 和 route_humu_embedding belief 并传给 generator dispatch，ValidationAgent 已能消费既有 validation_status，RetroSynAgent 已能消费 failed validation belief 和 `retrosyn_routes=0`，SupplyAgent 已能消费既有 supply_feasibility 和 `retrosyn_routes=0` belief，SRBAgent 已能消费 unavailable supply_feasibility，CriticAgent 已能消费既有 critic_verdict、validation/supply 失败 belief 和 `retrosyn_routes=0`；上述 agent 的重复执行与失败反馈级 CRG 读回已接入；BaseAgent JSON-LD payload 编码、signed AgentMessage envelope 发布、UUIDv7 message_id 默认生成、订阅接收端解包、recipient、message_type、payload_type_url 与 ttl 防循环校验和跨 agent 验证 helper 已接入，并可外接 SIGSTORE_SIGN_COMMAND/SIGSTORE_VERIFY_COMMAND；更深层跨 agent 联合优化、真实 Fulcio/Rekor 命令、生产身份令牌投放和外部系统闭环仍未完成。
12. CoreArchitecture v2 原文中的 NATS JetStream 路线；当前项目已改用 Redis，不作为本阶段补齐目标。
13. Sigstore/Rekor 真实签名审计链生产部署；当前 BaseAgent 已有 JSON-LD payload 编码、signed AgentMessage envelope 发布、UUIDv7 message_id 默认生成/订阅接收端解包、recipient、message_type、payload_type_url 与 ttl 防循环校验和跨 agent 验证 helper，并支持 `SIGSTORE_SIGN_COMMAND` 和 `SIGSTORE_VERIFY_COMMAND` 外部命令，签名命令可接收 `SIGSTORE_IDENTITY_TOKEN`，验证命令可接收 sender、recipient、message_type 与 expected_identity，且 BaseAgent 执行签名/验证命令前会校验首个可执行文件可用；`mf_agents.lineage.SigstoreSigner` 已支持同一组外部签名/验证命令并传递 payload hash、identity、identity token 与 Rekor URL，未配置时保留本地 HMAC-SHA256 fallback，且执行签名/验证命令前会校验首个可执行文件可用；provenance-svc 已有 `SIGSTORE_SIGN_COMMAND` 外部签名入口、`SIGSTORE_VERIFY_COMMAND` 外部验证入口、identity token/expected identity 上下文、`SIGSTORE_REKOR_URL` 配置和 Rekor bundle 读回，签名/验证命令执行前会校验首个可执行文件可用；Compose/Kubernetes/Helm 已补齐 provenance-svc 的 Sigstore env / ConfigMap / Secret wiring，并声明空 command/identity 默认数据；仍缺真实 Fulcio/Rekor command 值、生产身份令牌与 expected identity 实际投放，以及集群发布验证。
14. L4 GPU4PySCF/ORCA 本地量子校正生产部署；当前已有外部 `L4_QUANTUM_ORACLE_TARGET` 接入点、通用 `L4_QUANTUM_ORACLE_COMMAND` JSON command wrapper、执行前 executable preflight，以及 `L4_GPU4PYSCF_COMMAND` / `L4_ORCA_COMMAND` 命名 command wiring，`orchestrator-svc` 的 Compose/Kubernetes/Helm env wiring 已补齐；但真实 GPU4PySCF/ORCA runner command、artifact 值和集群发布验证仍未完成。
15. 完整 PCBO 优化服务；当前已有 HV、PoF、约束 HVI、批量 constrained HVI 候选排序、EHVI、HUMU log-map 切空间映射、tangent-space RBF GP constrained HVI/EHVI/PoF ranking、库级异步 oracle 采样循环、多轮 optimization scheduler、独立 `pareto_bo` service package、env-configured callable CLI 入口、外部 JSON command runner 接入点、command executable preflight、FastAPI optimize endpoint 和 `pareto-bo-svc` 部署 wiring，但真实 candidate provider/oracle evaluator command/env 与生产验收未完成。
16. Supply Oracle 本地供应目录；当前支持本地 JSON catalog、AiZynth HDF5 stock、runtime status、启动 preflight、服务入口 env client 构建前 preflight 和非 `file://` catalog URI fail-fast。
17. MOSES/GuacaMol/PMO/CrossDocked 全量 benchmark。
18. KRAS G12C 真实 pilot 全链路跑通；当前只补到 `KRAS_E2E_SCOPE=engineering` 的本地 resource-light scope 可非 skip 执行，其中 HFM/Boltz/AiZynth full-only 步骤在 engineering scope 中显式 skip；设置当前本地 HFM/Boltz/AiZynth artifact env 后，full scope preflight 仍缺 service ready、外部 DKI、provenance 与 Sigstore 生产依赖，真实 full pilot 仍未完成。

第 16 节逐项证据映射：

| 编号 | 未完成事项 | 当前证据 | 剩余 gate |
|---|---|---|---|
| 1 | JMCG 联合流形共生成 | HFM-3D 已消费 `route_humu_feedback` / `generation_feedback` / `jmcg_feedback` 做局部 Lorentz latent steering；服务与编排层可继续透传 generator params；`moleculeforge/docs/todo/2026-06-03-jmcg-feedback-contract-brief.md` 已起草 future feedback envelope 和 legacy mapping；GeneratorCoordAgent 已产出 contract-shaped route feedback；默认 HFM 不直接读取 shared CRG 的决策已确认；`moleculeforge/docs/todo/2026-06-03-jmcg-feedback-semantics-gate.md` 已记录本地 weight/confidence、polarity、per-kind aggregation 和 dropped-record metadata 语义增强；`moleculeforge/docs/todo/2026-06-03-property-feedback-producer-gate.md` 已记录 non-steering property context producer；`moleculeforge/docs/todo/2026-06-03-pocket-intent-feedback-producer-gate.md` 已记录 non-steering pocket / intent context producer；Owner A W2 已补 pocket/intent steering producer：intent 仅接受已有 finite 且满足 Lorentz hyperboloid 方程的 129 维 full-coordinate axis，pocket 仅在结构化 pocket geometry 可由 `HUMU_ENCODER_TARGET` 编码时带 finite 且满足 Lorentz hyperboloid 方程的 129 维 HUMU embedding；Owner A W8-E 已补 `JMCGEngineeringSampler` 本地 joint sample skeleton，可输出 `moleculeforge.jmcg.joint_sample.v1` engineering skeleton records；Owner A 已补 shared Lorentz embedding validation hardening，W2/HFM/W8-E 对非法 129 维 embedding fail closed，文件级 focused pytest 已通过 | 缺真实联合采样训练模型、W8-R 质量证据、端到端生产验证 |
| 2 | 生产级 CIG 编译链路 | `uv run pytest tests/unit/test_cic_compiler.py -q` 退出码 0，覆盖 JSON-LD、hyperedges、grounding、refinement command、HCIV baseline 和 W10 train/export smoke | 缺仓库内置 LLM/SRM、真实 refinement runner、训练好的 `Enc_intent` checkpoint |
| 3 | 生产级 `Enc_intent` HCIV checkpoint | CIG learned encoder baseline 已有 node/edge/hyperedge message passing；Owner A 已补 supervised JSON/JSONL train/export 工程路径，focused W10 gate 4 项通过，`tests/unit/test_cic_compiler.py` 文件级 31 项通过 | 缺真实 supervised CIG/HCIV 数据、`HCIV_CHECKPOINT_PATH` 指向的 production-quality checkpoint、集群验收和下游质量验证 |
| 4 | SE(3)/E(3) HUMU 三塔 | mol/pocket/route tower 已补 E(3)-invariant 或 topology 统计特征 | 缺最终 SE(3)/E(3) message passing、TreeLSTM/Transformer 结构和训练验收 |
| 5 | HFM-3D 生产 decoder | 本地 checkpoint/decoder artifact 存在，HFM benchmark smoke 可非 skip 执行；Owner A 已补 `mf_generators.hfm_3d.decoder.neural_geometry_decoder` 和 `train_geometry_decoder.py` 本地 train/export/runner 工程路径，focused W9 + legacy decoder gate 6 项通过，`tests/unit/test_generators.py` 文件级 65 项通过 | 缺真实 production-quality neural geometry decoder artifact、`HFM_MOLECULAR_DECODER_COMMAND`/artifact 生产投放、集群验收和几何质量 benchmark |
| 6 | FragFM 共享 HUMU 条件空间生产闭环 | FragFM vocab/checkpoint/rate-matrix 本地 artifact 存在，service 已接 intent-cone sampler；Owner A 已新增 training-time HUMU embedding 保留/校验和 `mf_generators.fragfm.quality` coverage/loadability report | 本地工程质量门已落地；当前本地 artifact HUMU coverage=0，仍缺真实 HUMU-labeled production artifact、正式质量阈值、benchmark 和集群发布验证 |
| 7 | CReM-pharm-3D 生产 scorer 闭环 | CReM service 已有 docking/pharmacophore/HUMU scorer command preflight 和本地 MMP artifact wiring | 缺真实 DiffDock-L、pharmacophore、HUMU scorer runner 值和集群验证 |
| 8 | MMPT-RAG 专利 RAG 对比解码 | `uv run pytest tests/unit/test_service_artifact_status.py -k mmpt -q` 退出码 0；本地 `file://` index 与 command preflight 已验证 | 缺真实 Seq2Seq Transformer、专利 RAG 服务、生产 command 和集群验证 |
| 9 | TAR ProxylessNAS 搜索训练闭环 | TAR 已有 scheduler、architecture gate、reward-cost update、`RunProxylessSearch` 接口和 `python -m generator_router_svc.tar_proxyless_runner` 本地 command target；命令级 smoke 已验证 stdin payload 可输出 `rounds`、`architecture_probabilities` 和 `architecture_logits`；W6 focused pytest 和 `tests/unit/test_task_router.py` 文件级 pytest（30 项）已通过 | 缺真实训练数据集、生产环境 `TAR_PROXYLESS_SEARCH_COMMAND` 值投放和集群验证 |
| 10 | Cross-Paradigm KD 生产蒸馏 | `uv run pytest tests/unit/test_cross_paradigm_kd.py -q` 退出码 0；KD layer、HypSeek app 和 iCLM update wiring 已验证 | 缺生产 teacher 服务和真实集群发布验证 |
| 11 | CRG 持久化共享一致性状态 | `uv run pytest tests/unit/test_graph_repo.py -q` 退出码 0；agent CRG read/write 本地路径已覆盖 | 缺真实 Neo4j/DKI、Fulcio/Rekor identity 和外部系统闭环验证 |
| 12 | NATS JetStream 路线 | 当前项目已确认 Redis 替代 NATS | 非本阶段补齐项 |
| 13 | Sigstore/Rekor 生产审计链 | `uv run pytest tests/unit/test_provenance.py -q` 退出码 0；BaseAgent/provenance-svc command preflight 已覆盖；`uv run pytest tests/test_mvp_pipeline.py::TestMVPPipeline::test_sigstore_signer_sign_command_preflight_rejects_missing_executable tests/test_mvp_pipeline.py::TestMVPPipeline::test_sigstore_signer_verify_command_preflight_rejects_missing_executable tests/test_mvp_pipeline.py::TestMVPPipeline::test_sigstore_signer_uses_configured_commands -q` 退出码 0，3 项通过 | 缺真实 Fulcio/Rekor command、identity token、expected identity 和集群验证 |
| 14 | L4 GPU4PySCF/ORCA 生产部署 | `uv run pytest tests/unit/test_validation_agent.py -k "quantum_command or l4_quantum" -q` 退出码 0，6 项通过；ValidationAgent 已有 `L4_QUANTUM_ORACLE_COMMAND`、`L4_GPU4PYSCF_COMMAND`、`L4_ORCA_COMMAND` wiring、JSON parsing 和缺 executable preflight | 缺真实 command、artifact 和集群验证 |
| 15 | 完整 PCBO 优化服务 | `uv run pytest tests/unit/test_mf_eval.py -q` 退出码 0；PCBO service、runner、FastAPI 和 deployment wiring 已覆盖 | 缺真实 candidate provider/oracle evaluator command/env 与生产验收 |
| 16 | Supply Oracle 本地供应目录 | `uv run pytest tests/unit/test_service_artifact_status.py -k supply -q` 退出码 0；本地 JSON catalog、AiZynth HDF5 stock 和 fail-fast 已覆盖 | 缺集群发布验证 |
| 17 | MOSES/GuacaMol/PMO/CrossDocked 全量 benchmark | 默认 `tests/benchmark` 全 skip；临时资源 smoke 可跑 18 项无 skip | 缺正式 benchmark 数据、正式阈值和生产 artifact 质量验收 |
| 18 | KRAS G12C 真实 full pilot | engineering scope 3 pass / 3 full-only skip；默认 full scope 命令退出码 1 且 6 项 preflight error；设置本地 HFM/Boltz/AiZynth artifact env 后 full scope pytest 仍退出码 1，6 项 setup error | 缺 service ready、DKI、provenance 和 Sigstore 生产依赖 |

## 17. Completion audit

本轮目标不是单纯改写本文档，而是按本文“当前实现 vs CoreArchitecture v2 目标预期”的对照继续补齐未实现项。完成判断必须同时满足以下交付条件：

目标拆解为四个可验收交付物：

1. 指定文件 `docs/architecture/current-implementation-vs-corearchitecture-v2.md` 保持为当前实现和 CoreArchitecture v2 目标预期的事实对照。
2. 对照中标出的本地可写、无需外部生产资源的工程接入点已经补齐，并有代码或测试证据。
3. 仍依赖真实模型 artifact、runner command、凭据、外部 DKI 或集群环境的事项明确保留为未完成，不能用本地 smoke 结果替代。
4. 目标完成判断必须由实际命令和文件证据支撑；测试通过、manifest 存在或文档记录不能单独视为架构完成。

Prompt-to-artifact checklist：

| 要求 / gate | 证据 artifact | 覆盖结论 |
|---|---|---|
| 指定文档必须继续作为对照源 | `docs/architecture/current-implementation-vs-corearchitecture-v2.md` 第 2-16 节 | 已覆盖当前和目标预期对照 |
| 不新增额外说明文档 | 本轮只修改指定对照文档、代码和既有测试文件 | 已满足 |
| 本地 command runner preflight 不被 artifact 掩盖 | `services/admet-svc/src/admet_svc/main.py`、`services/dock-svc/src/dock_svc/main.py`、`services/mmpt-generator-svc/src/mmpt_generator_svc/main.py`、`services/retrosyn-svc/src/retrosyn_svc/main.py`、`services/supply-oracle-svc/src/supply_oracle_svc/main.py` 与 `tests/unit/test_service_artifact_status.py` | 已补齐可本地验证路径 |
| Orchestrator import 不应在非 reasoning 路径提前触发 LangGraph warning | `agents/orchestrator/src/orchestrator/workflow/graph_builder.py`、`tests/e2e/test_predict_api.py::test_orchestrator_import_does_not_eagerly_load_langgraph` | 已补齐回归覆盖 |
| JMCG 本地联合生成语义变更 | `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py` 当前已解析 `route_humu_feedback`、`generation_feedback` 与 `jmcg_feedback`；`services/hfm-generator-svc/src/hfm_generator_svc/main.py`、`services/orchestrator-svc/src/orchestrator_svc/main.py` 和 `agents/generator_coord/src/generator_coord/agent.py` 已能透传 generator params；`agents/generator_coord/src/generator_coord/agent.py` 已把 route HUMU feedback 同步封装为 contract-shaped envelope；`services/orchestrator-svc/src/orchestrator_svc/main.py` 已把 workflow feedback 派生为 non-steering property context records，并把 eligible intent / pocket context 升级为 finite 且满足 Lorentz hyperboloid 方程的 129 维 steering-capable records；GeneratorCoordAgent 已能合并 property / intent / pocket 与 route records；HFM-3D 本地 steering 已补初始 weight/confidence、polarity、per-kind aggregation、dropped-record metadata 和 shared Lorentz embedding validation；`mf_generators.hfm_3d.inference.jmcg_sampler` 已补 W8-E engineering skeleton joint sample output；相关 gate 文档已记录 contract 与实施边界 | 当前仍只是工程骨架 + HUMU steering，缺 W8-R 真实联合采样训练质量和端到端生产验证；未完成 |
| 测试不应写入仓库默认 SQLite DB | `tests/conftest.py` 的临时 `MF_DB_PATH` 隔离记录在第 12 和第 17 节 | 已补齐边界 |
| 安全加载 torch checkpoint | `rg -n "torch\\.load\\(" models pipelines services agents libs tests` 和相关 generator / HUMU training 测试 | 已补齐显式 `weights_only=True` 证据 |
| anti-degradation 质量门 | `uv run pytest tests/anti_degradation -q` | 8 项通过 |
| Unit / E2E / integration / full pytest 不能替代生产验收 | 第 16 节和第 17 节记录 skip、warning、外部 env 与 artifact 缺口 | 未完成项已保留 |
| Audit completeness 生产验收 | `env RUN_AUDIT_E2E=1 uv run pytest tests/e2e/test_audit_completeness.py -q` | 4 项 preflight error，缺 provenance、Sigstore、OTel 和 DKI 生产依赖，未完成 |
| KRAS G12C full pilot 生产验收 | `env RUN_KRAS_G12C_E2E=1 KRAS_E2E_SCOPE=full uv run pytest tests/e2e/test_kras_g12c_pilot.py -q`；设置当前本地 HFM/Boltz/AiZynth artifact env 后重新运行同一 full scope pytest | 默认 env 下 6 项 preflight error；本地 artifact env 下仍退出码 1，6 项 setup error，缺 service ready、DKI、provenance 和 Sigstore 生产依赖，未完成 |
| `mf_agents.lineage.SigstoreSigner` executable preflight | `libs/mf-agents/src/mf_agents/lineage/sigstore_signer.py` | 已完成 |
| Pydantic V2 config 更新 | `libs/mf-core/src/mf_core/types/humu.py` | 已完成 |
| 生产资源验收 | `HCIV_CHECKPOINT_PATH`、CIG runner、HFM/MMPT/TAR runner、Sigstore、L4 quantum 和 DKI env | 当前环境未投放，未完成 |

最新阻塞复核：`sudo -n chown FWY:FWY libs/mf-agents/src/mf_agents/lineage/sigstore_signer.py libs/mf-core/src/mf_core/types/humu.py libs/mf-core/src/mf_core/types/__init__.py libs/mf-core/src/mf_core/types/hciv.py` 退出码 0；上述四个文件现在均为 `FWY:FWY 664` 且当前用户可写。关键生产 env 复核仍为 unset。

改动范围复核：`git diff --name-only -- docs/architecture/current-implementation-vs-corearchitecture-v2.md agents/orchestrator/src/orchestrator/workflow/graph_builder.py tests/e2e/test_predict_api.py services/admet-svc/src/admet_svc/main.py services/dock-svc/src/dock_svc/main.py services/mmpt-generator-svc/src/mmpt_generator_svc/main.py services/retrosyn-svc/src/retrosyn_svc/main.py services/supply-oracle-svc/src/supply_oracle_svc/main.py tests/unit/test_service_artifact_status.py libs/mf-agents/src/mf_agents/lineage/sigstore_signer.py libs/mf-core/src/mf_core/types/humu.py tests/test_mvp_pipeline.py` 列出 12 个既有文件；`find docs -maxdepth 2 -type f -name '*.md' -newermt '2026-06-02 00:00:00'` 只列出本指定对照文档。`git status --short -- README.md ../README.md docs/architecture/current-implementation-vs-corearchitecture-v2.md docs/architecture/current-architecture-readme.md` 显示 `../README.md` 有未提交修改、`docs/architecture/current-architecture-readme.md` 为未跟踪文件，均未纳入本次补齐范围；README 更新需等待用户确认。

| 交付条件 | 当前证据 | 状态 |
|---|---|---|
| 不新建额外说明文档 | 本轮继续维护指定对照文件 `docs/architecture/current-implementation-vs-corearchitecture-v2.md`，并在现有代码/测试中补齐可本地验证的工程接入点；未为本轮 audit 新建额外说明文档 | 已满足 |
| 当前实现与目标预期有逐层对照 | 第 2-12 节覆盖 JMCG、CIC/HCIV、HUMU、AMGE、MARB、Oracle/PCBO、SRB/Supply、DKI、工程实施和 benchmark | 已满足 |
| 已补齐不依赖外部资源的工程接入点 | 已接入多处 command runner executable preflight、agent gRPC 默认 event loop 保护、HFM/MMPT/CReM/iCLM 执行前校验、RetroSyn JSON planner ensemble 启动 preflight、BaseAgent/provenance-svc Sigstore 执行前校验，以及 pytest 临时 `MF_DB_PATH` 隔离，并在第 3-16 节记录 | 部分满足 |
| 已验证本轮相关改动 | 已完成 ADMET、Dock、Supply、MMPT、RetroSyn 等 runtime/preflight 聚焦验证，以及 orchestrator LangGraph import lazy-load 回归验证；`uv run pytest tests/unit -q` 退出码 0，pytest 输出 598 items 到 `[100%]`；`uv run pytest -q` 退出码 0，pytest 输出 694 items 到 `[100%]`，38 个 skip 均为 benchmark、audit/KRAS 开关或外部 DKI 环境缺失；相关 `uv run ruff check ... --select E,F,I,W,B,UP` 与 `git diff --check -- ...` 均退出码 0 | 已满足 |
| 不把测试通过当作整体完成 | 第 16 节仍列出需要真实 artifact、runner command、凭据或集群验证的缺口 | 已满足 |
| 权限阻塞已复核解除 | `sudo -n chown FWY:FWY ...` 对 `SigstoreSigner`、`humu.py`、`mf_core.types.__init__` 和 `mf_core.types.hciv` 退出码 0；`ls -l` 与 `test -w` 均显示四个文件当前为 `FWY:FWY 664` 且可写 | 已满足 |
| 外部生产资源缺口有边界 | 第 16 节列明 HFM 生产 decoder、CIG/HCIV checkpoint、DiffDock-L/pharmacophore/HUMU scorer、RAscore/RSGPT/UAlign/AiZynth runner、Seq2Seq/RAG artifact、TAR 训练 runner、Fulcio/Rekor identity、GPU4PySCF/ORCA/OpenFE、全量 benchmark 与 KRAS pilot | 未完成 |

本地 artifact/env 边界补充核对：`stat -c '%s %n'` 显示当前存在 `checkpoints/hfm3d_4h200/best_model.pt` 9886754 bytes、`checkpoints/hfm3d_4h200/decoder.json` 3523 bytes、`checkpoints/fragfm/vocab.json` 16253 bytes、`checkpoints/fragfm/best_model.pt` 25406754 bytes、`checkpoints/fragfm/rate_matrix.pt` 149552 bytes、`models/artifacts/mmpt/mmpt_index.json` 1038709 bytes、`models/artifacts/crem/crem_mmp_database.json` 469309 bytes、`models/artifacts/gnina/gnina.1.3.2.cuda12.8` 2052029472 bytes、`models/artifacts/diffdock/score_model/best_ema_inference_epoch_model.pt` 121063553 bytes、`models/artifacts/diffdock/confidence_model/best_model_epoch75.pt` 19312669 bytes、`models/artifacts/boltz-2/boltz2_aff.ckpt` 2062139170 bytes、`models/artifacts/boltz-2/boltz2_conf.ckpt` 2286561469 bytes、`models/artifacts/boltz-input-templates/6OIM.yaml` 347 bytes、`models/artifacts/aizynthfinder/config.yml` 396 bytes、`models/artifacts/aizynthfinder/uspto_model.onnx` 91518243 bytes、`models/artifacts/aizynthfinder/uspto_templates.csv.gz` 3313598 bytes、`models/artifacts/aizynthfinder/zinc_stock.hdf5` 1339073560 bytes 和 `models/artifacts/iclm/novomolgen_157m_smiles_bpe/model.safetensors` 631855376 bytes；这些只能证明本地 smoke / local runtime artifact 存在。当前环境中 `HCIV_CHECKPOINT_PATH`、`CIG_SEMANTIC_PARSER_COMMAND`、`CIG_REFINEMENT_COMMAND`、`HFM_MOLECULAR_DECODER_COMMAND`、`MMPT_PATENT_RAG_COMMAND`、`MMPT_SEQ2SEQ_DECODER_COMMAND`、`TAR_PROXYLESS_SEARCH_COMMAND`、`SIGSTORE_SIGN_COMMAND`、`SIGSTORE_VERIFY_COMMAND`、`SIGSTORE_IDENTITY_TOKEN`、`L4_QUANTUM_ORACLE_COMMAND`、`L4_GPU4PYSCF_COMMAND`、`L4_ORCA_COMMAND`、`RUN_AUDIT_E2E`、`RUN_KRAS_G12C_E2E`、`NEO4J_URI`、`MINIO_ENDPOINT_URL`、`TEST_DATABASE_URL`、`QDRANT_HOST` / `QDRANT_URL` 和 `REDIS_HOST` / `REDIS_URL` 均为 unset，因此生产 runner、凭据和外部 DKI 验收仍不能声称完成。

剩余缺口可写性补充核对：`services/cig-compiler-svc`、`services/hfm-generator-svc`、`models/mf-generators/hfm_3d`、`services/mmpt-generator-svc` 和 `models/mf-generators/mmpt_rag` 等目录当前可写；`libs/mf-core/src/mf_core/routing/task_router.py`、`libs/mf-core/src/mf_core/routing/cross_paradigm_kd.py`、`agents/orchestrator/src/orchestrator/workflow/graph_builder.py` 与 `agents/validation_agent/src/validation_agent/agent.py` 等关键文件当前可写，但其所在的 `libs/mf-core/src/mf_core/routing`、`agents/orchestrator/src/orchestrator/workflow` 和 `agents/validation_agent/src/validation_agent` 目录当前不可写。上述可写文件已覆盖第 16 节对应的 runner 接入点、runtime preflight、env wiring 或本地 smoke 路径；`SigstoreSigner` 与 `humu.py` 两个原权限阻塞点已补齐。剩余未完成项不是缺少可写 wrapper，而是缺真实生产 artifact、runner command、凭据、外部 DKI/集群环境，或待确认的 JMCG 生成语义变更。

补充验证：`uv run pytest tests/unit/test_task_router.py tests/unit/test_srb_agent.py tests/unit/test_validation_agent.py tests/unit/test_artifact_requirements.py -q` 退出码 0，pytest 输出 69 items 到 `[100%]`，覆盖 router、SRB、ValidationAgent 和 artifact requirement 的本地 wiring / preflight / fallback 边界。

Unit 全量补充验证：`uv run pytest tests/unit -q` 退出码 0，pytest 输出 598 items 到 `[100%]`；warning summary 只剩 `tests/unit/test_service_artifact_status.py::test_orchestrator_service_tracks_real_workflow_state` 在实际构建 LangGraph 时触发的第三方 LangGraph pending deprecation warning，未导致本轮单元测试失败。

Anti-degradation 补充验证：`uv run pytest tests/anti_degradation -q` 退出码 0，pytest 输出 8 项到 `[100%]`。

Torch checkpoint 加载补充修复：`models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py`、`models/mf-generators/fragfm/train.py`、`models/mf-generators/hfm_3d/train.py`、`models/mf-encoders/humu_pocket_encoder/src/mf_encoders/humu_pocket/encoder.py`、`pipelines/humu_pretrain/src/humu_pretrain/pipeline.py` 以及相关测试读取点已显式传入 `weights_only=True`；`rg -n "torch\\.load\\(" models pipelines services agents libs tests` 显示当前匹配的生产/test `torch.load` 调用均显式带 `weights_only=True`；`uv run pytest tests/unit/test_generators.py::TestFragFMGenerator::test_loads_checkpoint_and_rate_matrix_artifacts tests/unit/test_generators.py::TestFragFMGenerator::test_training_cli_writes_checkpoint_and_vocab_artifacts tests/unit/test_generators.py::TestUASGenerator::test_training_cli_writes_autoencoder_and_reference_artifacts tests/unit/test_humu_training.py::test_checkpoint_save_load tests/unit/test_humu_training.py::test_checkpoint_excludes_frozen_esm2_model_weights tests/unit/test_humu_training.py::test_step_checkpoint_saves_training_resume_metadata -q -W error::FutureWarning` 退出码 0，6 项通过；`uv run pytest tests/unit/test_generators.py::TestHFM3DTraining -q -W error::FutureWarning` 退出码 0，9 项通过。

API gateway / orchestrator LangGraph lazy-load 补充修复：`services/api-gateway/src/api_gateway/routers/reason.py` 已把 `orchestrator.pipeline.get_pipeline` 从模块级导入改为 endpoint 调用时懒加载，避免 `/health` 这类非 reasoning 路径在 import API gateway 时触发 LangGraph；本轮进一步把 `agents/orchestrator/src/orchestrator/workflow/graph_builder.py` 的 `langgraph.graph.END` / `StateGraph` 从模块级导入改为 `WorkflowGraph.build()` 内延迟导入，使 `import orchestrator` 不再因包初始化触发 LangGraph pending deprecation；新增 `tests/e2e/test_predict_api.py::test_orchestrator_import_does_not_eagerly_load_langgraph` 先以退出码 1 复现 `LangChainPendingDeprecationWarning`，修复后 `uv run pytest tests/e2e/test_predict_api.py::test_orchestrator_import_does_not_eagerly_load_langgraph -q` 退出码 0，1 项通过；`uv run pytest tests/e2e/test_predict_api.py::test_health_reports_devices -q -W error::langchain_core._api.deprecation.LangChainPendingDeprecationWarning` 退出码 0，1 项通过；`uv run pytest tests/e2e/test_reason_workbench.py::test_reason_run_completes -q -W error::langchain_core._api.deprecation.LangChainPendingDeprecationWarning` 退出码 0，1 项通过；`uv run pytest tests/e2e/test_predict_api.py tests/e2e/test_reason_workbench.py -q` 退出码 0，15 项通过；`uv run pytest tests/e2e -q` 退出码 0，25 项中 15 项通过、10 项按 audit/KRAS 显式开关 skip。

本地 warning 修复补充验证：`libs/mf-core/src/mf_core/types/humu.py` 已从 Pydantic V1 `class Config` 迁移到 V2 `ConfigDict`；`uv run python -c "import warnings; from pydantic.warnings import PydanticDeprecatedSince20; warnings.simplefilter('error', PydanticDeprecatedSince20); from mf_core.types.humu import HCIV, IntentCone; HCIV(); IntentCone()"` 退出码 0，不再触发 `PydanticDeprecatedSince20`。

Pytest 全量补充验证：`uv run pytest -q` 退出码 0，pytest 输出 694 items 到 `[100%]`；short test summary 中有 38 个 skip，原因集中在 benchmark 数据/模型 artifact 缺失、`RUN_AUDIT_E2E=1` 和 `RUN_KRAS_G12C_E2E=1` 未开启，以及 MinIO、Neo4j、PostgreSQL/provenance DB、Qdrant、Redis 外部 DKI 环境变量未配置；warning summary 只剩 `tests/test_mvp_pipeline.py::TestMVPPipeline::test_orchestrator_graph` 在实际构建 LangGraph 时触发的第三方 LangGraph pending deprecation warning；该结果只能证明默认本地测试集合无 failure/error，不能替代第 16 节的生产资源验收。

Benchmark 非 skip 补充验证：使用临时 MOSES/PMO/CrossDocked 资源文件运行 FragFM MOSES validity、PMO DRD2、PMO JNK3/GSK3B 和 CrossDocked 全文件测试，退出码 0，7 项通过；使用当前本地 HFM checkpoint/decoder 和临时 MOSES reference 运行 HFM MOSES validity、uniqueness、novelty 小批量 smoke，退出码 0，3 项通过；使用当前本地 HFM checkpoint/decoder、临时 MOSES/FragFM/PMO/CrossDocked 资源、batch size 8 和最低质量阈值运行全 `tests/benchmark` smoke，退出码 0，18 项通过且无 skip，输出包含已知 Pydantic V2 deprecation warning；这些 smoke 只验证 MOSES/GuacaMol/PMO/CrossDocked benchmark 路径可执行，不等于正式 benchmark 指标达标。

Benchmark 默认环境复核：`uv run pytest tests/benchmark -q` 退出码 0，18 项全部 skip；skip 原因包含 `CROSSDOCKED_BENCHMARK_JSONL is required`、`MOSES_REFERENCE_SMILES_PATH is required`、`FRAGFM_MOSES_GENERATED_SMILES_PATH is required`、`PMO_SCORE_TABLE_PATH is required`，以及 GuacaMol/PMO HFM 路径缺 `HFM_CHECKPOINT_PATH` 和 `HFM_DECODER_PATH`。该结果再次确认默认环境不能声称完成 MOSES/GuacaMol/PMO/CrossDocked 全量 benchmark。

Integration 补充验证：`uv run pytest tests/integration -q` 退出码 0，32 项中 22 项通过、10 项 skip；skip 来自 MinIO、Neo4j、PostgreSQL/provenance DB、Qdrant 和 Redis 外部 DKI 环境变量未配置。

Audit completeness 生产开关补充验证：`env RUN_AUDIT_E2E=1 uv run pytest tests/e2e/test_audit_completeness.py -q` 退出码 1，4 项均在 fixture preflight 阶段 error；原始缺口为 `PROVENANCE_SVC_URL`、`SIGSTORE_E2E_READY`、`SIGSTORE_IDENTITY_TOKEN`、`SIGSTORE_EXPECTED_IDENTITY`、`SIGSTORE_SIGN_COMMAND`、`SIGSTORE_VERIFY_COMMAND`、`SIGSTORE_REKOR_URL`、`OTEL_EXPORTER_OTLP_ENDPOINT`、`PROVENANCE_STORE_MODE=production_real`、`PROVENANCE_DATABASE_URL or TEST_DATABASE_URL`、`NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`、`MINIO_ENDPOINT_URL`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY` 和 `MINIO_BUCKET`。该结果证明 audit completeness 不是本地代码路径缺少 skip，而是当前环境未投放生产 provenance、Sigstore、OTel 和 DKI 依赖。

CRG repository 补充验证：`uv run pytest tests/unit/test_graph_repo.py -q` 退出码 0，11 项通过，覆盖 GraphRepository 本地 repository 行为；真实 Neo4j 环境仍需外部 DKI 环境变量投放后验证。

Oracle 补充验证：`uv run pytest tests/unit/test_l0_oracle.py -q` 退出码 0，19 项通过，覆盖 RDKit L0 score/uncertainty、PAINS metadata、ADMET uncertainty wrapper、GNINA/Boltz provenance enforcement 和 OpenFE runner 缺失时显式 skip 边界；L1-L4 真实 runner 仍需生产 command/env、full inference smoke 和集群发布验证。

H5 Oracle command wrapper 补充验证：`.env` 已投放 `DOCK_ORACLE_COMMAND`、`BOLTZ2_ORACLE_COMMAND`、`FEP_ORACLE_COMMAND`、`ADMET_ORACLE_COMMAND` 及 Boltz/OpenADMET/OpenFE registry 运行 key，未记录任何 secret/token/key 具体值；配置生效检查退出码 0，四个 command env 均为 set 且首个可执行均可解析，`OPENFE_RUNNER_PATH` 首个可执行可解析，`OPENFE_CLI_PATH`、`OPENFE_TRANSFORMATION_REGISTRY`、`OPENFE_RESULT_REGISTRY`、`OPENFE_WORK_DIR`、`FEP_JOB_DIR` 均存在。`uv run pytest tests/unit/test_h5_oracle_wrappers.py -q` 退出码 0，13 项通过；服务 command 合同回归退出码 0，4 项通过；FEP service focused 回归退出码 0，3 项通过；`PYTHONPYCACHEPREFIX=/tmp/mforge-pycache-h5 uv run python -m py_compile ...` 退出码 0；`git diff --check -- ...` 退出码 0。OpenADMET 主预测 smoke 退出码 0，JSON parse OK，1 条 clearance float prediction；Boltz GPU affinity smoke 使用 6OIM/CCO、GPU、结构采样 10、affinity 采样 10，退出码 0，stdout 278 bytes，stderr 0 bytes，JSON parse OK，affinity_count=1，并生成 CIF、confidence JSON 和 affinity JSON。FEP/OpenFE 补验：`openfe fetch rbfe-tutorial`、`openfe fetch rbfe-tutorial-results`、`openfe gather ... --report dg --tsv`、`openfe gather ... --report ddg --tsv` 均退出码 0；`openfe plan-rbfe-network ... --n-protocol-repeats 1 -s settings.yaml` 退出码 0，持续约 17 分 58 秒，stderr 摘要为 multiprocessing fork DeprecationWarning 与 element-change UserWarning，无 fatal error。已生成 `models/artifacts/openfe/tyk2/transformation_registry.json`（9 条完整 complex/solvent 边）和 `models/artifacts/openfe/tyk2/result_registry.json`（18 条正反向 ddG 边）；官方 TYK2 结果 `final_results_ddg.tsv` 为 9 rows，DDG 范围 -0.89 到 1.4 kcal/mol，range 2.29；`final_results_dg.tsv` 为 10 rows，DG(MLE) 范围 -1.25 到 2.0 kcal/mol，range 3.25。FEP wrapper registry smoke 退出码 0，stderr 空，返回 `ddg_kcal_mol=0.8`、`ddg_uncertainty=0.1`、`n_repeats=3`；FEP service background job smoke 退出码 0，最终 `state=completed`、`results=1`、`ddg_kcal_mol=0.8`。TYK2 教程输入和 SDF 未包含实验 Ki/IC50 标签，因此该 registry 证明真实 OpenFE 模拟结果具有区分分布，不登记实验相关性；本地仅发现 KRAS G12C 6OIM 共价复合物 PDB 与 Boltz template，未发现 KRAS OpenFE 配体系列、实验 ddG 或 covalent-FEP registry，KRAS full pilot 仍归 H11。H5 本地 command wrapper / ADMET / Boltz / FEP TYK2 registry gate 已完成；集群发布和 KRAS full pilot 仍归 H10/H11。

L3/L4 runtime 补充验证：`command -v openfe` 和 `command -v orca` 当前均无输出；`uv run python` 使用 `importlib.util.find_spec()` 探测得到 `openfe=False`、`gpu4pyscf=False`、`pyscf=False`。`uv run pytest tests/unit/test_validation_agent.py -k "quantum_command or l4_quantum" -q` 退出码 0，6 项通过，覆盖 L4 quantum cascade、`L4_QUANTUM_ORACLE_COMMAND`、`L4_GPU4PYSCF_COMMAND`、`L4_ORCA_COMMAND` env wiring、JSON command 解析和缺 executable preflight。该结果不代表真实 OpenFE/GPU4PySCF/ORCA runner command、Python package、artifact 或集群调用已验收。

ADMET runtime 补充验证：`services/admet-svc/src/admet_svc/main.py` 的 `_require_runtime()` 已在 `ADMET_ORACLE_COMMAND` 显式配置但不可用时 fail-fast，不再被可用的 `ADMET_MODEL_PATH` 掩盖；`uv run pytest tests/unit/test_service_artifact_status.py -k admet -q` 退出码 0，5 项通过；该结果验证 ADMET command runtime preflight、JSON runner 本地路径、HTTP runner 回退、gRPC 注册和部署 env wiring，不代表真实 ADMET-AI 服务或外部 runner 集群调用已验收。

Dock runtime 补充验证：`services/dock-svc/src/dock_svc/main.py` 的 `_require_runtime()` 已在 `DOCK_ORACLE_COMMAND` 显式配置但不可用时 fail-fast，不再被可用的 `GNINA_BINARY` 或 `DIFFDOCK_MODEL_PATH` 掩盖；`DockServicer.Dock()` 的 abort 路径会保留原始 command preflight 错误，不再降级为泛化 runner 未配置消息；`uv run pytest tests/unit/test_service_artifact_status.py -k dock -q` 退出码 0，5 项通过；该结果验证 Dock command runtime preflight、JSON runner 本地路径、service 错误传播、gRPC 注册和部署 env wiring，不代表真实 GNINA/DiffDock-L runner 集群调用已验收。

KD 补充验证：`uv run pytest tests/unit/test_cross_paradigm_kd.py -q` 退出码 0，16 项通过，覆盖 CrossParadigmKDLayer oracle feedback、Boltz2/HypSeek teacher distribution adapter、teacher embedding target distillation loss 和 generator quality ranking；生产级 teacher 服务和真实集群发布仍需外部环境验证。

W13 teacher embedding artifact gate 补充：新增 `libs/mf-core/src/mf_core/routing/kd_artifacts.py`，支持 `python -m mf_core.routing.kd_artifacts --input <teacher_records.jsonl> --output <teacher_embeddings.json> --expected-dim <dim> --min-embeddings <n> --strict`，从 JSON/JSONL teacher records 导出 canonical `cross_paradigm_teacher_embeddings.v1`，并报告 finite、consistent dimension、expected_dim 与 min_embeddings。新增 W13 focused pytest 2 项先 RED 后 GREEN；`uv run pytest tests/unit/test_cross_paradigm_kd.py -q` 当前退出码 0，18 项通过；CLI smoke 输出 `pass 2 2 cross_paradigm_teacher_embeddings.v1`；`python -m py_compile` 与 `git diff --check` 退出码 0。该 gate 只证明 teacher embedding artifact handoff 可验收；真实 production teacher source、真实蒸馏训练、benchmark 质量证据和集群发布验证仍未完成。

Generator 补充验证：`uv run pytest tests/unit/test_phase_b_generators.py -q` 退出码 0，7 项通过，覆盖 iCLM 缺 model/runner fail-fast、iCLM online learner EWC/PackNet/KD、UAS unfamiliarity filter、CReM fragment replacement 和 FragFM vocabulary + SA-aware rate matrix 本地生成路径；共享 HUMU 多范式生产闭环仍未完成。

W11 FragFM shared HUMU 质量门补充：`models/mf-generators/fragfm/train.py` 现在会校验并保留 valid 129 维 Lorentz full-coordinate `humu_embedding` 到 FragFM vocabulary artifact，并在 `training_manifest.json` 写入 `humu_embedding_count` / `humu_embedding_coverage`；新增 `models/mf-generators/fragfm/src/mf_generators/fragfm/quality.py`，可用 `python -m mf_generators.fragfm.quality --vocab <vocab> --checkpoint <pt> --rate-matrix <pt> --min-humu-coverage <x> --strict` 生成 `fragfm_quality_report.v1`。W11 focused pytest 4 项退出码 0；FragFM 子集 pytest 9 项退出码 0；`python -m py_compile` 和 `git diff --check` 退出码 0；当前本地 `checkpoints/fragfm` quality CLI smoke 在 `--min-humu-coverage 0.0` 下输出 `pass 50 0 0.0 True True`，说明 vocabulary 有 50 条规则、HUMU coverage=0、checkpoint/rate-matrix 可加载。该 smoke 仍不能作为生产 HUMU 条件质量证据；真实 HUMU-labeled FragFM 训练数据、production-quality artifact、正式 coverage/benchmark 阈值与集群发布验证仍未完成。

MMPT runtime 补充验证：`services/mmpt-generator-svc/src/mmpt_generator_svc/main.py` 的 runtime status 已与实际 `_index_path_from_uri()` 支持范围对齐，显式 `MMPT_INDEX_URI` 只接受本地 `file://` URI，非 `file://` URI 或非本地 file URI 会 fail-fast；`uv run pytest tests/unit/test_service_artifact_status.py -k mmpt -q` 退出码 0，7 项通过；该结果验证 MMPT index artifact、本地 `file://` URI、patent RAG / Seq2Seq command runtime status、部署 env wiring 和 service info，不代表真实 Seq2Seq Transformer、专利 RAG 服务或生产 command 值已投放。

CIC/HCIV 补充验证：`uv run pytest tests/unit/test_cic_compiler.py -q` 退出码 0，31 项通过，覆盖 CIG JSON-LD context、objective edges/hyperedges、schema、grounding、refinement command、production HCIV checkpoint fail-fast、HCIV feature generation、directed topology sensitivity，以及 W10 supervised HCIV train/export smoke；生产级 LLM/SRM parser、refinement runner command/env、真实 supervised CIG/HCIV 数据和训练好的 `Enc_intent` checkpoint 仍未投放。

CIG service 边界补充验证：`tests/unit/test_cig_service.py::test_compile_service_uses_injected_compiler` 已在注入式 local demo compiler 中显式关闭 grounding，并 monkeypatch UniProt 查询为失败函数，确保该 unit test 不访问外网；`uv run pytest tests/unit/test_cig_service.py -q` 退出码 0，2 项通过。该修复只证明 service 注入边界离线可测，不代表 production semantic parser 或 grounding 外部源已投放。

Agent/SRB/Validation 补充验证：`uv run pytest tests/unit/test_srb_agent.py tests/unit/test_generator_coord_agent.py tests/unit/test_validation_agent.py -q` 退出码 0，50 项通过，覆盖 SRB SSP/XDL/SiLA2 adapter、GeneratorCoord target/discovery/health/CRG routing 和 ValidationAgent L0-L4 cascade/wiring/CRG skip 边界；真实外部 generator registry、SiLA2 硬件和 L1-L4 runner 仍需投放后验证。

Supply Oracle 补充验证：`services/supply-oracle-svc/src/supply_oracle_svc/main.py` 的 runtime status、`_require_runtime()` 和服务入口 env client 构建前 preflight 只接受当前实际支持的 `file://` catalog URI，非 `file://` URI 会 fail-fast；`uv run pytest tests/unit/test_service_artifact_status.py -k supply -q` 退出码 0，覆盖本地 catalog、AiZynth HDF5 stock、非 `file://` catalog URI 拒绝和 orchestrator supply hook。

RetroSyn 补充验证：`services/retrosyn-svc/src/retrosyn_svc/main.py` 的 runtime status 和 `_require_planner_runtime()` 已把 `RETROSYN_PLANNER_COMMANDS_JSON` 内的 planner command 纳入 executable preflight；只配置 JSON planner ensemble、没有 `AIZYNTH_CONFIG_PATH` 时可通过启动 preflight，缺失、非法或部分不可用的 JSON planner command 会以 planner command 状态失败；`uv run pytest tests/unit/test_service_artifact_status.py -k retrosyn -q` 退出码 0，30 项通过；该结果验证注入式 planner、单 command、JSON ensemble、命名 planner env、runtime/startup preflight、部署 env wiring 和 RetroSyn 相关 orchestrator hook；RAscore/RSGPT/UAlign/AiZynth 本地真实 command path 已另行验收，集群调用仍未验收。

Provenance/Sigstore 补充验证：`uv run pytest tests/unit/test_provenance.py -q` 退出码 0，19 项通过，覆盖 provenance model/schema、本地 dev signature、Sigstore sign/verify command preflight、identity token、expected identity、Rekor URL 传递、Rekor bundle cache 和 provenance-svc 部署 env wiring；`uv run pytest tests/test_mvp_pipeline.py::TestMVPPipeline::test_sigstore_signer_sign_command_preflight_rejects_missing_executable tests/test_mvp_pipeline.py::TestMVPPipeline::test_sigstore_signer_verify_command_preflight_rejects_missing_executable tests/test_mvp_pipeline.py::TestMVPPipeline::test_sigstore_signer_uses_configured_commands -q` 退出码 0，3 项通过，覆盖 lineage `SigstoreSigner` sign/verify command executable preflight 与已配置 command 的正常路径。

PCBO/Eval 补充验证：`uv run pytest tests/unit/test_mf_eval.py -q` 退出码 0，20 项通过，覆盖 distortion/activity-cliff 指标、HV/HVI/PoF/constrained HVI/EHVI、HUMU log-map tangent GP acquisition、异步 PCBO oracle loop、scheduler、`ParetoBOService` JSON command runner、FastAPI endpoint 和部署 wiring；真实 candidate provider/oracle evaluator command/env 与生产验收仍未投放。

KRAS engineering scope 补充验证：`tests/e2e/test_kras_g12c_pilot.py` 的 preflight 已区分 `full` 与 `engineering` scope，`engineering` scope 不再要求 Neo4j、MinIO、Redis、Provenance production DB 和 Sigstore 生产环境变量；`services/orchestrator-svc/src/orchestrator_svc/main.py` 的 EngineeringWorkflowClients 已用本地 `MolPredictEngine` 为 validation rows 补 RDKit 描述符、ADMET 派生值和 critic 规则别名，并修正 PAINS/hERG 类型桥接；`env RUN_KRAS_G12C_E2E=1 KRAS_E2E_SCOPE=engineering ORCHESTRATOR_E2E_READY=1 uv run pytest tests/e2e/test_kras_g12c_pilot.py -q` 退出码 0，6 项中 3 项通过、3 项因 HFM/Boltz/AiZynth 属于 full-only 范围而 skip。该结果只证明 KRAS 本地 engineering workflow 可执行到 validation/critic/escalation，不等于 KRAS G12C 真实 full pilot 跑通。

KRAS full scope 补充验证：`env RUN_KRAS_G12C_E2E=1 KRAS_E2E_SCOPE=full uv run pytest tests/e2e/test_kras_g12c_pilot.py -q` 在默认 env 下退出码 1，6 项均在 fixture preflight 阶段 error；默认 env 原始缺口为 `HFM_CHECKPOINT_PATH`、`HFM_DECODER_PATH`、`BOLTZ_MODEL_PATH`、`BOLTZ_INPUT_TEMPLATE_DIR`、`AIZYNTH_CONFIG_PATH`、`CRITIC_AGENT_READY`、`ORCHESTRATOR_E2E_READY`、`NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`、`MINIO_ENDPOINT_URL`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY`、`MINIO_BUCKET`、`REDIS_HOST`、`REDIS_PORT`、`PROVENANCE_DATABASE_URL or TEST_DATABASE_URL`、`PROVENANCE_STORE_MODE=production_real`、`SIGSTORE_IDENTITY_TOKEN`、`SIGSTORE_EXPECTED_IDENTITY`、`SIGSTORE_SIGN_COMMAND`、`SIGSTORE_VERIFY_COMMAND` 和 `SIGSTORE_REKOR_URL`。补充设置 `HFM_CHECKPOINT_PATH=checkpoints/hfm3d_4h200/best_model.pt`、`HFM_DECODER_PATH=checkpoints/hfm3d_4h200/decoder.json`、`BOLTZ_MODEL_PATH=models/artifacts/boltz-2`、`BOLTZ_INPUT_TEMPLATE_DIR=models/artifacts/boltz-input-templates`、`AIZYNTH_CONFIG_PATH=models/artifacts/aizynthfinder/config.yml` 后，重新运行 `uv run pytest tests/e2e/test_kras_g12c_pilot.py -q` 仍退出码 1，6 项 setup error，剩余缺口为 `CRITIC_AGENT_READY`、`ORCHESTRATOR_E2E_READY`、`NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`、`MINIO_ENDPOINT_URL`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY`、`MINIO_BUCKET`、`REDIS_HOST`、`REDIS_PORT`、`PROVENANCE_DATABASE_URL or TEST_DATABASE_URL`、`PROVENANCE_STORE_MODE=production_real`、`SIGSTORE_IDENTITY_TOKEN`、`SIGSTORE_EXPECTED_IDENTITY`、`SIGSTORE_SIGN_COMMAND`、`SIGSTORE_VERIFY_COMMAND` 和 `SIGSTORE_REKOR_URL`；补充调用 `check_tool(ToolRequirement('boltz', executable='boltz', env_var='BOLTZ_BINARY'))` 显示 `boltz` executable 可解析到 `.venv/bin/boltz`。该结果证明当前本地 artifact 能覆盖部分 KRAS full preflight，但真实 full pilot 仍缺 service ready、外部 DKI、provenance 和 Sigstore 生产依赖。

基于以上审计，当前目标不能标记为完成。后续继续推进时，应优先处理两类事项：

1. 在已确认“默认 HFM 不直接读取 shared CRG”的边界下，继续把当前兼容旧字段的 `jmcg_feedback` 从 context records 推进到 evidence-backed HUMU embedding producers 和 W8-E engineering skeleton；property feedback 仍保持 non-steering context，intent feedback 仅在已有 finite 且满足 Lorentz hyperboloid 方程的 129 维 full-coordinate axis 时 steering，pocket feedback 仅在结构化 pocket geometry 可由 `HUMU_ENCODER_TARGET` 编码为 finite 且满足 Lorentz hyperboloid 方程的 129 维 HUMU embedding 时 steering。W8-E 已有本地 joint sample skeleton，但在联合训练/采样质量证据出现前，不能把该路径描述为完成 JMCG。
2. 投放真实 artifact、runner command、凭据和集群环境后，逐项把第 16 节的生产缺口转为可验证实现。

## 18. 结束判断

当前项目已经具备 CoreArchitecture v2 的工程骨架和若干真实子系统，但仍没有达到文档描述的核心架构闭环。若按文档逐条验收，应判定为：

```text
架构方向      :  与 v2 基本一致
工程骨架      :  较完整
当前冻结项    :  HUMU 预训练部分保持现状，不作为本阶段修改项
核心算法      :  除 HUMU 预训练冻结范围外，其余简化或未接入部分需要补齐
端到端闭环    :  未完成
评估验证      :  未完成
技术选型一致性:  DKI 已确认采用 Qdrant/Redis，不回迁 Milvus/NATS
```
