# CoreArchitecture v2 补齐 —— 两人对接验收规范（方案一）

## 0. 本文定位

- 上位文档：`docs/architecture/corearchitecture-v2-completion-tasksplit.md`（工作量划分，已选定**候选方案一**）。
- 真值基线：`docs/architecture/current-implementation-vs-corearchitecture-v2.md`。
- 治理规则：`docs/todo/2026-06-03-corearchitecture-v2-continuation-governance-plan.md`。
- 本文只解决一件事：**甲（生成上游）与乙（验证-供应-存证下游）之间如何交接、如何验收、如何防止互相破坏**。不重复划分工作量，不改业务代码，不跑测试。
- 文中所有契约字段均已核对真实源码（2026-06-03）。字段如与代码不符，以代码为准并回本文登记修订。

---

## 1. 角色与边界

主链路 `CIG→HCIV→generate→validate→retrosyn→supply→srb→critic→provenance`，以 generation 为界：

| | 甲（生成上游） | 乙（验证-供应-存证下游） |
|---|---|---|
| 主目录 | `models/mf-generators/*`、`agents/generator_coord/`、`agents/nl2obj/`、`services/cig-compiler-svc/`、`services/*generator*svc/`、`libs/mf-core/src/mf_core/routing/` | `agents/validation_agent/`、`agents/retrosyn_agent/`、`agents/supply_agent/`、`agents/srb_agent/`、`agents/critic_agent/`、`services/*oracle*`、`services/provenance-svc/`、`libs/mf-core/src/mf_core/db/`、`libs/mf-eval/` |
| work item | W2,W6,W8,W9,W10,W11,W13 | W1,W3,W5,W12 |
| 公共测试 | 提供改动清单给乙 | W4 统一执行（授权后） |

**共享但需协调的文件**（双方都可能改，改前必须登记，见 §4）：
- `services/orchestrator-svc/src/orchestrator_svc/main.py` —— 甲改 generation 派生、乙改 validate/retrosyn/supply/srb/provenance hook。
- `agents/generator_coord/src/generator_coord/agent.py` —— 甲主，乙只读。
- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py` —— 甲主，乙只读。

---

## 2. 三个契约面（精确 schema + 验收）

### 契约 C1：`generator_params` 反馈透传

**唯一 envelope**：`moleculeforge.jmcg.feedback.v1`。三个 generator_params key 共存：`jmcg_feedback`（新）、`route_humu_feedback`（legacy）、`generation_feedback`（legacy）。

**envelope 结构**（`generator_coord._jmcg_feedback_envelope` / `orchestrator._property_jmcg_feedback_from_generation_feedback`）：
```json
{
  "schema": "moleculeforge.jmcg.feedback.v1",
  "run_id": "run-...",
  "project_id": "project-...",
  "records": []
}
```

**record 完整字段**（steering-capable，以 `generator_coord._jmcg_route_feedback_record` 为准）：
```json
{
  "kind": "route",
  "source": "generator_coord",
  "run_id": "run-...",
  "subject": {"type": "route", "id": "route-..."},
  "humu_embedding": [0.0],
  "curvature": 1.0,
  "weight": 1.0,
  "polarity": "attract",
  "confidence": 1.0,
  "evidence_ids": ["belief-..."],
  "metadata": {}
}
```

**消费侧规则**（`hfm_3d/generator.py`）：
- HFM 扫描顺序：`("jmcg_feedback", "route_humu_feedback", "generation_feedback")`。
- embedding 取自 `humu_embedding` 或 `route_humu_embedding`；**两者皆无 = non-steering，被忽略但保留作 provenance**。
- 当前 `kind="property"` 仍省略 `humu_embedding` 和 `curvature` → non-steering；`kind="intent"` 仅在已有 finite 且满足 Lorentz hyperboloid 方程的 129 维 Lorentz full-coordinate axis 时可 steering；`kind="pocket"` 仅在结构化 pocket geometry 可由 HUMU encoder 编码为 finite 且满足 Lorentz hyperboloid 方程的 129 维 embedding 时可 steering。

**对接验收（C1）**：

| 提供方 | 消费方 | 验收点 | 期望证据 |
|---|---|---|---|
| 乙：orchestrator 派生 property/intent/pocket records | 甲：HFM | non-steering record 不改变 latent | `feedback_steering_count` 不计入这些 record；`feedback_steering_dropped_count` 记录 |
| 甲：generator_coord 产 route records | 甲：HFM | route record 触发有界 steering | `feedback_steering_kinds` 含 `route`；metadata 含 accepted/dropped 计数 |
| 甲：W2 让 pocket/intent 带 embedding | 甲：HFM | 升级为 steering-capable | `feedback_steering_kinds` 出现 `pocket`/`intent`；不破坏 route 既有行为 |

> **W2 红线**：甲给 pocket/intent 加 `humu_embedding` 时，embedding 必须是 finite、满足 Lorentz hyperboloid 方程、且维度等于 HFM 活跃 latent 维度；否则被 producer 或 HFM `_valid_feedback_records` 丢弃。维度来源见契约 C3。

> **W8-E 工程验收挂钩 C1**：上表三行全绿（五类 record 全部按契约 C1 被 HFM 正确消费/忽略）+ `JMCGEngineeringSampler` 能本地构造 `moleculeforge.jmcg.joint_sample.v1` engineering skeleton output，即满足 tasksplit §2.2 的 **W8-E 工程验收**。W8-R 研究验收（联合采样质量）不在本契约范围，单列里程碑。

### 契约 C2：CRG belief 谓词表

belief 经 `GraphRepository.write_workflow_belief(run_id, belief_id, subject, predicate, object_value, confidence, source_agent, ...)` 写入；读回 `get_run_crg(run_id) -> dict`；edge 用 `write_crg_edge(source_belief_id, target_belief_id, relation, ...)`。

**现有谓词（predicate）登记表**：

| predicate | 写入方 | 读取方 | 归属 |
|---|---|---|---|
| `workflow_status` | OrchestratorAgent | OrchestratorAgent | 公共 |
| `parsed_intent` / `compiled_cig` | NL2ObjAgent | NL2ObjAgent | 甲 |
| `selected_generators` | GeneratorCoordAgent | GeneratorCoordAgent | 甲 |
| `route_humu_embedding` | RetroSynAgent | GeneratorCoordAgent | **跨界**（乙写甲读）|
| `validation_status` | ValidationAgent | RetroSyn/Critic | 乙 |
| `retrosyn_routes` | RetroSynAgent | Supply/SRB/Critic | 乙 |
| `supply_feasibility` | SupplyAgent | SRB/Critic | 乙 |
| `ssp_compiled` | SRBAgent | — | 乙 |
| `critic_verdict` | CriticAgent | CriticAgent | 乙 |

**对接验收（C2）**：

| 场景 | 验收点 | 期望证据 |
|---|---|---|
| `route_humu_embedding` 跨界（乙→甲） | 乙写的 payload 字段能被甲 `_route_humu_feedback_from_crg` 解析 | payload 含 `humu_embedding`,`route_id`，可选 `source/weight/polarity/confidence/evidence_ids/metadata` |
| W1：乙合并 agent belief 进 final_state CRG | provenance metadata CRG 包含 agent belief 而非仅 stage belief | `crg_belief_count` 增大；合并发生在 `_record_workflow_provenance()` 前 |
| 新增 predicate | 必须先在本表登记 | 本文 §4 变更记录有对应行 |

> **W1 边界**（CRG brief §8.2 gap#2）：乙在写 provenance 前调 `get_run_crg(run_id)` 合并，但**不得改变 `add_edge` 版本语义**（已于 6.3 修正为递增 `CRG.version`），也不得让 HFM 直接读 CRG（默认 HFM 不读 shared CRG 的决策已确认，见 jmcg-feedback-contract-brief §11）。

### 契约 C3：HUMU encoder 接口

`humu-encoder-svc` 的 `Encode(request)`，`input_type ∈ {molecule, pocket, route}`（也接受 `entity_type`）。

| input_type | 输入来源（优先级） | payload schema |
|---|---|---|
| molecule | `request` smiles + `input_data.coords/coordinates` | `{"smiles": str, "coords": [[x,y,z],...]}` |
| pocket | `request.pocket_data` → `input_data` | pocket 点云/残基结构（dict） |
| route | `request.route_data` / `request.payload` → `input_data` | `{"steps": [{parent/children...}]}` |

输出：`humu_embedding`（当前为 129 维 Lorentz 全坐标；`dim=128` 是空间维度参数，实际向量长度为 `dim + 1`）+ `humu_curvature`。消费方必须用 finite + Lorentz hyperboloid 合法性校验，而不只检查长度。

**对接验收（C3）**：

| 调用方 | 验收点 | 期望证据 |
|---|---|---|
| 甲 W2（pocket/intent embedding） | pocket 编码返回 finite 且满足 Lorentz hyperboloid 方程的 129 维 Lorentz 全坐标 | embedding 合法且长度 == HFM latent dim；否则 C1 被丢弃 |
| 甲 W8（JMCG 联合采样） | mol/route/pocket 三类 embedding 同空间且均通过 Lorentz 合法性校验 | 同 checkpoint、同 curvature 来源 |
| 乙 RetroSynAgent（已存在） | route 编码写回 `route_humu_embedding` belief | 既有行为不被甲改动破坏 |

> dim 来源：encoder 构造 `dim=128`（见 `humu_encoder_svc/main.py`），但 mol/pocket/route encoder 均投影到 Lorentz `dim + 1` 全坐标；`humu/encoder.proto` 与 `retrosyn/route.proto` 也把 embedding 注释为 129 维。甲 W2/W8 与乙 RetroSyn 必须共用同一 `HUMU_CHECKPOINT_PATH`（本地 `checkpoints/humu/best_model.pt`），否则嵌入不可比。

---

## 3. work item 之间的交接点

| 交接 | 上游交付 | 下游接收 | 交接物 |
|---|---|---|---|
| W2 → W8 | 甲：pocket/intent steering-capable record | 甲：JMCG 联合采样 | 三类 embedding 可作联合 target |
| W1 ← C2 | 乙：合并 agent belief | provenance | 完整 run CRG |
| W1 ← H1 | 乙：DKI Neo4j 就位 | W1 真实验收 | `NEO4J_*` env |
| W3 ← H5 | 乙：本地 oracle 可达 | PCBO 参考 provider | oracle service target |
| W5 ← H8 | 人工：官方 benchmark 数据 | 乙：benchmark harness | `*_PATH` env |
| W6/W9/W10 ← 人工 | 训练数据/算力 | 甲：训练脚本 | 数据集 + 产物 artifact |
| W11 ← 人工 | HUMU-labeled FragFM 训练数据 + production artifact | 甲：质量门 | `fragfm.quality` JSON report + benchmark/cluster evidence |
| W13 ← 人工 | production teacher records / embeddings | 甲：KD artifact gate | `kd_artifacts` JSON report + distillation run evidence |

**交接协议**：上游完成后在 §5 执行日志登记「交付物 + 验收命令 + 当前剩余 gate」，下游据此接手，不口头交接。

### 3.1 W6 TAR runner command contract

Owner A 已提供本地 command target：

```bash
python -m generator_router_svc.tar_proxyless_runner
```

可配置为：

```bash
TAR_PROXYLESS_SEARCH_COMMAND="python -m generator_router_svc.tar_proxyless_runner"
```

stdin payload 与 `GeneratorRouterServicer.RunProxylessSearch()` 外部 command contract 一致：

```json
{
  "reward_batches_by_dataset": {
    "kras": [
      {"hfm_3d": 0.2, "fragfm": 0.8}
    ]
  },
  "generator_costs": {"hfm_3d": 5.0, "fragfm": 1.0},
  "cost_weight": 0.1,
  "learning_rate": 1.0,
  "temperature": 1.0
}
```

stdout result 至少包含：

```json
{
  "rounds": [],
  "architecture_probabilities": {},
  "architecture_logits": {},
  "generator_names": []
}
```

验收边界：

- 本地 AI 代码侧：runner command target exists，且复用 `ProxylessSearchScheduler`。
- 人工/生产侧：真实 reward 数据集、生产 `TAR_PROXYLESS_SEARCH_COMMAND` 值投放、集群发布验证仍未完成。

### 3.2 W11 FragFM shared HUMU quality contract

Owner A 已提供本地 quality gate：

```bash
python -m mf_generators.fragfm.quality \
  --vocab <fragfm_vocab.json> \
  --checkpoint <best_model.pt> \
  --rate-matrix <rate_matrix.pt> \
  --min-humu-coverage <0.0-1.0> \
  --strict
```

report 至少包含：

```json
{
  "schema_version": "fragfm_quality_report.v1",
  "status": "pass",
  "rules": 0,
  "fragments": 0,
  "humu_embedding_count": 0,
  "humu_embedding_coverage": 0.0,
  "invalid_humu_embeddings": 0,
  "checkpoint_loadable": true,
  "rate_matrix_loadable": true,
  "messages": []
}
```

验收边界：

- 本地 AI 代码侧：training CLI 保留 valid 129 维 Lorentz `humu_embedding` 到 vocabulary artifact，quality CLI 能报告 coverage、invalid embedding、checkpoint/rate-matrix loadability。
- 人工/生产侧：投放真实 HUMU-labeled FragFM 训练数据，训练 production-quality artifact，设定正式 `--min-humu-coverage` 和 benchmark 阈值，并完成集群发布验证。
- 当前本地 `checkpoints/fragfm` artifact 的 quality CLI smoke 在 `--min-humu-coverage 0.0` 下可 pass，但 `humu_embedding_coverage=0.0`，不得作为生产 HUMU 条件质量证据。
- 当前本地 `checkpoints/fragfm_humu_5k` artifact strict quality coverage 1.0，且 Docker Compose/Kubernetes/Helm 默认值已指向它；这只算本地 deployment-default hardening，不算 production artifact promotion 或 cluster acceptance。

### 3.3 W13 KD teacher embedding artifact contract

Owner A 已提供本地 teacher embedding artifact utility：

```bash
python -m mf_core.routing.kd_artifacts \
  --input <teacher_records.jsonl> \
  --output <teacher_embeddings.json> \
  --expected-dim <dim> \
  --min-embeddings <n> \
  --strict
```

canonical artifact:

```json
{
  "schema_version": "cross_paradigm_teacher_embeddings.v1",
  "embedding_count": 2,
  "embedding_dim": 2,
  "teacher_embeddings": [[0.1, 0.2], [0.3, 0.4]]
}
```

report 字段包括 `status`、`embedding_count`、`embedding_dim`、`expected_dim`、`min_embeddings`、`messages`。

验收边界：

- 本地 AI 代码侧：JSON/JSONL teacher records 可导出 canonical artifact；artifact preflight 会检查 finite、consistent dimension、expected dimension 和 minimum count。
- 人工/生产侧：提供真实 teacher record / embedding 来源，运行真实蒸馏训练，并提交 benchmark 质量证据和集群发布验证。

---

## 4. 防冲突与变更登记

### 4.1 共享文件双写协调

改 §1 列出的 3 个共享文件前，在下表登记意图（占用期内对方不并行改同一函数）：

| 文件 | 函数/区段 | 占用方 | 意图 | 状态 |
|---|---|---|---|---|
| orchestrator-svc/main.py | `_record_workflow_provenance` 前合并 | 乙(W1) | CRG 合并 | **已完成** |
| orchestrator-svc/main.py | `_jmcg_context_feedback_from_state` | 甲(W2) | pocket/intent 加 embedding | **已完成** |
| generator_coord/agent.py | `_route_humu_feedback_from_crg` | 甲 | — | 空闲 |
| hfm_3d/generator.py | `_feedback_embedding_records` | 甲(W2/W8) | 联合 target | **已完成** |

> W1（乙）与 W2（甲）都碰 `orchestrator-svc/main.py` 但函数不同——可并行，仅需各自登记区段，提交时分别 commit、避免同一 hunk。

### 4.2 契约变更登记（C1/C2/C3 任一字段/谓词变更）

| 日期 | 契约 | 变更 | 发起方 | 对方确认 |
|---|---|---|---|---|
| 2026-06-03 | — | 初始登记，无变更 | — | — |

变更规则：契约字段/谓词的增删改需双方在此表签字（发起方填写、对方确认）后方可落地；单方不得静默改 schema。

---

## 5. 验收命令清单（授权后执行，归 W4 统一跑）

| work item | 验收命令 | 期望证据 |
|---|---|---|
| W1 CRG 合并 | `uv run pytest tests/unit/test_graph_repo.py -q` + 新增合并用例 | 退出码 0；provenance CRG 含 agent belief |
| W2 embedding producer | `uv run pytest tests/unit/test_generators.py tests/unit/test_service_artifact_status.py -q` | pocket/intent steering 生效，route 不回归 |
| W3 PCBO provider | `uv run pytest tests/unit/test_mf_eval.py -q` | 本地 provider/oracle 闭环可跑 |
| W5 benchmark | `uv run pytest tests/benchmark -q`（默认全 skip）；临时资源 smoke 非 skip | 资源就位零代码改动可跑 |
| W11 FragFM quality | `uv run pytest tests/unit/test_generators.py -q -k 'FragFMGenerator and not training_cli_writes_checkpoint_and_vocab_artifacts and not training_cli_writes_kd_embedding_loss_metadata'` + `python -m mf_generators.fragfm.quality ... --strict` + focused deployment default regression | training artifact 保留 valid HUMU embedding；quality report 输出 coverage/loadability，并严格要求 checkpoint/rate-matrix 关键 schema；当前 `fragfm_humu_5k` local candidate coverage=1.0 且 deployment defaults 指向它，但仍非 production/cluster acceptance |
| W13 KD artifact | `uv run pytest tests/unit/test_cross_paradigm_kd.py -q` + `python -m mf_core.routing.kd_artifacts ... --strict` | teacher embedding artifact 可导出/报告；维度与 finite 校验生效 |
| C1 透传回归 | `uv run pytest tests/unit/test_generator_coord_agent.py tests/unit/test_generators.py -q` | envelope 合并/透传不破坏 legacy |
| C2 谓词回归 | `uv run pytest tests/unit/test_validation_agent.py tests/unit/test_srb_agent.py -q` | belief 读写跳过逻辑不变 |
| 全量 | `uv run pytest tests/unit -q`（基线 598 项）；`uv run pytest -q`（基线 694 项，38 skip） | 退出码 0，新增项不引入 failure |

> 跑测试前须经用户显式授权（项目 no-test 规则）。基线项数引自对照文档 §17。

---

## 6. 双周对接检查清单

每个对接周期末，两人共同核对：

- [ ] 三契约面（C1/C2/C3）字段无单方静默变更，§4.2 登记完整。
- [ ] 共享文件无未登记双写冲突。
- [ ] 已完成 work item 已在对照文档对应层同步更新（governance §1.3）。
- [ ] 交接点（§3）上游交付物有验收命令与剩余 gate 记录。
- [ ] C 类资源投放卡状态已更新（见 tasksplit §5.4）。
- [ ] 未验收的生产依赖仍明确标注未完成，未用本地 smoke 冒充。

---

## 7. 执行日志

- 2026-06-03：创建本对接验收文档。核对 C1/C2/C3 契约真实字段（graph_repo 三方法、jmcg.feedback.v1 record、humu-encoder Encode input_type）。未改业务代码，未跑测试。
- 2026-06-03：按用户决定移除专利（MMPT 专利 RAG/Seq2Seq，原 W7）与湿实验（SiLA2 硬件，原 H7），同步 work item 行与交接表。
- 2026-06-03（乙）：完成 W1/W3/W5/W12 全部乙方 AI 编码任务。W1 经静态核查为已完整实现（`_merge_agent_beliefs_into_crg` + 3 条单元测试）；W3 新建 `pareto_bo/providers.py` + 修改 `service.py` from_env 默认回退 + 4 条测试；W5 补 `tests/benchmark/__init__.py` gzip 支持；W12 补 3 条 CReM scorer 端到端测试。§4.1 共享文件占用表 W1 状态更新为"已完成"。`python -m py_compile` 通过，`git diff --check` 通过，未跑 pytest（遵守 no-test 规则）。C3 契约无变更，C2 谓词表无新增谓词。
- 2026-06-03（甲）：完成 W6 本地 TAR runner command target。新增 `generator_router_svc.tar_proxyless_runner`，可用 `python -m generator_router_svc.tar_proxyless_runner` 作为 `TAR_PROXYLESS_SEARCH_COMMAND`，stdin/stdout schema 见 §3.1；新增 3 条 focused 规格到 `tests/unit/test_task_router.py`。静态验证和命令级 smoke 通过；2026-06-04 已补 W6 focused pytest 和 `tests/unit/test_task_router.py` 文件级 pytest（30 项）通过。剩余 gate：真实 reward 数据集、生产 env 值投放和集群发布验证。
- 2026-06-04（甲）：完成 W8-E 本地 JMCG engineering skeleton。新增 `mf_generators.hfm_3d.inference.jmcg_sampler`，输出 `moleculeforge.jmcg.joint_sample.v1` JSON-serializable records，校验 HUMU embedding 并保留 128 维/缺 embedding 为 non-steering context；新增 3 条 focused 规格到 `tests/unit/test_generators.py`。静态验证和命令级 smoke 通过，后续 hardening gate 已把校验升级为 finite + Lorentz hyperboloid 合法性检查并补 pytest 回归。剩余 gate：W8-R 真实联合采样训练质量、训练 artifact 和端到端生产验证。
- 2026-06-04（甲）：完成 embedding validation hardening。新增 `mf_core.geometry.lorentz.normalize_lorentz_embedding()`，W2/HFM/W8-E 共用 finite + Lorentz hyperboloid 合法性校验；W8-E 支持 packed float32 `Molecule.humu_embedding` bytes。新增 4 条 focused 规格先 RED 后 GREEN；`uv run pytest tests/unit/test_generators.py tests/unit/test_service_artifact_status.py -q` 通过 273 项，剩余 gate：W8-R 真实联合采样训练质量、W6 生产 reward 数据/部署、W9 生产 artifact/部署。
- 2026-06-04（甲）：完成 W9 本地 HFM neural geometry decoder 工程路径。新增 `mf_generators.hfm_3d.decoder.neural_geometry_decoder`、`decoder/__init__.py` 和 `models/mf-generators/hfm_3d/train_geometry_decoder.py`，可从 SDF-backed HFM decoder artifact 训练 torch geometry decoder artifact，并可用 `python -m mf_generators.hfm_3d.decoder.neural_geometry_decoder --artifact <artifact.pt>` 作为 `HFM_MOLECULAR_DECODER_COMMAND` 兼容 command target。`HFM3DGenerator` 保留 decoder payload 自带 `metadata.decoder_mode`；旧 payload 未声明时仍默认 `molecular_decoder`。5 条新 W9 规格先 RED 后 GREEN；focused W9 + legacy decoder gate 6 项通过；`uv run pytest tests/unit/test_generators.py -q` 通过 65 项；`python -m py_compile` 与 `git diff --check` 通过。剩余 gate：真实 production-quality decoder artifact、生产 env/command 投放、集群发布和几何质量 benchmark。
- 2026-06-04（甲）：完成 W10 本地 Enc_intent checkpoint 训练/export 工程路径。新增 `cig_compiler_svc.domain.hciv_training` 和 `services/cig-compiler-svc/train_hciv_encoder.py`，可从 supervised `cig + target_hciv` JSON/JSONL 数据训练现有 `HCIVEncoder` 并导出兼容 `HCIV_CHECKPOINT_PATH` 的 torch checkpoint；`HCIVEncoder` 新增可微 `forward_coordinates(cig)`，`encode()` 输出契约保持不变。3 条新 W10 规格先 RED 后 GREEN；focused W10 gate 4 项通过；`uv run pytest tests/unit/test_cic_compiler.py -q` 通过 31 项；`python -m py_compile` 与 `git diff --check` 通过。剩余 gate：真实 supervised CIG/HCIV 数据、production-quality checkpoint、`HCIV_CHECKPOINT_PATH` 投放、集群验收和下游质量验证。
- 2026-06-04（甲）：完成 W11 FragFM shared HUMU 本地质量门。训练 CLI 现在会校验并保留 valid 129 维 Lorentz `humu_embedding`，并写入 manifest coverage；新增 `mf_generators.fragfm.quality` JSON report/CLI，覆盖 vocabulary HUMU coverage、invalid embedding、checkpoint/rate-matrix loadability。W11 focused 4 项通过，FragFM 子集 9 项通过，quality CLI smoke 对当前本地 artifact 输出 `pass 50 0 0.0 True True`（阈值 0.0），证明 artifact 可加载但 HUMU coverage=0。剩余 gate：真实 HUMU-labeled 数据、production artifact、正式 coverage/benchmark 阈值、集群验证。C1/C2/C3 无变更。
- 2026-06-04（甲）：完成 W13 KD teacher embedding artifact 本地质量门。新增 `mf_core.routing.kd_artifacts`，可从 JSON/JSONL teacher records 导出 canonical `cross_paradigm_teacher_embeddings.v1`，并报告 finite、维度、expected_dim、min_embeddings。2 条新 W13 规格先 RED 后 GREEN；`uv run pytest tests/unit/test_cross_paradigm_kd.py -q` 通过 18 项；CLI smoke 输出 `pass 2 2 cross_paradigm_teacher_embeddings.v1`；`python -m py_compile` 与 `git diff --check` 通过。剩余 gate：真实 production teacher source / teacher embeddings、真实蒸馏训练、benchmark 质量证据和集群发布验证。C1/C2/C3 无变更。
- 2026-06-04（甲）：完成 W9/W10/W11 阶段复验 hardening。W9 decoder source artifact `latent`、W10 supervised `target_hciv` 现在都复用 shared Lorentz validator；W11 FragFM quality gate 现在严格要求 checkpoint `fragment_encoder.weight` 和 rate-matrix `base_rate` schema。4 条新 hardening 规格先 RED 后 GREEN；相邻 focused pytest 13 项通过；`python -m py_compile` 与 `git diff --check` 通过；W11 strict CLI smoke 对当前本地 artifact 仍输出 `pass 50 0 0.0 True True`（coverage=0，仅 runtime smoke）。§4.1 已同步 W2/HFM 共享文件占用状态为已完成。剩余 gate 不变：真实 production artifacts/data、benchmark 阈值和集群验证。C1/C2/C3 无变更。
- 2026-06-04（乙）：完成 W1 真实 DKI 验收。未改业务代码；`.env` 中 DKI 必需 env 均已投放，`FEAST_REPO_PATH` 存在，Postgres/Neo4j/Qdrant/MinIO/Redis 端口可达。验证命令：`bash -lc 'set -a; source .env; set +a; unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"; uv run pytest tests/integration/test_dki_*.py -q'`。实际结果：exit code 0，10 passed，0 skipped；有 1 条 Qdrant client 1.18.0 与 server 1.12.4 minor version compatibility warning。W1 真实 Neo4j/DKI gate 已验收；C1/C2/C3 无变更。
- 2026-06-05（H2）：完成 Sigstore/Rekor 生产审计链验收。`.env` 已投放 `SIGSTORE_SIGN_COMMAND`、`SIGSTORE_VERIFY_COMMAND`、`SIGSTORE_REKOR_URL`、`PROVENANCE_SVC_URL`；GitHub Actions self-hosted runner 运行时投放 `SIGSTORE_IDENTITY_TOKEN`、`SIGSTORE_EXPECTED_IDENTITY`、`SIGSTORE_E2E_READY=1`，未写入或泄露 token。验证链路：`H2 Audit Sigstore E2E` workflow run `27016836066`（commit `d54f536`）在 self-hosted runner 上启动 `production_real` provenance service，真实 `cosign sign-blob` 写入 Rekor bundle，`cosign verify-blob` 按 GitHub Actions expected identity 验证通过，日志输出 `sigstore_rekor_smoke=pass`。验收命令：`RUN_AUDIT_E2E=1 PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/e2e/test_audit_completeness.py -q`；实际结果：GitHub Actions job success，4 items `.... [100%]`，exit code 0，4 passed，0 skipped，0 failed。剩余 gate：H2 已解除 audit E2E / H11 上游阻塞；H11 仍需 H5、H6 与 service ready 等前置。
- 2026-06-05（甲）：完成 W4 focused validation 复跑并记录到 `docs/todo/owner-a-generation-upstream/2026-06-05-W4-focused-validation-record.md`。通过项：W2 producer 8 项、HFM/JMCG consumer 12 项、C1 generator_coord 20 项、C2 validation/srb 32 项、W3 mf_eval 24 项、W11 quality 6 项 + strict CLI smoke、W13 18 项 + KD CLI smoke、W9/W10/W11 hardening 4 项。未通过项已分类：W1 unit gate 14 项中 3 项失败，根因是乙侧测试 patch `orchestrator_svc.main.build_shared_crg_repository_from_env` 但该 symbol 不在模块级导出；W5 benchmark 18 项中 8 failed/10 skipped，失败来自本地 `CCO` baseline 达不到 GuacaMol/PMO 生产阈值，skip 来自官方 benchmark env 缺失。未改业务代码，未改 Owner B 代码，C1/C2/C3 字段无变更。
- 2026-06-06（甲）：完成 W11 HUMU-labeled FragFM 50-record smoke gate。新增 `mf_generators.fragfm.humu_labeling`，直接用 frozen `HUMUMoleculeEncoder` + `checkpoint["encoder_mol"]` 派生 129 维 Lorentz `humu_embedding`，写入 `data/processing/generator_artifacts/fragfm_records_train_humu_labeled.jsonl`，report 显示 `status=pass`、`encoded_records=50`、`humu_embedding_coverage=1.0`。训练到新目录 `checkpoints/fragfm_humu_smoke/`，未覆盖 `checkpoints/fragfm`；`training_manifest.json` 显示 `humu_embedding_count=50`、coverage 1.0；`mf_generators.fragfm.quality --min-humu-coverage 1.0 --strict` exit code 0，checkpoint/rate-matrix loadable。新增 4 条 focused HUMU-labeling tests 通过；`python3 -m py_compile` 通过。该 artifact 仍是 50-record/1-epoch local engineering smoke，不是 production-quality W11 evidence；C1/C2/C3 字段无变更。
- 2026-06-06（甲）：完成 W11 5000-record HUMU labeling input gate，但未完成 5k training artifact。`data/processing/generator_artifacts/fragfm_records_humu_labeled.jsonl` 和 paired report 已生成，report `status=pass`、`encoded_records=5000`、`humu_embedding_coverage=1.0`、invalid counts 0；scan 确认全部 embedding 为 finite 129 维 Lorentz full-coordinate。两次 CPU training attempt 试图产出 `checkpoints/fragfm_humu_5k/`，分别使用 batch 64 与 batch 5000，均长时间运行后只写出 `vocab.json`，没有 checkpoint/manifest；partial 目录已删除。`checkpoints/fragfm_humu_5k/` 不存在，不得当作 artifact 引用。剩余 gate：GPU/cluster 训练或训练优化、production-quality artifact、benchmark/coverage threshold、cluster validation；C1/C2/C3 字段无变更。
- 2026-06-06（甲）：W11 FragFM 5k CPU training bottleneck 诊断与窄优化。诊断发现 5000-record dataset 有 2860 个 unique fragments，原 `_rate_transition_loss()` 会为每个 batch 物化 `[batch, vocab, vocab]` full rate tensor；第一版 sparse path 仍会为每个样本读取完整 `vocab*vocab` SA modulation 后切行。新增 sparse transition-row + SA row-gather path，并保留 custom rate matrix fallback；focused regression 证明 sparse loss 与 full matrix loss 等价，且不再调用 full `sa_score_embedding.forward()`。验证：`python3 -m py_compile` 通过；focused pytest 6 项通过；50-record `/tmp` training smoke 产出 checkpoint/rate-matrix/vocab/manifest 且 coverage 1.0。重试 5k training 仍未完成；`checkpoints/fragfm_humu_5k/` 不存在，后续需要 GPU/cluster 或进一步训练优化；C1/C2/C3 字段无变更。
- 2026-06-06（甲）：W11 FragFM rate optimizer controls。新增显式 `--rate-optimizer {adamw,sgd}` 与 `--disable-rate-grad-clip`，默认仍为 AdamW + rate grad clipping；manifest 记录 `rate_optimizer` 与 `rate_grad_clip`，checkpoint/rate-matrix schema 不变。验证：新 CLI regression 先 RED 后 GREEN；focused pytest 7 项通过；50-record HUMU-labeled `/tmp` smoke 和 256-record 5k-subset smoke 均能用 `--rate-optimizer sgd --disable-rate-grad-clip` 产出 checkpoint/rate-matrix/vocab/manifest 且 coverage 1.0。该时点 `checkpoints/fragfm_humu_5k/` 尚未产出，后续 5000-record local candidate 记录见下一条；C1/C2/C3 字段无变更。
- 2026-06-06（甲）：W11 5000-record HUMU-labeled FragFM local candidate 完成。使用 `data/processing/generator_artifacts/fragfm_records_humu_labeled.jsonl` 训练到 `checkpoints/fragfm_humu_5k/`，命令为 1 epoch、batch 64、hidden dim 8、CPU、`--rate-optimizer sgd --disable-rate-grad-clip`。manifest：`records=5000`、`fragments=2860`、`humu_embedding_count=5000`、coverage 1.0、`rate_optimizer=sgd`、`rate_grad_clip=false`；schema check：checkpoint `fragment_encoder.weight` shape `(2860, 8)`，rate matrix `base_rate` shape `(2860, 2860)`。strict quality gate `--min-humu-coverage 1.0` 输出 `status=pass`、checkpoint/rate-matrix loadable、messages empty。该 artifact 是 local engineering candidate，不是 final production W11 acceptance；剩余 benchmark/production training/deployment/cluster gate 未完成；C1/C2/C3 字段无变更。
- 2026-06-06（甲）：W11 FragFM HUMU 5k deployment defaults hardening 完成。Docker Compose、raw Kubernetes、Helm `fragfm-generator-config` 默认值从旧 `checkpoints/fragfm/{vocab.json,best_model.pt,rate_matrix.pt}` 切到 `checkpoints/fragfm_humu_5k/{vocab.json,best_model.pt,rate_matrix.pt}`，保留 env override；focused deployment regression 先 RED 后 GREEN，当前 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/unit/test_service_artifact_status.py::test_fragfm_deployment_wires_artifact_and_sampler_env -q` 通过 1 项，并校验 artifact 文件存在、`quality_report.json` status pass、coverage 1.0。该步骤不是 cluster acceptance；剩余 production-quality training、benchmark、正式阈值、artifact promotion policy、cluster validation；C1/C2/C3 字段无变更。
- 2026-06-05（H8）：H8 官方 benchmark 数据对接阻塞记录。已投放 MOSES 官方 test split：`MOSES_REFERENCE_SMILES_PATH=data/benchmarks/moses_reference_smiles.smi`，来源为 `molecularsets/moses` 官方 `data/dataset_v1.csv`，筛选 `SPLIT=test` 得到 176074 条 SMILES；`.env` 中 `HFM_CHECKPOINT_PATH`、`HFM_DECODER_PATH` 已设置且路径存在，但当前 decoder JSON 带本地 pytest 临时产物痕迹，不能作为 production-quality HFM artifact 完成证据。验证：`PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark -q -ra` 首次 exit code 0，18 skipped；加载 `.env` 后 `timeout 180s bash -lc 'set -a; source .env; set +a; PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark -q -ra'` exit code 124，无通过证据；最小 GuacaMol smoke `timeout 120s bash -lc 'set -a; source .env; set +a; export GUACAMOL_BENCHMARK_BATCH_SIZE=1; PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark/guacamol_benchmark.py::TestGuacaMolBenchmark::test_celecoxib_rediscovery -q -ra'` exit code 124。剩余 gate：甲方/人工提供可信 `PMO_SCORE_TABLE_PATH`（需 `smiles,drd2,jnk3,gsk3b`）、可信 `CROSSDOCKED_BENCHMARK_JSONL`（需 `pocket_id,ligand_smiles,split`，正式 gate 还需真实 `docking_score`）、production-quality `HFM_CHECKPOINT_PATH`+`HFM_DECODER_PATH` 或生产 decoder command，并设定正式阈值；H8 未完成，不登记完成验收。
- 2026-06-08（H8）：推进 H8 smoke 资源。基于本地 `/workspace/MForge/zzzzz/types/it2_tt_v1.3_completeset_test0.types` 与 `data/processing/crossdocked_full_extract.tmp` 中真实 SDF/GNINA 数据生成 `data/benchmarks/crossdocked_benchmark.jsonl`，共 1000 条 `split=test` 记录，字段含 `pocket_id`、`ligand_smiles`、`docking_score`；`.env` 新增 `CROSSDOCKED_BENCHMARK_JSONL=data/benchmarks/crossdocked_benchmark.jsonl`。验证：`timeout 180s bash -lc 'set -a; source .env; set +a; PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark/crossdocked_benchmark.py -q -ra'` exit code 0，4 passed，0 skipped。PMO 方向按官方 `wenhao-gao/mol_opt` 的 `data/zinc.csv.gz` 下载到 `data/benchmarks/pmo_zinc.csv.gz`，但当前环境缺 `tdc`；`timeout 600s uv pip install PyTDC` exit code 124，`timeout 60s env PYTHONDONTWRITEBYTECODE=1 uv run python - <<'PY' ... from tdc import Oracle ... PY` exit code 1，错误 `ModuleNotFoundError: No module named 'tdc'`，因此 `PMO_SCORE_TABLE_PATH` 仍未生成。HFM 现有 `checkpoints/hfm3d_4h200` 仅作为 smoke artifact 使用。剩余 gate：完成 PyTDC/PMO oracle 环境安装并生成可信 `PMO_SCORE_TABLE_PATH`，再跑 W5 benchmark 非 skip gate；H8 未完成。
- 2026-06-08（H8）：解除 PMO score table blocker，但不登记 H8 完成。新建专用 PMO 环境 `.venv-h8-pmo`，用 PMO 官方 `wenhao-gao/mol_opt` 的 `data/zinc.csv.gz` 和 PyTDC/TDC oracle 模型 `drd2_current`、`jnk3_current`、`gsk3b_current` 生成 `data/benchmarks/pmo_score_table.csv`，`.env` 新增 `PMO_SCORE_TABLE_PATH=data/benchmarks/pmo_score_table.csv`；表结构为 `smiles,drd2,jnk3,gsk3b`，2 条真实 scored ZINC rows，预检 `max_drd2=0.952971043107`、`max_pair_jnk3_gsk3b=0.635`。验证：PMO focused gate `timeout 180s bash -lc 'set -a; source .env; set +a; PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark/pmo_benchmark.py::TestPMOBenchmark::test_drd2_optimization tests/benchmark/pmo_benchmark.py::TestPMOBenchmark::test_multi_objective_jnk3_gsk3b -q -ra'` exit code 0，2 passed，0 skipped；CrossDocked 回归 `timeout 180s bash -lc 'set -a; source .env; set +a; PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark/crossdocked_benchmark.py -q -ra'` exit code 0，4 passed，0 skipped；总 W5 benchmark `timeout 300s bash -lc 'set -a; source .env; set +a; PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark -q -ra'` exit code 1，8 passed，9 failed，1 skipped。失败集中在当前 HFM smoke artifact 只生成重复 `CCO`，未达到 GuacaMol/MOSES/PMO 生成质量阈值（如 Celecoxib similarity 0.028571 < 0.75、MOSES uniqueness 0.00390625 < 0.95、PMO LogP/QED/LogP_QED 低于默认阈值）。剩余 gate：production-quality HFM checkpoint/decoder 或生产 decoder command、正式生成质量阈值、`uv run pytest tests/benchmark -q` 全量通过；H8 仍未完成。
- 2026-06-05（H6）：完成 H6 AiZynth真实 command path smoke gate。改动：`tools/retrosyn/aizynth_planner_wrapper.py` 支持真实 ONNX AiZynthFinder + 显式 inline stock，保留原 service/schema；`.env` 启用 H6 search limits 与 `RETROSYN_PLANNER_COMMANDS_JSON` AiZynth command；`tests/unit/test_h6_retrosyn_wrapper.py` 补 focused contract。验证：`uv run pytest tests/unit/test_h6_retrosyn_wrapper.py -q` exit code 0，2 passed；retrosyn command focused pytest exit code 0，3 passed；`uv run ruff check tools/retrosyn/aizynth_planner_wrapper.py tests/unit/test_h6_retrosyn_wrapper.py` exit code 0；`python -m py_compile tools/retrosyn/aizynth_planner_wrapper.py tests/unit/test_h6_retrosyn_wrapper.py` exit code 0；source `.env` 后 `runtime_status` 显示 `retrosyn_aizynth_planner_command configured=True available=True`；真实 wrapper smoke `timeout 360 uv run python tools/retrosyn/aizynth_planner_wrapper.py` 对 `CCO/max_routes=1/engine=aizynth` exit code 0，`total_routes_found=1`，非空真实 route；真实 service command path smoke `RetrosynServicer().FindRoutes(... engine="ensemble")` exit code 0，`total_routes_found=1`。剩余 gate：RetroGNN/RSGPT/UAlign 生产 runner、集群发布与生产多引擎验收仍未完成；C1/C2/C3 无变更。
- 2026-06-05（H4）：完成 H4 L4 PySCF quantum command path smoke gate。资源投放：`.env` 新增 H4 key：`L4_QUANTUM_ORACLE_COMMAND` 指向 `tools/oracles/pyscf_quantum_oracle_wrapper.py`，`L4_QUANTUM_ENGINE=pyscf`，`L4_PYSCF_METHOD=RHF`，`L4_PYSCF_BASIS=sto-3g`；当前 `.venv` 已安装 `pyscf==2.13.1` 与 `gpu4pyscf-cuda12x==1.7.1`。真实 smoke：wrapper 命令对 `C` exit code 0，返回非空 `scores.quantum_correction=-39.72460094981219`；从 `.env` 读取 H4 key 后构造 `QuantumCommandOracle` 的 command path smoke exit code 0，返回 `{"C": {"engine": "pyscf", "quantum_correction": -39.724600949812164}}`。focused gate：`uv run pytest tests/unit/test_validation_agent.py -k "quantum_command or l4_quantum" -q` exit code 0，6 passed。剩余 gate：GPU4PySCF `from gpu4pyscf import scf` 本机超过 5 分钟未返回，未登记 GPU4PySCF 真实计算完成；`command -v orca` 仍 missing，ORCA 未投放；集群发布验证仍未完成。C1/C2/C3 无变更。
- 2026-06-06（H4）：完成 H4 L4 GPU4PySCF quantum command path smoke gate。资源投放：`.env` H4 默认 engine 切到 `L4_QUANTUM_ENGINE=gpu4pyscf`，`L4_QUANTUM_ORACLE_COMMAND="uv run python tools/oracles/pyscf_quantum_oracle_wrapper.py"`、`L4_PYSCF_METHOD=RHF`、`L4_PYSCF_BASIS=sto-3g` 保持 H4 wrapper 合同；`.venv` 中 `pyscf==2.13.1`、`gpu4pyscf-cuda12x==1.7.1`、`cupy-cuda12x==13.6.0` 可探测。native artifact：投放 H200/sm_90 可执行 GPU4PySCF 库 `libgvhf_rys.so`、`libgvhf_md.so`、`libcupy_helper.so`，并投放 H2/STO-3G RHF 所需真实 `libgint.so`（含 `cart2sph_*` + `GINTinit_*`/basis cache 符号；原 wheel 备份在 `h4_sm90_backup_20260606/`）。真实 smoke：wrapper smoke `timeout 760 bash -lc 'printf ... engine=gpu4pyscf | L4_PYSCF_METHOD=RHF L4_PYSCF_BASIS=sto-3g L4_QUANTUM_ENGINE=gpu4pyscf CUDA_VISIBLE_DEVICES=2 uv run python tools/oracles/pyscf_quantum_oracle_wrapper.py'` exit code 0，返回非空 `scores.quantum_correction=-1.1174874250696716`；从 `.env` 读取 H4 key 后构造 `QuantumCommandOracle` 的 service command path smoke `timeout 820 bash -lc 'set -a; source .env; set +a; export CUDA_VISIBLE_DEVICES=2; uv run python ...'` exit code 0，返回 `{"[H][H]": {"engine": "gpu4pyscf", "quantum_correction": -1.1174874250696716}}`。focused gate：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -p pytest_asyncio.plugin tests/unit/test_h4_quantum_wrapper.py tests/unit/test_validation_agent.py -k "quantum_command or l4_quantum or gpu4pyscf_rhf" -q` exit code 0，7 passed。剩余 gate：`command -v orca` exit code 1，ORCA 未投放；完整 upstream `gint` 全目标 sm_90 构建此前 timeout 124，当前验收仅覆盖 H4 wrapper H2/STO-3G RHF 所需真实 GPU4PySCF path；集群发布验证仍未完成。C1/C2/C3 无变更。
- 2026-06-07（H4）：完成 H4 L4 GPU4PySCF wrapper path 修复与复验。修复：`tools/oracles/pyscf_quantum_oracle_wrapper.py` 的 GPU 分支从 PySCF `.to_gpu()` 宽转换改为直接构造真实 `gpu4pyscf.scf.hf.RHF(mol).kernel()`，保留窄加载 `_patch_pyscf.py` 与 H4 stdin/stdout command 合同。真实 smoke：`timeout 760 bash -lc 'printf ... engine=gpu4pyscf | L4_PYSCF_METHOD=RHF L4_PYSCF_BASIS=sto-3g L4_QUANTUM_ENGINE=gpu4pyscf CUDA_VISIBLE_DEVICES=2 uv run python tools/oracles/pyscf_quantum_oracle_wrapper.py'` exit code 0，返回 `engine=gpu4pyscf`、`scores.quantum_correction=-1.1174874250696716`、`elapsed_ms=177998`。command path 验证：从 `.env` 读取 H4 key 后构造 `QuantumCommandOracle`，`timeout 820 bash -lc 'set -a; source .env; set +a; export CUDA_VISIBLE_DEVICES=2; uv run python ...'` exit code 0，返回 `{"[H][H]": {"engine": "gpu4pyscf", "quantum_correction": -1.1174874250696716}}`。focused gate：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -p pytest_asyncio.plugin tests/unit/test_h4_quantum_wrapper.py tests/unit/test_validation_agent.py -k "quantum_command or l4_quantum or gpu4pyscf_rhf" -q` exit code 0，8 passed；`uv run python -m py_compile tools/oracles/pyscf_quantum_oracle_wrapper.py tests/unit/test_h4_quantum_wrapper.py` exit code 0。剩余 gate：ORCA 未登记完成；集群发布验证仍未完成。C1/C2/C3 无变更。
- 2026-06-07（H5）：完成 H5 L1-L3 oracle command wrapper 本地验收，并补齐 FEP/OpenFE TYK2 registry gate。`.env` 已投放 `DOCK_ORACLE_COMMAND`、`BOLTZ2_ORACLE_COMMAND`、`FEP_ORACLE_COMMAND`、`ADMET_ORACLE_COMMAND` 及 Boltz/OpenADMET/OpenFE registry 运行所需 key，未记录任何 secret/token/key 具体值；本轮仅补充修改 H5 key `OPENFE_TRANSFORMATION_REGISTRY`、`OPENFE_RESULT_REGISTRY`。配置生效检查 `set -a; source .env; set +a; python ...` exit code 0，四个 command env 均为 set 且首个可执行均可解析，`OPENFE_RUNNER_PATH` 首个可执行可解析，`OPENFE_CLI_PATH`、`OPENFE_TRANSFORMATION_REGISTRY`、`OPENFE_RESULT_REGISTRY`、`OPENFE_WORK_DIR`、`FEP_JOB_DIR` 均存在。focused gate：`uv run pytest tests/unit/test_h5_oracle_wrappers.py -q` exit code 0，13 passed；服务 command 合同回归 `uv run pytest tests/unit/test_service_artifact_status.py -k 'dock_service_runs_configured_json_command or boltz2_service_runs_configured_json_command or fep_service_runs_configured_json_command or admet_service_runs_configured_json_command' -q` exit code 0，4 passed；FEP service focused 回归 `uv run pytest tests/unit/test_service_artifact_status.py -k 'fep_service_submits_background_json_command_job or fep_service_runs_configured_json_command or fep_oracle_service_maps_evaluations_to_rbfe_scores' -q` exit code 0，3 passed；`PYTHONPYCACHEPREFIX=/tmp/mforge-pycache-h5 uv run python -m py_compile ...` exit code 0；`git diff --check -- ...` exit code 0。真实 smoke：OpenADMET 主预测 smoke exit code 0，JSON parse OK，1 条 clearance float prediction；Boltz GPU affinity smoke 使用 6OIM/CCO、GPU、结构采样 10、affinity 采样 10，exit code 0，stdout 278 bytes，stderr 0 bytes，JSON parse OK，affinity_count=1，并生成 `6OIM_0_model_0.cif`、`confidence_6OIM_0_model_0.json`、`affinity_6OIM_0.json`。FEP/OpenFE 补验：`openfe fetch rbfe-tutorial`、`openfe fetch rbfe-tutorial-results`、`openfe gather ... --report dg --tsv`、`openfe gather ... --report ddg --tsv` 均 exit code 0；`openfe plan-rbfe-network ... --n-protocol-repeats 1 -s settings.yaml` exit code 0，持续约 17 分 58 秒，stderr 摘要为 multiprocessing fork DeprecationWarning 与 element-change UserWarning，无 fatal error。已生成 `models/artifacts/openfe/tyk2/transformation_registry.json`（9 条完整 complex/solvent 边）和 `models/artifacts/openfe/tyk2/result_registry.json`（18 条正反向 ddG 边）；官方 TYK2 结果 `final_results_ddg.tsv` 为 9 rows，DDG 范围 -0.89 到 1.4 kcal/mol，range 2.29；`final_results_dg.tsv` 为 10 rows，DG(MLE) 范围 -1.25 到 2.0 kcal/mol，range 3.25。FEP wrapper registry smoke exit code 0，stderr 空，返回 `ddg_kcal_mol=0.8`、`ddg_uncertainty=0.1`、`n_repeats=3`；FEP service background job smoke exit code 0，最终 `state=completed`、`results=1`、`ddg_kcal_mol=0.8`。诊断中确认 Boltz 标准 CLI 卡点来自 checkpoint load 前大模型随机初始化，已新增 fast CLI 入口跳过会被 checkpoint 覆盖的初始化，并把 `--num_workers` 固化到 runner；`sampling_steps=1` / `sampling_steps_affinity=1` 会触发 SVD 数值失败，已改用通过 smoke 的采样配置。TYK2 教程输入和 SDF 未包含实验 Ki/IC50 标签，因此该 registry 证明真实 OpenFE 模拟结果具有区分分布，不登记实验相关性；本地仅发现 KRAS G12C 6OIM 共价复合物 PDB 与 Boltz template，未发现 KRAS OpenFE 配体系列、实验 ddG 或 covalent-FEP registry，KRAS full pilot 仍归 H11。H5 本地 command wrapper / ADMET / Boltz / FEP TYK2 registry gate 已完成；集群发布与 KRAS full pilot 仍归 H10/H11。C1/C2/C3 无变更。
- 2026-06-07（H6）：完成 H6 多引擎 retrosynthesis 本地真实 command path gate。`.env` 已接入 AiZynth、UAlign、RSGPT 三个 `RETROSYN_PLANNER_COMMANDS_JSON` runner；RSGPT 使用真实 `finetune_50k.pth`、官方 `rxngpt_llama1B.json` 和匹配 1000 vocab `vocab.json`，RetroGNN 按用户决定舍弃。验证：`uv run pytest tests/unit/test_h6_retrosyn_wrapper.py -q` exit code 0，7 passed；retrosyn command focused pytest exit code 0，3 passed；`uv run ruff check tools/retrosyn/aizynth_planner_wrapper.py tools/retrosyn/rsgpt_planner_wrapper.py tools/retrosyn/ualign_planner_wrapper.py tests/unit/test_h6_retrosyn_wrapper.py` exit code 0；`python -m py_compile ...` exit code 0；source `.env` 后 runtime_status 显示 `retrosyn_aizynth_planner_command`、`retrosyn_ualign_planner_command`、`retrosyn_rsgpt_planner_command` 均 configured=True available=True。真实 smoke：RSGPT wrapper 对 `CCO/max_routes=1/engine=rsgpt` exit code 0，`total_routes_found=1`，route `CCOC(=O)CBr>>CCO`；UAlign wrapper exit code 0，`total_routes_found=1`；service ensemble command path `RetrosynServicer().FindRoutes(... engine="ensemble")` exit code 0，`total_routes_found=3`，返回 AiZynth/RSGPT/UAlign 非空 routes。剩余 gate：集群发布验证与 KRAS full pilot 仍归 H10/H11；C1/C2/C3 无变更。
- 2026-06-07（H6）：完成 H6 Layer A 从 RetroGNN 设想替换为真实 RAscore 快筛并纳入本地四引擎 command path。改动：`tools/retrosyn/rascore_planner_wrapper.py` 使用 RAscore XGB ChEMBL ECFP counts 真实模型，官方旧 pickle 已转换为当前 XGBoost Booster JSON 且分数对齐；`RETROSYN_PLANNER_COMMANDS_JSON` 接入 `rascore/aizynth/ualign/rsgpt`；service/agent 命名 runner env 从 `RETROGNN_PLANNER_COMMAND` 改为 `RASCORE_PLANNER_COMMAND`，并保证 RAscore 可及性分数不会在 `max_routes` 截断时挤掉真实 reaction route。验证：`uv run pytest tests/unit/test_h6_retrosyn_wrapper.py -q` exit code 0，9 passed；retrosyn command focused pytest exit code 0，3 passed；RAscore named/ranking focused pytest exit code 0，4 passed；`uv run ruff check tools/retrosyn/rascore_planner_wrapper.py tools/retrosyn/aizynth_planner_wrapper.py tools/retrosyn/rsgpt_planner_wrapper.py tools/retrosyn/ualign_planner_wrapper.py tests/unit/test_h6_retrosyn_wrapper.py` exit code 0；`python -m py_compile tools/retrosyn/rascore_planner_wrapper.py tools/retrosyn/aizynth_planner_wrapper.py tools/retrosyn/rsgpt_planner_wrapper.py tools/retrosyn/ualign_planner_wrapper.py tests/unit/test_h6_retrosyn_wrapper.py` exit code 0；source `.env` 后 runtime_status 显示 `retrosyn_rascore_planner_command`、`retrosyn_aizynth_planner_command`、`retrosyn_ualign_planner_command`、`retrosyn_rsgpt_planner_command` 均 configured=True available=True。真实 smoke：RAscore wrapper 对 `CCO/max_routes=1/engine=rascore` exit code 0，`total_routes_found=1`，`accessibility_score=0.990022599697113`；service ensemble command path `RetrosynServicer().FindRoutes(... engine="ensemble", max_routes=4)` exit code 0，`total_routes_found=4`，返回 AiZynth/RSGPT/UAlign 非空真实 routes + RAscore 可及性评分，`elapsed_ms=1990311`。剩余 gate：H6 本地真实 command path 已完成；集群发布验证与 KRAS full pilot 仍归 H10/H11；C1/C2/C3 无变更。
- 2026-06-07（H3）：H3 商业供应商真实 API 设想按用户决定舍弃，不登记完成验收。H3 已从 C 类资源域移除，不再采购或投放 Enamine/Mcule/eMolecules/Chemspace 四组真实 API / sandbox，也不运行四家商业供应商真实 smoke。清理范围：删除 `supply-oracle-svc` 商业 HTTP provider / aggregator / retry-backoff provider wiring；保留 `SUPPLY_CATALOG_URI=file://...` 本地 JSON catalog 与 AiZynth HDF5 stock 路径；Docker Compose、raw Kubernetes、Helm 删除 `SUPPLY_COMMERCIAL_*`、四家 `SUPPLY_*_API_*`、`commercial-supply-config`、`commercial-supply-credentials`；删除 `data/ingestion/enamine_real/` Enamine REAL FAISS placeholder；清理相关待办/架构/测试文案中的供应商 API、Enamine REAL 占位和供应商特定样例源名。验证：`.env` H3 key 搜索无匹配；`PYTHONDONTWRITEBYTECODE=1 uv run python ... compile(...)` exit code 0，`syntax_ok`；Supply focused gate `timeout 180s env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 uv run pytest -p pytest_asyncio.plugin ... -q --tb=short` exit code 0，7 passed；`services/supply-oracle-svc/src/supply_oracle_svc/__init__.py` 包级旧描述已清理并恢复 mode `664`。H3 无剩余 blocker；C1/C2/C3 无变更。
- 2026-06-07（W12）：完成 W12 CReM-pharm-3D 本地真实 scorer runner 闭环。`.env` 已投放 `CREM_DOCK_ORACLE_TARGET`、`DOCK_ORACLE_RECEPTOR_PDB`、`CREM_PHARMACOPHORE_REFERENCE_SDF`、`CREM_PHARMACOPHORE_SCORER_COMMAND`、`CREM_HUMU_SCORER_COMMAND`、`CREM_SCORER_COMMAND_TIMEOUT_SECONDS`，未记录任何 secret/token/key；DiffDock-L 按用户决定移出 W12 本地 gate。真实 artifact：RCSB 6OIM/MOV receptor/reference SDF、既有 `HUMU_CHECKPOINT_PATH`、H5 `DOCK_ORACLE_COMMAND`/GNINA。验证：`PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/unit/test_phase_b_generators.py -q -k "crem"` exit code 0，6 passed；`PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/unit/test_service_artifact_status.py -q -k "crem_service_builds_configured_external_scorers or crem_external_json_score_provider_returns_smiles_records or crem_external_json_score_provider_preflight_rejects_missing_executable or crem_deployment_wires_mmp_and_external_scorer_env or crem_runtime_rejects_missing_external_scorer_command or dock_oracle_uses_default_receptor_for_oracle_requests"` exit code 0，6 passed；`PYTHONPYCACHEPREFIX=/tmp/... uv run python -m py_compile ...` exit code 0；`git diff --check -- .env tools/scorers/crem_humu_scorer.py tools/scorers/crem_pharmacophore_scorer.py services/dock-svc/src/dock_svc/main.py tests/unit/test_phase_b_generators.py tests/unit/test_service_artifact_status.py` exit code 0。真实 smoke：pharmacophore command 返回 `pharmacophore_score=0.5055679672060487`；HUMU command 返回 129 维 embedding 与 `humu_alignment_score=-0.12193988789716237`；dock gRPC scorer 返回 `oracle_name=gnina`、`docking_score=2.11969`；CReM generator 全链路 smoke exit code 0，1 条 molecule 同时包含 `docking_score=2.37787`、`pharmacophore_score=0.6224669558183266`、`humu_alignment_score=-0.07110921506371735` 且 `humu_embedding` 非空。剩余 gate：H10 集群发布验证；生产级 pharmacophore reference 策略和正式 benchmark 归后续生产验收。C1/C2/C3 无变更。
- 2026-06-08（H9）：完成 H9 CIG 云 LLM parser/refiner 真实接入本地 smoke gate。`.env` 已投放 H9 key：`CIG_DEEPSEEK_MODEL=deepseek-v4-flash`、`CIG_SEMANTIC_PARSER_COMMAND`、`CIG_SEMANTIC_PARSER_TIMEOUT_SECONDS`、`CIG_REFINEMENT_COMMAND`、`CIG_REFINEMENT_TIMEOUT_SECONDS`；DeepSeek API key 仅验证为 set，未记录任何 secret/token/key。真实 smoke：parser command exit code 0，返回 KRAS G12C target 与 `max_mw` 约束；`ProductionSemanticParserAdapter()` command path smoke exit code 0；`CIGCompilerServicer(compiler=CIGCompiler(enable_grounding=False)).Compile(...)` exit code 0，输出 CIG、129 维 HCIV 和 129 维 cone；refiner direct command smoke exit code 0；`CIGCompilerServicer().Refine(...)` service command path smoke exit code 0；`runtime_status()` 显示 `cig_semantic_parser_command` 与 `cig_refinement_command` configured=true、available=true；`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/unit/test_h9_cig_llm_wrappers.py -q` exit code 0，8 passed；`py_compile` 与 `git diff --check` 通过。设想调整：真实项目场景直接使用云 LLM API 解析用户意图，不再要求为 H9 parser/refiner 提前训练本地模型；此前 `h9_sklearn_hashing_smoke.pt` 和 teacher JSONL 仅作为历史 smoke artifact，不作为 W10 production-quality 目标。剩余 gate：W10 改为无训练 canonical CIG → deterministic Lorentz HCIV production encoder；`HCIV_CHECKPOINT_PATH` 仅保留为显式 learned 模式增强；仍需集群发布验证、外部 grounding 打开后的端到端验收和下游质量验证。C1/C2/C3 无变更。
- 2026-06-09（H2/H11 runner 操作记录）：已将 self-hosted runner `mforge-h2-audit` 使用方法落地到 `/workspace/MForge/actions-runner/RUNNER_USAGE.md`。运行方式：一次性执行 `cd /workspace/MForge/actions-runner && ./run.sh` 并保持终端打开；监控 run 使用 `cd /workspace/MForge && gh run watch <run-id> --repo weiyu1218/MForge --interval 10 --exit-status`；状态检查使用 `gh api repos/weiyu1218/MForge/actions/runners --jq '.runners[] | select(.name=="mforge-h2-audit") | {status,busy,labels:[.labels[].name]}'`；job 完成后在 runner 终端 `Ctrl+C` 停止，长期在线可用 `sudo ./svc.sh install/start/status/stop`。H11 retry：`gh run rerun 27187527634 --repo weiyu1218/MForge --failed` exit code 0，新 job `80311379572`；记录时 run/job 仍 queued，runner `mforge-h2-audit` offline、busy=false，steps 为空，因此尚无新的 H11 测试结果。Sigstore token 仍只由 GitHub Actions OIDC 运行时注入并 mask，不写入 `.env`，未记录任何 secret/token/key。H11 未完成，不登记完成验收。
- 2026-06-10（H9/W10 设想调整）：按用户确认，将真实项目默认路径改为云 LLM API 直接解析用户意图，不再要求为 H9 parser/refiner 预训练本地模型；W10 默认从 `HCIV_CHECKPOINT_PATH` learned checkpoint 切换为无训练 canonical CIG → deterministic Lorentz HCIV encoder，`HCIV_CHECKPOINT_PATH` 仅保留为显式 `learned` 模式增强。`.env` 已设置 `CIG_HCIV_ENCODING_MODE=canonical`，并移除默认 `HCIV_CHECKPOINT_PATH`。验证：新增 canonical HCIV 稳定性/合法性测试与 production 默认无 checkpoint 测试；`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/unit/test_cic_compiler.py -q` exit code 0，35 passed；真实 service smoke 在 `.env` 无 `HCIV_CHECKPOINT_PATH` 情况下 exit code 0，`encoding_mode=canonical`，输出 129 维 HCIV 和 129 维 cone。剩余 gate：集群发布验证、外部 grounding enabled 后端到端验收、下游质量验证。C1/C2/C3 无变更。
