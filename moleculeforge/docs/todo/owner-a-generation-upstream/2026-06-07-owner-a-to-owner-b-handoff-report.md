# MoleculeForge 甲方到乙方项目对接报告

日期：2026-06-07
发送方：甲方，Owner A，generation-upstream
接收方：乙方，Owner B，validation / retrosynthesis / supply / SRB / critic / provenance downstream
项目根目录：`/workspace/MForge/moleculeforge`

## 1. 报告结论

甲方 Owner A 的生成上游代码主体已经进入可对接状态，可以交给乙方进行接口复核、W1/W3/W5/W12 下游接手和联调准备。

但本报告必须明确区分三个状态：

- 甲方代码工程完成度：主体已完成，当前接近 code-freeze，可进入乙方对接。
- 双方联调完成度：尚未完成，需要乙方按本报告流程执行 W1/W3/W5/W12 复核、修正和验收。
- 生产交付完成度：尚未完成，还缺正式 benchmark、真实生产训练、cluster 验证、production artifact promotion 和最终集成验收。

因此，当前不是“项目全部完成”，而是“甲方生成上游代码进入乙方对接阶段”。乙方接手后，应优先处理 W1 unit patch seam、W5 benchmark resource gate、C1/C2/C3 契约复核，以及 W3/W12 下游链路确认。

## 2. 项目主流程和双方边界

MoleculeForge 的 CoreArchitecture v2 主链路为：

```text
natural language intent
  -> CIG / HCIV intent compilation
  -> generation upstream
  -> validation / oracle cascade
  -> retrosynthesis / supply / SRB / critic
  -> provenance and CRG belief recording
  -> feedback into the next generation round
```

甲方和乙方按 generation 边界切分：

| 范围 | 甲方 Owner A | 乙方 Owner B |
|---|---|---|
| 主职责 | 生成上游、条件生成、HFM/FragFM/TAR/JMCG/KD/HCIV 相关工程路径 | 验证、oracle、retrosynthesis、supply、SRB、critic、provenance、benchmark |
| Work items | W2, W6, W8, W9, W10, W11, W13 | W1, W3, W5, W12 |
| 主要目录 | `models/mf-generators/*`, `agents/generator_coord/`, `agents/nl2obj/`, `services/cig-compiler-svc/`, `services/*generator*svc/`, `libs/mf-core/src/mf_core/routing/` | `agents/validation_agent/`, `agents/retrosyn_agent/`, `agents/supply_agent/`, `agents/srb_agent/`, `agents/critic_agent/`, `services/*oracle*`, `services/provenance-svc/`, `libs/mf-core/src/mf_core/db/`, `libs/mf-eval/` |

共享且需要协调的文件：

- `services/orchestrator-svc/src/orchestrator_svc/main.py`
- `agents/generator_coord/src/generator_coord/agent.py`
- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py`

乙方如果需要修改共享文件，应先在对应交接记录中写清楚函数或 hunk，不要和甲方并行改同一区段。

## 3. 不可突破的边界

以下边界在乙方接手时仍然有效：

- 不要修改 HUMU pretraining 配置、loss、encoder 架构或 checkpoint continuation。
- 不要覆盖以下受保护 artifact：
  - `checkpoints/fragfm`
  - `checkpoints/humu`
  - `checkpoints/hfm3d_4h200`
- `/workspace/SemMol` 和 `/workspace/Projects` 只允许读取或复制参考内容，不允许写入、执行或作为输出目录。
- 不要打印 `.env` 中的 secret 值。
- 不要为了让 W5 变绿而放宽 MOSES、GuacaMol 或 PMO 阈值。
- 不要把 `checkpoints/fragfm_humu_5k/` 宣称为 production artifact。它是 strict-local engineering evidence。
- 不要把 W8-E 工程骨架宣称为 W8-R 研究完成。W8-R 仍需要真实联合训练质量证据。
- 不要启动大规模训练或 distillation，除非甲方和项目方明确重新授权。

## 4. 关键技术契约

乙方对接时必须先复核 C1/C2/C3 三个契约面。若乙方需要新增字段或 predicate，应先登记并获得双方确认。

### 4.1 C1：generator_params feedback envelope

统一 envelope：

```json
{
  "schema": "moleculeforge.jmcg.feedback.v1",
  "run_id": "run-id",
  "project_id": "project-id",
  "records": []
}
```

当前 generator 参数中存在三类 key：

- `jmcg_feedback`：新 envelope。
- `route_humu_feedback`：legacy route feedback。
- `generation_feedback`：legacy generation feedback。

HFM 消费规则：

- HFM 扫描顺序为 `jmcg_feedback`、`route_humu_feedback`、`generation_feedback`。
- feedback record 中有合法 `humu_embedding` 或 `route_humu_embedding` 才可用于 steering。
- metadata-only property / pocket / intent record 会保留 provenance，但不会 steering。
- 甲方 W2 已实现 pocket / intent 的保守升级：只有 valid 129-dimensional Lorentz full-coordinate embedding 才能成为 steering-capable。
- 128-dimensional payload 不能被静默当作 steering embedding。

乙方需要确认：

- 下游 orchestrator 派生 property / intent / pocket record 时，不要破坏 envelope schema。
- 乙方写入的 route feedback 或 CRG route HUMU belief 能被甲方 generator_coord 解析。
- non-steering record 不应改变 HFM latent，只应保留上下文和 provenance。

### 4.2 C2：CRG belief predicate table

当前跨界关键 predicate：

| predicate | 写入方 | 读取方 | 说明 |
|---|---|---|---|
| `workflow_status` | OrchestratorAgent | OrchestratorAgent | 公共 workflow 状态 |
| `parsed_intent` | NL2ObjAgent | NL2ObjAgent | 甲方 intent parse |
| `compiled_cig` | NL2ObjAgent | NL2ObjAgent | 甲方 CIG compile |
| `selected_generators` | GeneratorCoordAgent | GeneratorCoordAgent | 甲方 generator selection |
| `route_humu_embedding` | RetroSynAgent | GeneratorCoordAgent | 乙方写，甲方读，跨界重点 |
| `validation_status` | ValidationAgent | RetroSyn/Critic | 乙方 |
| `retrosyn_routes` | RetroSynAgent | Supply/SRB/Critic | 乙方 |
| `supply_feasibility` | SupplyAgent | SRB/Critic | 乙方 |
| `ssp_compiled` | SRBAgent | 下游使用方 | 乙方 |
| `critic_verdict` | CriticAgent | CriticAgent | 乙方 |

乙方需要确认：

- `route_humu_embedding` payload 至少包含 `humu_embedding` 和 `route_id`。
- 可选字段包括 `source`、`weight`、`polarity`、`confidence`、`evidence_ids`、`metadata`。
- W1 的 final-state CRG merge 应在 provenance 记录前完成。
- 不要让 HFM 直接读 shared CRG；HFM 默认通过 generator_params feedback 消费。

### 4.3 C3：HUMU encoder interface

HUMU 当前维度契约非常关键：

- encoder 构造参数 `dim=128` 表示空间维度。
- Lorentz full-coordinate 实际长度为 `dim + 1 = 129`。
- molecule / pocket / route encoder 都输出 129-dimensional Lorentz full-coordinate。
- HFM 当前 active latent 与 steering-capable feedback embedding 也要求 129 维。
- 乙方 RetroSyn 写回 route HUMU embedding 时，应使用同一 HUMU checkpoint 空间，否则 embedding 不可比。

乙方需要确认：

- route HUMU embedding 仍按 129-dimensional Lorentz full-coordinate 写回。
- consumer 不只检查长度，还要检查 finite 和 Lorentz hyperboloid 合法性。
- `HUMU_CHECKPOINT_PATH` 应指向同一 local HUMU checkpoint：`checkpoints/humu/best_model.pt`，除非后续正式 artifact promotion。

## 5. 甲方已完成任务明细

### 5.1 W2：pocket / intent HUMU embedding producer

目标：

- 将 intent / pocket context 转为可被 HFM/JMCG 消费的 feedback record。
- 只在 embedding 合法时提供 steering；否则保留 metadata-only record。

已完成：

- orchestrator 可以产出 `moleculeforge.jmcg.feedback.v1` records。
- intent feedback 只有在已有 valid 129-dimensional Lorentz full-coordinate axis 时才 steering-capable。
- pocket feedback 只有在 structured pocket geometry 存在且 HUMU encoder 返回合法 129 维 embedding 时才 steering-capable。
- property feedback 保持 non-steering metadata，不直接改变 HFM latent。
- optional pocket HUMU enrichment 使用 `HUMU_ENCODER_TARGET`。
- HFM 侧会校验 feedback embedding，不合格则丢弃 steering，但保留上下文。

乙方影响：

- 乙方如果在 orchestrator 或下游流程中提供 property / intent / pocket record，需要保持 C1 schema。
- 乙方不要把 128-dimensional HCIV payload 直接当成 HUMU steering embedding。

当前证据：

- W2 producer focused tests 8 passed。
- W2/W8 HFM JMCG consumer focused tests 12 passed。
- C1 generator coordinator regression 20 passed。

剩余门：

- 无新的甲方代码主线缺口。
- 乙方需在联调时确认下游 record 不破坏 C1。

### 5.2 W6：TAR ProxylessNAS runner

目标：

- 提供一个可配置到 `TAR_PROXYLESS_SEARCH_COMMAND` 的本地 command target。
- 复用已有 `ProxylessSearchScheduler`，不重新实现 routing algorithm。

已完成：

- 新增本地 command target：

```bash
python -m generator_router_svc.tar_proxyless_runner
```

- stdin 接收 reward / cost JSON。
- stdout 输出 service-compatible JSON，包括 rounds、architecture probabilities、architecture logits、generator names 等。
- 适配 `GeneratorRouterServicer.RunProxylessSearch()` 的外部 command contract。

乙方影响：

- 乙方一般不需要修改 W6。
- 如果乙方 benchmark 或 downstream validation 要解释 generator routing 结果，需要保留 W6 输出字段。

当前证据：

- runner command smoke 已记录。
- focused task_router tests 已通过。

剩余门：

- 真实 reward payload。
- production `TAR_PROXYLESS_SEARCH_COMMAND` env value。
- cluster 发布和 service request evidence。

### 5.3 W8-E：JMCG engineering skeleton

目标：

- 先完成联合采样工程骨架，打通 `(molecule, route, pocket/property/intent)` context 输出。
- 明确 W8-E 工程验收和 W8-R 研究验收的边界。

已完成：

- `JMCGEngineeringSampler` 可输出 JSON-serializable `moleculeforge.jmcg.joint_sample.v1` engineering skeleton records。
- 可消费 HFM candidate、route/property/pocket/intent feedback。
- 对 HUMU embedding 使用 finite + Lorentz 合法性校验。
- 对缺失 embedding 或 128-dimensional payload 保持 non-steering context，不伪装为 steering。

乙方影响：

- 乙方的 RetroSyn route HUMU output 是甲方 JMCG/HFM feedback 的关键来源之一。
- 乙方需要保证 `route_humu_embedding` predicate payload 合法。

当前证据：

- W8-E local engineering path 已完成。
- HFM/JMCG focused regression 已通过。

剩余门：

- W8-R 真实联合采样训练。
- 联合质量 benchmark。
- production model artifact。

### 5.4 W9：HFM-3D neural geometry decoder path

目标：

- 提供真实 neural geometry decoder 的 train/export/runner 工程路径。
- 保留现有 `HFM_MOLECULAR_DECODER_COMMAND` command contract。

已完成：

- 新增 neural geometry decoder 训练和 runner 路径：
  - `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/neural_geometry_decoder.py`
  - `models/mf-generators/hfm_3d/train_geometry_decoder.py`
- decoder source artifact latent 会走 Lorentz validation。
- runner 可作为 `HFM_MOLECULAR_DECODER_COMMAND` 兼容 command target。
- `HFM3DGenerator` 保留 decoder payload 自带 `metadata.decoder_mode`。

乙方影响：

- 乙方 W5 benchmark 如果评估 HFM，需要知道当前默认 HFM decoder artifact 仍是 smoke-only，不是 production-quality decoder。
- 不应把 `checkpoints/hfm3d_4h200/decoder.json` 作为 production-quality benchmark artifact。

当前证据：

- W9 focused + legacy decoder gate 6 items passed。
- `tests/unit/test_generators.py -q` 的相关验证已通过。
- `python -m py_compile` 和 `git diff --check` 已通过。

剩余门：

- 真实 latent/SDF decoder source data。
- production-quality decoder artifact。
- `HFM_MOLECULAR_DECODER_COMMAND` 或 production decoder path 部署。
- geometry benchmark 和 cluster evidence。

### 5.5 W10：HCIV supervised train/export path

目标：

- 为 Enc_intent / HCIV 提供真实 supervised checkpoint 的训练和导出路径。
- 导出的 checkpoint 兼容 `HCIV_CHECKPOINT_PATH`。

已完成：

- 新增 supervised CIG + target HCIV 数据加载、训练和 export 路径：
  - `services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_training.py`
  - `services/cig-compiler-svc/train_hciv_encoder.py`
- `target_hciv` 通过 Lorentz validation。
- `HCIVEncoder` 新增可微 `forward_coordinates(cig)`，但 `encode()` 输出契约保持不变。

乙方影响：

- 乙方如果依赖 intent-conditioned generation 的下游质量，需要等真实 HCIV checkpoint。
- 当前工程路径已存在，但不能声称已有 production HCIV checkpoint。

当前证据：

- W10 focused gate 4 passed。
- `tests/unit/test_cic_compiler.py -q` 31 passed。
- compile 与 diff check 通过。

剩余门：

- 真实 supervised CIG/HCIV data。
- production-quality checkpoint。
- `HCIV_CHECKPOINT_PATH` 部署。
- downstream intent-conditioned generation evidence。
- cluster service evidence。

### 5.6 W11：FragFM shared HUMU conditional-space path

目标：

- 让 FragFM 进入共享 HUMU 条件空间。
- 提供 HUMU-labeled data、training manifest、quality CLI、runtime service 和 sample export 的工程闭环。

已完成的核心代码和 artifact：

- `mf_generators.fragfm.humu_labeling` 可用 frozen HUMU molecule encoder 生成 129-dimensional HUMU embeddings。
- FragFM training CLI 会校验并保留 valid 129-dimensional Lorentz `humu_embedding` 到 vocab artifact。
- training manifest 记录 HUMU coverage、requested device、actual device、`log_every`、rate optimizer controls 等信息。
- `mf_generators.fragfm.quality` 可输出 quality JSON report，检查 HUMU coverage、invalid embedding、checkpoint loadability、rate matrix loadability、strict schema。
- 50-record HUMU-labeled smoke 已完成。
- 5000-record HUMU-labeled input gate 已完成：
  - `data/processing/generator_artifacts/fragfm_records_humu_labeled.jsonl`
  - paired report shows 5000/5000 encoded and coverage 1.0。
- local 5k candidate 已完成：
  - `checkpoints/fragfm_humu_5k/vocab.json`
  - `checkpoints/fragfm_humu_5k/best_model.pt`
  - `checkpoints/fragfm_humu_5k/rate_matrix.pt`
  - `checkpoints/fragfm_humu_5k/quality_report.json`
- Docker Compose、raw Kubernetes、Helm 默认 FragFM paths 已从 old `checkpoints/fragfm` smoke artifact 切到 `checkpoints/fragfm_humu_5k`。
- FragFM service runtime hardening 已完成：
  - configured-but-missing optional checkpoint 会 fail fast。
  - configured-but-missing optional rate matrix 会 fail fast。
  - invalid `intent_cone` 会映射到 invalid argument，不会泄漏内部 parser/model exception。
- FragFM sample export hardening 已完成：
  - `sample_export.py` 会在构造 generator 前验证 output/report parent directory。
  - write failure 时保留 atomic cleanup 行为。
  - 已有 8 / 64 / 256 sample local wiring evidence。

乙方影响：

- 乙方 W5 benchmark 可以使用 FragFM sample export 生成 benchmark input，但必须区分 local wiring evidence 和 official acceptance。
- 乙方不能把 8/64/256 sample smoke 当作 production benchmark。
- 乙方不能把 `checkpoints/fragfm_humu_5k/` 宣称为 final production W11 artifact。
- W5 benchmark 仍要等待 official benchmark resources 和 production-quality generated samples。

当前证据：

- W11 service focused code-freeze shard：4 passed。
- W11 sample export focused code-freeze shard：4 passed。
- compile check 通过。
- deployment default scan 确认 Docker Compose、Kubernetes、Helm 指向 `checkpoints/fragfm_humu_5k`。
- process scan 没有训练或 pytest 残留。

剩余门：

- stronger production training，经明确授权后才可执行。
- production-scale generated samples。
- MOSES / GuacaMol / PMO official benchmark evidence，不放宽阈值。
- cluster cold-start、readiness、request/response evidence。
- production artifact promotion decision。

### 5.7 W13：Cross-paradigm KD teacher embedding artifact path

目标：

- 为 cross-paradigm KD 提供 canonical teacher embedding artifact export/report gate。

已完成：

- 新增 `mf_core.routing.kd_artifacts`。
- 可从 JSON/JSONL teacher records 导出 canonical artifact：

```json
{
  "schema_version": "cross_paradigm_teacher_embeddings.v1",
  "embedding_count": 2,
  "embedding_dim": 2,
  "teacher_embeddings": [[0.1, 0.2], [0.3, 0.4]]
}
```

- strict gate 检查 finite、consistent dimension、expected dimension、minimum embedding count。

乙方影响：

- 如果乙方 benchmark 或 downstream evaluation 需要比较 KD 前后质量，需要等真实 teacher source 和 distillation evidence。

当前证据：

- `tests/unit/test_cross_paradigm_kd.py -q` 18 passed。
- KD CLI smoke 已记录。

剩余门：

- real production teacher source。
- real teacher embeddings。
- distillation run。
- baseline-vs-KD benchmark。
- cluster evidence。

## 6. 乙方当前需要接手的事项

### 6.1 第一优先级：W1 unit patch seam

当前现象：

- `uv run pytest tests/unit/test_graph_repo.py -q` 曾出现 14 items, 11 passed, 3 failed。
- 三个失败测试为：
  - `test_merge_agent_beliefs_merges_shared_crg_into_final_state`
  - `test_merge_agent_beliefs_deduplicates_existing_beliefs`
  - `test_merge_agent_beliefs_falls_through_when_no_repository`

失败原因：

- 测试 patch 目标为：

```text
orchestrator_svc.main.build_shared_crg_repository_from_env
```

- 但实现中该 symbol 在 `_merge_agent_beliefs_into_crg()` 内部局部 import：

```python
from mf_core.db.repositories import build_shared_crg_repository_from_env
```

乙方推荐处理方式二选一：

- 修改测试 patch target，改为 patch `mf_core.db.repositories.build_shared_crg_repository_from_env`。
- 或在 `orchestrator_svc.main` 明确暴露 module-level import seam，然后让测试 patch 该 seam。

注意：

- 这是 Owner B W1 unit-test compatibility issue。
- 甲方没有改 Owner B 测试，也没有改 W1 业务实现。
- 乙方处理后应重新跑 `uv run pytest tests/unit/test_graph_repo.py -q`。

### 6.2 第二优先级：W5 benchmark gate

当前状态：

- W5 benchmark failure 不是甲方本轮代码回归。
- 失败来自 production-quality/data gate。
- GuacaMol / PMO 的失败样本来自 repeated local `CCO` baseline，无法代表 production-quality generated samples。
- 资源缺失导致 skip 或无法正式验收。

乙方需要准备或确认：

- `MOSES_REFERENCE_SMILES_PATH`
- `FRAGFM_MOSES_GENERATED_SMILES_PATH`
- `PMO_SCORE_TABLE_PATH`
- `CROSSDOCKED_BENCHMARK_JSONL`
- production-quality HFM/FragFM generated samples。

禁止事项：

- 不要降低 MOSES / GuacaMol / PMO threshold。
- 不要把 8/64/256 local FragFM sample smoke 当作 official W5 pass。
- 不要把 smoke-only HFM decoder artifact 当作 production-quality HFM artifact。

乙方建议命令：

```bash
uv run pytest tests/benchmark -q -ra
```

该命令只有在 official benchmark resources 和 production-quality generated samples 就位后才应作为正式验收依据。

### 6.3 第三优先级：W3 / mf-eval 下游确认

当前记录：

- W3 mf-eval local provider gate 曾记录 24 passed。

乙方需要确认：

- W3 provider/oracle evaluator 与下游 validation pipeline 的接口没有被 Owner A 生成侧改动破坏。
- 若乙方引入真实 oracle resources，应保留当前 provider fallback 逻辑，不要破坏本地 smoke path。

建议命令：

```bash
uv run pytest tests/unit/test_mf_eval.py -q
```

### 6.4 第四优先级：W12 scorer integration 复核

当前边界：

- W12 属乙方范围。
- 甲方没有修改 Owner B implementation files。

乙方需要确认：

- CReM-pharm-3D scorer integration 与 downstream validation/supply path 仍然可用。
- 如果接入真实 DiffDock-L / pharmacophore / HUMU scorer runner，应先做 command preflight，再跑 service smoke。

建议流程：

1. 先复核 W12 当前配置和 runner env。
2. 跑 W12 focused tests。
3. 再接真实 scorer runner。
4. 最后进入 benchmark 或 cluster validation。

## 7. 乙方对接执行顺序

建议乙方按以下顺序执行，不要直接跑全量大测试：

1. 阅读本报告，确认边界和禁止事项。
2. 复核 C1/C2/C3 三个契约面，确认没有字段或 predicate 静默变更。
3. 处理 W1 unit patch seam，并跑 `uv run pytest tests/unit/test_graph_repo.py -q`。
4. 复跑 W3 local provider gate：`uv run pytest tests/unit/test_mf_eval.py -q`。
5. 复核 W12 scorer integration focused gate。
6. 准备 W5 official benchmark resources，不改阈值。
7. 用 production-quality generated samples 跑 W5 benchmark。
8. 双方确认 Owner A local code-freeze 状态和 Owner B downstream 状态。
9. 获得明确授权后，再进入真实训练、cluster validation、artifact promotion。

## 8. 对接验收命令建议

### 8.1 乙方优先命令

```bash
uv run pytest tests/unit/test_graph_repo.py -q
```

```bash
uv run pytest tests/unit/test_mf_eval.py -q
```

```bash
uv run pytest tests/unit/test_validation_agent.py tests/unit/test_srb_agent.py -q
```

```bash
uv run pytest tests/unit/test_generator_coord_agent.py -q
```

### 8.2 甲乙共同契约回归

```bash
uv run pytest tests/unit/test_service_artifact_status.py -q
```

```bash
uv run pytest tests/unit/test_generators.py::TestHFM3DGenerator -q
```

### 8.3 W11 对接确认命令

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/unit/test_service_artifact_status.py::test_fragfm_service_rejects_invalid_intent_cone_as_invalid_argument \
  tests/unit/test_service_artifact_status.py::test_fragfm_service_builds_generator_with_trained_artifacts \
  tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_checkpoint \
  tests/unit/test_service_artifact_status.py::test_fragfm_runtime_rejects_configured_missing_rate_matrix -q
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_writes_smiles_and_report \
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_does_not_leave_smiles_when_report_write_fails \
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_blocked_output_parent_before_generation \
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_rejects_same_output_and_report_path -q
```

```bash
python -m py_compile \
  services/fragfm-generator-svc/src/fragfm_generator_svc/main.py \
  models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py \
  tests/unit/test_service_artifact_status.py \
  tests/unit/test_generators.py
```

### 8.4 FragFM deployment default scan

```bash
rg -n "FRAGFM_(VOCAB|CHECKPOINT|RATE_MATRIX)_PATH|checkpoints/fragfm_humu_5k" \
  infra/docker/docker-compose.dev.yml \
  infra/kubernetes/deployments/moleculeforge-services.yaml \
  infra/helm/moleculeforge/values.yaml \
  tests/unit/test_service_artifact_status.py
```

期望默认路径：

```text
checkpoints/fragfm_humu_5k/vocab.json
checkpoints/fragfm_humu_5k/best_model.pt
checkpoints/fragfm_humu_5k/rate_matrix.pt
```

### 8.5 进程安全扫描

```bash
ps -eo pid,etime,stat,cmd | \
  rg -n 'pytest|models/mf-generators/.*/train.py|fragfm_generator_svc|mf_generators.fragfm|hfm_3d/train|train_hciv|tar_proxyless' || true
```

如果看到实际训练进程，应先确认归属，不要擅自 kill 外部进程。

## 9. 当前 artifact 和数据状态

### 9.1 可作为本地工程证据的路径

- `checkpoints/fragfm_humu_5k/`
- `data/processing/generator_artifacts/fragfm_records_humu_labeled.jsonl`
- `data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.smi`
- `data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.report.json`
- `data/processing/generator_artifacts/fragfm_humu_5k_sample_64.smi`
- `data/processing/generator_artifacts/fragfm_humu_5k_sample_64.report.json`
- `data/processing/generator_artifacts/fragfm_humu_5k_sample_256.smi`
- `data/processing/generator_artifacts/fragfm_humu_5k_sample_256.report.json`

### 9.2 不可误用的路径

- `checkpoints/fragfm`：old protected smoke/runtime artifact，不是 HUMU-conditioned production evidence。
- `checkpoints/humu`：protected HUMU checkpoint，不允许覆盖。
- `checkpoints/hfm3d_4h200`：protected HFM smoke/full-flow artifact，不是 production-quality HFM evidence。
- `checkpoints/fragfm_humu_candidate_20260606_155439/`：aborted-run evidence only，不是 candidate。

## 10. 乙方最终输出要求

乙方完成对接后，请输出一份接收报告，至少包含：

- W1 unit patch seam 是否修复，命令结果和失败/通过数量。
- W3 provider/evaluator focused gate 结果。
- W5 benchmark resource 状态，哪些 env 已投放，哪些仍缺。
- W12 scorer integration 状态。
- C1/C2/C3 契约是否有变更；如有变更，列出字段、原因、双方确认记录。
- 是否有 Owner A 文件需要甲方回接处理。
- 是否有 production resource、cluster、artifact promotion 决策需要项目方授权。

## 11. 当前最终判断

从甲方角度：

- Owner A generation-upstream 代码主体已经完成到乙方可接手状态。
- W11 是当前最完整的 local engineering path，但仍不是 production acceptance。
- 现在应由乙方处理 W1/W3/W5/W12 和契约复核。
- 双方对接完成后，才进入正式 benchmark、真实训练、cluster validation、artifact promotion。

本报告即为甲方向乙方发出的当前对接依据。乙方应按本报告流程接手，不要把本地工程 smoke 或 strict-local candidate 误登记为最终生产交付。
