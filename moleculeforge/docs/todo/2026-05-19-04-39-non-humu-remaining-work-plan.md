# 非 HUMU 剩余工作实施方案

## 目标

继续完成 CoreArchitecture v2 对齐中除以下内容之外的未完成项：

- HUMU 新 joint 数据。
- HUMU 新 intent 数据。
- HUMU 真实模型 checkpoint。

本方案不伪造任何外部 runner、checkpoint、专利数据、benchmark 数据或 E2E 结果。缺少真实资源时，生产路径必须 fail-fast 并输出明确缺失项。

## 当前证据

- DKI 基础设施已按 `mf-dki-bare` 启动并通过 smoke，后端为 PostgreSQL、Neo4j、Qdrant、MinIO、Redis。
- MoleculeForge 已完成 Milvus/NATS 到 Qdrant/Redis 的替换，活动代码和配置扫描未发现 Milvus/NATS 残留。
- DKI 真实集成测试已通过，覆盖 PostgreSQL、Neo4j、Qdrant、MinIO、Redis。
- `docs/architecture/current-implementation-vs-corearchitecture-v2.md` 仍记录以下非 HUMU 缺口：
  - Provenance production store 尚缺真实 Neo4j/Postgres/MinIO 端到端写入验收。
  - Oracle L1-L4 wrapper 存在，但缺真实 runner 和生产运行证据。
  - Retrosyn runner 依赖外部 AiZynth/RSGPT/UAlign runner，未完成真实链路。
  - KRAS/Audit E2E 有环境标记和测试文件，但没有真实运行通过证据。
  - Benchmark 依赖真实 generator、数据集和服务栈，尚未形成可复现结果。

## 调用链路分析

### DKI production provenance path

目标链路：

```text
API / agent / pipeline event
  -> run_id / trace_id / artifact metadata
  -> provenance service production_real store
  -> Neo4j graph nodes / relationships
  -> Postgres audit events
  -> MinIO artifact object
  -> response returns persisted ids and hashes
```

当前断点：

```text
production store adapter exists
  -> DKI clients exist
  -> real DKI integration exists
  -> missing end-to-end provenance write verification against mf-dki-bare
```

### KRAS / Audit E2E path

目标链路：

```text
E2E test entry
  -> environment preflight
  -> orchestrator workflow
  -> provenance production write
  -> DKI readback
  -> audit completeness assertion
```

当前断点：

```text
E2E tests exist
  -> env flags control execution
  -> DKI now available
  -> external runner/model/resource envs still need exact validation
```


目标链路：

```text
```

当前断点：

```text
pipeline exists
  -> Qdrant path exists
```

### Oracle / retrosyn / generator resource path

目标链路：

```text
service or agent request
  -> manifest/env resource preflight
  -> real runner or real artifact load
  -> deterministic metadata capture
  -> result persisted to provenance
```

当前断点：

```text
wrappers exist
  -> fail-fast boundaries exist in several places
  -> no full resource inventory proving all non-HUMU runners/checkpoints are available
```

## 文件影响评估

### 主要修改文件

- `services/provenance-svc/src/provenance_svc/main.py`
  - 增加或修正 production store 对真实 DKI 的写入/readback 验证入口。
  - 风险：不能影响 `local_demo` in-memory 行为。

- `tests/integration/`
  - 增加或完善真实 Provenance -> Neo4j/Postgres/MinIO 集成测试。
  - 增加 DKI 环境 fixture 复用，避免每个测试重复构造连接信息。

- `tests/e2e/test_kras_g12c_pilot.py`
  - 增加严格 preflight，列出缺失 runner、model、data、index、DKI 配置。
  - 只在真实依赖齐全时执行 E2E 主流程。

- `tests/e2e/test_audit_completeness.py`
  - 接入真实 DKI provenance readback。
  - 缺非 HUMU 资源时 fail-fast 或明确 skip 原因，不把空流程标记为通过。

  - 只在发现真实输入源时执行生产 indexing。

- `tests/unit/test_indexing_pipelines.py`

- `models/artifacts/manifest.json`
  - 审计非 HUMU runner/artifact 声明。
  - 只允许反映真实存在的资源路径或环境变量契约。

### 关联影响文件

- `configs/services/*`
  - 如现有服务配置缺少非 HUMU runner 或 DKI resource key，需要按真实配置补齐。

- `configs/agents/*`
  - 如 agent 仍依赖旧资源命名，需要和服务配置同步。

- `infra/docker/*`
  - 如 E2E 需要容器化服务连接 `mf-dki-bare`，只做必要环境变量映射，不新增伪服务。

### 文档文件

- `README.md`
  - 按项目规则，实施完成后先向用户简述更新内容，待用户确认后再更新。

## 实施步骤

### 步骤 1：资源库存审计

目标：列出所有非 HUMU 真实资源是否存在，形成代码可执行的 preflight 输入。

执行内容：

- 读取 `models/artifacts/manifest.json`。
- 对每项资源执行存在性检查或服务连通性检查。
- 输出结构化缺失项，不补造默认值。

验收：

- 缺失资源会明确列出资源名、期望 env/path、调用入口。
- 已存在资源会给出真实路径或真实 endpoint。

### 步骤 2：Provenance production DKI 写入验收

目标：在 `mf-dki-bare` 真实 DKI 上验证 graph、event、object 三类数据都能持久化。

执行内容：

- 使用现有 production store adapter。
- 构造真实 `run_id`、`trace_id`、artifact metadata。
- 写入 Neo4j 节点/关系、Postgres audit event、MinIO object。
- 从三个后端 readback，断言数据一致。

验收：

- 新增或更新的集成测试在 `mf-dki-bare` 环境下通过。
- 测试输出不能依赖 in-memory fallback。
- 缺任一后端配置时测试明确跳过或 fail-fast，不能假通过。



执行内容：

- 如果资源存在，运行 indexing 小批量真实样本并写入 Qdrant。
- 如果资源不存在，保留 fail-fast，并在 E2E preflight 中列为阻塞项。

验收：


### 步骤 4：Oracle / retrosyn / generator preflight 严格化

目标：让非 HUMU runner 和 artifact 的缺失在入口处被明确报告。

执行内容：

- 对 ADMET、GNINA、DiffDock、Boltz、OpenFE、retrosyn、非 HUMU generator runner 做统一 preflight。
- 保持现有 wrapper contract，不修改业务逻辑规避缺资源错误。
- 将 preflight 接到 KRAS/Audit E2E 前置检查。

验收：

- 缺资源时，错误信息包含具体 env/path。
- 资源存在时，测试执行真实 runner 或真实 artifact load。

### 步骤 5：KRAS / Audit E2E 尝试解锁

目标：在 DKI 已真实可用的前提下，运行非 HUMU 剩余链路的 E2E。

执行内容：

- 使用 `mf-dki-bare` 的真实环境变量。
- 启用 `RUN_KRAS_G12C_E2E=1` 和 `RUN_AUDIT_E2E=1` 前先运行 preflight。
- 若 preflight 缺资源，记录阻塞项并停止，不降低断言。
- 若资源齐全，执行 E2E 并验证 run_id、trace_id、artifact hash、DKI readback。

验收：

- E2E 通过时必须有真实 DKI readback 和真实资源调用证据。
- E2E 不通过时必须保留原始错误，不改成空流程通过。

## KISS 四问

1. 这是现实问题还是想象问题？


2. 有没有更简单的做法？

   有。先做资源 preflight 和真实 DKI readback，不重构架构，不新建并行版本，不引入新的抽象层。

3. 会破坏什么？

   主要风险是把原先可本地运行的 demo 路径误改为强依赖生产资源。控制方式是保持 `local_demo` 路径不变，只在 production/E2E 路径严格要求真实资源。

4. 当前项目真的需要这个功能吗？

   需要。用户明确要求除 HUMU joint/intent 数据和真实模型 checkpoint 外继续完成未完成项；这些项是当前“全栈是否打通、是否真实资源”的直接缺口。

## 风险和处理

- 缺真实外部资源：不伪造，不生成占位文件，输出阻塞项。
- DKI 服务状态变化：实施前重新运行 `sudo -n ./status.sh` 和必要 smoke。
- 代理环境影响本地 Qdrant：涉及 `mf-dki-bare` 的初始化和 smoke 继续使用去代理环境。
- E2E 涉及多个服务：先做 preflight，再跑主流程，避免半路失败时难以定位。
- README 更新：实施完成后先汇报本次更新，等用户确认后再修改 README。

## 审批状态

等待用户确认后实施。确认后按上述步骤从资源库存审计开始执行，不处理 HUMU 新 joint/intent 数据和 HUMU 真实模型 checkpoint。
