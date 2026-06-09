# H5 Oracle Wrapper 方案

## 1. 目标

为 P2/H5 补齐四个真实 oracle runner command：

- `DOCK_ORACLE_COMMAND`
- `BOLTZ2_ORACLE_COMMAND`
- `FEP_ORACLE_COMMAND`
- `ADMET_ORACLE_COMMAND`

本方案只设计 wrapper，不修改四个 oracle service 的业务逻辑。审批通过后再编码。

## 2. 依据

### 2.1 源文档依据

- `docs/architecture/corearchitecture-v2-completion-tasksplit.md`：H5 要投放真实 runner command 值 `DOCK_ORACLE_COMMAND`、`BOLTZ2_ORACLE_COMMAND`、`FEP_ORACLE_COMMAND`、`ADMET_ORACLE_COMMAND` 及 OpenFE 可执行环境，并做 Boltz full inference smoke。
- `docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`：W3 依赖 H5，本地 oracle 可达后 PCBO reference provider 才能做生产验收。
- `docs/todo/owner-b-validation-downstream/2026-06-04-p2-missing-resource-acquisition-todo.md`：H5 当前缺四个 command env；底层 artifact/tool env 已有，但不能替代 command wrapper。

### 2.2 Web 检索依据

没有找到公开的 MoleculeForge 专用 H5 wrapper，也没有公开结果命中 `DOCK_ORACLE_COMMAND`、`BOLTZ2_ORACLE_COMMAND`、`FEP_ORACLE_COMMAND`、`ADMET_ORACLE_COMMAND` 这些项目内 env 名。

可用的一手底层工具入口是：

- GNINA 官方仓库提供命令行 docking 工具：https://github.com/gnina/gnina
- Boltz 官方仓库提供 `boltz predict`，并记录可通过 `boltz predict --help` 查看输入格式与选项：https://github.com/jwohlwend/boltz
- Boltz prediction 文档记录 `boltz predict` 使用输入路径和选项参数，并包含 affinity 相关参数：https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md
- OpenFE CLI 文档包含 `plan-rbfe-network`、`quickrun`、`gather` 命令：https://docs.openfree.energy/en/v0.11.0/reference/cli/index.html
- ADMET-AI 官方仓库记录命令行工具 `admet_predict` 与 Python API：https://github.com/swansonk14/admet_ai

结论：H5 需要项目内 wrapper 将 MoleculeForge JSON stdin/stdout contract 翻译到底层工具 CLI 或 Python API。

### 2.3 本地代码依据

当前 `.env` preflight：

- 已设置：`GNINA_BINARY`、`DIFFDOCK_MODEL_PATH`、`BOLTZ_BINARY`、`BOLTZ_MODEL_PATH`、`BOLTZ_INPUT_TEMPLATE_DIR`、`OPENFE_RUNNER_PATH`、`CHEMPROP_ADMET_ROOT`、`ADMET_MODEL_PATH`、`ADMET_SERVICE_URL`、`ADMET_TARGETS`。
- 缺失：`DOCK_ORACLE_COMMAND`、`BOLTZ2_ORACLE_COMMAND`、`FEP_ORACLE_COMMAND`、`ADMET_ORACLE_COMMAND`。

现有服务已经支持 command env：

- `services/dock-svc/src/dock_svc/main.py`：读取 `DOCK_ORACLE_COMMAND`，stdin JSON 包含 `engine`、`smiles`、可选 `protein_pdb`，stdout JSON 需要 docking 分数字段。
- `services/boltz2-svc/src/boltz2_svc/main.py`：读取 `BOLTZ2_ORACLE_COMMAND`，stdin JSON 包含 `protein_pdb_id`、`ligand_smiles`、`ensemble_size`，stdout JSON 需要 affinity rows。
- `services/fep-svc/src/fep_svc/main.py`：读取 `FEP_ORACLE_COMMAND`，stdin JSON 包含 `project_id`、`protein_pdb_id`、`reference_ligand_smiles`、`test_ligand_smiles`、`method`、`n_repeats`，stdout JSON 需要非空 `results`。
- `services/admet-svc/src/admet_svc/main.py`：读取 `ADMET_ORACLE_COMMAND`，stdin JSON 包含 `smiles`、`properties`、`return_uncertainty`，stdout JSON 需要 predictions/scores。

现有测试已经固定 contract：

- `tests/unit/test_service_artifact_status.py` 覆盖四个 command runner 的 stdin/stdout 行为和 missing executable preflight。

## 3. 现实问题判断

这是现实问题，不是想象问题。P2/H5 当前确实缺四个 command env；底层工具路径存在不等于 H5 完成，因为服务只通过 `*_ORACLE_COMMAND` 进入生产 command path。

## 4. 深度调用链路分析

### 4.1 W3/PCBO 路径

```text
ParetoBOService.optimize()
-> PCBOOptimizationScheduler.run()
-> pareto_bo.providers.LocalOracleEvaluator.__call__()
-> LocalOracleEvaluator._evaluate_via_grpc()
-> OracleService gRPC target
-> dock-svc / boltz2-svc / fep-svc / admet-svc
-> service reads *_ORACLE_COMMAND
-> subprocess.run(wrapper)
-> wrapper calls real tool
-> wrapper prints JSON
-> service maps JSON to OracleEvaluation
```

关键点：

- PCBO 不直接调用 GNINA/Boltz/OpenFE/ADMET。
- PCBO 通过 gRPC 调 oracle service。
- oracle service 通过 command env 调 wrapper。
- wrapper 是缺失点。

### 4.2 Dock 路径

```text
DockServicer.Dock()
-> _require_runtime(engine)
-> _run_dock_command(request, engine)
-> subprocess.run(shlex.split(DOCK_ORACLE_COMMAND), stdin=json)
-> wrapper calls GNINA or DiffDock
-> wrapper stdout JSON
-> DockOracleServicer.Evaluate()
-> OracleEvaluation(scores={"docking_score": value})
```

输入字段：

- `engine`: `gnina` 或 `diffdock`
- `smiles`: 单个 SMILES
- `protein_pdb`: 可选；由 request 提供时传入

输出字段：

- `engine`
- `score` 或 `docking_score`
- 可选 `scores`
- 可选 `uncertainties`
- 可选 `elapsed_ms`

### 4.3 Boltz2 路径

```text
Boltz2Servicer.PredictAffinity()
-> BoltzCommandRunner.predict_affinity()
-> subprocess.run(shlex.split(BOLTZ2_ORACLE_COMMAND), stdin=json)
-> wrapper writes Boltz input file
-> wrapper calls boltz predict
-> wrapper parses affinity output
-> stdout JSON {"affinities": [...]}
-> service maps rows to Boltz2BindingAffinity
```

输入字段：

- `protein_pdb_id`
- `ligand_smiles`
- `ensemble_size`

输出每行至少需要：

- `protein_pdb_id`
- `ligand_smiles`
- `delta_g_kcal_mol`

可选字段：

- `uncertainty`
- `ki_nm`
- `ensemble_size`
- `per_member_dg`

### 4.4 FEP 路径

```text
FEPServicer.RunFEP()
-> _run_fep_command(request)
-> subprocess.run(shlex.split(FEP_ORACLE_COMMAND), stdin=json)
-> wrapper prepares OpenFE planning/quickrun/gather workflow
-> wrapper parses gathered result
-> stdout JSON {"results": [...]}
-> service maps rows to FEPResult
```

输入字段：

- `project_id`
- `protein_pdb_id`
- `reference_ligand_smiles`
- `test_ligand_smiles`
- `method`
- `n_repeats`

输出每行字段：

- `ligand_a_smiles`
- `ligand_b_smiles`
- `ddg_kcal_mol`
- `ddg_uncertainty`
- `n_repeats`
- `method`
- `per_repeat_ddg`
- `converged`

### 4.5 ADMET 路径

```text
ADMETServicer.Predict()
-> ADMETCommandRunner.predict()
-> subprocess.run(shlex.split(ADMET_ORACLE_COMMAND), stdin=json)
-> wrapper calls ADMET-AI Python API or admet_predict CLI
-> wrapper stdout JSON {"results": [...]}
-> service maps predictions to OracleEvaluation
```

输入字段：

- `smiles`: SMILES list
- `properties`: requested property list
- `return_uncertainty`: boolean

输出每行字段：

- `smiles`
- `predictions` or `scores`
- optional `uncertainties`
- optional `elapsed_ms`

## 5. 方案设计

### 5.1 推荐方案

新增四个独立 wrapper 文件，放在 `tools/oracles/`：

- `tools/oracles/dock_oracle_wrapper.py`
- `tools/oracles/boltz2_oracle_wrapper.py`
- `tools/oracles/fep_oracle_wrapper.py`
- `tools/oracles/admet_oracle_wrapper.py`

理由：

- 每个 wrapper 对应一个外部工具域，职责单一。
- 不改 service 已有 command contract。
- 不新增并行 service 版本。
- `.env` 可以直接指向 wrapper。
- 单测可单独覆盖每个 wrapper 的 JSON contract。

### 5.2 不采用的方案

不直接修改 `services/*-svc/src/*/main.py`：

- 这些服务已经有 command env、preflight、subprocess 调用和 JSON 解析。
- 修改服务会扩大影响面，并可能破坏现有测试。

不把 wrapper 放进 `scripts/`：

- `scripts/` 当前只有 `mine_mmp.py`，偏一次性脚本。
- `tools/` 已用于项目工具，`tools/oracles/` 更适合作为生产 command wrapper 集合。

不使用 mock 分数：

- H5 是生产 oracle acceptance gate，mock/fallback 不能算完成。

## 6. 文件影响评估

### 6.1 新增文件

- `tools/oracles/dock_oracle_wrapper.py`
  - 读取 MoleculeForge docking JSON。
  - 调 GNINA 或 DiffDock。
  - 输出 docking JSON。

- `tools/oracles/boltz2_oracle_wrapper.py`
  - 读取 MoleculeForge Boltz2 JSON。
  - 生成 Boltz input。
  - 调 `BOLTZ_BINARY` / `boltz predict`。
  - 输出 affinity JSON。

- `tools/oracles/fep_oracle_wrapper.py`
  - 读取 MoleculeForge FEP JSON。
  - 调 OpenFE CLI。
  - 输出 RBFE JSON。

- `tools/oracles/admet_oracle_wrapper.py`
  - 读取 MoleculeForge ADMET JSON。
  - 优先调用 ADMET-AI Python API；如果项目要求使用外部 HTTP service，则按 `ADMET_SERVICE_URL` 调本地服务。
  - 输出 ADMET JSON。

### 6.2 新增测试文件

- `tests/unit/test_h5_oracle_wrappers.py`
  - 测 wrapper stdin/stdout contract。
  - 通过 monkeypatch/subprocess fake 模拟底层工具输出。
  - 不跑真实 GNINA、Boltz、OpenFE、ADMET heavy inference。

### 6.3 可能修改文件

- `.env`
  - 审批后由用户或我们写入四个 command env。
  - 不提交密钥；这些 command env 不含密钥，但仍属于环境配置。

- `docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
  - 实现和验收后追加执行日志。

- `docs/architecture/corearchitecture-v2-completion-tasksplit.md`
  - 实现和验收后追加执行日志。

### 6.4 不修改文件

- `services/dock-svc/src/dock_svc/main.py`
- `services/boltz2-svc/src/boltz2_svc/main.py`
- `services/fep-svc/src/fep_svc/main.py`
- `services/admet-svc/src/admet_svc/main.py`

除非实现时发现现有 contract 与测试矛盾，否则不改服务。

## 7. Wrapper Contract

### 7.1 Dock Wrapper

stdin:

```json
{
  "engine": "gnina",
  "smiles": "CCO",
  "protein_pdb": "optional-pdb-text-or-path"
}
```

stdout:

```json
{
  "engine": "gnina",
  "scores": {
    "docking_score": -8.5
  },
  "uncertainties": {
    "docking_score": 0.0
  },
  "elapsed_ms": 1234
}
```

### 7.2 Boltz2 Wrapper

stdin:

```json
{
  "protein_pdb_id": "6OIM",
  "ligand_smiles": ["CCO"],
  "ensemble_size": 5
}
```

stdout:

```json
{
  "affinities": [
    {
      "protein_pdb_id": "6OIM",
      "ligand_smiles": "CCO",
      "delta_g_kcal_mol": -8.2,
      "uncertainty": 0.2,
      "ki_nm": 12.0,
      "ensemble_size": 5,
      "per_member_dg": [-8.0, -8.4]
    }
  ]
}
```

### 7.3 FEP Wrapper

stdin:

```json
{
  "project_id": "project-1",
  "protein_pdb_id": "6OIM",
  "reference_ligand_smiles": "CCO",
  "test_ligand_smiles": ["CCN"],
  "method": "openfe",
  "n_repeats": 1
}
```

stdout:

```json
{
  "batch_id": "project-1",
  "total_elapsed_ms": 1000,
  "results": [
    {
      "ligand_a_smiles": "CCO",
      "ligand_b_smiles": "CCN",
      "ddg_kcal_mol": -1.2,
      "ddg_uncertainty": 0.3,
      "n_repeats": 1,
      "method": "openfe",
      "per_repeat_ddg": {
        "repeat_1": -1.2
      },
      "converged": true
    }
  ]
}
```

### 7.4 ADMET Wrapper

stdin:

```json
{
  "smiles": ["CCO"],
  "properties": ["clearance", "herg"],
  "return_uncertainty": false
}
```

stdout:

```json
{
  "results": [
    {
      "smiles": "CCO",
      "predictions": {
        "clearance": 1.5,
        "herg": 0.2
      },
      "uncertainties": {
        "clearance": 0.0,
        "herg": 0.0
      },
      "elapsed_ms": 100
    }
  ]
}
```

## 8. 实施步骤

### Task 1: Dock Wrapper

- 新增 `tools/oracles/dock_oracle_wrapper.py`。
- 新增 `tests/unit/test_h5_oracle_wrappers.py` 中 dock wrapper 单测。
- 测试点：
  - stdin JSON 解析。
  - GNINA command argv 构造。
  - 底层命令失败时 stderr 原样进入 RuntimeError。
  - stdout JSON 包含 `scores.docking_score`。
- 验证命令：

```bash
uv run pytest tests/unit/test_h5_oracle_wrappers.py -q
```

### Task 2: Boltz2 Wrapper

- 新增 `tools/oracles/boltz2_oracle_wrapper.py`。
- 扩展 `tests/unit/test_h5_oracle_wrappers.py`。
- 测试点：
  - 为每个 ligand 生成 Boltz input。
  - 调用 `BOLTZ_BINARY`。
  - 解析 affinity JSON。
  - 输出 `affinities` rows。
- 验证命令：

```bash
uv run pytest tests/unit/test_h5_oracle_wrappers.py -q
```

### Task 3: FEP Wrapper

- 新增 `tools/oracles/fep_oracle_wrapper.py`。
- 扩展 `tests/unit/test_h5_oracle_wrappers.py`。
- 测试点：
  - 调用 `OPENFE_RUNNER_PATH` 或 `openfe`。
  - 处理 reference ligand 和 test ligand list。
  - 输出非空 `results`。
  - 对 OpenFE 失败返回明确错误。
- 验证命令：

```bash
uv run pytest tests/unit/test_h5_oracle_wrappers.py -q
```

### Task 4: ADMET Wrapper

- 新增 `tools/oracles/admet_oracle_wrapper.py`。
- 扩展 `tests/unit/test_h5_oracle_wrappers.py`。
- 测试点：
  - 通过 ADMET-AI Python API 或已配置 HTTP endpoint 获取 predictions。
  - 若 requested properties 为空，使用 `ADMET_TARGETS`。
  - 输出 `results` list。
  - 对无 numeric predictions 的结果 fail-fast。
- 验证命令：

```bash
uv run pytest tests/unit/test_h5_oracle_wrappers.py -q
```

### Task 5: Command Env Preflight

- 临时在当前 shell 设置四个 command env，不直接写 `.env`：

```bash
export DOCK_ORACLE_COMMAND="uv run python tools/oracles/dock_oracle_wrapper.py"
export BOLTZ2_ORACLE_COMMAND="uv run python tools/oracles/boltz2_oracle_wrapper.py"
export FEP_ORACLE_COMMAND="uv run python tools/oracles/fep_oracle_wrapper.py"
export ADMET_ORACLE_COMMAND="uv run python tools/oracles/admet_oracle_wrapper.py"
```

- 跑现有 service command contract 测试：

```bash
uv run pytest tests/unit/test_service_artifact_status.py -q
```

### Task 6: H5 Smoke

H5 smoke 分两层：

- 轻量 smoke：wrapper contract 单测通过。
- 真实工具 smoke：使用真实 GNINA/Boltz/OpenFE/ADMET 资源跑小样本。

真实工具 smoke 只能在资源确认后执行。Boltz/OpenFE 可能较慢，执行前需要用户明确授权。

## 9. 测试策略

### 9.1 不需要真实工具的测试

```bash
uv run pytest tests/unit/test_h5_oracle_wrappers.py -q
uv run pytest tests/unit/test_service_artifact_status.py -q
```

这些测试证明 wrapper contract 和 service preflight 正确。

### 9.2 需要真实资源的 smoke

审批后再确定具体命令。候选 smoke：

```bash
bash -lc 'set -a; source .env; set +a; export DOCK_ORACLE_COMMAND="uv run python tools/oracles/dock_oracle_wrapper.py"; uv run python tools/oracles/dock_oracle_wrapper.py'
```

该命令不能直接执行，因为 wrapper 需要 stdin JSON。真实 smoke 应由实施后提供完整 stdin payload，并记录 exit code 和 stdout JSON。

### 9.3 H5/W3 验收

源文档 W3 focused command：

```bash
uv run pytest tests/unit/test_mf_eval.py -q
```

H5 生产验收还需证明 PCBO provider/evaluator 能走真实 L0-L3 oracle path，而不是只走 `embedding_proxy`。该验收必须在四个 command env 实际配置后执行。

## 10. KISS 四问

1. 这是现实问题还是想象问题？
   - 现实问题。H5 明确缺四个 command env，源文档和 `.env` preflight 都能证明。

2. 有没有更简单的做法？
   - 有。最简单做法是只新增 wrapper，不改 service，不改 PCBO，不改 protobuf。

3. 会破坏什么？
   - 主要风险是 wrapper 输出 schema 与 service contract 不一致。用现有 tests 中的 contract 固定输出即可控制。

4. 当前项目真的需要这个功能吗？
   - 需要。H5 是 P2 剩余 gate，也是 W3 和 H11 的前置。

## 11. 性能影响

- 单元测试不会跑真实 heavy inference，性能影响低。
- 真实 Boltz/OpenFE smoke 会较慢，必须单独授权。
- wrapper 会使用临时工作目录，避免不同请求覆盖输出。
- Boltz batch 可以按 `ligand_smiles` list 处理，避免每个 ligand 重启过多流程；具体批处理深度以官方 CLI 和现有 artifact 能力为准。

## 12. 风险与缓解

- 风险：GNINA/DiffDock 需要 protein structure，但当前 dock service 只保证 `protein_pdb` 可选。
  - 缓解：wrapper 对缺 protein input fail-fast，不生成伪 docking score。

- 风险：Boltz input template 对 protein id 和 ligand SMILES 的映射要求不清。
  - 缓解：复用 `services/boltz2-svc/src/boltz2_svc/main.py` 中 `BoltzCliRunner` 的输入处理思路；必要时在 wrapper 中加载 `BOLTZ_INPUT_TEMPLATE_DIR`。

- 风险：OpenFE 真实 RBFE 需要 ligand/protein 准备和较长计算。
  - 缓解：FEP wrapper 先实现 contract 和 fail-fast；真实 smoke 只在 OpenFE 环境确认后运行。

- 风险：ADMET-AI 与当前 `CHEMPROP_ADMET_ROOT` 服务形态不一致。
  - 缓解：优先使用当前项目已有 `ADMET_SERVICE_URL` HTTP path；若服务不可达，再使用 ADMET-AI Python API，并在错误里明确依赖缺口。

- 风险：把 demo/fallback 当生产结果。
  - 缓解：wrapper 禁止随机、固定值、hash proxy；没有真实工具输出就 exit non-zero。

## 13. 需用户审批的范围

审批后允许新增：

- `tools/oracles/dock_oracle_wrapper.py`
- `tools/oracles/boltz2_oracle_wrapper.py`
- `tools/oracles/fep_oracle_wrapper.py`
- `tools/oracles/admet_oracle_wrapper.py`
- `tests/unit/test_h5_oracle_wrappers.py`

审批后允许追加执行日志：

- `docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
- `docs/architecture/corearchitecture-v2-completion-tasksplit.md`

审批后不允许改动：

- C1/C2/C3 schema。
- `services/*-svc/src/*/main.py`，除非实现中发现现有代码与现有测试矛盾，并重新申请 scope。
- 任何训练数据、模型权重或密钥。

## 14. 执行日志模板

```text
日期/角色: 2026-06-04（乙）
ID: H5 oracle wrapper implementation
改动文件: tools/oracles/*.py, tests/unit/test_h5_oracle_wrappers.py
验证命令: 实际执行的完整命令
实际结果: exit code、pass/fail/skip 数、关键 warning
剩余 gate: 真实工具 smoke / W3 production oracle acceptance / 无
契约变更: C1/C2/C3 无变更
```
