# CoreArchitecture v2 阶段 A/B 完成情况

## 结论

阶段 A 和阶段 B 已按对应执行文档完成验收。

- 阶段 A 执行文档：`docs/todo/2026-05-19-08-26-corearchitecture-v2-phase-a-execution-plan.md`
- 阶段 B 执行文档：`docs/todo/2026-05-19-08-33-corearchitecture-v2-phase-b-execution-plan.md`
- 验收日期：2026-05-19

## 阶段 A 完成情况

阶段 A 的工程债务清扫项已完成：

- SSP compiler 测试与当前 `compile_ssp()` contract 对齐。
- Benchmark 文件已进入 pytest 收集，缺真实资源时按 benchmark 自身条件 skip。
- CRG schema 与 proto/Pydantic 字段统一。
- K8s network policy、Docker compose、Helm 服务编排已同步当前服务端口和 Qdrant/Redis DKI 路线。
- GNINA checksum 占位已移除。
- 生产路径中的吞错、mock/dummy 命名、无效 SQLAlchemy repository 路径已清理。
- NL parser、CIG stage1、nl2obj service 已统一到同一 parser 路径。
- TAR router 和 generator coordinator 已统一到真实 generator 名称。
- gRPC proto 生成与服务注册链路已可校验。

## 阶段 B 完成情况

阶段 B 的算法补全项已完成：

- iCLM / UAS / CReM / FragFM 已接入仓内算法模块或真实 runner contract；缺 runner、checkpoint、candidate source 或 decoder 时 fail-fast。
- `SRBAgent.process` 已接入 `compile_ssp()`，不再返回空 protocol 占位。
- `ValidationAgent` L1-L3 已使用 `predict_with_uncertainty` 和不确定度阈值。
- Patent Dead Zone 已改为 Lorentz 距离与批量矩阵化实现。
- HUMU 已加入可学习曲率封装，并保持原有 float curvature API 兼容。
- PAINS 过滤已接入 L0 oracle。
- `mf-eval` 已补齐 `distortion`、`cliff_analysis`、`hv_evaluator`。

## 验收命令结果

| 命令 | 结果 |
|---|---|
| `uv run pytest tests/unit -q` | 通过，退出码 0 |
| `uv run pytest tests/integration -q` | 通过，退出码 0 |
| `uv run pytest tests/anti_degradation/test_no_degradation.py tests/benchmark -q` | 反退化测试通过；benchmark 按真实资源条件 skip，退出码 0 |
| `bash tools/codegen/check_proto_sync.sh` | 通过，退出码 0 |
| `docker compose -f infra/docker/docker-compose.test.yml config` | 通过，退出码 0 |
| `docker compose -f infra/docker/docker-compose.dev.yml config` | 通过，退出码 0 |
| `.venv/bin/helm template moleculeforge infra/helm/moleculeforge` | 通过，退出码 0 |

## 当前非阻塞告警

以下告警未阻塞阶段 A/B 验收：

- Pydantic V2 class-based `config` deprecation。
- `torch.load(weights_only=False)` future warning。
- LangGraph pending deprecation warning。
- Qdrant client/server minor version compatibility warning。

## 边界

- 阶段 A/B 不包含真实训练 checkpoint 补齐。
- 阶段 A/B 不包含 KRAS pilot、audit pilot、阶段 C-E 的真实外部模型和商业合规闭环。
- Benchmark skip 的原因是缺真实训练资源或数据集，不计为伪通过。
