# H6 Retrosynthesis AiZynth Runner Wrapper 方案

## 目标

为 H6 补齐最小可验收的 retrosynthesis production runner command：

- 先接入 `aizynth` 单引擎。
- 通过 `RETROSYN_PLANNER_COMMANDS_JSON` 投放给 `retrosyn-svc`。
- 不修改 `services/retrosyn-svc/src/retrosyn_svc/main.py` 的服务合同。

## 调用链路

```text
RetrosynServicer.FindRoutes()
-> RetrosynServicer._find_routes(engine="ensemble" 或 "aizynth")
-> ExternalCommandRoutePlanner.find_routes()
-> _run_planner_command()
-> subprocess.run(shlex.split(command), stdin=json)
-> tools/retrosyn/aizynth_planner_wrapper.py
-> AiZynthRetrosyn.from_env().find_routes(smiles, max_routes)
-> stdout {"routes": [...]}
```

## 输入输出合同

Wrapper stdin:

```json
{
  "smiles": "CCO",
  "max_routes": 1,
  "engine": "aizynth"
}
```

Wrapper stdout:

```json
{
  "routes": [
    {
      "route_id": "route-1",
      "score": 0.8,
      "steps": [
        {"reaction": "CCO>>CC=O"}
      ]
    }
  ],
  "total_routes_found": 1,
  "elapsed_ms": 10
}
```

## 修改范围

- 新增 `tools/retrosyn/aizynth_planner_wrapper.py`
- 新增 `tests/unit/test_h6_retrosyn_wrapper.py`
- 更新 `.env` 的 `RETROSYN_PLANNER_COMMANDS_JSON`

不修改：

- `services/retrosyn-svc/src/retrosyn_svc/main.py`
- protobuf schema
- 两份架构执行日志

## 验证

1. TDD RED：先新增 wrapper focused 测试，确认因 wrapper 缺失失败。
2. GREEN：实现 wrapper 后复跑 focused 测试。
3. 配置生效：`source .env` 后检查 `RETROSYN_PLANNER_COMMANDS_JSON` set，`retrosyn-svc.runtime_status()` 能看到 planner command available。
4. 回归：跑 retrosyn command 相关 focused pytest。
5. 真实 smoke：用 `.env` 当前 AiZynth 配置对 `CCO` 跑最小 route 规划。若真实 planner 输出空 route 或外部模型失败，按原始错误报告，不登记 H6 完成。

## KISS 检查

- 真问题：H6 缺 production runner command，H11 full pilot 依赖它。
- 更简单做法：优先单引擎 `aizynth`，不引入多引擎抽象。
- 会破坏什么：不改 service main 和 schema，保持现有命令合同。
- 当前需要：H6 是文档列出的 C 类 runner gate。
