# ADR-0002: 三层依赖纪律

**状态**：已采纳
**日期**：2025-Q1
**决策者**：MoleculeForge 架构团队

## 背景

MoleculeForge 代码库包含多层组件：共享库（libs）、模型（models）、服务（services）、智能体（agents）。需要防止循环依赖和架构腐化。

## 决策

采用严格的**三层依赖纪律**，由 `import-linter` 在 CI 中强制检查：

```
Layer 3: agents/         ← 最上层，编排/决策
Layer 2: services/       ← 中间层，执行
Layer 1: models/         ← 模型层，推理
Layer 0: libs/           ← 共享内核，无上层依赖
```

规则：
1. **libs/ 不得 import services/agents/models**（共享内核独立）
2. **models/ 不得 import services/agents**（模型不依赖服务）
3. **services/ 之间不得直接 import**（通过 client SDK 通信）
4. **agents/ 可以 import 所有下层**

## 理由

1. **可测试性**：每个 lib 可以独立测试，不依赖重服务
2. **部署灵活性**：模型可以在不同环境中独立部署（GPU 节点 vs CPU 节点）
3. **防循环**：严格的单向依赖图消除了循环依赖的可能性
4. **CI 可执行**：通过 `uv run import-linter` 自动检查

## 后果

- `pyproject.toml` 中配置 `[tool.importlinter]` 三个合约
- CI 流水线 `ci-lint.yml` 中 `uv run lint-imports` 失败 → 不允许合并
- 服务间通信只能通过 gRPC/Redis message bus，不能用 Python import
- 新模块必须先确定层级再实现

## 验证

- `uv run import-linter` 全通过
- 手工扫描 `grep -rE "^from services" libs/` 返回空
