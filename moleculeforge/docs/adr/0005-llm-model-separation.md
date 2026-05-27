# ADR-0005: Orchestrator 与 Critic LLM 模型族分离（防合谋设计）

**状态**：已采纳
**日期**：2025-Q2
**决策者**：MoleculeForge 架构团队

## 背景

MoleculeForge 的 MARB（Multi-Agent Reflective Board）中有两个关键 LLM 驱动的 Agent：

- **Orchestrator**（Agent-0）：主管，负责计划-执行决策、流程编排
- **Critic**（Agent-7）：科学质疑者，负责审核 Orchestrator 的决策和生成结果

如果两者使用同一 LLM 模型族，存在**认知偏差放大风险**：同一模型的"盲点"会在 Orchestrator 和 Critic 之间共振，导致 Critic 无法有效质疑 Orchestrator 的决策（架构 §4.2 防合谋设计）。

## 决策

**Orchestrator 和 Critic 必须使用不同的 LLM 模型族**。

推荐配置：
- Orchestrator：**Claude Sonnet 4.5**（Anthropic）— 推理深度 + 长上下文
- Critic：**DeepSeek-V3**（DeepSeek）或 **Gemini 2.5 Pro** — 不同的训练数据和架构

## 理由

1. **防合谋**：不同模型族的训练数据、RLHF 策略、架构差异 → Critic 更可能发现 Orchestrator 的盲点
2. **多样性**：两个模型对同一化学假设可能给出不同评估 → 更稳健的决策
3. **审计可追溯**：每条 CRG belief 记录来源模型，便于事后分析模型偏差
4. **成本优化**：Critic 调用频次低（仅在 batch/pareto_change/outlier 触发时），可以用高成本模型

## 后果

- `configs/agents/orchestrator.yaml` 配置 `llm: claude-sonnet-4-5`
- `configs/agents/critic.yaml` 配置 `llm: deepseek-v3`
- CI 测试 `test_critic_uses_different_llm_from_orchestrator` 强制检查两者 LLM 不同
- 同一 LLM 实例 → 🔴 BLOCKER（PR 不可合并）

## 验证

- 读 `configs/agents/orchestrator.yaml` 和 `configs/agents/critic.yaml`
- 断言 `orchestrator.llm.model_family != critic.llm.model_family`
