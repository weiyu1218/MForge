# ADR-0003: Redis 作为 Agent 间消息总线

**状态**：已采纳
**日期**：2025-Q1
**决策者**：MoleculeForge 架构团队

## 背景

8 个 Agent（orchestrator/nl2obj/generator_coord/retrosyn_agent/validation_agent/fto_agent/supply_agent/critic_agent）需要异步通信和事件驱动协同。当前 DKI 标准由 `mf-dki-bare` 提供，基础设施包含 PostgreSQL、Neo4j、Qdrant、MinIO、Redis。

备选方案：
- **gRPC 直连**：同步调用，紧耦合。
- **Redis Pub/Sub**：与 DKI 标准一致，满足当前 publish/subscribe/request 基础语义。
- **持久化消息队列**：提供更强投递语义，但当前代码没有使用持久化消费组能力。

## 决策

**选择 Redis Pub/Sub** 作为 Agent 间消息总线。

## 理由

1. **基础设施一致**：与 `mf-dki-bare` 的 Redis 服务一致，不再维护额外消息中间件。
2. **当前语义匹配**：代码只使用 publish、subscribe、request 基础语义，没有持久化队列调用点。
3. **审计职责清晰**：可审计事实写入 Neo4j/Postgres/MinIO，消息总线不承担长期事实存储。

## 后果

- Agent 不直接调用彼此，通过 Redis subject 通信。
- Redis Pub/Sub 不提供持久化投递保证，不能把消息总线作为审计来源。
- `mf_agents.messaging.redis_bus.RedisBus` 提供 Redis 连接和进程内 fallback；integration test 禁止 fallback。
- 主题命名遵循 `{domain}.{action}.{state}` 规范（如 `generation.request.hit_finding`）。

## 验证

- Agent 通信图中无直接 Python import。
- RedisBus 单元测试覆盖 trace_id 传播和 agent callback 签名。
- DKI Redis integration test 在配置真实 Redis 后验证 publish/subscribe。
