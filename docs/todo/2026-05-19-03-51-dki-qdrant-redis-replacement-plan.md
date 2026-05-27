# DKI Qdrant Redis 替换方案

## 目标

按用户确认的 A 方案，将 MoleculeForge 生产 DKI 标准从 `Milvus + NATS` 正式替换为 `mf-dki-bare` 提供的 `Qdrant + Redis`，并继续使用 `PostgreSQL + Neo4j + MinIO`。本方案不处理 HUMU 预训练数据，不停止当前 HUMU baseline 进程。

## 当前证据

- `/workspace/mf-dki-bare/README-bare.md` 明确该裸金属 DKI 由 PostgreSQL 16、Neo4j 5、Qdrant 1.12、MinIO、Redis 7 组成。
- `/workspace/mf-dki-bare/README-bare.md` 明确“Milvus -> Qdrant”，并声明 Qdrant 端口为 HTTP `16333`、gRPC `16334`。
- `/workspace/mf-dki-bare/status.sh` 当前输出 `supervisord not running`，端口 `15432`、`17687`、`16333`、`19000`、`16379` 全部 FAIL，因此当前不能声称 DKI 真实后端已经可用。
- MoleculeForge 当前仍存在 Milvus 路径：
  - `libs/mf-core/src/mf_core/db/milvus_client.py`
  - `services/humu-index-svc/src/humu_index_svc/main.py`
  - `tests/integration/test_dki_milvus.py`
  - `tests/unit/test_vector_store.py`
  - `models/artifacts/manifest.json` 中的 `MILVUS_URI`
  - `pipelines/patent_indexing/src/patent_indexing/pipeline.py` 中的 `milvus_client`、`milvus_collection`
- MoleculeForge 当前仍存在 NATS 路径：
  - `libs/mf-agents/src/mf_agents/messaging/nats_bus.py`
  - `libs/mf-agents/src/mf_agents/base/agent.py` 中 `nats_client` 命名和注释
  - `tests/unit/test_nats_bus.py`
  - 多个 agent 构造函数使用 `nats_client`
- `mf-dki-bare` 没有 NATS 服务；它提供 Redis，端口 `16379`。
- `mf-dki-bare/dki_client/vector_store.py` 已有 Qdrant 后端实现，集合名包括 `molecules_humu`、`pockets_humu`、`patents_embedding`，并提供 upsert/search/delete/count 语义。

## 调用链路分析

### HUMU index vector path

```text
humu-index-svc REST
  -> _require_milvus_config()
  -> MILVUS_URI
  -> _milvus_client()
  -> mf_core.db.milvus_client.MilvusCollectionClient
  -> pymilvus Collection
  -> insert/search/delete/stats
```

替换后目标链路：

```text
humu-index-svc REST
  -> _require_qdrant_config()
  -> QDRANT_URL 或 QDRANT_HOST/QDRANT_HTTP_PORT
  -> _vector_client()
  -> mf_core.db.qdrant_client.QdrantCollectionClient
  -> qdrant-client
  -> upsert/search/delete/stats
```

### Patent indexing path

```text
patent_indexing.run()
  -> index_surechembl_to_milvus()
  -> _required_index_client(cfg["milvus_client"])
  -> insert(collection, records) 或 upsert(data)
  -> search_patent_similarity()
  -> _required_search_client(cfg["milvus_client"])
```

替换后目标链路：

```text
patent_indexing.run()
  -> index_surechembl_to_vector_store()
  -> _required_index_client(cfg["vector_client"])
  -> insert(collection, records) 或 upsert(data)
  -> search_patent_similarity()
  -> _required_search_client(cfg["vector_client"])
```

默认 collection 从 `patent_molecules` 改为 `patents_embedding`，对齐 `mf-dki-bare`。

### Agent messaging path

```text
Agent
  -> BaseAgent(nats_client)
  -> NATSBus.connect()
  -> nats.connect()
  -> publish/subscribe/request
  -> fallback in-process pub/sub
```

替换后目标链路：

```text
Agent
  -> BaseAgent(message_bus)
  -> RedisBus.connect()
  -> redis.asyncio.Redis
  -> publish/subscribe/request
  -> fallback in-process pub/sub
```

Redis pub/sub 不天然提供 NATS request-reply inbox 语义，因此 request 会用独立 reply channel 和 timeout 实现。若没有 Redis 连接，继续使用 in-process fallback，保证单元测试和本地开发可运行。

### DKI integration validation path

当前：

```text
tests/integration/test_dki_postgres.py -> TEST_DATABASE_URL
tests/integration/test_dki_neo4j.py -> NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD
tests/integration/test_dki_milvus.py -> MILVUS_HOST/MILVUS_PORT
MinIO 只在单元 fake client 中验证
NATS 只在 fallback 单元测试中验证
```

替换后：

```text
tests/integration/test_dki_postgres.py -> TEST_DATABASE_URL
tests/integration/test_dki_neo4j.py -> NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD
tests/integration/test_dki_qdrant.py -> QDRANT_HOST/QDRANT_HTTP_PORT
tests/integration/test_dki_minio.py -> MINIO_ENDPOINT_URL/MINIO_ACCESS_KEY/MINIO_SECRET_KEY/MINIO_BUCKET
tests/integration/test_dki_redis.py -> REDIS_HOST/REDIS_PORT/REDIS_PASSWORD
```

## 文件影响评估

### 主要修改文件

- `libs/mf-core/src/mf_core/db/milvus_client.py`
  - 重命名为 `qdrant_client.py`，实现 Qdrant collection client。
  - 风险：所有导入路径需要同步更新。
- `libs/mf-core/pyproject.toml`
  - `db` extra 从 `pymilvus` 改为 `qdrant-client`。
  - 风险：依赖锁文件需要同步。
- `services/humu-index-svc/src/humu_index_svc/main.py`
  - backend、配置、错误信息、客户端构造改为 Qdrant。
  - 风险：接口响应字段变化影响测试和调用方。
- `services/humu-index-svc/pyproject.toml`
  - `pymilvus` 改为 `qdrant-client`。
- `pipelines/patent_indexing/src/patent_indexing/pipeline.py`
  - 函数和配置命名从 Milvus 改为 neutral vector/Qdrant。
  - 风险：旧配置键会失效；方案选择 A 表示接受架构替换，但仍应使用 `vector_client` 作为稳定抽象，避免未来再改后端时重复改业务名。
- `libs/mf-agents/src/mf_agents/messaging/nats_bus.py`
  - 替换为 Redis bus 实现，或重命名为 `redis_bus.py` 并更新导入。
  - 推荐重命名，避免代码继续声称 NATS。
- `libs/mf-agents/src/mf_agents/base/agent.py`
  - `nats_client` 改为 `message_bus`，注释改为 Redis-backed message bus。
- `libs/mf-agents/pyproject.toml`
  - 移除 `nats-py`，增加 `redis>=5.0`。
- `agents/orchestrator/src/orchestrator/agent.py`
  - 移除未使用的 `NATSBus` 导入或改为 `RedisBus`。
- `models/artifacts/manifest.json`
  - `milvus_uri/MILVUS_URI` 改为 `qdrant_endpoint/QDRANT_URL` 或 `QDRANT_HOST/QDRANT_HTTP_PORT`。

### 测试文件

- `tests/unit/test_vector_store.py`
  - 从 Milvus mock 测试改为 Qdrant mock 测试。
- `tests/unit/test_service_artifact_status.py`
  - HUMU index service 测试改为 Qdrant client 和 QDRANT env。
- `tests/unit/test_indexing_pipelines.py`
  - 函数名、配置名和默认 collection 改为 vector/Qdrant。
- `tests/unit/test_nats_bus.py`
  - 改为 `test_redis_bus.py`，覆盖 fallback 和 trace_id。
- `tests/integration/test_dki_milvus.py`
  - 改为 `test_dki_qdrant.py`，连接 Qdrant 真实服务。
- 新增 `tests/integration/test_dki_minio.py`。
- 新增 `tests/integration/test_dki_redis.py`。

### 配置和文档文件

- `configs/services/*`
  - 按需新增或更新 Qdrant/Redis 配置。
- `infra/docker/docker-compose.test.yml`
  - 如果继续保留 Docker test stack，应同步 Milvus/NATS -> Qdrant/Redis；若仅使用 `mf-dki-bare`，则标注该 compose 不再是 DKI 标准验收入口。
- `docs/architecture/current-implementation-vs-corearchitecture-v2.md`
  - 更新 DKI 目标描述为 `mf-dki-bare` 路线。
- `docs/todo/2026-05-19-02-51-corearchitecture-v2-alignment-plan.md`
  - 更新未完成 DKI 项中的 Milvus/NATS 文字。
- `README.md`
  - 按项目规则，开发完成后先简述更新内容，待用户确认后再更新。

## 方案设计

### 后端标准

正式采用：

- PostgreSQL/TimescaleDB：`TEST_DATABASE_URL`
- Neo4j：`NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`
- Qdrant：`QDRANT_HOST`、`QDRANT_HTTP_PORT`、`QDRANT_GRPC_PORT`，可选 `QDRANT_URL`
- MinIO：`MINIO_ENDPOINT_URL`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY`、`MINIO_BUCKET`
- Redis：`REDIS_HOST`、`REDIS_PORT`、`REDIS_PASSWORD`

`mf-dki-bare/.env` 到 MoleculeForge 的映射：

```text
PG_HOST/PG_PORT/PG_USER/PG_PASSWORD/PG_DB
  -> TEST_DATABASE_URL=postgresql+asyncpg://PG_USER:PG_PASSWORD@PG_HOST:PG_PORT/PG_DB
NEO4J_BOLT_PORT/NEO4J_USER/NEO4J_PASSWORD
  -> NEO4J_URI=bolt://127.0.0.1:NEO4J_BOLT_PORT
QDRANT_HTTP_PORT
  -> QDRANT_HOST=127.0.0.1, QDRANT_HTTP_PORT=16333
MINIO_API_PORT/MINIO_USER/MINIO_PASSWORD
  -> MINIO_ENDPOINT_URL=http://127.0.0.1:19000
REDIS_PORT/REDIS_PASSWORD
  -> REDIS_HOST=127.0.0.1, REDIS_PORT=16379
```

### Qdrant client

实现 `mf_core.db.qdrant_client.QdrantCollectionClient`，对外提供当前服务需要的最小方法：

- `connect()`
- `upsert(data: dict[str, list]) -> int`
- `flush()`
- `search(vector, top_k, output_fields) -> list[dict]`
- `delete(ids) -> int`
- `get_stats(collection) -> dict`
- `drop_collection()`
- `disconnect()`

数据转换规则：

- 主键字段保留原始 id，并写入 payload `_id_str`。
- Qdrant point id 用 UUIDv5 从原始 id 稳定生成。
- vector 字段默认 `vector`，测试和 HUMU index service 仍可指定集合名。
- search 结果统一返回 `{id, distance, entity}`，与原服务响应兼容。

### Redis message bus

实现 `mf_agents.messaging.redis_bus.RedisBus`，替代 NATS：

- `connect()`：读取 Redis 配置或显式 url。
- `subscribe(subject, cb)`：使用 Redis pub/sub，后台 task 消费消息。
- `publish(subject, payload)`：发布 bytes。
- `request(subject, payload, timeout)`：创建 reply channel，把 `reply_to` 包入 JSON envelope；如果对端不支持，timeout 后返回空 bytes。
- `close()`：取消订阅 task 并关闭连接。
- 连接失败时使用 `_FallbackBus`，保持现有 fallback 测试能力。

`BaseAgent` 改为接收 `message_bus`，保留 `nats_client` 作为构造参数会继续传播旧语义，因此本轮不保留旧参数名。同步更新所有 agent 构造函数。

### 启动和验收

实施后真实 DKI 验收顺序：

1. 启动 `mf-dki-bare`：
   - `cd /workspace/mf-dki-bare && ./start.sh`
   - 若未初始化，执行 `make init`
2. 运行 `./status.sh`，要求 Postgres、Neo4j、Qdrant、MinIO、Redis 均 OK。
3. 导出 MoleculeForge 环境变量。
4. 运行 integration tests：
   - `uv run pytest tests/integration/test_dki_postgres.py tests/integration/test_dki_neo4j.py tests/integration/test_dki_qdrant.py tests/integration/test_dki_minio.py tests/integration/test_dki_redis.py -q`
5. 运行相关单元测试和 lint。

## KISS 四问

1. 这是现实问题还是想象问题？
   - 是现实问题。当前代码要求 Milvus/NATS，`mf-dki-bare` 实际提供 Qdrant/Redis，导致 DKI 验收无法直接用现有环境完成。
2. 有没有更简单的做法？
   - 有。正式替换为 Qdrant/Redis，而不是同时维护 Milvus/NATS 和 Qdrant/Redis 两套生产后端。
3. 会破坏什么？
   - 会破坏依赖 `MILVUS_URI`、`MILVUS_HOST`、`NATS_URL`、`NATSBus`、`milvus_client` 命名的调用方。缓解方式是一次性同步所有已发现调用点和测试，不保留并行旧实现。
4. 当前项目真的需要这个功能吗？
   - 需要。用户明确选择 `mf-dki-bare` 替换路线，且当前环境没有可用 Docker Milvus/NATS stack。

## 风险

- `mf-dki-bare` 当前未运行。即使代码完成，也不能声称真实 DKI 通过，必须启动并初始化后验收。
- Redis pub/sub 与 NATS JetStream 持久化语义不同。当前代码只使用 publish/subscribe/request 基础语义，没有发现 JetStream 特性调用；因此本轮只实现基础语义，不伪造持久化能力。
- Qdrant 与 Milvus schema/index API 不同。必须通过新的 Qdrant integration test 验证真实 upsert/search/delete/count。
- `mf-dki-bare` 的 Python SDK 不应直接作为 MoleculeForge 内部依赖复制使用；可以参考其实现，但 MoleculeForge 需在 `mf_core` 内维护自己的最小 client，避免跨项目路径依赖。

## 验收标准

- 代码中生产 DKI 路径不再要求 `MILVUS_URI`、`MILVUS_HOST`、`MILVUS_PORT`、`NATS_URL`。
- HUMU index service health 返回 `backend: qdrant`。
- Qdrant unit test 覆盖 point id、payload、search result 转换。
- Redis bus unit test 覆盖 fallback trace_id 传播。
- `mf-dki-bare` 未启动时，integration test 明确 skip 或 fail-fast 列出缺失 Qdrant/Redis 环境变量。
- `mf-dki-bare` 启动并初始化后，Postgres、Neo4j、Qdrant、MinIO、Redis integration tests 真实读写通过。
- 不声称 Milvus/NATS 生产路径仍存在。

## 实施计划摘要

1. TDD 替换 vector client：先写 Qdrant unit test，再实现 `qdrant_client.py`，删除 Milvus client 生产导入。
2. TDD 更新 HUMU index service：先改测试期待 Qdrant，再改服务配置和响应。
3. TDD 更新 patent indexing：先改测试为 `vector_client` 和 `patents_embedding`，再改 pipeline 命名。
4. TDD 替换 message bus：先写 Redis/fallback 测试，再实现 RedisBus 和更新 BaseAgent/agent 构造函数。
5. TDD 更新 integration tests：Milvus -> Qdrant，新增 MinIO/Redis。
6. 更新 artifact manifest、配置和架构文档中的 DKI 后端描述。
7. 启动 `mf-dki-bare` 后运行真实 DKI 验收；若启动失败，原样报告 stderr 和端口状态。

## 审批状态

等待用户确认本方案后再实施代码修改。
