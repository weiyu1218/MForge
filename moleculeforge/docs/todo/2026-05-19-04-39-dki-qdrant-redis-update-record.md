# DKI Qdrant Redis 替换更新记录

## 范围

本次更新按 `mf-dki-bare` 路线，将 MoleculeForge 的 DKI 标准从 `Milvus + NATS` 替换为 `Qdrant + Redis`，并继续使用 `PostgreSQL + Neo4j + MinIO`。本记录只描述已经执行并验证过的内容，不包含 HUMU 新 joint/intent 数据和真实模型 checkpoint。

## 已完成内容

### DKI 后端替换

- 新增 `mf_core.db.qdrant_client.QdrantCollectionClient`，用于替代 Milvus collection client。
- HUMU index service 改为读取 Qdrant 配置，并通过 Qdrant client 完成 embedding upsert、search、delete、stats。
- Agent messaging 从 NATS 路径改为 Redis 路径，新增 `RedisBus`。
- Python 依赖移除 `pymilvus`、`nats-py`，新增 `qdrant-client>=1.12,<1.14` 和 `redis>=5.0`。
- `uv.lock` 已更新，锁定的项目依赖中包含 `qdrant-client 1.13.3`。

### 配置与部署对齐

- `configs/services`、`configs/agents` 中的 DKI 相关配置改为 Qdrant/Redis。
- Docker compose 测试、开发、minimal、DKI 配置均同步到 Qdrant/Redis。
- Helm values 已同步为 Qdrant/Redis。
- 架构和 ADR 文档已移除当前 DKI 标准中的 Milvus/NATS 目标描述。

### 测试覆盖调整

- Milvus 单元和集成测试替换为 Qdrant 对应测试。
- NATS 单元测试替换为 RedisBus 对应测试。
- DKI 真实集成测试覆盖 PostgreSQL、Neo4j、Qdrant、MinIO、Redis。
- ORM timestamp 已调整为 naive UTC，避免 asyncpg 写入 timestamptz/naive datetime 不一致。
- Redis close 路径改为 `aclose()`。
- Neo4j integration fixture 的 inchikey 长度已修正。

## 真实 DKI 环境验证

### mf-dki-bare 服务状态

`mf-dki-bare` 已通过 root supervisord 启动。由于 supervisord 需要 `setuid` 到 postgres，启动命令使用了：

```bash
sudo -n make start
```

服务状态命令：

```bash
cd /workspace/mf-dki-bare
sudo -n ./status.sh
```

已验证端口：

- PostgreSQL: `15432`
- Neo4j HTTP: `17474`
- Neo4j Bolt: `17687`
- Qdrant HTTP: `16333`
- Qdrant gRPC: `16334`
- MinIO API: `19000`
- MinIO Console: `19001`
- Redis: `16379`

### 初始化和 smoke

本地环境存在代理变量时，Qdrant 本地 HTTP 请求会走 SOCKS 代理并触发 `socksio` 缺失错误。因此初始化和 smoke 使用了去代理环境：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
  NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1 \
  make init
```

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
  NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1 \
  make smoke
```

`make smoke` 已返回：

```text
5/5 subsystems healthy.
```

覆盖子系统：

- PostgreSQL / TimescaleDB
- Neo4j
- Qdrant
- MinIO
- Redis

## MoleculeForge 验证结果

### 定向单元测试

```bash
uv run pytest tests/unit/test_vector_store.py tests/unit/test_redis_bus.py tests/unit/test_service_artifact_status.py tests/unit/test_indexing_pipelines.py
```

结果：

```text
36 passed
```

### 真实 DKI 集成测试

使用 `/workspace/mf-dki-bare/.env` 映射 MoleculeForge 所需环境变量后运行 DKI 集成测试。

结果：

```text
10 passed
```

说明：运行过程中观察到一条 Qdrant client 版本提示，提示系统路径中的 `qdrant-client 1.18` 与 server `1.12.4` 不完全匹配。项目锁文件中的 MoleculeForge 依赖为 `qdrant-client 1.13.3`。

### Ruff

```bash
uv run ruff check libs services agents pipelines tests models
```

结果：

```text
All checks passed
```

### Compose 配置解析

以下 compose 配置已完成解析验证：

- `infra/docker/docker-compose.test.yml`
- `infra/docker/docker-compose.dki.yml`
- `infra/docker/docker-compose.minimal.yml`
- `infra/docker/docker-compose.dev.yml`

### Milvus / NATS 残留扫描

已对活动代码、配置、infra、架构文档、依赖文件执行关键词扫描：

```bash
rg -n "Milvus|milvus|MILVUS|NATS|nats|NATS_URL|nats_client|pymilvus|nats-py" \
  libs services agents pipelines tests models configs infra docs/adr docs/architecture pyproject.toml uv.lock
```

结果为无命中。

## 当前结论

DKI 基础设施链路已经按 `mf-dki-bare` 的 `PostgreSQL + Neo4j + Qdrant + MinIO + Redis` 组合打通，并且 MoleculeForge 的 DKI 客户端、配置、测试和依赖已经完成 Qdrant/Redis 替换。

不能据此声称全栈业务流程已经全部打通。尚未完成的业务级证据包括：

- 非 HUMU checkpoint 的真实 generator / oracle / retrosyn runner 资源验收。
- KRAS pilot E2E 真实运行证据。
- Audit E2E 真实运行证据。
- Provenance production store 对真实 Neo4j / Postgres / MinIO 的端到端写入验收。

## 操作注意事项

- `mf-dki-bare` 当前需要通过 `sudo -n` 管理 supervisord 状态。
- `make init` 和 `make smoke` 需要去掉代理环境，避免本地 Qdrant 请求被代理拦截。
- 后续所有声称“真实资源已接入”的结论，必须由实际路径、环境变量、服务状态和测试输出支撑。
