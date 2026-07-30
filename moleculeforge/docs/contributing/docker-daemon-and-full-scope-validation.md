# 未完成的外部环境与真实资源验证

当前代码级完整线路已经接通，本机全量测试和静态检查已经通过。开发环境中的合成数据仅用于验证控制流，所有合成结果均带有验证标记，并会在生产业务边界被拒绝。

以下内容因当前环境缺少容器、集群、真实模型、真实数据集和外部存储而未执行，不能视为生产验收完成。

## 容器镜像与 GPU 运行验证

当前环境没有 `docker`、`podman` 和 `nvidia-smi`，因此尚未完成：

- 实际构建 base、agent、chem、generator 和 oracle 镜像。
- 使用 Docker Compose 启动完整开发服务栈。
- 在 Linux AMD64 和真实 NVIDIA GPU 上验证 CUDA 13 运行时。
- 在最终镜像中实际调用 GNINA、Boltz、OpenFE、AiZynthFinder、RSGPT 和 UAlign。
- 验证镜像内动态库、模型缓存、文件权限和进程入口在容器运行时均可用。

完成条件：在具备 Docker daemon 和 NVIDIA GPU 的 Linux AMD64 环境构建全部镜像，启动完整服务栈，并逐项执行上述工具的真实输入冒烟验证。

## 真实模型与业务工件验证

当前环境没有配置以下生产必需工件：

- `HUMU_CHECKPOINT_PATH`
- `HFM_CHECKPOINT_PATH`
- `HFM_DECODER_PATH`
- `FRAGFM_VOCAB_PATH`
- `CREM_MMP_DB_PATH`
- `MMPT_INDEX_URI`
- `ICLM_MODEL_PATH`
- `ADMET_MODEL_PATH`
- `BOLTZ_MODEL_PATH`
- `BOLTZ_INPUT_TEMPLATE_DIR`
- `DIFFDOCK_MODEL_PATH`
- `QDRANT_URL`
- `SUPPLY_CATALOG_URI`
- `FEAST_REPO_PATH`
- `AIZYNTH_CONFIG_PATH`

因此尚未完成：

- 使用真实 HuMU 预训练权重执行编码、生成和下游验证。
- 使用 HFM、FragFM、CReM、MMPT 和 ICLM 的真实模型或索引执行候选生成。
- 使用真实 ADMET、Boltz、DiffDock、OpenFE 和 AiZynthFinder 资源执行 Oracle 与逆合成评估。
- 使用真实 HypSeek 教师模型生成 teacher embeddings，并完成一次 ICLM 在线更新和 checkpoint 持久化。
- 使用真实供应目录完成 Supply Agent 的可采购性验证。

完成条件：提供清单中的真实工件和服务地址，使相关服务状态进入非合成的可用状态，并使用真实分子与靶点输入完成一次 full scope 工作流。

## 外部基础设施与集群验证

当前环境没有 `helm` 和 `kubectl`，也没有配置 PostgreSQL、Neo4j、Qdrant、MinIO 和 Redis 的真实连接，因此尚未完成：

- 渲染并安装 Helm release。
- 在 Kubernetes 中验证 Secret、持久卷、GPU 调度、服务发现、健康检查和滚动启动。
- 验证真实 Redis 请求应答、任务恢复和并发路由状态。
- 验证 PostgreSQL、Neo4j、Qdrant 与 MinIO 的持久化、查询和 provenance 审计链路。
- 执行依赖真实基础设施的 KRAS G12C 与审计完整性 E2E 测试。

完成条件：部署上述基础设施并提供测试所需连接配置，在集群中完成服务就绪检查、持久化检查和无外部环境跳过项的集成与 E2E 测试。

## 真实数据训练与科学指标验证

当前仅使用小型合成数据验证数据流和训练控制逻辑，尚未完成：

- 使用完整 HuMu 训练数据执行预训练、恢复训练和模型产出。
- 使用 GuacaMol、MOSES、PMO 和 CrossDocked 真实资源执行生成与口袋条件基准。
- 使用真实 KRAS G12C 结构和实验数据验证完整发现流程。
- 验证真实模型的收敛性、生成质量、结合预测质量、逆合成成功率和供应可得性。

完成条件：提供对应数据集，执行完整训练和基准流程，并基于实际输出确定可接受的科学指标。合成数据测试结果不能替代该验收。

## 当前验证边界

已完成的代码验证结果为 `2487 passed, 39 skipped`。39 项跳过测试属于缺少外部数据、模型、数据库或显式 E2E 运行条件的场景。Ruff、导入边界、依赖锁、配置解析和差异格式检查均已通过。

在上述外部验证全部完成前，只能声明代码级完整线路和合成数据验证已通过，不能声明生产环境、真实模型效果或科学指标已经验收。
