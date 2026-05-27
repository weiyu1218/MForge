# CoreArchitecture v2 阶段 A 执行文档

> 执行阶段必须逐项验证。每个任务完成后先跑对应最小验证，再进入下一项；不得用占位结果、伪 runner、伪 checksum、空 manifest 或跳过测试来制造通过状态。

## 目标

完成 `docs/architecture/current-implementation-vs-corearchitecture-v2.md` 中「阶段 A — 工程债务清扫」的 11 项工作，使现有工程基础设施、测试收集、schema/proto、一致性配置、NL 解析、TAR 路由、gRPC 注册、K8s/docker-compose/Helm 编排进入可验证状态。

## 非目标

- 不实现阶段 B-E 的算法补全、真实模型训练、真实专利数据接入、真实 KRAS pilot 或商业合规工作。
- 不伪造 HFM/FragFM/LaMGen/CReM/EvoMol/UAS/iCLM/MMPT checkpoint。
- 不把缺真实 runner 的服务改成返回随机或硬编码成功结果。
- 不更新 `README.md`，直到阶段 A 代码完成后向用户简述变更并获得确认。

## 当前证据

### 已执行命令

```bash
uv run pytest tests/unit/test_ssp_compiler.py -q
uv run pytest tests/anti_degradation/test_no_degradation.py -q
uv run pytest tests/benchmark --collect-only -q
```

### 当前失败结果

- `tests/unit/test_ssp_compiler.py` collection error：`ImportError: cannot import name '_build_steps' from 'srb_agent.compiler'`。
- `tests/benchmark --collect-only` 输出 `Running 0 items in this shard`，退出码为 5。
- `tests/anti_degradation/test_no_degradation.py` 失败 3 项：
  - `test_no_print_in_production`：生产代码 `print()` 计数为 31，阈值要求 `< 30`。
  - `test_no_bare_except_pass`：`agents/orchestrator/src/orchestrator/pipeline.py:208` 和 `agents/nl2obj/src/nl2obj/parser.py:314` 吞错。
  - `test_no_mock_in_production_paths`：`pipelines/humu_pretrain/src/humu_pretrain/pipeline.py:562` 命中 `dummy_` 生产路径命名。

### 已确认的代码事实

- `agents/srb_agent/src/srb_agent/compiler.py` 当前真实 helper 名为 `_build_steps_from_route`，`compile_ssp()` 要求 `retrosyn_route.steps` 和 `retrosyn_route.route_id`。
- `tests/benchmark/` 下文件名为 `moses_benchmark.py`、`guacamol_benchmark.py`、`pmo_benchmark.py`；根 `pyproject.toml` 未设置 `python_files`，且未注册 `benchmark` marker。
- `schemas/crg.schema.json` 使用 `nodes/source_id/target_id`；`libs/mf-core/src/mf_core/types/crg.py` 与 `protos/moleculeforge/v1/core/crg.proto` 使用 `beliefs/source_belief_id/target_belief_id`。
- `infra/kubernetes/namespaces/mf-data-ns.yaml` 仍允许 Milvus `19530`，当前 Qdrant 配置为 `configs/services/qdrant.yaml` 的 `16333/16334`，docker compose 映射为 `16333:6333` 和 `16334:6334`。
- `infra/kubernetes/namespaces/mf-oracles-ns.yaml` 允许 `50061-50067`，但 oracle 服务入口实际端口包括 `boltz2=50053`、`dock=50054`、`fep=50055`、`admet=50056`、`retrosyn=50057`、`fto=50058`、`supply=50059`。
- `infra/docker/base/Dockerfile.oracle` 中 `GNINA_SHA256` 是 `PLACEHOLDER_UPDATE_WITH_ACTUAL_SHA256_FROM_RELEASE`。
- `libs/mf-core/src/mf_core/db/repositories/molecule_repo.py` 使用 `session.execute(None)` 和 `session.get(None)`，不是有效 SQLAlchemy repository。
- 24 个 `services/*/src/*/main.py` 存在，其中部分是 REST-only；gRPC 服务 `serve()` 创建 server 后未调用任何生成的 `add_*Servicer_to_server`。
- `tools/dev/generate_protos.py` 可生成 Python pb2 到 `libs/mf-core/src/mf_core/proto_gen/`；当前阶段需先生成再接服务注册。

## 调用链路分析

### SSP 编译链路

```text
SRBAgent / tests
  -> srb_agent.compiler.compile_ssp(molecule, retrosyn_route, run_id)
  -> _build_steps_from_route(retrosyn_route["steps"])
  -> _build_materials(steps)
  -> yield_estimator / cost_estimator
  -> mf_core.types.ssp.SSP
```

当前断点是测试仍导入旧 helper `_build_steps` 并使用旧 route fixture。阶段 A 只修测试与当前严格 route contract 的一致性，不恢复旧的隐式默认 route 生成。

### Benchmark 收集链路

```text
pytest
  -> pyproject.toml [tool.pytest.ini_options]
  -> python_files pattern
  -> tests/benchmark/*_benchmark.py
  -> benchmark marker
  -> pytest.skip(reason) for unavailable trained resources
```

当前断点是 pytest 默认只收集 `test_*.py`，且 `benchmark` marker 未注册。阶段 A 只让 benchmark 被收集并真实 skip，不把缺资源的 benchmark 标记为 pass。

### CRG schema 链路

```text
protos/moleculeforge/v1/core/crg.proto
  -> generated pb2
  -> libs/mf-core/src/mf_core/types/crg.py
  -> schemas/crg.schema.json
  -> agents using mf_agents.crg.graph.ChemicalReasoningGraph
```

当前断点是 JSON Schema 与 proto/Pydantic 字段名不一致。阶段 A 以 proto 和 `mf_core.types.crg` 为规范源，更新 JSON Schema 使用 `project_id/beliefs/source_belief_id/target_belief_id/version/provenance_id`。

### NL 解析链路

```text
NL input
  -> agents/nl2obj/src/nl2obj/parser.py
  -> service nl2obj Parse()
  -> cig-compiler stage1_semantic
  -> stage1b_grounding
  -> stage2_cig_build
  -> CIG / HCIV
```

当前断点是三套解析逻辑并存：service 硬编码 objectives、agent parser regex、CIG stage1 另有轻量 regex。阶段 A 以 `agents/nl2obj/src/nl2obj/parser.py` 为唯一 parser，service 与 CIG stage1 调用同一实现。

### TAR / generator router 链路

```text
HCIV + TaskProfile
  -> mf_core.routing.task_router.TaskAwareRouter
  -> generator-router-svc Route()
  -> GeneratorCoordAgent selected_generators
  -> 8 个真实 generator 名称
```

当前断点是 `generator-router-svc` 使用独立 Thompson learner 的 `gen-0..gen-7`，`GeneratorCoordAgent` 使用 5 个不存在的策略名。阶段 A 统一到 `mf_core.routing.task_router.GENERATOR_NAMES`。

### 服务化链路

```text
*.proto
  -> tools/dev/generate_protos.py
  -> mf_core.proto_gen.moleculeforge.v1...
  -> services/*/main.py Servicer
  -> generated add_*Servicer_to_server(server)
  -> grpc.aio.server
  -> docker-compose / K8s / Helm
```

当前断点是 pb2 未生成、服务没有注册生成的 servicer、编排端口与 main.py 端口不一致。阶段 A 先生成 proto，再逐服务注册，再同步 compose/K8s/Helm。

## 文件影响评估

### 主要修改文件

- `tests/unit/test_ssp_compiler.py`：修导入和 route fixture，使其匹配 `_build_steps_from_route` 与当前 `compile_ssp()` contract。
- `pyproject.toml`：增加 benchmark 收集规则与 `benchmark` marker。
- `schemas/crg.schema.json`：与 proto/Pydantic CRG 字段统一。
- `infra/kubernetes/namespaces/mf-data-ns.yaml`：移除 Milvus `19530`，加入 Qdrant 端口。
- `infra/kubernetes/namespaces/mf-oracles-ns.yaml`：按实际 oracle 端口重写 ingress 列表。
- `infra/docker/base/Dockerfile.oracle`：用真实 GNINA release binary SHA256 替换占位值。
- `agents/orchestrator/src/orchestrator/pipeline.py`：替换吞错逻辑，保留项目自创建幂等行为。
- `agents/nl2obj/src/nl2obj/parser.py`：替换 RDKit import/parse 吞错逻辑。
- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`：替换 `dummy_` 生产路径命名。
- `libs/mf-core/src/mf_core/db/repositories/molecule_repo.py`：实现真实 SQLAlchemy CRUD。
- `tests/unit/test_molecule_repo.py`：从 mock-only 验证改为断言 SQLAlchemy 语句行为；保留空 batch 行为。
- `services/nl2obj-svc/src/nl2obj_svc/main.py`：改为调用统一 NL parser。
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/stages/stage1_semantic.py`：改为调用统一 NL parser。
- `services/generator-router-svc/src/generator_router_svc/main.py`：改为使用 `TaskAwareRouter` 和真实 generator 名称。
- `services/generator-router-svc/src/generator_router_svc/domain/online_learner.py`：删除或降级为不再使用的内部实现；若仍保留，不能作为 Route 主路径。
- `agents/generator_coord/src/generator_coord/agent.py`：使用 `GENERATOR_NAMES`。
- `tools/dev/generate_protos.py` 与 `tools/codegen/check_proto_sync.sh`：确保生成输出可校验。
- `services/*/src/*/main.py`：为 gRPC 服务注册生成的 servicer；REST-only 服务不伪装 gRPC。
- `infra/docker/docker-compose.dev.yml`：补齐缺失服务并修正已有服务端口。
- `infra/helm/moleculeforge/Chart.yaml`、`infra/helm/moleculeforge/values.yaml`：移除不存在的 subchart 依赖或补齐本 chart template 所需 values。

### 需要新增的必要文件

- `libs/mf-core/src/mf_core/proto_gen/**`：由 `tools/dev/generate_protos.py` 生成的 pb2/pb2_grpc/pyi 文件。
- `infra/kubernetes/deployments/moleculeforge-services.yaml`：24 个 service 目录对应的 Kubernetes Deployment/Service 多文档 manifest。
- `infra/helm/moleculeforge/templates/_helpers.tpl`：Helm 名称、标签、service helper。
- `infra/helm/moleculeforge/templates/services.yaml`：服务 Deployment/Service 模板。

这些新增文件是阶段 A 第 9-11 项的直接产物，不是示例文件或占位文件。

### 关联影响文件

- `tests/unit/test_task_router.py`：统一 router 后需保持现有 TAR contract。
- `tests/unit/test_cic_compiler.py` 与 `tests/integration/cic/test_cic_end_to_end.py`：统一 NL parser 后需验证 CIG 行为不退化。
- `tests/unit/test_service_artifact_status.py`：服务注册和 fail-fast 边界调整后需验证。
- `infra/docker/base/Dockerfile.agent`、`Dockerfile.generator`、`Dockerfile.oracle`：compose/K8s 使用的 image build 入口。

## 执行顺序

### 任务 0：建立基线

执行命令：

```bash
uv run pytest tests/unit/test_ssp_compiler.py -q
uv run pytest tests/benchmark --collect-only -q
uv run pytest tests/anti_degradation/test_no_degradation.py -q
uv run pytest tests/unit/test_molecule_repo.py -q
```

预期：前三项复现当前失败；`test_molecule_repo.py` 当前通过但不证明 repository 真实可用，因为 mock 没检查 SQLAlchemy statement。

### 任务 1：修 SSP unit collection error

修改：

- `tests/unit/test_ssp_compiler.py`

执行内容：

- 将导入改为 `_build_steps_from_route`。
- 将 `SAMPLE_RETROSYN_ROUTE` 改为包含 `route_id` 与真实 `steps` list。
- 将 `_build_steps` 相关测试改为 `_build_steps_from_route`。
- 不在生产代码里新增 `_build_steps` 兼容别名，避免恢复旧的默认 route 生成逻辑。

验证：

```bash
uv run pytest tests/unit/test_ssp_compiler.py -q
```

通过标准：文件可 collect，且 SSP compiler tests 全部通过。

### 任务 2：修 benchmark 收集

修改：

- `pyproject.toml`

执行内容：

- 在 `[tool.pytest.ini_options]` 增加 `python_files = ["test_*.py", "*_benchmark.py"]`。
- 在 `markers` 增加 `benchmark: benchmark suites requiring trained generators or datasets`。

验证：

```bash
uv run pytest tests/benchmark --collect-only -q
uv run pytest tests/benchmark -q
```

通过标准：

- collect-only 能收集 `TestMosesBenchmark`、`TestGuacaMolBenchmark`、`TestPMOBenchmark` 下的测试。
- 实跑结果为 skip，skip 原因来自 benchmark 文件内的真实资源缺失说明。

### 任务 3：统一 CRG schema/proto 字段

修改：

- `schemas/crg.schema.json`

执行内容：

- 以 `protos/moleculeforge/v1/core/crg.proto` 和 `libs/mf-core/src/mf_core/types/crg.py` 为规范源。
- 将 schema 顶层 required 改为 `project_id`、`beliefs`、`edges`。
- belief 字段使用 `id`、`subject`、`predicate`、`object`、`confidence`、`evidence_ids`、`source_agent`、`timestamp_ns`。
- edge 字段使用 `source_belief_id`、`target_belief_id`、`relation`、`weight`。
- 保留 `version`、`provenance_id`。
- 不保留 `nodes/source_id/target_id` 双写兼容，因为仓内生产代码未读取 `schemas/crg.schema.json` 的旧字段。

验证：

```bash
uv run python -m json.tool schemas/crg.schema.json >/tmp/crg.schema.formatted.json
uv run pytest tests/unit/test_provenance.py tests/unit/test_service_artifact_status.py -q
```

通过标准：JSON 可解析，CRG/provenance 相关单测不退化。

### 任务 4：修 K8s netpol 端口与 GNINA checksum

修改：

- `infra/kubernetes/namespaces/mf-data-ns.yaml`
- `infra/kubernetes/namespaces/mf-oracles-ns.yaml`
- `infra/docker/base/Dockerfile.oracle`

执行内容：

- `mf-data-ns.yaml`：删除 `19530`；加入 `16333` 和 `16334`，匹配 `configs/services/qdrant.yaml` 与 docker compose host 端口。
- `mf-oracles-ns.yaml`：使用实际 oracle gRPC 端口 `50053`、`50054`、`50055`、`50056`、`50057`、`50058`、`50059`；REST 端口只保留服务实际暴露的 REST 端口。
- `Dockerfile.oracle`：下载 `https://github.com/gnina/gnina/releases/download/v1.3/gnina`，用 `sha256sum` 计算真实值后替换 `GNINA_SHA256`。下载失败时停止，不写入猜测值。

验证：

```bash
uv run python - <<'PY'
from pathlib import Path
for path in [
    "infra/kubernetes/namespaces/mf-data-ns.yaml",
    "infra/kubernetes/namespaces/mf-oracles-ns.yaml",
]:
    text = Path(path).read_text()
    assert "19530" not in text
assert "PLACEHOLDER_UPDATE_WITH_ACTUAL_SHA256_FROM_RELEASE" not in Path("infra/docker/base/Dockerfile.oracle").read_text()
PY
docker compose -f infra/docker/docker-compose.test.yml config >/tmp/mf-compose-test.yaml
```

通过标准：占位 checksum 消失，compose 配置仍可解析。

### 任务 5：修 anti_degradation 3 项失败

修改：

- `agents/orchestrator/src/orchestrator/pipeline.py`
- `agents/nl2obj/src/nl2obj/parser.py`
- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
- 可能涉及 2 个服务 `print()` 中任意 2 处，确保生产 `print()` 计数低于阈值；优先改正在当前阶段同时触达的服务入口。

执行内容：

- `orchestrator/pipeline.py`：捕获项目已存在或唯一约束冲突时忽略；其他异常继续抛出。
- `nl2obj/parser.py`：RDKit 不可用或 SMILES 解析失败时显式跳过当前 token，不使用 `except: pass`。
- `humu_pretrain/pipeline.py`：将 DDP padding 用的 `dummy_item/dummy_emb` 命名改为 `padding_item/padding_emb`，不改变训练逻辑。
- 服务入口 `print()` 改为 `logging.getLogger(__name__).info(...)`，只改最小数量以满足当前哨兵阈值；若后续任务触达更多服务入口，继续替换。

验证：

```bash
uv run pytest tests/anti_degradation/test_no_degradation.py -q
uv run pytest tests/unit/test_humu_training.py -q
```

通过标准：anti degradation 全部通过，HUMU training 单测不退化。

### 任务 6：修 MoleculeRepository 空壳

修改：

- `libs/mf-core/src/mf_core/db/repositories/molecule_repo.py`
- `tests/unit/test_molecule_repo.py`

执行内容：

- 使用 `sqlalchemy.select` 查询 `MoleculeORM`。
- `upsert()`：有 `inchikey` 时按 `inchikey` 查找并更新；无命中时创建 `MoleculeORM` 并 `session.add()`；最后 `flush()` 并返回 ORM。
- `get_by_inchikey()`：执行 `select(MoleculeORM).where(MoleculeORM.inchikey == inchikey)`。
- `get_by_id()`：执行 `session.get(MoleculeORM, mol_id)`。
- `batch_upsert()`：空列表返回 0；非空时逐条调用 `upsert()` 并返回数量。
- 单元测试检查传入 `session.execute()` 的不是 `None`，`session.get()` 的第一个参数是 `MoleculeORM`。

验证：

```bash
uv run pytest tests/unit/test_molecule_repo.py -q
uv run pytest tests/integration/test_dki_postgres.py -q
```

通过标准：unit 通过；integration 在未配置 `TEST_DATABASE_URL` 时只允许按现有 fixture skip，配置后应通过。

### 任务 7：整合 3 套 NL 解析并统一 TAR 与 generator-router

修改：

- `services/nl2obj-svc/src/nl2obj_svc/main.py`
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/stages/stage1_semantic.py`
- `services/generator-router-svc/src/generator_router_svc/main.py`
- `services/generator-router-svc/src/generator_router_svc/domain/online_learner.py`
- `tests/unit/test_cic_compiler.py`
- `tests/integration/cic/test_cic_end_to_end.py`
- `tests/unit/test_task_router.py`

执行内容：

- `nl2obj-svc` 调用 `nl2obj.parser.parse_intent` 或当前 parser 文件内实际公开入口；若公开入口名称不稳定，先在 parser 内补一个单一公开函数，再让 service 和 CIG stage1 同时调用。
- CIG `stage1_semantic` 不再维护另一套属性 keyword regex，改为把统一 parser 结果映射为 stage2 需要的 semantic dict。
- `generator-router-svc` 使用 `TaskAwareRouter`、`TaskProfile`、`GENERATOR_NAMES`，Route response 返回真实 generator 名称和权重。
- `SubmitFeedback` 使用 `TaskAwareRouter.update_with_feedback(generator_name, reward)`；如果 request 只含旧 `generator_idx`，通过 `GENERATOR_NAMES[index]` 映射。
- `online_learner.py` 不再作为主路径；如果保留文件，只保留兼容 adapter，不能产生 `gen-0` 作为外部 generator id。

验证：

```bash
uv run pytest tests/unit/test_cic_compiler.py tests/integration/cic/test_cic_end_to_end.py -q
uv run pytest tests/unit/test_task_router.py -q
uv run pytest tests/unit/test_service_artifact_status.py -q
```

通过标准：CIC 行为不退化；generator router 不再返回 `gen-0..gen-7` 作为业务 generator id。

### 任务 8：Generator Coordinator Agent 对齐 8 个真实 generator 名称

修改：

- `agents/generator_coord/src/generator_coord/agent.py`
- 需要时新增或更新 `tests/unit/test_generator_coord_agent.py`

执行内容：

- 从 `mf_core.routing.task_router import GENERATOR_NAMES`。
- `self.generators` 使用 `list(GENERATOR_NAMES)`。
- `_select_generators("auto", objectives)` 仍保持简单规则，但返回值只能来自 `GENERATOR_NAMES`。
- `strategy == "all"` 返回 8 个真实 generator。
- `strategy in self.generators` 返回该真实 generator。
- 未知 strategy 不回退到不存在的 `template_based`，改为返回稳定的默认真实 generator 列表，例如 `["hfm_3d", "fragfm"]`。

验证：

```bash
uv run pytest tests/unit -q
```

通过标准：无测试退化；新增测试证明 selected/available generators 均为 `GENERATOR_NAMES` 子集。

### 任务 9：生成 pb2 并注册 gRPC servicer

修改/新增：

- `libs/mf-core/src/mf_core/proto_gen/**`
- `tools/dev/generate_protos.py`
- `tools/codegen/check_proto_sync.sh`
- `services/*/src/*/main.py`

执行内容：

- 运行 `uv run python tools/dev/generate_protos.py` 生成 pb2/pb2_grpc/pyi。
- 修正生成脚本中直接写文件行为只限生成目录；手工代码修改继续使用 patch。
- 对有对应 proto service 的服务，导入生成的 `*_pb2_grpc` 并调用 `add_*Servicer_to_server(servicer, server)`。
- REST-only 服务保持 REST，不伪装 gRPC。
- 若某个 service main 的方法名与 proto RPC 不匹配，先写对应单元测试，再补齐方法名适配；不删除 fail-fast 资源检查。

验证：

```bash
uv run python tools/dev/generate_protos.py
bash tools/codegen/check_proto_sync.sh
uv run python - <<'PY'
from pathlib import Path
missing = []
for path in Path("services").glob("*/src/*/main.py"):
    text = path.read_text()
    if "grpc.aio.server" in text and "add_" not in text:
        missing.append(str(path))
assert not missing, "\n".join(missing)
PY
uv run pytest tests/unit/test_service_artifact_status.py -q
```

通过标准：生成目录存在，gRPC server 文件均有注册调用，服务 artifact status 单测通过。

### 任务 10：补 24 服务 K8s Deployment manifest 与 docker-compose 编排

修改/新增：

- `infra/kubernetes/deployments/moleculeforge-services.yaml`
- `infra/docker/docker-compose.dev.yml`

执行内容：

- K8s manifest 覆盖当前 24 个 service 目录：
  - `admet-svc`
  - `api-gateway`
  - `boltz2-svc`
  - `cig-compiler-svc`
  - `crem-generator-svc`
  - `critic-svc`
  - `dock-svc`
  - `evomol-rl-svc`
  - `feature-store-svc`
  - `fep-svc`
  - `fragfm-generator-svc`
  - `fto-patent-svc`
  - `generator-router-svc`
  - `hfm-generator-svc`
  - `humu-encoder-svc`
  - `humu-index-svc`
  - `iclm-svc`
  - `lamgen-generator-svc`
  - `mmpt-generator-svc`
  - `nl2obj-svc`
  - `orchestrator-svc`
  - `provenance-svc`
  - `retrosyn-svc`
  - `supply-oracle-svc`
- 端口以 `services/*/main.py` 实际监听端口为准；修正 `docker-compose.dev.yml` 中 `humu-encoder-svc=50051`、`generator-router-svc=50052`。
- compose 中每个服务使用已有 base Dockerfile，保持真实模块启动命令 `python -m <package>`。
- 依赖 DKI 的服务通过 compose service name 连接 `postgres/neo4j/qdrant/minio/redis` 的容器内端口，不使用宿主机端口。

验证：

```bash
docker compose -f infra/docker/docker-compose.dev.yml config >/tmp/mf-compose-dev.yaml
uv run python - <<'PY'
from pathlib import Path
text = Path("infra/kubernetes/deployments/moleculeforge-services.yaml").read_text()
for name in [p.name for p in Path("services").iterdir() if p.is_dir()]:
    assert name in text, name
PY
```

通过标准：compose 可解析；K8s manifest 覆盖所有服务目录。

### 任务 11：补全 Helm chart templates

修改/新增：

- `infra/helm/moleculeforge/Chart.yaml`
- `infra/helm/moleculeforge/values.yaml`
- `infra/helm/moleculeforge/templates/_helpers.tpl`
- `infra/helm/moleculeforge/templates/services.yaml`

执行内容：

- 移除当前 `Chart.yaml` 中没有本地 chart 目录支撑的 subchart dependencies，避免 `helm template` 因依赖缺失失败。
- 在 `values.yaml` 增加 `services` map，每个 service 声明 `enabled`、`image`、`command`、`ports`、`resources`。
- `templates/services.yaml` 根据 `values.services` 渲染 Deployment 和 Service。
- 不为缺真实 runner/checkpoint 的服务写成功探针；readiness/liveness 只检查进程端口或 HTTP health endpoint，不伪造业务可用性。

验证：

```bash
helm template moleculeforge infra/helm/moleculeforge >/tmp/moleculeforge-rendered.yaml
uv run python - <<'PY'
from pathlib import Path
rendered = Path("/tmp/moleculeforge-rendered.yaml").read_text()
for name in [p.name for p in Path("services").iterdir() if p.is_dir()]:
    assert name in rendered, name
PY
```

通过标准：Helm 可渲染，输出覆盖所有服务目录。

## 总体验证

阶段 A 全部任务完成后执行：

```bash
uv run pytest tests/unit -q
uv run pytest tests/anti_degradation/test_no_degradation.py -q
uv run pytest tests/benchmark --collect-only -q
uv run pytest tests/benchmark -q
docker compose -f infra/docker/docker-compose.test.yml config >/tmp/mf-compose-test.yaml
docker compose -f infra/docker/docker-compose.dev.yml config >/tmp/mf-compose-dev.yaml
bash tools/codegen/check_proto_sync.sh
helm template moleculeforge infra/helm/moleculeforge >/tmp/moleculeforge-rendered.yaml
```

如果 DKI 环境变量已配置，再执行：

```bash
uv run pytest tests/integration -q
```

## KISS 四问

1. 这是现实问题还是想象问题：是现实问题，当前已有 collection error、benchmark collect=0、anti_degradation 失败、schema/proto 不一致、配置端口不一致、repository 空壳和服务未注册证据。
2. 有没有更简单的做法：有，阶段 A 只修工程债务和一致性，不实现算法阶段，不引入新框架。
3. 会破坏什么：主要风险是 schema 字段统一、NL parser 统一和服务注册可能影响现有测试；每项都有最小验证命令和回归命令。
4. 当前项目真的需要这个功能吗：需要。阶段 A 是后续算法补全和真实模型接入前的工程前置条件。

## 风险与处理

- GNINA checksum 获取失败：停止任务 4，汇报下载命令和 stderr，不写猜测值。
- `grpcio-tools` 缺失：使用 `uv` 安装/同步已有依赖配置后再生成；不能手写 pb2。
- 生成 pb2 后 import path 不匹配：先修 `tools/dev/generate_protos.py` 和生成包 `__init__.py`，再接服务注册。
- Helm CLI 不存在：如实报告 `helm: command not found`，不声称 Helm 验证通过。
- Docker 不可用：如实报告 docker 命令 stderr，不跳过 compose config 验证。
- Unit 全量测试因既有无关失败阻塞：先隔离阶段 A 涉及测试；无关失败需单独记录，不改业务逻辑规避。

## 完成标准

- 阶段 A 11 项均有对应代码或配置变更。
- 所有阶段 A 最小验证命令按实际输出记录。
- 不能通过的验证必须给出原始错误和下一步处理选项。
- `README.md` 在用户确认阶段 A 更新摘要后再更新。
