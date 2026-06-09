# CoreArchitecture v2 补齐后全流程跑通验证 Runbook

## 0. 定位与判定边界

本文用于在甲乙两侧工作完成后，验证 MoleculeForge 从环境、契约、服务、工程闭环到生产全量 E2E 的跑通情况，并给出可直接执行的命令。

上位文档：

- `docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- `docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`
- `docs/architecture/current-implementation-vs-corearchitecture-v2.md`

本文不新增功能，不修改业务代码，不把本地 smoke 冒充生产验收。验收分三层：

| 层级 | 目标 | 通过标准 |
|---|---|---|
| L0 本地工程闭环 | 证明甲乙交付的代码、契约、runner、artifact gate 可本地运行 | focused unit gate、CLI smoke、engineering-scope E2E 退出码为 0 |
| L1 DKI/服务集成 | 证明真实 Postgres/Neo4j/Qdrant/MinIO/Redis 与服务栈可达 | DKI integration tests 退出码为 0，服务 health 全部可达 |
| L2 生产全量闭环 | 证明 full workflow、审计、benchmark、KRAS pilot 使用真实资源跑通 | audit E2E、KRAS full E2E、benchmark 全量退出码为 0 |

当前已知边界：

- `KRAS_E2E_SCOPE=engineering` 可验证工程链路；`KRAS_E2E_SCOPE=full` 必须等待 full 生产资源齐备。
- `tests/benchmark` 全量通过依赖 production-quality HFM checkpoint/decoder 或生产 decoder command；当前 smoke artifact 不能作为生成质量达标证据。
- H10 集群发布验证需要真实 Kubernetes/Helm 环境；本地 Docker Compose 只能作为服务 wiring smoke。

## 1. 通用执行前置

所有命令默认在仓库根目录执行：

```bash
cd /workspace/MForge/moleculeforge
```

安装/同步工作区依赖：

```bash
uv sync --all-extras
```

加载环境变量。不要把 `.env` 中的 token、API key、secret 打印到日志：

```bash
set -a
source .env
set +a

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
```

检查固定端口是否已被旧进程占用：

```bash
for port in \
  15432 17474 17687 16333 16334 19000 19001 16379 \
  50051 50052 50053 50054 50055 50056 50057 50059 50060 50061 50062 50063 \
  50065 50066 50067 50069 50070 50071 \
  8000 8008 8009 8010 8011 8012 8013
do
  lsof -iTCP:"$port" -sTCP:LISTEN -Pn || true
done
```

如端口被旧 MoleculeForge compose 栈占用，先按需关闭旧栈。确认旧栈数据不需要保留后再执行 `down -v`：

```bash
docker compose -f infra/docker/docker-compose.dev.yml down
docker compose -f infra/docker/docker-compose.dki.yaml down
```

仅清理临时 test stack 时使用：

```bash
docker compose -f infra/docker/docker-compose.test.yml down -v
```

## 2. L0 本地工程闭环

### 2.1 静态与 proto 同步

```bash
uv run ruff check .
uv run ruff format --check .
bash tools/codegen/check_proto_sync.sh
```

可选类型检查。当前 Makefile 对 mypy/import-linter 使用 `|| true`，因此这两项不能单独作为硬性通过证据：

```bash
uv run mypy libs/ services/ agents/ models/ || true
uv run lint-imports || true
```

### 2.2 甲乙对接契约回归

覆盖 C1 `generator_params` 透传、C2 CRG belief、C3 HUMU embedding 合法性，以及 W1/W2/W3/W5/W6/W8/W9/W10/W11/W12/W13 的本地工程 gate：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest \
  tests/unit/test_graph_repo.py \
  tests/unit/test_generator_coord_agent.py \
  tests/unit/test_generators.py \
  tests/unit/test_service_artifact_status.py \
  tests/unit/test_task_router.py \
  tests/unit/test_mf_eval.py \
  tests/unit/test_phase_b_generators.py \
  tests/unit/test_cross_paradigm_kd.py \
  tests/unit/test_validation_agent.py \
  tests/unit/test_srb_agent.py \
  -q
```

外部 runner wrapper focused gate：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest \
  tests/unit/test_h4_quantum_wrapper.py \
  tests/unit/test_h5_oracle_wrappers.py \
  tests/unit/test_h6_retrosyn_wrapper.py \
  tests/unit/test_h9_cig_llm_wrappers.py \
  -q
```

全量 unit 回归：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/unit -q
```

判定：

- 退出码必须为 0。
- 失败时保留 stderr、失败测试名、文件行号，不得用修改业务逻辑规避失败。

### 2.3 关键 CLI smoke

TAR ProxylessNAS runner：

```bash
printf '%s\n' '{
  "reward_batches_by_dataset": {
    "kras": [
      {"hfm_3d": 0.2, "fragfm": 0.8}
    ]
  },
  "generator_costs": {"hfm_3d": 5.0, "fragfm": 1.0},
  "cost_weight": 0.1,
  "learning_rate": 1.0,
  "temperature": 1.0
}' | uv run python -m generator_router_svc.tar_proxyless_runner
```

FragFM HUMU quality gate。`checkpoints/fragfm_humu_5k` 是本地 engineering candidate，不是 production-quality artifact：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m mf_generators.fragfm.quality \
  --vocab checkpoints/fragfm_humu_5k/vocab.json \
  --checkpoint checkpoints/fragfm_humu_5k/best_model.pt \
  --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt \
  --manifest checkpoints/fragfm_humu_5k/training_manifest.json \
  --min-humu-coverage 1.0 \
  --strict
```

KD teacher embedding artifact gate：

```bash
export KD_TEACHER_RECORDS_PATH="${KD_TEACHER_RECORDS_PATH:?provide real teacher records JSONL}"
export KD_TEACHER_EMBEDDINGS_PATH="${KD_TEACHER_EMBEDDINGS_PATH:-data/processing/generator_artifacts/kd_teacher_embeddings.json}"

PYTHONDONTWRITEBYTECODE=1 uv run python -m mf_core.routing.kd_artifacts \
  --input "$KD_TEACHER_RECORDS_PATH" \
  --output "$KD_TEACHER_EMBEDDINGS_PATH" \
  --expected-dim 129 \
  --min-embeddings 1 \
  --strict
```

如没有真实 teacher records，只运行 unit gate 中的 W13 测试；不要临时伪造 production teacher records。

H9 CIG LLM/SRM wrapper gate：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/unit/test_h9_cig_llm_wrappers.py -q
```

W9 HFM neural geometry decoder gate：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/unit/test_generators.py -q -k "neural_geometry_decoder or geometry_decoder"
```

W10 HCIV checkpoint training/export gate：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/unit/test_cic_compiler.py -q -k "hciv"
```

W12 CReM-pharm-3D scorer gate：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/unit/test_phase_b_generators.py -q -k "crem"
```

## 3. L1 DKI 基础设施集成

### 3.1 启动 DKI 栈

`infra/docker/docker-compose.dki.yaml` 使用固定端口：Postgres `15432`、Neo4j `17687`、Qdrant `16333`、MinIO `19000`、Redis `16379`。

为保证 compose 密码和 `.env` 中测试连接一致，启动前同步这些变量：

```bash
export PG_PASSWORD="${PG_PASSWORD:?}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:?}"
export REDIS_PASSWORD="${REDIS_PASSWORD:?}"
export MINIO_PASSWORD="${MINIO_SECRET_KEY:?}"

docker compose -f infra/docker/docker-compose.dki.yaml up -d postgres neo4j qdrant minio redis
```

等待健康检查：

```bash
docker compose -f infra/docker/docker-compose.dki.yaml ps
curl -sf "$QDRANT_URL/readyz"
curl -sf "$MINIO_ENDPOINT_URL/minio/health/live"
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" ping
```

创建 MinIO bucket。若 bucket 已存在，命令仍可继续：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python - <<'PY'
import asyncio
import os

from aiobotocore.session import get_session
from botocore.exceptions import ClientError

async def main():
    session = get_session()
    async with session.create_client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT_URL"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
    ) as client:
        try:
            await client.create_bucket(Bucket=os.environ["MINIO_BUCKET"])
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                raise

asyncio.run(main())
PY
```

### 3.2 运行 DKI integration tests

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/integration/test_dki_*.py -q
```

期望：

- 退出码为 0。
- Postgres、Neo4j、Qdrant、MinIO、Redis 相关测试均不是 skip。
- 若出现 Qdrant client/server minor version warning，只记录 warning，不等同失败。

## 4. L1 服务栈启动与健康检查

### 4.1 启动 dev 服务栈

保持与 compose 文件一致的端口，不临时改端口：

```bash
export PG_PASSWORD="${PG_PASSWORD:?}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:?}"
export REDIS_PASSWORD="${REDIS_PASSWORD:?}"
export MINIO_PASSWORD="${MINIO_SECRET_KEY:?}"

docker compose -f infra/docker/docker-compose.dev.yml up -d \
  postgres neo4j qdrant minio redis \
  humu-encoder-svc generator-router-svc hypseek-teacher-svc \
  admet-svc boltz2-svc dock-svc fep-svc retrosyn-svc supply-oracle-svc \
  crem-generator-svc fragfm-generator-svc hfm-generator-svc mmpt-generator-svc iclm-svc \
  cig-compiler-svc nl2obj-svc critic-svc provenance-svc orchestrator-svc pareto-bo-svc \
  api-gateway feature-store-svc humu-index-svc
```

查看启动状态：

```bash
docker compose -f infra/docker/docker-compose.dev.yml ps
```

### 4.2 REST health

```bash
curl -sf http://127.0.0.1:8000/health
curl -sf http://127.0.0.1:8010/health
curl -sf http://127.0.0.1:8011/health
curl -sf http://127.0.0.1:8012/healthz
curl -sf http://127.0.0.1:8013/health
curl -sf http://127.0.0.1:8008/health
curl -sf http://127.0.0.1:8009/health
```

### 4.3 API gateway smoke

```bash
curl -sf -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles":"CC(=O)Oc1ccccc1C(=O)O"}'
```

直接运行 API e2e：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/e2e/test_predict_api.py -q
```

## 5. Orchestrator 工程链路跑通

### 5.1 REST engineering-scope smoke

工程范围不要求 HFM/Boltz/AiZynth 全量生产资源，但会验证 `PLANNING -> GENERATING -> VALIDATING -> RETROSYN -> CRITIC` 的主链路：

```bash
curl -sf -X POST http://127.0.0.1:8011/v1/orchestrator/design \
  -H "Content-Type: application/json" \
  -d '{
    "nl_input": "Design covalent inhibitors for KRAS G12C with Molecular weight < 500 Da and LogP 1-4.",
    "workflow_scope": "engineering",
    "n_samples": 2,
    "seed": 42
  }'
```

返回中至少检查：

- `status` 为 `completed` 或 `escalated`。
- `history` 包含 `PLANNING`、`GENERATING`、`VALIDATING`、`RETROSYN`、`CRITIC`。
- `state.candidates` 非空。
- `state.critic.total_rules` 大于 0。

### 5.2 KRAS engineering-scope e2e

```bash
RUN_KRAS_G12C_E2E=1 \
KRAS_E2E_SCOPE=engineering \
ORCHESTRATOR_E2E_READY=1 \
PYTHONDONTWRITEBYTECODE=1 \
uv run pytest tests/e2e/test_kras_g12c_pilot.py -q
```

期望：

- 退出码为 0。
- HFM expert generation、external affinity oracles、AiZynthFinder production resources 对应测试会按 engineering scope skip。
- `test_critic_reviews_concerns` 和 `test_end_to_end_kras_g12c` 验证工程主链路。

## 6. 生产资源 full workflow 验收

### 6.1 full-scope 资源预检

检查 full KRAS e2e 所需 env 是否齐备：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python - <<'PY'
from tests.e2e.test_kras_g12c_pilot import kras_e2e_preflight_status

status = kras_e2e_preflight_status()
print(status)
raise SystemExit(0 if status["ready"] else 1)
PY
```

full scope 至少要求：

- `HFM_CHECKPOINT_PATH`
- `HFM_DECODER_PATH`
- `BOLTZ_MODEL_PATH`
- `BOLTZ_INPUT_TEMPLATE_DIR`
- `AIZYNTH_CONFIG_PATH`
- `BOLTZ_BINARY`
- `CRITIC_AGENT_READY=1`
- `ORCHESTRATOR_E2E_READY=1`
- `SIGSTORE_IDENTITY_TOKEN`
- `SIGSTORE_EXPECTED_IDENTITY`
- `SIGSTORE_SIGN_COMMAND`
- `SIGSTORE_VERIFY_COMMAND`
- `SIGSTORE_REKOR_URL`
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `MINIO_ENDPOINT_URL`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_BUCKET`
- `REDIS_HOST`
- `REDIS_PORT`
- `PROVENANCE_DATABASE_URL` 或 `TEST_DATABASE_URL`
- `PROVENANCE_STORE_MODE=production_real`

### 6.2 audit E2E

检查 audit e2e 所需 env 是否齐备：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python - <<'PY'
from tests.e2e.test_audit_completeness import audit_e2e_preflight_status

status = audit_e2e_preflight_status()
print(status)
raise SystemExit(0 if status["ready"] else 1)
PY
```

运行审计 E2E：

```bash
RUN_AUDIT_E2E=1 \
PROVENANCE_STORE_MODE=production_real \
PYTHONDONTWRITEBYTECODE=1 \
uv run pytest tests/e2e/test_audit_completeness.py -q
```

期望：

- 退出码为 0。
- 4 项 audit e2e 全部通过。
- Sigstore identity token 不写入仓库、不写入日志。

### 6.3 KRAS full-scope E2E

```bash
RUN_KRAS_G12C_E2E=1 \
KRAS_E2E_SCOPE=full \
CRITIC_AGENT_READY=1 \
ORCHESTRATOR_E2E_READY=1 \
PROVENANCE_STORE_MODE=production_real \
PYTHONDONTWRITEBYTECODE=1 \
uv run pytest tests/e2e/test_kras_g12c_pilot.py -q
```

期望：

- 退出码为 0。
- `test_generates_diverse_candidates` 使用 HFM production artifact。
- `test_oracle_cascade_validates_affinity` 使用 Boltz/FEP/ADMET 等真实 oracle 配置。
- `test_retrosyn_plans_synthesis` 使用真实 retrosyn runner。
- `test_end_to_end_kras_g12c` 的 `history` 精确为 `PLANNING, GENERATING, VALIDATING, RETROSYN, CRITIC`。

如该命令失败，不能降级为 engineering scope 宣称 full 跑通。

## 7. Benchmark 验收

### 7.1 资源预检

```bash
test -f "$MOSES_REFERENCE_SMILES_PATH"
test -f "$PMO_SCORE_TABLE_PATH"
test -f "$CROSSDOCKED_BENCHMARK_JSONL"
test -f "$HFM_CHECKPOINT_PATH"
test -f "$HFM_DECODER_PATH"
```

### 7.2 focused benchmark

MOSES：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark/moses_benchmark.py -q -ra
```

GuacaMol：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark/guacamol_benchmark.py -q -ra
```

PMO：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark/pmo_benchmark.py -q -ra
```

CrossDocked：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark/crossdocked_benchmark.py -q -ra
```

### 7.3 全量 benchmark

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark -q -ra
```

期望：

- 退出码为 0。
- 不存在因 `MOSES_REFERENCE_SMILES_PATH`、`PMO_SCORE_TABLE_PATH`、`CROSSDOCKED_BENCHMARK_JSONL`、`HFM_CHECKPOINT_PATH`、`HFM_DECODER_PATH` 缺失导致的 skip。
- 若当前 HFM smoke artifact 只生成重复 `CCO` 并导致 MOSES/GuacaMol/PMO 失败，应登记为 production-quality HFM artifact 未达标，不得降低阈值冒充通过。

## 8. PCBO 本地闭环

命令行方式调用 ParetoBO service wrapper。默认 candidate provider 为 `TangentSpaceNoiseCandidateProvider`，默认 oracle evaluator 为 embedding proxy：

```bash
printf '%s\n' '{
  "reference": [0.0, 0.0],
  "lower_bounds": [0.0],
  "upper_bounds": [1.0],
  "batch_size": 2,
  "n_rounds": 1,
  "maximize": true,
  "observed_embeddings": [[0.1, 0.2], [0.2, 0.1]],
  "observed_objectives": [[0.1, 0.2], [0.2, 0.1]],
  "observed_constraints": [[0.5], [0.5]]
}' | PYTHONDONTWRITEBYTECODE=1 uv run python -c 'from pareto_bo.service import main; main()'
```

REST 方式：

```bash
curl -sf -X POST http://127.0.0.1:8013/v1/pareto-bo/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "reference": [0.0, 0.0],
    "lower_bounds": [0.0],
    "upper_bounds": [1.0],
    "batch_size": 2,
    "n_rounds": 1,
    "maximize": true,
    "observed_embeddings": [[0.1, 0.2], [0.2, 0.1]],
    "observed_objectives": [[0.1, 0.2], [0.2, 0.1]],
    "observed_constraints": [[0.5], [0.5]]
  }'
```

生产 oracle evaluator 验收时，先投放对应 target：

```bash
export PARETO_BO_ORACLE_LEVEL=0
export L0_ADMET_ORACLE_TARGET=127.0.0.1:50056
export PARETO_BO_ORACLE_PROPERTIES=clearance
```

再重复上述 PCBO 命令。

## 9. Kubernetes/Helm 集群发布验证

本地服务栈通过后，再做 H10 集群发布。以下命令只验证 manifest 渲染和资源应用，不替代集群中的 readiness 观察。

Helm 模板渲染：

```bash
helm template moleculeforge infra/helm/moleculeforge > /tmp/moleculeforge-rendered.yaml
```

Kubernetes server-side dry-run：

```bash
kubectl apply --dry-run=server -f infra/kubernetes/namespaces/
kubectl apply --dry-run=server -f infra/kubernetes/deployments/moleculeforge-services.yaml
kubectl apply --dry-run=server -f /tmp/moleculeforge-rendered.yaml
```

实际发布：

```bash
kubectl apply -f infra/kubernetes/namespaces/
helm upgrade --install moleculeforge infra/helm/moleculeforge
```

观察状态：

```bash
kubectl get pods -A | grep -E 'mf-|moleculeforge' || true
kubectl get svc -A | grep -E 'mf-|moleculeforge' || true
kubectl rollout status deployment -n mf-agents --all
kubectl rollout status deployment -n mf-generators --all
kubectl rollout status deployment -n mf-oracles --all
```

期望：

- 所有目标 deployment rollout 成功。
- artifact path、ConfigMap、Secret 挂载与 `.env`/Helm values 一致。
- 集群中再次运行 H5/H6/W12 smoke 或对应 service health，不只看 pod Running。

## 10. 最终通过矩阵

全流程跑通时，以下命令均应有退出码 0：

```bash
uv run ruff check .
uv run ruff format --check .
bash tools/codegen/check_proto_sync.sh
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/unit -q
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/integration/test_dki_*.py -q
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/e2e/test_predict_api.py -q
RUN_KRAS_G12C_E2E=1 KRAS_E2E_SCOPE=engineering ORCHESTRATOR_E2E_READY=1 PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/e2e/test_kras_g12c_pilot.py -q
RUN_AUDIT_E2E=1 PROVENANCE_STORE_MODE=production_real PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/e2e/test_audit_completeness.py -q
RUN_KRAS_G12C_E2E=1 KRAS_E2E_SCOPE=full CRITIC_AGENT_READY=1 ORCHESTRATOR_E2E_READY=1 PROVENANCE_STORE_MODE=production_real PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/e2e/test_kras_g12c_pilot.py -q
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/benchmark -q -ra
```

验收记录必须包含：

- 命令原文。
- 退出码。
- pytest 通过/失败/skip 数。
- 失败时的原始 stderr 或 pytest failure 摘要。
- 未完成 gate 的明确原因。

## 11. 清理命令

停止本地 dev 服务栈：

```bash
docker compose -f infra/docker/docker-compose.dev.yml down
```

停止 DKI 栈但保留数据卷：

```bash
docker compose -f infra/docker/docker-compose.dki.yaml down
```

清理 DKI 数据卷。仅在确认不需要保留本地验证数据时执行：

```bash
docker compose -f infra/docker/docker-compose.dki.yaml down -v
```
