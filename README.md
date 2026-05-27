# MoleculeForge

> **End-to-end molecular inverse design platform**
> Monorepo · 微服务 · 协议优先 · 插件化 · 共享内核

This is the code skeleton scaffolded from `MoleculeForge_CodeArchitecture.md`. Follow `IMPLEMENTATION_GUIDE.md` for the recommended order of fleshing out each module.

## Quick Start

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install all workspace dependencies into ./.venv
uv sync --all-extras --all-packages

# 3. Run lint + tests
make lint
make test-unit
make test-e2e   # smoke tests for the prediction API

# 4. Boot the API gateway + UI on port 8000
.venv/bin/python -m uvicorn api_gateway.main:app --host 0.0.0.0 --port 8000

# 5. Open the dashboard
#   http://localhost:8000/
```

The API gateway hosts the UI at `/`, OpenAPI at `/docs`, and JSON endpoints
under `/v1/*`. The first call may take a few seconds while the multi-GPU
prediction engine warms up.

## DKI Integration Test Environment

Run this shell setup before integration tests that require the bare-metal DKI
stack at `/workspace/mf-dki-bare`:

```bash
set -a
source /workspace/mf-dki-bare/.env
set +a

export TEST_DATABASE_URL="postgresql+asyncpg://${PG_USER}:${PG_PASSWORD}@127.0.0.1:${PG_PORT}/${PG_DB}"
export PROVENANCE_DATABASE_URL="$TEST_DATABASE_URL"

export NEO4J_URI="bolt://127.0.0.1:${NEO4J_BOLT_PORT}"
export QDRANT_URL="http://127.0.0.1:${QDRANT_HTTP_PORT}"
export QDRANT_HOST="127.0.0.1"
export QDRANT_HTTP_PORT="${QDRANT_HTTP_PORT}"

export MINIO_ENDPOINT_URL="http://127.0.0.1:${MINIO_API_PORT}"
export MINIO_ACCESS_KEY="${MINIO_USER}"
export MINIO_SECRET_KEY="${MINIO_PASSWORD}"
export MINIO_BUCKET="mf-integration"

export REDIS_URL="redis://:${REDIS_PASSWORD}@127.0.0.1:${REDIS_PORT}/0"
export REDIS_HOST="127.0.0.1"
export REDIS_PORT="${REDIS_PORT}"

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
```

## Repository Layout

| Directory | Purpose |
|-----------|---------|
| `protos/`      | Protocol-first gRPC contracts (Buf) |
| `schemas/`     | JSON Schemas (CIG, CRG, SSP, audit) |
| `libs/`        | Shared kernel: types, plugin ABCs, manifolds, telemetry |
| `models/`      | ML implementations: 8 generators, 5 oracles, 3 retrosyn, 4 encoders |
| `services/`    | 14 microservices (gRPC + REST) |
| `agents/`      | 8 LangGraph-driven agents |
| `pipelines/`   | Batch / training pipelines |
| `data/`        | Ingestion, processing, validation, DVC |
| `configs/`     | Hydra YAML configs |
| `infra/`       | Docker, Kubernetes, Helm, Terraform |
| `tests/`       | unit / integration / e2e / benchmark |
| `docs/`        | Architecture, API, tutorials, ADRs |
| `tools/`       | Code generators, lint plugins, benchmarks |
| `ui/`, `wetlab/`, `commercial/` | Reserved modules (interfaces only) |

## Hardware Profile

This deployment is currently provisioned for **4 × NVIDIA H200 (141 GB HBM3e)**.
GPU resource requests in `infra/kubernetes/services/*` and `infra/helm/charts/*` are tuned for that profile — see `IMPLEMENTATION_GUIDE.md` for sharding strategies across services.

## License

TBD.
