# MoleculeForge

MoleculeForge is a molecular inverse-design monorepo. The current codebase
combines a runnable FastAPI workbench with a broader protocol-first service
architecture for generation, scoring, retrosynthesis, supply assessment,
provenance, and wet-lab protocol compilation.

## Current Scope

The active runtime entry is `/workspace/MForge/moleculeforge`.

The first-screen workflow is a natural-language molecular design workbench:

```text
User intent
  -> api-gateway `/v1/reason/*`
  -> agents/orchestrator ReasoningPipeline
  -> nl2obj parser
  -> candidate generation
  -> mf-chem prediction engine
  -> constraint filtering
  -> known-molecule novelty lookup
  -> ranking
  -> SQLite persistence
  -> static UI events and result views
```

The broader service workflow is protocol-first:

```text
API Gateway
  -> Orchestrator service
  -> CIG compiler / generator router / generator services
  -> oracle services / validation agent / critic agent
  -> retrosynthesis / supply / SRB agents
  -> provenance service and storage adapters
```

## Repository Layout

```text
/workspace/MForge/
|-- README.md
|-- actions-runner/  Self-hosted runner state outside the package graph
|-- moleculeforge/
|   |-- agents/       Agent packages for orchestration, validation, supply, SRB, and related roles
|   |-- configs/      Runtime and service configuration
|   |-- data/         SQLite store, migrations, ingestion, validation, and processing code
|   |-- docs/         Project documentation outside this root architecture overview
|   |-- feature_repo/ Feast feature repository
|   |-- infra/        Docker Compose, Kubernetes, Helm, Terraform, and build scripts
|   |-- libs/         Shared libraries: core types, chemistry, HUMU, evaluation, telemetry, agents
|   |-- models/       Generator, oracle, retrosynthesis, and encoder packages
|   |-- pipelines/    Implemented training and batch pipelines
|   |-- protos/       Source protobuf contracts
|   |-- schemas/      JSON and OpenAPI schemas
|   |-- services/     FastAPI and service packages
|   |-- tests/        Unit, integration, e2e, benchmark, and anti-degradation tests
|   |-- tools/        Developer, wrapper, oracle, scorer, codegen, and linting tools
|   |-- ui/public/    Static workbench UI served by api-gateway
|   `-- wetlab/      XDL compiler path used by SRB
|-- openadmet-models/ Editable external OpenADMET project location
|-- openadmet_models/ Local model asset directory selected by OPENADMET_MODEL_DIR
|-- recycle bin/     Inactive archive, preserved with original path shape
`-- zzzzz/           Local source datasets outside active runtime boundaries
```

## Core Packages

`libs/mf-core` defines shared domain types, plugin interfaces, storage helpers,
generated protobuf modules, routing utilities, artifact checks, and database
integration.

`libs/mf-chem` owns molecule parsing and prediction. Its prediction engine
combines RDKit descriptors, HUMU embeddings, and ADMET outputs behind a common
batch interface used by the API gateway and reasoning workbench.

`libs/mf-humu` contains Lorentz geometry, HUMU encoders, Gaussian-process
utilities, intent-cone operations, and unfamiliarity logic.

`libs/mf-agents` contains shared agent base classes, Redis messaging, CRG graph
helpers, and lineage signing helpers.

`libs/mf-eval` contains evaluation utilities for molecule quality, hypervolume,
distortion, and activity-cliff analysis.

Evaluation paths that require RDKit must fail explicitly when RDKit is
unavailable. They must not emit fixed fallback scores or synthetic success
values for missing scientific dependencies.

`libs/mf-telemetry` contains tracing integration.

## Services

`services/api-gateway` is the primary HTTP entry. It mounts the static UI from
`ui/public`, initializes the SQLite-backed store, exposes prediction and
reasoning endpoints, and proxies selected orchestrator service requests.

Generator services expose HFM, FragFM, CREM, MMPT, iCLM, and router paths.
Oracle services expose ADMET, docking, Boltz2, FEP, HUMU index, feature store,
and supply-oracle paths. `cig-compiler-svc`, `orchestrator-svc`,
`provenance-svc`, `retrosyn-svc`, and `humu-encoder-svc` provide the core
protocol-driven service layer.

Services are independent packages. Cross-service behavior should go through
protocol clients, HTTP clients, gRPC clients, or explicit adapter functions
rather than direct service-to-service imports.

Service console entry points must target active application factories.

## Agents

`agents/nl2obj` parses natural-language intent into structured objectives.
`agents/orchestrator` owns the single-process reasoning workbench pipeline.
`agents/generator_coord` coordinates generator selection and dispatch.
`agents/validation_agent`, `critic_agent`, `retrosyn_agent`, `supply_agent`,
and `srb_agent` implement downstream validation, critique, route embedding,
supply feasibility, and structured synthesis protocol compilation.

`srb_agent` uses `wetlab/xdl-compiler` for SSP-to-XDL export. Real hardware
execution remains an external command boundary through `SILA2_PLAN_COMMAND`.

## Models

Generators live under `models/mf-generators`:

- `hfm_3d`
- `fragfm`
- `crem_3d`
- `incremental_clm`
- `mmpt_rag`
- `rdkit_random`
- `uas`

Oracles live under `models/mf-oracles`:

- `admet_ai`
- `boltz2`
- `diffdock_l`
- `gnina`
- `openfe`
- `rdkit-oracle`

Retrosynthesis models live under `models/mf-retrosyn`.
HUMU encoders live under `models/mf-encoders`.

Runtime model artifacts are local runtime assets and are not part of this
source workspace contract.

## Data Flow

The workbench path stores run state and results in SQLite through
`libs/mf-core/src/mf_core/db/store.py`. The default database path is
`moleculeforge/data/moleculeforge.db`, while tests override `MF_DB_PATH` to a
temporary database.

The reasoning pipeline emits structured stages:

```text
nl_parse
objectives
generation
scoring
constraint_filter
novelty
ranking
summary
```

Each stage is persisted and can be streamed to the UI through server-sent
events.

## Workspace

The Python workspace is managed by `uv` from `moleculeforge/pyproject.toml`.
Implemented workspace package groups are:

- `libs/*`
- `models/mf-generators/*`
- `models/mf-oracles/*`
- `models/mf-retrosyn/*`
- `models/mf-encoders/*`
- `services/*`
- `agents/*`
- implemented `pipelines/*`

Implemented pipeline packages include `humu_pretrain`, `mvp_pipeline`,
`reaction_indexing`, and `pareto_bo`.

`supply.proto` and generated protocol modules are active contracts. Protocol
field changes belong in the source protobuf definition and regenerated outputs
together.

## External Assets

`openadmet-models` is an editable external project location.
`openadmet_models` is the runtime model directory selected by
`OPENADMET_MODEL_DIR`. `actions-runner` is self-hosted runner state.

These directories are not MoleculeForge source packages. Runtime code should
reach external assets through configuration, environment variables, service
adapters, or CI runner control paths.

## Repository Boundaries

`/workspace/MForge/recycle bin` is an inactive archive with the same relative
path shape as the active tree. It is not an active source tree.

The following areas are outside the active source boundary:

- `/workspace/MForge/zzzzz`
- `/workspace/MForge/moleculeforge/docs`
- runtime caches, logs, checkpoints, runs, local virtual environments, and local model artifacts

## Code Rules

Keep a single active implementation for each feature. Do not add parallel
`v1`/`v2` replacements unless explicitly requested.

Use existing package boundaries and shared helpers before introducing new
abstractions. Do not add placeholder packages, roadmap-only directories, or
README-only modules to the active source tree.

Missing external resources should fail clearly through configuration or adapter
preflight rather than falling back to fake success.
