# MoleculeForge · 完整代码工程架构文档

> **版本**：v1.0 (代码工程架构)
> **日期**：2026-04-29
> **定位**：端到端分子逆向设计平台的**代码仓库工程蓝图**
> **设计哲学**：Monorepo + 微服务 + 协议优先 + 插件化 + 共享内核

---

## 目录

1. [架构设计原则](#1-架构设计原则)
2. [仓库整体结构](#2-仓库整体结构)
3. [顶层目录详解](#3-顶层目录详解)
4. [protos · 协议定义层](#4-protos--协议定义层)
5. [libs · 共享内核库](#5-libs--共享内核库)
6. [services · 微服务实现](#6-services--微服务实现)
7. [agents · 智能体实现](#7-agents--智能体实现)
8. [models · 模型实现库](#8-models--模型实现库)
9. [data · 数据管线](#9-data--数据管线)
10. [infra · 基础设施](#10-infra--基础设施)
11. [tests · 测试体系](#11-tests--测试体系)
12. [docs · 文档系统](#12-docs--文档系统)
13. [tools · 开发工具](#13-tools--开发工具)
14. [预留扩展目录](#14-预留扩展目录)
15. [关键文件深度解析](#15-关键文件深度解析)
16. [开发工作流](#16-开发工作流)
17. [插件化扩展指南](#17-插件化扩展指南)

---

## 1. 架构设计原则

在动手设计目录之前，先确立 **8 条不可妥协的工程原则**，每条都直接影响后续的目录组织：

### 1.1 Monorepo + 多包并行

**为什么不用多 Repo**：MoleculeForge 有 8 个生成器、8 个 Agent、5 级 Oracle、20+ 微服务，如果每个独立 Repo，跨服务重构和版本同步会成为噩梦。

**为什么不是单包巨石**：单一 Python 包无法支持不同服务的不同依赖（如 Boltz-2 需要 CUDA 12.4，AiZynthFinder 需要 RDKit 2024.09，可能冲突）。

**最佳折中**：**Monorepo + Workspace**
- 仓库根目录用 `pyproject.toml` workspace 模式
- 每个服务/库是独立的 Python 包，有自己的依赖
- 共享内核（`libs/`）可被所有服务引用
- 用 `uv workspace` 或 `Pants` 构建系统管理

### 1.2 协议优先（Protocol-First）

所有微服务间的通信、数据格式、Agent 消息**先定义协议**，再写实现：
- Protobuf 定义服务间 gRPC 接口（强类型、跨语言、高性能）
- JSON Schema 定义 REST API 和文件格式（CIG、CRG、SSP）
- OpenAPI 3.1 描述对外的 RESTful API
- AsyncAPI 2.6 描述消息总线异步事件

**好处**：协议变更立即被所有依赖方感知；Mock 服务能基于协议自动生成；前后端可并行开发。

### 1.3 共享内核（Shared Kernel）

把**所有服务都需要的核心抽象**抽取到 `libs/` 中：
- `mf-core`：CIG/CRG/HCIV 等数据结构
- `mf-humu`：双曲流形数学库（多个生成器都要用）
- `mf-chem`：化学操作的统一接口（避免每个服务都直接调 RDKit）
- `mf-agents`：Agent 协议、消息总线封装

**关键原则**：共享内核**只放稳定的、无副作用的、跨服务通用的代码**。业务逻辑绝不放进去。

### 1.4 插件化（Plugin Architecture）

8 个生成器、5 级 Oracle、N 种 Agent 都遵循**统一的插件接口**：
- 抽象基类定义在 `libs/mf-core/plugins/`
- 每个具体实现作为独立插件包
- 通过 entry-points 注册（Python `importlib.metadata`）
- 配置文件指定使用哪个插件

**好处**：未来新增生成器（比如某个 NeurIPS 2027 的新 SOTA）只需写一个新插件包，不用动核心代码。

### 1.5 三层依赖纪律

```
┌──────────────────────────────────────┐
│  apps/services/agents (业务层)       │  ← 可依赖下两层
├──────────────────────────────────────┤
│  models, data (领域层)               │  ← 可依赖 libs，不能依赖 services
├──────────────────────────────────────┤
│  libs (共享内核)                     │  ← 不能依赖任何上层
└──────────────────────────────────────┘
```

CI 用 `import-linter` 强制检查，违反就拒绝合并。

### 1.6 配置与代码分离

所有可调参数（模型超参、oracle 阈值、Agent 行为）都放在 YAML 配置中，由 `Hydra` 管理：
- 默认配置：`configs/`
- 环境覆盖：`configs/env/{dev,staging,prod}.yaml`
- 实验配置：`configs/experiments/`
- 运行时合成：用 OmegaConf 动态合成

### 1.7 Docker 镜像分层

```
mf-base-image          # CUDA + Python + 通用依赖
   ├── mf-chem-image    # + RDKit + OpenBabel
   │     ├── mf-generator-image  # + 生成模型依赖
   │     └── mf-oracle-image     # + 对接/FEP 依赖
   └── mf-agent-image   # + LangGraph + LLM SDK
```

每个微服务的 Dockerfile 继承对应基础镜像，避免重复构建。

### 1.8 三个预留目录

按你的要求，前端、湿实验室、商业化保留**完整的目录占位**和**接口定义文件**，但内部只有 `README.md` 说明"待实施"。这样核心架构完成后，扩展团队可以立即上手。

---

## 2. 仓库整体结构

```
moleculeforge/
├── README.md                          # 项目总览
├── LICENSE                            # 开源协议
├── pyproject.toml                     # Workspace 根配置
├── uv.lock                            # 依赖锁文件
├── Makefile                           # 常用命令快捷方式
├── .gitignore
├── .editorconfig
├── .pre-commit-config.yaml            # Git pre-commit 钩子
├── .github/
│   ├── workflows/                     # CI/CD 流水线
│   │   ├── ci-lint.yml
│   │   ├── ci-test.yml
│   │   ├── ci-build-images.yml
│   │   └── release.yml
│   └── CODEOWNERS                     # 各模块负责人
│
├── protos/                            # 【协议定义层】
│   ├── README.md
│   ├── buf.yaml
│   ├── buf.gen.yaml
│   └── moleculeforge/v1/
│       ├── core/                      # 核心数据类型
│       ├── agent/                     # Agent 通信协议
│       ├── generator/                 # 生成器 gRPC
│       ├── oracle/                    # Oracle gRPC
│       ├── retrosyn/                  # 逆合成 gRPC
│       └── humu/                      # HUMU 编码 gRPC
│
├── schemas/                           # 【JSON Schema 定义】
│   ├── cig.schema.json                # Chemical Intent Graph
│   ├── crg.schema.json                # Chemical Reasoning Graph
│   ├── ssp.schema.json                # Structured Synthesis Protocol
│   ├── audit_message.schema.json      # 审计消息格式
│   └── openapi/
│       ├── public-api.v1.yaml
│       └── internal-api.v1.yaml
│
├── libs/                              # 【共享内核】
│   ├── mf-core/                       # 核心数据结构与接口
│   ├── mf-humu/                       # 双曲流形数学库
│   ├── mf-chem/                       # 化学操作封装
│   ├── mf-agents/                     # Agent 框架基础设施
│   ├── mf-eval/                       # 评估指标库
│   └── mf-telemetry/                  # 日志/监控/追踪
│
├── models/                            # 【模型实现库】
│   ├── README.md
│   ├── mf-generators/                 # 8 个生成器实现
│   │   ├── hfm_3d/
│   │   ├── fragfm/
│   │   ├── lamgen_3d/
│   │   ├── crem_3d/
│   │   ├── mmpt_rag/
│   │   ├── evomol_rl/
│   │   ├── incremental_clm/
│   │   └── uas/
│   ├── mf-oracles/                    # Oracle 实现
│   │   ├── boltz2/
│   │   ├── diffdock_l/
│   │   ├── gnina/
│   │   ├── openfe/
│   │   └── admet_ai/
│   ├── mf-retrosyn/                   # 逆合成模型
│   │   ├── aizynth_wrapper/
│   │   ├── rsgpt/
│   │   └── ualign/
│   └── mf-encoders/                   # HUMU 编码器
│       ├── humu_mol_encoder/
│       ├── humu_pocket_encoder/
│       └── humu_route_encoder/
│
├── services/                          # 【微服务实现】
│   ├── README.md
│   ├── humu-encoder-svc/
│   ├── generator-router-svc/
│   ├── retrosyn-svc/
│   ├── boltz2-svc/
│   ├── dock-svc/
│   ├── fep-svc/
│   ├── admet-svc/
│   ├── fto-patent-svc/
│   ├── supply-oracle-svc/
│   ├── humu-index-svc/
│   ├── provenance-svc/
│   ├── feature-store-svc/
│   ├── cig-compiler-svc/
│   └── api-gateway/
│
├── agents/                            # 【智能体实现】
│   ├── README.md
│   ├── orchestrator/
│   ├── nl2obj/
│   ├── generator_coord/
│   ├── retrosyn_agent/
│   ├── validation_agent/
│   ├── fto_agent/
│   ├── supply_agent/
│   └── critic_agent/
│
├── pipelines/                         # 【批量计算管线】
│   ├── README.md
│   ├── humu_pretrain/                 # HUMU 联合预训练
│   ├── generator_finetune/            # 生成器在线微调
│   ├── patent_indexing/               # 专利数据库索引构建
│   ├── reaction_indexing/             # 反应模板索引
│   ├── boltz2_eval/                   # Boltz-2 离线评估
│   └── pareto_bo/                     # Pareto 贝叶斯优化主循环
│
├── data/                              # 【数据管线与数据】
│   ├── README.md
│   ├── ingestion/                     # 数据接入
│   ├── processing/                    # 数据预处理
│   ├── validation/                    # 数据质量检查
│   └── samples/                       # 单元测试用小样本
│
├── configs/                           # 【全局配置】
│   ├── README.md
│   ├── default.yaml
│   ├── env/
│   ├── models/
│   ├── services/
│   ├── agents/
│   └── experiments/
│
├── infra/                             # 【基础设施代码】
│   ├── README.md
│   ├── docker/                        # Dockerfile
│   ├── kubernetes/                    # K8s 资源
│   ├── helm/                          # Helm Charts
│   ├── terraform/                     # 云资源 IaC
│   └── scripts/                       # 运维脚本
│
├── tests/                             # 【测试体系】
│   ├── README.md
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   ├── benchmark/
│   └── fixtures/
│
├── docs/                              # 【文档系统】
│   ├── README.md
│   ├── architecture/
│   ├── api/
│   ├── tutorials/
│   ├── adr/                           # 架构决策记录
│   └── papers/                        # 内部技术报告
│
├── tools/                             # 【开发工具】
│   ├── codegen/                       # 代码生成器
│   ├── linting/                       # 自定义 lint 规则
│   └── benchmarks/                    # 性能基准工具
│
├── ui/                                # 🔲【预留：前端】
│   └── README.md
├── wetlab/                            # 🔲【预留：湿实验室】
│   └── README.md
└── commercial/                        # 🔲【预留：商业化】
    └── README.md
```

---

## 3. 顶层目录详解

### 3.1 根目录核心文件

#### `pyproject.toml`（Workspace 根配置）

```toml
[project]
name = "moleculeforge"
version = "0.1.0"
description = "End-to-end molecular inverse design platform"
requires-python = ">=3.11,<3.13"

[tool.uv.workspace]
members = [
    "libs/*",
    "models/mf-generators/*",
    "models/mf-oracles/*",
    "models/mf-retrosyn/*",
    "models/mf-encoders/*",
    "services/*",
    "agents/*",
    "pipelines/*",
]

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "I", "N", "W", "B", "UP", "ANN", "S"]

[tool.mypy]
strict = true
python_version = "3.11"

[tool.import-linter]
# 强制三层依赖纪律：services 不能直接依赖其他 services
[[tool.import-linter.contracts]]
name = "Service Independence"
type = "forbidden"
source_modules = ["services.*"]
forbidden_modules = ["services.*"]
ignore_imports = ["services.*.client.*"]  # 客户端 SDK 例外
```

**关键作用**：
- 定义所有子包的 workspace 关系
- 统一 lint/format/type-check 配置
- 通过 `import-linter` 强制架构纪律

#### `Makefile`（开发命令快捷方式）

```makefile
.PHONY: install lint test build run-dev clean

# 安装所有依赖
install:
	uv sync --all-extras

# 全仓库 lint
lint:
	uv run ruff check .
	uv run mypy libs/ services/ agents/ models/
	uv run import-linter

# 运行测试
test:
	uv run pytest tests/ -n auto

test-unit:
	uv run pytest tests/unit -n auto

test-integration:
	docker-compose -f infra/docker/docker-compose.test.yml up -d
	uv run pytest tests/integration
	docker-compose -f infra/docker/docker-compose.test.yml down

# 构建协议
proto-gen:
	cd protos && buf generate

# 构建所有 Docker 镜像
build-images:
	bash infra/scripts/build_all_images.sh

# 本地启动核心服务
run-dev:
	docker-compose -f infra/docker/docker-compose.dev.yml up

# 数据库迁移
db-migrate:
	uv run alembic -c data/alembic.ini upgrade head

# 清理
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
```

#### `.pre-commit-config.yaml`（Git 钩子）

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic, types-requests]
  - repo: local
    hooks:
      - id: import-linter
        name: Check import architecture
        entry: uv run import-linter
        language: system
        pass_filenames: false
      - id: proto-check
        name: Check protobuf consistency
        entry: bash tools/codegen/check_proto_sync.sh
        language: system
        pass_filenames: false
```

---

## 4. protos · 协议定义层

> **职责**：所有跨服务通信的契约，先于实现

### 4.1 目录结构

```
protos/
├── README.md
├── buf.yaml                           # Buf 配置（lint + breaking change 检查）
├── buf.gen.yaml                       # 代码生成配置
└── moleculeforge/v1/
    ├── core/
    │   ├── molecule.proto              # Molecule, SMILES, InChIKey
    │   ├── humu.proto                  # HCIV, IntentCone, HUMU 嵌入
    │   ├── cig.proto                   # Chemical Intent Graph
    │   ├── crg.proto                   # Chemical Reasoning Graph
    │   ├── ssp.proto                   # Structured Synthesis Protocol
    │   ├── pareto.proto                # Pareto 前沿、HVI
    │   └── audit.proto                 # 审计消息
    ├── agent/
    │   ├── message.proto               # Agent 间消息基础结构
    │   ├── orchestrator.proto          # Orchestrator 服务接口
    │   └── critic.proto                # Critic 服务接口
    ├── generator/
    │   ├── generator.proto             # 生成器统一接口
    │   └── router.proto                # TAR 路由器接口
    ├── oracle/
    │   ├── oracle.proto                # Oracle 统一接口
    │   ├── boltz2.proto                # Boltz-2 专用扩展
    │   └── fep.proto                   # FEP 专用扩展
    ├── retrosyn/
    │   ├── retrosyn.proto              # 逆合成统一接口
    │   └── route.proto                 # 反应树、合成路径
    └── humu/
        └── encoder.proto                # HUMU 编码器接口
```

### 4.2 关键 Proto 文件示例

#### `core/molecule.proto`

```protobuf
syntax = "proto3";
package moleculeforge.v1.core;

import "google/protobuf/timestamp.proto";
import "moleculeforge/v1/core/humu.proto";

// 分子的统一表示
message Molecule {
  string id = 1;                            // UUID
  string smiles = 2;                        // SMILES 字符串
  string canonical_smiles = 3;
  string inchikey = 4;
  
  // 物化性质
  MolecularProperties properties = 10;
  
  // HUMU 嵌入
  HCIV humu_embedding = 20;
  float unfamiliarity_score = 21;          // OOD 分数
  
  // 来源信息
  string generator_name = 30;               // FragFM/HFM-3D/...
  string generator_version = 31;
  uint64 generation_seed = 32;
  
  // 谱系
  string run_id = 40;
  string parent_molecule_id = 41;          // 如果是从某分子变异而来
  
  google.protobuf.Timestamp created_at = 50;
}

message MolecularProperties {
  float molecular_weight = 1;
  float logp = 2;
  uint32 hbd = 3;
  uint32 hba = 4;
  float tpsa = 5;
  float qed = 6;
  float sa_score = 7;
  bytes ecfp4 = 8;                         // 1024-bit fingerprint
}
```

#### `generator/generator.proto`（生成器统一接口）

```protobuf
syntax = "proto3";
package moleculeforge.v1.generator;

import "moleculeforge/v1/core/molecule.proto";
import "moleculeforge/v1/core/humu.proto";
import "moleculeforge/v1/core/cig.proto";

// 所有生成器必须实现这个接口
service GeneratorService {
  // 单次生成
  rpc Generate(GenerateRequest) returns (GenerateResponse);
  
  // 流式生成（边生成边推送）
  rpc GenerateStream(GenerateRequest) returns (stream GeneratedMolecule);
  
  // 批量生成
  rpc BatchGenerate(BatchGenerateRequest) returns (BatchGenerateResponse);
  
  // 健康检查 + 模型信息
  rpc Info(InfoRequest) returns (InfoResponse);
}

message GenerateRequest {
  // 生成上下文
  moleculeforge.v1.core.HCIV intent_vector = 1;
  moleculeforge.v1.core.IntentCone intent_cone = 2;
  moleculeforge.v1.core.CIG cig = 3;
  
  // 生成参数
  uint32 n_samples = 10;
  uint32 max_atoms = 11;
  float temperature = 12;
  uint64 seed = 13;
  
  // 约束
  repeated string forbidden_substructures = 20;
  
  // 元数据
  string trace_id = 90;                    // 分布式追踪
  string run_id = 91;
}

message GenerateResponse {
  repeated GeneratedMolecule molecules = 1;
  GenerationStats stats = 2;
}

message GeneratedMolecule {
  moleculeforge.v1.core.Molecule molecule = 1;
  float generation_log_prob = 2;
  GenerationMetadata metadata = 3;
}

message GenerationStats {
  uint32 n_generated = 1;
  uint32 n_valid = 2;
  uint32 n_unique = 3;
  float wallclock_seconds = 4;
}

message InfoResponse {
  string generator_name = 1;
  string version = 2;
  string model_checkpoint = 3;
  repeated string supported_modes = 4;     // hit_finding/lead_opt/scaffold_hop
  ResourceRequirements resources = 5;
}
```

### 4.3 Proto 工作流

```bash
# 1. 修改 .proto 文件
vim protos/moleculeforge/v1/generator/generator.proto

# 2. Lint 检查
cd protos && buf lint

# 3. Breaking change 检查（防止破坏现有调用方）
buf breaking --against '.git#branch=main'

# 4. 生成 Python/Go/TS 代码
buf generate

# 5. 生成的代码自动放到 libs/mf-core/src/mf_core/proto_gen/
```

---

## 5. libs · 共享内核库

> **职责**：所有服务都依赖的核心抽象。**只放稳定、纯粹、跨服务通用的代码**

### 5.1 libs/mf-core · 核心数据结构与接口

```
libs/mf-core/
├── pyproject.toml
├── README.md
└── src/mf_core/
    ├── __init__.py
    ├── proto_gen/                      # 由 protos/ 自动生成（不要手动改）
    │   ├── molecule_pb2.py
    │   ├── molecule_pb2_grpc.py
    │   └── ...
    ├── types/                          # Pydantic 数据模型
    │   ├── __init__.py
    │   ├── molecule.py                 # MoleculeModel
    │   ├── cig.py                      # ChemicalIntentGraph
    │   ├── crg.py                      # ChemicalReasoningGraph  
    │   ├── hciv.py                     # HCIV
    │   ├── ssp.py                      # StructuredSynthesisProtocol
    │   ├── pareto.py                   # ParetoArchive, ParetoSolution
    │   └── audit.py                    # AuditMessage
    ├── plugins/                        # 插件抽象基类
    │   ├── __init__.py
    │   ├── base.py                     # BasePlugin
    │   ├── generator.py                # BaseGenerator (ABC)
    │   ├── oracle.py                   # BaseOracle (ABC)
    │   ├── retrosyn.py                 # BaseRetrosynModel (ABC)
    │   └── encoder.py                  # BaseEncoder (ABC)
    ├── registry/                       # 插件注册中心
    │   ├── __init__.py
    │   └── plugin_registry.py
    ├── exceptions/                     # 统一异常体系
    │   ├── __init__.py
    │   ├── core.py                     # MoleculeForgeError 基类
    │   ├── generation.py
    │   ├── validation.py
    │   └── synthesis.py
    └── utils/
        ├── ids.py                      # UUID/Trace ID 生成
        ├── hashing.py                  # 内容哈希（用于审计）
        └── time.py                     # 时间戳工具
```

#### 关键文件：`plugins/generator.py`

```python
"""
所有生成器必须实现的抽象基类。
新增生成器 = 继承此类 + 通过 entry-points 注册。
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator
from mf_core.types.molecule import MoleculeModel
from mf_core.types.cig import ChemicalIntentGraph
from mf_core.types.hciv import HCIV, IntentCone


class BaseGenerator(ABC):
    """所有分子生成器的统一接口。
    
    实现要点：
    - 必须支持异步流式生成（大批量时不阻塞）
    - 必须接收 HCIV + IntentCone 作为条件
    - 必须返回包含 generation_log_prob 的分子（供路由器决策）
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """生成器唯一标识，用于 TAR 路由"""
        ...
    
    @property
    @abstractmethod
    def version(self) -> str:
        """模型权重版本，用于审计"""
        ...
    
    @property
    @abstractmethod
    def supported_modes(self) -> list[str]:
        """声明此生成器擅长的任务模式
        Examples: ["hit_finding", "lead_opt", "scaffold_hop", "multi_target"]
        """
        ...
    
    @abstractmethod
    async def generate(
        self,
        intent_vector: HCIV,
        intent_cone: IntentCone,
        cig: ChemicalIntentGraph,
        n_samples: int = 100,
        seed: int | None = None,
    ) -> AsyncIterator[MoleculeModel]:
        """生成分子，逐个产出（生成器/异步迭代器）。
        
        实现者必须：
        1. 在 IntentCone 内采样（不要超出约束区域）
        2. 每个分子附带 humu_embedding 和 generation_log_prob
        3. 处理 cancellation（asyncio.CancelledError）以支持及时停止
        """
        ...
    
    @abstractmethod
    async def health_check(self) -> dict:
        """健康检查，返回模型加载状态、GPU 可用性等"""
        ...
```

#### 关键文件：`registry/plugin_registry.py`

```python
"""
插件注册中心，使用 importlib.metadata 的 entry points。

新增生成器只需在其 pyproject.toml 中声明：

[project.entry-points."moleculeforge.generators"]
hfm_3d = "mf_generators.hfm_3d:HFM3DGenerator"
fragfm = "mf_generators.fragfm:FragFMGenerator"
"""
from importlib.metadata import entry_points
from typing import TypeVar
from mf_core.plugins.generator import BaseGenerator
from mf_core.plugins.oracle import BaseOracle


T = TypeVar("T")


class PluginRegistry:
    """运行时发现并加载所有插件"""
    
    @staticmethod
    def load_generators() -> dict[str, type[BaseGenerator]]:
        """加载所有注册的生成器"""
        eps = entry_points(group="moleculeforge.generators")
        return {ep.name: ep.load() for ep in eps}
    
    @staticmethod
    def load_oracles() -> dict[str, type[BaseOracle]]:
        eps = entry_points(group="moleculeforge.oracles")
        return {ep.name: ep.load() for ep in eps}
    
    @staticmethod
    def get_generator(name: str) -> type[BaseGenerator]:
        gens = PluginRegistry.load_generators()
        if name not in gens:
            raise ValueError(f"Generator {name} not registered. Available: {list(gens.keys())}")
        return gens[name]
```

### 5.2 libs/mf-humu · 双曲流形数学库

```
libs/mf-humu/
├── pyproject.toml
├── README.md
└── src/mf_humu/
    ├── __init__.py
    ├── manifold/                       # Lorentz 流形数学
    │   ├── __init__.py
    │   ├── lorentz.py                  # Lorentz 模型核心运算
    │   ├── geodesic.py                 # 测地线、距离
    │   ├── exp_log.py                  # exp_map / log_map
    │   └── parallel_transport.py
    ├── encoders/                       # 编码器组件
    │   ├── __init__.py
    │   ├── base.py                     # BaseHUMUEncoder
    │   ├── lorentz_proj.py             # 切空间→流形投影层
    │   └── lorentz_attention.py        # 双曲注意力
    ├── operations/                     # 高级操作
    │   ├── __init__.py
    │   ├── intent_cone.py              # 意图锥定义和采样
    │   ├── dead_zone.py                # Patent Dead Zone 障碍势能
    │   ├── cliff_detection.py          # Activity Cliff 检测
    │   └── unfamiliarity.py            # OOD 不熟悉度
    ├── gp/                             # 双曲高斯过程
    │   ├── __init__.py
    │   ├── kernels.py                  # Matérn on geodesic distance
    │   ├── svgp.py                     # 稀疏变分 GP
    │   └── ehvi.py                     # EHVI 采集函数
    └── utils/
        ├── numerical.py                # 数值稳定的双曲运算
        └── visualization.py            # 双曲空间可视化（Poincaré 投影）
```

#### 关键文件：`manifold/lorentz.py`

```python
"""
Lorentz 模型的核心数学运算。

Lorentz 流形定义：
  H^d = {x ∈ R^{d+1} : <x, x>_L = -1, x_0 > 0}
  其中 <x, y>_L = -x_0*y_0 + sum(x_i * y_i for i=1..d)
"""
import torch
from torch import Tensor


class LorentzManifold:
    """Lorentz 双曲流形。
    
    所有运算在 PyTorch tensor 上，支持 autograd 和 GPU。
    """
    
    def __init__(self, curvature: float = 1.0, eps: float = 1e-7):
        self.c = curvature
        self.eps = eps  # 数值稳定性
    
    def inner(self, x: Tensor, y: Tensor, keepdim: bool = True) -> Tensor:
        """Lorentzian 内积 <x, y>_L = -x_0*y_0 + sum(x_i*y_i)"""
        prod = x * y
        prod[..., 0] = -prod[..., 0]
        return prod.sum(dim=-1, keepdim=keepdim)
    
    def distance(self, x: Tensor, y: Tensor) -> Tensor:
        """测地线距离 d(x, y) = arcosh(-<x, y>_L / c)"""
        inner = self.inner(x, y).squeeze(-1)
        # 数值稳定：clip to >= 1.0 + eps
        arg = torch.clamp(-inner / self.c, min=1.0 + self.eps)
        return torch.acosh(arg) * self.c**0.5
    
    def expmap(self, base: Tensor, tangent: Tensor) -> Tensor:
        """指数映射：从切空间投影到流形
        
        exp_x(v) = cosh(|v|) * x + sinh(|v|) * v / |v|
        """
        norm = torch.sqrt(torch.clamp(self.inner(tangent, tangent), min=self.eps))
        result = torch.cosh(norm) * base + torch.sinh(norm) * tangent / (norm + self.eps)
        return self._project(result)  # 数值修正回流形
    
    def logmap(self, base: Tensor, point: Tensor) -> Tensor:
        """对数映射：从流形投影到切空间
        
        log_x(y) = d(x, y) * (y - <x, y>_L * x) / |y - <x, y>_L * x|
        """
        inner = self.inner(base, point)
        diff = point + inner * base  # 注意 Lorentz 内积的符号
        norm = torch.sqrt(torch.clamp(self.inner(diff, diff), min=self.eps))
        dist = self.distance(base, point).unsqueeze(-1)
        return dist * diff / (norm + self.eps)
    
    def _project(self, x: Tensor) -> Tensor:
        """数值修正：把可能漂移的点投影回 Lorentz 流形"""
        # 强制 <x, x>_L = -1
        spatial = x[..., 1:]
        time_sq = 1.0 / self.c + (spatial ** 2).sum(dim=-1, keepdim=True)
        x_proj = torch.cat([torch.sqrt(time_sq), spatial], dim=-1)
        return x_proj
```

### 5.3 libs/mf-chem · 化学操作封装

```
libs/mf-chem/
├── pyproject.toml
├── README.md
└── src/mf_chem/
    ├── __init__.py
    ├── molecule/
    │   ├── parsing.py                  # SMILES/InChI 解析（统一入口）
    │   ├── canonicalization.py         # 规范化
    │   ├── descriptors.py              # 物化性质
    │   ├── conformers.py               # 3D 构象生成
    │   └── fingerprints.py             # ECFP/MACCS/...
    ├── pharmacophore/
    │   ├── extraction.py               # 从分子提取药效团
    │   ├── matching.py                 # 3D 药效团匹配
    │   └── alignment.py                # 药效团对齐
    ├── reaction/
    │   ├── parsing.py                  # 反应 SMARTS 解析
    │   ├── tree.py                     # 反应树数据结构（AND-OR 图）
    │   ├── enumeration.py              # 反应枚举
    │   └── templates.py                # 反应模板管理
    ├── filters/                        # 化学有效性过滤
    │   ├── pains.py                    # PAINS 过滤
    │   ├── reactive.py                 # 反应基团过滤
    │   ├── lipinski.py
    │   └── leadlike.py
    └── adapters/                       # 第三方库适配器
        ├── rdkit_adapter.py
        ├── openbabel_adapter.py
        └── openff_adapter.py
```

**为什么要这一层**：
- 所有服务都需要分子操作，但**直接调 RDKit 会让代码强耦合 RDKit 版本**
- 通过 `mf-chem` 适配器，未来切换底层库（比如换成更快的 OEChem）只需改一处
- 统一处理分子规范化（避免 SMILES 字符串大小写/顺序导致的去重失败）

### 5.4 libs/mf-agents · Agent 框架基础设施

```
libs/mf-agents/
├── pyproject.toml
├── README.md
└── src/mf_agents/
    ├── __init__.py
    ├── base/
    │   ├── agent.py                    # BaseAgent
    │   ├── tool.py                     # BaseTool
    │   └── memory.py                   # 短期记忆 / 长期记忆抽象
    ├── messaging/
    │   ├── nats_bus.py                 # NATS JetStream 封装
    │   ├── message_envelope.py         # 消息封装（含签名）
    │   └── routing.py                  # 主题路由
    ├── lineage/
    │   ├── tracker.py                  # 谱系追踪器
    │   ├── sigstore_signer.py          # Sigstore 签名
    │   └── neo4j_logger.py             # 写入 Neo4j Provenance
    ├── llm/
    │   ├── client.py                   # LLM 调用封装（多 Provider）
    │   ├── claude_provider.py
    │   ├── deepseek_provider.py
    │   ├── prompt_template.py
    │   └── retry.py                    # 智能重试 + 降级
    ├── crg/                            # Chemical Reasoning Graph
    │   ├── graph.py                    # CRG 数据结构
    │   ├── belief.py                   # 信念节点
    │   ├── conflict.py                 # 冲突检测
    │   └── persistence.py              # 持久化到 Neo4j
    └── workflow/
        ├── langgraph_helpers.py        # LangGraph 辅助
        └── state_machine.py            # 状态机框架
```

#### 关键文件：`base/agent.py`

```python
"""所有 Agent 的基类，定义统一的生命周期和接口"""
from abc import ABC, abstractmethod
from typing import Any
from mf_agents.messaging.nats_bus import NATSBus
from mf_agents.lineage.tracker import LineageTracker
from mf_agents.crg.graph import ChemicalReasoningGraph
from mf_agents.llm.client import LLMClient


class BaseAgent(ABC):
    """所有智能体的统一基类。
    
    核心职责：
    1. 通过 NATS 总线发送/接收消息
    2. 维护本地 CRG 视图，写入共享 CRG
    3. 自动记录所有操作到谱系追踪器
    4. 处理错误回退和重试
    """
    
    def __init__(
        self,
        agent_id: str,
        bus: NATSBus,
        lineage: LineageTracker,
        crg: ChemicalReasoningGraph,
        llm: LLMClient | None = None,
    ):
        self.agent_id = agent_id
        self.bus = bus
        self.lineage = lineage
        self.crg = crg
        self.llm = llm
    
    @abstractmethod
    async def handle(self, message: dict) -> dict:
        """处理单条消息，返回响应"""
        ...
    
    async def run(self):
        """主循环：订阅消息总线，处理消息"""
        async for message in self.bus.subscribe(self._subscription_subjects()):
            try:
                # 自动追踪谱系
                with self.lineage.span(agent_id=self.agent_id, msg=message) as span:
                    response = await self.handle(message)
                    span.set_response(response)
                    
                # 自动签名 + 发送响应
                await self.bus.publish(
                    subject=message["reply_to"],
                    payload=response,
                    sign=True,
                )
            except Exception as e:
                await self._handle_error(message, e)
    
    @abstractmethod
    def _subscription_subjects(self) -> list[str]:
        """声明此 Agent 订阅的 NATS 主题"""
        ...
    
    async def _handle_error(self, message: dict, error: Exception):
        """错误处理：根据严重程度选择降级/重试/上报"""
        # 实现略
        ...
```

### 5.5 libs/mf-eval · 评估指标库

```
libs/mf-eval/
├── pyproject.toml
└── src/mf_eval/
    ├── molecule/
    │   ├── validity.py                 # Validity/Uniqueness/Novelty
    │   ├── moses.py                    # MOSES 基准
    │   ├── guacamol.py                 # GuacaMol 基准
    │   └── pmo.py                      # PMO 23 任务
    ├── humu/
    │   ├── distortion.py               # 树嵌入蒸馏度
    │   ├── cliff_separation.py         # Activity Cliff 分辨力
    │   └── retrieval.py                # EF1% 检索增益
    ├── pareto/
    │   ├── hypervolume.py              # 超体积计算
    │   ├── spread.py                   # 多样性指标
    │   └── convergence.py              # 收敛性指标
    └── agent/
        ├── task_completion.py          # 任务完成率
        └── audit_completeness.py       # 审计完整性
```

### 5.6 libs/mf-telemetry · 日志/监控/追踪

```
libs/mf-telemetry/
└── src/mf_telemetry/
    ├── logging/
    │   ├── structured.py               # 结构化日志（JSON）
    │   └── correlation.py              # Trace ID 关联
    ├── metrics/
    │   ├── prometheus.py               # Prometheus 指标
    │   └── custom.py                   # 自定义业务指标
    └── tracing/
        ├── opentelemetry.py            # OTel 集成
        └── span_helpers.py
```

---

## 6. services · 微服务实现

> **职责**：每个服务是一个独立的微服务，可独立部署、独立扩展、独立故障

### 6.1 通用服务结构

每个服务遵循**统一的目录模板**：

```
services/{service-name}/
├── pyproject.toml                      # 服务自己的依赖
├── README.md                           # 服务文档
├── Dockerfile
├── src/{service_name}/
│   ├── __init__.py
│   ├── main.py                         # 服务入口
│   ├── config.py                       # 服务配置 (Hydra)
│   ├── api/                            # API 层（gRPC/REST）
│   │   ├── grpc_server.py
│   │   └── rest_server.py
│   ├── domain/                         # 业务逻辑（与框架无关）
│   │   ├── service.py
│   │   └── models.py
│   ├── infra/                          # 基础设施适配器
│   │   ├── db.py
│   │   ├── cache.py
│   │   └── nats_client.py
│   └── client/                         # 给其他服务使用的客户端 SDK
│       └── client.py
├── tests/
│   ├── unit/
│   └── integration/
└── deploy/
    ├── kubernetes/
    │   ├── deployment.yaml
    │   ├── service.yaml
    │   └── hpa.yaml
    └── helm/
        └── values.yaml
```

### 6.2 核心服务详解

#### 6.2.1 `humu-encoder-svc` · HUMU 编码服务

**职责**：把分子/口袋/合成路径编码到 ℍ¹²⁸

```
services/humu-encoder-svc/
├── pyproject.toml                      # 依赖：torch, geoopt, mf-humu, mf-core
├── Dockerfile                          # 基础：mf-base-image (CUDA)
├── src/humu_encoder_svc/
│   ├── main.py                         # 启动 gRPC 服务
│   ├── config.py
│   ├── api/
│   │   └── grpc_server.py              # 实现 humu/encoder.proto
│   ├── domain/
│   │   ├── encoding_service.py         # 核心业务逻辑
│   │   └── batching.py                 # 批量推理优化
│   └── infra/
│       ├── model_loader.py             # 加载预训练编码器权重
│       └── gpu_manager.py              # GPU 资源管理
├── tests/
└── deploy/kubernetes/
    └── deployment.yaml                 # 资源：4x A40, 共享 PVC for weights
```

**关键文件：`domain/encoding_service.py`**

```python
"""HUMU 编码服务的核心业务逻辑"""
from mf_core.types.molecule import MoleculeModel
from mf_core.types.hciv import HCIV
from mf_humu.manifold.lorentz import LorentzManifold
from models.mf_encoders.humu_mol_encoder import MoleculeHUMUEncoder


class EncodingService:
    """提供分子/口袋/路径的统一编码接口"""
    
    def __init__(
        self,
        mol_encoder: MoleculeHUMUEncoder,
        manifold: LorentzManifold,
        device: str = "cuda",
    ):
        self.mol_encoder = mol_encoder.to(device).eval()
        self.manifold = manifold
        self.device = device
    
    async def encode_molecule(self, mol: MoleculeModel) -> HCIV:
        """单分子编码"""
        graph = mol.to_3d_graph()
        with torch.no_grad():
            z = self.mol_encoder(graph.to(self.device))
        return HCIV.from_tensor(z, manifold="lorentz", curvature=1.0)
    
    async def encode_batch(self, mols: list[MoleculeModel]) -> list[HCIV]:
        """批量编码（GPU 利用率优化）"""
        graphs = collate_3d_graphs([m.to_3d_graph() for m in mols])
        with torch.no_grad():
            zs = self.mol_encoder(graphs.to(self.device))
        return [HCIV.from_tensor(z) for z in zs]
```

#### 6.2.2 `generator-router-svc` · 任务感知路由器服务

**职责**：根据任务画像决定调用哪些生成器、权重多少

```
services/generator-router-svc/
└── src/generator_router_svc/
    ├── main.py
    ├── domain/
    │   ├── tar_router.py               # TaskAwareRouter 实现
    │   ├── feature_extractor.py        # 任务画像特征
    │   ├── policy_network.py           # ProxylessNAS 路由策略
    │   ├── online_learner.py           # REINFORCE 在线更新
    │   └── generator_pool.py           # 管理所有生成器客户端
    ├── api/grpc_server.py
    └── infra/
        ├── generator_clients.py        # 调用各生成器的客户端
        └── reward_buffer.py            # 奖励缓存（用于学习）
```

#### 6.2.3 `boltz2-svc` · Boltz-2 亲和力预测服务

```
services/boltz2-svc/
├── pyproject.toml                      # 依赖：boltz, torch, mf-core
├── Dockerfile                          # 基础：mf-oracle-image
├── src/boltz2_svc/
│   ├── main.py
│   ├── config.py
│   ├── api/grpc_server.py              # 实现 oracle/oracle.proto
│   ├── domain/
│   │   ├── boltz2_oracle.py            # 实现 BaseOracle 接口
│   │   ├── batch_scheduler.py          # 动态批处理（吞吐优化）
│   │   └── uncertainty_estimator.py    # 不确定度估计
│   └── infra/
│       ├── model_loader.py
│       └── triton_client.py            # 部署到 Triton Inference Server
└── deploy/
    └── kubernetes/
        ├── deployment.yaml             # 资源：2x H100，常驻 minReplicas=2
        └── hpa.yaml                    # 基于 GPU 利用率自动扩缩
```

#### 6.2.4 `fto-patent-svc` · FTO/专利分析服务

```
services/fto-patent-svc/
└── src/fto_patent_svc/
    ├── main.py
    ├── domain/
    │   ├── fto_analyzer.py             # 主分析逻辑
    │   ├── markush_expander.py         # Markush 通式展开
    │   ├── claim_parser.py             # LLM 权利要求解析
    │   ├── similarity_search.py        # Milvus 相似性检索
    │   └── dead_zone_updater.py        # 更新 Patent Dead Zone
    ├── api/
    │   ├── grpc_server.py
    │   └── rest_server.py              # FTO 报告 PDF 生成
    └── infra/
        ├── milvus_client.py
        ├── neo4j_client.py
        ├── llm_client.py               # Claude Sonnet 4.5
        └── patent_data_sources/
            ├── surechembl.py
            ├── uspto.py
            ├── google_patents_bq.py
            └── reaxys.py
```

#### 6.2.5 `humu-index-svc` · HUMU 向量索引服务

```
services/humu-index-svc/
└── src/humu_index_svc/
    ├── main.py
    ├── domain/
    │   ├── index_service.py            # 检索/插入/删除
    │   ├── hyperbolic_distance.py      # 自定义 Lorentz 距离 metric
    │   └── deduplication.py            # 基于 HUMU 距离的去重
    ├── api/grpc_server.py
    └── infra/
        └── milvus_adapter.py
```

#### 6.2.6 `provenance-svc` · 谱系审计服务

```
services/provenance-svc/
└── src/provenance_svc/
    ├── main.py
    ├── domain/
    │   ├── lineage_writer.py           # 写入 Neo4j
    │   ├── sigstore_signer.py          # Sigstore 签名
    │   ├── audit_query.py              # 审计追溯查询
    │   └── compliance_reporter.py      # 21 CFR Part 11 报告生成
    ├── api/
    │   ├── grpc_server.py
    │   └── rest_server.py              # 审计追溯 UI 后端
    └── infra/
        ├── neo4j_client.py
        ├── sigstore_client.py
        └── s3_client.py                # 审计日志归档
```

#### 6.2.7 `cig-compiler-svc` · CIG 编译器服务

```
services/cig-compiler-svc/
└── src/cig_compiler_svc/
    ├── main.py
    ├── domain/
    │   ├── compiler.py                 # 主编译流程
    │   ├── stages/
    │   │   ├── stage1_semantic.py      # 语义解析
    │   │   ├── stage2_grounding.py     # 知识锚定
    │   │   ├── stage3_cig_build.py     # CIG 构建
    │   │   └── stage4_hciv_encode.py   # HCIV 编码
    │   ├── clarification.py            # 多轮澄清
    │   └── validation.py               # CIG 校验
    ├── api/
    │   └── rest_server.py              # 暴露给前端的 API
    └── infra/
        ├── llm_client.py
        └── tools/
            ├── uniprot.py
            ├── pdb.py
            ├── chembl.py
            ├── pubmed.py
            └── surechembl_search.py
```

#### 6.2.8 `api-gateway` · 对外 API 网关

```
services/api-gateway/
└── src/api_gateway/
    ├── main.py                         # FastAPI 应用
    ├── routers/
    │   ├── projects.py                 # 项目管理
    │   ├── design.py                   # 设计任务提交
    │   ├── molecules.py                # 分子查询
    │   ├── pareto.py                   # Pareto 前沿
    │   ├── fto.py                      # FTO 报告
    │   ├── routes.py                   # 合成路径
    │   └── stream.py                   # WebSocket/SSE 实时流
    ├── auth/
    │   ├── oidc.py                     # OIDC 认证
    │   └── rbac.py                     # 角色权限
    ├── middleware/
    │   ├── tracing.py
    │   ├── rate_limit.py
    │   └── error_handler.py
    └── clients/                        # 内部服务客户端
        ├── orchestrator_client.py
        ├── fto_client.py
        └── retrosyn_client.py
```

---

## 7. agents · 智能体实现

> **职责**：8 个智能体各为独立服务，通过 NATS 总线协同

### 7.1 通用 Agent 结构

```
agents/{agent_name}/
├── pyproject.toml
├── Dockerfile
├── src/{agent_name}/
│   ├── main.py                         # 启动 Agent 主循环
│   ├── config.py
│   ├── agent.py                        # Agent 实现（继承 BaseAgent）
│   ├── tools/                          # 此 Agent 专属工具
│   ├── prompts/                        # 提示词模板
│   └── policies/                       # 决策策略
└── tests/
```

### 7.2 各 Agent 详细结构

#### 7.2.1 `agents/orchestrator`

```
agents/orchestrator/
└── src/orchestrator/
    ├── main.py
    ├── agent.py                        # OrchestratorAgent
    ├── workflow/                       # LangGraph 状态机
    │   ├── state.py                    # MFState 定义
    │   ├── graph_builder.py            # 构建 StateGraph
    │   ├── nodes.py                    # 各节点函数
    │   └── routing.py                  # 条件路由逻辑
    ├── policies/
    │   ├── budget_policy.py            # 预算管理策略
    │   ├── stage_transition.py         # hit→lead_opt→refine 转换
    │   ├── escalation.py               # 错误升级策略
    │   └── reflection.py               # Reflexion 自我反思
    ├── prompts/
    │   ├── planning.txt
    │   ├── reflection.txt
    │   └── decision.txt
    └── tools/
        └── crg_inspector.py            # CRG 一致性检查
```

#### 7.2.2 `agents/nl2obj`

```
agents/nl2obj/
└── src/nl2obj/
    ├── main.py
    ├── agent.py                        # NL2ObjAgent
    ├── stages/
    │   ├── entity_extraction.py
    │   ├── knowledge_grounding.py
    │   ├── cig_construction.py
    │   └── clarification.py
    ├── prompts/
    │   ├── extract_entities.txt
    │   ├── clarify_intent.txt
    │   └── build_cig.txt
    └── tools/                          # 工具调用集合
        ├── uniprot_tool.py
        ├── pdb_tool.py
        ├── chembl_tool.py
        ├── pubmed_tool.py
        └── surechembl_tool.py
```

#### 7.2.3 `agents/critic`

```
agents/critic/
└── src/critic/
    ├── main.py
    ├── agent.py                        # CriticAgent (使用不同的 LLM)
    ├── rules/                          # 100+ 质疑规则
    │   ├── confidence_rules.py
    │   ├── diversity_rules.py
    │   ├── fto_rules.py
    │   ├── synthesis_rules.py
    │   └── safety_rules.py
    ├── triggers/
    │   ├── batch_trigger.py            # 批次结束触发
    │   ├── pareto_change_trigger.py    # Pareto 变化触发
    │   └── outlier_trigger.py          # 异常值触发
    └── prompts/
        ├── batch_review.txt
        └── escalation_review.txt
```

---

## 8. models · 模型实现库

> **职责**：所有 ML 模型的训练 + 推理代码，独立于服务部署

### 8.1 mf-generators · 8 个生成器

每个生成器是独立的包，遵循统一结构：

```
models/mf-generators/{generator_name}/
├── pyproject.toml                      # 注册为 entry-point 插件
├── README.md
├── src/mf_generators/{generator_name}/
│   ├── __init__.py
│   ├── generator.py                    # 主类（实现 BaseGenerator）
│   ├── model/                          # 神经网络模块
│   │   ├── architecture.py
│   │   ├── layers.py
│   │   └── losses.py
│   ├── training/                       # 训练代码
│   │   ├── trainer.py
│   │   ├── data_loader.py
│   │   └── augmentation.py
│   ├── inference/                      # 推理优化
│   │   ├── sampler.py
│   │   ├── beam_search.py
│   │   └── caching.py
│   └── configs/
│       ├── model_default.yaml
│       └── training_default.yaml
├── tests/
└── checkpoints/                        # （DVC 管理，不入 Git）
    └── .gitkeep
```

#### `mf-generators/hfm_3d/`

```
src/mf_generators/hfm_3d/
├── generator.py                        # HFM3DGenerator(BaseGenerator)
├── model/
│   ├── lorentz_flow_matching.py        # 双曲流匹配核心
│   ├── lorentz_equivariant_layer.py    # SE(3)-等变 + Lorentz
│   ├── intent_cone_sampler.py          # 意图锥内采样
│   └── ode_solver.py                   # Midpoint ODE solver
├── training/
│   ├── trainer.py                      # 训练循环
│   └── flow_matching_loss.py           # FM 损失函数
└── inference/
    ├── conditional_sampler.py          # 条件采样
    └── geodesic_interpolation.py
```

**关键文件：`generator.py`**

```python
"""HFM-3D 生成器：在 Lorentz 双曲流形切丛上的 Flow Matching"""
from mf_core.plugins.generator import BaseGenerator
from mf_core.types.molecule import MoleculeModel
from mf_core.types.cig import ChemicalIntentGraph
from mf_core.types.hciv import HCIV, IntentCone
from mf_humu.manifold.lorentz import LorentzManifold
from mf_humu.operations.intent_cone import sample_within_cone
from .model.lorentz_flow_matching import LorentzFlowMatchingModel
from .model.ode_solver import MidpointSolver


class HFM3DGenerator(BaseGenerator):
    """双曲流匹配 3D 分子生成器
    
    定位：Hit 阶段冷启动，新颖性最高，3D 几何最精确
    优势：意图锥约束采样，生成结果天然在目标区域
    """
    
    name = "hfm_3d"
    version = "2.3.0"
    supported_modes = ["hit_finding", "scaffold_hop"]
    
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.manifold = LorentzManifold(curvature=1.0)
        self.model = LorentzFlowMatchingModel.from_checkpoint(ckpt_path)
        self.model = self.model.to(device).eval()
        self.solver = MidpointSolver(n_steps=20)
        self.device = device
    
    async def generate(
        self,
        intent_vector: HCIV,
        intent_cone: IntentCone,
        cig: ChemicalIntentGraph,
        n_samples: int = 100,
        seed: int | None = None,
    ):
        # 1. 在意图锥内采样初始噪声点
        z_0 = sample_within_cone(
            cone=intent_cone,
            n_samples=n_samples,
            manifold=self.manifold,
            seed=seed,
        )
        
        # 2. 解 ODE 演化到目标分布
        z_1 = self.solver.solve(
            x_0=z_0,
            vector_field=lambda z, t: self.model(
                z, t, 
                pocket=cig.target_context.pocket_embedding,
                intent=intent_vector,
            ),
            t_span=(0.0, 1.0),
        )
        
        # 3. 解码为分子
        for z in z_1:
            mol = self.model.decode(z)
            if mol.is_valid:
                yield mol
    
    async def health_check(self):
        return {
            "model_loaded": self.model is not None,
            "device": self.device,
            "gpu_available": torch.cuda.is_available(),
        }
```

#### `mf-generators/fragfm/`

```
src/mf_generators/fragfm/
├── generator.py                        # FragFMGenerator
├── model/
│   ├── two_level_dfm.py                # 两层离散流匹配
│   ├── scaffold_ctmc.py                # Scaffold 连续时间马尔可夫链
│   ├── rgroup_dfm.py                   # R基团 DFM
│   ├── sa_aware_rate_matrix.py         # SA 内嵌的转移概率矩阵
│   └── fragment_vocabulary.py          # 片段词表
├── training/
│   ├── data/
│   │   ├── fragment_extraction.py      # BRICS/RECAP 片段提取
│   │   └── scaffold_decomposition.py
│   └── trainer.py
└── inference/
    └── hierarchical_sampler.py
```

#### `mf-generators/lamgen_3d/`

```
src/mf_generators/lamgen_3d/
├── generator.py                        # LaMGen3DProGenerator
├── model/
│   ├── multi_target_attention.py       # 多靶点交叉注意力门控
│   ├── rotation_aware_tokens.py        # 旋转感知 token
│   ├── humu_decoder.py                 # 直接输出 HUMU 坐标
│   └── speculative_decoder.py          # Speculative decoding 加速
├── training/
│   └── trainer.py                      # LoRA 微调
└── inference/
    └── batched_inference.py
```

#### `mf-generators/crem_3d/`

```
src/mf_generators/crem_3d/
├── generator.py                        # CReM3DGenerator
├── algorithm/
│   ├── fragment_replacement.py         # CReM 片段替换
│   ├── pharmacophore_matching.py       # 3D 药效团约束
│   └── docking_in_loop.py              # 循环中实时对接
├── data/
│   └── crem_database.py                # CReM 片段数据库
└── inference/
    └── greedy_search.py
```

#### `mf-generators/mmpt_rag/`

```
src/mf_generators/mmpt_rag/
├── generator.py                        # MMPTRAGGenerator
├── model/
│   ├── seq2seq_transformer.py
│   ├── retrieval_encoder.py
│   └── contrastive_decoder.py          # 专利负样本对比解码
├── retrieval/
│   ├── positive_corpus.py              # ChEMBL MMP pairs
│   └── negative_corpus.py              # SureChEMBL 专利变换
└── inference/
    └── fto_aware_beam_search.py
```

#### `mf-generators/evomol_rl/`

```
src/mf_generators/evomol_rl/
├── generator.py                        # EvoMolRLGenerator
├── algorithm/
│   ├── genetic_algorithm.py
│   ├── pareto_archive.py
│   ├── hypervolume_reward.py           # HVI 奖励
│   └── sleeping_bandit.py              # Sleeping Bandit 策略
├── operators/
│   ├── mutation.py
│   ├── crossover.py
│   └── scaffold_hop.py
└── inference/
    └── evolutionary_loop.py
```

#### `mf-generators/incremental_clm/`

```
src/mf_generators/incremental_clm/
├── generator.py                        # IncrementalCLMGenerator
├── model/
│   ├── base_clm.py                     # 基础化学语言模型
│   ├── ewc_regularizer.py              # 弹性权重整合
│   └── packnet.py                      # 参数空间隔离
├── learning/
│   ├── online_learner.py               # 在线持续学习
│   ├── sar_branch_manager.py           # SAR 分支管理
│   └── replay_buffer.py
└── inference/
    └── conditional_generation.py
```

#### `mf-generators/uas/`

```
src/mf_generators/uas/
├── unfamiliarity_estimator.py          # OOD 不熟悉度估计
├── autoencoder/
│   ├── molecule_ae.py                  # 重建误差作为 OOD 信号
│   └── trainer.py
├── sampler/
│   └── ood_aware_sampling.py
└── triggers/
    └── active_learning_trigger.py
```

### 8.2 mf-oracles · Oracle 实现

```
models/mf-oracles/
├── boltz2/
│   ├── pyproject.toml
│   └── src/mf_oracles/boltz2/
│       ├── oracle.py                   # Boltz2Oracle (实现 BaseOracle)
│       ├── model/
│       │   └── boltz2_wrapper.py       # 封装 Boltz-2 推理
│       ├── inference/
│       │   ├── batch_processor.py
│       │   └── triton_client.py
│       └── uncertainty/
│           └── ensemble_estimator.py   # 多个 head 集成估不确定度
├── diffdock_l/
│   └── src/mf_oracles/diffdock_l/
│       ├── oracle.py
│       └── model/
│           └── diffdock_wrapper.py
├── gnina/
│   └── src/mf_oracles/gnina/
│       ├── oracle.py
│       └── infra/
│           └── gnina_subprocess.py     # 调用 GNINA 二进制
├── openfe/
│   └── src/mf_oracles/openfe/
│       ├── oracle.py
│       ├── workflow/
│       │   ├── relative_fep.py
│       │   ├── nnp_mm_hybrid.py        # MACE-OFF24 混合
│       │   └── perturbation_network.py
│       └── infra/
│           └── hpc_submitter.py        # 提交到 HPC 集群
└── admet_ai/
    └── src/mf_oracles/admet_ai/
        ├── oracle.py
        └── model/
            ├── admet_ai_wrapper.py
            └── chemprop_wrapper.py
```

### 8.3 mf-retrosyn · 逆合成模型

```
models/mf-retrosyn/
├── aizynth_wrapper/
│   └── src/mf_retrosyn/aizynth/
│       ├── retrosyn.py                 # 实现 BaseRetrosynModel
│       ├── mcts/
│       │   ├── tree_search.py
│       │   ├── bond_prompting.py       # 用户约束注入
│       │   └── supply_aware_scoring.py # 供应链感知评分
│       └── infra/
│           └── aizynthfinder_runner.py
├── rsgpt/
│   └── src/mf_retrosyn/rsgpt/
│       ├── retrosyn.py
│       ├── model/
│       │   └── rsgpt_transformer.py
│       └── inference/
│           └── beam_search.py
└── ualign/
    └── src/mf_retrosyn/ualign/
        ├── retrosyn.py
        └── model/
            ├── graph_to_sequence.py
            └── unsupervised_alignment.py
```

### 8.4 mf-encoders · HUMU 编码器

```
models/mf-encoders/
├── humu_mol_encoder/
│   └── src/mf_encoders/humu_mol/
│       ├── encoder.py                  # MoleculeHUMUEncoder
│       ├── model/
│       │   ├── segnn_backbone.py       # SE(3)-等变主干
│       │   └── lorentz_projection.py   # 切空间投影
│       └── training/
│           └── contrastive_trainer.py
├── humu_pocket_encoder/
│   └── src/mf_encoders/humu_pocket/
│       ├── encoder.py
│       ├── model/
│       │   ├── equivariant_pocket_gnn.py
│       │   └── esm2_fusion.py          # 序列+结构融合
│       └── training/
├── humu_route_encoder/
│   └── src/mf_encoders/humu_route/
│       ├── encoder.py
│       ├── model/
│       │   ├── reaction_tree_gnn.py    # AND-OR 图编码
│       │   └── lorentz_treelstm.py     # 双曲 TreeLSTM
│       └── training/
└── humu_intent_encoder/
    └── src/mf_encoders/humu_intent/
        ├── encoder.py                  # CIG → HCIV
        ├── model/
        │   └── hypergraph_encoder.py
        └── training/
```

---

## 9. data · 数据管线

```
data/
├── README.md
├── alembic/                            # 数据库迁移
│   ├── alembic.ini
│   └── versions/
├── ingestion/                          # 数据接入
│   ├── chembl/
│   │   ├── downloader.py
│   │   └── importer.py
│   ├── pdb/
│   ├── surechembl/
│   │   ├── daily_sync.py               # 每日增量
│   │   └── markush_extractor.py
│   ├── enamine_real/
│   │   └── faiss_indexer.py            # 49B 化合物的本地索引
│   ├── uspto/
│   ├── reaxys/
│   └── pistachio/                      # 反应数据集
├── processing/
│   ├── molecule_canonicalization.py
│   ├── conformer_generation.py
│   ├── fingerprint_computation.py
│   ├── fragment_extraction.py
│   └── reaction_template_extraction.py
├── validation/                         # 数据质量检查
│   ├── schema_check.py
│   ├── duplicate_detection.py
│   └── outlier_detection.py
├── samples/                            # 测试用小样本（入 Git）
│   ├── molecules_100.csv
│   ├── pockets_10.json
│   └── reactions_50.txt
└── dvc/                                # DVC 配置（数据版本控制）
    ├── .dvcignore
    └── pipelines/
        ├── humu_pretrain_data.dvc.yaml
        └── patent_index.dvc.yaml
```

---

## 10. infra · 基础设施

### 10.1 Docker

```
infra/docker/
├── base/
│   ├── Dockerfile.mf-base              # CUDA 12.4 + Python 3.11 + 通用依赖
│   ├── Dockerfile.mf-chem              # + RDKit + OpenBabel
│   ├── Dockerfile.mf-generator         # + 生成模型依赖（torch, geoopt, e3nn）
│   ├── Dockerfile.mf-oracle            # + Boltz-2 + DiffDock + OpenFE
│   └── Dockerfile.mf-agent             # + LangGraph + LLM SDK
├── docker-compose.dev.yml              # 本地开发环境
├── docker-compose.test.yml             # 集成测试环境
└── docker-compose.minimal.yml          # 最小化 demo 环境
```

#### `infra/docker/base/Dockerfile.mf-base`

```dockerfile
# 通用基础镜像，所有其他镜像都继承
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV PYTHON_VERSION=3.11
ENV DEBIAN_FRONTEND=noninteractive

# 系统依赖
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev \
    python3-pip git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app
```

### 10.2 Kubernetes

```
infra/kubernetes/
├── README.md
├── namespaces/
│   ├── mf-generators.yaml
│   ├── mf-agents.yaml
│   ├── mf-oracles.yaml
│   ├── mf-data.yaml
│   └── mf-mlops.yaml
├── infrastructure/                     # 基础设施层
│   ├── milvus/
│   ├── neo4j/
│   ├── postgres/
│   ├── minio/
│   ├── nats/
│   └── redis/
├── services/                           # 各服务部署
│   ├── humu-encoder/
│   ├── boltz2/
│   ├── fto-patent/
│   └── ...
├── monitoring/
│   ├── prometheus/
│   ├── grafana/
│   ├── loki/
│   └── opentelemetry/
└── policies/                           # OPA Gatekeeper 策略
    ├── resource-limits.yaml
    ├── image-signing.yaml
    └── network-policies.yaml
```

### 10.3 Helm

```
infra/helm/
├── moleculeforge/                      # 总伞 Chart
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
└── charts/                             # 子 Charts
    ├── humu-encoder/
    │   ├── Chart.yaml
    │   ├── values.yaml
    │   └── templates/
    │       ├── deployment.yaml
    │       ├── service.yaml
    │       ├── hpa.yaml
    │       └── pvc.yaml
    ├── boltz2-oracle/
    └── ...
```

### 10.4 Terraform

```
infra/terraform/
├── README.md
├── environments/
│   ├── dev/
│   ├── staging/
│   └── prod/
└── modules/
    ├── eks-cluster/                    # EKS 集群（或 GKE/AKS）
    ├── rds-postgres/
    ├── s3-storage/
    ├── vpc-networking/
    └── iam-roles/
```

### 10.5 运维脚本

```
infra/scripts/
├── build_all_images.sh                 # 批量构建镜像
├── deploy_to_staging.sh
├── db_backup.sh
├── humu_index_rebuild.sh               # 重建 Milvus 索引
└── patent_db_sync.sh                   # 专利数据库同步
```

---

## 11. tests · 测试体系

```
tests/
├── README.md
├── unit/                               # 单元测试（每个 lib 内也有自己的 unit tests）
│   ├── libs/
│   ├── models/
│   └── services/
├── integration/                        # 集成测试（多服务协同）
│   ├── test_humu_encoding_pipeline.py
│   ├── test_generator_router_flow.py
│   ├── test_oracle_cascade.py
│   └── test_fto_pipeline.py
├── e2e/                                # 端到端测试（完整 NL → 候选）
│   ├── test_kras_g12c_pilot.py
│   ├── test_multi_target_design.py
│   └── test_audit_completeness.py
├── benchmark/                          # 基准测试
│   ├── moses_benchmark.py
│   ├── guacamol_benchmark.py
│   ├── pmo_benchmark.py
│   └── crossdocked_benchmark.py
├── fixtures/                           # 测试数据
│   ├── molecules.json
│   ├── pockets.pdb
│   ├── reactions.txt
│   └── cigs.json
├── conftest.py                         # Pytest 全局 fixtures
└── pytest.ini
```

### 关键文件：`tests/e2e/test_kras_g12c_pilot.py`

```python
"""端到端测试：KRAS G12C 单靶点全流程

验证从自然语言输入到生成候选分子的完整 pipeline。
作为 MVP 阶段（M8）的关键验收标准。
"""
import pytest
from mf_core.types.cig import ChemicalIntentGraph
from agents.orchestrator.workflow.graph_builder import build_main_workflow


@pytest.mark.e2e
@pytest.mark.slow
async def test_kras_g12c_full_pipeline(integration_environment):
    """从 NL 提示出发，验证 6 小时内产出至少 5 个 Pareto 前沿候选"""
    
    nl_input = (
        "Design a selective KRAS G12C covalent inhibitor "
        "with nanomolar potency, oral bioavailability, "
        "avoiding Mirati's patent US11186593, "
        "synthesizable in <= 5 steps from Enamine REAL building blocks."
    )
    
    workflow = build_main_workflow(env=integration_environment)
    
    result = await workflow.ainvoke({
        "user_input": nl_input,
        "budget": {
            "wallclock_hours": 6,
            "oracle_L2_calls": 2000,
            "oracle_L3_calls": 50,
        },
    })
    
    # 验收条件
    assert result["status"] == "success"
    assert len(result["pareto_front"]) >= 5
    assert all(c["fto_score"] > 0.8 for c in result["pareto_front"])
    assert all(c["sa_score"] < 4.0 for c in result["pareto_front"])
    assert result["audit_chain_complete"] is True
```

---

## 12. docs · 文档系统

```
docs/
├── README.md                           # 文档导航
├── mkdocs.yml                          # MkDocs 配置
├── architecture/                       # 架构文档
│   ├── 01_overview.md                  # 系统概览
│   ├── 02_humu_design.md               # HUMU 设计深度解析
│   ├── 03_agent_system.md              # 智能体系统
│   ├── 04_data_flow.md                 # 数据流详解
│   └── 05_security.md                  # 安全与合规
├── api/                                # API 文档
│   ├── public/
│   │   ├── rest_api.md
│   │   └── websocket_api.md
│   └── internal/
│       ├── grpc_services.md
│       └── nats_subjects.md
├── tutorials/                          # 教程
│   ├── 01_quickstart.md
│   ├── 02_first_design_run.md
│   ├── 03_custom_generator.md          # 如何添加新生成器
│   └── 04_custom_oracle.md
├── adr/                                # Architecture Decision Records
│   ├── 0001-use-lorentz-not-poincare.md
│   ├── 0002-monorepo-with-uv-workspace.md
│   ├── 0003-langgraph-for-orchestration.md
│   ├── 0004-protobuf-for-internal-rpc.md
│   └── 0005-sigstore-for-audit-signing.md
├── papers/                             # 内部技术报告
│   ├── humu_theory.tex
│   └── benchmark_results_2026.md
└── contributing/
    ├── code_style.md
    ├── pr_guidelines.md
    └── testing_strategy.md
```

### `docs/adr/` · 架构决策记录

每个重大设计决策都用一份 ADR 记录，格式：

```markdown
# ADR-0001: 使用 Lorentz 模型而非 Poincaré 球

## 状态
已采纳 (2026-04-01)

## 背景
HUMU 需要选择一个具体的双曲空间模型...

## 决策
采用 Lorentz 模型（Hyperboloid model）

## 理由
1. 解析梯度：exp_map / log_map 有闭式解
2. 数值稳定：Poincaré 球在边界处梯度爆炸
3. 计算高效：geoopt 库原生支持
...

## 后果
- 优势：...
- 劣势：...
- 缓解：...
```

---

## 13. tools · 开发工具

```
tools/
├── codegen/
│   ├── proto_to_python.sh              # Proto → Python 生成
│   ├── openapi_to_client.sh            # OpenAPI → 客户端 SDK
│   ├── check_proto_sync.sh             # 检查 Proto 是否已同步生成
│   └── generate_grpc_stubs.py
├── linting/
│   ├── custom_ruff_rules/              # 自定义 lint 规则
│   ├── chemistry_linter.py             # 化学相关代码 lint
│   └── secrets_scanner.py
├── benchmarks/
│   ├── molecule_generation_benchmark.py
│   ├── oracle_throughput_benchmark.py
│   └── humu_retrieval_benchmark.py
└── dev/
    ├── seed_dev_db.py                  # 开发数据库种子数据
    ├── mock_oracle_server.py           # Mock Oracle（开发时用）
    └── generate_test_cig.py
```

---

## 14. 预留扩展目录

按你的要求，三个预留模块**保留完整目录占位**：

### 14.1 `ui/` · 前端

```
ui/
├── README.md                           # 「待实施」说明 + 预留接口列表
├── ARCHITECTURE.md                     # 前端架构设计（已规划）
├── package.json                        # 占位（NextJS/React + TS）
├── tsconfig.json                       # 占位
├── src/
│   └── README.md                       # 「核心架构完成后实施」
├── public/
│   └── README.md
└── design/
    ├── README.md
    └── mockups/                        # 设计稿（可先放）
```

### 14.2 `wetlab/` · 湿实验室硬件接口

```
wetlab/
├── README.md                           # 「待实施」说明
├── ARCHITECTURE.md                     # 已规划：XDL + SiLA2 + 硬件适配
├── xdl-compiler/
│   └── README.md                       # 待实施：SSP → XDL 2.0 编译器
├── sila2-adapter/
│   └── README.md                       # 待实施：XDL → SiLA2 gRPC
├── hardware-drivers/
│   ├── chemputer/
│   ├── opentrons/
│   ├── chemspeed/
│   ├── ecl/
│   └── strateos/
└── eln-integrations/
    ├── benchling/
    ├── idbs/
    └── dotmatics/
```

### 14.3 `commercial/` · 商业化

```
commercial/
├── README.md                           # 「待实施」说明
├── ARCHITECTURE.md
├── multi-tenancy/
│   └── README.md                       # 待实施：多租户隔离
├── billing/
│   └── README.md                       # 待实施：计量计费
├── compliance/
│   ├── 21cfr-part11/
│   ├── eu-annex-11/
│   ├── gdpr/
│   └── china-dsl/
└── customer-deployments/
    └── README.md                       # 客户私有化部署模板
```

---

## 15. 关键文件深度解析

### 15.1 仓库根 `pyproject.toml`（已展示）—— 工作区根配置

### 15.2 `libs/mf-core/src/mf_core/types/cig.py` —— CIG 数据模型

```python
"""Chemical Intent Graph 的 Pydantic 数据模型。

CIG 是用户自然语言意图编译后的精确目标函数表示。
"""
from datetime import datetime
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class ObjectiveType(str, Enum):
    CONTINUOUS_MAXIMIZE = "continuous_maximize"
    CONTINUOUS_MINIMIZE = "continuous_minimize"
    RATIO_MAXIMIZE = "ratio_maximize"
    MULTI_CONSTRAINT_SATISFY = "multi_constraint_satisfy"


class ObjectiveNode(BaseModel):
    """目标节点（Pareto 优化对象之一）"""
    id: str
    type: ObjectiveType
    oracle: str
    target_value: float | None = None
    uncertainty_tolerance: float = 0.5
    weight: float = Field(ge=0.0, le=1.0)
    pareto_tier: Literal[1, 2, 3] = 2
    constraints: dict | None = None
    soft_penalty: bool = False


class TargetContext(BaseModel):
    """靶点上下文"""
    pocket_embedding: list[float] | None = None    # ESMFold 口袋向量
    pharmacophore_3d: dict | None = None           # 3D 药效团
    binding_mode_prior: str | None = None
    pdb_ids: list[str] = Field(default_factory=list)
    uniprot_ids: list[str] = Field(default_factory=list)


class GenerativePriors(BaseModel):
    """生成先验偏好"""
    scaffold_bias: str | None = None
    mw_range: tuple[float, float] | None = None
    ring_systems: list[str] = Field(default_factory=list)
    forbidden_substructures: list[str] = Field(default_factory=list)
    novelty_vs_analogy: float = Field(0.5, ge=0.0, le=1.0)


class BudgetConstraints(BaseModel):
    """预算约束"""
    oracle_L2_calls_max: int = 5000
    oracle_L3_calls_max: int = 200
    oracle_L4_calls_max: int = 20
    wallclock_hours: float = 12.0
    cost_usd_max: float | None = None


class ChemicalIntentGraph(BaseModel):
    """化学意图图 - CIG 核心数据结构"""
    model_config = ConfigDict(extra="forbid")
    
    intent_id: str
    version: str = "2.0"
    signature: str | None = None                   # Sigstore 签名
    
    target_context: TargetContext
    objective_nodes: list[ObjectiveNode]
    generative_priors: GenerativePriors = GenerativePriors()
    budget_constraints: BudgetConstraints = BudgetConstraints()
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str | None = None                  # NL2Obj agent ID
    source_user_input: str                          # 原始 NL 输入（审计用）
    
    def validate_consistency(self) -> list[str]:
        """业务级别一致性校验，返回警告列表"""
        warnings = []
        weight_sum = sum(o.weight for o in self.objective_nodes)
        if not (0.95 <= weight_sum <= 1.05):
            warnings.append(f"Objective weights sum to {weight_sum}, expected ~1.0")
        # 更多校验...
        return warnings
```

### 15.3 `services/{any-service}/src/{name}/main.py` —— 服务入口模板

```python
"""服务启动入口 - 所有服务遵循此模板"""
import asyncio
import signal
import sys
import hydra
from omegaconf import DictConfig
from mf_telemetry.logging.structured import setup_logging
from mf_telemetry.tracing.opentelemetry import setup_tracing

from .api.grpc_server import GRPCServer
from .domain.service import CoreService
from .infra.db import init_db
from .config import ServiceConfig


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    setup_logging(level=cfg.log_level, service=cfg.service_name)
    setup_tracing(service=cfg.service_name, endpoint=cfg.tracing.otel_endpoint)
    
    asyncio.run(_async_main(ServiceConfig(**cfg)))


async def _async_main(config: ServiceConfig):
    # 1. 初始化基础设施
    db = await init_db(config.db)
    
    # 2. 初始化业务服务
    service = CoreService(config=config, db=db)
    await service.startup()
    
    # 3. 启动 gRPC 服务器
    server = GRPCServer(service=service, port=config.grpc_port)
    
    # 4. 优雅关闭
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda: asyncio.create_task(_shutdown(server, service))
        )
    
    await server.serve()


async def _shutdown(server, service):
    await server.stop(grace_period=30)
    await service.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
```

### 15.4 `agents/orchestrator/src/orchestrator/workflow/graph_builder.py` —— LangGraph 主流程

```python
"""Orchestrator 的 LangGraph 状态机定义"""
from langgraph.graph import StateGraph, END
from .state import MFState
from .nodes import (
    nl2obj_node, humu_encode_node, generate_node,
    validate_node, fto_check_node, retrosyn_node,
    critic_node, orchestrate_node, refine_node,
)
from .routing import (
    route_after_validation, route_after_critic, orchestrator_decision,
)


def build_main_workflow() -> StateGraph:
    """构建主工作流状态机"""
    builder = StateGraph(MFState)
    
    # 注册节点
    builder.add_node("nl2obj", nl2obj_node)
    builder.add_node("humu_encode", humu_encode_node)
    builder.add_node("generate", generate_node)
    builder.add_node("validate", validate_node)
    builder.add_node("fto_check", fto_check_node)
    builder.add_node("retrosyn", retrosyn_node)
    builder.add_node("critic", critic_node)
    builder.add_node("orchestrate", orchestrate_node)
    builder.add_node("refine", refine_node)
    
    # 入口
    builder.set_entry_point("nl2obj")
    
    # 线性边
    builder.add_edge("nl2obj", "humu_encode")
    builder.add_edge("humu_encode", "generate")
    builder.add_edge("generate", "validate")
    builder.add_edge("fto_check", "retrosyn")
    builder.add_edge("retrosyn", "critic")
    
    # 条件路由
    builder.add_conditional_edges("validate", route_after_validation, {
        "fto_check": "fto_check",
        "regenerate": "generate",
        "escalate_L3": "validate",
    })
    builder.add_conditional_edges("critic", route_after_critic, {
        "proceed": "orchestrate",
        "block_regenerate": "generate",
        "refine": "refine",
        "human_review": END,
    })
    builder.add_conditional_edges("orchestrate", orchestrator_decision, {
        "continue_generate": "generate",
        "final_output": END,
        "escalate_budget": "validate",
    })
    builder.add_edge("refine", "validate")
    
    return builder.compile()
```

### 15.5 `configs/default.yaml` —— Hydra 默认配置

```yaml
# MoleculeForge 默认配置（开发环境）

defaults:
  - _self_
  - env: dev
  - models: default
  - services: default
  - agents: default

run:
  name: ${oc.env:USER}-${now:%Y%m%d-%H%M%S}
  output_dir: ./outputs/${run.name}
  seed: 42

log_level: INFO

tracing:
  otel_endpoint: http://localhost:4317
  service_name_prefix: mf

# HUMU 配置
humu:
  manifold:
    type: lorentz
    curvature: 1.0
    dim: 128
  encoder:
    mol_ckpt: ${oc.env:MF_HUMU_MOL_CKPT,/checkpoints/humu_mol_v1.pt}
    pocket_ckpt: ${oc.env:MF_HUMU_POCKET_CKPT,/checkpoints/humu_pocket_v1.pt}
    route_ckpt: ${oc.env:MF_HUMU_ROUTE_CKPT,/checkpoints/humu_route_v1.pt}

# Oracle 级联预算
oracle_cascade:
  L0: { enabled: true, batch_size: 1000 }
  L1: { enabled: true, batch_size: 64, model: boltz2_v1.5 }
  L2: { enabled: true, batch_size: 16, model: diffdock_l_v1.2 }
  L3: { enabled: true, batch_size: 4, model: openfe_rbfe }
  L4: { enabled: false, batch_size: 1, model: gpu4pyscf }

# Agent 路由
agents:
  orchestrator:
    llm: claude-sonnet-4.5
    max_cycles: 20
    reflection_interval: 5
  critic:
    llm: deepseek-v3        # 不同模型族
    rules_path: agents/critic/rules
    severity_threshold: WARN
```

### 15.6 顶层 `Makefile`（已展示）

### 15.7 `protos/buf.gen.yaml` —— Proto 代码生成

```yaml
version: v2
plugins:
  - remote: buf.build/protocolbuffers/python:v25.0
    out: ../libs/mf-core/src/mf_core/proto_gen
  - remote: buf.build/grpc/python:v1.62
    out: ../libs/mf-core/src/mf_core/proto_gen
  - remote: buf.build/community/nrfta-pyi:v25.0
    out: ../libs/mf-core/src/mf_core/proto_gen
```

---

## 16. 开发工作流

### 16.1 新开发者首次上手

```bash
# 1. Clone 仓库
git clone https://github.com/moleculeforge/moleculeforge.git
cd moleculeforge

# 2. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 安装所有依赖（uv 工作区会自动处理子包）
make install

# 4. 安装 pre-commit 钩子
uv run pre-commit install

# 5. 启动本地开发环境（Docker Compose）
make run-dev

# 6. 运行单元测试验证
make test-unit

# 7. 看一下文档
cd docs && uv run mkdocs serve
```

### 16.2 添加一个新生成器（最常见的扩展）

```bash
# 1. 创建新生成器包
mkdir -p models/mf-generators/my_new_generator/src/mf_generators/my_new_generator

# 2. 配置 pyproject.toml（注册插件）
cat > models/mf-generators/my_new_generator/pyproject.toml <<EOF
[project]
name = "mf-generators-my-new-generator"
version = "0.1.0"
dependencies = ["mf-core", "mf-humu", "torch>=2.6"]

[project.entry-points."moleculeforge.generators"]
my_new_generator = "mf_generators.my_new_generator:MyNewGenerator"
EOF

# 3. 实现 BaseGenerator
# 编辑 src/mf_generators/my_new_generator/generator.py

# 4. 写测试
# 编辑 tests/

# 5. 同步 workspace
uv sync

# 6. 在配置中启用
# 编辑 configs/models/default.yaml，加入新生成器

# 7. CI 自动测试 + lint
git add .
git commit -m "feat: add MyNewGenerator"
git push origin feature/my-new-generator
# → CI 自动运行 ruff + mypy + import-linter + pytest
```

### 16.3 修改 Proto 协议（破坏性变更）

```bash
# 1. 修改 proto 文件
vim protos/moleculeforge/v1/generator/generator.proto

# 2. 检查破坏性变更（防止破坏现有调用）
cd protos && buf breaking --against '.git#branch=main'

# 3. 如果有破坏性变更，要么：
#    a. 加版本号（v2/）保持向后兼容
#    b. 协调所有调用方一起升级

# 4. 生成新代码
buf generate

# 5. 提交（CI 会再次验证）
```

---

## 17. 插件化扩展指南

整个架构的灵魂在于：**核心代码极其稳定，所有创新都通过插件实现**。

### 17.1 可插拔的层

| 层 | 插件接口 | 注册方式 | 当前实现 |
|----|----------|----------|----------|
| 生成器 | `BaseGenerator` | `moleculeforge.generators` entry-point | 8 个 |
| Oracle | `BaseOracle` | `moleculeforge.oracles` entry-point | 5 个 |
| 逆合成模型 | `BaseRetrosynModel` | `moleculeforge.retrosyn` entry-point | 3 个 |
| HUMU 编码器 | `BaseHUMUEncoder` | `moleculeforge.encoders` entry-point | 3 个 |
| LLM Provider | `BaseLLMProvider` | `moleculeforge.llm_providers` | 4 个 |
| 数据源 | `BaseDataSource` | `moleculeforge.data_sources` | 多个 |

### 17.2 添加自定义 Oracle 示例

假设 2027 年出现了一个比 Boltz-2 更准的亲和力预测模型 `MegaAffinity-v3`：

```python
# models/mf-oracles/megaaffinity/src/mf_oracles/megaaffinity/oracle.py

from mf_core.plugins.oracle import BaseOracle


class MegaAffinityOracle(BaseOracle):
    name = "megaaffinity_v3"
    version = "3.0.0"
    oracle_level = "L1"             # L0/L1/L2/L3/L4
    
    async def predict(self, mol, target):
        # 实现略
        ...
    
    async def predict_with_uncertainty(self, mol, target):
        # 实现略
        ...
```

```toml
# pyproject.toml
[project.entry-points."moleculeforge.oracles"]
megaaffinity = "mf_oracles.megaaffinity:MegaAffinityOracle"
```

```yaml
# configs/oracles/cascade.yaml 改成使用新 oracle
L1:
  enabled: true
  model: megaaffinity_v3        # ← 一行配置切换！
```

**就这样**——核心代码一行不改，新 Oracle 立即生效。这就是插件化架构的力量。

---

## 18. 总结：架构灵活性的体现

### 18.1 这套架构如何应对各种变化？

| 场景 | 应对方式 | 影响范围 |
|------|----------|----------|
| 新增生成器 | 新建插件包 + 注册 entry-point | 0 行核心代码改动 |
| 替换底层化学库（RDKit→OEChem）| 改 `libs/mf-chem/adapters/` | 1 个文件 |
| 切换 LLM 提供商 | 改配置 + 实现 LLMProvider | 1 个文件 + 配置 |
| 升级双曲流形理论 | 改 `libs/mf-humu/` | 共享内核内 |
| 新增 Agent | 新建 `agents/` 子目录 | 0 影响其他 Agent |
| 新增 Oracle | 新建插件 | 0 影响其他 Oracle |
| 协议变更 | 改 Proto + 重生成 | 自动传播到所有调用方 |
| 数据库切换 | 改 `infra/db.py` 适配器 | 1 处 |
| K8s → 自建机房 | 改 `infra/kubernetes/` | 部署层独立 |
| **接入前端** | **填入 `ui/` 即可** | **核心架构 0 改动** |
| **接入湿实验室** | **填入 `wetlab/` 即可** | **核心架构 0 改动** |
| **商业化部署** | **填入 `commercial/` 即可** | **核心架构 0 改动** |

### 18.2 关键架构指标

| 指标 | 设计目标 |
|------|----------|
| 服务独立部署性 | 100%（每个服务可独立发布）|
| 接口向后兼容性 | Proto v1 永远不删除 |
| 插件热替换 | 修改配置即可，不需要重启核心 |
| 测试覆盖率 | 单元测试 > 80%，关键路径 > 95% |
| 平均文件行数 | < 300 行（单一职责）|
| 跨服务调用延迟 | P99 < 100ms（gRPC）|
| 全仓 lint 时间 | < 30 秒 |
| CI 全流程时间 | < 15 分钟 |

---

## 19. 各预留模块的接口契约（供未来扩展者参考）

### 19.1 前端预留接口

```yaml
# 已在核心架构中定义的对外 API（位于 services/api-gateway/）
# 前端实施时直接消费这些 API 即可

公共 REST API:
  POST   /v1/projects/{id}/design          提交设计任务
  GET    /v1/projects/{id}/pareto          获取 Pareto 前沿
  GET    /v1/molecules/{smiles}/fto        FTO 报告
  ...
  
WebSocket/SSE:
  GET /v1/stream/{job_id}                  Agent 思考链实时流
  GET /v1/audit/{run_id}                   审计事件流
```

### 19.2 湿实验室预留接口

```yaml
# 核心架构产出的 SSP（StructuredSynthesisProtocol）即是湿实验室的输入
# 已在 libs/mf-core/src/mf_core/types/ssp.py 中定义

预留接入点:
  - SSP → XDL 编译器：wetlab/xdl-compiler/
  - XDL → SiLA2 适配器：wetlab/sila2-adapter/
  - 各硬件驱动：wetlab/hardware-drivers/
  - 数据回流到 Incremental CLM：通过 NATS subject `wetlab.results.*`
```

### 19.3 商业化预留接口

```yaml
预留接入点:
  - 多租户隔离：commercial/multi-tenancy/
    （在 services/api-gateway 的 middleware 中扩展 tenant_id 维度）
  - 计量计费：commercial/billing/
    （订阅 NATS subject `oracle.calls.*` 累计计费）
  - GxP 合规：commercial/compliance/
    （已有的 Sigstore + Neo4j 谱系直接可用）
```

---

> **文档结束**
>
> 这份代码工程架构覆盖了：
> - **顶层组织**（Monorepo、Workspace、依赖纪律）
> - **8 大目录**的详细职责（protos/libs/services/agents/models/data/infra/tests）
> - **关键文件**的具体内容示例（Proto、Pydantic 模型、Service 模板、LangGraph 流程）
> - **插件化机制**（如何通过 entry-points 扩展生成器/Oracle）
> - **三个预留模块**的清晰接口（ui/wetlab/commercial）
>
> **核心特点**：
> 1. **灵活**：插件化让新增模型/Oracle 不需要动核心代码
> 2. **严谨**：协议优先 + 三层依赖纪律 + import-linter 强制执行
> 3. **可扩展**：Monorepo + uv workspace 支持任意子包
> 4. **可审计**：Sigstore + Neo4j 谱系贯穿所有代码层
> 5. **预留扩展**：前端/湿实验室/商业化目录已就位

---

*MoleculeForge Code Architecture Documentation v1.0*
*Generated: 2026-04-29*
