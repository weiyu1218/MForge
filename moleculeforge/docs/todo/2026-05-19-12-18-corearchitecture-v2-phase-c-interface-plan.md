# CoreArchitecture v2 阶段 C 接口接入方案

## 目标

按当前阶段边界只接入阶段 C 的 1、2：

- ESM-2 接入 HUMU pocket encoder。
- AiZynthFinder 接入 retrosyn service / retrosyn agent / SRB 上游 route 链路。


## 当前证据

- ESM-2 权重存在于 `models/esm2/esm2_t33_650M_UR50D.pt`，alphabet 文件存在于 `models/esm2/esm2_alphabet.pkl`。
- `configs/models/humu_pretrain.yaml` 当前 pocket encoder 配置写入了 `use_esm2: true`，但 checkpoint 路径指向不存在的 `checkpoints/esm2/esm2_t33_650M_UR50D.pt`。
- 当前环境缺少 `esm` Python 包，不能声称 ESM-2 已可运行加载。
- `HUMUPocketEncoder` 当前只消费 pocket atom 坐标、元素、残基类型，未消费 protein sequence 或 ESM-2 embedding。
- `AiZynthRetrosyn` wrapper 已存在，要求 runner 返回带 route_id 与完整 steps 的 route。
- `services/retrosyn-svc` 当前只检查 `RETROSYN_RUNNER_URI` / `RETROSYN_SCORER_URI`，检查后仍抛出未配置错误。
- `RetroSynAgent.process()` 当前返回空 routes，不调用 `AiZynthRetrosyn`。
- 当前环境缺少 `aizynthfinder` Python 包，且未发现 AiZynthFinder config / USPTO policy / stock 模型路径。

## 调用链路分析

### ESM-2 pocket encoder

目标链路：

```text
HUMU train config
  -> _build_encoders()
  -> HUMUPocketEncoder(use_esm2=True, esm2_checkpoint=...)
  -> pocket_data sequence/protein_sequence or precomputed esm2_embedding
  -> ESM-2 residue representation mean pooling
  -> projection into pocket point-cloud representation
  -> LorentzAttention
  -> Lorentz embedding
```

当前断点：

```text
config has use_esm2
  -> _build_encoders ignores encoders.pocket settings
  -> HUMUPocketEncoder has no ESM-2 loader
  -> pocket data has no sequence field in existing CrossDocked sidecars
```

### AiZynthFinder retrosyn

目标链路：

```text
Retrosyn request / agent payload
  -> AiZynthRunner from local AiZynthFinder config or injected runner
  -> AiZynthRetrosyn.find_routes()
  -> validate_retrosyn_routes()
  -> Retrosyn service SyntheticRoute response
  -> SRB receives retrosyn_route.steps and compiles SSP
```

当前断点：

```text
AiZynthRetrosyn wrapper exists
  -> no concrete AiZynthFinder runner adapter
  -> service does not call wrapper
  -> agent does not call wrapper
```

## 实施方案

### 任务 1：ESM-2 encoder 接线

修改：

- `models/mf-encoders/humu_pocket_encoder/src/mf_encoders/humu_pocket/encoder.py`
- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
- `configs/models/humu_pretrain.yaml`
- `tests/unit/test_humu_training.py`

策略：

- `HUMUPocketEncoder` 增加 `use_esm2`、`esm2_checkpoint`、`esm2_layer`、`esm2_dim` 参数。
- 支持两种真实输入：
  - `protein_sequence` / `sequence`：需要 `esm` 包和 checkpoint，加载 ESM-2 后计算 mean pooled representation。
  - `esm2_embedding`：用于离线预计算特征和单元测试，不要求加载 2.5G 权重。
- `use_esm2=True` 时，既无 sequence 也无 `esm2_embedding` 直接报错，不回退到伪特征。
- `_build_encoders()` 将 `cfg["encoders"]["pocket"]` 传入 `HUMUPocketEncoder`。
- 修正默认 ESM-2 路径为 `models/esm2/esm2_t33_650M_UR50D.pt`。

### 任务 2：AiZynthFinder runner 接线

修改：

- `models/mf-retrosyn/aizynth_wrapper/src/mf_retrosyn/aizynth/retrosyn.py`
- `services/retrosyn-svc/src/retrosyn_svc/main.py`
- `agents/retrosyn_agent/src/retrosyn_agent/agent.py`
- `tests/unit/test_indexing_pipelines.py`
- `tests/unit/test_service_artifact_status.py`

策略：

- 新增 `AiZynthFinderRunner`，只在配置了真实 `AIZYNTH_CONFIG_PATH` 且 `aizynthfinder` 可导入时加载。
- `AiZynthRetrosyn.from_env()` 从环境构造 runner，缺包或缺 config 时 fail-fast。
- `RetrosynServicer.FindRoutes()` 根据 `engine` 调用 AiZynth wrapper，并把完整 route 压缩映射到现有 proto 的 `SyntheticRoute` 字段。
- `RetroSynAgent.process()` 调用 AiZynth wrapper，返回真实 routes；缺依赖时抛出明确错误。
- 不新增 RSGPT/UAlign 真实 runner，不新增伪 route。

## KISS 四问

1. 这是现实问题还是想象问题？
   - 是现实问题。代码和配置已经存在接口断点，阶段 C 明确要求接入 1、2。
2. 有没有更简单的做法？
   - 有。只接已有 wrapper 和配置入口，不重写 encoder 架构，不创建并行服务。
3. 会破坏什么？
   - 风险是默认启用 ESM-2 后旧 pocket fixture 无 sequence 失败。控制方式是训练配置显式启用，测试和旧路径可使用 `use_esm2=False` 或提供 `esm2_embedding`。
4. 当前项目真的需要这个功能吗？
   - 需要。它们是阶段 C 的明确接口接入项，且已有权重/wrapper 基础。

## 验证

- 先写失败测试：
  - pocket encoder 启用 ESM-2 时必须使用 `esm2_embedding` 或 sequence，且 embedding 会改变输出。
  - `_build_encoders()` 会传递 pocket ESM-2 配置。
  - Retrosyn service 会调用 runner 并返回 route。
  - Retrosyn agent 会返回真实 route，缺 runner 时 fail-fast。
- 实现后运行：
  - `uv run pytest tests/unit/test_humu_training.py tests/unit/test_indexing_pipelines.py tests/unit/test_service_artifact_status.py -q`
  - `uv run pytest tests/anti_degradation/test_no_degradation.py -q`

