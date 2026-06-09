# CoreArchitecture v2 阶段 B 执行文档

> 阶段 B 是算法补全，不是模型资源补全。所有改动必须接入仓内已有算法模块或真实 runner contract；缺 checkpoint、候选源、外部 runner 时必须 fail-fast，不能生成随机、hash、硬编码成功结果。

## 目标

完成 `docs/architecture/current-implementation-vs-corearchitecture-v2.md` 中「阶段 B — 算法补全」的 7 项工作：

1. 把 iCLM / UAS / CReM / FragFM 孤立算法模块接入 `generator.generate`。
2. `SRBAgent.process` 接 `compile_ssp`。
3. `ValidationAgent` L1-L3 使用 `predict_with_uncertainty` 和不确定度阈值。
5. HUMU 加入可学习曲率。
6. PAINS 过滤接入 L0 oracle。
7. `mf-eval` 补全 `distortion`、`cliff_analysis`、`hv_evaluator`。

## 非目标

- 不训练 iCLM、UAS、FragFM、CReM 的真实权重。
- 不接入阶段 C 的真实 ESM-2、AiZynthFinder、chemprop、OpenTelemetry 或 Sigstore。
- 不绕过缺 runner、缺 checkpoint、缺候选源、缺真实数据的问题。
- 不把 benchmark 或 E2E 测试从 skip 改成伪通过。
- 不更新 `README.md`，直到阶段 B 代码完成后向用户简述变更并获得确认。

## 当前证据

### 生成器孤立模块

- `models/mf-generators/incremental_clm/src/mf_generators/incremental_clm/generator.py` 当前通过 MD5 从 10 个 SMILES 固定池选择分子，未引用 `learning/online_learner.py`、`model/ewc_regularizer.py`、`model/packnet.py`。
- `models/mf-generators/uas/src/mf_generators/uas/generator.py` 内有 `_Autoencoder`，但 `generate()` 只转发 `runner.generate()`；`autoencoder/molecule_ae.py`、`sampler/ood_aware_sampling.py`、`unfamiliarity_estimator.py` 未进入主路径。
- `models/mf-generators/crem_3d/src/mf_generators/crem_3d/generator.py` 从 JSON `mutations` 轮询 `product`；`fragment_replacement.py` 的 `get_attachment_points()` 和 `replace_fragment()` 未进入主路径。
- `models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py` 轮询 `assembly_rules.product`；`model/two_level_dfm.py`、`model/sa_aware_rate_matrix.py` 只存在定义，未被 `generate()` 调用。

### SRB / Validation / HUMU / Oracle / Eval

- `agents/srb_agent/src/srb_agent/agent.py` 当前 `process()` 返回 `steps=[]`、`estimated_overall_yield=0.0` 的 protocol 占位，不调用 `compile_ssp()`。
- `agents/srb_agent/src/srb_agent/compiler.py` 已有真实 `compile_ssp(molecule, retrosyn_route, run_id)`，要求 `retrosyn_route.route_id` 和非空 `steps`。
- `agents/validation_agent/src/validation_agent/agent.py` 当前 L0 检查 `admet_score >= l0_threshold`；L1-L3 只要不是 skipped 就恒 `return True`。
- 多个 oracle wrapper 已有 `predict_with_uncertainty`：`models/mf-oracles/gnina`、`diffdock_l`、`boltz2`、`openfe`、`admet_ai`。
- `libs/mf-humu/src/mf_humu/operations/dead_zone.py` 当前用欧氏距离、Python 双重循环。
- `libs/mf-humu/src/mf_humu/manifold/lorentz.py` 当前 `self.k = curvature` 是普通 float，不是 `nn.Parameter`。
- `models/mf-oracles/rdkit-oracle/src/mf_oracles/rdkit_oracle/scorer.py` 当前计算 SA、QED、Lipinski、composite，没有 PAINS。
- `libs/mf-eval/src/mf_eval/` 当前只有 `molecule/moses.py`，没有 `distortion.py`、`cliff_analysis.py`、`hv_evaluator.py`。

## 调用链路分析

### 生成器调用链

```text
Generator service / agent / tests
  -> generator.generate(...)
  -> generator-specific artifact / runner / algorithm module
  -> RDKit validity check
  -> mf_core.types.molecule.Molecule or MoleculeModel
```

阶段 B 的原则：

- 已有 runner contract 的生成器仍优先使用真实 runner。
- 仓内已有算法模块只在输入资源满足 contract 时接入。
- 资源不足时抛出明确错误，不降级到 hash、random 或固定池。

### SRB 调用链

```text
SRBAgent.process(data)
  -> data["molecule"] / data["smiles"]
  -> data["retrosyn_route"] or data["pathways"][i]
  -> compile_ssp(molecule, route, run_id)
  -> SSP Pydantic object
  -> protocol dict / optional XDL export
```

当前断点是 agent 没调用 compiler。阶段 B 只接 `compile_ssp`，不 invent route。

### Validation 调用链

```text
ValidationAgent.process(data)
  -> L0 oracle evaluate / predict_with_uncertainty
  -> L1-L3 oracle predict_with_uncertainty
  -> score thresholds + uncertainty thresholds
  -> cascade result with scores and uncertainty
```

当前断点是 L1-L3 无阈值逻辑。阶段 B 使用每级显式 threshold；缺 threshold 时使用保守默认值并写入返回结果。

### HUMU 曲率链路

```text
HUMU encoders / LorentzAttention / HUMU training pipeline
  -> LorentzManifold(curvature=...)
  -> origin / project / expmap / logmap / distance
  -> curvature regularization / checkpoints
```

当前断点是 `LorentzManifold` 不是 `nn.Module`，曲率不进入 optimizer。阶段 B 增加可选 learnable curvature wrapper，同时保持现有 float API 兼容。

## 文件影响评估

### 主要修改文件

- `models/mf-generators/incremental_clm/src/mf_generators/incremental_clm/generator.py`
- `models/mf-generators/incremental_clm/src/mf_generators/incremental_clm/learning/online_learner.py`
- `models/mf-generators/incremental_clm/src/mf_generators/incremental_clm/model/ewc_regularizer.py`
- `models/mf-generators/incremental_clm/src/mf_generators/incremental_clm/model/packnet.py`
- `models/mf-generators/uas/src/mf_generators/uas/generator.py`
- `models/mf-generators/uas/src/mf_generators/uas/autoencoder/molecule_ae.py`
- `models/mf-generators/uas/src/mf_generators/uas/sampler/ood_aware_sampling.py`
- `models/mf-generators/uas/src/mf_generators/uas/unfamiliarity_estimator.py`
- `models/mf-generators/crem_3d/src/mf_generators/crem_3d/generator.py`
- `models/mf-generators/crem_3d/src/mf_generators/crem_3d/fragment_replacement.py`
- `models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py`
- `models/mf-generators/fragfm/src/mf_generators/fragfm/model/two_level_dfm.py`
- `models/mf-generators/fragfm/src/mf_generators/fragfm/model/sa_aware_rate_matrix.py`
- `agents/srb_agent/src/srb_agent/agent.py`
- `agents/validation_agent/src/validation_agent/agent.py`
- `libs/mf-humu/src/mf_humu/operations/dead_zone.py`
- `libs/mf-humu/src/mf_humu/manifold/lorentz.py`
- `libs/mf-humu/src/mf_humu/encoders/lorentz_attention.py`
- `models/mf-encoders/humu_mol_encoder/src/mf_encoders/humu_mol/encoder.py`
- `models/mf-encoders/humu_pocket_encoder/src/mf_encoders/humu_pocket/encoder.py`
- `models/mf-encoders/humu_route_encoder/src/mf_encoders/humu_route/encoder.py`
- `models/mf-encoders/humu_intent_encoder/src/mf_encoders/humu_intent/encoder.py`
- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
- `models/mf-oracles/rdkit-oracle/src/mf_oracles/rdkit_oracle/scorer.py`
- `models/mf-oracles/rdkit-oracle/src/mf_oracles/rdkit_oracle/oracle.py`
- `libs/mf-eval/src/mf_eval/__init__.py`

### 需要新增的必要文件

- `libs/mf-humu/src/mf_humu/manifold/learnable_lorentz.py`：封装可学习曲率，保留现有 `LorentzManifold` float API。
- `libs/mf-eval/src/mf_eval/distortion.py`：流形/嵌入距离失真评估。
- `libs/mf-eval/src/mf_eval/cliff_analysis.py`：activity cliff 分析。
- `libs/mf-eval/src/mf_eval/hv_evaluator.py`：hypervolume 和 hypervolume improvement 评估。
- `tests/unit/test_phase_b_generators.py`：覆盖四个生成器接入仓内算法模块。
- `tests/unit/test_dead_zone.py`：覆盖 Lorentz 批量 dead zone。
- `tests/unit/test_learnable_curvature.py`：覆盖曲率参数进入 optimizer 和保持正值。
- `tests/unit/test_mf_eval.py`：覆盖 distortion、cliff、HV evaluator。

新增文件均对应阶段 B 明确要求，不创建示例版、占位版或临时测试文件。

### 关联测试文件

- `tests/unit/test_generators.py`
- `tests/unit/test_srb_agent.py`
- `tests/unit/test_validation_agent.py`
- `tests/unit/test_lorentz_manifold.py`
- `tests/unit/test_humu_training.py`
- `tests/unit/test_l0_oracle.py`
- `tests/unit/test_rdkit_generator.py`

## 执行顺序

### 任务 0：建立阶段 B 基线

执行命令：

```bash
uv run pytest tests/unit/test_generators.py -q
uv run pytest tests/unit/test_srb_agent.py -q
uv run pytest tests/unit/test_validation_agent.py -q
uv run pytest tests/unit/test_lorentz_manifold.py -q
uv run pytest tests/unit/test_l0_oracle.py -q
```

通过标准：记录当前通过、失败或 collection error 状态。若阶段 A 尚未完成导致 collection error，先完成阶段 A 或记录阻塞，不在阶段 B 中绕过阶段 A 问题。

### 任务 1：iCLM 接入 EWC / PackNet / OnlineLearner

修改：

- `models/mf-generators/incremental_clm/src/mf_generators/incremental_clm/generator.py`
- `models/mf-generators/incremental_clm/src/mf_generators/incremental_clm/learning/online_learner.py`
- `models/mf-generators/incremental_clm/src/mf_generators/incremental_clm/model/ewc_regularizer.py`
- `models/mf-generators/incremental_clm/src/mf_generators/incremental_clm/model/packnet.py`
- `tests/unit/test_phase_b_generators.py`

执行内容：

- `IncrementalCLMGenerator.__init__` 增加可选 `model`、`tokenizer`、`decoder`、`online_learner`、`ewc_regularizer`、`packnet` 参数。
- `generate()` 逻辑：
  - 有 `runner` 时调用真实 runner。
  - 无 runner 但有 `model` 和 `decoder` 时，用模型输出经 decoder 得到 SMILES。
  - 有 `online_batch` 时调用 `OnlineLearner.update()`。
  - 有 `ewc_regularizer` 时把 `ewc_loss` 写入 metadata。
  - 有 `packnet` 时在生成前调用 `apply_mask()`。
  - 无 runner 且无 model/decoder 时抛出 `RuntimeError("IncrementalCLM model or runner is required")`。
- 删除 MD5 固定池路径。

验证：

```bash
uv run pytest tests/unit/test_phase_b_generators.py::test_iclm_requires_model_or_runner -q
uv run pytest tests/unit/test_phase_b_generators.py::test_iclm_uses_online_learner_ewc_and_packnet -q
uv run pytest tests/anti_degradation/test_no_degradation.py -q
```

通过标准：iCLM 不再出现 MD5 固定池生成；无资源 fail-fast；有 fixture model 时走 OnlineLearner/EWC/PackNet。

### 任务 2：UAS 接入 autoencoder / OOD sampler / unfamiliarity estimator

修改：

- `models/mf-generators/uas/src/mf_generators/uas/generator.py`
- `models/mf-generators/uas/src/mf_generators/uas/autoencoder/molecule_ae.py`
- `models/mf-generators/uas/src/mf_generators/uas/sampler/ood_aware_sampling.py`
- `models/mf-generators/uas/src/mf_generators/uas/unfamiliarity_estimator.py`
- `tests/unit/test_phase_b_generators.py`

执行内容：

- `UASGenerator.__init__` 增加可选 `candidate_source`、`reference_embeddings`、`decoder`、`unfamiliarity_threshold`。
- `generate()` 逻辑：
  - 有 runner 时保持 runner 主路径。
  - 无 runner 时要求 `candidate_source`、`reference_embeddings`、`decoder` 同时存在。
  - 使用 `MoleculeAutoencoder` 或现有 `_Autoencoder` 计算 reconstruction error。
  - 使用 `OODAwareSampler` 根据 `compute_unfamiliarity()` 过滤候选 embedding。
  - decoder 将 accepted embedding 转为 SMILES；RDKit 可用时校验 SMILES 合法。
  - 缺资源抛出明确 `RuntimeError`。
- 消除 `generator.py` 内 `_Autoencoder` 与 `autoencoder/molecule_ae.py` 的重复实现，保留一个实现并更新测试。

验证：

```bash
uv run pytest tests/unit/test_phase_b_generators.py::test_uas_requires_candidate_source_reference_and_decoder -q
uv run pytest tests/unit/test_phase_b_generators.py::test_uas_filters_candidates_by_unfamiliarity -q
uv run pytest tests/unit/test_generators.py::TestUASGenerator -q
```

通过标准：runner 路径兼容；非 runner 路径真实调用 AE、unfamiliarity estimator 和 OOD sampler。

### 任务 3：CReM 接入 fragment_replacement

修改：

- `models/mf-generators/crem_3d/src/mf_generators/crem_3d/generator.py`
- `models/mf-generators/crem_3d/src/mf_generators/crem_3d/fragment_replacement.py`
- `tests/unit/test_phase_b_generators.py`
- `tests/unit/test_generators.py`

执行内容：

- 扩展 CReM MMP artifact record：
  - `id`
  - `seed_smiles`
  - `fragment_smiles`
  - `attachment_index`
  - `product` 可选，用于 curated product 兼容。
- `generate()` 在 `fragment_smiles` 存在时：
  - 解析 `seed_smiles` 或调用入参 `seed_smiles`。
  - 使用 `get_attachment_points()` 得到候选 attachment。
  - 使用 `replace_fragment()` 生成 RDKit mol。
  - canonicalize 后返回 `Molecule`。
- `product` 仅作为已验证 artifact 的兼容路径；不能作为 fragment replacement 已接入的唯一证据。
- `replace_fragment()` 不吞掉所有异常；对 RDKit sanitize 失败返回 `None`，对配置错误抛出明确异常。

验证：

```bash
uv run pytest tests/unit/test_phase_b_generators.py::test_crem_generate_uses_fragment_replacement -q
uv run pytest tests/unit/test_generators.py::TestCReM3DGenerator -q
```

通过标准：新测试能证明 `generate()` 使用 `replace_fragment()` 产物；既有 product artifact 测试不退化。

### 任务 4：FragFM 接入 TwoLevelDFM / SAAwareRateMatrix / FragmentVocabulary

修改：

- `models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py`
- `models/mf-generators/fragfm/src/mf_generators/fragfm/model/two_level_dfm.py`
- `models/mf-generators/fragfm/src/mf_generators/fragfm/model/sa_aware_rate_matrix.py`
- `models/mf-generators/fragfm/src/mf_generators/fragfm/model/fragment_vocabulary.py`
- `tests/unit/test_phase_b_generators.py`
- `tests/unit/test_generators.py`

执行内容：

- `FragFMGenerator.__init__` 增加可选 `model`、`rate_matrix`、`decoder`。
- `checkpoint_path` 存在时加载 `TwoLevelDFM` state_dict；不存在时继续要求 `vocab_path`，不伪造 checkpoint。
- `generate()`：
  - 使用 `FragmentVocabulary.encode()` 编码 assembly rule fragments。
  - 使用 `SAAwareRateMatrix` 根据 `sa_score_bin` 计算 transition scores。
  - 有 model/decoder 时通过 `TwoLevelDFM` logits 解码产品。
  - 无 model/decoder 时仍可使用 artifact assembly_rules，但要在 metadata 写入 `rate_matrix_applied=True` 和 `fragment_indices`，证明 vocabulary/rate_matrix 已参与排序。
- 不把 DFM logits 随机采样作为生产结果；测试 fixture decoder 必须确定。

验证：

```bash
uv run pytest tests/unit/test_phase_b_generators.py::test_fragfm_uses_vocabulary_and_sa_rate_matrix -q
uv run pytest tests/unit/test_generators.py::TestFragFMGenerator -q
```

通过标准：FragFM 主路径使用 `FragmentVocabulary` 与 `SAAwareRateMatrix`；有 fixture model 时使用 `TwoLevelDFM`。

### 任务 5：SRBAgent.process 接 compile_ssp

修改：

- `agents/srb_agent/src/srb_agent/agent.py`
- `tests/unit/test_srb_agent.py`

执行内容：

- `process(data)` 接受：
  - `run_id`
  - `molecule` 或 `smiles`
  - `retrosyn_route` 或 `pathways`
- 对每条 route 调用 `compile_ssp(molecule, route, run_id)`。
- 返回 `protocols` 中每项包含：
  - `ssp_id`
  - `route_id`
  - `target_smiles`
  - `materials`
  - `steps`
  - `total_estimated_yield`
  - `total_estimated_cost_usd`
  - `xdl_version`
- 缺 route steps 或 route_id 时让 `compile_ssp` 的错误向上传播；不生成空 steps。

验证：

```bash
uv run pytest tests/unit/test_srb_agent.py::test_srb_agent_process_compiles_ssp -q
uv run pytest tests/unit/test_srb_agent.py -q
```

通过标准：`SRBAgent.process` 返回真实 SSP 字段，不再返回空 steps 占位。

### 任务 6：ValidationAgent L1-L3 接 predict_with_uncertainty 和阈值

修改：

- `agents/validation_agent/src/validation_agent/agent.py`
- `tests/unit/test_validation_agent.py`

执行内容：

- 增加每级默认配置：
  - L0：`admet_score >= l0_threshold`。
  - L1：`docking_score <= l1_max_docking_score`，默认 `-6.0`。
  - L2：`affinity <= l2_max_affinity`，默认 `-7.0`。
  - L3：`rbfe <= l3_max_rbfe`，默认 `0.0`。
  - 每级 uncertainty 默认阈值：`l1_max_uncertainty=1.0`、`l2_max_uncertainty=1.0`、`l3_max_uncertainty=1.0`。
- Oracle 调用优先顺序：
  - 有 `predict_with_uncertainty`：调用并解析 `(scores, uncertainty)` 或 `{smiles: (scores, uncertainty)}`。
  - 只有 `evaluate`：保持兼容，但结果必须包含 required score；uncertainty 记为 `None`。
- `skipped=True` 仍按现有逻辑通过，但 cascade 中必须保留 skip reason。
- L1-L3 分数不达阈值或 uncertainty 超阈值时停止 cascade。

验证：

```bash
uv run pytest tests/unit/test_validation_agent.py -q
```

通过标准：新增测试覆盖 L1 分数失败、L2 uncertainty 失败、L3 skipped 兼容；原有 L0 停止逻辑不退化。


修改：

- `libs/mf-humu/src/mf_humu/operations/dead_zone.py`
- `tests/unit/test_dead_zone.py`

执行内容：

- 函数签名增加可选 `manifold: LorentzManifold | None = None`。
- 将 `dead_zones: list[Tensor]` 合并为 `(n_total, d+1)`。
- 使用 Lorentz 距离：
  - `inner = -z0*dz0 + sum(z_i*dz_i)`
  - `distance = arccosh(clamp(-inner, min=1+eps))`
- 用 batch matrix 运算得到 `(batch, n_total)` 距离矩阵。
- potential 使用 `exp(-min_distance^2 / (2 * radius^2))`。
- 空 dead_zones 返回全 0 tensor。

验证：

```bash
uv run pytest tests/unit/test_dead_zone.py -q
uv run pytest tests/unit/test_humu_training.py -q
```

通过标准：新实现无 Python 双重 for-loop；结果 shape 正确；HUMU training 不退化。

### 任务 8：HUMU 加入可学习曲率

修改/新增：

- `libs/mf-humu/src/mf_humu/manifold/learnable_lorentz.py`
- `libs/mf-humu/src/mf_humu/manifold/lorentz.py`
- `libs/mf-humu/src/mf_humu/encoders/lorentz_attention.py`
- `models/mf-encoders/humu_mol_encoder/src/mf_encoders/humu_mol/encoder.py`
- `models/mf-encoders/humu_pocket_encoder/src/mf_encoders/humu_pocket/encoder.py`
- `models/mf-encoders/humu_route_encoder/src/mf_encoders/humu_route/encoder.py`
- `models/mf-encoders/humu_intent_encoder/src/mf_encoders/humu_intent/encoder.py`
- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
- `tests/unit/test_learnable_curvature.py`
- `tests/unit/test_lorentz_manifold.py`
- `tests/unit/test_humu_training.py`

执行内容：

- 保持 `LorentzManifold(curvature=float)` 的现有 API。
- 新增 `LearnableLorentzManifold(nn.Module)`：
  - 内部参数为 `raw_curvature = nn.Parameter(...)`。
  - 对外 `k` 和 `c` 通过 `softplus(raw_curvature) + eps` 保证正值。
  - 提供与 `LorentzManifold` 一致的 `origin()`、`inner()`、`distance()`、`expmap()`、`logmap()`、`project_tangent()`。
- 编码器增加 `learnable_curvature: bool = False` 参数；默认 false 保持兼容。
- HUMU training config 增加可选 `learnable_curvature`，启用后 optimizer 包含曲率参数。
- checkpoint 保存和恢复包含曲率参数。

验证：

```bash
uv run pytest tests/unit/test_learnable_curvature.py -q
uv run pytest tests/unit/test_lorentz_manifold.py -q
uv run pytest tests/unit/test_humu_training.py -q
```

通过标准：默认 float 曲率测试仍通过；启用 learnable 后参数在 `named_parameters()` 中可见且梯度可回传。

### 任务 9：PAINS 过滤接入 L0 oracle

修改：

- `models/mf-oracles/rdkit-oracle/src/mf_oracles/rdkit_oracle/scorer.py`
- `models/mf-oracles/rdkit-oracle/src/mf_oracles/rdkit_oracle/oracle.py`
- `tests/unit/test_l0_oracle.py`

执行内容：

- 使用 RDKit `FilterCatalog` 的 PAINS catalogs；如果当前 RDKit build 不提供 catalog，抛出 `RuntimeError("RDKit PAINS filter catalog is unavailable")`，不使用手写假规则。
- 新增函数：
  - `has_pains_alert(smiles: str) -> bool | None`
  - `pains_alerts(smiles: str) -> list[str]`
- `compute_composite_score()` 对 PAINS 命中分子施加明确 penalty，penalty 值写成常量。
- `RDKitOracle.predict_with_uncertainty()` 返回 metadata 需求时包含 PAINS 信息；若现有 API 只能返回 `(score, uncertainty)`，则新增 `evaluate()` 返回 dict，不破坏 `predict()`。

验证：

```bash
uv run pytest tests/unit/test_l0_oracle.py -q
```

通过标准：PAINS 阳性 fixture 分数低于非 PAINS 对照；RDKit catalog 不可用时测试明确 skip 或 fail-fast，不伪造 PAINS。

### 任务 10：补全 mf-eval distortion / cliff_analysis / hv_evaluator

修改/新增：

- `libs/mf-eval/src/mf_eval/distortion.py`
- `libs/mf-eval/src/mf_eval/cliff_analysis.py`
- `libs/mf-eval/src/mf_eval/hv_evaluator.py`
- `libs/mf-eval/src/mf_eval/__init__.py`
- `tests/unit/test_mf_eval.py`

执行内容：

- `distortion.py`：
  - `pairwise_distance_distortion(source_distances, embedding_distances) -> dict`
  - 返回 `mean_absolute_error`、`mean_relative_error`、`spearman_r`、`n_pairs`。
  - 输入 shape 不一致时抛 `ValueError`。
- `cliff_analysis.py`：
  - `find_activity_cliffs(smiles, activities, similarity_threshold, activity_delta_threshold) -> list[dict]`。
  - 用 RDKit ECFP4 + Tanimoto；RDKit 不可用时抛 `RuntimeError`。
  - `cliff_separation_auroc(embeddings, cliff_labels) -> float | None`，缺正负样本返回 `None`。
- `hv_evaluator.py`：
  - `filter_non_dominated(points, maximize=True)`。
  - `hypervolume_2d(points, reference, maximize=True)`。
  - `hypervolume_improvement(candidate, front, reference, maximize=True)`。
  - 先实现 2D 明确 contract；多维输入抛 `ValueError`，避免伪通用。

验证：

```bash
uv run pytest tests/unit/test_mf_eval.py -q
uv run pytest tests/unit/test_humu_training.py -q
```

通过标准：三个模块有确定单元测试；HUMU training 中已有 cliff metric 缺标签逻辑不退化。

## 总体验证

阶段 B 全部任务完成后执行：

```bash
uv run pytest tests/unit/test_phase_b_generators.py -q
uv run pytest tests/unit/test_generators.py -q
uv run pytest tests/unit/test_srb_agent.py -q
uv run pytest tests/unit/test_validation_agent.py -q
uv run pytest tests/unit/test_dead_zone.py -q
uv run pytest tests/unit/test_learnable_curvature.py -q
uv run pytest tests/unit/test_lorentz_manifold.py -q
uv run pytest tests/unit/test_l0_oracle.py -q
uv run pytest tests/unit/test_mf_eval.py -q
uv run pytest tests/unit/test_humu_training.py -q
uv run pytest tests/anti_degradation/test_no_degradation.py -q
```

如果阶段 A 已完成且 unit suite 可 collect，再执行：

```bash
uv run pytest tests/unit -q
```

## KISS 四问

1. 这是现实问题还是想象问题：是现实问题，阶段 B 的 7 项都有当前代码证据支撑。
2. 有没有更简单的做法：有，只接入仓内已有算法模块和真实 contract，不引入新训练系统。
3. 会破坏什么：主要风险是生成器 API、HUMU checkpoint 兼容、Validation 阈值行为；每项都有独立回归测试。
4. 当前项目真的需要这个功能吗：需要。阶段 B 是让现有算法文件进入真实调用链，为阶段 C 的真实模型资源接入提供稳定接口。

## 风险与处理

- iCLM 没有真实 CLM 模型类：阶段 B 只接可注入 `model/decoder/runner` contract；缺资源 fail-fast。
- UAS 缺候选源和 decoder：非 runner 路径必须要求 `candidate_source/reference_embeddings/decoder`，不自动生成候选。
- FragFM checkpoint 缺失：保留 artifact assembly path，但必须让 vocabulary/rate matrix 参与排序或 metadata；不伪造 DFM 权重。
- PAINS catalog 不可用：报告 RDKit build 问题，不手写假 PAINS。
- Learnable curvature 改动影响 checkpoint：默认 `learnable_curvature=False`，老 checkpoint 路径保持兼容。
- 多维 hypervolume 容易过度实现：阶段 B 只实现 2D contract，多维明确报错。

## 完成标准

- 阶段 B 7 项均有对应代码和测试覆盖。
- 四个生成器不再依赖孤立未调用模块或固定伪生成路径。
- SRB、Validation、Dead Zone、HUMU curvature、L0 PAINS、mf-eval 均有最小验证命令输出。
- 所有不能通过的验证必须原样报告 stderr、失败测试名和下一步处理选项。
