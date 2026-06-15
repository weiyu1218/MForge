# HUMU 预训练新数据接入方案

## 目标

先把已处理的新数据集接入 HUMU 预训练数据契约和验证链路，验证通过后再升级 HUMU 编码器结构。当前阶段不改 HUMU checkpoint 外部服务接口，不恢复 intent encoder，不创建并行 v2 训练入口。

## 当前证据

- 当前分支：`feature/humu-pretrain-optimization`。
- 最新提交：`9a4a555 chore: 同步 humu 预训练数据进度`。
- 当前工作区干净。
- `data/processing/humu_pretrain/manifest.json` 已记录 `protacdb`、`protacpedia`、`rcsb_mmcif` 等新组件。
- 新数据 manifest 显示：
  - `bindingdb_activity`: `n_records=3145942`，字段含 `ligand_smiles`、`target_id`、`activity_value`。
  - `protacpedia`: `n_records=1203`，字段含 `protac_canonical_smiles`、`e3_binder_canonical_smiles`、`ligand_canonical_smiles`、`linker_canonical_smiles`。
  - `protacdb`: `total_sdf_records=46115`，主表和组件表含 `canonical_smiles` / `smiles`。
  - `pdcdb` 与 `interface_skempi2` 样例没有稳定 SMILES 或可直接编码结构字段，不能直接进入当前三塔训练。
- HUMU 预训练主链路当前只支持 `mol`、`pocket`、`route` 三塔。
- `loss_weights.intent` 和 `data.intent_source` 已在 `data_loader.py` 与 `pipeline.py` 中 fail-fast 拒绝。

## 调用链路

```text
pipelines/humu_pretrain/train.py
  -> yaml.safe_load(config)
  -> humu_pretrain.pipeline.run(config)
  -> _validate_config(config)
  -> _build_encoders(config, device)
     -> HUMUMoleculeEncoder
     -> HUMUPocketEncoder
     -> HUMURouteEncoder
  -> humu_pretrain.data_loader.create_dataloaders(config)
     -> PairedHUMUDataset
     -> _record_collate
  -> _forward_paired_batch(encoders, paired_batch, config)
  -> _compute_losses(...)
  -> _log_step(...)
  -> _validate_epoch(...)
  -> _save_checkpoint(...)
```

## 文件影响

主要修改文件：

- `pipelines/humu_pretrain/src/humu_pretrain/data_loader.py`
  - 增加 activity 多源读取配置。
  - 增加 PROTAC 组件样本读取和数据契约校验。
  - 保持 `PairedHUMUDataset` 作为唯一训练 dataset 入口。
- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
  - 增加同一 molecule encoder 内的 PROTAC-component 对齐 loss。
  - 增加训练日志和 validation 聚合键。
  - 保持 checkpoint key 为 `encoder_mol`、`encoder_pocket`、`encoder_route`。
- `configs/models/humu_pretrain.yaml`
  - 引用 `bindingdb_activity` 作为 activity 第二来源。
  - 引用 `protacpedia` 作为默认 PROTAC 组件来源。
  - 不引用 `pdcdb`、`interface_skempi2`、`rcsb_mmcif` 进入训练。
- `tests/unit/test_humu_training.py`
  - 先写失败测试覆盖新数据 contract、loader、loss、config。

关联影响文件：

- `services/humu-encoder-svc/src/humu_encoder_svc/main.py`
  - 本阶段不改。服务仍只支持 molecule/pocket/route。
- `models/mf-encoders/*`
  - 本阶段不改编码器结构。

不修改文件：

- 不修改 CIG、HCIV、IntentCone、AMGE、orchestrator。
- 不更新 README，待本次代码验证通过后按用户确认再决定。

## 设计

### Activity 多源

新增 `data.activity_sources`，支持多个 JSONL 目录。保留 `data.activity_source` 作为向后兼容入口。preflight 对每个目录执行现有 activity record 校验。validation 的 activity cliff 标签从所有配置源合并，SMILES canonical 后按 ligand 聚合。

### PROTAC 组件数据

新增 `data.protac_sources`。每个 source 是目录路径，当前支持从以下文件读取：

- `protacpedia/protacpedia.jsonl`
  - anchor: `protac_canonical_smiles` 或 `protac_smiles`
  - component: `e3_binder_canonical_smiles`、`ligand_canonical_smiles`、`linker_canonical_smiles`
- `protacdb/protacdb.jsonl`
  - anchor: `canonical_smiles` 或 `smiles`
  - 若主表缺少可解析组件 SMILES，仅作为 PROTAC molecule 样本，不产生组件对。
- `protacdb/e3_ligand.jsonl`、`protacdb/warhead.jsonl`、`protacdb/linker.jsonl`、`protacdb/mg.jsonl`、`protacdb/xtac.jsonl`
  - 可作为组件 molecule 样本来源，但只有存在同一记录 anchor/component 配对时才生成 `protac_component` 对。

训练样本使用现有 `PairedHUMUDataset`，新增 `pair_type="protac_component"`：

```python
{
    "pair_type": "protac_component",
    "mol_id": "protac_component:<record_id>:<component>",
    "ligand_smiles": "<protac_smiles>",
    "component_smiles": "<component_smiles>",
    "component_type": "<e3_binder|target_ligand|linker|warhead>",
    "source_dataset": "<protacpedia|protacdb>",
    "split": "train",
    "pocket": None,
    "route": None,
}
```

`_forward_paired_batch()` 对 `protac_component` 样本用同一个 molecule encoder 编码 anchor 和 component，新增 `l_protac_component`。该 loss 只在配置 `loss_weights.protac_component > 0` 且 batch 中有有效配对时生效。

### 跳过数据

`pdcdb` 和 `interface_skempi2` 当前样例没有稳定 SMILES 或当前三塔可编码结构字段。preflight 可以报告它们未接入，但不把它们加入默认训练配置，避免伪造成功。

## KISS 四问

1. 这是现实问题还是想象问题？
   - 现实问题。新数据已处理但 manifest 明确未接入默认 HUMU 配置。
2. 有没有更简单做法？
   - 有。先扩展现有三塔数据契约，不引入新 encoder 或 v2 训练入口。
3. 会破坏什么？
   - 风险集中在 batch collate、loss key 和 activity validation。通过保留旧配置字段、默认新 loss 为 0、checkpoint key 不变降低风险。
4. 当前项目真的需要这个功能吗？
   - 需要。用户明确要求先将大分子、PROTAC 等相关数据接入 HUMU 预训练，验证通过后再升级模型结构。

## 风险

- PROTAC-DB 主表缺组件 SMILES 时不能生成组件对，只能作为 molecule 数据来源；不能臆造 linker 或 ligand 拆分。
- BindingDB 记录数很大，validation 加载全量会增加启动成本；实现要支持 `eval.activity_max_records` 限制，默认不限制以保持真实数据完整性，测试使用小样本。
- 同塔 PROTAC-component contrastive 是当前三塔约束下的最小训练信号，不等价于最终三元复合体几何建模。

## 实施计划

### 任务 1：测试 activity 多源合同

- 修改：`tests/unit/test_humu_training.py`
- 预期失败命令：

```bash
uv run pytest tests/unit/test_humu_training.py::test_preflight_reports_activity_sources_contract -q
```

- 预期失败：`KeyError` 或未识别 `activity_sources`。

### 任务 2：实现 activity 多源

- 修改：`pipelines/humu_pretrain/src/humu_pretrain/data_loader.py`
- 修改：`pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
- 验证命令：

```bash
uv run pytest tests/unit/test_humu_training.py::test_preflight_reports_activity_sources_contract tests/unit/test_humu_training.py::test_validate_epoch_computes_activity_cliff_auroc_from_activity_sources -q
```

### 任务 3：测试 PROTAC 组件样本合同

- 修改：`tests/unit/test_humu_training.py`
- 预期失败命令：

```bash
uv run pytest tests/unit/test_humu_training.py::test_paired_dataset_builds_protac_component_contract -q
```

- 预期失败：dataset 不产生 `protac_component` 样本。

### 任务 4：实现 PROTAC component loader

- 修改：`pipelines/humu_pretrain/src/humu_pretrain/data_loader.py`
- 验证命令：

```bash
uv run pytest tests/unit/test_humu_training.py::test_paired_dataset_builds_protac_component_contract tests/unit/test_humu_training.py::test_preflight_reports_protac_source_contract -q
```

### 任务 5：测试并实现 PROTAC component loss

- 修改：`tests/unit/test_humu_training.py`
- 修改：`pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
- 验证命令：

```bash
uv run pytest tests/unit/test_humu_training.py::test_forward_paired_batch_computes_protac_component_loss tests/unit/test_humu_training.py::test_compute_losses_uses_protac_component_objective -q
```

### 任务 6：更新默认配置

- 修改：`configs/models/humu_pretrain.yaml`
- 验证命令：

```bash
uv run python pipelines/humu_pretrain/train.py --config configs/models/humu_pretrain.yaml --preflight-only
```

### 任务 7：聚焦回归

验证命令：

```bash
uv run pytest tests/unit/test_humu_training.py tests/unit/test_learnable_curvature.py -q
```

## 自检

- 不创建并行 HUMU v2 训练入口。
- 不恢复 `intent_encoder`。
- 不把缺少 SMILES 的 PDCdb 和 SKEMPI2 强行接入训练。
- 不修改 HUMU encoder service API。
- 不声明模型质量提升，只声明数据合同和训练链路接入验证结果。
