# HUMU 预训练阻断修复验收日志

日期：2026-06-13

## 背景

本日志跟踪 HUMU 默认预训练配置进入大规模训练前的三类阻断修复：

1. ESM/结构输入合同不一致：默认启用 `encoders.pocket.use_esm2: true`，但 `protein_interface` / 结构数据源可只提供几何结构字段。
2. SKEMPI2 多突变解析不完整：真实 SKEMPI2 数据中存在大量逗号分隔多突变，且单突变字符串格式应解释为 `wildtype + chain + residue_number + mutant`。
3. PROTAC-DB 采样覆盖缺失：PROTAC-DB 组件库样本被加载为 `protac_component_library`，但默认目标采样只覆盖 `protac_component`。

这些问题会影响默认配置的 one-batch forward gate，因此修复顺序为：
SKEMPI2 → ESM/结构合同 → PROTAC-DB 覆盖 → 默认 one-batch forward gate。

## 证据

### SKEMPI2 多突变

- 旧 `_parse_skempi_mutation("LI38G")` 将第一个氨基酸 `L` 误作为 chain，科学含义错误；正确 chain 为 `I`。
- 旧 `_parse_skempi_mutation("SI40E,RI39M")` 返回不可解析，导致真实多突变记录被跳过或在加载时失败。
- 默认配置小样本 dataloader 曾在 SKEMPI2 记录上抛出 `ValueError: SKEMPI2 record requires parseable mutation`。

### ESM/结构合同

- 默认配置启用 pocket ESM2。
- `rcsb_mmcif` / SKEMPI2 / PDCdb 结构视图可以只提供 `coords`、`elements`、`residue_types`。
- 当前 one-batch forward 会在结构源上遇到 ESM2 输入缺失风险。

### PROTAC-DB 覆盖

- PROTAC-DB 组件库样本的 `pair_type` 是 `protac_component_library`，`source_name` 是 `protacdb`。
- 默认 objective sampling 只配置 `protac_component`。
- 2026-06-13 小样本 dataloader 回归显示 `source_counts["protacdb"] == 0`，说明默认采样没有覆盖 PROTAC-DB。

## 修复计划

1. SKEMPI2：
   - 增加 `_parse_skempi_mutations()`，返回 mutation dict 列表。
   - 支持逗号、分号和空白分隔的多突变。
   - 将 `LI38G` 解释为 `wildtype=L, chain_id=I, residue_number=38, mutant=G`。
   - `_mutated_residue_payload()` 对所有突变逐一应用 residue type 更新。
   - `_iter_interface_mutation_records()` 只过滤不可解析突变，不再过滤可解析多突变。

2. ESM/结构合同：
   - 在 pocket encoder 配置中增加源级 ESM 必需策略。
   - 对 `pocket` / `joint` 等原始口袋源保持 ESM 必需。
   - 对 `rcsb_mmcif`、`interface_skempi2`、`pdcdb` 等结构源允许 geometry-only 前向，除非样本明确带有 ESM 输入。

3. PROTAC-DB：
   - 增加 `protac_component_library` objective / loss / metric 覆盖。
   - 默认 objective sampling 和 loss weights 覆盖该目标。
   - 确保默认 batch 能采到 `source_name=protacdb`。

4. One-batch gate：
   - 增加默认配置 one-batch forward 测试，CPU、小样本、单 batch，要求总 loss 有限，且关键目标和来源计数符合预期。

## 执行日志

### 2026-06-13 SKEMPI2 修复

修改文件：

- `pipelines/humu_pretrain/src/humu_pretrain/data_loader.py`
- `tests/unit/test_humu_training.py`

新增/调整测试：

- `test_skempi_parser_handles_chain_and_multi_mutations`
- `test_skempi_multi_mutation_payload_applies_all_residue_changes`
- `test_skempi2_multimutation_without_explicit_views_is_loaded`

红灯验证：

```bash
uv run pytest moleculeforge/tests/unit/test_humu_training.py -q -k "skempi_parser_handles_chain_and_multi_mutations or skempi_multi_mutation_payload_applies_all_residue_changes or skempi2_multimutation_without_explicit_views_is_loaded"
```

结果：3 个测试按预期失败，失败原因分别为缺少 `_parse_skempi_mutations`、旧 payload 只接受单 tuple、多突变记录仍被跳过。

绿灯验证：

```bash
uv run pytest moleculeforge/tests/unit/test_humu_training.py -q -k "skempi_parser_handles_chain_and_multi_mutations or skempi_multi_mutation_payload_applies_all_residue_changes or skempi2_multimutation_without_explicit_views_is_loaded"
```

结果：3 个测试通过。

真实 dataloader 回归：

```bash
uv run python - <<'PY'
# 加载默认 humu_pretrain.yaml，覆盖 CPU / max_samples=100 / batch_size=256 / num_workers=0 / steps_per_epoch=1，
# 然后 create_dataloaders(cfg)["paired"] 并取一个 batch。
PY
```

结果：

```text
pair_type_counts {'activity_pair': 31, 'interface_mutation': 13, 'mol_pocket': 31, 'mol_pocket_route': 36, 'mol_route': 31, 'mol_self': 25, 'pdc_component': 7, 'protac_component': 20, 'protac_ternary': 13, 'protein_interface': 18, 'route_template': 31}
source_counts {'activity': 11, 'bindingdb_activity': 20, 'interface_skempi2': 13, 'joint': 36, 'mol': 25, 'pdcdb': 7, 'pocket': 31, 'protac8k': 13, 'protacdb': 0, 'protacpedia': 20, 'rcsb_mmcif': 18, 'retropath_templates': 13, 'route': 31, 'route_eval': 18}
batch_size 256
```

回头检查：

- SKEMPI2 解析修复是合理且科学的，因为 SKEMPI 突变 token 的第一个字母是野生型氨基酸，第二个字段才是链 ID。
- 多突变作为同一复合物突变集合同时应用到 residue payload，比丢弃多突变更符合 SKEMPI2 记录语义。
- 真实 dataloader 已不再因 SKEMPI2 多突变阻断 batch 构建。
- `source_counts["protacdb"] == 0` 仍然存在，进入 PROTAC-DB 覆盖任务处理。

### 2026-06-13 ESM/结构合同修复

修改文件：

- `models/mf-encoders/humu_pocket_encoder/src/mf_encoders/humu_pocket/encoder.py`
- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
- `pipelines/humu_pretrain/src/humu_pretrain/data_loader.py`
- `configs/models/humu_pretrain.yaml`
- `models/mf-encoders/humu_pocket_encoder/pyproject.toml`
- `uv.lock`
- `tests/unit/test_humu_training.py`

修复内容：

- `HUMUPocketEncoder` 新增 `esm2_required_sources`。
- 默认配置设置 `esm2_required_sources: ["pocket", "joint"]`。
- `pocket` / `joint` 源仍要求 `protein_sequence`、`sequence` 或 `esm2_embedding`。
- `rcsb_mmcif`、`interface_skempi2`、`pdcdb` 等结构源在无 ESM 输入时走 geometry-only 编码。
- dataloader 将 `source_name` 透传进嵌套 pocket / protein payload，使 encoder 能按源执行合同。
- pocket encoder package 声明 `fair-esm>=2.0.0` 依赖，并执行 `uv lock`。

红灯验证：

```bash
uv run pytest tests/unit/test_humu_training.py -q -k "pocket_encoder_requires_esm2_input_when_enabled or pocket_encoder_allows_geometry_only_for_structure_source_when_esm2_enabled or build_encoders_passes_pocket_esm2_config"
```

结果：3 个测试按预期失败，失败原因是 `HUMUPocketEncoder` 不接受 `esm2_required_sources`，且 `_build_encoders` 未传递该配置。

绿灯验证：

```bash
uv run pytest tests/unit/test_humu_training.py -q -k "pocket_encoder_requires_esm2_input_when_enabled or pocket_encoder_allows_geometry_only_for_structure_source_when_esm2_enabled or build_encoders_passes_pocket_esm2_config or pocket_encoder_batches_sequence_esm2_embeddings or forward_route_only_batch_keeps_pocket_encoder_ddp_path_with_esm2 or skempi"
```

结果：8 个测试通过。

依赖验证：

```bash
uv lock
```

结果：解析成功，`uv.lock` 已更新。

真实 forward 探测：

使用默认配置、CPU、`max_samples=100`、`batch_size=256` 运行 one-batch forward。该探测未出现结构源缺 ESM 输入错误；手动中断栈显示耗时点在 `pocket` / `joint` 必需源加载 650M ESM2 模型。

回头检查：

- 结构源允许 geometry-only 是合理的，因为 mmCIF/PDB/SKEMPI/PDC 结构 payload 本身具有可训练几何/残基特征，不应因缺少序列嵌入而整批失败。
- `pocket` 和 `joint` 源仍要求 ESM 输入是合理的，因为默认配置显式启用 ESM2，且这些源的数据合同应提供序列或预计算 embedding。
- one-batch gate 后续应在测试中 stub ESM 计算，验证默认 batch 前向合同；真实大规模训练仍会使用 `fair-esm` 和配置 checkpoint。

### 2026-06-13 PROTAC-DB 采样覆盖修复

修改文件：

- `pipelines/humu_pretrain/src/humu_pretrain/data_loader.py`
- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
- `configs/models/humu_pretrain.yaml`
- `tests/unit/test_humu_training.py`

修复内容：

- 将 `protacdb` 的 source objective 定义为 `protac_component_library`。
- 默认配置新增 `loss_weights.protac_component_library: 0.05`。
- 默认 objective sampling 新增 `protac_component_library: 0.03`。
- pipeline 新增 `l_protac_component_library` 和 `protac_component_library_retrieval_top1`。
- `protac_component_library` 使用组件 SMILES 的自监督双视图对比目标，不伪造 PROTAC anchor。

红灯验证：

```bash
uv run pytest tests/unit/test_humu_training.py -q -k "default_objective_sampler_includes_protacdb_library_source or forward_paired_batch_computes_protac_component_library_loss"
```

结果：`forward_paired_batch` 测试按预期失败，缺少 `l_protac_component_library`。

绿灯验证：

```bash
uv run pytest tests/unit/test_humu_training.py -q -k "default_objective_sampler_includes_protacdb_library_source or forward_paired_batch_computes_protac_component_library_loss"
uv run pytest tests/unit/test_humu_training.py -q -k "protac_component or objective_sampling_uses_all_trainable_humu_sources or target_ratio_sampler_matches_configured_objective_mix or training_batch_reports_source_coverage_stats or compute_losses_uses_protac_component_objective or compute_losses_uses_protac_ternary_and_pdc_objectives or forward_paired_batch_computes_all_enabled_losses"
```

结果：定向 2 个测试通过，较宽的采样/loss/forward 9 个测试通过。

默认 dataloader 覆盖回归：

```text
pair_type_counts {'activity_pair': 30, 'interface_mutation': 12, 'mol_pocket': 30, 'mol_pocket_route': 35, 'mol_route': 30, 'mol_self': 25, 'pdc_component': 7, 'protac_component': 20, 'protac_component_library': 8, 'protac_ternary': 12, 'protein_interface': 17, 'route_template': 30}
source_counts {'activity': 10, 'bindingdb_activity': 20, 'interface_skempi2': 12, 'joint': 35, 'mol': 25, 'pdcdb': 7, 'pocket': 30, 'protac8k': 12, 'protacdb': 8, 'protacpedia': 20, 'rcsb_mmcif': 17, 'retropath_templates': 12, 'route': 30, 'route_eval': 18}
```

回头检查：

- `protac_component_library` 独立于 `protac_component` 是合理的，因为 PROTAC-DB component library 记录只有组件 SMILES，没有完整 PROTAC anchor。
- 使用组件自监督双视图对比能让 PROTAC-DB 进入训练和检索指标，同时不制造错误的分子-组件配对语义。
- 默认采样已从 `source_counts["protacdb"] == 0` 修复为 `source_counts["protacdb"] == 8`。

### 2026-06-13 默认 one-batch forward gate

修改文件：

- `tests/unit/test_humu_training.py`

新增测试：

- `test_default_config_one_batch_forward_gate`

测试设计：

- 读取真实 `configs/models/humu_pretrain.yaml`。
- 仅覆盖 CPU、`max_samples=100`、`batch_size=256`、`num_workers=0`、`steps_per_epoch=1`。
- 使用真实 dataloader 生成默认比例 one-batch。
- 使用轻量 deterministic encoders 跑 `_forward_paired_batch`，避免单元测试加载 650M ESM2 或逐 SMILES 运行重模型。
- 验证 `losses["total"]` 有限。
- 验证所有默认关键目标均被采样：包括 `protac_component_library`。
- 验证所有关键来源均被采样：包括 `protacdb`。
- 验证所有默认关键 loss key 存在且为有限值。

验证：

```bash
uv run pytest tests/unit/test_humu_training.py -q -k "default_config_one_batch_forward_gate"
uv run pytest tests/unit/test_humu_training.py -q -k "compute_losses_uses_mol_self_activity_route_template_and_hard_negative or compute_losses_uses_large_source_objectives or forward_paired_batch_computes_all_enabled_losses or default_config_one_batch_forward_gate or protac_component_library"
```

结果：默认 gate 单测通过；相关 5 个 gate/loss/forward 回归通过。

回头检查：

- gate 使用真实默认配置和真实 batch，因此能覆盖配置、采样、collate、pair/source counts 和 `_forward_paired_batch` 合同。
- gate 不使用真实重 encoder，是合理的单元/集成测试边界；真实 encoder 的 ESM 和几何逻辑已有单独定向测试覆盖。
- 大规模训练前仍建议保留 `--preflight-only` 和短作业 smoke test，但默认 one-batch 合同现在已有自动化回归。

## 最终验收记录

### Manifest 同步

以下 manifest 的 `integration_status` 已从 `not_referenced_by_default_humu_pretrain_config` 更新为 `referenced_by_default_humu_pretrain_config`，并补充默认目标字段：

- `data/processing/humu_pretrain/protac8k/manifest.json`：`default_humu_pretrain_objective=protac_ternary`
- `data/processing/humu_pretrain/interface_skempi2/manifest.json`：`default_humu_pretrain_objective=interface_mutation`
- `data/processing/humu_pretrain/protacdb/manifest.json`：`default_humu_pretrain_objective=protac_component_library`
- `data/processing/humu_pretrain/protacpedia/manifest.json`：`default_humu_pretrain_objective=[protac_component, protac_ternary]`
- `data/processing/humu_pretrain/bindingdb_activity/manifest.json`：`default_humu_pretrain_objective=activity_pair`
- `data/processing/humu_pretrain/pdcdb/manifest.json`：`default_humu_pretrain_objective=pdc_component`

### 最终命令

```bash
uv run python -m py_compile \
  pipelines/humu_pretrain/src/humu_pretrain/pipeline.py \
  pipelines/humu_pretrain/src/humu_pretrain/data_loader.py \
  models/mf-encoders/humu_pocket_encoder/src/mf_encoders/humu_pocket/encoder.py \
  models/mf-encoders/humu_mol_encoder/src/mf_encoders/humu_mol/encoder.py \
  models/mf-encoders/humu_route_encoder/src/mf_encoders/humu_route/encoder.py

timeout 180s uv run python pipelines/humu_pretrain/train.py \
  --config configs/models/humu_pretrain.yaml \
  --preflight-only

uv run pytest tests/unit/test_humu_training.py tests/unit/test_learnable_curvature.py -q

git diff --check
```

结果：

- `py_compile`：通过。
- `preflight-only`：通过，source registry 显示 `protacdb` objective 为 `protac_component_library`，全部默认 HUMU source configured/trainable。
- HUMU 单元测试：113 个测试通过。
- `git diff --check`：通过。

### 科学复核结论

- SKEMPI2：多突变解析和 residue payload 同步应用符合 SKEMPI2 记录语义；单突变 chain 解析已修正。
- ESM/结构合同：源级 ESM 必需策略合理，保留 pocket/joint ESM 质量要求，同时允许纯结构源 geometry-only 训练。
- PROTAC-DB：组件库没有完整 PROTAC anchor，因此独立 `protac_component_library` 自监督目标比混入 `protac_component` 更科学。
- Default gate：默认配置已经能构建包含所有关键目标和来源的一批数据，并能通过 `_forward_paired_batch` 合同验证。
