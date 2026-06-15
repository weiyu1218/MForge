# HUMU 预训练

HUMU 预训练将分子、口袋、路线和辅助特征编码器联合训练到同一个 Lorentz 流形空间中。默认配置来自 `configs/models/humu_pretrain.yaml`，训练入口和核心流程位于 `pipelines/humu_pretrain/src/humu_pretrain`。

## 模型结构

- `mol`：分子图编码器，用于 SMILES、分子组件和分子自监督目标。
- `pocket`：蛋白口袋和蛋白界面编码器，支持 ESM2 payload。
- `route`：逆合成路线编码器，用于路线和反应模板目标。
- `protac_feature`：PROTAC-8K 行对齐 PROTAC 数值特征投影器。
- `protac_context_feature`：PROTAC-8K target 和 E3 上下文数值特征投影器。

所有 trainable encoder 都由 `_build_encoders` 构建，DDP、optimizer 和 checkpoint 共享同一份 encoder map。

## 数据合同

默认配置要求 HUMU source registry 中的所有训练源都被配置。各数据源映射到以下训练目标：

- `mol`：分子自监督对比样本。
- `pocket`：分子-口袋对比样本。
- `route`：分子-路线对比样本。
- `joint`：分子-口袋-路线联合样本。
- `activity` 和 `bindingdb_activity`：同靶点活性监督样本。
- `protacpedia` 和 `protacdb`：PROTAC 分子-组件样本和组件库样本。
- `protac8k`：PROTAC-8K 三元特征样本。
- `rcsb_mmcif`：蛋白界面对比样本。
- `interface_skempi2`：SKEMPI2 突变亲和力样本。
- `pdcdb`：PDC 分子-linker 组件样本。
- `route_eval` 和 `retropath_templates`：路线-模板样本。

PROTAC-8K 使用真实行对齐特征矩阵：`protac_feature.npy`、`target_feature.npy`、`e3_feature.npy`。PDCdb 只使用可验证的 PDC SMILES 和 linker SMILES；`Peptide_Name` 不作为 peptide sequence 使用。

## 输入限制

ESM2 序列限制由 `encoders.pocket.esm2_max_sequence_length` 控制，默认值为 `1022`。该值表示氨基酸残基长度；ESM2 batch converter 会额外加入 BOS/EOS token，因此 1022 个残基对应 1024 个输入 token。`pocket` 和 `joint` 源在启用 ESM2 时会按该限制过滤超长序列；已提供 `esm2_embedding` 的记录不再按序列长度过滤。

Pocket 点云限制由 `data.max_pocket_points` 控制，默认值为 `1536`。该限制作用于 `pocket_source` 和 `joint_source` 的 pocket 坐标 payload，在进入 pocket encoder 前对 `coords`、`elements`、`residue_types` 等逐点字段做确定性等距下采样。该限制不删除数据记录，只限制单条 pocket 的点数上限，用于约束 pocket encoder 中 `torch.cdist` 的 `O(N^2)` 计算规模。

## 采样策略

当前训练使用 objective-ratio batch sampling，不按原始数据量自然采样，也不使用固定倍数放大采样。每个 batch 由 `TargetRatioMultiSourceBatchSampler` 按配置比例组装。

默认 `batch_size` 为 `64`。当前每个 rank 的 batch 整数目标配额如下：

| Objective | 样本数 |
| --- | ---: |
| `mol_self` | 6 |
| `mol_pocket` | 8 |
| `mol_route` | 8 |
| `mol_pocket_route` | 9 |
| `activity_pair` | 8 |
| `protac_component` | 5 |
| `protac_component_library` | 2 |
| `route_template` | 8 |
| `protac_ternary` | 3 |
| `protein_interface` | 4 |
| `interface_mutation` | 3 |
| `pdc_component` | 0 |

同一 objective 下的多 source 调度使用 `size ** alpha`，默认 `alpha=0.5`。该策略保留大数据源的影响力，同时避免小但必要的数据源在训练中消失。PDCdb 保留在 source registry 和 preflight 数据合同中；由于当前可训练 PDC 记录只有 5 条，默认主训练不启用 `pdc_component` objective。

`source_sample_cap` 与 `objective_sampling.source_cap` 默认为 `null`，即每个数据源全量加载。每个 epoch 由 `TargetRatioMultiSourceBatchSampler.set_epoch` 触发一次确定性洗牌（按 `seed` 和 epoch 派生随机排列），使跨 epoch 的 batch 组合和负样本集随 epoch 变化，并覆盖到全部数据。DDP 下各 rank 共享同一排列、按 `rank` 做 stride 分片。

## 损失结构

训练总损失由启用目标的加权损失组成：

- `mol_pocket`：分子-口袋对比。
- `mol_route`：分子-路线对比。
- `pocket_route`：口袋-路线对比。
- `mol_self`：分子自监督对比。
- `activity_supervised`：同靶点活性监督。
- `protac_component`：PROTAC 分子-组件对比。
- `protac_component_library`：PROTAC-DB 组件库自监督对比。
- `route_template`：反应模板-路线对比。
- `protein_interface`：蛋白界面对比。
- `interface_mutation`：SKEMPI2 突变亲和力监督。
- `protac_ternary`：PROTAC-8K 三元特征对比。
- `pdc_component`：PDC 分子-linker 对比；默认权重为 `0.0`，数据补强后再启用。
- `curvature_reg`：流形曲率正则。

默认对比学习使用 in-batch negatives，temperature 为 `0.07`。

流形曲率锁定为 `1.0`：`manifold.distance` 的测地距离实现仅对单位曲率正确，`curvature != 1.0` 会在配置校验阶段报错。

## 优化策略

- Optimizer：`AdamW`。
- Learning rate：`3.0e-4`。
- Weight decay：`1.0e-5`。
- Warmup：前 `warmup_steps` 步 lr 从 0 线性升到 `3.0e-4`，warmup 期间保持 cosine scheduler 不前进。
- Scheduler：warmup 之后使用 `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)`。
- Epochs：`100`。
- Mixed precision：CUDA 下启用 `bfloat16`。
- Gradient clipping：`1.0`。
- Gradient accumulation：`1`。
- Bad batch policy：跳过坏 batch，最多 `max_skipped_batches` 次。

## 验证和 checkpoint

启用验证 split 时，每 `5` 个 epoch 验证一次，默认 `eval_split_ratio=0.1`。验证集按 `pair_type` 分层划分，保证训练子集包含每个 objective，同时验证集对小数据源有代表性。

`retrieval_top1` 聚合指标排除 `mol_self` 和 `protac_component_library`：这两个 objective 的 anchor 与 positive 编码同一输入（SimCSE 式，仅靠 dropout 区分），验证时 dropout 关闭使其 top1 恒为 1.0，单独上报但不计入总检索指标。

Early stopping 监控 `val_loss`，`mode=min`，`patience=5`，`min_delta=0.001`。

checkpoint 写入 `checkpoints/humu/`。训练每 `5` 个 epoch 保存 epoch checkpoint，每 `500` step 保存 step checkpoint。恢复状态包含 encoder 权重、optimizer state、scheduler state、epoch、step 和 best validation loss。
