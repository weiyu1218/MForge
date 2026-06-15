# HUMU 大规模预训练前置复核验收报告

日期：2026-06-14

## 复核范围

本次复核只做验收检查，不启动大规模训练。检查范围包括：

- 默认训练配置 `configs/models/humu_pretrain.yaml`
- 训练入口与后台启动脚本
- HUMU dataloader、目标采样、DDP wrapper、checkpoint/resume、validation/early stopping
- 真实 smoke 日志与 checkpoint 产物
- GPU/磁盘/进程状态

## 新鲜验证证据

已重新执行：

```bash
uv run python pipelines/humu_pretrain/train.py --config configs/models/humu_pretrain.yaml --preflight-only
uv run python -m py_compile pipelines/humu_pretrain/train.py \
  pipelines/humu_pretrain/src/humu_pretrain/pipeline.py \
  pipelines/humu_pretrain/src/humu_pretrain/data_loader.py \
  models/mf-encoders/humu_mol_encoder/src/mf_encoders/humu_mol/encoder.py \
  models/mf-encoders/humu_pocket_encoder/src/mf_encoders/humu_pocket/encoder.py \
  models/mf-encoders/humu_route_encoder/src/mf_encoders/humu_route/encoder.py
uv run pytest tests/unit/test_humu_training.py tests/unit/test_learnable_curvature.py -q
git diff --check
```

结果：

- 默认配置 preflight exit 0。
- 关键 Python 文件编译 exit 0。
- 相关单测 118 项通过。
- `git diff --check` exit 0。
- 当前无活跃 HUMU 训练进程；4 张 H200 显存占用为 0。
- 工作区磁盘可用约 9.9T。

CodeRabbit 自动审查未完成：CLI 已安装，`coderabbit --version` 为 `0.6.0`；但 `coderabbit auth status --agent` 未认证，`coderabbit auth login --agent` 返回 `environment_unsupported`，要求使用 `--api-key`。本报告不把本地人工审查冒充为 CodeRabbit 结果。

## 已通过的前置条件

- 真实单卡 1-step smoke 通过，完成 forward/backward/checkpoint。
- 真实单卡 5-step smoke 通过，覆盖连续 step、optimizer/scheduler、缓存复用。
- 真实 4-GPU DDP 2-step smoke 通过，覆盖 torchrun/NCCL/DDP 极短路径。
- `amp_dtype: bfloat16` 已真正进入 autocast；bf16 不再使用 `GradScaler`。
- H200/cuDNN SDPA backward 失败已用最小复现确认，并通过 `cuda_backends.enable_cudnn_sdp: false` 规避。
- 默认配置 source registry 全部 trainable source 配置存在。
- Validation loader 会按默认 `eval.every_n_epochs: 5` 创建；相关 validation/early stopping 单测存在。

## 重要问题

### 1. 目标采样器跨 epoch 完全重复，长训覆盖不充分

证据：

- `TargetRatioMultiSourceBatchSampler` 没有 `set_epoch()`。
- `_set_sampler_epoch()` 只检查 `loader.sampler`，不会作用到 `loader.batch_sampler`。
- 用默认配置构建 loader 后，连续两次 `list(iter(batch_sampler))` 完全相同：

```text
batch_sampler TargetRatioMultiSourceBatchSampler has_set_epoch False len 3
same_batches_across_iter True
```

默认完整 epoch 量化：

```text
dataset_len 420402 steps 1000 batch_size 256
same_epoch_sequence_when_reiterated True
```

影响：

- 100 epoch 会重复相同采样顺序。
- 对大源样本，单 epoch 有一定覆盖；但跨 epoch 不增加采样多样性。
- 对长训科学性不理想，可能降低泛化，尤其是多源目标的 curriculum/negative diversity。

判断：

- 这是大规模预训练前的科学合理性 blocker。短 smoke 不会暴露这个问题。

建议修复：

- 给 `TargetRatioMultiSourceBatchSampler` 增加 `seed`、`epoch`、`set_epoch()`。
- 在每个 epoch 内对 objective/source 内索引做确定性随机 permutation。
- `_set_sampler_epoch()` 同时检查 `loader.batch_sampler`。
- 增加单测：同一 epoch 可复现，不同 epoch batch 序列不同；DDP rank 之间仍不重叠。

### 2. 小源过采样极高，PDCdb 只有 5 条 trainable records

默认一个 epoch 采样统计：

```text
pdc_component 7000 seen, 5 unique, repeat_rate 0.9993
protac_component 20000 seen, 3581 unique, repeat_rate 0.8210
mol_pocket_route 35000 seen, 4275 unique, repeat_rate 0.8779
interface_mutation 12000 seen, 6798 unique, repeat_rate 0.4335
protac_ternary 12000 seen, 8640 unique, repeat_rate 0.2800
```

影响：

- `pdc_component` 权重虽只有 0.05，但每 epoch 7000 次来自 5 条记录，过拟合和梯度偏置风险很高。
- 这不一定导致工程失败，但从科学训练角度不稳健。

判断：

- 建议作为大训前配置 blocker 处理，至少降低/关闭 `pdc_component`，或先补充 PDC trainable records。

建议修复：

- 将 `pdc_component` objective ratio 从 `0.03` 暂时降到 `0.0` 或极低值，直到 PDCdb trainable records 数量合理。
- 或在 sampler 中按 `max_repeats_per_epoch` / `min_unique_records` 对小源做约束，无法满足时 fail-fast。
- 把 `preflight_humu_data_contract` 扩展为报告每个 objective 的 trainable unique count 和预计 repeat rate。

### 3. 默认输出目录会混入并覆盖历史 checkpoint

证据：

- 默认 `output_dir` 是 `/workspace/MForge/moleculeforge/checkpoints/humu/`。
- 该目录已有历史训练产物：238 个文件，总计约 1.1G。
- 其中已有 `best_model.pt`、大量 `checkpoint_step_*.pt`、`validation_metrics.jsonl`。
- 新 smoke checkpoint 约 34MB，旧历史 checkpoint 约 4.8MB，明显不是同一模型尺寸/配置。

影响：

- 直接启动默认大训练会覆盖 `best_model.pt` / `final_model.pt`。
- 新旧 `checkpoint_step_*.pt` 和 `validation_metrics.jsonl` 会混在一起，后续恢复、审计、曲线分析都容易误判。

判断：

- 这是大规模训练前的运维/可追踪性 blocker。

建议修复：

- 启动前生成独立 run config，把 `output_dir` 改为带 UTC 时间戳的目录，例如 `checkpoints/humu_4h200_YYYYMMDDTHHMMSSZ/`。
- 后台脚本 manifest 记录 config hash、git status、run name、输出目录。
- 不要直接写默认 `checkpoints/humu/`。

## 非阻塞但需记录的问题

### 配置项未被训练代码消费

`rg` 检查显示：

- `warmup_steps` 未用于 scheduler warmup。
- `gradient_accumulation_steps` 未用于训练 loop。
- `gpu_ids` 未由 `train.py` 或 `pipeline.py` 使用，实际多 GPU 由 `CUDA_VISIBLE_DEVICES` 和 torchrun 控制。

判断：

- 当前值分别是 `warmup_steps: 2000`、`gradient_accumulation_steps: 1`、`gpu_ids: [0]`。
- 因为 accumulation 当前为 1，不影响行为。
- `warmup_steps` 写在配置里但不生效，属于可解释性风险，不是立即崩溃 blocker。

建议：

- 若需要 warmup，应实现 scheduler warmup 或从配置移除，避免误导。
- 启动文档明确 GPU 选择以 launch env 为准。

### checkpoint 数量增长可接受但需隔离

默认：

```text
steps_per_epoch=1000
epochs=100
save_every_n_steps=500 -> 约 200 个 step checkpoint
save_every_n_epochs=5 -> 约 20 个 epoch checkpoint
keep_last_n=null
估算 checkpoint 空间约 7.37GB
```

磁盘空间充足，因此不是容量 blocker；但必须使用独立输出目录。

## 当前结论

不能再简单说“完全放心可以直接大训”。更准确的结论是：

- 工程执行路径已经被 smoke 证明可跑。
- 默认配置 preflight、单测、DDP 小测试都通过。
- 但发现两个大规模预训练前应修复的问题：
  1. `TargetRatioMultiSourceBatchSampler` 跨 epoch 完全重复。
  2. 默认 `output_dir` 会混入并覆盖历史 checkpoint。
- 还建议处理 `pdc_component` 极端小源过采样。

因此，本次复核结论：

**不建议立刻启动最终大规模预训练。**

建议先修复采样 epoch 随机性、隔离输出目录，并调整/约束小源过采样；修复后重新跑：

1. 目标采样单测。
2. 默认配置 preflight。
3. 单卡 5-step smoke。
4. 4-GPU DDP 2-step smoke。

这些 gate 再次通过后，再进入用户确认的大规模训练启动阶段。
