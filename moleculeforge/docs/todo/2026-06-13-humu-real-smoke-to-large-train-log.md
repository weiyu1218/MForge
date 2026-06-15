# HUMU 真实 Smoke 验收日志

日期：2026-06-14

## 目标

在启动大规模 HUMU 预训练前，验证真实 GPU、真实 ESM2 checkpoint、真实 encoder、真实 dataloader、真实 backward 和 checkpoint 写入路径。

执行计划：

- `docs/superpowers/plans/2026-06-13-humu-real-smoke-to-large-train.md`

## 修复前阻塞

### AMP dtype 合同未生效

证据：

- `configs/models/humu_pretrain.yaml` 和 smoke YAML 均配置 `use_amp: true`、`amp_dtype: bfloat16`。
- 训练代码原先使用 `torch.amp.autocast("cuda")`，没有传入 `dtype`，PyTorch CUDA autocast 默认 dtype 为 `torch.float16`。
- 真实 `smoke1` 在 backward 阶段失败：`RuntimeError: cuDNN Frontend error: [cudnn_frontend] Error: No valid execution plans built.`

修复：

- 在 `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py` 新增 `_amp_dtype_from_config()`。
- 训练和验证 autocast 均使用 `torch.amp.autocast("cuda", dtype=amp_dtype)`。
- 仅在 `amp_dtype is torch.float16` 时启用 `GradScaler`；bf16 不使用 scaler。
- 在 `tests/unit/test_humu_training.py` 增加 `bfloat16/bf16/float16/fp16/half` 解析和非法 dtype fail-fast 测试。

### H200/cuDNN SDPA backward 后端失败

证据：

- 环境：`torch 2.6.0a0+df5bbc09d1.nv24.12`、CUDA 12.6、cuDNN 9.6、NVIDIA H200。
- PyTorch 后端状态：`torch.backends.cuda.cudnn_sdp_enabled() == True`。
- 最小复现：
  - `nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)` 在 bf16 autocast 下 backward 失败，错误同为 `cuDNN Frontend error: No valid execution plans built`。
  - 调用 `torch.backends.cuda.enable_cudnn_sdp(False)` 后，同一最小复现 backward 通过。
  - `HUMURouteEncoder(dim=128, hidden_dim=256, n_layers=4, n_heads=8)` 同样在禁用 cuDNN SDPA 后 backward 通过。

修复：

- 在 `pipeline.py` 新增 `_cuda_sdp_backend_config()` 和 `_configure_cuda_backends()`。
- 训练入口拿到 CUDA device 后立即应用 `torch.backends.cuda.enable_cudnn_sdp(False)`。
- 在 `configs/models/humu_pretrain.yaml` 增加：

```yaml
cuda_backends:
  enable_cudnn_sdp: false
```

科学性判断：

- 该修复只改变 PyTorch scaled-dot-product attention 的 kernel backend 选择，不改变模型结构、损失函数、数据采样或优化目标。
- H200 当前软件栈下已用最小复现证明 cuDNN SDPA backward 不可用；禁用该 backend 后保留 flash/math/mem-efficient SDPA，训练语义合理。

## 环境记录

- GPU：4 x NVIDIA H200，smoke 前后 `nvidia-smi` 显示显存空闲。
- ESM checkpoint：`models/esm2/esm2_t33_650M_UR50D.pt` 存在。
- `fair-esm` import 正常。
- DDP smoke 使用 NCCL `2.23.4+cuda12.6`。

## Smoke 配置

- 单卡 1-step：`logs/humu_pretrain/humu_real_smoke_20260613_smoke1.yaml`
  - `batch_size: 64`
  - `steps_per_epoch: 1`
  - `max_samples/source_cap: 128`
  - `use_amp: true`
  - `amp_dtype: bfloat16`
  - `cuda_backends.enable_cudnn_sdp: false`
  - 输出：`checkpoints/humu_real_smoke_20260613_smoke1`

- 单卡 5-step：`logs/humu_pretrain/humu_real_smoke_20260613_smoke5.yaml`
  - `batch_size: 64`
  - `steps_per_epoch: 5`
  - `max_samples/source_cap: 256`
  - `use_amp: true`
  - `amp_dtype: bfloat16`
  - `cuda_backends.enable_cudnn_sdp: false`
  - 输出：`checkpoints/humu_real_smoke_20260613_smoke5`

- 4-GPU DDP 2-step：`logs/humu_pretrain/humu_real_smoke_20260613_ddp.yaml`
  - `batch_size: 256`
  - `steps_per_epoch: 2`
  - `max_samples/source_cap: 512`
  - `use_amp: true`
  - `amp_dtype: bfloat16`
  - `cuda_backends.enable_cudnn_sdp: false`
  - 输出：`checkpoints/humu_real_smoke_20260613_ddp`

## 验证命令

已通过：

```bash
uv run pytest tests/unit/test_humu_training.py -q -k "amp_dtype_from_config or cuda_sdp_backend_config"
uv run python -m py_compile pipelines/humu_pretrain/src/humu_pretrain/pipeline.py
uv run pytest tests/unit/test_humu_training.py tests/unit/test_learnable_curvature.py -q
uv run python pipelines/humu_pretrain/train.py --config logs/humu_pretrain/humu_real_smoke_20260613_smoke1.yaml --preflight-only
uv run python pipelines/humu_pretrain/train.py --config logs/humu_pretrain/humu_real_smoke_20260613_smoke5.yaml --preflight-only
uv run python pipelines/humu_pretrain/train.py --config logs/humu_pretrain/humu_real_smoke_20260613_ddp.yaml --preflight-only
git diff --check
```

`tests/unit/test_humu_training.py tests/unit/test_learnable_curvature.py` 共 118 项通过。

## Smoke 结果

### 单卡 1-step

命令：

```bash
CUDA_VISIBLE_DEVICES=2 timeout 1800s uv run python -u pipelines/humu_pretrain/train.py \
  --config logs/humu_pretrain/humu_real_smoke_20260613_smoke1.yaml
```

结果：

- exit 0。
- `Batch 1/1` 完成。
- `train_loss=6.4846`。
- 时间：`53.0s`，首批 forward `49.5s`，backward `0.6s`。
- GPU 显存：约 `2663MB`。
- checkpoint：`best_model.pt`、`checkpoint_epoch_0001.pt`、`final_model.pt` 均写出，每个约 34MB。
- 日志未发现 `Skipped HUMU batch`、`Traceback`、`RuntimeError`、`CUDA out of memory`。

### 单卡 5-step

命令：

```bash
CUDA_VISIBLE_DEVICES=2 timeout 3600s uv run python -u pipelines/humu_pretrain/train.py \
  --config logs/humu_pretrain/humu_real_smoke_20260613_smoke5.yaml
```

结果：

- exit 0。
- `Batch 1/5` 到 `Batch 5/5` 全部完成。
- `train_loss=6.5113`。
- 总时间：`71.9s`。
- 首批 forward `46.6s`；后续 batch forward 约 `1.8s-3.2s`；backward 约 `0.4s-0.6s`。
- GPU 显存：约 `2662MB-2666MB`。
- checkpoint：`best_model.pt`、`checkpoint_epoch_0001.pt`、`final_model.pt` 均写出，每个约 34MB。
- 日志未发现 `Skipped HUMU batch`、`Traceback`、`RuntimeError`、`CUDA out of memory`。

### 4-GPU DDP 2-step

命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=WARN OMP_NUM_THREADS=4 timeout 2400s \
  uv run python -u -m torch.distributed.run --standalone --nproc_per_node=4 \
  pipelines/humu_pretrain/train.py --config logs/humu_pretrain/humu_real_smoke_20260613_ddp.yaml
```

结果：

- exit 0。
- NCCL 启动正常，版本 `2.23.4+cuda12.6`。
- `Batch 1/2` 和 `Batch 2/2` 完成。
- `train_loss=10.0487`。
- 总时间：`71.4s`。
- backward 约 `1.4s-1.5s`。
- rank0 GPU 显存：约 `2673MB-2677MB`。
- 4 个 rank 均返回 `status: completed`，rank0 `best_loss=10.048651695251465`。
- checkpoint：`best_model.pt`、`checkpoint_epoch_0001.pt`、`final_model.pt` 均写出，每个约 34MB。
- 日志未发现 `Skipped HUMU batch`、`Traceback`、`RuntimeError`、`CUDA out of memory`、`NCCL error`。
- smoke 后 `nvidia-smi` 显示 4 张 GPU 显存占用为 0；未发现活跃 HUMU 训练进程。

## 当前结论

短 smoke gate 已通过：默认 HUMU 训练路径已覆盖真实数据合同、真实 ESM2、真实 encoder、真实 bf16 autocast、真实 backward、checkpoint 写入、单卡连续 step 和 4-GPU DDP 极短路径。

本次不启动最终大规模训练。

在用户确认前，只建议保留当前修复并等待下一步决策。若之后要启动大规模训练，建议使用已修复的默认配置，并在启动后先监控前 10-20 个 step 的 loss、skip、GPU 显存、DDP rank 状态和 checkpoint 写入。
