# HUMU Real Smoke To Large Train Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the real HUMU training path with real GPU, real ESM2 checkpoint, real dataloader, and real encoders before starting long-running large-scale pretraining.

**Architecture:** Keep the default production config as the source of truth, generate small smoke YAML files from it, and run foreground jobs with isolated output directories. The smoke ladder is preflight -> single GPU 1-batch -> single GPU 3-5 batch -> optional 4xH200 DDP smoke -> large-train launch recommendation.

**Tech Stack:** PyTorch 2.6 CUDA, H200 GPUs, fair-esm, HUMU `pipelines/humu_pretrain/train.py`, YAML configs, shell timeout guards, JSONL checkpoint/log inspection.

---

### Task 1: Smoke Environment And Config

**Files:**
- Create: `logs/humu_pretrain/humu_real_smoke_20260613_smoke1.yaml`
- Create: `logs/humu_pretrain/humu_real_smoke_20260613_smoke5.yaml`
- Create: `docs/todo/2026-06-13-humu-real-smoke-to-large-train-log.md`

- [ ] **Step 1: Verify runtime environment**

Run:

```bash
uv run python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.mem_get_info(i))
import esm
print("esm_import", "ok")
PY
test -f models/esm2/esm2_t33_650M_UR50D.pt
```

Expected: CUDA available, at least one H200 visible, `esm_import ok`, checkpoint file exists.

- [ ] **Step 2: Generate isolated smoke configs**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path
import copy
import yaml

root = Path.cwd()
base = yaml.safe_load((root / "configs/models/humu_pretrain.yaml").read_text())

def write_smoke(name: str, steps: int, batch_size: int, max_samples: int) -> None:
    cfg = copy.deepcopy(base)
    cfg["epochs"] = 1
    cfg["batch_size"] = batch_size
    cfg["max_samples"] = max_samples
    cfg["device"] = "cuda"
    cfg["use_amp"] = True
    cfg["skip_bad_batches"] = False
    cfg["max_skipped_batches"] = 0
    cfg["save_every_n_epochs"] = 1
    cfg["save_every_n_steps"] = 0
    cfg["keep_last_n"] = 1
    cfg["resume_from"] = None
    cfg["output_dir"] = str(root / "checkpoints" / name)
    cfg["data"]["num_workers"] = 0
    cfg["data"]["pin_memory"] = False
    cfg["data"]["objective_sampling"]["steps_per_epoch"] = steps
    cfg["data"]["objective_sampling"]["source_cap"] = max_samples
    cfg["data"]["source_sample_cap"] = max_samples
    cfg["eval"]["every_n_epochs"] = 0
    cfg["early_stopping"]["enabled"] = False
    cfg["logging"]["log_every_n_steps"] = 1
    cfg["logging"]["preserve_step_logs"] = True
    out = root / "logs" / "humu_pretrain" / f"{name}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(out)

write_smoke("humu_real_smoke_20260613_smoke1", steps=1, batch_size=64, max_samples=128)
write_smoke("humu_real_smoke_20260613_smoke5", steps=5, batch_size=64, max_samples=256)
PY
```

Expected: both config paths printed; both output directories point under `checkpoints/humu_real_smoke_20260613_*`.

- [ ] **Step 3: Run preflight against smoke configs**

Run:

```bash
uv run python pipelines/humu_pretrain/train.py --config logs/humu_pretrain/humu_real_smoke_20260613_smoke1.yaml --preflight-only
uv run python pipelines/humu_pretrain/train.py --config logs/humu_pretrain/humu_real_smoke_20260613_smoke5.yaml --preflight-only
```

Expected: both commands exit 0 and source registry shows default trainable sources configured.

### Task 2: Single GPU One-Batch Real Smoke

**Files:**
- Read: `logs/humu_pretrain/humu_real_smoke_20260613_smoke1.yaml`
- Write: `logs/humu_pretrain/humu_real_smoke_20260613_smoke1.log`
- Write: `checkpoints/humu_real_smoke_20260613_smoke1/`

- [ ] **Step 1: Run one real GPU batch**

Run:

```bash
CUDA_VISIBLE_DEVICES=2 timeout 1800s uv run python -u pipelines/humu_pretrain/train.py \
  --config logs/humu_pretrain/humu_real_smoke_20260613_smoke1.yaml \
  2>&1 | tee logs/humu_pretrain/humu_real_smoke_20260613_smoke1.log
```

Expected: exit 0, one step log appears, no skipped batch, `final_model.pt` exists.

- [ ] **Step 2: Inspect outputs**

Run:

```bash
test -f checkpoints/humu_real_smoke_20260613_smoke1/final_model.pt
test -f checkpoints/humu_real_smoke_20260613_smoke1/best_model.pt
rg -n "Skipped HUMU batch|error=|Traceback|RuntimeError|CUDA out of memory|train_loss" \
  logs/humu_pretrain/humu_real_smoke_20260613_smoke1.log
```

Expected: checkpoint files exist; `train_loss=` appears; no `Skipped HUMU batch`, traceback, runtime error, or CUDA OOM.

### Task 3: Single GPU Multi-Batch Real Smoke

**Files:**
- Read: `logs/humu_pretrain/humu_real_smoke_20260613_smoke5.yaml`
- Write: `logs/humu_pretrain/humu_real_smoke_20260613_smoke5.log`
- Write: `checkpoints/humu_real_smoke_20260613_smoke5/`

- [ ] **Step 1: Run five real GPU batches**

Run:

```bash
CUDA_VISIBLE_DEVICES=2 timeout 3600s uv run python -u pipelines/humu_pretrain/train.py \
  --config logs/humu_pretrain/humu_real_smoke_20260613_smoke5.yaml \
  2>&1 | tee logs/humu_pretrain/humu_real_smoke_20260613_smoke5.log
```

Expected: exit 0, five step logs appear, no skipped batch, `final_model.pt` exists.

- [ ] **Step 2: Inspect outputs**

Run:

```bash
test -f checkpoints/humu_real_smoke_20260613_smoke5/final_model.pt
test -f checkpoints/humu_real_smoke_20260613_smoke5/best_model.pt
rg -n "Skipped HUMU batch|error=|Traceback|RuntimeError|CUDA out of memory|train_loss" \
  logs/humu_pretrain/humu_real_smoke_20260613_smoke5.log
```

Expected: checkpoint files exist; `train_loss=` appears; no `Skipped HUMU batch`, traceback, runtime error, or CUDA OOM.

### Task 4: Large-Train Launch Decision

**Files:**
- Modify: `docs/todo/2026-06-13-humu-real-smoke-to-large-train-log.md`

- [ ] **Step 1: Record evidence**

Append the command outputs, checkpoint paths, GPU memory observations, and pass/fail decision to the log document.

- [ ] **Step 2: Decide launch gate**

If both single-GPU smokes pass, recommend either:

```bash
CONFIG_PATH=/workspace/MForge/moleculeforge/configs/models/humu_pretrain.yaml \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUN_NAME=humu_4h200_$(date -u +%Y%m%dT%H%M%SZ) \
bash pipelines/humu_pretrain/run_humu_4h200_background.sh
```

or, if DDP should be tested first, generate a 4-GPU smoke config with `steps_per_epoch=2`, `batch_size=256`, `max_samples=512`, and isolated output under `checkpoints/humu_real_smoke_20260613_ddp/`.

Expected: launch only after smoke evidence is recorded.

---

## Self-Review

- Spec coverage: covers environment, config generation, real GPU one-batch, multi-batch, output inspection, and large-train decision.
- Placeholder scan: no TBD/TODO placeholders.
- Type consistency: config keys match `humu_pretrain.yaml` and `train.py`/`pipeline.py` expectations.
