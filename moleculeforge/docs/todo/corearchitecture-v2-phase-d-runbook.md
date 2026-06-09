# CoreArchitecture v2 阶段 C/D Runbook

更新时间：2026-05-19

本文只记录阶段 C 接口状态、阶段 C3 预留接法、阶段 D 六项工作的后台执行命令和未完成边界。

所有命令默认在仓库根目录执行，并先加载 `.env` 中的启动配置：

```bash
cd /workspace/MForge/moleculeforge
set -a
source .env
set +a
```

每个任务的启动路径、GPU、训练参数和服务依赖变量统一在 `.env` 中维护；运行前只修改 `.env`，不在启动命令前临时拼接环境变量。

## 1. 阶段 C 接口状态

| 阶段 C 接口 | 目标 | 当前完成情况 | 代码入口 |
|---|---|---|---|
| C1 | ESM-2 接入 HUMU pocket encoder | 已完成接口接线。`HUMUPocketEncoder` 支持 `use_esm2`、`esm2_checkpoint`、`esm2_layer`、`esm2_dim`；输入可用 `protein_sequence`、`sequence` 或预计算 `esm2_embedding`。缺输入时 fail-fast，不生成伪特征。 | `models/mf-encoders/humu_pocket_encoder/src/mf_encoders/humu_pocket/encoder.py` |
| C2 | AiZynthFinder 接入 retrosyn service / retrosyn agent / SRB 上游 route 链路 | 已完成接口接线。`AiZynthRetrosyn.from_env()` 从 `AIZYNTH_CONFIG_PATH` 构造真实 runner；`retrosyn-svc` 和 `RetroSynAgent` 都调用 planner routes；SRB 可消费 retrosyn route 编译 SSP。真实运行仍需要 AiZynthFinder config、stock、policy 资源。 | `models/mf-retrosyn/aizynth_wrapper/src/mf_retrosyn/aizynth/retrosyn.py`、`services/retrosyn-svc/src/retrosyn_svc/main.py`、`agents/retrosyn_agent/src/retrosyn_agent/agent.py` |
| C3 | chemprop ADMET 真实模型接入 | 外部 Chemprop ADMET 微服务配置已放在 `/workspace/MForge/Decoupling/chempropADMET`，提供 `/health` 和 `/predict` HTTP 接口；MoleculeForge 内部 `admet-svc` 仍需接 `ADMET_SERVICE_URL` runner，当前代码仍会抛出 `ADMET model runner is not configured`。 | `/workspace/MForge/Decoupling/chempropADMET/app.py`、`/workspace/MForge/Decoupling/chempropADMET/client.py`、`services/admet-svc/src/admet_svc/main.py`、`models/mf-oracles/admet_ai/src/mf_oracles/admet_ai/oracle.py` |

C1 验证点：

- `tests/unit/test_humu_training.py::test_pocket_encoder_uses_precomputed_esm2_embedding`
- `tests/unit/test_humu_training.py::test_pocket_encoder_requires_esm2_input_when_enabled`
- `tests/unit/test_humu_training.py::test_build_encoders_passes_pocket_esm2_config`

C2 验证点：

- `tests/unit/test_indexing_pipelines.py::test_aizynth_from_env_requires_config`
- `tests/unit/test_indexing_pipelines.py::test_retrosyn_agent_uses_planner_routes`
- `tests/unit/test_service_artifact_status.py` 中 retrosyn service planner 注入测试

## 2. C3 chemprop ADMET 接法

C3 的真实 Chemprop ADMET 资源位于 `/workspace/MForge/Decoupling/chempropADMET`，当前形态是独立 HTTP 微服务：

- `app.py`：FastAPI 服务入口，暴露 `GET /health` 和 `POST /predict`。
- `config.py`：服务配置，端口为 `8901`，模型目录为 `models/`，默认 endpoint 包含 `solubility`、`lipophilicity`、`permeability`、`bbb`、`hia`、`bioavailability`、`cyp_inhibition`、`herg`、`ld50`、`clearance`。
- `model_manager.py`：按 endpoint 懒加载 `models/<endpoint>/model.ckpt`。
- `client.py`：读取 `ADMET_SERVICE_URL` 调用 HTTP 服务。

`.env` 中的 C3 配置：

```text
CHEMPROP_ADMET_ROOT
ADMET_SERVICE_URL
ADMET_HOST
ADMET_PORT
ADMET_MODEL_PATH
ADMET_TARGETS
ADMET_BATCH_SIZE
ADMET_MAX_BATCH_SIZE
ADMET_DEVICE
ADMET_UNCERTAINTY_METHOD
ADMET_CAL_PATH
ADMET_TEST_CSV
```

启动外部 ADMET 服务：

```bash
cd "$CHEMPROP_ADMET_ROOT"
uvicorn app:app --host "$ADMET_HOST" --port "$ADMET_PORT"
```

健康检查：

```bash
curl -sf "$ADMET_SERVICE_URL/health"
```

预测 smoke：

```bash
curl -sf -X POST "$ADMET_SERVICE_URL/predict" \
  -H "Content-Type: application/json" \
  -d "{\"smiles\":[\"CCO\"],\"batch_size\":$ADMET_BATCH_SIZE}"
```

MoleculeForge 内部接入边界：

1. 不在 `admet-svc` 内直接写启发式预测。
2. 在 `models/mf-oracles/admet_ai/src/mf_oracles/admet_ai/` 内接入 HTTP runner，读取 `ADMET_SERVICE_URL`、`ADMET_TARGETS`、`ADMET_BATCH_SIZE`。
3. runner 输入复用 `ADMETAIOracle` 已生成的 `descriptor_rows`，取 `row["smiles"]` 调用 `/predict`。
4. 返回值继续保持现有 oracle runner 契约：

```python
{
    "<canonical_smiles>": {
        "<target_name>": <predicted_value>,
    },
}
```

5. 在 `services/admet-svc/src/admet_svc/main.py` 中：
   - `ADMETServicer.__init__(runner=None)` 支持注入 runner。
   - 未注入 runner 时从 `ADMET_SERVICE_URL` 构造 HTTP runner。
   - `Predict()` 调用 `ADMETAIOracle(runner=runner).evaluate(...)`。
   - `Screen()` 调用 `predict_with_uncertainty(...)`；无不确定度 runner 时继续 fail-fast。
6. 增加单元测试：
   - runner 注入后 `Predict()` 返回模型值。
   - 缺 `ADMET_SERVICE_URL`、服务不可达、缺 `ADMET_TARGETS` 均 fail-fast。
   - `Screen()` 无 uncertainty runner 时明确报错。

## 3. 阶段 D 后台命令

### D1. HFM-3D 真实预训练

当前仓库入口：

- `models/mf-generators/hfm_3d/train.py`
- `models/mf-generators/hfm_3d/run_hfm_4h200_background.sh`

当前 `train.py` 已支持 `torch.distributed.run`：

- 从 `RANK`、`WORLD_SIZE`、`LOCAL_RANK` 读取 DDP 上下文。
- CUDA 下使用 NCCL，CPU 下使用 gloo。
- 使用 `DistributedSampler` 划分训练样本。
- 只在 rank 0 写 checkpoint。
- 空数据目录会 fail-fast，不会静默产出无效权重。

4xH200 后台启动：

```bash
bash models/mf-generators/hfm_3d/run_hfm_4h200_background.sh
```

恢复训练：

先在 `.env` 中设置 `HFM_RESUME` 为已有 checkpoint 路径，再执行同一个后台脚本。

```bash
bash models/mf-generators/hfm_3d/run_hfm_4h200_background.sh
```

单卡调试命令：

```bash
"$PYTHON_BIN" -u models/mf-generators/hfm_3d/train.py \
  --data "$HFM_DATA_DIR" \
  --epochs "$HFM_EPOCHS" \
  --batch-size "$HFM_BATCH_SIZE" \
  --lr "$HFM_LR" \
  --dim "$HFM_DIM" \
  --n-steps "$HFM_N_STEPS" \
  --device "$HFM_DEVICE" \
  --output-dir "$HFM_OUTPUT_DIR" \
  --save-every "$HFM_SAVE_EVERY"
```

训练产物：

```text
$HFM_OUTPUT_DIR/best_model.pt
$HFM_OUTPUT_DIR/final_model.pt
$HFM_OUTPUT_DIR/checkpoint_epoch_*.pt
```

### D2. FragFM SA-aware DFM 真实预训练

当前仓库入口：

- `models/mf-generators/fragfm/train.py`

训练数据要求：

- JSON 或 JSONL 文件/目录。
- 每条记录必须包含 `fragments` 和 `product`。
- `sa_score_bin` 可选，取值必须在 `[0, 9]`。
- `product` 必须是 RDKit 可解析 SMILES。

后台启动：

```bash
mkdir -p "$FRAGFM_LOG_DIR"

nohup "$PYTHON_BIN" -u models/mf-generators/fragfm/train.py \
  --data "$FRAGFM_DATA_DIR" \
  --output-dir "$FRAGFM_OUTPUT_DIR" \
  --epochs "$FRAGFM_EPOCHS" \
  --batch-size "$FRAGFM_BATCH_SIZE" \
  --lr "$FRAGFM_LR" \
  --hidden-dim "$FRAGFM_HIDDEN_DIM" \
  --rate-loss-weight "$FRAGFM_RATE_LOSS_WEIGHT" \
  --device "$FRAGFM_DEVICE" \
  --save-every "$FRAGFM_SAVE_EVERY" \
  > "$FRAGFM_LOG_DIR/fragfm_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 &
echo $! > "$FRAGFM_LOG_DIR/fragfm.pid"
```

训练产物：

```text
$FRAGFM_OUTPUT_DIR/vocab.json
$FRAGFM_OUTPUT_DIR/best_model.pt
$FRAGFM_OUTPUT_DIR/rate_matrix.pt
$FRAGFM_OUTPUT_DIR/final_model.pt
$FRAGFM_OUTPUT_DIR/final_rate_matrix.pt
$FRAGFM_OUTPUT_DIR/training_manifest.json
```

服务接入：

```text
FRAGFM_VOCAB_PATH
FRAGFM_CHECKPOINT_PATH
FRAGFM_RATE_MATRIX_PATH
```

### D3. HUMU foundation model 大规模联合预训练

当前仓库入口：

- `pipelines/humu_pretrain/train.py`
- `pipelines/humu_pretrain/run_humu_4h200_background.sh`

当前可运行内容：

- mol-pocket contrastive loss
- mol-route contrastive loss
- pocket-route contrastive loss
- molecule-intent contrastive loss
- curvature regularization

当前不可声明的内容：


数据契约预检：

```bash
"$PYTHON_BIN" -u pipelines/humu_pretrain/train.py \
  --config "$HUMU_CONFIG_PATH" \
  --preflight-only
```

4 卡后台启动：

```bash
bash pipelines/humu_pretrain/run_humu_4h200_background.sh
```



停止后台训练：

```bash
bash pipelines/humu_pretrain/stop_humu_background.sh <pid-file>
```

训练产物：

```text
checkpoints/humu/best_model.pt
checkpoints/humu/final_model.pt
checkpoints/humu/checkpoint_epoch_*.pt
```

未完成项：

- 解决方式：

### D4. KRAS G12C pilot 端到端真实跑通

当前仓库入口：

- `tests/e2e/test_kras_g12c_pilot.py`

D4 不是训练脚本，而是端到端验证入口。必须先跑 preflight；缺依赖时不要设置 flag 绕过。DKI 连接、provenance、对象存储、Redis、代理清理和 KRAS 启动 flag 均在 `.env` 中维护。

preflight：

```bash
uv run python - <<'PY'
from tests.e2e.test_kras_g12c_pilot import kras_e2e_preflight_status

status = kras_e2e_preflight_status()
print(status)
if not status["ready"]:
    raise SystemExit(1)
PY
```

preflight 通过后后台启动：

```bash
mkdir -p "$KRAS_E2E_LOG_DIR"

nohup uv run pytest tests/e2e/test_kras_g12c_pilot.py -q \
  > "$KRAS_E2E_LOG_DIR/kras_g12c_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 &
echo $! > "$KRAS_E2E_LOG_DIR/kras_g12c.pid"
```

preflight 必需项：

```text
HFM_CHECKPOINT_PATH
HFM_DECODER_PATH
BOLTZ_MODEL_PATH
RETROSYN_RUNNER_URI
CRITIC_AGENT_READY=1
ORCHESTRATOR_E2E_READY=1
GNINA_BINARY or DIFFDOCK_MODEL_PATH
NEO4J_URI
NEO4J_USER
NEO4J_PASSWORD
MINIO_ENDPOINT_URL
MINIO_ACCESS_KEY
MINIO_SECRET_KEY
MINIO_BUCKET
REDIS_HOST
REDIS_PORT
PROVENANCE_DATABASE_URL or TEST_DATABASE_URL
PROVENANCE_STORE_MODE=production_real
```

未完成项：

- DKI 环境变量的加载方式已在上方给出，仍需要 `/workspace/mf-dki-bare` 服务本身处于可用状态。
- 当前测试文件只校验端到端依赖是否齐备，真实业务执行依赖 D1/D3/D5 和 oracle/retrosyn/orchestrator 资源。
- 解决方式：先补齐 preflight 中列出的真实 artifact、真实服务和 DKI 环境，再运行 D4；只有 D4 真实通过，才能声明 KRAS G12C pilot 跑通。


当前仓库入口：

- `pipelines/reaction_indexing/src/reaction_indexing/pipeline.py::run(config)`

当前不能给出真实后台接入命令。原因：

- `reaction_indexing.run(config)` 需要 `source_paths["uspto"]`、`source_paths["pistachio"]`、`source_paths["reaxys"]` 等真实本地数据路径。

未完成项：

- Reaxys 需要商业授权和法务确认，当前只有 client 骨架。

解决方式：

2. 提供真实数据路径或 API 凭证，不用 placeholder client 伪造成功。
4. 实现 reaction indexing CLI：从环境读取 USPTO / Pistachio / Reaxys 本地路径，调用 `reaction_indexing.run(config)`。

## 4. 未完成项汇总

| 项 | 未完成内容 | 解决路径 |
|---|---|---|
| D4 | KRAS G12C 真实业务跑通 | 补齐 preflight 所有 artifact、服务和 DKI 环境 |
