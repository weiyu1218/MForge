#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

: "${HFM_DATA_DIR:?set HFM_DATA_DIR to processed ChEMBL 34 JSONL directory}"
: "${HFM_OUTPUT_DIR:?set HFM_OUTPUT_DIR to HFM checkpoint output directory}"
HUMU_CHECKPOINT_PATH="${HUMU_CHECKPOINT_PATH:-${HUMU_CHECKPOINT:-}}"
: "${HUMU_CHECKPOINT_PATH:?set HUMU_CHECKPOINT_PATH to pretrained HUMU checkpoint}"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
HFM_EPOCHS="${HFM_EPOCHS:-50}"
HFM_BATCH_SIZE="${HFM_BATCH_SIZE:-128}"
HFM_LR="${HFM_LR:-3e-4}"
HFM_DIM="${HFM_DIM:-128}"
HFM_N_STEPS="${HFM_N_STEPS:-20}"
HFM_SAVE_EVERY="${HFM_SAVE_EVERY:-5}"
HFM_SEED="${HFM_SEED:-42}"
HFM_RESUME="${HFM_RESUME:-}"
HFM_DECODER_PATH="${HFM_DECODER_PATH:-$HFM_OUTPUT_DIR/decoder.json}"
HFM_DECODER_MAX_ENTRIES="${HFM_DECODER_MAX_ENTRIES:-4096}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/hfm_3d}"
RUN_NAME="${RUN_NAME:-hfm_4h200_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_NAME.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/$RUN_NAME.pid}"
RUN_MANIFEST="${RUN_MANIFEST:-$LOG_DIR/$RUN_NAME.manifest.json}"

if [[ ! -d "$HFM_DATA_DIR" ]]; then
  echo "HFM_DATA_DIR must point to an existing directory: $HFM_DATA_DIR" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$HUMU_CHECKPOINT_PATH" ]]; then
  echo "HUMU checkpoint not found: $HUMU_CHECKPOINT_PATH" >&2
  exit 1
fi

if [[ -n "$HFM_RESUME" && ! -f "$HFM_RESUME" ]]; then
  echo "Resume checkpoint not found: $HFM_RESUME" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$HFM_OUTPUT_DIR"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "HFM training is already running for PID file $PID_FILE: $existing_pid" >&2
    exit 1
  fi
fi

export PROJECT_ROOT HFM_DATA_DIR HFM_OUTPUT_DIR PYTHON_BIN NPROC_PER_NODE
export CUDA_VISIBLE_DEVICES NCCL_DEBUG OMP_NUM_THREADS HFM_EPOCHS HFM_BATCH_SIZE
export HFM_LR HFM_DIM HFM_N_STEPS HFM_SAVE_EVERY HFM_SEED HFM_RESUME
export HUMU_CHECKPOINT_PATH HFM_DECODER_PATH HFM_DECODER_MAX_ENTRIES
export LOG_FILE RUN_MANIFEST PYTHONUNBUFFERED=1

setsid bash -c '
  set -Eeuo pipefail
  cd "$PROJECT_ROOT"
  exec > >(stdbuf -oL tee -a "$LOG_FILE" >/dev/null) 2>&1
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "project_root=$PROJECT_ROOT"
  echo "data_dir=$HFM_DATA_DIR"
  echo "output_dir=$HFM_OUTPUT_DIR"
  echo "humu_checkpoint=$HUMU_CHECKPOINT_PATH"
  echo "decoder_artifact=$HFM_DECODER_PATH"
  echo "run_manifest=$RUN_MANIFEST"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "nproc_per_node=$NPROC_PER_NODE"
  echo "resume=$HFM_RESUME"
  echo "log_file=$LOG_FILE"
  train_args=(
    models/mf-generators/hfm_3d/train.py
    --data "$HFM_DATA_DIR"
    --output-dir "$HFM_OUTPUT_DIR"
    --humu-checkpoint "$HUMU_CHECKPOINT_PATH"
    --decoder-artifact "$HFM_DECODER_PATH"
    --decoder-max-entries "$HFM_DECODER_MAX_ENTRIES"
    --epochs "$HFM_EPOCHS"
    --batch-size "$HFM_BATCH_SIZE"
    --lr "$HFM_LR"
    --dim "$HFM_DIM"
    --n-steps "$HFM_N_STEPS"
    --device cuda
    --save-every "$HFM_SAVE_EVERY"
    --seed "$HFM_SEED"
  )
  if [[ -n "$HFM_RESUME" ]]; then
    train_args+=(--resume "$HFM_RESUME")
  fi
  exec stdbuf -oL -eL "$PYTHON_BIN" -u -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    "${train_args[@]}"
' &

runner_pid="$!"
echo "$runner_pid" > "$PID_FILE"
cat > "$RUN_MANIFEST" <<EOF
{
  "started_at": "$STARTED_AT",
  "project_root": "$PROJECT_ROOT",
  "data_dir": "$HFM_DATA_DIR",
  "output_dir": "$HFM_OUTPUT_DIR",
  "humu_checkpoint": "$HUMU_CHECKPOINT_PATH",
  "decoder_artifact": "$HFM_DECODER_PATH",
  "decoder_max_entries": $HFM_DECODER_MAX_ENTRIES,
  "python_bin": "$PYTHON_BIN",
  "world_size": $NPROC_PER_NODE,
  "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES",
  "epochs": $HFM_EPOCHS,
  "batch_size": $HFM_BATCH_SIZE,
  "learning_rate": "$HFM_LR",
  "dim": $HFM_DIM,
  "n_steps": $HFM_N_STEPS,
  "save_every": $HFM_SAVE_EVERY,
  "seed": $HFM_SEED,
  "resume": "$HFM_RESUME",
  "log_file": "$LOG_FILE",
  "pid_file": "$PID_FILE",
  "runner_pid": $runner_pid,
  "command": "$PYTHON_BIN -u -m torch.distributed.run --standalone --nproc_per_node=$NPROC_PER_NODE models/mf-generators/hfm_3d/train.py --data $HFM_DATA_DIR --output-dir $HFM_OUTPUT_DIR"
}
EOF

echo "Started HFM-3D training in background."
echo "PID: $runner_pid"
echo "PID file: $PID_FILE"
echo "Log file: $LOG_FILE"
echo "Run manifest: $RUN_MANIFEST"
echo "Follow logs: tail -f \"$LOG_FILE\""
