#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file not found: $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -n "$RESUME_FROM" && ! -f "$RESUME_FROM" ]]; then
  echo "Resume checkpoint not found: $RESUME_FROM" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CONFIG_HASH="$(sha256sum "$CONFIG_PATH" | awk '{print $1}')"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Training is already running for PID file $PID_FILE: $existing_pid" >&2
    exit 1
  fi
fi

export PROJECT_ROOT CONFIG_PATH PYTHON_BIN NPROC_PER_NODE
export CUDA_VISIBLE_DEVICES NCCL_DEBUG TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC
export TORCH_NCCL_ENABLE_MONITORING OMP_NUM_THREADS RESUME_FROM LOG_FILE RUN_MANIFEST
export PYTORCH_CUDA_ALLOC_CONF PYTHONUNBUFFERED

setsid bash -c '
  set -Eeuo pipefail
  cd "$PROJECT_ROOT"
  exec > >(stdbuf -oL tee -a "$LOG_FILE" >/dev/null) 2>&1
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "project_root=$PROJECT_ROOT"
  echo "config=$CONFIG_PATH"
  echo "run_manifest=$RUN_MANIFEST"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "nproc_per_node=$NPROC_PER_NODE"
  echo "torch_nccl_heartbeat_timeout_sec=$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"
  echo "torch_nccl_enable_monitoring=$TORCH_NCCL_ENABLE_MONITORING"
  echo "resume_from=$RESUME_FROM"
  echo "log_file=$LOG_FILE"
  train_args=(pipelines/humu_pretrain/train.py --config "$CONFIG_PATH")
  if [[ -n "$RESUME_FROM" ]]; then
    train_args+=(--resume "$RESUME_FROM")
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
  "config_path": "$CONFIG_PATH",
  "config_hash": "$CONFIG_HASH",
  "python_bin": "$PYTHON_BIN",
  "world_size": $NPROC_PER_NODE,
  "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES",
  "torch_nccl_heartbeat_timeout_sec": "$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
  "torch_nccl_enable_monitoring": "$TORCH_NCCL_ENABLE_MONITORING",
  "resume_from": "$RESUME_FROM",
  "log_file": "$LOG_FILE",
  "pid_file": "$PID_FILE",
  "runner_pid": $runner_pid,
  "command": "$PYTHON_BIN -u -m torch.distributed.run --standalone --nproc_per_node=$NPROC_PER_NODE pipelines/humu_pretrain/train.py --config $CONFIG_PATH${RESUME_FROM:+ --resume $RESUME_FROM}"
}
EOF

echo "Started HUMU training in background."
echo "PID: $runner_pid"
echo "PID file: $PID_FILE"
echo "Log file: $LOG_FILE"
echo "Run manifest: $RUN_MANIFEST"
echo "Follow logs: tail -f \"$LOG_FILE\""
