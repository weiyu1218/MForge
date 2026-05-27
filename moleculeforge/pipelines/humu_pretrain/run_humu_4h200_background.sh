#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-/tmp/humu_4h200.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
RESUME_FROM="${RESUME_FROM:-}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/humu_pretrain}"
RUN_NAME="${RUN_NAME:-humu_4h200_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_NAME.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/$RUN_NAME.pid}"
RUN_MANIFEST="${RUN_MANIFEST:-$LOG_DIR/$RUN_NAME.manifest.json}"

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
export CUDA_VISIBLE_DEVICES NCCL_DEBUG OMP_NUM_THREADS RESUME_FROM LOG_FILE RUN_MANIFEST
export PYTHONUNBUFFERED=1

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
