#!/usr/bin/env bash
set -Eeuo pipefail

PID_FILE="${1:-}"

if [[ -z "$PID_FILE" ]]; then
  echo "Usage: $0 <pid-file>" >&2
  exit 2
fi

if [[ ! -f "$PID_FILE" ]]; then
  echo "PID file not found: $PID_FILE" >&2
  exit 1
fi

pid="$(cat "$PID_FILE")"
if [[ -z "$pid" ]]; then
  echo "PID file is empty: $PID_FILE" >&2
  exit 1
fi

if ! kill -0 "$pid" 2>/dev/null; then
  echo "Process is not running: $pid"
  exit 0
fi

pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
if [[ -z "$pgid" ]]; then
  echo "Process group not found for PID: $pid" >&2
  exit 1
fi

kill -TERM "-$pgid"
for _ in $(seq 1 30); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Stopped process group: $pgid"
    exit 0
  fi
  sleep 1
done

kill -KILL "-$pgid"
echo "Force stopped process group: $pgid"
