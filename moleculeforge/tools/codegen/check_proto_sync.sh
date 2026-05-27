#!/bin/bash
# Check that generated proto code is in sync with .proto definitions
set -euo pipefail

PROTO_DIR="protos/"
GEN_DIR="libs/mf-core/src/mf_core/proto_gen/"

echo "Checking proto file count..."
PROTO_COUNT=$(find "$PROTO_DIR" -name "*.proto" | wc -l)
echo "Proto files: $PROTO_COUNT"

echo "Checking generated files..."
if [ -d "$GEN_DIR" ]; then
  GEN_COUNT=$(find "$GEN_DIR" -name "*.py" | wc -l)
  echo "Generated Python files: $GEN_COUNT"
else
  echo "proto_gen/ directory not found - run uv run python tools/dev/generate_protos.py"
  exit 1
fi

if [ -x ".venv/bin/python" ]; then
  PYTHON_CMD=(".venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=("uv" "run" "python")
else
  PYTHON_CMD=("python")
fi

PB2_COUNT=$(find "$GEN_DIR" -name "*_pb2.py" | wc -l)
if [ "$PB2_COUNT" -ne "$PROTO_COUNT" ]; then
  echo "Generated pb2 count $PB2_COUNT does not match proto count $PROTO_COUNT"
  exit 1
fi

PYTHONPATH="libs/mf-core/src:${PYTHONPATH:-}" "${PYTHON_CMD[@]}" - <<'PY'
import importlib
from pathlib import Path

root = Path("libs/mf-core/src/mf_core/proto_gen")
for path in sorted(root.rglob("*_pb2_grpc.py")):
    rel = path.relative_to(root).with_suffix("")
    module = "mf_core.proto_gen." + ".".join(rel.parts)
    importlib.import_module(module)
PY

echo "Proto sync check complete."
