#!/usr/bin/env bash
# Smoke test for the ADMET inference service.
# Usage: bash test_service.sh [BASE_URL]

set -euo pipefail

BASE_URL="${1:-http://localhost:8901}"

echo "=== Health check ==="
curl -sf "${BASE_URL}/health" | python3 -m json.tool
echo ""

echo "=== Predict 3 molecules ==="
curl -sf -X POST "${BASE_URL}/predict" \
  -H 'Content-Type: application/json' \
  -d '{
    "smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
    "batch_size": 32
  }' | python3 -m json.tool
echo ""

echo "=== Predict with specific endpoints ==="
curl -sf -X POST "${BASE_URL}/predict" \
  -H 'Content-Type: application/json' \
  -d '{
    "smiles": ["CCO", "c1ccccc1"],
    "endpoints": ["solubility", "lipophilicity"],
    "batch_size": 16
  }' | python3 -m json.tool
echo ""

echo "=== Batch stress test (50 molecules) ==="
# Generate 50 test SMILES
SMILES=$(python3 -c "
smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCCC', 'c1ccc(O)cc1'] * 10
import json; print(json.dumps(smiles))
")
curl -sf -X POST "${BASE_URL}/predict" \
  -H 'Content-Type: application/json' \
  -d "{\"smiles\": ${SMILES}, \"batch_size\": 64}" \
  -w '\nHTTP %{http_code} — %{time_total}s\n' -o /dev/null

echo ""
echo "All tests passed."
