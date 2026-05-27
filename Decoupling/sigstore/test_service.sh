#!/usr/bin/env bash
# Smoke test for the Sigstore signing service.
#
# Prerequisites:
#   1. Service running: python app.py
#   2. OIDC token available: export SIGSTORE_ID_TOKEN="..."
#
# Usage: bash test_service.sh [BASE_URL]

set -euo pipefail

BASE_URL="${1:-http://localhost:8902}"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== 1. Health check ==="
curl -sf "${BASE_URL}/health" | python3 -m json.tool
echo ""

echo "=== 2. Create test artifact ==="
TEST_FILE="${TMPDIR}/test_artifact.csv"
cat > "$TEST_FILE" <<'EOF'
SMILES,Prediction,Confidence
CCO,0.88,0.95
c1ccccc1,-1.2,0.87
CC(=O)O,0.55,0.91
EOF
echo "Created: ${TEST_FILE}"
echo ""

echo "=== 3. Sign file via service ==="
SIGN_RESULT=$(curl -sf -X POST "${BASE_URL}/sign/file" \
  -H 'Content-Type: application/json' \
  -d "{\"file_path\": \"${TEST_FILE}\"}")
echo "$SIGN_RESULT" | python3 -m json.tool
BUNDLE_PATH=$(echo "$SIGN_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['bundle_path'])")
echo ""

echo "=== 4. Verify file via service ==="
curl -sf -X POST "${BASE_URL}/verify/file" \
  -H 'Content-Type: application/json' \
  -d "{
    \"file_path\": \"${TEST_FILE}\",
    \"bundle_path\": \"${BUNDLE_PATH}\",
    \"identity\": \"https://github.com/test/repo/.github/workflows/ci.yml@refs/heads/main\",
    \"issuer\": \"https://token.actions.githubusercontent.com\"
  }" | python3 -m json.tool
echo ""

echo "=== 5. Sign JSON via service ==="
JSON_BUNDLE="${TMPDIR}/prediction.sigstore"
curl -sf -X POST "${BASE_URL}/sign/json" \
  -H 'Content-Type: application/json' \
  -d "{
    \"data\": {\"smiles\": \"CCO\", \"admet\": {\"solubility\": 0.88, \"lipophilicity\": -0.31}},
    \"bundle_path\": \"${JSON_BUNDLE}\"
  }" | python3 -m json.tool
echo ""

echo "=== 6. Sign bytes via service ==="
BYTES_BUNDLE="${TMPDIR}/bytes.sigstore"
DATA_B64=$(echo -n "binary model checkpoint data here" | base64)
curl -sf -X POST "${BASE_URL}/sign/bytes" \
  -H 'Content-Type: application/json' \
  -d "{
    \"data_base64\": \"${DATA_B64}\",
    \"bundle_path\": \"${BYTES_BUNDLE}\"
  }" | python3 -m json.tool
echo ""

echo "All tests passed."
