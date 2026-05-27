# Sigstore Signing Microservice

Standalone signing & verification service using the real [sigstore-python](https://sigstore.github.io/sigstore-python/) v3 API. Communicates with production Fulcio (certificate authority) and Rekor (transparency log).

## Architecture

```
┌──────────────┐     OIDC JWT      ┌─────────┐
│  OIDC Provider├──────────────────►│         │
│  (GitHub/     │                   │  This   │
│   Google/     │   sign request    │ Service │──► Fulcio (short-lived cert)
│   Keycloak)   │◄──────────────────┤         │──► Rekor  (transparency log)
└──────────────┘                    │         │
                                    └────┬────┘
                                         │ .sigstore bundle
                                         ▼
                                    artifact + bundle
```

**Signing flow:**
1. Get OIDC JWT (env var, GitHub Actions, or browser)
2. Send JWT + artifact digest to Fulcio → receive short-lived x509 cert
3. Sign artifact hash with the private key (ephemeral, generated in-process)
4. Record signature + cert in Rekor transparency log
5. Output `.sigstore` bundle (cert + signature + Rekor proof)

**Verification flow:**
1. Load `.sigstore` bundle
2. Verify signature matches artifact digest
3. Verify certificate chain → OIDC identity
4. Verify Rekor log entry inclusion proof
5. Confirm identity matches expected policy

## Quick Start

### 1. Install

```bash
conda env create -f environment.yml
conda activate sigstore-svc
```

Or manually:
```bash
conda create -n sigstore-svc python=3.11 -y
conda activate sigstore-svc
pip install "sigstore>=3.0.0" fastapi "uvicorn[standard]" pydantic httpx
```

### 2. Set OIDC Token

```bash
# Option A: direct token
export SIGSTORE_ID_TOKEN="eyJhbGciOiJSUzI1NiIs..."

# Option B: GitHub Actions (automatic in workflows with id-token: write)
# No action needed — the service detects ACTIONS_ID_TOKEN_REQUEST_URL

# Option C: interactive browser (local dev only)
# Set OIDC_STRATEGY=interactive in config.py
```

### 3. Start Service

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8902
```

### 4. Test

```bash
bash test_service.sh
```

## API

### `GET /health`
Returns service status, OIDC token presence, and configuration.

### `POST /sign/file`
Sign a file on disk. Produces a `.sigstore` bundle.
```json
{
  "file_path": "/data/dataset.csv",
  "bundle_path": "/data/dataset.csv.sigstore"
}
```

### `POST /sign/json`
Sign a JSON object using canonical encoding.
```json
{
  "data": {"smiles": ["CCO"], "score": 0.88},
  "bundle_path": "/data/prediction.sigstore"
}
```

### `POST /sign/bytes`
Sign raw bytes (base64-encoded).
```json
{
  "data_base64": "SGVsbG8gV29ybGQ=",
  "bundle_path": "/data/model.sigstore"
}
```

### `POST /verify/file`
Verify a file against its bundle and an identity policy.
```json
{
  "file_path": "/data/dataset.csv",
  "bundle_path": "/data/dataset.csv.sigstore",
  "identity": "https://github.com/org/repo/.github/workflows/ci.yml@refs/heads/main",
  "issuer": "https://token.actions.githubusercontent.com"
}
```

## CLI

```bash
# Sign
export SIGSTORE_ID_TOKEN="..."
python cli.py sign --file dataset.csv
python cli.py sign --json '{"key":"val"}' --bundle out.sigstore

# Verify
python cli.py verify \
  --file dataset.csv \
  --bundle dataset.csv.sigstore \
  --identity "https://github.com/org/repo/.github/workflows/ci.yml@refs/heads/main" \
  --issuer "https://token.actions.githubusercontent.com"
```

## Client Library

```python
from client import SigstoreClient

c = SigstoreClient("http://localhost:8902")
result = c.sign_file("/data/dataset.csv")
print(result["bundle_path"])

ok = c.verify_file(
    "/data/dataset.csv",
    result["bundle_path"],
    identity="https://github.com/org/repo/...",
    issuer="https://token.actions.githubusercontent.com",
)
```

## OIDC Token Sources

| Source | Strategy | How |
|---|---|---|
| Environment variable | `env` | `export SIGSTORE_ID_TOKEN="..."` |
| GitHub Actions | `github` | Automatic with `permissions: id-token: write` |
| Browser OAuth | `interactive` | Opens browser for Google/GitHub login (dev only) |

Set in `config.py` via `OIDC_STRATEGY`.

## GitHub Actions Example

```yaml
jobs:
  sign-artifacts:
    permissions:
      id-token: write  # required for OIDC
      contents: read
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: python cli.py sign --file build/output.tar.gz
```

## Config

Edit `config.py`:
- `SIGSTORE_ENV`: "production" or "staging"
- `OIDC_STRATEGY`: "env", "github", or "interactive"
- `PORT`: service port (default 8902)
- `BUNDLE_DIR`: where to write bundles (None = alongside artifact)

## Docker

```bash
docker build -t sigstore-svc .
docker run -p 8902:8902 -e SIGSTORE_ID_TOKEN="$SIGSTORE_ID_TOKEN" sigstore-svc
```
