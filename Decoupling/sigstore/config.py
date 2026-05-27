"""Configuration for the Sigstore signing service."""

from pathlib import Path

# Server
HOST = "0.0.0.0"
PORT = 8902

# Sigstore environment: "production" or "staging"
SIGSTORE_ENV = "production"

# OIDC token source priority:
#   1. env       — read SIGSTORE_ID_TOKEN from process environment
#   2. github    — GitHub Actions OIDC (ACTIONS_ID_TOKEN_REQUEST_URL)
#   3. interactive — browser-based OAuth (local dev only)
OIDC_STRATEGY = "env"

# Where to store .sigstore bundle files (None = alongside the artifact)
BUNDLE_DIR: Path | None = None

# API authentication — optional bearer token to protect the service
API_TOKEN: str | None = None
