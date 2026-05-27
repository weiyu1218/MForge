"""Pydantic models for the Sigstore signing service API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SignFileRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to the artifact to sign")
    bundle_path: str | None = Field(
        default=None,
        description="Output path for .sigstore bundle. None = alongside artifact.",
    )


class SignBytesRequest(BaseModel):
    data_base64: str = Field(..., description="Base64-encoded artifact data")
    bundle_path: str = Field(..., description="Output path for .sigstore bundle")


class SignJsonRequest(BaseModel):
    data: dict | list = Field(..., description="JSON-serializable object to sign")
    bundle_path: str = Field(..., description="Output path for .sigstore bundle")


class VerifyFileRequest(BaseModel):
    file_path: str = Field(..., description="Path to the original artifact")
    bundle_path: str = Field(..., description="Path to the .sigstore bundle")
    identity: str = Field(
        ...,
        description=(
            "Expected certificate identity "
            "(e.g. https://github.com/org/repo/.github/workflows/ci.yml@refs/heads/main)"
        ),
    )
    issuer: str = Field(
        ...,
        description=(
            "Expected OIDC issuer "
            "(e.g. https://token.actions.githubusercontent.com)"
        ),
    )


class VerifyBytesRequest(BaseModel):
    data_base64: str = Field(..., description="Base64-encoded original data")
    bundle_path: str = Field(..., description="Path to the .sigstore bundle")
    identity: str
    issuer: str


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SignResponse(BaseModel):
    success: bool
    artifact_path: str
    bundle_path: str
    digest_hex: str
    rekor_log_index: int | None = None
    error: str | None = None


class VerifyResponse(BaseModel):
    valid: bool
    artifact_path: str
    bundle_path: str
    identity: str | None = None
    issuer: str | None = None
    rekor_log_index: int | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    sigstore_env: str
    oidc_strategy: str
    oidc_token_present: bool
