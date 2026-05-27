"""Core Sigstore signing and verification engine.

Uses the real sigstore-python v3 API to communicate with
Fulcio (certificate authority) and Rekor (transparency log).

Install: pip install "sigstore>=3.0.0"
Docs:    https://sigstore.github.io/sigstore-python/
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sigstore.models import Bundle
from sigstore.sign import SigningContext
from sigstore.verify import Verifier
from sigstore.verify.policy import Identity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SigstoreConfig:
    """Runtime config for signing operations."""

    env: str = "production"           # "production" | "staging"
    oidc_strategy: str = "env"        # "env" | "github" | "interactive"
    bundle_dir: Path | None = None    # None = alongside artifact

    def get_signing_context(self) -> SigningContext:
        if self.env == "staging":
            return SigningContext.staging()
        return SigningContext.production()

    def get_verifier(self) -> Verifier:
        if self.env == "staging":
            return Verifier.staging()
        return Verifier.production()


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

@dataclass
class SignResult:
    artifact_path: str
    bundle_path: str
    digest_hex: str
    rekor_log_index: int | None = None
    certificate_pem: str | None = None


@dataclass
class VerifyResult:
    artifact_path: str
    bundle_path: str
    valid: bool
    identity: str | None = None
    issuer: str | None = None
    rekor_log_index: int | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Signer
# ---------------------------------------------------------------------------

class SigstoreManager:
    """Production-grade Sigstore signing and verification.

    Usage:
        mgr = SigstoreManager(config=SigstoreConfig())

        # Sign — requires OIDC token in environment
        result = mgr.sign_file("/data/dataset.csv")

        # Verify
        ok = mgr.verify_file(
            "/data/dataset.csv",
            "/data/dataset.csv.sigstore",
            identity="https://github.com/org/repo/.github/workflows/ci.yml@refs/heads/main",
            issuer="https://token.actions.githubusercontent.com",
        )
    """

    def __init__(self, config: SigstoreConfig | None = None):
        self.config = config or SigstoreConfig()

    # ------------------------------------------------------------------
    # Signing
    # ------------------------------------------------------------------

    def sign_file(self, file_path: str | Path, bundle_path: str | Path | None = None) -> SignResult:
        """Sign a file and write the .sigstore bundle.

        Requires a valid OIDC token (see oidc_token.py).
        """
        artifact = Path(file_path)
        if not artifact.is_file():
            raise FileNotFoundError(f"Artifact not found: {artifact}")

        digest_hex = _sha256_hex(artifact)

        # Resolve bundle output path
        if bundle_path is None:
            if self.config.bundle_dir:
                bundle_out = self.config.bundle_dir / f"{artifact.name}.sigstore"
            else:
                bundle_out = artifact.with_suffix(artifact.suffix + ".sigstore")
        else:
            bundle_out = Path(bundle_path)
        bundle_out.parent.mkdir(parents=True, exist_ok=True)

        # Inject OIDC token into env if strategy is not interactive
        _inject_oidc_token(self.config.oidc_strategy)

        logger.info("Signing %s (sha256=%s)", artifact, digest_hex[:16])

        ctx = self.config.get_signing_context()
        with ctx.signer() as signer:
            # sign_artifact accepts raw bytes
            artifact_bytes = artifact.read_bytes()
            bundle: Bundle = signer.sign_artifact(artifact_bytes)

        # Persist bundle
        bundle_out.write_text(bundle.to_json())
        logger.info("Bundle written to %s", bundle_out)

        # Try to extract Rekor log index from bundle (may not always be present)
        rekor_idx = _extract_rekor_log_index(bundle)

        return SignResult(
            artifact_path=str(artifact),
            bundle_path=str(bundle_out),
            digest_hex=digest_hex,
            rekor_log_index=rekor_idx,
        )

    def sign_bytes(self, data: bytes, bundle_path: str | Path) -> SignResult:
        """Sign raw bytes and write the .sigstore bundle."""
        digest_hex = hashlib.sha256(data).hexdigest()

        bundle_out = Path(bundle_path)
        bundle_out.parent.mkdir(parents=True, exist_ok=True)

        _inject_oidc_token(self.config.oidc_strategy)

        logger.info("Signing %d bytes (sha256=%s)", len(data), digest_hex[:16])

        ctx = self.config.get_signing_context()
        with ctx.signer() as signer:
            bundle: Bundle = signer.sign_artifact(data)

        bundle_out.write_text(bundle.to_json())
        logger.info("Bundle written to %s", bundle_out)

        rekor_idx = _extract_rekor_log_index(bundle)

        return SignResult(
            artifact_path="<bytes>",
            bundle_path=str(bundle_out),
            digest_hex=digest_hex,
            rekor_log_index=rekor_idx,
        )

    def sign_json(self, obj: Any, bundle_path: str | Path) -> SignResult:
        """Sign a JSON-serializable object (canonical JSON encoding)."""
        canonical = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
        return self.sign_bytes(canonical, bundle_path)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_file(
        self,
        file_path: str | Path,
        bundle_path: str | Path,
        identity: str,
        issuer: str,
    ) -> VerifyResult:
        """Verify a signed artifact against a certificate identity policy.

        Parameters
        ----------
        file_path : path to the original artifact
        bundle_path : path to the .sigstore bundle
        identity : expected certificate identity
            (e.g. "https://github.com/org/repo/.github/workflows/ci.yml@refs/heads/main")
        issuer : expected OIDC issuer
            (e.g. "https://token.actions.githubusercontent.com")
        """
        artifact = Path(file_path)
        bpath = Path(bundle_path)

        if not artifact.is_file():
            raise FileNotFoundError(f"Artifact not found: {artifact}")
        if not bpath.is_file():
            raise FileNotFoundError(f"Bundle not found: {bpath}")

        logger.info("Verifying %s against identity=%s issuer=%s", artifact, identity, issuer)

        try:
            bundle = Bundle.from_json(bpath.read_text())
            policy = Identity(identity=identity, issuer=issuer)
            verifier = self.config.get_verifier()

            with verifier as vctx:
                materials = vctx.verify_artifact(
                    artifact.read_bytes(),
                    bundle,
                    policy,
                )

            rekor_idx = _extract_rekor_log_index(bundle)

            logger.info("Verification PASSED for %s", artifact)
            return VerifyResult(
                artifact_path=str(artifact),
                bundle_path=str(bpath),
                valid=True,
                identity=identity,
                issuer=issuer,
                rekor_log_index=rekor_idx,
            )

        except Exception as exc:
            logger.warning("Verification FAILED for %s: %s", artifact, exc)
            return VerifyResult(
                artifact_path=str(artifact),
                bundle_path=str(bpath),
                valid=False,
                identity=identity,
                issuer=issuer,
                error=str(exc),
            )

    def verify_bytes(
        self,
        data: bytes,
        bundle_path: str | Path,
        identity: str,
        issuer: str,
    ) -> VerifyResult:
        """Verify raw bytes against a bundle and identity policy."""
        bpath = Path(bundle_path)
        if not bpath.is_file():
            raise FileNotFoundError(f"Bundle not found: {bpath}")

        try:
            bundle = Bundle.from_json(bpath.read_text())
            policy = Identity(identity=identity, issuer=issuer)
            verifier = self.config.get_verifier()

            with verifier as vctx:
                vctx.verify_artifact(data, bundle, policy)

            rekor_idx = _extract_rekor_log_index(bundle)

            return VerifyResult(
                artifact_path="<bytes>",
                bundle_path=str(bpath),
                valid=True,
                identity=identity,
                issuer=issuer,
                rekor_log_index=rekor_idx,
            )

        except Exception as exc:
            return VerifyResult(
                artifact_path="<bytes>",
                bundle_path=str(bpath),
                valid=False,
                identity=identity,
                issuer=issuer,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _inject_oidc_token(strategy: str) -> None:
    """If using a non-interactive strategy, ensure SIGSTORE_ID_TOKEN is set."""
    if strategy == "interactive":
        return  # sigstore handles it

    if os.environ.get("SIGSTORE_ID_TOKEN"):
        return  # already set

    from oidc_token import resolve_token
    token = resolve_token(strategy)
    if token:
        os.environ["SIGSTORE_ID_TOKEN"] = token


def _extract_rekor_log_index(bundle: Bundle) -> int | None:
    """Try to extract the Rekor transparency log index from a bundle."""
    try:
        # The bundle's internal structure may expose log index
        # This depends on the sigstore-python version
        if hasattr(bundle, "_inner"):
            tlog_entries = bundle._inner.verification_material.tlog_entries
            if tlog_entries:
                return int(tlog_entries[0].log_index)
    except Exception:
        pass
    return None
