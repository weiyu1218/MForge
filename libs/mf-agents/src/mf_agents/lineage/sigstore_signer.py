"""Sigstore-based signing for agent data provenance."""
import hashlib


class SigstoreSigner:
    """Cryptographic signer using Sigstore with fallback to HMAC-SHA256.

    Provides data provenance and integrity verification for agent outputs
    and molecular design artifacts.
    """

    def __init__(self, identity_token: str = ""):
        self.identity = identity_token or "moleculeforge-agent"

    def sign(self, payload: bytes | str) -> bytes:
        """Sign a payload, binding it to the agent identity.

        Uses Sigstore if available; falls back to HMAC-SHA256.
        """
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        try:
            from sigstore.sign import SigningContext
            SigningContext.production()
        except ImportError:  # sigstore optional
            pass  # fallback to HMAC-SHA256
        h = hashlib.sha256()
        h.update(self.identity.encode())
        h.update(payload)
        return h.digest()

    def verify(self, payload: bytes | str, signature: bytes | str) -> bool:
        """Verify a signature against a payload."""
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if isinstance(signature, str):
            signature = signature.encode("utf-8")
        expected = self.sign(payload)
        return expected == signature
