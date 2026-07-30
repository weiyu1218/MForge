"""Sigstore-based signing for agent data provenance."""

import hashlib
import hmac
import json
import os
import shlex
import subprocess

from mf_core.artifacts import CommandRequirement, check_command, require_available

_SIGSTORE_SIGN_COMMAND = CommandRequirement(
    "sigstore_sign_command",
    "SIGSTORE_SIGN_COMMAND",
    required=False,
)
_SIGSTORE_VERIFY_COMMAND = CommandRequirement(
    "sigstore_verify_command",
    "SIGSTORE_VERIFY_COMMAND",
    required=False,
)


class SigstoreSigner:
    """Cryptographic signer using Sigstore with fallback to HMAC-SHA256.

    Provides data provenance and integrity verification for agent outputs
    and molecular design artifacts.
    """

    def __init__(self, identity_token: str = ""):
        self.identity = identity_token or "moleculeforge-agent"
        self.sign_command = os.getenv("SIGSTORE_SIGN_COMMAND", "").strip()
        self.verify_command = os.getenv("SIGSTORE_VERIFY_COMMAND", "").strip()
        self.rekor_url = os.getenv("SIGSTORE_REKOR_URL", "https://rekor.sigstore.dev")
        self.identity_token = os.getenv("SIGSTORE_IDENTITY_TOKEN", "").strip()
        self._signature_cache: dict[str, dict] = {}

    def sign(self, payload: bytes | str) -> bytes:
        """Sign a payload, binding it to the agent identity.

        Uses Sigstore if available; falls back to HMAC-SHA256.
        """
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if self.sign_command:
            return self._sign_with_command(payload)
        return self._local_signature(payload)

    def _local_signature(self, payload: bytes) -> bytes:
        return hmac.new(self.identity.encode(), payload, hashlib.sha256).digest()

    def verify(self, payload: bytes | str, signature: bytes | str) -> bool:
        """Verify a signature against a payload."""
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if isinstance(signature, str):
            signature = signature.encode("utf-8")
        if self.verify_command:
            return self._verify_with_command(payload, signature)
        signature_text = signature.decode("utf-8", errors="replace")
        if self.sign_command:
            cached = self._signature_cache.get(signature_text)
            return bool(cached and cached.get("payload_hash") == _payload_hash(payload))
        expected = self._local_signature(payload)
        return hmac.compare_digest(expected, signature)

    def _sign_with_command(self, payload: bytes) -> bytes:
        _require_command_available(_SIGSTORE_SIGN_COMMAND, self.sign_command)
        payload_hash = _payload_hash(payload)
        result = subprocess.run(
            shlex.split(self.sign_command),
            input=json.dumps(
                {
                    "artifact_id": f"agent-lineage-{payload_hash}",
                    "artifact_type": "agent_lineage_payload",
                    "payload_hash": payload_hash,
                    "identity": self.identity,
                    "identity_token": self.identity_token,
                    "rekor_url": self.rekor_url,
                },
                sort_keys=True,
            ).encode("utf-8"),
            capture_output=True,
            timeout=float(os.getenv("SIGSTORE_COMMAND_TIMEOUT_SECONDS", "30")),
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Sigstore signing command failed: {stderr}")
        try:
            response = json.loads(result.stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Sigstore signing command returned invalid JSON") from exc
        if not isinstance(response, dict) or not response.get("signature"):
            raise RuntimeError("Sigstore signing command must return a signature")
        signature = str(response["signature"])
        self._signature_cache[signature] = {
            "payload_hash": str(response.get("payload_hash") or payload_hash),
            "signature_type": str(response.get("signature_type") or "sigstore_rekor"),
            "certificate": response.get("certificate"),
            "rekor_entry": response.get("rekor_entry"),
            "bundle": response.get("bundle"),
        }
        return signature.encode("utf-8")

    def _verify_with_command(self, payload: bytes, signature: bytes) -> bool:
        _require_command_available(_SIGSTORE_VERIFY_COMMAND, self.verify_command)
        payload_hash = _payload_hash(payload)
        signature_text = signature.decode("utf-8", errors="replace")
        result = subprocess.run(
            shlex.split(self.verify_command),
            input=json.dumps(
                {
                    "artifact_id": f"agent-lineage-{payload_hash}",
                    "artifact_type": "agent_lineage_payload",
                    "payload_hash": payload_hash,
                    "signature": signature_text,
                    "identity": self.identity,
                    "expected_identity": self.identity,
                    "bundle": self._signature_cache.get(signature_text) or {},
                    "rekor_url": self.rekor_url,
                },
                sort_keys=True,
            ).encode("utf-8"),
            capture_output=True,
            timeout=float(os.getenv("SIGSTORE_COMMAND_TIMEOUT_SECONDS", "30")),
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Sigstore verification command failed: {stderr}")
        try:
            response = json.loads(result.stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Sigstore verification command returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise RuntimeError("Sigstore verification command must return a JSON object")
        if "valid" in response:
            return response["valid"] is True
        if "signature_valid" in response:
            return response["signature_valid"] is True
        raise RuntimeError("Sigstore verification command must return valid")


def _payload_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_command_available(
    requirement: CommandRequirement,
    command: str,
) -> None:
    required_requirement = CommandRequirement(
        requirement.name,
        requirement.env_var,
        required=True,
    )
    env = {**os.environ, requirement.env_var: command}
    require_available([check_command(required_requirement, env=env)])
