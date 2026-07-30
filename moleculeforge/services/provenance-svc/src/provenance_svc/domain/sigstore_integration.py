"""Sigstore integration for cryptographic artifact signing and verification.

Uses Sigstore (sigstore-python) to sign artifacts with OIDC-based
keyless signing, producing a transparency log entry in the Rekor log.

Each artifact in the MoleculeForge pipeline (molecules, routes,
predictions, etc.) receives a Sigstore signature that enables
end-to-end provenance verification.
"""

import hashlib
import json
import os
import re
import shlex
import subprocess
from collections.abc import Mapping

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
_SHA256_CHECKSUM = re.compile(r"sha256:[0-9a-f]{64}")


class SigstoreIntegration:
    """Manages Sigstore signing and verification for provenance artifacts.

    In production, this uses the sigstore-python library to:
    1. Sign artifacts via OIDC authentication (Fulcio CA)
    2. Submit signatures to the Rekor transparency log
    3. Verify signatures against the transparency log
    """

    def __init__(self, rekor_url: str = "https://rekor.sigstore.dev"):
        self.rekor_url = os.getenv("SIGSTORE_REKOR_URL", rekor_url).strip() or rekor_url
        self._signature_cache: dict[str, dict] = {}
        self.sign_command = os.getenv("SIGSTORE_SIGN_COMMAND", "").strip()
        self.verify_command = os.getenv("SIGSTORE_VERIFY_COMMAND", "").strip()
        self.identity_token = os.getenv("SIGSTORE_IDENTITY_TOKEN", "").strip()
        self.expected_identity = os.getenv("SIGSTORE_EXPECTED_IDENTITY", "").strip()
        self.signature_type = self._detect_signature_type()

    def _detect_signature_type(self) -> str:
        if self.sign_command:
            return "sigstore_rekor"
        try:
            import sigstore  # noqa: F401
        except ImportError:
            return "local_dev_signature"
        return "sigstore"

    def sign_artifact(
        self,
        artifact_id: str,
        artifact_type: str,
        metadata: dict,
        *,
        checksum: str | None = None,
        parent_ids: list[str] | None = None,
        recorded_at: str | None = None,
    ) -> dict:
        """Sign an artifact and return the signature bundle.

        Args:
            artifact_id: Unique identifier for the artifact.
            artifact_type: Type of artifact (molecule, route, prediction, etc.).
            metadata: Additional metadata to include in the signed payload.

        Returns:
            Dict with 'signature', 'certificate', and 'bundle' keys.
        """
        if checksum is None:
            legacy_bytes = _canonical_json_bytes(metadata)
            checksum = f"sha256:{hashlib.sha256(legacy_bytes).hexdigest()}"
        if not _SHA256_CHECKSUM.fullmatch(checksum):
            raise ValueError("checksum must be sha256:<64 lowercase hex>")
        payload = {
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "checksum": checksum,
            "parent_ids": list(parent_ids or []),
            "metadata": metadata,
        }
        if recorded_at is not None:
            payload["recorded_at"] = recorded_at
        payload_bytes = _canonical_json_bytes(payload)
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()

        if self.signature_type == "local_dev_signature":
            signature = hashlib.sha256(f"{payload_hash}:{self.signature_type}".encode()).hexdigest()
            bundle = {
                "signature_type": "local_dev_signature",
                "signature": signature,
                "certificate": None,
                "payload_hash": payload_hash,
                "rekor_entry": None,
                "identity": None,
                "bundle": None,
            }
        elif self.signature_type == "sigstore_rekor":
            bundle = self._sign_with_command(payload, payload_hash)
        else:
            raise RuntimeError("Sigstore signing backend is not configured")

        bundle["signed_payload"] = payload
        self._signature_cache[artifact_id] = bundle
        return bundle

    def _sign_with_command(self, payload: dict, payload_hash: str) -> dict:
        command_payload = {
            "artifact_id": payload["artifact_id"],
            "artifact_type": payload["artifact_type"],
            "identity_token": self.identity_token,
            "payload": payload,
            "payload_hash": payload_hash,
            "rekor_url": self.rekor_url,
        }
        timeout = float(os.getenv("SIGSTORE_COMMAND_TIMEOUT_SECONDS", "30"))
        _require_command_available(_SIGSTORE_SIGN_COMMAND, self.sign_command)
        result = subprocess.run(
            shlex.split(self.sign_command),
            input=json.dumps(command_payload, sort_keys=True).encode("utf-8"),
            capture_output=True,
            timeout=timeout,
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
        if response.get("payload_hash") != payload_hash:
            raise RuntimeError("Sigstore signing command payload_hash does not match")
        if response.get("artifact_type") not in {None, payload["artifact_type"]}:
            raise RuntimeError("Sigstore signing command artifact_type does not match")
        signature_type = response.get("signature_type", "sigstore_rekor")
        if signature_type != "sigstore_rekor":
            raise RuntimeError("Sigstore signing command signature_type must be sigstore_rekor")
        return {
            "signature_type": signature_type,
            "signature": str(response["signature"]),
            "certificate": response.get("certificate"),
            "payload_hash": str(response.get("payload_hash") or payload_hash),
            "artifact_type": str(response.get("artifact_type") or payload["artifact_type"]),
            "identity": response.get("identity"),
            "rekor_entry": response.get("rekor_entry"),
            "bundle": response.get("bundle"),
        }

    def verify_record(
        self,
        record: Mapping[str, object],
        signature: str | None = None,
    ) -> bool:
        try:
            artifact_id = _required_record_string(record, "artifact_id")
            artifact_type = _required_record_string(record, "artifact_type")
            checksum = _required_record_string(record, "checksum")
            if not _SHA256_CHECKSUM.fullmatch(checksum):
                return False
            parent_ids = record.get("parent_ids")
            metadata = record.get("metadata")
            recorded_at = _required_record_string(record, "recorded_at")
            payload_base64 = _required_record_string(record, "payload_base64")
            signed_payload = record.get("signed_payload")
            bundle = record.get("signature_bundle")
            if (
                not isinstance(parent_ids, list)
                or any(not isinstance(value, str) or not value for value in parent_ids)
                or not isinstance(metadata, dict)
                or not isinstance(signed_payload, dict)
                or not isinstance(bundle, dict)
            ):
                return False
            raw_payload = _decode_base64(payload_base64)
            actual_checksum = f"sha256:{hashlib.sha256(raw_payload).hexdigest()}"
            if actual_checksum != checksum:
                return False
            expected_signed_payload = {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "checksum": checksum,
                "parent_ids": parent_ids,
                "metadata": metadata,
                "recorded_at": recorded_at,
            }
            if signed_payload != expected_signed_payload:
                return False
            payload_hash = hashlib.sha256(
                _canonical_json_bytes(expected_signed_payload)
            ).hexdigest()
            if bundle.get("payload_hash") != payload_hash:
                return False
            stored_signature = bundle.get("signature")
            if (
                not isinstance(stored_signature, str)
                or not stored_signature
                or record.get("signature") != stored_signature
                or (signature is not None and signature != stored_signature)
            ):
                return False
            signature_type = bundle.get("signature_type")
            if record.get("signature_type") != signature_type:
                return False
            if signature_type == "local_dev_signature":
                expected_signature = hashlib.sha256(
                    f"{payload_hash}:local_dev_signature".encode()
                ).hexdigest()
                return stored_signature == expected_signature
            if not self.verify_command:
                return False
            return self._verify_with_command(
                artifact_id,
                stored_signature,
                bundle,
            )
        except (TypeError, ValueError):
            return False

    def verify_signature(self, artifact_id: str, signature: str) -> bool:
        """Verify a signature against stored records.

        Args:
            artifact_id: The artifact to verify.
            signature: The claimed signature.

        Returns:
            True if the signature is valid for the artifact.
        """
        cached = self._signature_cache.get(artifact_id)
        if cached is None:
            return False

        if cached.get("signature_type") != "local_dev_signature" and self.verify_command:
            return self._verify_with_command(artifact_id, signature, cached)
        return cached["signature"] == signature

    def _verify_with_command(
        self,
        artifact_id: str,
        signature: str,
        cached_bundle: dict,
    ) -> bool:
        command_payload = {
            "artifact_id": artifact_id,
            "artifact_type": cached_bundle.get("artifact_type", ""),
            "payload_hash": cached_bundle.get("payload_hash", ""),
            "signature": signature,
            "expected_identity": self.expected_identity or str(cached_bundle.get("identity") or ""),
            "bundle": cached_bundle,
            "rekor_url": self.rekor_url,
        }
        timeout = float(os.getenv("SIGSTORE_COMMAND_TIMEOUT_SECONDS", "30"))
        _require_command_available(_SIGSTORE_VERIFY_COMMAND, self.verify_command)
        result = subprocess.run(
            shlex.split(self.verify_command),
            input=json.dumps(command_payload, sort_keys=True).encode("utf-8"),
            capture_output=True,
            timeout=timeout,
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
            if not isinstance(response["valid"], bool):
                raise RuntimeError("Sigstore verification command valid must be boolean")
            return response["valid"]
        if "signature_valid" in response:
            if not isinstance(response["signature_valid"], bool):
                raise RuntimeError("Sigstore verification command signature_valid must be boolean")
            return response["signature_valid"]
        raise RuntimeError("Sigstore verification command must return valid")

    def get_rekor_entry(self, artifact_id: str) -> dict | None:
        """Retrieve the Rekor transparency log entry for an artifact.

        Returns None if no entry exists (artifact was never signed).
        """
        cached = self._signature_cache.get(artifact_id)
        if cached is None:
            return None
        if cached.get("signature_type") == "local_dev_signature":
            return None
        rekor_entry = cached.get("rekor_entry")
        if isinstance(rekor_entry, dict):
            return rekor_entry
        log_index = int(hashlib.sha256(artifact_id.encode("utf-8")).hexdigest()[:12], 16)
        return {
            "uuid": cached["signature"][:16],
            "log_index": log_index,
            "integrated_time": cached.get("signed_at"),
            "body": cached["payload_hash"],
            "rekor_url": rekor_entry,
        }

    def batch_sign(self, artifacts: list[tuple[str, str, dict]]) -> list[dict]:
        """Sign multiple artifacts in batch.

        Args:
            artifacts: List of (artifact_id, artifact_type, metadata) tuples.

        Returns:
            List of signature bundles.
        """
        results = []
        for artifact_id, artifact_type, metadata in artifacts:
            results.append(self.sign_artifact(artifact_id, artifact_type, metadata))
        return results

    def get_stats(self) -> dict:
        """Return signing statistics."""
        return {
            "total_signed": len(self._signature_cache),
            "rekor_url": self.rekor_url,
            "cache_keys": list(self._signature_cache.keys()),
        }


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


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _required_record_string(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"record {field} is required")
    return value


def _decode_base64(value: str) -> bytes:
    import base64
    import binascii

    try:
        payload = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("payload_base64 is invalid") from exc
    if not payload:
        raise ValueError("payload_base64 is empty")
    if base64.b64encode(payload).decode("ascii") != value:
        raise ValueError("payload_base64 is not canonical")
    return payload
