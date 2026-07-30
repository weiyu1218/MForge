#!/usr/bin/env python3
"""Cosign bridge for MoleculeForge Sigstore/Rekor audit commands."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

DEFAULT_OIDC_ISSUER = "https://token.actions.githubusercontent.com"
DEFAULT_REKOR_URL = "https://rekor.sigstore.dev"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sign", "verify"))
    args = parser.parse_args()

    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError("stdin must contain a JSON object")
        response = sign(request) if args.mode == "sign" else verify(request)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(response, sort_keys=True))
    return 0


def sign(request: dict[str, Any]) -> dict[str, Any]:
    payload_hash = _required_text(request, "payload_hash")
    identity_token = _identity_token(request)
    rekor_url = _text(request.get("rekor_url")) or DEFAULT_REKOR_URL
    cosign = _cosign_binary()

    with tempfile.TemporaryDirectory(prefix="mf-sigstore-") as tmp:
        tmpdir = Path(tmp)
        blob_path = tmpdir / "payload_hash.txt"
        token_path = tmpdir / "identity_token.txt"
        bundle_path = tmpdir / "bundle.json"
        blob_path.write_text(payload_hash, encoding="utf-8")
        token_path.write_text(identity_token, encoding="utf-8")
        token_path.chmod(0o600)

        result = subprocess.run(
            [
                cosign,
                "sign-blob",
                "--yes",
                "--identity-token",
                str(token_path),
                "--rekor-url",
                rekor_url,
                "--bundle",
                str(bundle_path),
                str(blob_path),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(_stderr("cosign sign-blob failed", result))
        signature = result.stdout.strip()
        bundle = _read_json_file(bundle_path)

    if not signature:
        signature = _find_bundle_signature(bundle)
    if not signature:
        raise RuntimeError("cosign sign-blob did not return a signature")

    return {
        "signature_type": "sigstore_rekor",
        "signature": signature,
        "payload_hash": payload_hash,
        "artifact_type": _text(request.get("artifact_type")),
        "identity": _text(request.get("identity")),
        "rekor_entry": _first_tlog_entry(bundle),
        "bundle": bundle,
    }


def verify(request: dict[str, Any]) -> dict[str, bool]:
    payload_hash = _required_text(request, "payload_hash")
    expected_identity = _required_text(request, "expected_identity")
    rekor_url = _text(request.get("rekor_url")) or DEFAULT_REKOR_URL
    oidc_issuer = os.environ.get("SIGSTORE_OIDC_ISSUER", DEFAULT_OIDC_ISSUER)
    bundle = _extract_bundle(request.get("bundle"))
    cosign = _cosign_binary()

    with tempfile.TemporaryDirectory(prefix="mf-sigstore-") as tmp:
        tmpdir = Path(tmp)
        blob_path = tmpdir / "payload_hash.txt"
        bundle_path = tmpdir / "bundle.json"
        blob_path.write_text(payload_hash, encoding="utf-8")
        bundle_path.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")

        result = subprocess.run(
            [
                cosign,
                "verify-blob",
                "--bundle",
                str(bundle_path),
                "--certificate-identity",
                expected_identity,
                "--certificate-oidc-issuer",
                oidc_issuer,
                "--rekor-url",
                rekor_url,
                str(blob_path),
            ],
            capture_output=True,
            check=False,
            text=True,
        )

    if result.returncode == 0:
        return {"valid": True}
    if os.environ.get("SIGSTORE_VERIFY_STRICT", "1") == "1":
        raise RuntimeError(_stderr("cosign verify-blob failed", result))
    return {"valid": False}


def _required_text(request: dict[str, Any], key: str) -> str:
    value = _text(request.get(key))
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _cosign_binary() -> str:
    return os.environ.get("COSIGN_BINARY", "cosign").strip() or "cosign"


def _identity_token(request: dict[str, Any]) -> str:
    github_token = _github_actions_identity_token()
    if github_token:
        return github_token
    token_file = os.environ.get("SIGSTORE_IDENTITY_TOKEN_FILE", "").strip()
    if token_file:
        token_path = Path(token_file)
        if not token_path.is_file():
            raise RuntimeError(f"Sigstore identity token file does not exist: {token_path}")
        file_token = token_path.read_text(encoding="utf-8").strip()
        if not file_token:
            raise RuntimeError("Sigstore identity token file is empty")
        return file_token
    return _required_text(request, "identity_token")


def _github_actions_identity_token() -> str:
    request_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL", "").strip()
    request_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "").strip()
    if not request_url or not request_token:
        return ""

    timeout = float(os.environ.get("SIGSTORE_OIDC_REQUEST_TIMEOUT_SECONDS", "10"))
    request = Request(
        _with_audience(request_url, "sigstore"),
        headers={"Authorization": f"bearer {request_token}"},
    )
    with urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    value = _text(data.get("value")) if isinstance(data, dict) else ""
    if not value:
        raise RuntimeError("GitHub OIDC token response did not contain .value")
    return value


def _with_audience(url: str, audience: str) -> str:
    parsed = urlsplit(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "audience"
    ]
    query.append(("audience", audience))
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
    )


def _read_json_file(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"cosign did not create bundle file: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _extract_bundle(value: Any) -> Any:
    if isinstance(value, dict) and isinstance(value.get("bundle"), dict):
        return value["bundle"]
    if isinstance(value, dict):
        return value
    raise ValueError("bundle is required")


def _first_tlog_entry(bundle: Any) -> dict[str, Any] | None:
    if not isinstance(bundle, dict):
        return None
    verification_material = bundle.get("verificationMaterial")
    if not isinstance(verification_material, dict):
        return None
    entries = verification_material.get("tlogEntries")
    if isinstance(entries, list) and entries and isinstance(entries[0], dict):
        return entries[0]
    return None


def _find_bundle_signature(bundle: Any) -> str:
    if not isinstance(bundle, dict):
        return ""
    message_signature = bundle.get("messageSignature")
    if isinstance(message_signature, dict):
        signature = _text(message_signature.get("signature"))
        if signature:
            return signature
    return ""


def _stderr(prefix: str, result: subprocess.CompletedProcess[str]) -> str:
    stderr = result.stderr.strip()
    stdout = result.stdout.strip()
    if stderr:
        return f"{prefix}: {stderr}"
    if stdout:
        return f"{prefix}: {stdout}"
    return prefix


if __name__ == "__main__":
    raise SystemExit(main())
