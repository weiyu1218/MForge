"""Provenance signer — deterministic SHA-256 signing for audit trail."""
from __future__ import annotations

import hashlib
import json


def sign(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def verify(payload: dict, signature: str) -> bool:
    return sign(payload) == signature
