"""Unit tests for provenance service models and signer."""

from __future__ import annotations

import pytest
from provenance_svc.models import ProvenanceEdge, ProvenanceNode
from provenance_svc.signer import sign, verify

# ── Test: ProvenanceNode ─────────────────────────────────────────────────────

def test_provenance_node_creation():
    node = ProvenanceNode(
        node_id="test-node-1",
        node_type="NL_Input",
        run_id="run-001",
        trace_id="trace-001",
        content_hash="abc123",
        payload={"text": "Design EGFR inhibitor"},
    )
    assert node.node_id == "test-node-1"
    assert node.node_type == "NL_Input"
    assert node.signature is None
    assert node.payload["text"] == "Design EGFR inhibitor"


def test_provenance_node_extra_forbid():
    with pytest.raises(Exception):
        ProvenanceNode(
            node_id="t1", node_type="CIG", run_id="r1",
            trace_id="tr1", content_hash="h", bad_field="x",
        )


# ── Test: ProvenanceEdge ─────────────────────────────────────────────────────

def test_provenance_edge_creation():
    edge = ProvenanceEdge(
        from_node_id="nl-1",
        to_node_id="cig-1",
        relation="COMPILED_BY",
        agent="NL2Obj",
        metadata={"seed": 42},
    )
    assert edge.relation == "COMPILED_BY"
    assert edge.agent == "NL2Obj"
    assert edge.metadata["seed"] == 42
    assert edge.signature is None


def test_provenance_edge_extra_forbid():
    with pytest.raises(Exception):
        ProvenanceEdge(
            from_node_id="a", to_node_id="b", relation="X",
            agent="test", bad_field="x",
        )


# ── Test: signer ─────────────────────────────────────────────────────────────

def test_sign_returns_string():
    sig = sign({"key": "value"})
    assert isinstance(sig, str)
    assert len(sig) == 64  # SHA-256 hex


def test_sign_deterministic():
    payload = {"key": "value", "num": 42}
    sig1 = sign(payload)
    sig2 = sign(payload)
    assert sig1 == sig2


def test_sign_different_payloads():
    sig1 = sign({"key": "value1"})
    sig2 = sign({"key": "value2"})
    assert sig1 != sig2


def test_verify_valid():
    payload = {"data": "test"}
    sig = sign(payload)
    assert verify(payload, sig) is True


def test_verify_invalid():
    payload = {"data": "test"}
    sig = sign(payload)
    assert verify({"data": "tampered"}, sig) is False


# ── Test: ProvenanceNode with signature ──────────────────────────────────────

def test_node_with_signature():
    payload = {"text": "test input"}
    sig = sign({"node_id": "n1", "content_hash": "h1", "payload": payload})
    node = ProvenanceNode(
        node_id="n1", node_type="NL_Input", run_id="r1",
        trace_id="tr1", content_hash="h1", signature=sig, payload=payload,
    )
    assert node.signature is not None
    assert len(node.signature) == 64


def test_sigstore_unavailable_uses_explicit_local_dev_signature():
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    signer = SigstoreIntegration()
    bundle = signer.sign_artifact("artifact-1", "molecule", {"smiles": "CCO"})

    assert bundle["signature_type"] == "local_dev_signature"
    assert bundle["certificate"] is None
    assert bundle["rekor_entry"] is None
    assert signer.get_rekor_entry("artifact-1") is None
    assert signer.verify_signature("artifact-1", bundle["signature"]) is True
