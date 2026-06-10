"""Unit tests for provenance service models and signer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from provenance_svc.models import ProvenanceEdge, ProvenanceNode
from provenance_svc.signer import sign, verify
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]


def _load_cosign_audit_wrapper():
    path = ROOT / "tools/sigstore/cosign_audit_wrapper.py"
    spec = importlib.util.spec_from_file_location("cosign_audit_wrapper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cosign_audit_wrapper.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

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
    with pytest.raises(ValidationError):
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
    with pytest.raises(ValidationError):
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


def test_sigstore_sign_command_preflight_rejects_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
):
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", "missing-sigstore-sign --json")
    signer = SigstoreIntegration()

    with pytest.raises(RuntimeError, match="not found"):
        signer.sign_artifact("artifact-missing-sign", "molecule", {"smiles": "CCO"})


def test_sigstore_verify_command_preflight_rejects_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
):
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", "missing-sigstore-verify --json")
    signer = SigstoreIntegration()
    signer._signature_cache["artifact-missing-verify"] = {
        "artifact_type": "molecule",
        "payload_hash": "payload-hash",
        "signature": "sigstore-signature",
        "signature_type": "sigstore_rekor",
    }

    with pytest.raises(RuntimeError, match="not found"):
        signer.verify_signature("artifact-missing-verify", "sigstore-signature")


def test_sigstore_command_returns_rekor_bundle(monkeypatch: pytest.MonkeyPatch):
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    command = (
        f"{sys.executable} -c "
        "\"import json,sys; "
        "payload=json.load(sys.stdin); "
        "print(json.dumps({"
        "'signature_type':'sigstore_rekor',"
        "'signature':'sigstore-signature',"
        "'certificate':'fulcio-cert',"
        "'payload_hash':payload['payload_hash'],"
        "'rekor_entry':{'uuid':'rekor-uuid','log_index':42},"
        "'bundle':{'mediaType':'application/vnd.dev.sigstore.bundle+json'}"
        "}))\""
    )
    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", command)
    signer = SigstoreIntegration()

    bundle = signer.sign_artifact("artifact-2", "molecule", {"smiles": "CCN"})

    assert bundle["signature_type"] == "sigstore_rekor"
    assert bundle["signature"] == "sigstore-signature"
    assert bundle["certificate"] == "fulcio-cert"
    assert bundle["rekor_entry"]["uuid"] == "rekor-uuid"
    assert signer.get_rekor_entry("artifact-2")["uuid"] == "rekor-uuid"
    assert signer.verify_signature("artifact-2", "sigstore-signature") is True


def test_sigstore_sign_command_receives_identity_token(
    monkeypatch: pytest.MonkeyPatch,
):
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    command = (
        f"{sys.executable} -c "
        "\"import json,sys; "
        "payload=json.load(sys.stdin); "
        "assert payload['artifact_id']=='artifact-identity'; "
        "assert payload['artifact_type']=='molecule'; "
        "assert payload['identity_token']=='oidc-token'; "
        "print(json.dumps({"
        "'signature_type':'sigstore_rekor',"
        "'signature':'identity-signature',"
        "'payload_hash':payload['payload_hash']"
        "}))\""
    )
    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", command)
    monkeypatch.setenv("SIGSTORE_IDENTITY_TOKEN", "oidc-token")
    signer = SigstoreIntegration()

    bundle = signer.sign_artifact("artifact-identity", "molecule", {"smiles": "CCO"})

    assert bundle["signature"] == "identity-signature"


def test_sigstore_verify_command_controls_rekor_validation(
    monkeypatch: pytest.MonkeyPatch,
):
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    sign_command = (
        f"{sys.executable} -c "
        "\"import json,sys; "
        "payload=json.load(sys.stdin); "
        "print(json.dumps({"
        "'signature_type':'sigstore_rekor',"
        "'signature':'sigstore-signature',"
        "'payload_hash':payload['payload_hash'],"
        "'rekor_entry':{'uuid':'rekor-uuid'}"
        "}))\""
    )
    verify_command = (
        f"{sys.executable} -c "
        "\"import json,sys; "
        "json.load(sys.stdin); "
        "print(json.dumps({'valid': False}))\""
    )
    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", sign_command)
    monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", verify_command)
    signer = SigstoreIntegration()

    signer.sign_artifact("artifact-3", "molecule", {"smiles": "CCC"})

    assert signer.verify_signature("artifact-3", "sigstore-signature") is False


def test_sigstore_verify_command_receives_artifact_identity_context(
    monkeypatch: pytest.MonkeyPatch,
):
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    sign_command = (
        f"{sys.executable} -c "
        "\"import json,sys; "
        "payload=json.load(sys.stdin); "
        "print(json.dumps({"
        "'signature_type':'sigstore_rekor',"
        "'signature':'sigstore-signature',"
        "'payload_hash':payload['payload_hash'],"
        "'rekor_entry':{'uuid':'rekor-uuid'}"
        "}))\""
    )
    verify_command = (
        f"{sys.executable} -c "
        "\"import json,sys; "
        "payload=json.load(sys.stdin); "
        "assert payload['artifact_type']=='molecule'; "
        "assert payload['payload_hash']; "
        "assert payload['expected_identity']=='fulcio@example.com'; "
        "print(json.dumps({'valid': True}))\""
    )
    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", sign_command)
    monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", verify_command)
    monkeypatch.setenv("SIGSTORE_EXPECTED_IDENTITY", "fulcio@example.com")
    signer = SigstoreIntegration()

    signer.sign_artifact("artifact-identity-verify", "molecule", {"smiles": "CCO"})

    assert signer.verify_signature("artifact-identity-verify", "sigstore-signature") is True


def test_sigstore_commands_use_configured_rekor_url(
    monkeypatch: pytest.MonkeyPatch,
):
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    sign_command = (
        f"{sys.executable} -c "
        "\"import json,sys; "
        "payload=json.load(sys.stdin); "
        "assert payload['rekor_url']=='https://rekor.example'; "
        "print(json.dumps({"
        "'signature_type':'sigstore_rekor',"
        "'signature':'sigstore-signature',"
        "'payload_hash':payload['payload_hash'],"
        "'rekor_entry':{'uuid':'rekor-uuid'}"
        "}))\""
    )
    verify_command = (
        f"{sys.executable} -c "
        "\"import json,sys; "
        "payload=json.load(sys.stdin); "
        "assert payload['rekor_url']=='https://rekor.example'; "
        "print(json.dumps({'valid': True}))\""
    )
    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", sign_command)
    monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", verify_command)
    monkeypatch.setenv("SIGSTORE_REKOR_URL", "https://rekor.example")
    signer = SigstoreIntegration()

    signer.sign_artifact("artifact-rekor-url", "molecule", {"smiles": "CCO"})

    assert signer.verify_signature("artifact-rekor-url", "sigstore-signature") is True


def test_cosign_wrapper_refreshes_github_actions_oidc_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    wrapper = _load_cosign_audit_wrapper()
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"value":"fresh-oidc-token"}'

    def fake_urlopen(request, timeout):
        captured["oidc_url"] = request.full_url
        captured["oidc_auth"] = request.headers.get("Authorization")
        captured["oidc_timeout"] = timeout
        return _Response()

    def fake_run(command, capture_output, check, text):
        captured["cosign_command"] = command
        token_path = Path(command[command.index("--identity-token") + 1])
        captured["cosign_token"] = token_path.read_text(encoding="utf-8")
        bundle_path = Path(command[command.index("--bundle") + 1])
        bundle_path.write_text(
            '{"messageSignature":{"signature":"bundle-signature"},'
            '"verificationMaterial":{"tlogEntries":[{"uuid":"rekor-uuid"}]}}',
            encoding="utf-8",
        )
        return type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "stdout-signature\n", "stderr": ""},
        )()

    monkeypatch.setenv(
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "https://token.actions.githubusercontent.com?existing=1",
    )
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "request-token")
    monkeypatch.setenv("SIGSTORE_OIDC_REQUEST_TIMEOUT_SECONDS", "4")
    monkeypatch.setenv("COSIGN_BINARY", "cosign")
    monkeypatch.setattr(wrapper, "urlopen", fake_urlopen)
    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)

    bundle = wrapper.sign(
        {
            "artifact_type": "audit",
            "identity_token": "stale-oidc-token",
            "payload_hash": "payload-hash",
            "rekor_url": "https://rekor.sigstore.dev",
        }
    )

    assert captured["oidc_url"].endswith("existing=1&audience=sigstore")
    assert captured["oidc_auth"] == "bearer request-token"
    assert captured["oidc_timeout"] == 4
    assert captured["cosign_token"] == "fresh-oidc-token"
    assert bundle["signature"] == "stdout-signature"
    assert bundle["rekor_entry"]["uuid"] == "rekor-uuid"


def test_sigstore_provenance_deployment_wires_production_env() -> None:
    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(
        encoding="utf-8"
    )
    helm_template = (
        ROOT / "infra/helm/moleculeforge/templates/services.yaml"
    ).read_text(encoding="utf-8")

    for env_name in (
        "SIGSTORE_SIGN_COMMAND",
        "SIGSTORE_VERIFY_COMMAND",
        "SIGSTORE_IDENTITY_TOKEN",
        "SIGSTORE_EXPECTED_IDENTITY",
        "SIGSTORE_REKOR_URL",
        "SIGSTORE_COMMAND_TIMEOUT_SECONDS",
    ):
        assert env_name in compose
        assert env_name in k8s
        assert env_name in helm_values

    assert "SIGSTORE_REKOR_URL: ${SIGSTORE_REKOR_URL:-https://rekor.sigstore.dev}" in compose
    assert "SIGSTORE_COMMAND_TIMEOUT_SECONDS: ${SIGSTORE_COMMAND_TIMEOUT_SECONDS:-30}" in compose
    assert "name: sigstore-provenance" in k8s
    assert "secretKeyRef:" in k8s
    assert "configMapKeyRef:" in k8s
    assert "envValueFrom:" in helm_values
    assert "$service.envValueFrom" in helm_template
    assert "valueFrom:" in helm_template
