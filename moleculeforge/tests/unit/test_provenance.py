"""Unit tests for provenance service models and signer."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
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
            node_id="t1",
            node_type="CIG",
            run_id="r1",
            trace_id="tr1",
            content_hash="h",
            bad_field="x",
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
            from_node_id="a",
            to_node_id="b",
            relation="X",
            agent="test",
            bad_field="x",
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


def test_provenance_record_requires_strict_non_empty_base64_payload():
    from provenance_svc.main import ProvenanceRecord

    with pytest.raises(ValidationError, match="payload_base64"):
        ProvenanceRecord(
            artifact_type="workflow_state",
            artifact_id="artifact-invalid",
            payload_base64="not-base64",
        )

    with pytest.raises(ValidationError, match="payload_base64"):
        ProvenanceRecord(
            artifact_type="workflow_state",
            artifact_id="artifact-empty",
            payload_base64="",
        )


@pytest.mark.asyncio
async def test_provenance_routes_require_configured_internal_service_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx
    from provenance_svc import main as provenance

    store = provenance.InMemoryProvenanceStore()
    monkeypatch.setattr(provenance.rest_app.state, "provenance_store", store, raising=False)
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "service-secret")
    transport = httpx.ASGITransport(app=provenance.rest_app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://provenance.test",
    ) as client:
        health = await client.get("/health")
        missing = [
            await client.post("/v1/provenance/record", json={}),
            await client.get("/v1/provenance/record/missing"),
            await client.get("/v1/provenance/missing"),
            await client.post("/v1/provenance/verify", json={}),
            await client.get("/v1/provenance/audit/project-1"),
        ]
        wrong = await client.get(
            "/v1/provenance/missing",
            headers={"X-MoleculeForge-Service-Token": "wrong"},
        )
        accepted = await client.get(
            "/v1/provenance/missing",
            headers={"X-MoleculeForge-Service-Token": "service-secret"},
        )

    assert health.status_code == 200
    assert {response.status_code for response in missing} == {401}
    assert wrong.status_code == 401
    assert accepted.status_code == 404


@pytest.mark.asyncio
async def test_health_initializes_configured_production_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from provenance_svc import main as provenance

    class Graph:
        async def count_artifacts(self) -> int:
            return 0

    class Audit:
        def __init__(self) -> None:
            self.schema_calls = 0

        async def _ensure_schema(self) -> None:
            self.schema_calls += 1

    class Objects:
        def __init__(self) -> None:
            self.bucket_calls = 0

        async def ensure_bucket(self) -> None:
            self.bucket_calls += 1

    audit = Audit()
    objects = Objects()
    store = provenance.ProductionProvenanceStore(Graph(), audit, objects)
    monkeypatch.setattr(provenance.rest_app.state, "provenance_store", store, raising=False)

    response = await provenance.health()

    assert response["status"] == "healthy"
    assert audit.schema_calls == 1
    assert objects.bucket_calls == 1


def test_provenance_deployment_exposes_only_registered_http_transport():
    import yaml
    from provenance_svc import main as provenance

    compose = yaml.safe_load(
        (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    )
    kubernetes = list(
        yaml.safe_load_all(
            (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    helm = yaml.safe_load(
        (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")
    )
    deployment = next(
        document
        for document in kubernetes
        if document
        and document.get("kind") == "Deployment"
        and document["metadata"]["name"] == "provenance-svc"
    )
    service = next(
        document
        for document in kubernetes
        if document
        and document.get("kind") == "Service"
        and document["metadata"]["name"] == "provenance-svc"
    )

    assert compose["services"]["provenance-svc"]["ports"] == ["8010:8010"]
    assert deployment["spec"]["template"]["spec"]["containers"][0]["ports"] == [
        {"name": "http", "containerPort": 8010}
    ]
    assert service["spec"]["ports"] == [{"name": "http", "port": 8010, "targetPort": 8010}]
    assert helm["services"]["provenance-svc"]["ports"] == [{"name": "http", "port": 8010}]
    assert not hasattr(provenance, "ProvenanceServicer")
    assert not hasattr(provenance, "serve_grpc")


def test_provenance_and_orchestrator_deployments_wire_persistent_storage():
    import yaml

    compose = yaml.safe_load(
        (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    )
    kubernetes = list(
        yaml.safe_load_all(
            (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    helm = yaml.safe_load(
        (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")
    )

    compose_orchestrator = compose["services"]["orchestrator-svc"]
    compose_provenance = compose["services"]["provenance-svc"]
    assert compose_orchestrator["environment"]["PROVENANCE_SVC_URL"] == (
        "${PROVENANCE_SVC_URL:-http://provenance-svc:8010}"
    )
    assert compose_orchestrator["environment"]["INTERNAL_SERVICE_TOKEN"] == (
        "${INTERNAL_SERVICE_TOKEN:-mf_dev_internal_service_token}"
    )
    assert compose_provenance["environment"]["INTERNAL_SERVICE_TOKEN"] == (
        "${INTERNAL_SERVICE_TOKEN:-mf_dev_internal_service_token}"
    )
    assert compose_orchestrator["environment"]["MF_DB_PATH"] == (
        "/var/lib/moleculeforge/moleculeforge.db"
    )
    assert "orchestrator_state_data:/var/lib/moleculeforge" in (compose_orchestrator["volumes"])
    assert compose_provenance["environment"]["PROVENANCE_STORE_MODE"] == (
        "${PROVENANCE_STORE_MODE:-local_demo}"
    )
    for name in (
        "PROVENANCE_DATABASE_URL",
        "MINIO_ENDPOINT_URL",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MINIO_BUCKET",
    ):
        assert name in compose_provenance["environment"]

    resources = {
        (resource["kind"], resource["metadata"]["name"]): resource
        for resource in kubernetes
        if resource
    }
    orchestrator = resources[("Deployment", "orchestrator-svc")]
    provenance = resources[("Deployment", "provenance-svc")]
    orchestrator_container = orchestrator["spec"]["template"]["spec"]["containers"][0]
    provenance_container = provenance["spec"]["template"]["spec"]["containers"][0]
    orchestrator_env = {item["name"]: item for item in orchestrator_container["env"]}
    provenance_env = {item["name"]: item for item in provenance_container["env"]}

    assert orchestrator_env["PROVENANCE_SVC_URL"]["value"] == (
        "http://provenance-svc.mf-agents.svc.cluster.local:8010"
    )
    assert orchestrator_env["PROVENANCE_REQUIRED_SIGNATURE_TYPE"]["value"] == ("sigstore_rekor")
    assert orchestrator_env["MF_DB_PATH"]["value"] == ("/var/lib/moleculeforge/moleculeforge.db")
    assert provenance_env["INTERNAL_SERVICE_TOKEN"]["valueFrom"]["secretKeyRef"] == {
        "name": "agent-runtime-secrets",
        "key": "INTERNAL_SERVICE_TOKEN",
    }
    assert orchestrator_container["volumeMounts"] == [
        {"name": "orchestrator-state", "mountPath": "/var/lib/moleculeforge"}
    ]
    assert orchestrator["spec"]["template"]["spec"]["volumes"] == [
        {
            "name": "orchestrator-state",
            "persistentVolumeClaim": {"claimName": "orchestrator-state"},
        }
    ]
    assert resources[("PersistentVolumeClaim", "orchestrator-state")]["spec"]["accessModes"] == [
        "ReadWriteOnce"
    ]

    assert provenance_env["PROVENANCE_STORE_MODE"]["value"] == "production_real"
    for name in (
        "PROVENANCE_DATABASE_URL",
        "MINIO_ENDPOINT_URL",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MINIO_BUCKET",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
    ):
        assert "valueFrom" in provenance_env[name]

    helm_orchestrator = helm["services"]["orchestrator-svc"]
    helm_provenance = helm["services"]["provenance-svc"]
    expected_orchestrator_env = {
        "HUMU_ENCODER_TARGET": "humu-encoder-svc.mf-generators.svc.cluster.local:50051",
        "MF_DB_PATH": "/var/lib/moleculeforge/moleculeforge.db",
        "PROVENANCE_REQUIRED_SIGNATURE_TYPE": "sigstore_rekor",
        "PROVENANCE_SVC_URL": "http://provenance-svc.mf-agents.svc.cluster.local:8010",
    }
    assert {
        name: helm_orchestrator["env"][name]
        for name in expected_orchestrator_env
    } == expected_orchestrator_env
    assert helm_orchestrator["volumeMounts"] == [
        {"name": "orchestrator-state", "mountPath": "/var/lib/moleculeforge"}
    ]
    assert helm_orchestrator["volumes"] == [
        {
            "name": "orchestrator-state",
            "persistentVolumeClaim": {"claimName": "orchestrator-state"},
        }
    ]
    assert helm["persistentVolumeClaims"]["orchestrator-state"]["storage"] == "1Gi"
    assert helm_provenance["env"]["PROVENANCE_STORE_MODE"] == "production_real"
    assert "INTERNAL_SERVICE_TOKEN" in helm_orchestrator["envValueFrom"]
    assert "INTERNAL_SERVICE_TOKEN" in helm_provenance["envValueFrom"]
    for name in (
        "PROVENANCE_DATABASE_URL",
        "MINIO_ENDPOINT_URL",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MINIO_BUCKET",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
    ):
        assert name in helm_provenance["envValueFrom"]


@pytest.mark.asyncio
async def test_provenance_signature_binds_raw_bytes_and_verifies_after_restart(
    monkeypatch: pytest.MonkeyPatch,
):
    from provenance_svc import main as provenance
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    raw_payload = json.dumps(
        {"candidate_id": "candidate-1", "status": "completed"},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    store = provenance.InMemoryProvenanceStore()
    monkeypatch.setattr(provenance.rest_app.state, "provenance_store", store, raising=False)
    monkeypatch.setattr(provenance, "sigstore", SigstoreIntegration())

    await provenance.create_record(
        provenance.ProvenanceRecord(
            artifact_type="nl_query",
            artifact_id="artifact-parent",
            metadata={"project_id": "project-1", "run_id": "run-1"},
            payload_base64=base64.b64encode(b"source input").decode("ascii"),
        )
    )
    created = await provenance.create_record(
        provenance.ProvenanceRecord(
            artifact_type="workflow_state",
            artifact_id="artifact-restart",
            parent_ids=["artifact-parent"],
            metadata={"project_id": "project-1", "run_id": "run-1"},
            payload_base64=base64.b64encode(raw_payload).decode("ascii"),
        )
    )
    stored = store.records["artifact-restart"]

    assert stored["checksum"] == f"sha256:{hashlib.sha256(raw_payload).hexdigest()}"
    assert base64.b64decode(stored["payload_base64"], validate=True) == raw_payload
    assert stored["signed_payload"] == {
        "artifact_id": "artifact-restart",
        "artifact_type": "workflow_state",
        "checksum": stored["checksum"],
        "metadata": {"project_id": "project-1", "run_id": "run-1"},
        "parent_ids": ["artifact-parent"],
        "recorded_at": stored["recorded_at"],
    }
    assert (
        stored["signature_bundle"]["payload_hash"]
        == hashlib.sha256(
            json.dumps(
                stored["signed_payload"],
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )

    monkeypatch.setattr(provenance, "sigstore", SigstoreIntegration())
    verified = await provenance.verify_provenance(
        provenance.VerifyRequest(
            artifact_id="artifact-restart",
            signature=created["signature"],
        )
    )
    chain = await provenance.get_provenance("artifact-restart")
    audit = await provenance.audit_project("project-1")

    assert verified["signature_valid"] is True
    assert chain["verified"] is True
    assert audit["verified_count"] == 2
    assert all(artifact["verified"] for artifact in audit["artifacts"])

    stored["payload_base64"] = base64.b64encode(b"tampered").decode("ascii")
    tampered = await provenance.verify_provenance(
        provenance.VerifyRequest(
            artifact_id="artifact-restart",
            signature=created["signature"],
        )
    )
    tampered_chain = await provenance.get_provenance("artifact-restart")
    tampered_audit = await provenance.audit_project("project-1")

    assert tampered["signature_valid"] is False
    assert tampered_chain["verified"] is False
    assert tampered_audit["verified_count"] == 1
    assert tampered_audit["unverified_count"] == 1


@pytest.mark.asyncio
async def test_provenance_record_detail_returns_verified_signed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from provenance_svc import main as provenance
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    store = provenance.InMemoryProvenanceStore()
    monkeypatch.setattr(provenance.rest_app.state, "provenance_store", store, raising=False)
    monkeypatch.setattr(provenance, "sigstore", SigstoreIntegration())
    payload = {
        "schema_version": "external_validation_evidence.v1",
        "project_id": "project-evidence",
        "run_id": "run-evidence",
        "candidate_id": "candidate-evidence",
        "canonical_smiles": "CCO",
        "metrics": {"activity": 0.81},
        "uncertainties": {"activity": 0.02},
    }
    created = await provenance.create_record(
        provenance.ProvenanceRecord(
            artifact_type="external_validation_evidence",
            artifact_id="artifact-evidence-detail",
            payload_base64=base64.b64encode(
                json.dumps(
                    payload,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).decode("ascii"),
            metadata={
                "project_id": "project-evidence",
                "run_id": "run-evidence",
                "candidate_id": "candidate-evidence",
                "canonical_smiles": "CCO",
            },
        )
    )

    detail = await provenance.get_provenance_record("artifact-evidence-detail")

    assert detail["artifact_id"] == "artifact-evidence-detail"
    assert detail["artifact_type"] == "external_validation_evidence"
    assert detail["payload_base64"]
    assert detail["signature"] == created["signature"]
    assert detail["verified"] is True


@pytest.mark.asyncio
async def test_provenance_artifact_id_is_immutable_and_identical_retry_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
):
    from fastapi import HTTPException
    from provenance_svc import main as provenance
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    store = provenance.InMemoryProvenanceStore()
    monkeypatch.setattr(provenance.rest_app.state, "provenance_store", store, raising=False)
    monkeypatch.setattr(provenance, "sigstore", SigstoreIntegration())
    first_record = provenance.ProvenanceRecord(
        artifact_type="workflow_state",
        artifact_id="artifact-idempotent",
        payload_base64=base64.b64encode(b"stable payload").decode("ascii"),
        metadata={"project_id": "project-1"},
    )

    first = await provenance.create_record(first_record)
    retry = await provenance.create_record(first_record)

    assert retry == first
    with pytest.raises(HTTPException) as exc:
        await provenance.create_record(
            provenance.ProvenanceRecord(
                artifact_type="workflow_state",
                artifact_id="artifact-idempotent",
                payload_base64=base64.b64encode(b"different payload").decode("ascii"),
                metadata={"project_id": "project-1"},
            )
        )
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_production_provenance_retry_is_idempotent_after_store_restart(
    monkeypatch: pytest.MonkeyPatch,
):
    from fastapi import HTTPException
    from provenance_svc import main as provenance
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    class _Graph:
        def __init__(self) -> None:
            self.artifacts: list[str] = []

        async def write_artifact(self, **kwargs) -> None:
            artifact_id = str(kwargs["artifact_id"])
            if artifact_id not in self.artifacts:
                self.artifacts.append(artifact_id)

        async def write_artifact_parent(self, parent_id: str, child_id: str) -> None:
            return None

    class _Audit:
        def __init__(self) -> None:
            self.events: list[dict] = []

        async def write_event(self, stored: dict) -> None:
            if all(event["artifact_id"] != stored["artifact_id"] for event in self.events):
                self.events.append(stored)

    class _Objects:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}
            self.put_count = 0

        async def object_exists(self, object_name: str) -> bool:
            return object_name in self.objects

        async def get_object(self, object_name: str) -> bytes:
            return self.objects[object_name]

        async def put_object(
            self,
            object_name: str,
            data: bytes,
            content_type: str,
        ) -> None:
            assert content_type == "application/json"
            self.put_count += 1
            self.objects[object_name] = data

        async def put_object_if_absent(
            self,
            object_name: str,
            data: bytes,
            content_type: str,
        ) -> bool:
            if object_name in self.objects:
                return False
            await self.put_object(object_name, data, content_type)
            return True

    graph = _Graph()
    audit = _Audit()
    objects = _Objects()
    signer = SigstoreIntegration()
    raw_payload = b"restart-stable payload"
    record = provenance.ProvenanceRecord(
        artifact_type="workflow_state",
        artifact_id="artifact-production-retry",
        payload_base64=base64.b64encode(raw_payload).decode("ascii"),
        metadata={"project_id": "project-1"},
    )
    first_recorded_at = "2026-07-29T00:00:00Z"
    retry_recorded_at = "2026-07-29T00:01:00Z"
    signed = signer.sign_artifact(
        record.artifact_id,
        record.artifact_type,
        record.metadata,
        checksum=f"sha256:{hashlib.sha256(raw_payload).hexdigest()}",
        parent_ids=record.parent_ids,
        recorded_at=first_recorded_at,
    )
    signed["signature_type"] = "sigstore_rekor"

    class AlwaysValidVerifier:
        def verify_record(self, *args, **kwargs) -> bool:
            return True

    monkeypatch.setattr(provenance, "sigstore", AlwaysValidVerifier())

    first_store = provenance.ProductionProvenanceStore(graph, audit, objects)
    first = await first_store.record(record, signed, first_recorded_at)
    retry_signed = signer.sign_artifact(
        record.artifact_id,
        record.artifact_type,
        record.metadata,
        checksum=f"sha256:{hashlib.sha256(raw_payload).hexdigest()}",
        parent_ids=record.parent_ids,
        recorded_at=retry_recorded_at,
    )
    retry_signed["signature_type"] = "sigstore_rekor"
    restarted_store = provenance.ProductionProvenanceStore(graph, audit, objects)
    retry = await restarted_store.record(record, retry_signed, retry_recorded_at)

    assert retry == first
    assert graph.artifacts == ["artifact-production-retry"]
    assert len(audit.events) == 1
    assert objects.put_count == 1

    changed_payload = b"different payload"
    changed_record = record.model_copy(
        update={
            "payload_base64": base64.b64encode(changed_payload).decode("ascii"),
        }
    )
    changed_signed = signer.sign_artifact(
        changed_record.artifact_id,
        changed_record.artifact_type,
        changed_record.metadata,
        checksum=f"sha256:{hashlib.sha256(changed_payload).hexdigest()}",
        parent_ids=changed_record.parent_ids,
        recorded_at="2026-07-29T00:02:00Z",
    )
    changed_signed["signature_type"] = "sigstore_rekor"
    with pytest.raises(HTTPException) as exc:
        await provenance.ProductionProvenanceStore(graph, audit, objects).record(
            changed_record,
            changed_signed,
            "2026-07-29T00:02:00Z",
        )

    assert exc.value.status_code == 409
    assert graph.artifacts == ["artifact-production-retry"]
    assert len(audit.events) == 1
    assert objects.put_count == 1


@pytest.mark.asyncio
async def test_create_record_checks_existing_content_before_signing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException
    from provenance_svc import main as provenance
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    class CountingSigner:
        def __init__(self) -> None:
            self.delegate = SigstoreIntegration()
            self.calls = 0

        def sign_artifact(self, *args, **kwargs):
            self.calls += 1
            return self.delegate.sign_artifact(*args, **kwargs)

        def verify_record(self, *args, **kwargs):
            return self.delegate.verify_record(*args, **kwargs)

    store = provenance.InMemoryProvenanceStore()
    signer = CountingSigner()
    monkeypatch.setattr(provenance.rest_app.state, "provenance_store", store, raising=False)
    monkeypatch.setattr(provenance, "sigstore", signer)
    record = provenance.ProvenanceRecord(
        artifact_type="workflow_state",
        artifact_id="artifact-preflight",
        payload_base64=base64.b64encode(b"stable payload").decode("ascii"),
    )

    first = await provenance.create_record(record)
    retry = await provenance.create_record(record)

    assert retry == first
    assert signer.calls == 1

    with pytest.raises(HTTPException) as exc:
        await provenance.create_record(
            record.model_copy(
                update={"payload_base64": base64.b64encode(b"conflicting payload").decode("ascii")}
            )
        )

    assert exc.value.status_code == 409
    assert signer.calls == 1


@pytest.mark.asyncio
async def test_create_record_rejects_unknown_parent_before_signing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException
    from provenance_svc import main as provenance

    class RejectUnexpectedSigning:
        def sign_artifact(self, *args, **kwargs):
            raise AssertionError("unknown parents must be rejected before signing")

    store = provenance.InMemoryProvenanceStore()
    monkeypatch.setattr(provenance.rest_app.state, "provenance_store", store, raising=False)
    monkeypatch.setattr(provenance, "sigstore", RejectUnexpectedSigning())

    with pytest.raises(HTTPException) as exc:
        await provenance.create_record(
            provenance.ProvenanceRecord(
                artifact_type="workflow_state",
                artifact_id="artifact-orphan",
                parent_ids=["artifact-missing"],
                payload_base64=base64.b64encode(b"orphan").decode("ascii"),
            )
        )

    assert exc.value.status_code == 409
    assert store.records == {}


@pytest.mark.asyncio
async def test_provenance_signature_binds_recorded_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException
    from provenance_svc import main as provenance
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    store = provenance.InMemoryProvenanceStore()
    monkeypatch.setattr(provenance.rest_app.state, "provenance_store", store, raising=False)
    monkeypatch.setattr(provenance, "sigstore", SigstoreIntegration())

    await provenance.create_record(
        provenance.ProvenanceRecord(
            artifact_type="workflow_state",
            artifact_id="artifact-recorded-at",
            payload_base64=base64.b64encode(b"signed time").decode("ascii"),
        )
    )
    stored = store.records["artifact-recorded-at"]

    assert stored["signed_payload"]["recorded_at"] == stored["recorded_at"]
    assert await provenance._verify_stored_record(stored) is True

    stored["recorded_at"] = "2030-01-01T00:00:00+00:00"

    assert await provenance._verify_stored_record(stored) is False
    with pytest.raises(HTTPException) as exc:
        await provenance.create_record(
            provenance.ProvenanceRecord(
                artifact_type="workflow_state",
                artifact_id="artifact-recorded-at",
                payload_base64=base64.b64encode(b"signed time").decode("ascii"),
            )
        )
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_production_store_rejects_local_dev_signature() -> None:
    from fastapi import HTTPException
    from provenance_svc import main as provenance
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    class NoSideEffects:
        def __getattr__(self, name):
            async def unexpected(*args, **kwargs):
                raise AssertionError(f"unexpected production side effect: {name}")

            return unexpected

    record = provenance.ProvenanceRecord(
        artifact_type="workflow_state",
        artifact_id="artifact-local-signature",
        payload_base64=base64.b64encode(b"production payload").decode("ascii"),
    )
    recorded_at = "2026-07-30T00:00:00+00:00"
    signed = SigstoreIntegration().sign_artifact(
        record.artifact_id,
        record.artifact_type,
        record.metadata,
        checksum=f"sha256:{hashlib.sha256(b'production payload').hexdigest()}",
        parent_ids=record.parent_ids,
        recorded_at=recorded_at,
    )
    store = provenance.ProductionProvenanceStore(
        NoSideEffects(),
        NoSideEffects(),
        NoSideEffects(),
    )

    with pytest.raises(HTTPException) as exc:
        await store.record(record, signed, recorded_at)

    assert exc.value.status_code == 503
    assert "local_dev_signature" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_production_store_uses_object_conditional_create_as_commit_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from provenance_svc import main as provenance
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    class RecordingGraph:
        def __init__(self) -> None:
            self.calls = 0

        async def write_artifact(self, **kwargs) -> None:
            self.calls += 1

    class RecordingAudit:
        def __init__(self) -> None:
            self.calls = 0

        async def write_event(self, stored: dict) -> None:
            self.calls += 1

    class RacingObjects:
        def __init__(self, existing: dict) -> None:
            self.existing = json.dumps(existing, sort_keys=True).encode("utf-8")
            self.if_absent_calls = 0

        async def object_exists(self, object_name: str) -> bool:
            return False

        async def put_object_if_absent(
            self,
            object_name: str,
            data: bytes,
            content_type: str,
        ) -> bool:
            self.if_absent_calls += 1
            return False

        async def get_object(self, object_name: str) -> bytes:
            return self.existing

    record = provenance.ProvenanceRecord(
        artifact_type="workflow_state",
        artifact_id="artifact-race",
        payload_base64=base64.b64encode(b"race payload").decode("ascii"),
    )
    first_recorded_at = "2026-07-30T00:00:00+00:00"
    retry_recorded_at = "2026-07-30T00:01:00+00:00"
    signer = SigstoreIntegration()
    checksum = f"sha256:{hashlib.sha256(b'race payload').hexdigest()}"
    first_signed = signer.sign_artifact(
        record.artifact_id,
        record.artifact_type,
        record.metadata,
        checksum=checksum,
        parent_ids=record.parent_ids,
        recorded_at=first_recorded_at,
    )
    existing = provenance._stored_record(record, first_signed, first_recorded_at)
    retry_signed = signer.sign_artifact(
        record.artifact_id,
        record.artifact_type,
        record.metadata,
        checksum=checksum,
        parent_ids=record.parent_ids,
        recorded_at=retry_recorded_at,
    )
    first_signed["signature_type"] = "sigstore_rekor"
    first_signed["signature_bundle"] = "rekor-bundle"
    retry_signed["signature_type"] = "sigstore_rekor"
    retry_signed["signature_bundle"] = "rekor-bundle"
    existing = provenance._stored_record(record, first_signed, first_recorded_at)

    class AlwaysValidVerifier:
        def verify_record(self, *args, **kwargs) -> bool:
            return True

    monkeypatch.setattr(provenance, "sigstore", AlwaysValidVerifier())
    graph = RecordingGraph()
    audit = RecordingAudit()
    objects = RacingObjects(existing)
    store = provenance.ProductionProvenanceStore(graph, audit, objects)

    result = await store.record(record, retry_signed, retry_recorded_at)

    assert result == existing
    assert objects.if_absent_calls == 1
    assert graph.calls == 1
    assert audit.calls == 1


@pytest.mark.asyncio
async def test_production_store_retry_repairs_secondary_indexes_after_partial_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from provenance_svc import main as provenance
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    class IdempotentGraph:
        def __init__(self) -> None:
            self.artifacts: set[str] = set()

        async def write_artifact(self, **kwargs) -> None:
            self.artifacts.add(kwargs["artifact_id"])

    class FailingAudit:
        def __init__(self) -> None:
            self.attempts = 0
            self.artifacts: set[str] = set()

        async def write_event(self, stored: dict) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("postgres unavailable")
            self.artifacts.add(stored["artifact_id"])

    class ConditionalObjects:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}

        async def object_exists(self, object_name: str) -> bool:
            return object_name in self.objects

        async def put_object_if_absent(
            self,
            object_name: str,
            data: bytes,
            content_type: str,
        ) -> bool:
            if object_name in self.objects:
                return False
            self.objects[object_name] = data
            return True

        async def get_object(self, object_name: str) -> bytes:
            return self.objects[object_name]

    class AlwaysValidVerifier:
        def verify_record(self, *args, **kwargs) -> bool:
            return True

    monkeypatch.setattr(provenance, "sigstore", AlwaysValidVerifier())
    record = provenance.ProvenanceRecord(
        artifact_type="workflow_state",
        artifact_id="artifact-partial-retry",
        payload_base64=base64.b64encode(b"partial retry").decode("ascii"),
    )
    recorded_at = "2026-07-30T00:00:00+00:00"
    signed = SigstoreIntegration().sign_artifact(
        record.artifact_id,
        record.artifact_type,
        record.metadata,
        checksum=f"sha256:{hashlib.sha256(b'partial retry').hexdigest()}",
        parent_ids=record.parent_ids,
        recorded_at=recorded_at,
    )
    signed["signature_type"] = "sigstore_rekor"
    graph = IdempotentGraph()
    audit = FailingAudit()
    objects = ConditionalObjects()
    store = provenance.ProductionProvenanceStore(graph, audit, objects)
    monkeypatch.setattr(provenance.rest_app.state, "provenance_store", store, raising=False)

    with pytest.raises(RuntimeError, match="postgres unavailable"):
        await store.record(record, signed, recorded_at)

    result = await provenance.create_record(record)

    assert result["artifact_id"] == "artifact-partial-retry"
    assert graph.artifacts == {"artifact-partial-retry"}
    assert audit.artifacts == {"artifact-partial-retry"}
    assert audit.attempts == 2


@pytest.mark.asyncio
async def test_postgres_audit_schema_and_insert_are_artifact_id_idempotent() -> None:
    from provenance_svc import main as provenance

    statements: list[str] = []

    class Connection:
        async def execute(self, statement, parameters=None):
            statements.append(str(statement))

    class BeginContext:
        async def __aenter__(self):
            return Connection()

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    class Engine:
        def begin(self):
            return BeginContext()

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def execute(self, statement, parameters=None):
            statements.append(str(statement))
            return type(
                "Result",
                (),
                {"scalar_one_or_none": lambda self: parameters["payload"]},
            )()

        async def commit(self):
            return None

    writer = object.__new__(provenance.PostgresAuditWriter)
    writer.engine = Engine()
    writer.sessionmaker = Session
    writer._schema_ready = False

    await writer.write_event(
        {
            "artifact_id": "artifact-audit-idempotent",
            "metadata": {},
        }
    )

    sql = "\n".join(statements)
    assert "UNIQUE" in sql
    assert "artifact_id" in sql
    assert "ON CONFLICT (artifact_id) DO UPDATE" in sql
    assert "audit_events.payload = EXCLUDED.payload" in sql


def test_provenance_service_installs_mf_core_db_extra() -> None:
    import tomllib

    config = tomllib.loads(
        (ROOT / "services/provenance-svc/pyproject.toml").read_text(encoding="utf-8")
    )

    assert "mf-core[db]" in config["project"]["dependencies"]


# ── Test: ProvenanceNode with signature ──────────────────────────────────────


def test_node_with_signature():
    payload = {"text": "test input"}
    sig = sign({"node_id": "n1", "content_hash": "h1", "payload": payload})
    node = ProvenanceNode(
        node_id="n1",
        node_type="NL_Input",
        run_id="r1",
        trace_id="tr1",
        content_hash="h1",
        signature=sig,
        payload=payload,
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
        '"import json,sys; '
        "payload=json.load(sys.stdin); "
        "print(json.dumps({"
        "'signature_type':'sigstore_rekor',"
        "'signature':'sigstore-signature',"
        "'certificate':'fulcio-cert',"
        "'payload_hash':payload['payload_hash'],"
        "'rekor_entry':{'uuid':'rekor-uuid','log_index':42},"
        "'bundle':{'mediaType':'application/vnd.dev.sigstore.bundle+json'}"
        '}))"'
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
        '"import json,sys; '
        "payload=json.load(sys.stdin); "
        "assert payload['artifact_id']=='artifact-identity'; "
        "assert payload['artifact_type']=='molecule'; "
        "assert payload['identity_token']=='oidc-token'; "
        "print(json.dumps({"
        "'signature_type':'sigstore_rekor',"
        "'signature':'identity-signature',"
        "'payload_hash':payload['payload_hash']"
        '}))"'
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
        '"import json,sys; '
        "payload=json.load(sys.stdin); "
        "print(json.dumps({"
        "'signature_type':'sigstore_rekor',"
        "'signature':'sigstore-signature',"
        "'payload_hash':payload['payload_hash'],"
        "'rekor_entry':{'uuid':'rekor-uuid'}"
        '}))"'
    )
    verify_command = (
        f"{sys.executable} -c "
        '"import json,sys; '
        "json.load(sys.stdin); "
        "print(json.dumps({'valid': False}))\""
    )
    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", sign_command)
    monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", verify_command)
    signer = SigstoreIntegration()

    signer.sign_artifact("artifact-3", "molecule", {"smiles": "CCC"})

    assert signer.verify_signature("artifact-3", "sigstore-signature") is False


def test_sigstore_persisted_bundle_verifies_without_process_cache(
    monkeypatch: pytest.MonkeyPatch,
):
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    sign_command = (
        f"{sys.executable} -c "
        '"import json,sys; '
        "payload=json.load(sys.stdin); "
        "print(json.dumps({"
        "'signature_type':'sigstore_rekor',"
        "'signature':'persisted-signature',"
        "'artifact_type':payload['artifact_type'],"
        "'payload_hash':payload['payload_hash'],"
        "'identity':'fulcio@example.com',"
        "'rekor_entry':{'uuid':'rekor-persisted'},"
        "'bundle':{'mediaType':'application/vnd.dev.sigstore.bundle+json'}"
        '}))"'
    )
    verify_command = (
        f"{sys.executable} -c "
        '"import json,sys; '
        "payload=json.load(sys.stdin); "
        "assert payload['signature']=='persisted-signature'; "
        "assert payload['bundle']['rekor_entry']['uuid']=='rekor-persisted'; "
        "assert payload['bundle']['bundle']['mediaType']; "
        "print(json.dumps({'valid': True}))\""
    )
    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", sign_command)
    monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", verify_command)
    raw_payload = b"persisted raw bytes"
    checksum = f"sha256:{hashlib.sha256(raw_payload).hexdigest()}"
    signed = SigstoreIntegration().sign_artifact(
        "artifact-persisted",
        "workflow_state",
        {"project_id": "project-1"},
        checksum=checksum,
        parent_ids=["artifact-parent"],
        recorded_at="2026-07-29T00:00:00+00:00",
    )
    signed_payload = signed["signed_payload"]
    signature_bundle = {key: value for key, value in signed.items() if key != "signed_payload"}
    record = {
        "artifact_id": "artifact-persisted",
        "artifact_type": "workflow_state",
        "parent_ids": ["artifact-parent"],
        "metadata": {"project_id": "project-1"},
        "payload_base64": base64.b64encode(raw_payload).decode("ascii"),
        "checksum": checksum,
        "signed_payload": signed_payload,
        "payload_hash": signature_bundle["payload_hash"],
        "signature_bundle": signature_bundle,
        "signature": signature_bundle["signature"],
        "certificate": signature_bundle["certificate"],
        "recorded_at": "2026-07-29T00:00:00+00:00",
        "signature_type": signature_bundle["signature_type"],
    }

    restarted = SigstoreIntegration()

    assert restarted.get_rekor_entry("artifact-persisted") is None
    assert restarted.verify_record(record, "persisted-signature") is True


def test_sigstore_verify_command_receives_artifact_identity_context(
    monkeypatch: pytest.MonkeyPatch,
):
    from provenance_svc.domain.sigstore_integration import SigstoreIntegration

    sign_command = (
        f"{sys.executable} -c "
        '"import json,sys; '
        "payload=json.load(sys.stdin); "
        "print(json.dumps({"
        "'signature_type':'sigstore_rekor',"
        "'signature':'sigstore-signature',"
        "'payload_hash':payload['payload_hash'],"
        "'rekor_entry':{'uuid':'rekor-uuid'}"
        '}))"'
    )
    verify_command = (
        f"{sys.executable} -c "
        '"import json,sys; '
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
        '"import json,sys; '
        "payload=json.load(sys.stdin); "
        "assert payload['rekor_url']=='https://rekor.example'; "
        "print(json.dumps({"
        "'signature_type':'sigstore_rekor',"
        "'signature':'sigstore-signature',"
        "'payload_hash':payload['payload_hash'],"
        "'rekor_entry':{'uuid':'rekor-uuid'}"
        '}))"'
    )
    verify_command = (
        f"{sys.executable} -c "
        '"import json,sys; '
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


def test_cosign_wrapper_reads_rotated_identity_token_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wrapper = _load_cosign_audit_wrapper()
    token_file = tmp_path / "identity-token"
    token_file.write_text("first-token\n", encoding="utf-8")
    monkeypatch.delenv("ACTIONS_ID_TOKEN_REQUEST_URL", raising=False)
    monkeypatch.delenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", raising=False)
    monkeypatch.setenv("SIGSTORE_IDENTITY_TOKEN_FILE", str(token_file))

    assert wrapper._identity_token({"identity_token": "stale-token"}) == "first-token"

    token_file.write_text("rotated-token\n", encoding="utf-8")
    assert wrapper._identity_token({"identity_token": "stale-token"}) == "rotated-token"


def test_agent_runtime_image_contains_verified_cosign_and_sigstore_wrapper() -> None:
    dockerfile = (ROOT / "infra/docker/base/Dockerfile.agent").read_text(encoding="utf-8")

    assert "COPY tools ./tools" in dockerfile
    assert "COSIGN_VERSION=v3.1.2" in dockerfile
    assert "f7622ed3cf22e55e1ae6377c080979ff77a22da9981c11df222a2e444991e7cf" in dockerfile
    assert "90e7ae0b5dfd60f20816b52c012addf7fc055ebcc7bea4ce81c428ca8518c302" in dockerfile
    assert "sha256sum -c -" in dockerfile


def test_sigstore_provenance_deployment_wires_production_env() -> None:
    import yaml

    compose = (ROOT / "infra/docker/docker-compose.dev.yml").read_text(encoding="utf-8")
    k8s = (ROOT / "infra/kubernetes/deployments/moleculeforge-services.yaml").read_text(
        encoding="utf-8"
    )
    helm_values = (ROOT / "infra/helm/moleculeforge/values.yaml").read_text(encoding="utf-8")
    helm_template = (ROOT / "infra/helm/moleculeforge/templates/services.yaml").read_text(
        encoding="utf-8"
    )

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

    k8s_documents = [doc for doc in yaml.safe_load_all(k8s) if isinstance(doc, dict)]
    k8s_config = next(
        doc
        for doc in k8s_documents
        if doc.get("kind") == "ConfigMap"
        and doc["metadata"]["name"] == "sigstore-provenance-config"
    )["data"]
    assert k8s_config["sign-command"] == (
        "python /workspace/tools/sigstore/cosign_audit_wrapper.py sign"
    )
    assert k8s_config["verify-command"] == (
        "python /workspace/tools/sigstore/cosign_audit_wrapper.py verify"
    )
    k8s_deployment = next(
        doc
        for doc in k8s_documents
        if doc.get("kind") == "Deployment"
        and doc["metadata"]["name"] == "provenance-svc"
    )
    k8s_container = k8s_deployment["spec"]["template"]["spec"]["containers"][0]
    k8s_env = {item["name"]: item for item in k8s_container["env"]}
    assert k8s_env["SIGSTORE_IDENTITY_TOKEN_FILE"]["value"] == (
        "/var/run/secrets/moleculeforge/sigstore/token"
    )
    assert {item["name"] for item in k8s_container["volumeMounts"]} >= {
        "sigstore-identity"
    }
    assert {item["name"] for item in k8s_deployment["spec"]["template"]["spec"]["volumes"]} >= {
        "sigstore-identity"
    }

    helm = yaml.safe_load(helm_values)
    helm_config = helm["configMaps"]["sigstore-provenance-config"]["data"]
    assert helm_config["sign-command"] == k8s_config["sign-command"]
    assert helm_config["verify-command"] == k8s_config["verify-command"]
    helm_provenance = helm["services"]["provenance-svc"]
    assert helm_provenance["env"]["SIGSTORE_IDENTITY_TOKEN_FILE"] == (
        "/var/run/secrets/moleculeforge/sigstore/token"
    )


def test_production_provenance_accepts_rotating_identity_token_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from provenance_svc import main as provenance

    token_file = tmp_path / "identity-token"
    token_file.write_text("projected-token\n", encoding="utf-8")
    for name, value in {
        "NEO4J_URI": "bolt://neo4j:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password",
        "PROVENANCE_DATABASE_URL": "postgresql+asyncpg://test",
        "MINIO_ENDPOINT_URL": "http://minio:9000",
        "MINIO_ACCESS_KEY": "access",
        "MINIO_SECRET_KEY": "secret",
        "MINIO_BUCKET": "mf-data",
        "SIGSTORE_SIGN_COMMAND": "python wrapper.py sign",
        "SIGSTORE_VERIFY_COMMAND": "python wrapper.py verify",
        "SIGSTORE_EXPECTED_IDENTITY": "service-account@example.test",
        "SIGSTORE_REKOR_URL": "https://rekor.sigstore.dev",
        "SIGSTORE_IDENTITY_TOKEN_FILE": str(token_file),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("SIGSTORE_IDENTITY_TOKEN", raising=False)

    assert "SIGSTORE_IDENTITY_TOKEN" not in provenance._missing_production_config()
