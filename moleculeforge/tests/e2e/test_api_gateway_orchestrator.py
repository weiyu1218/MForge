"""API Gateway orchestrator proxy tests."""

from __future__ import annotations

import asyncio
import json
import time
import tomllib
from pathlib import Path

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]


def _oidc_rsa_key_pair(
    key_id: str,
) -> tuple[rsa.RSAPrivateKey, jwt.PyJWK]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    public_jwk.update({"alg": "RS256", "kid": key_id, "use": "sig"})
    return private_key, jwt.PyJWK.from_dict(public_jwk)


class _StaticJWKClient:
    def __init__(self, signing_key: jwt.PyJWK) -> None:
        self._signing_key = signing_key

    def get_signing_key_from_jwt(self, token: str) -> jwt.PyJWK:
        header = jwt.get_unverified_header(token)
        if header.get("kid") != self._signing_key.key_id:
            raise jwt.InvalidKeyError("unknown signing key")
        return self._signing_key


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("ORCHESTRATOR_SVC_URL", "http://orchestrator.test")
    from api_gateway.main import app

    class _AuthenticatedOIDCAuth:
        async def authenticate(self, request: object) -> dict:
            return {"sub": "scientist-test"}

    monkeypatch.setattr(app.state, "oidc_auth", _AuthenticatedOIDCAuth(), raising=False)
    with TestClient(app) as test_client:
        yield test_client


def test_api_gateway_forwards_design_to_orchestrator(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict | None, dict[str, str] | None]] = []

    class _Response:
        status_code = 202

        def __init__(self, payload: dict) -> None:
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self) -> dict:
            return self._payload

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(
            self,
            url: str,
            json: dict,
            headers: dict[str, str] | None = None,
        ):
            calls.append(("POST", url, json, headers))
            return _Response(
                {
                    "design_id": "design-1",
                    "run_id": "run-1",
                    "status": "completed",
                    "history": ["PLANNING", "GENERATING", "VALIDATING", "RETROSYN", "CRITIC"],
                    "state": {
                        "candidates": [{"canonical_smiles": "CCO"}],
                        "critic": {"total_rules": 1},
                    },
                }
            )

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "gateway-service-token")

    response = client.post(
        "/v1/orchestrator/design",
        json={
            "nl_input": "Design KRAS G12C inhibitors",
            "project_id": "project-full",
            "max_refinements": 1,
            "n_samples": 2,
            "validation_policy": {"oracle_level": 0},
            "teacher_policy": {"teacher_source": "hypseek"},
            "selection_policy": {"criteria": []},
        },
    )

    assert response.status_code == 202
    assert response.json()["design_id"] == "design-1"
    assert calls == [
        (
            "POST",
            "http://orchestrator.test/v1/orchestrator/design",
            {
                "nl_input": "Design KRAS G12C inhibitors",
                "project_id": "project-full",
                "workflow_scope": "full",
                "max_refinements": 1,
                "n_samples": 2,
                "validation_policy": {"oracle_level": 0},
                "teacher_policy": {"teacher_source": "hypseek"},
                "selection_policy": {"criteria": []},
            },
            {
                "X-MoleculeForge-Service-Token": "gateway-service-token",
                "X-MoleculeForge-Principal": "scientist-test",
            },
        )
    ]


def test_api_gateway_forwards_external_evidence_resume(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api_gateway.main import app

    evidence = [
        {
            "candidate_id": "candidate-1",
            "metrics": {"activity": 0.8},
            "evidence_ids": ["artifact:measurement-1"],
        }
    ]
    calls: list[tuple[str, dict, dict[str, str] | None]] = []

    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "design_id": "run-evidence",
                "run_id": "run-evidence",
                "status": "running",
            }

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(
            self,
            url: str,
            json: dict,
            headers: dict[str, str] | None = None,
        ):
            calls.append((url, json, headers))
            return _Response()

    class _AuthenticatedOIDCAuth:
        async def authenticate(self, request: object) -> dict:
            return {"sub": "scientist-1"}

    monkeypatch.setattr(
        app.state,
        "oidc_auth",
        _AuthenticatedOIDCAuth(),
        raising=False,
    )
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "gateway-service-token")
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.post(
        "/v1/orchestrator/runs/run-evidence/evidence/resume",
        json={"external_evidence": evidence},
        headers={"Authorization": "Bearer signed-oidc-token"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert calls == [
        (
            ("http://orchestrator.test/v1/orchestrator/runs/run-evidence/evidence/resume"),
            {"external_evidence": evidence},
            {
                "X-MoleculeForge-Service-Token": "gateway-service-token",
                "X-MoleculeForge-Principal": "scientist-1",
            },
        )
    ]


def test_api_gateway_binds_full_design_to_authenticated_principal(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api_gateway.main import app

    calls: list[dict[str, str] | None] = []

    class _Response:
        status_code = 202

        def json(self) -> dict:
            return {"design_id": "run-owned", "run_id": "run-owned", "status": "queued"}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(
            self,
            url: str,
            json: dict,
            headers: dict[str, str] | None = None,
        ):
            calls.append(headers)
            return _Response()

    class _AuthenticatedOIDCAuth:
        async def authenticate(self, request: object) -> dict:
            return {"sub": "scientist-owner"}

    monkeypatch.setattr(app.state, "oidc_auth", _AuthenticatedOIDCAuth(), raising=False)
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "gateway-service-token")
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.post(
        "/v1/orchestrator/design",
        json={
            "nl_input": "Design an evidence-backed molecule",
            "workflow_scope": "full",
            "max_refinements": 1,
            "validation_policy": {"oracle_level": 4},
            "teacher_policy": {"teacher_source": "hypseek"},
            "selection_policy": {"criteria": []},
        },
        headers={"Authorization": "Bearer signed-oidc-token"},
    )

    assert response.status_code == 202
    assert calls == [
        {
            "X-MoleculeForge-Service-Token": "gateway-service-token",
            "X-MoleculeForge-Principal": "scientist-owner",
        }
    ]


def test_api_gateway_rejects_anonymous_external_evidence_resume(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api_gateway.main import app

    calls: list[str] = []

    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {"status": "running"}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict):
            calls.append(url)
            return _Response()

    class _AnonymousOIDCAuth:
        async def authenticate(self, request: object) -> dict:
            return {"anonymous": True}

    monkeypatch.setattr(app.state, "oidc_auth", _AnonymousOIDCAuth(), raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.post(
        "/v1/orchestrator/runs/run-evidence/evidence/resume",
        json={"external_evidence": []},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}
    assert calls == []


def test_api_gateway_rejects_evidence_resume_without_oidc_provider(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api_gateway.auth.oidc import OIDCAuth
    from api_gateway.main import app

    calls: list[str] = []

    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {"status": "running"}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict):
            calls.append(url)
            return _Response()

    monkeypatch.setattr(app.state, "oidc_auth", OIDCAuth(), raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.post(
        "/v1/orchestrator/runs/run-evidence/evidence/resume",
        json={"external_evidence": []},
        headers={"Authorization": "Bearer token-longer-than-ten-characters"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "OIDC provider is not configured"}
    assert calls == []


def test_api_gateway_accepts_injected_oidc_provider_and_verifier(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api_gateway.auth.oidc import OIDCAuth
    from api_gateway.main import app

    issuer = "https://identity.test"
    audience = "moleculeforge-api"
    token = jwt.encode(
        {
            "sub": "scientist-1",
            "iss": issuer,
            "aud": audience,
            "exp": 4102444800,
        },
        "test-only-selection-key-with-32-bytes",
        algorithm="HS256",
    )
    calls: list[str] = []

    class _Verifier:
        def verify(self, encoded_token: str, provider: dict) -> dict:
            if encoded_token != token:
                raise AssertionError("authenticator passed the wrong token")
            if provider != {
                "issuer": issuer,
                "client_id": audience,
                "jwks_uri": "https://identity.test/keys",
            }:
                raise AssertionError("authenticator selected the wrong provider")
            return {
                "sub": "scientist-1",
                "email": "scientist@example.test",
                "name": "Scientist",
                "preferred_username": "scientist",
            }

    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {"status": "running"}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict):
            calls.append(url)
            return _Response()

    try:
        authenticator = OIDCAuth(
            providers={
                "test": {
                    "issuer": issuer,
                    "client_id": audience,
                    "jwks_uri": "https://identity.test/keys",
                }
            },
            verifier=_Verifier(),
        )
    except TypeError as exc:
        pytest.fail(f"OIDC provider and verifier injection is unavailable: {exc}")
    monkeypatch.setattr(app.state, "oidc_auth", authenticator, raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.post(
        "/v1/orchestrator/runs/run-evidence/evidence/resume",
        json={"external_evidence": []},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert calls == [("http://orchestrator.test/v1/orchestrator/runs/run-evidence/evidence/resume")]


def test_oidc_auth_reads_provider_from_environment(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api_gateway.auth.oidc import OIDCAuth
    from api_gateway.main import app

    issuer = "https://environment-identity.test"
    audience = "environment-api"
    jwks_uri = "https://environment-identity.test/keys"
    monkeypatch.setenv("OIDC_ISSUER", issuer)
    monkeypatch.setenv("OIDC_AUDIENCE", audience)
    monkeypatch.setenv("OIDC_JWKS_URI", jwks_uri)
    token = jwt.encode(
        {
            "sub": "scientist-2",
            "iss": issuer,
            "aud": audience,
            "exp": 4102444800,
        },
        "another-test-only-selection-key-32-bytes",
        algorithm="HS256",
    )
    calls: list[str] = []

    class _Verifier:
        def verify(self, encoded_token: str, provider: dict) -> dict:
            if encoded_token != token:
                raise AssertionError("authenticator passed the wrong token")
            if provider != {
                "issuer": issuer,
                "client_id": audience,
                "jwks_uri": jwks_uri,
            }:
                raise AssertionError("environment provider was not registered")
            return {"sub": "scientist-2"}

    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {"status": "running"}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict):
            calls.append(url)
            return _Response()

    authenticator = OIDCAuth.from_environment(verifier=_Verifier())
    monkeypatch.setattr(app.state, "oidc_auth", authenticator, raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.post(
        "/v1/orchestrator/runs/run-evidence/evidence/resume",
        json={"external_evidence": []},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert calls == [("http://orchestrator.test/v1/orchestrator/runs/run-evidence/evidence/resume")]


def test_oidc_auth_rejects_incomplete_environment_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api_gateway.auth.oidc import OIDCAuth

    monkeypatch.delenv("OIDC_ISSUER", raising=False)
    monkeypatch.delenv("OIDC_AUDIENCE", raising=False)
    monkeypatch.delenv("OIDC_JWKS_URI", raising=False)
    monkeypatch.setenv("OIDC_ISSUER", "https://identity.test")

    with pytest.raises(ValueError, match="must be configured together"):
        OIDCAuth.from_environment()


def test_api_gateway_accepts_rs256_oidc_token_from_static_jwks(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api_gateway.auth.oidc import OIDCAuth, PyJWTVerifier
    from api_gateway.main import app

    issuer = "https://identity.test"
    audience = "moleculeforge-api"
    private_key, signing_key = _oidc_rsa_key_pair("key-1")
    token = jwt.encode(
        {
            "sub": "scientist-1",
            "iss": issuer,
            "aud": audience,
            "exp": int(time.time()) + 300,
            "email": "scientist@example.test",
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "key-1"},
    )
    calls: list[str] = []

    def jwks_client_factory(uri: str) -> _StaticJWKClient:
        if uri != "https://identity.test/keys":
            raise AssertionError("verifier used an unconfigured JWKS endpoint")
        return _StaticJWKClient(signing_key)

    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {"status": "running"}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict):
            calls.append(url)
            return _Response()

    authenticator = OIDCAuth(
        providers={
            "test": {
                "issuer": issuer,
                "client_id": audience,
                "jwks_uri": "https://identity.test/keys",
            }
        },
        verifier=PyJWTVerifier(jwks_client_factory=jwks_client_factory),
    )
    monkeypatch.setattr(app.state, "oidc_auth", authenticator, raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.post(
        "/v1/orchestrator/runs/run-evidence/evidence/resume",
        json={"external_evidence": []},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert calls == [("http://orchestrator.test/v1/orchestrator/runs/run-evidence/evidence/resume")]


@pytest.mark.parametrize(
    "invalid_token_kind",
    [
        "signature",
        "expired",
        "issuer",
        "audience",
        "missing_exp",
        "unknown_kid",
    ],
)
def test_api_gateway_rejects_invalid_oidc_token(
    invalid_token_kind: str,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api_gateway.auth.oidc import OIDCAuth, PyJWTVerifier
    from api_gateway.main import app

    issuer = "https://identity.test"
    audience = "moleculeforge-api"
    private_key, signing_key = _oidc_rsa_key_pair("key-1")
    wrong_private_key, _ = _oidc_rsa_key_pair("key-1")
    claims: dict[str, object] = {
        "sub": "scientist-1",
        "iss": issuer,
        "aud": audience,
        "exp": int(time.time()) + 300,
    }
    signing_private_key = private_key
    signing_key_id = "key-1"
    if invalid_token_kind == "signature":
        signing_private_key = wrong_private_key
    elif invalid_token_kind == "expired":
        claims["exp"] = int(time.time()) - 60
    elif invalid_token_kind == "issuer":
        claims["iss"] = "https://attacker.test"
    elif invalid_token_kind == "audience":
        claims["aud"] = "another-api"
    elif invalid_token_kind == "missing_exp":
        claims.pop("exp")
    elif invalid_token_kind == "unknown_kid":
        signing_key_id = "unknown-key"
    token = jwt.encode(
        claims,
        signing_private_key,
        algorithm="RS256",
        headers={"kid": signing_key_id},
    )
    calls: list[str] = []

    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {"status": "running"}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict):
            calls.append(url)
            return _Response()

    authenticator = OIDCAuth(
        providers={
            "test": {
                "issuer": issuer,
                "client_id": audience,
                "jwks_uri": "https://identity.test/keys",
            }
        },
        verifier=PyJWTVerifier(jwks_client_factory=lambda uri: _StaticJWKClient(signing_key)),
    )
    monkeypatch.setattr(app.state, "oidc_auth", authenticator, raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.post(
        "/v1/orchestrator/runs/run-evidence/evidence/resume",
        json={"external_evidence": []},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}
    assert calls == []


def test_api_gateway_rejects_evidence_resume_when_jwks_is_unavailable(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api_gateway.auth.oidc import OIDCAuth, PyJWTVerifier
    from api_gateway.main import app

    issuer = "https://identity.test"
    audience = "moleculeforge-api"
    private_key, _ = _oidc_rsa_key_pair("key-1")
    token = jwt.encode(
        {
            "sub": "scientist-1",
            "iss": issuer,
            "aud": audience,
            "exp": int(time.time()) + 300,
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "key-1"},
    )
    calls: list[str] = []

    class _UnavailableJWKClient:
        def get_signing_key_from_jwt(self, encoded_token: str) -> jwt.PyJWK:
            raise jwt.PyJWKClientConnectionError("JWKS endpoint is unavailable")

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict):
            calls.append(url)
            raise AssertionError("the gateway must not call the orchestrator")

    authenticator = OIDCAuth(
        providers={
            "test": {
                "issuer": issuer,
                "client_id": audience,
                "jwks_uri": "https://identity.test/keys",
            }
        },
        verifier=PyJWTVerifier(jwks_client_factory=lambda uri: _UnavailableJWKClient()),
    )
    monkeypatch.setattr(app.state, "oidc_auth", authenticator, raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.post(
        "/v1/orchestrator/runs/run-evidence/evidence/resume",
        json={"external_evidence": []},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "OIDC verification is unavailable"}
    assert calls == []


def test_api_gateway_forwards_design_status_to_orchestrator(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api_gateway.main import app

    calls: list[tuple[str, str]] = []

    class _Response:
        status_code = 200
        text = '{"design_id":"design-1","status":"completed"}'

        def json(self) -> dict:
            return {"design_id": "design-1", "status": "completed"}

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            calls.append(("GET", url))
            return _Response()

    class _AuthenticatedOIDCAuth:
        async def authenticate(self, request: object) -> dict:
            return {"sub": "scientist-1"}

    monkeypatch.setattr(app.state, "oidc_auth", _AuthenticatedOIDCAuth(), raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.get(
        "/v1/orchestrator/runs/design-1",
        headers={"Authorization": "Bearer signed-oidc-token"},
    )

    assert response.status_code == 200
    assert response.json() == {"design_id": "design-1", "status": "completed"}
    assert calls == [("GET", "http://orchestrator.test/v1/orchestrator/runs/design-1")]


def test_api_gateway_exposes_canonical_async_run_lifecycle(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api_gateway.main import app

    calls: list[
        tuple[
            str,
            str,
            dict[str, object] | None,
            dict[str, object] | None,
            dict[str, str] | None,
        ]
    ] = []

    class _Response:
        status_code = 200
        text = '{"status":"ok"}'

        def json(self) -> dict[str, str]:
            return {"status": "ok"}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(
            self,
            url: str,
            params: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
        ) -> _Response:
            calls.append(("GET", url, None, params, headers))
            return _Response()

        async def post(
            self,
            url: str,
            json: dict[str, object],
            headers: dict[str, str] | None = None,
        ) -> _Response:
            calls.append(("POST", url, json, None, headers))
            return _Response()

    class _AuthenticatedOIDCAuth:
        async def authenticate(self, request: object) -> dict[str, str]:
            return {"sub": "scientist-1"}

    monkeypatch.setattr(app.state, "oidc_auth", _AuthenticatedOIDCAuth(), raising=False)
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "gateway-service-token")
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    headers = {"Authorization": "Bearer signed-oidc-token"}

    responses = [
        client.get("/v1/orchestrator/runs?page_size=30", headers=headers),
        client.get("/v1/orchestrator/runs/run-1", headers=headers),
        client.get(
            "/v1/orchestrator/runs/run-1/events?after_step=3",
            headers=headers,
        ),
        client.post("/v1/orchestrator/runs/run-1/pause", json={}, headers=headers),
        client.post("/v1/orchestrator/runs/run-1/resume", json={}, headers=headers),
        client.post(
            "/v1/orchestrator/runs/run-1/evidence/resume",
            json={"external_evidence": []},
            headers=headers,
        ),
        client.post("/v1/orchestrator/runs/run-1/cancel", json={}, headers=headers),
    ]

    assert [response.status_code for response in responses] == [200] * len(responses)
    service_headers = {
        "X-MoleculeForge-Service-Token": "gateway-service-token",
        "X-MoleculeForge-Principal": "scientist-1",
    }
    assert calls == [
        (
            "GET",
            "http://orchestrator.test/v1/orchestrator/runs",
            None,
            {"page_size": 30},
            service_headers,
        ),
        (
            "GET",
            "http://orchestrator.test/v1/orchestrator/runs/run-1",
            None,
            None,
            service_headers,
        ),
        (
            "GET",
            "http://orchestrator.test/v1/orchestrator/runs/run-1/events",
            None,
            {"after_step": 3},
            service_headers,
        ),
        (
            "POST",
            "http://orchestrator.test/v1/orchestrator/runs/run-1/pause",
            {},
            None,
            service_headers,
        ),
        (
            "POST",
            "http://orchestrator.test/v1/orchestrator/runs/run-1/resume",
            {},
            None,
            service_headers,
        ),
        (
            "POST",
            "http://orchestrator.test/v1/orchestrator/runs/run-1/evidence/resume",
            {"external_evidence": []},
            None,
            service_headers,
        ),
        (
            "POST",
            "http://orchestrator.test/v1/orchestrator/runs/run-1/cancel",
            {},
            None,
            service_headers,
        ),
    ]


def test_static_ui_submits_runs_through_orchestrator_gateway() -> None:
    gateway_source = (ROOT / "services/api-gateway/src/api_gateway/main.py").read_text(
        encoding="utf-8"
    )
    markup = (ROOT / "ui/public/index.html").read_text(encoding="utf-8")
    script = (ROOT / "ui/public/app.js").read_text(encoding="utf-8")

    assert "/workspace" not in gateway_source
    assert 'api("/orchestrator/design"' in script
    for field_id in (
        "bearer-token",
        "project-id",
        "max-refinements",
        "n-samples",
        "generation-strategy",
        "retrosyn-engine",
        "validation-policy",
        "teacher-version",
        "allow-synthetic",
        "kd-weight",
        "selection-policy",
    ):
        assert f'id="{field_id}"' in markup
    assert 'id="workflow-scope"' not in markup
    assert 'id="validation-passed"' not in markup
    assert 'id="known-modal"' not in markup
    assert 'id="show-known"' not in markup
    submit_block = script.split('$("#run").addEventListener("click"', 1)[1].split(
        "/* ---------------- run rendering ---------------- */",
        1,
    )[0]
    assert "/reason/runs" not in submit_block
    assert 'workflow_scope: "full"' in submit_block
    assert 'teacher_source: "hypseek"' in submit_block
    assert 'max_refinements: Number($("#max-refinements").value)' in submit_block
    assert "openRun(r.run_id" in submit_block
    assert "Authorization" in script
    assert "bearerToken" in script
    assert "localStorage" not in script
    assert "sessionStorage" not in script
    assert "EventSource" not in script
    assert "/reason/" not in script
    assert "/stream/" not in script
    assert "api(`/design/" not in script
    assert "/orchestrator/runs/" in script
    assert "after_step=" in script
    assert "ownsActiveRun(runId, generation)" in script
    assert "live: !isTerminalRun(run.status)" in script


def test_api_gateway_does_not_expose_orchestrator_compatibility_routes(
    client: TestClient,
) -> None:
    responses = [
        client.get("/v1/design/run-1"),
        client.get("/v1/reason/known"),
        client.get("/v1/stream/run-1"),
    ]

    assert [response.status_code for response in responses] == [404, 404, 404]


def test_gateway_deployment_wires_orchestrator_and_oidc_configuration() -> None:
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

    compose_env = compose["services"]["api-gateway"]["environment"]
    assert compose_env["ORCHESTRATOR_SVC_URL"] == (
        "${ORCHESTRATOR_SVC_URL:-http://orchestrator-svc:8011}"
    )
    for name in ("OIDC_ISSUER", "OIDC_AUDIENCE", "OIDC_JWKS_URI"):
        assert name in compose_env
    assert compose_env["INTERNAL_SERVICE_TOKEN"] == (
        "${INTERNAL_SERVICE_TOKEN:-mf_dev_internal_service_token}"
    )

    deployment = next(
        resource
        for resource in kubernetes
        if resource
        and resource.get("kind") == "Deployment"
        and resource["metadata"]["name"] == "api-gateway"
    )
    env = {
        item["name"]: item
        for item in deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert env["ORCHESTRATOR_SVC_URL"]["value"] == (
        "http://orchestrator-svc.mf-agents.svc.cluster.local:8011"
    )
    for name in ("OIDC_ISSUER", "OIDC_AUDIENCE", "OIDC_JWKS_URI"):
        assert env[name]["valueFrom"]["configMapKeyRef"]["name"] == (
            "api-gateway-config"
        )
    assert env["INTERNAL_SERVICE_TOKEN"]["valueFrom"]["secretKeyRef"] == {
        "name": "agent-runtime-secrets",
        "key": "INTERNAL_SERVICE_TOKEN",
    }

    helm_gateway = helm["services"]["api-gateway"]
    assert helm_gateway["env"]["ORCHESTRATOR_SVC_URL"] == (
        "http://orchestrator-svc.mf-agents.svc.cluster.local:8011"
    )
    assert set(helm_gateway["envValueFrom"]) == {
        "OIDC_ISSUER",
        "OIDC_AUDIENCE",
        "OIDC_JWKS_URI",
        "INTERNAL_SERVICE_TOKEN",
    }
    assert set(helm["configMaps"]["api-gateway-config"]["data"]) == {
        "oidc-issuer",
        "oidc-audience",
        "oidc-jwks-uri",
    }
    gateway_project = tomllib.loads(
        (ROOT / "services/api-gateway/pyproject.toml").read_text(encoding="utf-8")
    )
    assert any(
        dependency.startswith("pyjwt[crypto]")
        for dependency in gateway_project["project"]["dependencies"]
    )


def test_gateway_orchestrator_error_detail_is_not_nested(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 400
        text = '{"detail":"workflow_scope must be full"}'

        def json(self) -> dict:
            return {"detail": "workflow_scope must be full"}

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.post(
        "/v1/orchestrator/design",
        json={"nl_input": "intent", "workflow_scope": "engineering"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "workflow_scope must be full"}


def test_gateway_does_not_expose_legacy_orchestrator_aliases(
    client: TestClient,
) -> None:
    from api_gateway.main import app

    status = client.get("/v1/orchestrator/run-legacy")
    evidence = client.post(
        "/v1/orchestrator/run-legacy/evidence/resume",
        json={"external_evidence": []},
    )
    paths = {getattr(route, "path", "") for route in app.routes}

    assert status.status_code == 404
    assert evidence.status_code in {404, 405}
    assert "/v1/orchestrator/{design_id}" not in paths
    assert "/v1/orchestrator/{design_id}/evidence/resume" not in paths


@pytest.mark.parametrize(
    ("project_id", "external_project_id"),
    [
        ("shared-project", "shared-project"),
        ("space name", "space%20name"),
        ("R&D #1", "R%26D%20%231"),
        ("A/B", "A%2FB"),
    ],
)
def test_project_routes_proxy_to_independent_orchestrator_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    project_id: str,
    external_project_id: str,
) -> None:
    from api_gateway.main import app
    from mf_core.db.store import RunStore
    from orchestrator_svc import main as orchestrator_main
    from orchestrator_svc.main import RunControl

    gateway_database_path = tmp_path / "gateway.db"
    orchestrator_database_path = tmp_path / "orchestrator.db"
    monkeypatch.setenv("MF_DB_PATH", str(gateway_database_path))
    monkeypatch.setenv("ORCHESTRATOR_SVC_URL", "http://orchestrator.test")
    store = RunStore(orchestrator_database_path)
    asyncio.run(store.initialize())
    monkeypatch.setattr(orchestrator_main, "_RUN_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUN_CONTROL", RunControl(store))
    monkeypatch.setattr(orchestrator_main, "_RUN_INITIALIZED_STORE", store)
    monkeypatch.setattr(orchestrator_main, "_RUNTIME_INIT_LOCK", None)
    monkeypatch.setattr(orchestrator_main, "_RUN_TASKS", {})
    monkeypatch.setattr(
        orchestrator_main,
        "_register_design_run_task",
        lambda run_id, request, initial_state, **kwargs: None,
    )
    real_async_client = httpx.AsyncClient

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(
            self,
            exc_type: object,
            exc: object,
            tb: object,
        ) -> None:
            return None

        async def get(
            self,
            url: str,
            params: dict | None = None,
        ) -> httpx.Response:
            return await self._request("GET", url, params=params)

        async def post(self, url: str, json: dict) -> httpx.Response:
            return await self._request("POST", url, json=json)

        async def delete(self, url: str) -> httpx.Response:
            return await self._request("DELETE", url)

        async def _request(
            self,
            method: str,
            url: str,
            *,
            json: dict | None = None,
            params: dict | None = None,
        ) -> httpx.Response:
            transport = httpx.ASGITransport(app=orchestrator_main.rest_app)
            async with real_async_client(
                transport=transport,
                base_url="http://orchestrator.test",
            ) as upstream:
                path = url.removeprefix("http://orchestrator.test")
                return await upstream.request(method, path, json=json, params=params)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    with TestClient(app) as gateway:
        created = gateway.post(
            "/v1/projects/",
            json={"name": project_id, "description": "first"},
        )
        assert created.status_code == 200
        created_at = created.json()["created_at"]
        assert created.json() == {
            "project_id": project_id,
            "name": project_id,
            "description": "first",
            "status": "active",
            "created_at": created_at,
            "designs": [],
        }

        orchestrator_project = asyncio.run(store.get_project(project_id))
        assert orchestrator_project == {
            "project_id": project_id,
            "name": project_id,
            "description": "first",
            "created_at": created_at,
        }
        gateway_store = RunStore(gateway_database_path)
        assert asyncio.run(gateway_store.get_project(project_id)) is None

        updated = gateway.post(
            "/v1/projects/",
            json={"name": project_id, "description": "updated"},
        )
        assert updated.json() == {
            **created.json(),
            "description": "updated",
        }
        fetched = gateway.get(f"/v1/projects/{external_project_id}")
        listed = gateway.get("/v1/projects/")
        deleted = gateway.delete(f"/v1/projects/{external_project_id}")
        missing_get = gateway.get(f"/v1/projects/{external_project_id}")
        missing_delete = gateway.delete(f"/v1/projects/{external_project_id}")

        assert fetched.status_code == 200
        assert fetched.json() == updated.json()
        assert listed.json() == {
            "projects": [updated.json()],
            "n_projects": 1,
        }
        assert deleted.status_code == 200
        assert deleted.json() == {
            "deleted": True,
            "project_id": project_id,
        }
        assert missing_get.status_code == 404
        assert missing_delete.status_code == 404

    assert asyncio.run(store.list_projects()) == []


def test_pareto_routes_read_canonical_orchestrator_snapshot(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forwarded_headers: list[dict[str, str] | None] = []

    class _Response:
        status_code = 200
        text = ""

        def json(self) -> dict:
            return {
                "run_id": "run-pareto",
                "status": "completed",
                "state": {
                    "objectives": ["qed", "sa_score"],
                    "results": [
                        {
                            "rank": 1,
                            "canonical_smiles": "CCO",
                            "valid": True,
                            "pareto_optimal": True,
                            "qed": 0.7,
                            "sa_score": 2.0,
                            "logp": 1.0,
                            "composite_score": 0.8,
                        }
                    ],
                    "validation": {
                        "results": [
                            {
                                "rank": 1,
                                "canonical_smiles": "CCO",
                                "valid": True,
                                "pareto_optimal": True,
                                "qed": 0.7,
                                "sa_score": 2.0,
                                "logp": 1.0,
                                "composite_score": 0.8,
                            }
                        ]
                    },
                },
            }

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(
            self,
            url: str,
            params: dict | None = None,
            headers: dict[str, str] | None = None,
        ):
            forwarded_headers.append(headers)
            return _Response()

    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "gateway-service-token")
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    frontier = client.get("/v1/pareto/run-pareto/frontier")
    hypervolume = client.get("/v1/pareto/run-pareto/hypervolume")
    missing_weights = client.post("/v1/pareto/run-pareto/select", json={})
    selected = client.post(
        "/v1/pareto/run-pareto/select",
        json={"weights": {"qed": 1.0}, "top_k": 1},
    )

    assert frontier.status_code == 200
    assert frontier.json()["frontier"][0]["smiles"] == "CCO"
    assert hypervolume.status_code == 200
    assert hypervolume.json()["n_points"] == 1
    assert missing_weights.status_code == 400
    assert missing_weights.json() == {"detail": "weights is required"}
    assert selected.status_code == 200
    assert selected.json()["selected"][0]["smiles"] == "CCO"
    assert forwarded_headers == [
        {
            "X-MoleculeForge-Service-Token": "gateway-service-token",
            "X-MoleculeForge-Principal": "scientist-test",
        }
    ] * 4


def test_pareto_only_uses_explicitly_matched_top_level_validations(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "run_id": "run-validation-gate",
                "status": "completed",
                "state": {
                    "candidates": [
                        {
                            "candidate_id": "candidate-unvalidated",
                            "canonical_smiles": "CCC",
                            "pareto_optimal": True,
                            "properties": {
                                "valid": True,
                                "qed": 0.99,
                                "sa_score": 1.0,
                            },
                        },
                        {
                            "candidate_id": "candidate-invalid",
                            "canonical_smiles": "CCN",
                            "pareto_optimal": True,
                            "properties": {
                                "valid": True,
                                "qed": 0.98,
                                "sa_score": 1.1,
                            },
                        },
                        {
                            "candidate_id": "candidate-nested-valid",
                            "canonical_smiles": "CCCl",
                            "pareto_optimal": True,
                            "properties": {
                                "valid": True,
                                "qed": 0.97,
                                "sa_score": 1.2,
                            },
                        },
                        {
                            "candidate_id": "candidate-valid",
                            "canonical_smiles": "CCO",
                            "pareto_optimal": True,
                            "properties": {
                                "qed": 0.70,
                                "sa_score": 2.0,
                            },
                        },
                    ],
                    "validation": {
                        "results": [
                            {
                                "candidate_id": "candidate-invalid",
                                "canonical_smiles": "CCN",
                                "valid": False,
                            },
                            {
                                "candidate_id": "candidate-nested-valid",
                                "canonical_smiles": "CCCl",
                                "properties": {"valid": True},
                            },
                            {
                                "candidate_id": "candidate-valid",
                                "canonical_smiles": "CCO",
                                "valid": True,
                            },
                        ]
                    },
                },
            }

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    frontier = client.get("/v1/pareto/run-validation-gate/frontier")
    hypervolume = client.get("/v1/pareto/run-validation-gate/hypervolume")
    selected = client.post(
        "/v1/pareto/run-validation-gate/select",
        json={"weights": {"qed": 1.0}, "top_k": 10},
    )

    assert frontier.status_code == 200
    assert [row["smiles"] for row in frontier.json()["frontier"]] == ["CCO"]
    assert hypervolume.status_code == 200
    assert hypervolume.json()["n_points"] == 1
    assert selected.status_code == 200
    assert [row["smiles"] for row in selected.json()["selected"]] == ["CCO"]


@pytest.mark.parametrize("verified_source", ["snapshot", "results", "ranked"])
def test_pareto_filters_already_verified_sources_by_their_own_top_level_valid(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    verified_source: str,
) -> None:
    verified_rows = [
        {
            "canonical_smiles": "CCO",
            "valid": True,
            "pareto_optimal": True,
            "properties": {"valid": False, "qed": 0.8, "sa_score": 2.0},
        },
        {
            "canonical_smiles": "CCN",
            "valid": False,
            "pareto_optimal": True,
            "properties": {"valid": True, "qed": 0.9, "sa_score": 1.0},
        },
        {
            "canonical_smiles": "CCC",
            "pareto_optimal": True,
            "properties": {"valid": True, "qed": 1.0, "sa_score": 1.0},
        },
    ]
    snapshot = {
        "run_id": f"run-verified-{verified_source}",
        "status": "completed",
        "state": {},
    }
    if verified_source == "snapshot":
        snapshot["results"] = verified_rows
    else:
        snapshot["state"][verified_source] = verified_rows

    class _Response:
        status_code = 200

        def json(self) -> dict:
            return snapshot

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    design_id = snapshot["run_id"]

    frontier = client.get(f"/v1/pareto/{design_id}/frontier")
    hypervolume = client.get(f"/v1/pareto/{design_id}/hypervolume")
    selected = client.post(
        f"/v1/pareto/{design_id}/select",
        json={"weights": {"qed": 1.0}, "top_k": 10},
    )

    assert [row["smiles"] for row in frontier.json()["frontier"]] == ["CCO"]
    assert hypervolume.json()["n_points"] == 1
    assert [row["smiles"] for row in selected.json()["selected"]] == ["CCO"]


def test_pareto_accepts_only_explicit_canonical_validation_success(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "run_id": "run-canonical-validation-facts",
                "status": "completed",
                "state": {
                    "results": [
                        {
                            "canonical_smiles": "CCO",
                            "rank": 2,
                            "overall_passed": True,
                            "pareto_optimal": True,
                            "properties": {"qed": 0.8, "sa_score": 2.0},
                        },
                        {
                            "canonical_smiles": "CCN",
                            "rank": 1,
                            "status": "validated",
                            "pareto_optimal": True,
                            "properties": {"qed": 0.9, "sa_score": 1.5},
                        },
                        {
                            "canonical_smiles": "CCC",
                            "rank": 3,
                            "overall_passed": False,
                            "status": "validated",
                            "pareto_optimal": True,
                            "properties": {
                                "valid": True,
                                "qed": 0.99,
                                "sa_score": 1.0,
                            },
                        },
                        {
                            "canonical_smiles": "CCCl",
                            "rank": 4,
                            "pareto_optimal": True,
                            "properties": {
                                "valid": True,
                                "qed": 1.0,
                                "sa_score": 1.0,
                            },
                        },
                    ]
                },
            }

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    frontier = client.get("/v1/pareto/run-canonical-validation-facts/frontier")
    hypervolume = client.get("/v1/pareto/run-canonical-validation-facts/hypervolume")

    assert frontier.status_code == 200
    assert [(row["rank"], row["smiles"]) for row in frontier.json()["frontier"]] == [
        (1, "CCN"),
        (2, "CCO"),
    ]
    assert hypervolume.status_code == 200
    assert hypervolume.json()["n_points"] == 2


def test_pareto_orders_duplicate_occurrences_by_validation_rank(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "run_id": "run-canonical-ranked-occurrences",
                "status": "completed",
                "state": {
                    "candidates": [
                        {
                            "candidate_id": "candidate-duplicate",
                            "canonical_smiles": "CCO",
                        },
                        {
                            "candidate_id": "candidate-duplicate",
                            "canonical_smiles": "CCN",
                        },
                    ],
                    "validation": {
                        "results": [
                            {
                                "candidate_id": "candidate-duplicate",
                                "canonical_smiles": "CCN",
                                "rank": 1,
                                "overall_passed": True,
                                "pareto_optimal": True,
                                "properties": {"qed": 0.9, "sa_score": 1.5},
                            },
                            {
                                "candidate_id": "candidate-duplicate",
                                "canonical_smiles": "CCO",
                                "rank": 2,
                                "status": "validated",
                                "pareto_optimal": True,
                                "properties": {"qed": 0.8, "sa_score": 2.0},
                            },
                        ]
                    },
                },
            }

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.get("/v1/pareto/run-canonical-ranked-occurrences/frontier")

    assert response.status_code == 200
    assert [(row["rank"], row["smiles"]) for row in response.json()["frontier"]] == [
        (1, "CCN"),
        (2, "CCO"),
    ]


def test_pareto_candidate_index_selects_the_canonical_duplicate_occurrence(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "run_id": "run-canonical-candidate-index",
                "status": "completed",
                "state": {
                    "candidates": [
                        {
                            "candidate_id": "candidate-duplicate",
                            "canonical_smiles": "CCO",
                            "composite_score": 1.0,
                        },
                        {
                            "candidate_id": "candidate-duplicate",
                            "canonical_smiles": "CCO",
                            "composite_score": 2.0,
                        },
                    ],
                    "validation": {
                        "results": [
                            {
                                "candidate_index": 1,
                                "candidate_id": "candidate-duplicate",
                                "canonical_smiles": "CCO",
                                "rank": 1,
                                "overall_passed": True,
                                "pareto_optimal": True,
                                "properties": {"qed": 0.9, "sa_score": 1.5},
                            },
                            {
                                "candidate_id": "candidate-duplicate",
                                "canonical_smiles": "CCO",
                                "rank": 2,
                                "overall_passed": True,
                                "pareto_optimal": True,
                                "properties": {"qed": 0.8, "sa_score": 2.0},
                            },
                        ]
                    },
                },
            }

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.get("/v1/pareto/run-canonical-candidate-index/frontier")

    assert response.status_code == 200
    assert [
        (
            row["rank"],
            row["composite_score"],
            row["objectives"]["qed"],
        )
        for row in response.json()["frontier"]
    ] == [
        (1, 2.0, 0.9),
        (2, 1.0, 0.8),
    ]


@pytest.mark.parametrize(
    ("candidate_index", "candidate_id", "canonical_smiles"),
    [
        (2, "candidate-1", "CCO"),
        (True, "candidate-1", "CCO"),
        (None, "candidate-1", "CCO"),
        (0, "candidate-other", "CCO"),
        (0, "candidate-1", "CCN"),
    ],
)
def test_pareto_rejects_invalid_explicit_candidate_index_without_fallback(
    candidate_index: object,
    candidate_id: str,
    canonical_smiles: str,
) -> None:
    from api_gateway.routers.pareto import _merge_candidate_results

    merged = _merge_candidate_results(
        [
            {
                "candidate_id": "candidate-1",
                "canonical_smiles": "CCO",
            }
        ],
        [
            {
                "candidate_index": candidate_index,
                "candidate_id": candidate_id,
                "canonical_smiles": canonical_smiles,
                "rank": 1,
                "overall_passed": True,
            }
        ],
        require_validated=True,
    )

    assert merged == []


def test_pareto_treats_explicit_empty_verified_results_as_authoritative(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "run_id": "run-empty-verified-results",
                "status": "completed",
                "results": [],
                "state": {
                    "candidates": [
                        {
                            "candidate_id": "candidate-raw",
                            "canonical_smiles": "CCO",
                            "pareto_optimal": True,
                        }
                    ],
                    "validation": {
                        "results": [
                            {
                                "candidate_id": "candidate-raw",
                                "canonical_smiles": "CCO",
                                "valid": True,
                            }
                        ]
                    },
                },
            }

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.get("/v1/pareto/run-empty-verified-results/frontier")

    assert response.status_code == 200
    assert response.json()["frontier"] == []


def test_pareto_hypervolume_trusts_matched_validation_top_level_valid(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "run_id": "run-top-level-valid",
                "status": "completed",
                "state": {
                    "candidates": [
                        {
                            "candidate_id": "candidate-1",
                            "canonical_smiles": "CCO",
                            "properties": {
                                "valid": False,
                                "qed": 0.8,
                                "sa_score": 2.0,
                            },
                        }
                    ],
                    "validation": {
                        "results": [
                            {
                                "candidate_id": "candidate-1",
                                "canonical_smiles": "CCO",
                                "valid": True,
                            }
                        ]
                    },
                },
            }

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    response = client.get("/v1/pareto/run-top-level-valid/hypervolume")

    assert response.status_code == 200
    assert response.json()["n_points"] == 1
    assert response.json()["hypervolume"] > 0


def test_pareto_merges_production_candidates_and_validation_results(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200
        text = ""

        def json(self) -> dict:
            return {
                "run_id": "run-production-shape",
                "status": "completed",
                "state": {
                    "candidates": [
                        {
                            "candidate_id": "candidate-1",
                            "canonical_smiles": "CCO",
                            "properties": {"qed": 0.1},
                        },
                        {
                            "candidate_id": "candidate-2",
                            "canonical_smiles": "CCN",
                        },
                    ],
                    "validation": {
                        "results": [
                            {
                                "candidate_id": "candidate-1",
                                "valid": True,
                                "pareto_optimal": True,
                                "properties": {
                                    "qed": 0.8,
                                    "sa_score": 2.0,
                                    "logp": 1.2,
                                },
                            },
                            {
                                "canonical_smiles": "CCN",
                                "valid": True,
                                "properties": {
                                    "qed": 0.6,
                                    "sa_score": 3.0,
                                    "logp": 1.5,
                                },
                            },
                        ]
                    },
                },
            }

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    frontier = client.get("/v1/pareto/run-production-shape/frontier")
    hypervolume = client.get("/v1/pareto/run-production-shape/hypervolume")
    selected = client.post(
        "/v1/pareto/run-production-shape/select",
        json={"weights": {"qed": 1.0}, "top_k": 2},
    )

    assert frontier.status_code == 200
    assert frontier.json()["frontier"][0]["objectives"]["qed"] == 0.8
    assert hypervolume.status_code == 200
    assert hypervolume.json()["n_points"] == 2
    assert hypervolume.json()["hypervolume"] > 0
    assert selected.status_code == 200
    assert [row["smiles"] for row in selected.json()["selected"]] == ["CCO", "CCN"]
    assert selected.json()["selected"][0]["qed"] == 0.8


def test_pareto_merges_repeated_smiles_by_occurrence(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "run_id": "run-repeated-smiles",
                "status": "completed",
                "state": {
                    "candidates": [
                        {"canonical_smiles": "CCO"},
                        {"canonical_smiles": "CCN"},
                        {"canonical_smiles": "CCO"},
                    ],
                    "validation": {
                        "results": [
                            {
                                "canonical_smiles": "CCO",
                                "rank": 1,
                                "valid": True,
                                "pareto_optimal": True,
                                "properties": {"qed": 0.9, "sa_score": 2.0},
                            },
                            {
                                "canonical_smiles": "CCN",
                                "rank": 2,
                                "valid": True,
                                "pareto_optimal": False,
                                "properties": {"qed": 0.8, "sa_score": 2.5},
                            },
                            {
                                "canonical_smiles": "CCO",
                                "rank": 3,
                                "valid": True,
                                "pareto_optimal": True,
                                "properties": {"qed": 0.7, "sa_score": 3.0},
                            },
                        ]
                    },
                },
            }

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    frontier = client.get("/v1/pareto/run-repeated-smiles/frontier")
    hypervolume = client.get("/v1/pareto/run-repeated-smiles/hypervolume")
    selected = client.post(
        "/v1/pareto/run-repeated-smiles/select",
        json={"weights": {"qed": 1.0}, "top_k": 3},
    )

    assert frontier.status_code == 200
    assert frontier.json()["n_points"] == 2
    assert [row["smiles"] for row in frontier.json()["frontier"]] == ["CCO", "CCO"]
    assert [row["rank"] for row in frontier.json()["frontier"]] == [1, 3]
    assert [row["objectives"]["qed"] for row in frontier.json()["frontier"]] == [0.9, 0.7]
    assert hypervolume.status_code == 200
    assert hypervolume.json()["n_points"] == 3
    assert selected.status_code == 200
    assert [row["smiles"] for row in selected.json()["selected"]] == ["CCO", "CCN", "CCO"]
    assert [row["qed"] for row in selected.json()["selected"]] == [0.9, 0.8, 0.7]


def test_pareto_merges_duplicate_candidate_ids_without_extra_rows(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "run_id": "run-duplicate-candidate-ids",
                "status": "completed",
                "state": {
                    "candidates": [
                        {"candidate_id": "candidate-duplicate", "canonical_smiles": "CCO"},
                        {"candidate_id": "candidate-duplicate", "canonical_smiles": "CCO"},
                    ],
                    "validation": {
                        "results": [
                            {
                                "candidate_id": "candidate-duplicate",
                                "canonical_smiles": "CCO",
                                "rank": 1,
                                "valid": True,
                                "pareto_optimal": True,
                                "properties": {"qed": 0.9, "sa_score": 2.0},
                            },
                            {
                                "candidate_id": "candidate-duplicate",
                                "canonical_smiles": "CCO",
                                "rank": 2,
                                "valid": True,
                                "pareto_optimal": True,
                                "properties": {"qed": 0.7, "sa_score": 3.0},
                            },
                        ]
                    },
                },
            }

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    frontier = client.get("/v1/pareto/run-duplicate-candidate-ids/frontier")
    hypervolume = client.get("/v1/pareto/run-duplicate-candidate-ids/hypervolume")
    selected = client.post(
        "/v1/pareto/run-duplicate-candidate-ids/select",
        json={"weights": {"qed": 1.0}, "top_k": 3},
    )

    assert frontier.status_code == 200
    assert frontier.json()["n_points"] == 2
    assert [row["rank"] for row in frontier.json()["frontier"]] == [1, 2]
    assert hypervolume.status_code == 200
    assert hypervolume.json()["n_points"] == 2
    assert selected.status_code == 200
    assert len(selected.json()["selected"]) == 2
    assert [row["qed"] for row in selected.json()["selected"]] == [0.9, 0.7]


def test_pareto_matches_duplicate_id_by_smiles_before_occurrence(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "run_id": "run-duplicate-id-distinct-smiles",
                "status": "completed",
                "state": {
                    "candidates": [
                        {
                            "candidate_id": "candidate-duplicate",
                            "canonical_smiles": "CCO",
                            "composite_score": 1.0,
                        },
                        {
                            "candidate_id": "candidate-duplicate",
                            "canonical_smiles": "CCN",
                            "composite_score": 2.0,
                        },
                    ],
                    "validation": {
                        "results": [
                            {
                                "candidate_id": "candidate-duplicate",
                                "canonical_smiles": "CCN",
                                "rank": 2,
                                "valid": True,
                                "pareto_optimal": True,
                                "properties": {"qed": 0.8, "sa_score": 2.5},
                            },
                            {
                                "candidate_id": "candidate-duplicate",
                                "canonical_smiles": "CCO",
                                "rank": 1,
                                "valid": True,
                                "pareto_optimal": True,
                                "properties": {"qed": 0.9, "sa_score": 2.0},
                            },
                        ]
                    },
                },
            }

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    frontier = client.get("/v1/pareto/run-duplicate-id-distinct-smiles/frontier")
    hypervolume = client.get("/v1/pareto/run-duplicate-id-distinct-smiles/hypervolume")
    selected = client.post(
        "/v1/pareto/run-duplicate-id-distinct-smiles/select",
        json={"weights": {"qed": 1.0}, "top_k": 2},
    )

    assert frontier.status_code == 200
    assert [
        (row["smiles"], row["rank"], row["composite_score"]) for row in frontier.json()["frontier"]
    ] == [
        ("CCO", 1, 1.0),
        ("CCN", 2, 2.0),
    ]
    assert hypervolume.status_code == 200
    assert hypervolume.json()["n_points"] == 2
    assert selected.status_code == 200
    assert [(row["smiles"], row["composite_score"]) for row in selected.json()["selected"]] == [
        ("CCO", 1.0),
        ("CCN", 2.0),
    ]


def test_pareto_reserves_later_exact_duplicate_id_match_before_vague_occurrence(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "run_id": "run-vague-before-exact",
                "status": "completed",
                "state": {
                    "candidates": [
                        {
                            "candidate_id": "candidate-duplicate",
                            "canonical_smiles": "CCO",
                            "composite_score": 1.0,
                        },
                        {
                            "candidate_id": "candidate-duplicate",
                            "canonical_smiles": "CCN",
                            "composite_score": 2.0,
                        },
                    ],
                    "validation": {
                        "results": [
                            {
                                "candidate_id": "candidate-duplicate",
                                "rank": 2,
                                "valid": True,
                                "pareto_optimal": True,
                                "properties": {"qed": 0.8, "sa_score": 2.5},
                            },
                            {
                                "candidate_id": "candidate-duplicate",
                                "canonical_smiles": "CCO",
                                "rank": 1,
                                "valid": True,
                                "pareto_optimal": True,
                                "properties": {"qed": 0.9, "sa_score": 2.0},
                            },
                        ]
                    },
                },
            }

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    frontier = client.get("/v1/pareto/run-vague-before-exact/frontier")
    hypervolume = client.get("/v1/pareto/run-vague-before-exact/hypervolume")
    selected = client.post(
        "/v1/pareto/run-vague-before-exact/select",
        json={"weights": {"qed": 1.0}, "top_k": 2},
    )

    assert frontier.status_code == 200
    assert [
        (row["smiles"], row["rank"], row["composite_score"]) for row in frontier.json()["frontier"]
    ] == [
        ("CCO", 1, 1.0),
        ("CCN", 2, 2.0),
    ]
    assert hypervolume.status_code == 200
    assert hypervolume.json()["n_points"] == 2
    assert selected.status_code == 200
    assert [(row["smiles"], row["composite_score"]) for row in selected.json()["selected"]] == [
        ("CCO", 1.0),
        ("CCN", 2.0),
    ]


def test_pareto_keeps_unknown_candidate_id_unmatched_when_smiles_exists(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "run_id": "run-unknown-candidate-id",
                "status": "completed",
                "state": {
                    "candidates": [
                        {
                            "candidate_id": "candidate-known",
                            "canonical_smiles": "CCO",
                            "rank": 10,
                            "valid": True,
                            "pareto_optimal": True,
                            "properties": {"qed": 0.1, "sa_score": 5.0},
                        }
                    ],
                    "validation": {
                        "results": [
                            {
                                "candidate_id": "candidate-unknown",
                                "canonical_smiles": "CCO",
                                "rank": 1,
                                "valid": True,
                                "pareto_optimal": False,
                                "properties": {"qed": 0.9, "sa_score": 2.0},
                            }
                        ]
                    },
                },
            }

    class _Client:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict | None = None):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    frontier = client.get("/v1/pareto/run-unknown-candidate-id/frontier")
    hypervolume = client.get("/v1/pareto/run-unknown-candidate-id/hypervolume")
    selected = client.post(
        "/v1/pareto/run-unknown-candidate-id/select",
        json={"weights": {"qed": 1.0}, "top_k": 3},
    )

    assert frontier.status_code == 200
    assert frontier.json()["frontier"] == []
    assert hypervolume.status_code == 200
    assert hypervolume.json()["n_points"] == 0
    assert selected.status_code == 200
    assert selected.json()["selected"] == []


def test_pareto_reserves_every_duplicate_id_occurrence_before_smiles_fallback() -> None:
    from api_gateway.routers.pareto import _merge_candidate_results

    merged = _merge_candidate_results(
        [
            {"candidate_id": "candidate-duplicate", "canonical_smiles": "CCO"},
            {"candidate_id": "candidate-duplicate", "canonical_smiles": "CCO"},
            {"candidate_id": "candidate-other", "canonical_smiles": "CCO"},
        ],
        [
            {"canonical_smiles": "CCO", "rank": 3},
            {
                "candidate_id": "candidate-duplicate",
                "canonical_smiles": "CCO",
                "rank": 1,
            },
            {
                "candidate_id": "candidate-duplicate",
                "canonical_smiles": "CCO",
                "rank": 2,
            },
        ],
    )

    assert [(row["candidate_id"], row["rank"]) for row in merged] == [
        ("candidate-duplicate", 1),
        ("candidate-duplicate", 2),
        ("candidate-other", 3),
    ]


def test_pareto_does_not_reassign_excess_known_id_occurrences() -> None:
    from api_gateway.routers.pareto import _merge_candidate_results

    merged = _merge_candidate_results(
        [
            {"candidate_id": "candidate-duplicate", "canonical_smiles": "CCO"},
            {"candidate_id": "candidate-other", "canonical_smiles": "CCO"},
        ],
        [
            {
                "candidate_id": "candidate-duplicate",
                "canonical_smiles": "CCO",
                "rank": 1,
            },
            {
                "candidate_id": "candidate-duplicate",
                "canonical_smiles": "CCO",
                "rank": 99,
            },
            {"canonical_smiles": "CCO", "rank": 2},
        ],
    )

    assert [(row["candidate_id"], row["rank"]) for row in merged] == [
        ("candidate-duplicate", 1),
        ("candidate-other", 2),
        ("candidate-duplicate", 99),
    ]


def test_pareto_appends_an_unmatched_validation_row_unchanged() -> None:
    from api_gateway.routers.pareto import _merge_candidate_results

    merged = _merge_candidate_results(
        [{"candidate_id": "candidate-known", "canonical_smiles": "CCO"}],
        [
            {
                "candidate_id": "candidate-unknown",
                "canonical_smiles": "NNN",
                "rank": 4,
            }
        ],
    )

    assert merged == [
        {"candidate_id": "candidate-known", "canonical_smiles": "CCO"},
        {
            "candidate_id": "candidate-unknown",
            "canonical_smiles": "NNN",
            "rank": 4,
        },
    ]


@pytest.mark.parametrize(
    "validation_rows",
    [
        [
            {
                "canonical_smiles": "CCO",
                "rank": 2,
                "pareto_optimal": False,
            },
            {
                "candidate_id": "candidate-1",
                "canonical_smiles": "CCO",
                "rank": 1,
                "pareto_optimal": True,
            },
        ],
        [
            {
                "candidate_id": "candidate-1",
                "canonical_smiles": "CCO",
                "rank": 1,
                "pareto_optimal": True,
            },
            {
                "canonical_smiles": "CCO",
                "rank": 2,
                "pareto_optimal": False,
            },
        ],
    ],
)
def test_pareto_reserves_explicit_ids_before_smiles_fallback(
    validation_rows: list[dict],
) -> None:
    from api_gateway.routers.pareto import _merge_candidate_results

    merged = _merge_candidate_results(
        [
            {"candidate_id": "candidate-1", "canonical_smiles": "CCO"},
            {"candidate_id": "candidate-2", "canonical_smiles": "CCO"},
        ],
        validation_rows,
    )

    assert [
        (row["candidate_id"], row.get("rank"), row.get("pareto_optimal")) for row in merged
    ] == [
        ("candidate-1", 1, True),
        ("candidate-2", 2, False),
    ]
