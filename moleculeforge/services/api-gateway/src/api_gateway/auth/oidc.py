"""OIDC authentication for the API Gateway.

Supports multiple OIDC providers (Google, Azure AD, Okta, Keycloak) for
single sign-on authentication. Validates JWT tokens and extracts user
identity claims for downstream authorization.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from typing import Any, Protocol

import jwt
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)
_OIDC_SIGNING_ALGORITHMS = frozenset(
    {
        "EdDSA",
        "ES256",
        "ES384",
        "ES512",
        "PS256",
        "PS384",
        "PS512",
        "RS256",
        "RS384",
        "RS512",
    }
)


class SigningKeyClient(Protocol):
    def get_signing_key_from_jwt(self, token: str) -> jwt.PyJWK:
        """Resolve the configured signing key for a JWT."""
        ...


class TokenVerifier(Protocol):
    def verify(self, token: str, provider: dict[str, str]) -> dict[str, Any]:
        """Validate a token for one configured provider."""
        ...


class PyJWTVerifier:
    """Validate OIDC JWTs with signing keys from configured JWKS endpoints."""

    def __init__(
        self,
        *,
        jwks_client_factory: Callable[[str], SigningKeyClient] = jwt.PyJWKClient,
    ) -> None:
        self._jwks_client_factory = jwks_client_factory
        self._jwks_clients: dict[str, SigningKeyClient] = {}

    def verify(self, token: str, provider: dict[str, str]) -> dict[str, Any]:
        header = jwt.get_unverified_header(token)
        algorithm = header.get("alg")
        if algorithm not in _OIDC_SIGNING_ALGORITHMS:
            raise jwt.InvalidAlgorithmError("Unsupported OIDC signing algorithm")

        jwks_uri = provider["jwks_uri"]
        jwks_client = self._jwks_clients.get(jwks_uri)
        if jwks_client is None:
            jwks_client = self._jwks_client_factory(jwks_uri)
            self._jwks_clients[jwks_uri] = jwks_client
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        if signing_key.algorithm_name != algorithm:
            raise jwt.InvalidAlgorithmError("JWT and JWK algorithms do not match")

        return jwt.decode(
            token,
            key=signing_key.key,
            algorithms=[algorithm],
            audience=provider["client_id"],
            issuer=provider["issuer"],
            options={"require": ["aud", "exp", "iss", "sub"]},
        )


class OIDCAuth:
    """OpenID Connect authenticator for API Gateway.

    Validates JWT tokens against configured OIDC providers and
    returns user identity information. Supports multi-tenant
    deployments with per-tenant provider configurations.
    """

    def __init__(
        self,
        *,
        providers: dict[str, dict[str, str]] | None = None,
        verifier: TokenVerifier | None = None,
    ) -> None:
        self.providers: dict[str, dict[str, str]] = {}
        for name, provider in (providers or {}).items():
            self.register_provider(
                name=name,
                issuer=provider["issuer"],
                client_id=provider["client_id"],
                jwks_uri=provider["jwks_uri"],
            )
        self._verifier = verifier or PyJWTVerifier()

    @classmethod
    def from_environment(
        cls,
        *,
        verifier: TokenVerifier | None = None,
    ) -> OIDCAuth:
        provider = {
            "issuer": os.environ.get("OIDC_ISSUER", "").strip(),
            "client_id": os.environ.get("OIDC_AUDIENCE", "").strip(),
            "jwks_uri": os.environ.get("OIDC_JWKS_URI", "").strip(),
        }
        configured_values = [bool(value) for value in provider.values()]
        if not any(configured_values):
            return cls(verifier=verifier)
        if not all(configured_values):
            raise ValueError(
                "OIDC_ISSUER, OIDC_AUDIENCE, and OIDC_JWKS_URI must be configured together"
            )
        return cls(providers={"environment": provider}, verifier=verifier)

    def register_provider(
        self,
        name: str,
        issuer: str,
        client_id: str,
        jwks_uri: str | None = None,
    ) -> None:
        """Register an OIDC provider configuration.

        Args:
            name: Provider name (e.g., 'google', 'azure', 'keycloak').
            issuer: OIDC issuer URL.
            client_id: OIDC client ID.
            jwks_uri: JWKS endpoint URL.
        """
        if not name or not issuer or not client_id or not jwks_uri:
            raise ValueError("OIDC provider name, issuer, client_id, and jwks_uri are required")
        self.providers[name] = {
            "issuer": issuer,
            "client_id": client_id,
            "jwks_uri": jwks_uri,
        }

    async def authenticate(self, request: Request) -> dict:
        """Authenticate a request and return user identity.

        Args:
            request: The incoming FastAPI request.

        Returns:
            Dict with user identity claims (sub, email, name, etc.).

        Raises:
            HTTPException: If authentication fails.
        """
        credentials: HTTPAuthorizationCredentials | None = await security(request)
        if credentials is None:
            return {"anonymous": True}
        if not self.providers:
            raise HTTPException(
                status_code=503,
                detail="OIDC provider is not configured",
            )

        token = credentials.credentials

        try:
            provider = self._provider_for_token(token)
        except jwt.InvalidTokenError:
            provider = None
        if provider is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        try:
            claims = await asyncio.to_thread(self._verifier.verify, token, provider)
        except jwt.exceptions.PyJWKClientConnectionError as exc:
            raise HTTPException(
                status_code=503,
                detail="OIDC verification is unavailable",
            ) from exc
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=401, detail="Invalid token") from exc

        return {
            "sub": claims["sub"],
            "email": claims.get("email", ""),
            "name": claims.get("name", ""),
            "preferred_username": claims.get("preferred_username", ""),
            "roles": claims.get("roles", []),
            "token": token[:10] + "...",
        }

    def _provider_for_token(self, token: str) -> dict[str, str] | None:
        claims = jwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_aud": False,
                "verify_iss": False,
            },
        )
        issuer = claims.get("iss")
        if not isinstance(issuer, str):
            return None
        return next(
            (provider for provider in self.providers.values() if provider["issuer"] == issuer),
            None,
        )

    def require_roles(self, *roles: str):
        """Return a dependency that enforces specific roles.

        Usage:
            @router.get("/admin")
            async def admin_endpoint(user=Depends(auth.require_roles("admin"))):
                ...
        """

        async def role_checker(request: Request) -> dict:
            user = await self.authenticate(request)
            if user.get("anonymous"):
                raise HTTPException(status_code=401, detail="Authentication required")
            user_roles = user.get("roles", [])
            if not any(r in user_roles for r in roles):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return user

        return role_checker
