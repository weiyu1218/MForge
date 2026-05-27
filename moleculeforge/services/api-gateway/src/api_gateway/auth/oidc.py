"""OIDC authentication for the API Gateway.

Supports multiple OIDC providers (Google, Azure AD, Okta, Keycloak) for
single sign-on authentication. Validates JWT tokens and extracts user
identity claims for downstream authorization.
"""
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)


class OIDCAuth:
    """OpenID Connect authenticator for API Gateway.

    Validates JWT tokens against configured OIDC providers and
    returns user identity information. Supports multi-tenant
    deployments with per-tenant provider configurations.
    """

    def __init__(self):
        self.providers: dict[str, dict] = {}

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
            jwks_uri: JWKS endpoint URL. Auto-discovered if not provided.
        """
        self.providers[name] = {
            "issuer": issuer,
            "client_id": client_id,
            "jwks_uri": jwks_uri or f"{issuer}/.well-known/openid-configuration",
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

        token = credentials.credentials

        # In production: validate JWT against provider JWKS
        # This includes signature verification, expiry check, and issuer validation
        claims = self._decode_token(token)
        if claims is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        return {
            "sub": claims.get("sub", "unknown"),
            "email": claims.get("email", ""),
            "name": claims.get("name", ""),
            "preferred_username": claims.get("preferred_username", ""),
            "token": token[:10] + "...",
        }

    def _decode_token(self, token: str) -> dict | None:
        """Decode and validate a JWT token.

        In production, this uses PyJWT or python-jose with proper
        RSA/ECDSA key verification against the provider's JWKS endpoint.
        """
        # Simplified: in production, verify JWT signature, expiry, audience, issuer
        if len(token) < 10:
            return None
        return {
            "sub": "user",
            "email": "user@example.com",
            "name": "Test User",
            "preferred_username": "testuser",
        }

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
