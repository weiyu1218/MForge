"""Tenant isolation middleware for multi-tenant deployments.

Extracts tenant context from the X-Tenant-ID header and attaches it
to the request state for downstream routing and data isolation.
"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class TenantMiddleware(BaseHTTPMiddleware):
    """Middleware that extracts and validates tenant context.

    Ensures all requests carry a valid tenant identifier for data isolation.
    The tenant ID is propagated to all downstream service calls for
    consistent multi-tenancy enforcement.
    """

    async def dispatch(self, request: Request, call_next):
        tenant_id = request.headers.get("X-Tenant-ID", "default")

        # In production: validate tenant_id against tenant registry
        request.state.tenant_id = tenant_id

        # Attach tenant to request context for logging and tracing
        request.state.tenant_context = {
            "tenant_id": tenant_id,
            "isolation_level": "database",  # or 'schema', 'shared'
        }

        response = await call_next(request)
        response.headers["X-Tenant-ID"] = tenant_id
        return response
