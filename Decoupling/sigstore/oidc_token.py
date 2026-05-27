"""OIDC token acquisition for headless (non-interactive) environments.

Supports three strategies:
  1. Direct environment variable (SIGSTORE_ID_TOKEN)
  2. GitHub Actions OIDC token exchange
  3. Fallback to interactive browser flow (dev only)
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


class TokenProvider(Protocol):
    """Any callable that returns a raw OIDC JWT string."""

    def __call__(self) -> str: ...


# ---------------------------------------------------------------------------
# Strategy 1: environment variable
# ---------------------------------------------------------------------------

def token_from_env() -> str:
    """Read SIGSTORE_ID_TOKEN directly from the process environment."""
    token = os.environ.get("SIGSTORE_ID_TOKEN")
    if not token:
        raise RuntimeError(
            "SIGSTORE_ID_TOKEN is not set. "
            "Export a valid OIDC JWT before calling sign operations."
        )
    logger.info("OIDC token acquired from SIGSTORE_ID_TOKEN (len=%d)", len(token))
    return token


# ---------------------------------------------------------------------------
# Strategy 2: GitHub Actions OIDC
# ---------------------------------------------------------------------------

def token_from_github_actions(audience: str = "sigstore") -> str:
    """Exchange the GitHub Actions runtime token for an OIDC JWT.

    Requires the calling workflow to have:
        permissions:
          id-token: write

    The runtime provides:
      - ACTIONS_ID_TOKEN_REQUEST_URL
      - ACTIONS_ID_TOKEN_REQUEST_TOKEN
    """
    request_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    request_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")

    if not request_url or not request_token:
        raise RuntimeError(
            "GitHub Actions OIDC variables not found. "
            "Ensure 'id-token: write' permission is set in the workflow."
        )

    resp = httpx.get(
        request_url,
        params={"audience": audience},
        headers={"Authorization": f"bearer {request_token}"},
        timeout=10,
    )
    resp.raise_for_status()

    token = resp.json().get("value")
    if not token:
        raise RuntimeError("GitHub Actions OIDC response did not contain a token.")

    logger.info("OIDC token acquired from GitHub Actions (audience=%s)", audience)
    return token


# ---------------------------------------------------------------------------
# Strategy 3: interactive browser flow (passthrough — sigstore handles it)
# ---------------------------------------------------------------------------

def token_interactive() -> str:
    """Placeholder — sigstore-python handles browser OAuth internally
    when no SIGSTORE_ID_TOKEN is set and OIDC_STRATEGY is 'interactive'.

    We return an empty string so that sigstore's default provider kicks in.
    """
    logger.info("OIDC: delegating to sigstore's interactive browser flow")
    return ""  # sigstore will use its built-in OAuth provider


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, TokenProvider] = {
    "env": token_from_env,
    "github": token_from_github_actions,
    "interactive": token_interactive,
}


def resolve_token(strategy: str = "env") -> str:
    """Resolve an OIDC token using the specified strategy.

    Parameters
    ----------
    strategy : str
        One of "env", "github", "interactive".

    Returns
    -------
    str
        The OIDC JWT.  Empty string means sigstore should use its own
        provider (interactive browser flow).
    """
    provider = STRATEGIES.get(strategy)
    if provider is None:
        raise ValueError(
            f"Unknown OIDC strategy '{strategy}'. "
            f"Available: {list(STRATEGIES.keys())}"
        )
    return provider()
