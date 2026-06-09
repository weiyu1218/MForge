"""Repository layer for Neo4j graph operations."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from mf_core.db.repositories.graph_repo import GraphRepository

_NEO4J_ENV_VARS = ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")


def build_shared_crg_repository_from_env(
    env: Mapping[str, str] | None = None,
    driver_factory: Any = None,
) -> GraphRepository | None:
    config = env or os.environ
    configured = [name for name in _NEO4J_ENV_VARS if config.get(name)]
    if not configured:
        return None
    missing = [name for name in _NEO4J_ENV_VARS if not config.get(name)]
    if missing:
        raise RuntimeError(f"Shared CRG repository missing config: {', '.join(missing)}")
    if driver_factory is None:
        from neo4j import AsyncGraphDatabase

        driver_factory = AsyncGraphDatabase.driver
    driver = driver_factory(
        str(config["NEO4J_URI"]),
        auth=(str(config["NEO4J_USER"]), str(config["NEO4J_PASSWORD"])),
    )
    return GraphRepository(driver)


__all__ = ["GraphRepository", "build_shared_crg_repository_from_env"]
