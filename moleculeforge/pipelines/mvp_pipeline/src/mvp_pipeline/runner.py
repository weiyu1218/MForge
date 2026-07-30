"""MVP pipeline client for the canonical Orchestrator API."""
from __future__ import annotations

import asyncio
import os

import httpx

_TERMINAL_STATUSES = {"completed", "rejected", "failed", "interrupted"}


async def run_pipeline(
    nl_query: str,
    n_samples: int = 100,
    seed: int | None = None,
    *,
    workflow_scope: str | None = None,
    validation_passed: bool | None = None,
    max_refinements: int | None = None,
) -> dict:
    if not workflow_scope:
        raise ValueError("workflow_scope is required")
    if validation_passed is None:
        raise ValueError("validation_passed is required")
    if max_refinements is None:
        raise ValueError("max_refinements is required")
    if max_refinements < 0:
        raise ValueError("max_refinements must be non-negative")
    base_url = os.environ.get(
        "ORCHESTRATOR_SVC_URL",
        "http://orchestrator-svc:8011",
    ).rstrip("/")
    request = {
        "nl_input": nl_query,
        "n_samples": n_samples,
        "seed": seed,
        "workflow_scope": workflow_scope,
        "validation_passed": validation_passed,
        "max_refinements": max_refinements,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        submitted = await client.post(
            f"{base_url}/v1/orchestrator/design",
            json=request,
        )
        submitted.raise_for_status()
        submission = submitted.json()
        run_id = str(submission["run_id"])
        while True:
            response = await client.get(
                f"{base_url}/v1/orchestrator/runs/{run_id}"
            )
            response.raise_for_status()
            snapshot = response.json()
            if snapshot.get("status") in _TERMINAL_STATUSES:
                return snapshot
            await asyncio.sleep(0.5)


def run_pipeline_sync(
    nl_query: str,
    n_samples: int = 100,
    seed: int | None = None,
    *,
    workflow_scope: str | None = None,
    validation_passed: bool | None = None,
    max_refinements: int | None = None,
) -> dict:
    return asyncio.run(
        run_pipeline(
            nl_query,
            n_samples=n_samples,
            seed=seed,
            workflow_scope=workflow_scope,
            validation_passed=validation_passed,
            max_refinements=max_refinements,
        )
    )
