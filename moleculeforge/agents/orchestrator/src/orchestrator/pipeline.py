"""Compatibility client for the canonical Orchestrator run API."""
from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx


class ReasoningPipeline:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (
            base_url
            or os.environ.get(
                "ORCHESTRATOR_SVC_URL",
                "http://orchestrator-svc:8011",
            )
        ).rstrip("/")

    async def submit(
        self,
        intent: str,
        *,
        workflow_scope: str,
        validation_passed: bool,
        max_refinements: int,
        project_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            **dict(extra or {}),
            "intent": intent,
            "workflow_scope": workflow_scope,
            "validation_passed": validation_passed,
            "max_refinements": max_refinements,
            "project_id": project_id,
        }
        response = await self._post("/v1/orchestrator/design", payload)
        return str(response["run_id"])

    async def get(self, run_id: str) -> dict[str, Any]:
        return await self._get(f"/v1/orchestrator/runs/{run_id}")

    async def list_runs(
        self,
        *,
        page_size: int,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        return await self._get(
            "/v1/orchestrator/runs",
            params={"page_size": page_size, "page_token": page_token},
        )

    async def subscribe(self, run_id: str) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        asyncio.create_task(
            self._poll_events(run_id, queue),
            name=f"orchestrator-subscription-{run_id}",
        )
        return queue

    async def _poll_events(
        self,
        run_id: str,
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        after_step = -1
        while True:
            event_page = await self._get(
                f"/v1/orchestrator/runs/{run_id}/events",
                params={"after_step": after_step},
            )
            for event in event_page.get("events", []):
                after_step = int(event["step_index"])
                await queue.put({"type": "step", **event})
            snapshot = await self.get(run_id)
            if snapshot.get("status") in {
                "completed",
                "rejected",
                "failed",
                "interrupted",
            }:
                await queue.put({"type": "done", "run_id": run_id})
                return
            await asyncio.sleep(0.5)

    async def _get(
        self,
        path: str,
        *,
        params: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.base_url}{path}", params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("orchestrator service returned invalid response")
        return payload

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}{path}", json=payload)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError("orchestrator service returned invalid response")
        return result


_default_pipeline: ReasoningPipeline | None = None


def get_pipeline() -> ReasoningPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = ReasoningPipeline()
    return _default_pipeline
