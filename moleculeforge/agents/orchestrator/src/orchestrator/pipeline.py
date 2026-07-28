"""Compatibility client for the canonical Orchestrator run API."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class _Subscription:
    queue: asyncio.Queue[dict[str, Any]]
    terminal_sent: bool = False


class ReasoningPipeline:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (
            base_url
            or os.environ.get(
                "ORCHESTRATOR_SVC_URL",
                "http://orchestrator-svc:8011",
            )
        ).rstrip("/")
        self._subscription_tasks: dict[str, asyncio.Task[None]] = {}
        self._subscriptions: dict[str, _Subscription] = {}
        self._subscription_lock = asyncio.Lock()
        self._closed = False

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
        }
        payload.pop("project_id", None)
        if project_id is not None:
            payload["project_id"] = project_id
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
        params: dict[str, str | int] = {"page_size": page_size}
        if page_token is not None:
            params["page_token"] = page_token
        return await self._get(
            "/v1/orchestrator/runs",
            params=params,
        )

    async def subscribe(self, run_id: str) -> asyncio.Queue[dict[str, Any]]:
        async with self._subscription_lock:
            if self._closed:
                raise RuntimeError("reasoning pipeline is closed")
            await self._unsubscribe(run_id)
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            subscription = _Subscription(queue=queue)
            task = asyncio.create_task(
                self._poll_events(run_id, subscription),
                name=f"orchestrator-subscription-{run_id}",
            )
            self._subscription_tasks[run_id] = task
            self._subscriptions[run_id] = subscription
            task.add_done_callback(
                lambda completed, key=run_id, target=subscription: self._finish_subscription(
                    key,
                    target,
                    completed,
                )
            )
        return queue

    async def unsubscribe(self, run_id: str) -> None:
        async with self._subscription_lock:
            await self._unsubscribe(run_id)

    async def _unsubscribe(self, run_id: str) -> None:
        task = self._subscription_tasks.pop(run_id, None)
        subscription = self._subscriptions.pop(run_id, None)
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if subscription is not None:
            self._finish_queue(run_id, subscription)

    async def aclose(self) -> None:
        async with self._subscription_lock:
            self._closed = True
            subscriptions = [
                (run_id, task, self._subscriptions.get(run_id))
                for run_id, task in self._subscription_tasks.items()
            ]
            self._subscription_tasks.clear()
            self._subscriptions.clear()
            for _, task, _ in subscriptions:
                task.cancel()
            if subscriptions:
                await asyncio.gather(
                    *(task for _, task, _ in subscriptions),
                    return_exceptions=True,
                )
            for run_id, _, subscription in subscriptions:
                if subscription is not None:
                    self._finish_queue(run_id, subscription)

    def _finish_subscription(
        self,
        run_id: str,
        subscription: _Subscription,
        task: asyncio.Task[None],
    ) -> None:
        if not task.cancelled():
            exception = task.exception()
            if exception is not None:
                self._finish_queue(
                    run_id,
                    subscription,
                    {
                        "type": "done",
                        "run_id": run_id,
                        "status": "failed",
                        "error_type": type(exception).__name__,
                        "error_message": str(exception),
                    },
                )
        if self._subscription_tasks.get(run_id) is task:
            self._subscription_tasks.pop(run_id, None)
            self._subscriptions.pop(run_id, None)

    @staticmethod
    def _finish_queue(
        run_id: str,
        subscription: _Subscription,
        message: dict[str, Any] | None = None,
    ) -> None:
        if subscription.terminal_sent:
            return
        subscription.terminal_sent = True
        subscription.queue.put_nowait(message or {"type": "done", "run_id": run_id})

    async def _poll_events(
        self,
        run_id: str,
        subscription: _Subscription,
    ) -> None:
        after_step = -1
        while True:
            event_page = await self._get(
                f"/v1/orchestrator/runs/{run_id}/events",
                params={"after_step": after_step},
            )
            for event in event_page.get("events", []):
                after_step = int(event["step_index"])
                await subscription.queue.put({"type": "step", **event})
            snapshot = await self.get(run_id)
            if snapshot.get("status") in {
                "completed",
                "rejected",
                "failed",
                "interrupted",
            }:
                self._finish_queue(run_id, subscription)
                return
            await asyncio.sleep(0.5)

    async def _get(
        self,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
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
