"""Integration tests for DKI Redis message bus."""

from __future__ import annotations

import asyncio
import json
import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_redis_publish_subscribe_preserves_trace_id() -> None:
    if not (os.environ.get("REDIS_HOST") or os.environ.get("REDIS_URL")):
        pytest.skip("REDIS_HOST or REDIS_URL is required for Redis integration tests")
    from mf_agents.messaging.redis_bus import RedisBus

    bus = RedisBus(connect_timeout=1.0, allow_fallback=False)
    received: list[dict] = []

    async def on_message(message):
        received.append(json.loads(message["data"].decode("utf-8")))

    await bus.connect()
    await bus.subscribe("mf.integration.trace", on_message)
    await bus.publish(
        "mf.integration.trace",
        json.dumps({"trace_id": "trace-redis", "event": "created"}).encode("utf-8"),
    )
    for _ in range(20):
        if received:
            break
        await asyncio.sleep(0.05)
    await bus.close()

    assert received == [{"trace_id": "trace-redis", "event": "created"}]
