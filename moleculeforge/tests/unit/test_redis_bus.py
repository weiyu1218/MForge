"""Redis message bus publish/subscribe behavior."""

from __future__ import annotations

import json

import pytest


@pytest.mark.asyncio
async def test_redis_bus_fallback_preserves_trace_id() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    bus = RedisBus(url="redis://127.0.0.1:1/0", connect_timeout=0.1)
    received = []

    async def on_message(message):
        received.append(json.loads(message["data"].decode("utf-8")))

    await bus.connect()
    await bus.subscribe("mf.trace", on_message)
    await bus.publish(
        "mf.trace",
        json.dumps({"trace_id": "trace-001", "event": "created"}).encode("utf-8"),
    )

    assert received == [{"trace_id": "trace-001", "event": "created"}]
    await bus.close()


@pytest.mark.asyncio
async def test_redis_bus_fallback_supports_agent_callback_signature() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    bus = RedisBus(url="redis://127.0.0.1:1/0", connect_timeout=0.1)
    received = []

    async def on_message(subject, payload, reply_to=""):
        received.append((subject, json.loads(payload.decode("utf-8")), reply_to))

    await bus.connect()
    await bus.subscribe("mf.agent", on_message)
    await bus.publish(
        "mf.agent",
        json.dumps({"trace_id": "trace-002", "event": "handled"}).encode("utf-8"),
    )

    assert received == [
        ("mf.agent", {"trace_id": "trace-002", "event": "handled"}, "")
    ]
    await bus.close()
