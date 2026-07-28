"""Redis message bus publish/subscribe behavior."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

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
    await asyncio.wait_for(_wait_until(lambda: bool(received)), timeout=0.2)

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
    await asyncio.wait_for(_wait_until(lambda: bool(received)), timeout=0.2)

    assert received == [("mf.agent", {"trace_id": "trace-002", "event": "handled"}, "")]
    await bus.close()


@pytest.mark.asyncio
async def test_inmemory_publish_does_not_wait_for_long_callback() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus

    bus = InMemoryBus()
    await bus.connect()
    callback_started = asyncio.Event()
    callback_release = asyncio.Event()
    fast_callback_completed = asyncio.Event()

    async def blocking_callback(message):
        callback_started.set()
        await callback_release.wait()

    async def fast_callback(message):
        fast_callback_completed.set()

    await bus.subscribe("mf.concurrent", blocking_callback)
    await bus.subscribe("mf.concurrent", fast_callback)

    try:
        await asyncio.wait_for(bus.publish("mf.concurrent", b"payload"), timeout=0.02)
        await asyncio.wait_for(callback_started.wait(), timeout=0.02)
        await asyncio.wait_for(fast_callback_completed.wait(), timeout=0.02)
    finally:
        callback_release.set()
        await bus.close()


@pytest.mark.asyncio
async def test_inmemory_callback_failure_does_not_stop_other_callbacks() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus

    bus = InMemoryBus()
    await bus.connect()
    completed = asyncio.Event()

    async def failing_callback(message):
        raise RuntimeError("callback failed")

    async def healthy_callback(message):
        completed.set()

    await bus.subscribe("mf.errors", failing_callback)
    await bus.subscribe("mf.errors", healthy_callback)

    await bus.publish("mf.errors", b"payload")
    await asyncio.wait_for(completed.wait(), timeout=0.05)
    await asyncio.wait_for(_wait_until(lambda: bus.callback_task_count == 0), timeout=0.05)
    await bus.close()


@pytest.mark.asyncio
async def test_bus_close_cancels_managed_callback_tasks() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus

    bus = InMemoryBus()
    await bus.connect()
    started = asyncio.Event()

    async def hanging_callback(message):
        started.set()
        await asyncio.Event().wait()

    await bus.subscribe("mf.close", hanging_callback)
    publish_task = asyncio.create_task(bus.publish("mf.close", b"payload"))
    await asyncio.wait_for(started.wait(), timeout=0.05)

    await asyncio.wait_for(bus.close(), timeout=0.05)

    try:
        assert publish_task.done()
        assert bus.callback_task_count == 0
    finally:
        if not publish_task.done():
            publish_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await publish_task


@pytest.mark.asyncio
async def test_concurrent_redis_connect_keeps_one_resource_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.messaging.redis_bus import RedisBus
    from redis import asyncio as redis_async

    class PubSub:
        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1

    class Client:
        def __init__(self) -> None:
            self.ping_started = asyncio.Event()
            self.release_ping = asyncio.Event()
            self.pubsub_calls = 0
            self.pubsub_resource = PubSub()
            self.close_calls = 0

        async def ping(self) -> None:
            self.ping_started.set()
            await self.release_ping.wait()

        def pubsub(self) -> PubSub:
            self.pubsub_calls += 1
            return self.pubsub_resource

        async def aclose(self) -> None:
            self.close_calls += 1

    clients: list[Client] = []

    def from_url(url: str) -> Client:
        client = Client()
        clients.append(client)
        return client

    monkeypatch.setattr(redis_async, "from_url", from_url)
    bus = RedisBus(allow_fallback=False)
    first = asyncio.create_task(bus.connect())
    while not clients:
        await asyncio.sleep(0)
    await asyncio.wait_for(clients[0].ping_started.wait(), timeout=0.05)
    second = asyncio.create_task(bus.connect())
    await asyncio.sleep(0)
    for client in clients:
        client.release_ping.set()
    await asyncio.gather(first, second)
    retained_client = bus._client
    retained_pubsub = bus._pubsub

    await bus.close()

    assert len(clients) == 1
    assert retained_client is clients[0]
    assert retained_pubsub is clients[0].pubsub_resource
    assert clients[0].pubsub_calls == 1
    assert clients[0].pubsub_resource.close_calls == 1
    assert clients[0].close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_redis_connect_closes_local_client_and_can_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.messaging.redis_bus import RedisBus
    from redis import asyncio as redis_async

    class PubSub:
        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1

    class Client:
        def __init__(self, *, block_ping: bool) -> None:
            self.block_ping = block_ping
            self.ping_started = asyncio.Event()
            self.pubsub_resource = PubSub()
            self.close_calls = 0

        async def ping(self) -> None:
            self.ping_started.set()
            if self.block_ping:
                await asyncio.Event().wait()

        def pubsub(self) -> PubSub:
            return self.pubsub_resource

        async def aclose(self) -> None:
            self.close_calls += 1

    clients = [Client(block_ping=True), Client(block_ping=False)]

    def from_url(url: str) -> Client:
        return clients.pop(0)

    monkeypatch.setattr(redis_async, "from_url", from_url)
    bus = RedisBus(allow_fallback=False)
    first_client = clients[0]
    connect_task = asyncio.create_task(bus.connect())
    await asyncio.wait_for(first_client.ping_started.wait(), timeout=0.05)

    connect_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await connect_task
    cancelled_state = (
        first_client.close_calls,
        bus._client,
        bus._pubsub,
        bus._fallback,
    )

    second_client = clients[0]
    await bus.connect()
    await bus.close()

    assert cancelled_state == (1, None, None, None)
    assert second_client.pubsub_resource.close_calls == 1
    assert second_client.close_calls == 1


@pytest.mark.asyncio
async def test_redis_pubsub_initialization_failure_closes_client_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.messaging.redis_bus import RedisBus
    from redis import asyncio as redis_async

    class PubSub:
        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1

    class Client:
        def __init__(self, *, fail_pubsub: bool) -> None:
            self.fail_pubsub = fail_pubsub
            self.pubsub_resource = PubSub()
            self.close_calls = 0

        async def ping(self) -> None:
            return None

        def pubsub(self) -> PubSub:
            if self.fail_pubsub:
                raise RuntimeError("pubsub initialization failed")
            return self.pubsub_resource

        async def aclose(self) -> None:
            self.close_calls += 1

    clients = [Client(fail_pubsub=True), Client(fail_pubsub=False)]

    def from_url(url: str) -> Client:
        return clients.pop(0)

    monkeypatch.setattr(redis_async, "from_url", from_url)
    bus = RedisBus(allow_fallback=False)
    first_client = clients[0]

    with pytest.raises(RuntimeError, match="pubsub initialization failed"):
        await bus.connect()
    failed_state = (
        first_client.close_calls,
        bus._client,
        bus._pubsub,
        bus._fallback,
    )

    second_client = clients[0]
    await bus.connect()
    await bus.close()

    assert failed_state == (1, None, None, None)
    assert second_client.pubsub_resource.close_calls == 1
    assert second_client.close_calls == 1


@pytest.mark.asyncio
async def test_inmemory_request_timeout_covers_subscribe() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus

    class BlockingSubscribeBus(InMemoryBus):
        async def subscribe(self, subject, cb):
            await asyncio.Event().wait()

    bus = BlockingSubscribeBus()
    await bus.connect()
    request_task = asyncio.create_task(bus.request("mf.request", b"payload", timeout=0.01))

    await asyncio.sleep(0.03)

    assert request_task.done()
    with pytest.raises(TimeoutError):
        await request_task
    assert bus.callback_count == 0
    await bus.close()


@pytest.mark.asyncio
async def test_inmemory_request_timeout_does_not_wait_for_unsubscribe() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus

    class BlockingUnsubscribeBus(InMemoryBus):
        async def unsubscribe(self, subscription):
            await super().unsubscribe(subscription)
            await asyncio.Event().wait()

    bus = BlockingUnsubscribeBus()
    await bus.connect()
    request_task = asyncio.create_task(bus.request("mf.request", b"payload", timeout=0.01))

    await asyncio.sleep(0.03)

    assert request_task.done()
    with pytest.raises(TimeoutError):
        await request_task
    assert bus.callback_count == 0
    await bus.close()


@pytest.mark.asyncio
async def test_request_deadline_survives_unsubscribe_cancellation_suppression() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus

    class CancellationResistantCleanupBus(InMemoryBus):
        def __init__(self) -> None:
            super().__init__()
            self.release_cleanup = asyncio.Event()

        def discard_subscription(self, subscription):
            super().discard_subscription(subscription)
            return True

        async def _unsubscribe_subject(self, subject):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await self.release_cleanup.wait()

        async def publish(self, subject, payload):
            if subject == "mf.request":
                reply_to = next(
                    existing for existing in self._callbacks if existing.startswith("_reply.")
                )
                await super().publish(reply_to, b"response")
                return
            await super().publish(subject, payload)

    bus = CancellationResistantCleanupBus()
    await bus.connect()
    request_task = asyncio.create_task(bus.request("mf.request", b"payload", timeout=0.01))

    try:
        await asyncio.sleep(0.03)

        assert request_task.done()
        assert await request_task == b"response"
        assert bus.callback_count == 0
    finally:
        bus.release_cleanup.set()
        if not request_task.done():
            await asyncio.wait_for(request_task, timeout=0.05)
        await bus.close()


@pytest.mark.asyncio
async def test_redis_subscribe_waits_for_server_ack() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.subject = ""
            self.command_sent = asyncio.Event()
            self.release_ack = asyncio.Event()
            self.closed = asyncio.Event()

        async def subscribe(self, subject):
            self.subject = subject
            self.command_sent.set()

        async def listen(self):
            await self.release_ack.wait()
            yield {"type": "subscribe", "channel": self.subject, "data": 1}
            await self.closed.wait()

        async def aclose(self):
            self.closed.set()

    class Client:
        async def aclose(self):
            return None

    bus = RedisBus(allow_fallback=False)
    pubsub = PubSub()
    bus._client = Client()
    bus._pubsub = pubsub
    subscribe_task = asyncio.create_task(bus.subscribe("mf.ack", lambda message: None))
    await asyncio.wait_for(pubsub.command_sent.wait(), timeout=0.05)
    await asyncio.sleep(0)

    assert subscribe_task.done() is False

    pubsub.release_ack.set()
    subscription = await asyncio.wait_for(subscribe_task, timeout=0.05)
    assert subscription.subject == "mf.ack"
    await bus.close()


@pytest.mark.asyncio
async def test_redis_request_publishes_only_after_reply_subscription_ack() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.subject = ""
            self.command_sent = asyncio.Event()
            self.release_ack = asyncio.Event()
            self.messages: asyncio.Queue[dict] = asyncio.Queue()
            self.closed = asyncio.Event()

        async def subscribe(self, subject):
            self.subject = subject
            self.command_sent.set()

        async def listen(self):
            await self.release_ack.wait()
            yield {"type": "subscribe", "channel": self.subject, "data": 1}
            while not self.closed.is_set():
                yield await self.messages.get()

        async def unsubscribe(self, subject):
            return None

        async def aclose(self):
            self.closed.set()

    class Client:
        def __init__(self, pubsub: PubSub) -> None:
            self.pubsub = pubsub
            self.publish_called = asyncio.Event()

        async def publish(self, subject, payload):
            self.publish_called.set()
            await self.pubsub.messages.put(
                {
                    "type": "message",
                    "channel": self.pubsub.subject,
                    "data": b"response",
                }
            )

        async def aclose(self):
            return None

    bus = RedisBus(allow_fallback=False)
    pubsub = PubSub()
    client = Client(pubsub)
    bus._client = client
    bus._pubsub = pubsub
    request_task = asyncio.create_task(bus.request("mf.request", b"payload", timeout=0.05))
    await asyncio.wait_for(pubsub.command_sent.wait(), timeout=0.05)
    await asyncio.sleep(0)

    assert client.publish_called.is_set() is False

    pubsub.release_ack.set()
    assert await asyncio.wait_for(request_task, timeout=0.05) == b"response"
    assert client.publish_called.is_set() is True
    await bus.close()


@pytest.mark.asyncio
async def test_redis_subscribe_ack_timeout_rolls_back_and_is_retryable() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.subjects: asyncio.Queue[str] = asyncio.Queue()
            self.release_ack = asyncio.Event()
            self.closed = asyncio.Event()

        async def subscribe(self, subject):
            await self.subjects.put(subject)

        async def listen(self):
            while not self.closed.is_set():
                subject = await self.subjects.get()
                await self.release_ack.wait()
                yield {"type": "subscribe", "channel": subject, "data": 1}

        async def unsubscribe(self, subject):
            return None

        async def aclose(self):
            self.closed.set()

    class Client:
        async def aclose(self):
            return None

    bus = RedisBus(connect_timeout=0.01, allow_fallback=False)
    pubsub = PubSub()
    bus._client = Client()
    bus._pubsub = pubsub

    with pytest.raises(TimeoutError):
        await bus.subscribe("mf.retry-ack", lambda message: None)

    assert bus.callback_count == 0
    assert bus._subscribe_acks == {}

    pubsub.release_ack.set()
    subscription = await asyncio.wait_for(
        bus.subscribe("mf.retry-ack", lambda message: None),
        timeout=0.05,
    )
    assert subscription.subject == "mf.retry-ack"
    assert bus.callback_count == 1
    await bus.close()


@pytest.mark.asyncio
async def test_redis_late_subscribe_ack_triggers_orphan_cleanup() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.subjects: asyncio.Queue[str] = asyncio.Queue()
            self.release_ack = asyncio.Event()
            self.unsubscribe_called = asyncio.Event()
            self.closed = asyncio.Event()

        async def subscribe(self, subject):
            await self.subjects.put(subject)

        async def listen(self):
            while not self.closed.is_set():
                subject = await self.subjects.get()
                if subject == "mf.orphan":
                    await self.release_ack.wait()
                yield {"type": "subscribe", "channel": subject, "data": 1}

        async def unsubscribe(self, *subjects):
            self.unsubscribe_called.set()

        async def aclose(self):
            self.closed.set()

    class Client:
        async def aclose(self):
            return None

    bus = RedisBus(connect_timeout=0.01, allow_fallback=False)
    pubsub = PubSub()
    bus._client = Client()
    bus._pubsub = pubsub

    with pytest.raises(TimeoutError):
        await bus.subscribe("mf.orphan", lambda message: None)

    assert bus.callback_count == 0
    assert bus._subscribe_acks == {}

    pubsub.release_ack.set()
    await asyncio.wait_for(
        _wait_until(lambda: "mf.orphan" in bus._deferred_unsubscribe_subjects),
        timeout=0.05,
    )
    subscription = await asyncio.wait_for(
        bus.subscribe("mf.next", lambda message: None),
        timeout=0.05,
    )
    assert subscription.subject == "mf.next"
    assert bus._deferred_unsubscribe_subjects == set()
    assert pubsub.unsubscribe_called.is_set()
    await bus.close()


@pytest.mark.asyncio
async def test_redis_cancel_after_subscribe_ack_schedules_remote_unsubscribe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.messaging import redis_bus

    class PubSub:
        def __init__(self) -> None:
            self.subjects: asyncio.Queue[str] = asyncio.Queue()
            self.unsubscribe_called = asyncio.Event()
            self.closed = asyncio.Event()

        async def subscribe(self, subject):
            await self.subjects.put(subject)

        async def listen(self):
            while not self.closed.is_set():
                subject = await self.subjects.get()
                yield {"type": "subscribe", "channel": subject, "data": 1}

        async def unsubscribe(self, *subjects):
            self.unsubscribe_called.set()

        async def aclose(self):
            self.closed.set()

    class Client:
        async def aclose(self):
            return None

    original_wait_for = asyncio.wait_for

    async def cancel_after_ack(awaitable, timeout):
        await original_wait_for(awaitable, timeout=timeout)
        asyncio.current_task().cancel()
        await asyncio.sleep(0)

    bus = redis_bus.RedisBus(allow_fallback=False)
    pubsub = PubSub()
    bus._client = Client()
    bus._pubsub = pubsub

    with monkeypatch.context() as patch:
        patch.setattr(redis_bus.asyncio, "wait_for", cancel_after_ack)
        subscribe_task = asyncio.create_task(bus.subscribe("mf.ack-cancel", lambda message: None))
        with pytest.raises(asyncio.CancelledError):
            await subscribe_task

    assert bus.callback_count == 0
    assert bus._subscribe_acks == {}
    assert bus._orphaned_subscriptions == set()
    assert bus._deferred_unsubscribe_subjects == {"mf.ack-cancel"}
    subscription = await asyncio.wait_for(
        bus.subscribe("mf.next", lambda message: None),
        timeout=0.05,
    )
    assert subscription.subject == "mf.next"
    assert bus._deferred_unsubscribe_subjects == set()
    assert pubsub.unsubscribe_called.is_set()
    await bus.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("error", "cancel"))
async def test_redis_remote_unsubscribe_failure_is_retried_by_agent_stop(
    failure: str,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.subjects: asyncio.Queue[str] = asyncio.Queue()
            self.unsubscribe_calls: list[str] = []
            self.closed = False

        async def subscribe(self, subject: str) -> None:
            await self.subjects.put(subject)

        async def listen(self):
            while True:
                subject = await self.subjects.get()
                yield {"type": "subscribe", "channel": subject, "data": 1}

        async def unsubscribe(self, subject: str) -> None:
            self.unsubscribe_calls.append(subject)
            if len(self.unsubscribe_calls) == 1:
                if failure == "cancel":
                    raise asyncio.CancelledError
                raise RuntimeError("remote unsubscribe failed")

        async def aclose(self) -> None:
            self.closed = True

    class Client:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    class SubscriptionAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = ["mf.retry-unsubscribe"]

    bus = RedisBus(allow_fallback=False)
    pubsub = PubSub()
    client = Client()
    bus._client = client
    bus._pubsub = pubsub
    agent = SubscriptionAgent(bus)
    await agent.start()

    expected_error = asyncio.CancelledError if failure == "cancel" else RuntimeError
    with pytest.raises(expected_error):
        await agent.stop()
    first_stop_state = (
        tuple(pubsub.unsubscribe_calls),
        set(bus._deferred_unsubscribe_subjects),
        bus.callback_count,
        agent._started,
    )

    await agent.stop()
    second_stop_state = (
        tuple(pubsub.unsubscribe_calls),
        set(bus._deferred_unsubscribe_subjects),
        bus.callback_count,
        agent._started,
    )
    await bus.close()

    assert first_stop_state == (
        ("mf.retry-unsubscribe",),
        {"mf.retry-unsubscribe"},
        0,
        False,
    )
    assert second_stop_state == (
        ("mf.retry-unsubscribe", "mf.retry-unsubscribe"),
        set(),
        0,
        False,
    )
    assert pubsub.closed is True
    assert client.closed is True


@pytest.mark.asyncio
async def test_stale_unsubscribe_handle_does_not_remove_new_same_subject_callback() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.subjects: asyncio.Queue[str] = asyncio.Queue()
            self.unsubscribe_calls: list[str] = []

        async def subscribe(self, subject: str) -> None:
            await self.subjects.put(subject)

        async def listen(self):
            while True:
                subject = await self.subjects.get()
                yield {"type": "subscribe", "channel": subject, "data": 1}

        async def unsubscribe(self, subject: str) -> None:
            self.unsubscribe_calls.append(subject)
            if len(self.unsubscribe_calls) == 1:
                raise RuntimeError("remote unsubscribe failed")

        async def aclose(self) -> None:
            return None

    class Client:
        async def aclose(self) -> None:
            return None

    class SubscriptionAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = ["mf.same-subject"]

    bus = RedisBus(allow_fallback=False)
    pubsub = PubSub()
    bus._client = Client()
    bus._pubsub = pubsub
    old_agent = SubscriptionAgent(bus)
    new_agent = SubscriptionAgent(bus)
    await old_agent.start()

    with pytest.raises(RuntimeError, match="remote unsubscribe failed"):
        await old_agent.stop()
    await new_agent.start()
    state_after_new_start = (
        tuple(pubsub.unsubscribe_calls),
        set(bus._deferred_unsubscribe_subjects),
        bus.callback_count,
    )

    await old_agent.stop()
    state_after_stale_stop = (
        tuple(pubsub.unsubscribe_calls),
        set(bus._deferred_unsubscribe_subjects),
        bus.callback_count,
    )
    await new_agent.stop()
    final_state = (
        tuple(pubsub.unsubscribe_calls),
        set(bus._deferred_unsubscribe_subjects),
        bus.callback_count,
    )
    await bus.close()

    assert state_after_new_start == (
        ("mf.same-subject", "mf.same-subject"),
        set(),
        1,
    )
    assert state_after_stale_stop == state_after_new_start
    assert final_state == (
        ("mf.same-subject", "mf.same-subject", "mf.same-subject"),
        set(),
        0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("request", "roundtrip"))
async def test_redis_operation_deadline_covers_reply_subscription_ack(
    operation: str,
) -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.command_sent = asyncio.Event()
            self.closed = asyncio.Event()

        async def subscribe(self, subject):
            self.command_sent.set()

        async def listen(self):
            await self.closed.wait()
            if False:
                yield {}

        async def unsubscribe(self, subject):
            return None

        async def aclose(self):
            self.closed.set()

    class Client:
        def __init__(self) -> None:
            self.publish_called = False

        async def publish(self, subject, payload):
            self.publish_called = True

        async def aclose(self):
            return None

    bus = RedisBus(connect_timeout=1.0, allow_fallback=False)
    pubsub = PubSub()
    client = Client()
    bus._client = client
    bus._pubsub = pubsub
    if operation == "request":
        operation_task = asyncio.create_task(bus.request("mf.request", b"payload", timeout=0.01))
    else:
        operation_task = asyncio.create_task(bus.roundtrip(timeout=0.01))
    await asyncio.wait_for(pubsub.command_sent.wait(), timeout=0.05)
    await asyncio.sleep(0.03)

    assert operation_task.done()
    if operation == "request":
        with pytest.raises(TimeoutError):
            await operation_task
    else:
        assert await operation_task is False
    assert client.publish_called is False
    assert bus.callback_count == 0
    assert bus._subscribe_acks == {}
    await bus.close()


@pytest.mark.asyncio
async def test_redis_subscribe_failure_rolls_back_callback_registry() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.subscribe_calls = 0
            self.subject = ""
            self.closed = asyncio.Event()

        async def subscribe(self, subject):
            self.subscribe_calls += 1
            if self.subscribe_calls == 1:
                raise RuntimeError("subscribe failed")
            self.subject = subject

        async def unsubscribe(self, subject):
            return None

        async def listen(self):
            yield {"type": "subscribe", "channel": self.subject, "data": 1}
            await self.closed.wait()

        async def aclose(self):
            self.closed.set()

    class Client:
        async def aclose(self):
            return None

    bus = RedisBus(allow_fallback=False)
    pubsub = PubSub()
    bus._client = Client()
    bus._pubsub = pubsub

    with pytest.raises(RuntimeError, match="subscribe failed"):
        await bus.subscribe("mf.retry", lambda message: None)

    assert bus.callback_count == 0

    await bus.subscribe("mf.retry", lambda message: None)

    assert pubsub.subscribe_calls == 2
    assert bus.callback_count == 1
    await bus.close()


@pytest.mark.asyncio
async def test_redis_subscribe_cancellation_rolls_back_callback_registry() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.subscribe_started = asyncio.Event()
            self.closed = asyncio.Event()

        async def subscribe(self, subject):
            self.subscribe_started.set()

        async def listen(self):
            await self.closed.wait()
            if False:
                yield {}

        async def unsubscribe(self, subject):
            return None

        async def aclose(self):
            self.closed.set()

    class Client:
        async def aclose(self):
            return None

    bus = RedisBus(allow_fallback=False)
    pubsub = PubSub()
    bus._client = Client()
    bus._pubsub = pubsub
    subscribe_task = asyncio.create_task(bus.subscribe("mf.cancel", lambda message: None))
    await asyncio.wait_for(pubsub.subscribe_started.wait(), timeout=0.05)
    await asyncio.sleep(0)

    subscribe_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await subscribe_task

    assert bus.callback_count == 0
    assert bus._subscribe_acks == {}
    await bus.close()


@pytest.mark.asyncio
async def test_redis_reader_failure_rolls_back_pending_subscribe() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.listen_calls = 0
            self.subject = ""
            self.closed = asyncio.Event()

        async def subscribe(self, subject):
            self.subject = subject

        async def listen(self):
            self.listen_calls += 1
            if self.listen_calls == 1:
                raise PermissionError("ACL rejected SUBSCRIBE")
            yield {"type": "subscribe", "channel": self.subject, "data": 1}
            await self.closed.wait()

        async def unsubscribe(self, subject):
            return None

        async def aclose(self):
            self.closed.set()

    class Client:
        async def aclose(self):
            return None

    bus = RedisBus(allow_fallback=False)
    pubsub = PubSub()
    bus._client = Client()
    bus._pubsub = pubsub

    with pytest.raises(PermissionError, match="ACL"):
        await bus.subscribe("mf.reader-retry", lambda message: None)

    assert bus.callback_count == 0
    assert bus._subscribe_acks == {}

    subscription = await asyncio.wait_for(
        bus.subscribe("mf.reader-retry", lambda message: None),
        timeout=0.05,
    )
    assert subscription.subject == "mf.reader-retry"
    assert bus.callback_count == 1
    await bus.close()


@pytest.mark.asyncio
async def test_redis_reader_failure_drops_active_callbacks_and_allows_resubscribe() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.listen_calls = 0
            self.subject = ""
            self.fail_reader = asyncio.Event()
            self.closed = asyncio.Event()

        async def subscribe(self, subject):
            self.subject = subject

        async def listen(self):
            self.listen_calls += 1
            yield {"type": "subscribe", "channel": self.subject, "data": 1}
            if self.listen_calls == 1:
                await self.fail_reader.wait()
                raise RuntimeError("reader failed after subscribe")
            await self.closed.wait()

        async def unsubscribe(self, subject):
            return None

        async def aclose(self):
            self.closed.set()

    class Client:
        async def aclose(self):
            return None

    bus = RedisBus(allow_fallback=False)
    pubsub = PubSub()
    bus._client = Client()
    bus._pubsub = pubsub

    await bus.subscribe("mf.reader-retry", lambda message: None)
    pubsub.fail_reader.set()
    await asyncio.wait_for(
        _wait_until(lambda: bus._listener_task is not None and bus._listener_task.done()),
        timeout=0.05,
    )

    assert bus.callback_count == 0
    assert bus._subscribe_acks == {}

    subscription = await asyncio.wait_for(
        bus.subscribe("mf.reader-retry", lambda message: None),
        timeout=0.05,
    )
    assert subscription.subject == "mf.reader-retry"
    assert pubsub.listen_calls == 2
    assert bus.callback_count == 1
    await bus.close()


@pytest.mark.asyncio
async def test_concurrent_redis_subscribe_retries_after_first_failure() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.subscribe_calls = 0
            self.subject = ""
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()
            self.closed = asyncio.Event()

        async def subscribe(self, subject):
            self.subscribe_calls += 1
            if self.subscribe_calls == 1:
                self.first_started.set()
                await self.release_first.wait()
                raise RuntimeError("first subscribe failed")
            self.subject = subject

        async def listen(self):
            yield {"type": "subscribe", "channel": self.subject, "data": 1}
            await self.closed.wait()

        async def aclose(self):
            self.closed.set()

    class Client:
        async def aclose(self):
            return None

    bus = RedisBus(allow_fallback=False)
    pubsub = PubSub()
    bus._client = Client()
    bus._pubsub = pubsub
    first = asyncio.create_task(bus.subscribe("mf.concurrent", lambda message: None))
    await asyncio.wait_for(pubsub.first_started.wait(), timeout=0.05)
    second = asyncio.create_task(bus.subscribe("mf.concurrent", lambda message: None))
    await asyncio.sleep(0)
    second_waited_for_first = not second.done()
    pubsub.release_first.set()

    with pytest.raises(RuntimeError, match="first subscribe failed"):
        await first
    subscription = await second

    assert second_waited_for_first is True
    assert subscription.subject == "mf.concurrent"
    assert pubsub.subscribe_calls == 2
    assert bus.callback_count == 1
    await bus.close()


@pytest.mark.asyncio
async def test_redis_close_cleans_resources_after_listener_failure() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.listen_started = asyncio.Event()
            self.closed = False

        async def subscribe(self, subject):
            return None

        async def listen(self):
            self.listen_started.set()
            raise RuntimeError("reader failed")
            yield

        async def aclose(self):
            self.closed = True

    class Client:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self):
            self.closed = True

    bus = RedisBus(allow_fallback=False)
    pubsub = PubSub()
    client = Client()
    bus._client = client
    bus._pubsub = pubsub
    bus._listener_task = asyncio.create_task(bus._listen())
    await asyncio.wait_for(pubsub.listen_started.wait(), timeout=0.05)
    await asyncio.wait_for(
        _wait_until(lambda: bus._listener_task is not None and bus._listener_task.done()),
        timeout=0.05,
    )
    hanging_callback = asyncio.create_task(asyncio.Event().wait())
    bus._callback_tasks.add(hanging_callback)

    try:
        await asyncio.wait_for(bus.close(), timeout=0.05)
    finally:
        if not hanging_callback.done():
            hanging_callback.cancel()
            with pytest.raises(asyncio.CancelledError):
                await hanging_callback

    assert bus.callback_count == 0
    assert bus.callback_task_count == 0
    assert pubsub.closed is True
    assert client.closed is True


@pytest.mark.asyncio
async def test_redis_close_failure_preserves_failed_resource_for_retry() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("pubsub close failed")

    class Client:
        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self):
            self.close_calls += 1

    bus = RedisBus(allow_fallback=False)
    pubsub = PubSub()
    client = Client()
    bus._client = client
    bus._pubsub = pubsub

    with pytest.raises(RuntimeError, match="pubsub close failed"):
        await bus.close()
    failed_state = (
        bus._pubsub,
        bus._client,
        pubsub.close_calls,
        client.close_calls,
    )
    await bus.close()

    assert failed_state == (pubsub, None, 1, 1)
    assert pubsub.close_calls == 2
    assert client.close_calls == 1
    assert bus._pubsub is None
    assert bus._client is None


@pytest.mark.asyncio
async def test_redis_client_close_failure_preserves_client_for_retry() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1

    class Client:
        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("client close failed")

    bus = RedisBus(allow_fallback=False)
    pubsub = PubSub()
    client = Client()
    bus._client = client
    bus._pubsub = pubsub

    with pytest.raises(RuntimeError, match="client close failed"):
        await bus.close()
    failed_state = (
        bus._pubsub,
        bus._client,
        pubsub.close_calls,
        client.close_calls,
    )
    await bus.close()

    assert failed_state == (None, client, 1, 1)
    assert pubsub.close_calls == 1
    assert client.close_calls == 2
    assert bus._pubsub is None
    assert bus._client is None


@pytest.mark.asyncio
async def test_cancelled_redis_close_preserves_failed_resource_for_retry() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.close_calls = 0
            self.first_close_started = asyncio.Event()

        async def aclose(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                self.first_close_started.set()
                await asyncio.Event().wait()

    class Client:
        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1

    bus = RedisBus(allow_fallback=False)
    pubsub = PubSub()
    client = Client()
    bus._client = client
    bus._pubsub = pubsub
    close_task = asyncio.create_task(bus.close())
    await asyncio.wait_for(pubsub.first_close_started.wait(), timeout=0.05)

    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task
    cancelled_state = (
        bus._pubsub,
        bus._client,
        pubsub.close_calls,
        client.close_calls,
    )

    await bus.close()

    assert cancelled_state == (pubsub, None, 1, 1)
    assert pubsub.close_calls == 2
    assert client.close_calls == 1
    assert bus._pubsub is None
    assert bus._client is None


@pytest.mark.asyncio
async def test_concurrent_redis_close_serializes_resource_cleanup() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.close_calls = 0
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()

        async def aclose(self) -> None:
            self.close_calls += 1
            self.close_started.set()
            await self.release_close.wait()

    class Client:
        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1

    bus = RedisBus(allow_fallback=False)
    pubsub = PubSub()
    client = Client()
    bus._client = client
    bus._pubsub = pubsub
    first = asyncio.create_task(bus.close())
    await asyncio.wait_for(pubsub.close_started.wait(), timeout=0.05)
    second = asyncio.create_task(bus.close())
    await asyncio.sleep(0)
    state_while_closing = (
        second.done(),
        pubsub.close_calls,
        client.close_calls,
    )

    pubsub.release_close.set()
    await asyncio.gather(first, second)

    assert state_while_closing == (False, 1, 0)
    assert pubsub.close_calls == 1
    assert client.close_calls == 1
    assert bus._pubsub is None
    assert bus._client is None


@pytest.mark.asyncio
async def test_redis_roundtrip_timeout_covers_blocking_publish() -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.subject = ""
            self.closed = asyncio.Event()

        async def subscribe(self, subject):
            self.subject = subject

        async def unsubscribe(self, subject):
            return None

        async def listen(self):
            yield {"type": "subscribe", "channel": self.subject, "data": 1}
            await self.closed.wait()

        async def aclose(self):
            self.closed.set()

    class Client:
        async def publish(self, subject, payload):
            await asyncio.Event().wait()

        async def aclose(self):
            return None

    bus = RedisBus(allow_fallback=False)
    bus._client = Client()
    bus._pubsub = PubSub()
    try:
        result = await asyncio.wait_for(bus.roundtrip(timeout=0.01), timeout=0.06)
    finally:
        await bus.close()

    assert result is False
    assert bus.callback_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("request", "roundtrip"))
async def test_redis_timeout_does_not_wait_for_unsubscribe(operation: str) -> None:
    from mf_agents.messaging.redis_bus import RedisBus

    class PubSub:
        def __init__(self) -> None:
            self.subjects: asyncio.Queue[str] = asyncio.Queue()
            self.unsubscribe_called = asyncio.Event()
            self.closed = asyncio.Event()

        async def subscribe(self, subject):
            await self.subjects.put(subject)

        async def unsubscribe(self, subject):
            self.unsubscribe_called.set()

        async def listen(self):
            while not self.closed.is_set():
                subject = await self.subjects.get()
                yield {"type": "subscribe", "channel": subject, "data": 1}

        async def aclose(self):
            self.closed.set()

    class Client:
        async def publish(self, subject, payload):
            return None

        async def aclose(self):
            return None

    bus = RedisBus(allow_fallback=False)
    pubsub = PubSub()
    bus._client = Client()
    bus._pubsub = pubsub
    if operation == "request":
        operation_task = asyncio.create_task(bus.request("mf.request", b"payload", timeout=0.01))
    else:
        operation_task = asyncio.create_task(bus.roundtrip(timeout=0.01))

    try:
        await asyncio.sleep(0.03)

        assert operation_task.done()
        if operation == "request":
            with pytest.raises(TimeoutError):
                await operation_task
        else:
            assert await operation_task is False
        assert bus.callback_count == 0
        assert len(bus._deferred_unsubscribe_subjects) == 1
        assert pubsub.unsubscribe_called.is_set() is False
    finally:
        if not operation_task.done():
            try:
                await asyncio.wait_for(operation_task, timeout=0.05)
            except TimeoutError:
                pass
        await bus.close()


def test_redis_expired_deadline_does_not_leave_task_for_asyncio_run() -> None:
    script = """
import asyncio
from mf_agents.messaging.redis_bus import RedisBus

class PubSub:
    def __init__(self):
        self.subject = ""

    async def subscribe(self, subject):
        self.subject = subject

    async def listen(self):
        yield {"type": "subscribe", "channel": self.subject, "data": 1}
        await asyncio.Event().wait()

    async def unsubscribe(self, subject):
        while True:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass

    async def aclose(self):
        return None

class Client:
    async def publish(self, subject, payload):
        return None

    async def aclose(self):
        return None

async def main():
    bus = RedisBus(allow_fallback=False)
    bus._client = Client()
    bus._pubsub = PubSub()
    assert await bus.roundtrip(timeout=0.01) is False
    assert bus.callback_count == 0
    assert len(bus._deferred_unsubscribe_subjects) == 1
    await bus.close()

asyncio.run(main())
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=1.0,
    )

    assert result.returncode == 0, result.stderr


async def _wait_until(predicate) -> None:
    while not predicate():
        await asyncio.sleep(0)
