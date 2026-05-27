"""Redis message bus wrapper for agent communication."""
from __future__ import annotations

import asyncio
import inspect
import os
import uuid
from typing import Any, Callable


class RedisBus:
    """Redis-backed message bus with in-process fallback."""

    def __init__(
        self,
        url: str | None = None,
        connect_timeout: float = 2.0,
        allow_fallback: bool = True,
    ) -> None:
        self.url = url or _redis_url_from_env()
        self.connect_timeout = connect_timeout
        self.allow_fallback = allow_fallback
        self._client = None
        self._pubsub = None
        self._tasks: list[asyncio.Task] = []

    async def connect(self) -> None:
        try:
            from redis import asyncio as redis_async

            client = redis_async.from_url(self.url)
            await asyncio.wait_for(client.ping(), timeout=self.connect_timeout)
            self._client = client
        except Exception:
            if not self.allow_fallback:
                raise
            self._client = _FallbackBus()

    async def subscribe(self, subject: str, cb: Callable) -> Any:
        if self._client is None:
            return None
        if isinstance(self._client, _FallbackBus):
            return await self._client.subscribe(subject, cb)
        if self._pubsub is None:
            self._pubsub = self._client.pubsub()
        await self._pubsub.subscribe(subject)
        task = asyncio.create_task(self._listen(subject, cb))
        self._tasks.append(task)
        return task

    async def publish(self, subject: str, payload: bytes) -> None:
        if self._client is not None:
            await self._client.publish(subject, payload)

    async def request(self, subject: str, payload: bytes, timeout: float = 30.0) -> bytes:
        if self._client is None:
            return b""
        if isinstance(self._client, _FallbackBus):
            response = await self._client.request(subject, payload, timeout)
            return response.data
        reply_to = f"_reply.{uuid.uuid4().hex}"
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)

        async def on_reply(message):
            await queue.put(message["data"])

        subscription = await self.subscribe(reply_to, on_reply)
        await self.publish(subject, payload)
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except TimeoutError:
            return b""
        finally:
            if subscription in self._tasks:
                subscription.cancel()

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        if self._pubsub is not None:
            await _close_client(self._pubsub)
            self._pubsub = None
        if self._client is not None:
            await _close_client(self._client)
            self._client = None

    async def _listen(self, subject: str, cb: Callable) -> None:
        if self._pubsub is None:
            return
        async for message in self._pubsub.listen():
            if message.get("type") != "message":
                continue
            channel = message.get("channel")
            if isinstance(channel, bytes):
                channel = channel.decode("utf-8")
            if channel == subject:
                await _invoke_callback(cb, subject, message["data"])


class _FallbackBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable]] = {}

    async def subscribe(self, subject: str, cb: Callable) -> None:
        self._subs.setdefault(subject, []).append(cb)

    async def publish(self, subject: str, payload: bytes) -> None:
        for existing, callbacks in self._subs.items():
            if existing == subject:
                for cb in callbacks:
                    await _invoke_callback(cb, subject, payload)

    async def request(self, subject: str, payload: bytes, timeout: float = 30.0) -> Any:
        class _Resp:
            data = b""

        return _Resp()

    async def close(self) -> None:
        self._subs.clear()


def _redis_url_from_env() -> str:
    if os.environ.get("REDIS_URL"):
        return os.environ["REDIS_URL"]
    host = os.environ.get("REDIS_HOST", "127.0.0.1")
    port = os.environ.get("REDIS_PORT", "6379")
    password = os.environ.get("REDIS_PASSWORD")
    auth = f":{password}@" if password else ""
    return f"redis://{auth}{host}:{port}/0"


async def _invoke_callback(cb: Callable, subject: str, payload: bytes) -> None:
    parameters = list(inspect.signature(cb).parameters)
    if len(parameters) <= 1:
        result = cb({"subject": subject, "data": payload})
    else:
        result = cb(subject, payload, "")
    if inspect.isawaitable(result):
        await result


async def _close_client(client: Any) -> None:
    close = getattr(client, "aclose", None) or getattr(client, "close")
    result = close()
    if inspect.isawaitable(result):
        await result
