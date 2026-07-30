"""Redis and in-memory message buses for Agent communication."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

MessageCallback = Callable[..., Any | Awaitable[Any]]
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Subscription:
    subject: str
    token: str


class InMemoryBus:
    """Behavior-equivalent in-process bus used by tests and local development."""

    is_redis = False

    def __init__(self) -> None:
        self._callbacks: dict[str, dict[str, MessageCallback]] = {}
        self._callback_tasks: set[asyncio.Task[None]] = set()
        self._callback_task_tokens: dict[asyncio.Task[None], str] = {}
        self._connected = False
        self.last_published: dict[str, bytes] = {}

    @property
    def callback_count(self) -> int:
        return sum(len(callbacks) for callbacks in self._callbacks.values())

    @property
    def callback_task_count(self) -> int:
        return len(self._callback_tasks)

    async def connect(self) -> None:
        self._connected = True

    async def subscribe(self, subject: str, cb: MessageCallback) -> Subscription:
        if not self._connected:
            raise RuntimeError("message bus is not connected")
        token = uuid.uuid4().hex
        self._callbacks.setdefault(subject, {})[token] = cb
        return Subscription(subject, token)

    def discard_subscription(self, subscription: Subscription) -> bool:
        callbacks = self._callbacks.get(subscription.subject)
        if callbacks is None:
            return False
        callbacks.pop(subscription.token, None)
        if not callbacks:
            self._callbacks.pop(subscription.subject, None)
        return False

    async def unsubscribe(self, subscription: Subscription) -> None:
        self.discard_subscription(subscription)
        await _cancel_subscription_callback_tasks(
            self._callback_tasks,
            self._callback_task_tokens,
            subscription.token,
        )

    async def publish(self, subject: str, payload: bytes) -> None:
        if not self._connected:
            raise RuntimeError("message bus is not connected")
        self.last_published[subject] = payload
        callbacks = tuple(self._callbacks.get(subject, {}).items())
        _schedule_callbacks(
            self._callback_tasks,
            self._callback_task_tokens,
            callbacks,
            subject,
            payload,
        )

    async def request(self, subject: str, payload: bytes, timeout: float = 30.0) -> bytes:
        reply_to = f"_reply.{uuid.uuid4().hex}"
        future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        deadline = _deadline_after(timeout)

        async def on_reply(message: dict[str, Any]) -> None:
            if not future.done():
                future.set_result(message["data"])

        subscription: Subscription | None = None
        try:
            async with asyncio.timeout_at(deadline):
                subscription = await self.subscribe(reply_to, on_reply)
                await self.publish(subject, payload)
                return await future
        finally:
            if subscription is not None:
                await _bounded_unsubscribe(self, subscription, deadline)

    async def roundtrip(self, timeout: float = 1.0) -> bool:
        subject = f"_health.{uuid.uuid4().hex}"
        expected = uuid.uuid4().bytes
        future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        deadline = _deadline_after(timeout)

        async def on_message(message: dict[str, Any]) -> None:
            if not future.done():
                future.set_result(message["data"])

        subscription: Subscription | None = None
        try:
            async with asyncio.timeout_at(deadline):
                subscription = await self.subscribe(subject, on_message)
                await self.publish(subject, expected)
                received = await future
            return received == expected
        except TimeoutError:
            return False
        finally:
            if subscription is not None:
                await _bounded_unsubscribe(self, subscription, deadline)

    async def close(self) -> None:
        await _cancel_callback_tasks(self._callback_tasks)
        self._callback_task_tokens.clear()
        self._callbacks.clear()
        self._connected = False


class RedisBus:
    """Redis pub/sub bus with one listener and a channel callback registry."""

    def __init__(
        self,
        url: str | None = None,
        connect_timeout: float = 2.0,
        allow_fallback: bool = True,
    ) -> None:
        self.url = url or _redis_url_from_env()
        self.connect_timeout = connect_timeout
        self.allow_fallback = allow_fallback
        self._client: Any = None
        self._pubsub: Any = None
        self._fallback: InMemoryBus | None = None
        self._callbacks: dict[str, dict[str, MessageCallback]] = {}
        self._callback_tasks: set[asyncio.Task[None]] = set()
        self._callback_task_tokens: dict[asyncio.Task[None], str] = {}
        self._listener_task: asyncio.Task[None] | None = None
        self._subscribe_acks: dict[str, asyncio.Future[None]] = {}
        self._orphaned_subscriptions: set[str] = set()
        self._deferred_unsubscribe_subjects: set[str] = set()
        self._background_failure: BaseException | None = None
        self._background_failure_event = asyncio.Event()
        self._lifecycle_lock = asyncio.Lock()
        self._subscription_lock = asyncio.Lock()
        self._closed = False

    @property
    def is_redis(self) -> bool:
        return self._client is not None and self._fallback is None

    @property
    def callback_count(self) -> int:
        if self._fallback is not None:
            return self._fallback.callback_count
        return sum(len(callbacks) for callbacks in self._callbacks.values())

    @property
    def listener_count(self) -> int:
        return int(self._listener_task is not None and not self._listener_task.done())

    @property
    def callback_task_count(self) -> int:
        if self._fallback is not None:
            return self._fallback.callback_task_count
        return len(self._callback_tasks)

    async def connect(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("message bus is closed")
            if self._fallback is not None:
                return
            if self._client is not None and self._pubsub is not None:
                return
            if self._client is not None or self._pubsub is not None:
                raise RuntimeError(
                    "message bus has incomplete Redis resources; close before reconnecting"
                )
            client = None
            try:
                from redis import asyncio as redis_async

                client = redis_async.from_url(self.url)
                await asyncio.wait_for(client.ping(), timeout=self.connect_timeout)
                pubsub = client.pubsub()
            except BaseException as error:
                cleanup_error: BaseException | None = None
                if client is not None:
                    try:
                        await _close_client(client)
                    except BaseException as close_error:
                        self._client = client
                        cleanup_error = close_error
                if cleanup_error is not None:
                    raise BaseExceptionGroup(
                        "Redis connect and local resource cleanup failed",
                        [error, cleanup_error],
                    ) from None
                if not isinstance(error, Exception):
                    raise
                if not self.allow_fallback:
                    raise
                fallback = InMemoryBus()
                try:
                    await fallback.connect()
                except BaseException:
                    await fallback.close()
                    raise
                self._fallback = fallback
                return
            self._client = client
            self._pubsub = pubsub

    async def subscribe(self, subject: str, cb: MessageCallback) -> Subscription:
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("message bus is closed")
            return await self._subscribe_locked(subject, cb)

    async def _subscribe_locked(
        self,
        subject: str,
        cb: MessageCallback,
    ) -> Subscription:
        if self._fallback is not None:
            return await self._fallback.subscribe(subject, cb)
        async with self._subscription_lock:
            if self._client is None or self._pubsub is None:
                raise RuntimeError("message bus is not connected")
            await self._flush_deferred_unsubscribes()
            token = uuid.uuid4().hex
            callbacks = self._callbacks.setdefault(subject, {})
            first_callback = not callbacks
            callbacks[token] = cb
            subscribe_ack: asyncio.Future[None] | None = None
            try:
                if first_callback:
                    subscribe_ack = asyncio.get_running_loop().create_future()
                    self._subscribe_acks[subject] = subscribe_ack
                    await self._pubsub.subscribe(subject)
                    self._start_listener()
                    await asyncio.wait_for(
                        asyncio.shield(subscribe_ack),
                        timeout=self.connect_timeout,
                    )
            except BaseException:
                ack_received = _future_completed_successfully(subscribe_ack)
                callbacks.pop(token, None)
                if not callbacks:
                    self._callbacks.pop(subject, None)
                if subscribe_ack is not None and self._subscribe_acks.get(subject) is subscribe_ack:
                    self._subscribe_acks.pop(subject, None)
                if subscribe_ack is not None and not subscribe_ack.done():
                    subscribe_ack.cancel()
                if subscribe_ack is not None:
                    if ack_received:
                        self._orphaned_subscriptions.discard(subject)
                        if not self._callbacks.get(subject):
                            self.defer_unsubscribe(subject)
                    else:
                        self._orphaned_subscriptions.add(subject)
                raise
            if subscribe_ack is not None and self._subscribe_acks.get(subject) is subscribe_ack:
                self._subscribe_acks.pop(subject, None)
            return Subscription(subject, token)

    def discard_subscription(self, subscription: Subscription) -> bool:
        if self._fallback is not None:
            return self._fallback.discard_subscription(subscription)
        callbacks = self._callbacks.get(subscription.subject)
        if callbacks is None or subscription.token not in callbacks:
            return False
        callbacks.pop(subscription.token)
        if callbacks:
            return False
        self._callbacks.pop(subscription.subject, None)
        return True

    async def unsubscribe(self, subscription: Subscription) -> None:
        if self._fallback is not None:
            await self._fallback.unsubscribe(subscription)
            return
        needs_remote_unsubscribe = self.discard_subscription(subscription)
        await _cancel_subscription_callback_tasks(
            self._callback_tasks,
            self._callback_task_tokens,
            subscription.token,
        )
        subject = subscription.subject
        if not needs_remote_unsubscribe and subject not in self._deferred_unsubscribe_subjects:
            return
        self.defer_unsubscribe(subject)
        await self._unsubscribe_subject(subject)

    def defer_unsubscribe(self, subject: str) -> None:
        self._deferred_unsubscribe_subjects.add(subject)

    async def _unsubscribe_subject(self, subject: str) -> None:
        if self._fallback is not None:
            return
        async with self._subscription_lock:
            if self._callbacks.get(subject):
                return
            if self._pubsub is not None:
                await self._pubsub.unsubscribe(subject)
                self._deferred_unsubscribe_subjects.discard(subject)

    async def _flush_deferred_unsubscribes(self) -> None:
        if self._pubsub is None:
            return
        subjects = tuple(
            subject
            for subject in self._deferred_unsubscribe_subjects
            if not self._callbacks.get(subject)
        )
        if not subjects:
            return
        await self._pubsub.unsubscribe(*subjects)
        self._deferred_unsubscribe_subjects.difference_update(subjects)

    async def publish(self, subject: str, payload: bytes) -> None:
        if self._fallback is not None:
            await self._fallback.publish(subject, payload)
            return
        if self._client is None:
            raise RuntimeError("message bus is not connected")
        await self._client.publish(subject, payload)

    async def request(self, subject: str, payload: bytes, timeout: float = 30.0) -> bytes:
        reply_to = f"_reply.{uuid.uuid4().hex}"
        future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        deadline = _deadline_after(timeout)

        async def on_reply(message: dict[str, Any]) -> None:
            if not future.done():
                future.set_result(message["data"])

        subscription: Subscription | None = None
        try:
            async with asyncio.timeout_at(deadline):
                subscription = await self.subscribe(reply_to, on_reply)
                await self.publish(subject, payload)
                return await future
        finally:
            if subscription is not None:
                await _bounded_unsubscribe(self, subscription, deadline)

    async def roundtrip(self, timeout: float = 1.0) -> bool:
        if self._fallback is not None:
            return await self._fallback.roundtrip(timeout)
        subject = f"_health.{uuid.uuid4().hex}"
        expected = uuid.uuid4().bytes
        future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        deadline = _deadline_after(timeout)

        async def on_message(message: dict[str, Any]) -> None:
            if not future.done():
                future.set_result(message["data"])

        subscription: Subscription | None = None
        try:
            async with asyncio.timeout_at(deadline):
                subscription = await self.subscribe(subject, on_message)
                await self.publish(subject, expected)
                received = await future
            return received == expected
        except TimeoutError:
            return False
        finally:
            if subscription is not None:
                await _bounded_unsubscribe(self, subscription, deadline)

    async def wait_for_background_failure(self) -> None:
        await self._background_failure_event.wait()
        failure = self._background_failure
        if failure is None:
            raise RuntimeError("Redis background task failed without an error")
        raise failure

    async def close(self) -> None:
        async with self._lifecycle_lock:
            self._closed = True
            await self._close_locked()

    async def _close_locked(self) -> None:
        errors: list[BaseException] = []
        cancellation: asyncio.CancelledError | None = None
        if self._fallback is not None:
            fallback = self._fallback
            try:
                await fallback.close()
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            except BaseException as error:
                errors.append(error)
            else:
                if self._fallback is fallback:
                    self._fallback = None
        if self._listener_task is not None:
            listener_task = self._listener_task
            listener_task.cancel()
            try:
                listener_result = await asyncio.gather(
                    listener_task,
                    return_exceptions=True,
                )
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            else:
                listener_error = listener_result[0]
                if isinstance(listener_error, Exception):
                    _LOGGER.error(
                        "Redis listener failed before close",
                        exc_info=(
                            type(listener_error),
                            listener_error,
                            listener_error.__traceback__,
                        ),
                    )
            if listener_task.done():
                self._listener_task = None
        for subscribe_ack in self._subscribe_acks.values():
            if not subscribe_ack.done():
                subscribe_ack.cancel()
        self._subscribe_acks.clear()
        self._orphaned_subscriptions.clear()
        try:
            await _cancel_callback_tasks(self._callback_tasks)
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
        self._callback_task_tokens.clear()
        self._callbacks.clear()
        self._deferred_unsubscribe_subjects.clear()
        if self._pubsub is not None:
            pubsub = self._pubsub
            try:
                await _close_client(pubsub)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            except BaseException as error:
                _LOGGER.error(
                    "Redis pubsub close failed",
                    exc_info=(type(error), error, error.__traceback__),
                )
                errors.append(error)
            else:
                if self._pubsub is pubsub:
                    self._pubsub = None
        if self._client is not None:
            client = self._client
            try:
                await _close_client(client)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            except BaseException as error:
                _LOGGER.error(
                    "Redis client close failed",
                    exc_info=(type(error), error, error.__traceback__),
                )
                errors.append(error)
            else:
                if self._client is client:
                    self._client = None
        if cancellation is not None:
            if errors:
                raise BaseExceptionGroup(
                    "Redis close failed",
                    [cancellation, *errors],
                )
            raise cancellation
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Redis close failed", errors)

    async def _listen(self) -> None:
        if self._pubsub is None:
            return
        try:
            async for message in self._pubsub.listen():
                message_type = message.get("type")
                channel = message.get("channel")
                if isinstance(channel, bytes):
                    channel = channel.decode("utf-8")
                subject = str(channel)
                if message_type == "subscribe":
                    subscribe_ack = self._subscribe_acks.get(subject)
                    if subscribe_ack is not None and not subscribe_ack.done():
                        subscribe_ack.set_result(None)
                    if subject in self._orphaned_subscriptions:
                        self._orphaned_subscriptions.discard(subject)
                        if not self._callbacks.get(subject):
                            self.defer_unsubscribe(subject)
                    continue
                if message_type != "message":
                    continue
                callbacks = tuple(self._callbacks.get(subject, {}).items())
                _schedule_callbacks(
                    self._callback_tasks,
                    self._callback_task_tokens,
                    callbacks,
                    subject,
                    message["data"],
                )
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                if self._background_failure is None:
                    self._background_failure = error
                    self._background_failure_event.set()
                self._callbacks.clear()
                self._orphaned_subscriptions.clear()
            for subscribe_ack in self._subscribe_acks.values():
                if subscribe_ack.done():
                    continue
                if isinstance(error, asyncio.CancelledError):
                    subscribe_ack.cancel()
                else:
                    subscribe_ack.set_exception(error)
            raise

    def _start_listener(self) -> None:
        if self._listener_task is not None and not self._listener_task.done():
            return
        if self._listener_task is not None:
            with contextlib.suppress(BaseException):
                self._listener_task.exception()
        self._listener_task = asyncio.create_task(self._listen())


def _deadline_after(timeout: float) -> float:
    return asyncio.get_running_loop().time() + max(0.0, float(timeout))


async def _bounded_unsubscribe(
    message_bus: Any,
    subscription: Subscription,
    deadline: float,
) -> None:
    discard_subscription = getattr(message_bus, "discard_subscription", None)
    if callable(discard_subscription):
        needs_remote_unsubscribe = bool(discard_subscription(subscription))
        if not needs_remote_unsubscribe:
            return
        unsubscribe_subject = getattr(message_bus, "_unsubscribe_subject", None)
        if callable(unsubscribe_subject):
            cleanup = unsubscribe_subject
            cleanup_target = subscription.subject
        else:
            cleanup = message_bus.unsubscribe
            cleanup_target = subscription
    else:
        cleanup = message_bus.unsubscribe
        cleanup_target = subscription
    remaining = deadline - asyncio.get_running_loop().time()
    defer_unsubscribe = getattr(message_bus, "defer_unsubscribe", None)
    if callable(defer_unsubscribe):
        defer_unsubscribe(subscription.subject)
        return
    if remaining <= 0.0:
        return
    cleanup_task = asyncio.create_task(cleanup(cleanup_target))
    try:
        done, _pending = await asyncio.wait((cleanup_task,), timeout=remaining)
    except BaseException:
        cleanup_task.cancel()
        cleanup_task.add_done_callback(_consume_cleanup_task)
        raise
    if cleanup_task not in done:
        cleanup_task.cancel()
        cleanup_task.add_done_callback(_consume_cleanup_task)
        return
    _consume_cleanup_task(cleanup_task)


def _consume_cleanup_task(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        _LOGGER.exception("Agent reply subscription cleanup failed")


def _future_completed_successfully(
    future: asyncio.Future[Any] | None,
) -> bool:
    return (
        future is not None
        and future.done()
        and not future.cancelled()
        and future.exception() is None
    )


def _redis_url_from_env() -> str:
    if os.environ.get("REDIS_URL"):
        return os.environ["REDIS_URL"]
    host = os.environ.get("REDIS_HOST", "127.0.0.1")
    port = os.environ.get("REDIS_PORT", "6379")
    password = os.environ.get("REDIS_PASSWORD")
    auth = f":{password}@" if password else ""
    return f"redis://{auth}{host}:{port}/0"


async def _invoke_callback(cb: MessageCallback, subject: str, payload: bytes) -> None:
    parameters = list(inspect.signature(cb).parameters)
    if len(parameters) <= 1:
        result = cb({"subject": subject, "data": payload})
    else:
        result = cb(subject, payload, "")
    if inspect.isawaitable(result):
        await result


def _schedule_callbacks(
    tasks: set[asyncio.Task[None]],
    task_tokens: dict[asyncio.Task[None], str],
    callbacks: tuple[tuple[str, MessageCallback], ...],
    subject: str,
    payload: bytes,
) -> None:
    for token, callback in callbacks:
        task = asyncio.create_task(_run_callback(callback, subject, payload))
        tasks.add(task)
        task_tokens[task] = token
        task.add_done_callback(
            lambda completed: _discard_callback_task(
                completed,
                tasks,
                task_tokens,
            )
        )


def _discard_callback_task(
    task: asyncio.Task[None],
    tasks: set[asyncio.Task[None]],
    task_tokens: dict[asyncio.Task[None], str],
) -> None:
    tasks.discard(task)
    task_tokens.pop(task, None)


async def _run_callback(
    callback: MessageCallback,
    subject: str,
    payload: bytes,
) -> None:
    try:
        await _invoke_callback(callback, subject, payload)
    except Exception:
        _LOGGER.exception("Agent message callback failed")


async def _cancel_callback_tasks(tasks: set[asyncio.Task[None]]) -> None:
    pending = tuple(tasks)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    tasks.clear()


async def _cancel_subscription_callback_tasks(
    tasks: set[asyncio.Task[None]],
    task_tokens: dict[asyncio.Task[None], str],
    token: str,
) -> None:
    current_task = asyncio.current_task()
    pending = tuple(
        task
        for task, task_token in task_tokens.items()
        if task_token == token and task is not current_task
    )
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    for task in pending:
        tasks.discard(task)
        task_tokens.pop(task, None)


async def _close_client(client: Any) -> None:
    close = getattr(client, "aclose", None) or client.close
    result = close()
    if inspect.isawaitable(result):
        await result
