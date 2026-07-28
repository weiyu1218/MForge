"""Independent runtime for the six request/reply Agents."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import logging
import os
import signal
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentHeartbeat

from mf_agents.base.agent import (
    AGENT_PROTOCOLS,
    agent_health_check_timeout_seconds,
    run_health_probe_in_daemon,
)
from mf_agents.messaging.redis_bus import RedisBus

AGENT_ENTRY_POINTS = {protocol.entry_point: protocol.subject for protocol in AGENT_PROTOCOLS}
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeHealth:
    ready: bool
    entry_point: bool
    redis: bool
    targets: dict[str, bool] = field(default_factory=dict)
    reason: str = ""


def load_agent_entry_point(name: str) -> Callable:
    if name not in AGENT_ENTRY_POINTS:
        raise LookupError(f"unsupported Agent entry point: {name}")
    matches = [item for item in entry_points(group="moleculeforge.agents") if item.name == name]
    if len(matches) != 1:
        raise LookupError(
            f"Agent entry point must resolve exactly once: {name}; found {len(matches)}"
        )
    loaded = matches[0].load()
    if not callable(loaded):
        raise TypeError(f"Agent entry point is not callable: {name}")
    return loaded


class AgentRuntime:
    def __init__(
        self,
        agent_name: str,
        *,
        entry_point_loader: Callable[[str], Callable] = load_agent_entry_point,
        message_bus=None,
        production: bool = False,
        heartbeat_interval: float = 10.0,
    ) -> None:
        if agent_name not in AGENT_ENTRY_POINTS:
            raise ValueError(f"unsupported Agent runtime: {agent_name}")
        self.agent_name = agent_name
        self.entry_point_loader = entry_point_loader
        self.message_bus = message_bus
        self.production = production
        self.heartbeat_interval = heartbeat_interval
        self.agent = None
        self._agent_factory: Callable | None = None
        self.health = RuntimeHealth(False, False, False)
        self.ready = False
        self._bus_connected = False
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()
        self._lifecycle_lock = asyncio.Lock()
        self._started = False
        self._agent_stopped = False
        self._closed = False

    async def check_readiness(self) -> RuntimeHealth:
        async with self._lifecycle_lock:
            return await self._check_readiness_locked()

    async def _check_readiness_locked(self) -> RuntimeHealth:
        if self._closed:
            return self._set_health(
                RuntimeHealth(
                    False,
                    self.health.entry_point,
                    self._bus_connected,
                    dict(self.health.targets),
                    "Agent runtime has been shut down",
                )
            )
        try:
            self._resolve_agent_factory()
        except Exception as exc:
            return self._set_health(RuntimeHealth(False, False, False, reason=str(exc)))
        try:
            bus = self._resolve_bus()
            if not self._bus_connected:
                await bus.connect()
                self._bus_connected = True
            if self.production and not bool(getattr(bus, "is_redis", False)):
                return self._set_health(
                    RuntimeHealth(
                        False,
                        True,
                        False,
                        reason="production Agent runtime requires Redis",
                    )
                )
            redis_timeout = min(1.0, agent_health_check_timeout_seconds())
            redis_ready = bool(
                await asyncio.wait_for(
                    bus.roundtrip(timeout=redis_timeout),
                    timeout=redis_timeout,
                )
            )
        except Exception as exc:
            return self._set_health(
                RuntimeHealth(False, True, False, reason=f"Redis roundtrip failed: {exc}")
            )
        if not redis_ready:
            return self._set_health(
                RuntimeHealth(False, True, False, reason="Redis roundtrip failed")
            )
        try:
            self._resolve_agent(bus)
            if self.production and not bool(
                getattr(self.agent, "production_signing_configured", False)
            ):
                return self._set_health(
                    RuntimeHealth(
                        False,
                        True,
                        True,
                        reason=(
                            "production Agent signing requires "
                            "AGENT_MESSAGE_HMAC_SECRET or both "
                            "SIGSTORE_SIGN_COMMAND and SIGSTORE_VERIFY_COMMAND"
                        ),
                    )
                )
            targets = self.agent.runtime_targets()
            if not isinstance(targets, Mapping):
                raise TypeError("Agent runtime_targets() must return a mapping")
            if not targets:
                raise RuntimeError("Agent runtime must declare required domain targets")
            target_names = [str(name) for name in targets]
            target_results = await asyncio.gather(
                *(_check_target_health(target) for target in targets.values())
            )
            target_health = dict(zip(target_names, target_results, strict=True))
        except Exception as exc:
            return self._set_health(
                RuntimeHealth(
                    False,
                    True,
                    True,
                    reason=f"domain target readiness failed: {exc}",
                )
            )
        ready = all(target_health.values())
        reason = "" if ready else "one or more required domain targets are unhealthy"
        return self._set_health(RuntimeHealth(ready, True, True, target_health, reason))

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Agent runtime has been shut down")
            if self._started:
                return
            try:
                health = await self._check_readiness_locked()
                if not health.ready:
                    raise RuntimeError(health.reason or "Agent runtime is not ready")
                self.ready = False
                await self.agent.start()
            except BaseException:
                self._set_health(
                    RuntimeHealth(
                        False,
                        self.health.entry_point,
                        self.health.redis,
                        dict(self.health.targets),
                        self.health.reason,
                    )
                )
                raise
            self._agent_stopped = False
            self._started = True
            self.ready = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def run(self) -> None:
        self.install_signal_handlers()
        try:
            await self.start()
            await self._shutdown_event.wait()
        finally:
            await self.shutdown()

    def install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(signum, self._shutdown_event.set)

    async def shutdown(self) -> None:
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await self._lifecycle_lock.acquire()
                break
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
        try:
            await self._shutdown_locked(cancellation)
        finally:
            self._lifecycle_lock.release()

    async def _shutdown_locked(
        self,
        cancellation: asyncio.CancelledError | None = None,
    ) -> None:
        self._closed = True
        self._set_health(
            RuntimeHealth(
                False,
                self.health.entry_point,
                self._bus_connected,
                dict(self.health.targets),
                "Agent runtime has been shut down",
            )
        )
        errors: list[BaseException] = []
        if self._heartbeat_task is not None:
            heartbeat_task = self._heartbeat_task
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError as error:
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    cancellation = error
            except BaseException as error:
                _LOGGER.error(
                    "Agent heartbeat failed before shutdown",
                    exc_info=(type(error), error, error.__traceback__),
                )
                errors.append(error)
            self._heartbeat_task = None
        if self.agent is not None and not self._agent_stopped:
            try:
                await self.agent.stop()
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
            except BaseException as error:
                _LOGGER.error(
                    "Agent stop failed during shutdown",
                    exc_info=(type(error), error, error.__traceback__),
                )
                errors.append(error)
            else:
                self._agent_stopped = True
        if self.message_bus is not None:
            if self._bus_connected:
                try:
                    await self.message_bus.close()
                except asyncio.CancelledError as error:
                    cancellation = cancellation or error
                except BaseException as error:
                    _LOGGER.error(
                        "Agent message bus close failed during shutdown",
                        exc_info=(type(error), error, error.__traceback__),
                    )
                    errors.append(error)
                else:
                    self._bus_connected = False
        self._started = False
        if cancellation is not None:
            if errors:
                raise BaseExceptionGroup(
                    "Agent runtime shutdown failed",
                    [cancellation, *errors],
                )
            raise cancellation
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Agent runtime shutdown failed", errors)

    def _resolve_agent_factory(self) -> None:
        if self._agent_factory is not None:
            return
        self._agent_factory = self.entry_point_loader(self.agent_name)
        if not callable(self._agent_factory):
            raise TypeError(f"Agent entry point is not callable: {self.agent_name}")

    def _resolve_agent(self, message_bus) -> None:
        if self.agent is not None:
            return
        if self._agent_factory is None:
            raise RuntimeError("Agent entry point has not been resolved")
        self.agent = self._agent_factory(message_bus=message_bus)
        if not callable(getattr(self.agent, "start", None)):
            raise TypeError("Agent entry point must return an Agent with start()")
        if not callable(getattr(self.agent, "stop", None)):
            raise TypeError("Agent entry point must return an Agent with stop()")
        if not callable(getattr(self.agent, "runtime_targets", None)):
            raise TypeError("Agent entry point must return an Agent with runtime_targets()")

    def _resolve_bus(self):
        if self.message_bus is not None:
            return self.message_bus
        self.message_bus = RedisBus(
            url=os.environ.get("REDIS_URL") or None,
            allow_fallback=not self.production,
        )
        return self.message_bus

    def _set_health(self, health: RuntimeHealth) -> RuntimeHealth:
        self.health = health
        self.ready = health.ready
        return health

    async def _heartbeat_loop(self) -> None:
        subject = AGENT_ENTRY_POINTS[self.agent_name].removesuffix(".request") + ".heartbeat"
        while True:
            heartbeat = AgentHeartbeat(
                agent_name=self.agent_name,
                status="healthy" if self.ready else "unavailable",
                active_jobs=0,
            )
            await self.message_bus.publish(subject, heartbeat.SerializeToString())
            await asyncio.sleep(self.heartbeat_interval)


async def _check_target_health(target: Any) -> bool:
    if target is None:
        return False
    health_check = getattr(target, "health_check", None)
    if not callable(health_check):
        return False

    async def invoke() -> Any:
        if inspect.iscoroutinefunction(health_check):
            result = health_check()
        else:
            result = await run_health_probe_in_daemon(health_check)
        if inspect.isawaitable(result):
            result = await result
        return result

    try:
        result = await asyncio.wait_for(
            invoke(),
            timeout=agent_health_check_timeout_seconds(),
        )
    except Exception:
        return False
    if isinstance(result, Mapping):
        return result.get("healthy") is True
    return result is True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=tuple(AGENT_ENTRY_POINTS), required=True)
    return parser.parse_args()


async def _run_cli() -> None:
    args = _parse_args()
    await AgentRuntime(args.agent, production=True).run()


def main() -> None:
    asyncio.run(_run_cli())


if __name__ == "__main__":
    main()
