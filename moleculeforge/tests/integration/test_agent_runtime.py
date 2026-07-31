"""Independent Agent runtime readiness and lifecycle tests."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import shutil
import struct
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

TEST_AGENT_HMAC_SECRET = "task-3-agent-test-secret"
_CRG_AGENT_TYPES = (
    ("nl2obj.agent", "NL2ObjAgent"),
    ("generator_coord.agent", "GeneratorCoordAgent"),
    ("validation_agent.agent", "ValidationAgent"),
    ("retrosyn_agent.agent", "RetroSynAgent"),
    ("supply_agent.agent", "SupplyAgent"),
    ("srb_agent.agent", "SRBAgent"),
    ("critic_agent.agent", "ScientificCriticAgent"),
)


@pytest.fixture(autouse=True)
def _configure_agent_hmac_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MESSAGE_HMAC_SECRET", TEST_AGENT_HMAC_SECRET)


@dataclass
class FakeTarget:
    healthy: bool
    calls: int = 0

    async def health_check(self):
        self.calls += 1
        return {"healthy": self.healthy}


class LiteralHealthTarget:
    def __init__(self, value) -> None:
        self.value = value

    async def health_check(self):
        return {"healthy": self.value}


class FakeAgent:
    def __init__(self, targets=None) -> None:
        self.targets = targets or {}
        self.started = 0
        self.stopped = 0

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    def runtime_targets(self):
        return self.targets


class FakeBus:
    is_redis = True

    def __init__(self, roundtrip_ok: bool = True) -> None:
        self.roundtrip_ok = roundtrip_ok
        self.connected = 0
        self.closed = 0
        self.published = []

    async def connect(self) -> None:
        self.connected += 1

    async def roundtrip(self, timeout: float = 1.0) -> bool:
        return self.roundtrip_ok

    async def publish(self, subject: str, payload: bytes) -> None:
        self.published.append((subject, payload))

    async def close(self) -> None:
        self.closed += 1


def _grpc_client_types():
    from generator_coord.agent import GeneratorGrpcClient
    from nl2obj.agent import CIGCompilerGrpcClient
    from retrosyn_agent.agent import HUMURouteEncoderGrpcClient
    from supply_agent.agent import SupplyOracleGrpcClient
    from validation_agent.agent import OracleGrpcClient

    return {
        "cig": CIGCompilerGrpcClient,
        "generator": GeneratorGrpcClient,
        "oracle": OracleGrpcClient,
        "route_encoder": HUMURouteEncoderGrpcClient,
        "supply": SupplyOracleGrpcClient,
    }


def _crg_agent_kwargs(module_name: str, module, monkeypatch: pytest.MonkeyPatch) -> dict:
    if module_name == "generator_coord.agent":
        monkeypatch.setattr(module, "_build_generator_clients", lambda targets: {})
        return {"generator_clients": {}, "generator_targets": {}}
    if module_name == "validation_agent.agent":
        return {"oracles": {}}
    if module_name == "retrosyn_agent.agent":
        return {
            "route_planners": {"test": object()},
            "route_encoder_client": object(),
        }
    if module_name == "supply_agent.agent":
        return {"supply_client": object()}
    if module_name == "nl2obj.agent":
        return {"cig_compiler_client": object()}
    return {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "agent_type_name"),
    _CRG_AGENT_TYPES,
    ids=[module_name.split(".", 1)[0] for module_name, _ in _CRG_AGENT_TYPES],
)
async def test_agents_close_internally_created_crg_repository_once(
    module_name: str,
    agent_type_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    class Driver:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    module = importlib.import_module(module_name)
    driver = Driver()
    repository = GraphRepository(driver)
    monkeypatch.setattr(
        module,
        "build_shared_crg_repository_from_env",
        lambda: repository,
    )
    agent_type = getattr(module, agent_type_name)
    agent = agent_type(**_crg_agent_kwargs(module_name, module, monkeypatch))

    assert repository in agent.runtime_targets().values()

    await agent.stop()
    await agent.stop()

    assert driver.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "agent_type_name"),
    _CRG_AGENT_TYPES,
    ids=[module_name.split(".", 1)[0] for module_name, _ in _CRG_AGENT_TYPES],
)
async def test_agents_do_not_close_injected_crg_repository(
    module_name: str,
    agent_type_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_core.db.repositories.graph_repo import GraphRepository

    class Driver:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    module = importlib.import_module(module_name)
    driver = Driver()
    repository = GraphRepository(driver)

    def unexpected_factory_call():
        raise AssertionError("injected CRG repository must bypass the shared factory")

    monkeypatch.setattr(
        module,
        "build_shared_crg_repository_from_env",
        unexpected_factory_call,
    )
    agent_type = getattr(module, agent_type_name)
    agent = agent_type(
        crg_repository=repository,
        **_crg_agent_kwargs(module_name, module, monkeypatch),
    )

    assert repository not in agent.runtime_targets().values()

    await agent.stop()
    await agent.stop()

    assert driver.close_calls == 0




@pytest.mark.asyncio
async def test_base_agent_stop_closes_unique_targets_without_closing_bus() -> None:
    from mf_agents.base.agent import BaseAgent

    events: list[str] = []

    class Target:
        def __init__(self, name: str, *, failure: Exception | None = None) -> None:
            self.name = name
            self.failure = failure
            self.closed = 0

        async def close(self) -> None:
            self.closed += 1
            events.append(self.name)
            if self.failure is not None:
                raise self.failure

    class RuntimeAgent(BaseAgent):
        def __init__(self, message_bus, targets) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self.targets = targets

        def runtime_targets(self):
            return self.targets

    failed = Target("failed", failure=RuntimeError("target close failed"))
    healthy = Target("healthy")
    bus = FakeBus()
    agent = RuntimeAgent(
        bus,
        {
            "failed": failed,
            "duplicate": failed,
            "healthy": healthy,
        },
    )

    with pytest.raises(RuntimeError, match="target close failed"):
        await agent.stop()

    assert failed.closed == 1
    assert healthy.closed == 1
    assert bus.closed == 0
    assert events == ["failed", "healthy"]


@pytest.mark.asyncio
async def test_base_agent_stop_is_idempotent_after_success() -> None:
    from mf_agents.base.agent import BaseAgent

    class Target:
        def __init__(self) -> None:
            self.closed = 0

        async def close(self) -> None:
            self.closed += 1

    class RuntimeAgent(BaseAgent):
        def __init__(self, message_bus, target) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self.target = target

        def runtime_targets(self):
            return {"target": self.target}

    target = Target()
    bus = FakeBus()
    agent = RuntimeAgent(bus, target)

    await agent.stop()
    await agent.stop()

    assert target.closed == 1
    assert bus.closed == 0


@pytest.mark.asyncio
async def test_base_agent_stop_retries_only_cleanup_that_failed() -> None:
    from mf_agents.base.agent import BaseAgent

    class Target:
        def __init__(self, *, fail_once: bool = False) -> None:
            self.fail_once = fail_once
            self.closed = 0

        async def close(self) -> None:
            self.closed += 1
            if self.fail_once and self.closed == 1:
                raise RuntimeError("target close failed")

    class RuntimeAgent(BaseAgent):
        def __init__(self, message_bus, failed, healthy) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self.targets = {"failed": failed, "healthy": healthy}

        def runtime_targets(self):
            return self.targets

    failed = Target(fail_once=True)
    healthy = Target()
    bus = FakeBus()
    agent = RuntimeAgent(bus, failed, healthy)

    with pytest.raises(RuntimeError, match="target close failed"):
        await agent.stop()

    await agent.stop()
    await agent.stop()

    assert failed.closed == 2
    assert healthy.closed == 1
    assert bus.closed == 0


@pytest.mark.asyncio
async def test_base_agent_stop_retries_target_cancelled_during_close() -> None:
    from mf_agents.base.agent import BaseAgent

    class Target:
        def __init__(self) -> None:
            self.closed = 0

        async def close(self) -> None:
            self.closed += 1
            if self.closed == 1:
                raise asyncio.CancelledError

    class RuntimeAgent(BaseAgent):
        def __init__(self, message_bus, target) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self.target = target

        def runtime_targets(self):
            return {"target": self.target}

    target = Target()
    bus = FakeBus()
    agent = RuntimeAgent(bus, target)

    with pytest.raises(asyncio.CancelledError):
        await agent.stop()

    await agent.stop()

    assert target.closed == 2
    assert bus.closed == 0


@pytest.mark.asyncio
async def test_base_agent_concurrent_stop_closes_each_resource_once() -> None:
    from mf_agents.base.agent import BaseAgent

    class Target:
        def __init__(self) -> None:
            self.closed = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def close(self) -> None:
            self.closed += 1
            self.started.set()
            await self.release.wait()

    class RuntimeAgent(BaseAgent):
        def __init__(self, message_bus, target) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self.target = target

        def runtime_targets(self):
            return {"target": self.target}

    target = Target()
    bus = FakeBus()
    agent = RuntimeAgent(bus, target)
    first = asyncio.create_task(agent.stop())
    await asyncio.wait_for(target.started.wait(), timeout=0.1)
    second = asyncio.create_task(agent.stop())
    await asyncio.sleep(0)

    assert target.closed == 1

    target.release.set()
    await asyncio.gather(first, second)

    assert target.closed == 1
    assert bus.closed == 0


@pytest.mark.asyncio
async def test_base_agent_stop_unsubscribes_without_closing_bus_and_is_one_shot() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class CountingBus(InMemoryBus):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            await super().close()

    class SubscriptionAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = ["mf.first", "mf.second"]

    bus = CountingBus()
    await bus.connect()
    agent = SubscriptionAgent(bus)

    await agent.start()
    first_start_count = bus.callback_count
    await agent.stop()
    first_stop_state = (bus.callback_count, bus.close_calls, agent._started)
    await bus.publish("mf.first", b"still-connected")
    restart_error: RuntimeError | None = None
    try:
        await agent.start()
    except RuntimeError as error:
        restart_error = error
    await agent.stop()
    second_stop_state = (bus.callback_count, bus.close_calls, agent._started)
    await bus.close()

    assert first_start_count == 2
    assert first_stop_state == (0, 0, False)
    assert restart_error is not None
    assert str(restart_error) == "Agent has been stopped"
    assert second_stop_state == (0, 0, False)
    assert bus.close_calls == 1


@pytest.mark.asyncio
async def test_base_agent_rejects_start_while_target_cleanup_needs_retry() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class Target:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("target close failed")

    class SubscriptionAgent(BaseAgent):
        def __init__(self, message_bus, target) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = ["mf.target-cleanup"]
            self.target = target

        def runtime_targets(self):
            return {"target": self.target}

    bus = InMemoryBus()
    await bus.connect()
    target = Target()
    agent = SubscriptionAgent(bus, target)
    await agent.start()

    with pytest.raises(RuntimeError, match="target close failed"):
        await agent.stop()
    first_stop_state = (
        bus.callback_count,
        agent._started,
        target.close_calls,
    )
    restart_error: RuntimeError | None = None
    try:
        await agent.start()
    except RuntimeError as error:
        restart_error = error

    await agent.stop()
    await bus.close()

    assert first_stop_state == (0, False, 1)
    assert restart_error is not None
    assert str(restart_error) == "Agent has been stopped"
    assert agent._started is False
    assert target.close_calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("error", "cancel"))
async def test_base_agent_stop_retries_only_failed_unsubscribe(
    failure: str,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class FailOnceUnsubscribeBus(InMemoryBus):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False
            self.close_calls = 0
            self.unsubscribe_calls: list[str] = []

        async def unsubscribe(self, subscription) -> None:
            self.unsubscribe_calls.append(subscription.subject)
            if subscription.subject == "mf.first" and not self.failed:
                self.failed = True
                if failure == "cancel":
                    raise asyncio.CancelledError
                raise RuntimeError("unsubscribe failed")
            await super().unsubscribe(subscription)

        async def close(self) -> None:
            self.close_calls += 1
            await super().close()

    class SubscriptionAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = ["mf.first", "mf.second"]

    bus = FailOnceUnsubscribeBus()
    await bus.connect()
    agent = SubscriptionAgent(bus)
    await agent.start()

    expected_error = asyncio.CancelledError if failure == "cancel" else RuntimeError
    with pytest.raises(expected_error):
        await agent.stop()
    first_stop_state = (
        bus.callback_count,
        bus.close_calls,
        agent._started,
        tuple(bus.unsubscribe_calls),
    )

    await agent.stop()
    second_stop_state = (
        bus.callback_count,
        bus.close_calls,
        agent._started,
        tuple(bus.unsubscribe_calls),
    )
    await bus.close()

    assert first_stop_state == (
        1,
        0,
        False,
        ("mf.first", "mf.second"),
    )
    assert second_stop_state == (
        0,
        0,
        False,
        ("mf.first", "mf.second", "mf.first"),
    )
    assert bus.close_calls == 1


@pytest.mark.asyncio
async def test_base_agent_stop_drains_only_its_callbacks_before_closing_target() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class Target:
        def __init__(self, callback_finished: asyncio.Event) -> None:
            self.callback_finished = callback_finished
            self.closed_after_callback = False

        async def close(self) -> None:
            self.closed_after_callback = self.callback_finished.is_set()

    class BlockingAgent(BaseAgent):
        def __init__(self, name: str, subject: str, message_bus) -> None:
            super().__init__(name, message_bus=message_bus)
            self._subscription_subjects = [subject]
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.finished = asyncio.Event()
            self.target = Target(self.finished)

        async def process(self, payload: Mapping) -> Mapping:
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            finally:
                self.finished.set()
            return payload

        def runtime_targets(self):
            return {"target": self.target}

    bus = InMemoryBus()
    await bus.connect()
    first = BlockingAgent("first", "legacy.first", bus)
    second = BlockingAgent("second", "legacy.second", bus)
    await first.start()
    await second.start()
    payload = json.dumps(
        {
            "run_id": "run-drain",
            "request_id": "request-drain",
            "schema_version": "drain.v1",
        }
    ).encode()
    await bus.publish("legacy.first", payload)
    await bus.publish("legacy.second", payload)
    await asyncio.wait_for(first.started.wait(), timeout=0.05)
    await asyncio.wait_for(second.started.wait(), timeout=0.05)

    try:
        await first.stop()
        state_after_first_stop = (
            first.cancelled.is_set(),
            first.finished.is_set(),
            first.target.closed_after_callback,
            second.cancelled.is_set(),
            bus.callback_count,
            bus.callback_task_count,
        )
    finally:
        first.release.set()
        second.release.set()
        await asyncio.wait_for(second.finished.wait(), timeout=0.05)
        await second.stop()
        await bus.close()

    assert state_after_first_stop == (True, True, True, False, 1, 1)


@pytest.mark.asyncio
async def test_retrosyn_stop_closes_shared_planner_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from retrosyn_agent.agent import RetroSynAgent

    monkeypatch.delenv("HUMU_ENCODER_TARGET", raising=False)

    class Planner:
        def __init__(self) -> None:
            self.closed = 0

        def find_routes(self, smiles: str, max_routes: int) -> list[dict]:
            return []

        async def close(self) -> None:
            self.closed += 1

    planner = Planner()
    bus = FakeBus()
    agent = RetroSynAgent(
        message_bus=bus,
        route_planners={"primary": planner, "fallback": planner},
        crg_repository=object(),
    )

    await agent.stop()
    await agent.stop()

    assert planner.closed == 1
    assert bus.closed == 0


def _loader(agent):
    return lambda name: lambda message_bus=None: agent


def test_runtime_allows_all_existing_agent_entry_points() -> None:
    from mf_agents.runtime import AGENT_ENTRY_POINTS, load_agent_entry_point

    assert AGENT_ENTRY_POINTS == {
        "generator_coord": "agent.generator_coord.request",
        "validation": "agent.validation.request",
        "retrosyn": "agent.retrosyn.request",
        "supply": "agent.supply.request",
        "srb": "agent.srb.request",
        "critic": "agent.critic.request",
        "nl2obj": "agent.nl2obj.request",
    }
    for name in AGENT_ENTRY_POINTS:
        assert callable(load_agent_entry_point(name))
    with pytest.raises(LookupError, match="unsupported"):
        load_agent_entry_point("unknown")


def test_agent_runtime_extra_installs_loadable_factories_in_isolation(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    runtime_environment = tmp_path / "runtime-environment"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("VIRTUAL_ENV", None)
    environment["UV_PROJECT_ENVIRONMENT"] = str(runtime_environment)
    uv_executable = shutil.which("uv")
    assert uv_executable is not None

    sync_result = subprocess.run(  # noqa: S603
        [
            uv_executable,
            "sync",
            "--frozen",
            "--no-dev",
            "--extra",
            "agent-runtime",
            "--no-editable",
            "--refresh-package",
            "mf-retrosyn-aizynth",
            "--python",
            sys.executable,
        ],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert sync_result.returncode == 0, sync_result.stderr

    runtime_python = runtime_environment / "bin" / "python"
    inspection_script = """
import importlib
import json
from importlib.metadata import distribution, entry_points

expected_distributions = (
    "nl2obj",
    "generator-coord",
    "validation-agent",
    "retrosyn-agent",
    "supply-agent",
    "srb-agent",
    "critic-agent",
    "orchestrator",
)
implicit_modules = (
    "cig_compiler_svc.main",
    "mf_generators.uas.generator",
    "mf_oracles.boltz2.oracle",
    "mf_oracles.gnina.oracle",
    "mf_oracles.openfe.oracle",
    "mf_oracles.rdkit_oracle.oracle",
    "mf_retrosyn.aizynth.retrosyn",
)
installed_distributions = {
    name: distribution(name).metadata["Name"]
    for name in expected_distributions
}
imported_modules = [importlib.import_module(name).__name__ for name in implicit_modules]
agent_entry_points = {
    item.name: item
    for item in entry_points(group="moleculeforge.agents")
}
factory_kwargs = {
    "nl2obj": {
        "cig_compiler_client": object(),
        "crg_repository": object(),
    },
    "generator_coord": {
        "generator_clients": {"hfm_3d": object(), "fragfm": object()},
        "generator_targets": {},
        "router_client": object(),
        "teacher_adapter": object(),
        "crg_repository": object(),
    },
    "validation": {
        "oracles": {f"L{level}": object() for level in range(5)},
        "crg_repository": object(),
    },
    "retrosyn": {
        "route_planners": {"aizynth": object()},
        "route_encoder_client": object(),
        "crg_repository": object(),
    },
    "supply": {
        "supply_client": object(),
        "crg_repository": object(),
    },
    "srb": {"crg_repository": object()},
    "critic": {"crg_repository": object()},
}
instances = {
    name: agent_entry_points[name].load()(**factory_kwargs[name])
    for name in factory_kwargs
}
print(
    json.dumps(
        {
            "distributions": installed_distributions,
            "entry_points": {
                name: item.value
                for name, item in agent_entry_points.items()
            },
            "implicit_modules": imported_modules,
            "factory_types": {
                name: f"{type(instance).__module__}:{type(instance).__name__}"
                for name, instance in instances.items()
            },
        },
        sort_keys=True,
    )
)
"""
    inspection_result = subprocess.run(  # noqa: S603
        [str(runtime_python), "-c", inspection_script],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert inspection_result.returncode == 0, inspection_result.stderr
    result = json.loads(inspection_result.stdout)

    assert result["distributions"] == {
        "critic-agent": "critic-agent",
        "generator-coord": "generator-coord",
        "nl2obj": "nl2obj",
        "orchestrator": "orchestrator",
        "retrosyn-agent": "retrosyn-agent",
        "srb-agent": "srb-agent",
        "supply-agent": "supply-agent",
        "validation-agent": "validation-agent",
    }
    assert result["entry_points"] == {
        "critic": "critic_agent.agent:ScientificCriticAgent",
        "generator_coord": "generator_coord.agent:GeneratorCoordAgent",
        "nl2obj": "nl2obj.agent:NL2ObjAgent",
        "retrosyn": "retrosyn_agent.agent:RetroSynAgent",
        "srb": "srb_agent.agent:SRBAgent",
        "supply": "supply_agent.agent:SupplyAgent",
        "validation": "validation_agent.agent:ValidationAgent",
    }
    assert result["implicit_modules"] == [
        "cig_compiler_svc.main",
        "mf_generators.uas.generator",
        "mf_oracles.boltz2.oracle",
        "mf_oracles.gnina.oracle",
        "mf_oracles.openfe.oracle",
        "mf_oracles.rdkit_oracle.oracle",
        "mf_retrosyn.aizynth.retrosyn",
    ]
    assert result["factory_types"] == {
        "critic": "critic_agent.agent:ScientificCriticAgent",
        "generator_coord": "generator_coord.agent:GeneratorCoordAgent",
        "nl2obj": "nl2obj.agent:NL2ObjAgent",
        "retrosyn": "retrosyn_agent.agent:RetroSynAgent",
        "srb": "srb_agent.agent:SRBAgent",
        "supply": "supply_agent.agent:SupplyAgent",
        "validation": "validation_agent.agent:ValidationAgent",
    }








@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_name", "heartbeat_subject"),
    (("nl2obj", "agent.nl2obj.heartbeat"),),
)
async def test_runtime_starts_existing_generic_signed_entry_points(
    agent_name: str,
    heartbeat_subject: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.runtime import AgentRuntime, _parse_args

    monkeypatch.setattr(sys, "argv", ["mf-agent-runtime", "--agent", agent_name])
    parsed = _parse_args()
    agent = FakeAgent({"domain": FakeTarget(True)})
    bus = FakeBus()
    runtime = AgentRuntime(
        agent_name,
        entry_point_loader=_loader(agent),
        message_bus=bus,
        heartbeat_interval=60.0,
    )

    await runtime.start()
    await asyncio.sleep(0)
    await runtime.shutdown()

    assert parsed.agent == agent_name
    assert agent.started == 1
    assert agent.stopped == 1
    assert any(subject == heartbeat_subject for subject, _ in bus.published)


@pytest.mark.asyncio
async def test_readiness_is_false_when_entry_point_does_not_resolve() -> None:
    from mf_agents.runtime import AgentRuntime

    def unresolved(name):
        raise LookupError(name)

    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=unresolved,
        message_bus=FakeBus(),
    )

    health = await runtime.check_readiness()

    assert health.ready is False
    assert health.entry_point is False
    assert health.redis is False
    assert health.targets == {}


@pytest.mark.asyncio
async def test_readiness_is_false_when_redis_roundtrip_fails() -> None:
    from mf_agents.runtime import AgentRuntime

    target = FakeTarget(True)
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(FakeAgent({"generator": target})),
        message_bus=FakeBus(roundtrip_ok=False),
    )

    health = await runtime.check_readiness()

    assert health.ready is False
    assert health.entry_point is True
    assert health.redis is False
    assert health.targets == {}
    assert target.calls == 0


@pytest.mark.asyncio
async def test_runtime_enforces_hard_timeout_around_bus_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.runtime import AgentRuntime

    class HangingRoundtripBus(FakeBus):
        async def roundtrip(self, timeout: float = 1.0) -> bool:
            await asyncio.Event().wait()

    monkeypatch.setenv("AGENT_HEALTH_CHECK_TIMEOUT_SECONDS", "0.01")
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(FakeAgent({"generator": FakeTarget(True)})),
        message_bus=HangingRoundtripBus(),
    )

    health = await asyncio.wait_for(runtime.check_readiness(), timeout=0.06)

    assert health.ready is False
    assert health.redis is False
    assert "Redis roundtrip failed" in health.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "targets",
    [
        {"generator": None, "oracle": FakeTarget(True)},
        {"generator": FakeTarget(False), "oracle": FakeTarget(True)},
        {"generator": FakeTarget(True), "oracle": None},
        {"generator": FakeTarget(True), "oracle": FakeTarget(False)},
    ],
)
async def test_readiness_is_false_for_each_missing_or_unhealthy_target(targets) -> None:
    from mf_agents.runtime import AgentRuntime

    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(FakeAgent(targets)),
        message_bus=FakeBus(),
    )

    health = await runtime.check_readiness()

    assert health.ready is False
    assert health.entry_point is True
    assert health.redis is True
    assert set(health.targets) == {"generator", "oracle"}
    assert all(isinstance(value, bool) for value in health.targets.values())
    assert not all(health.targets.values())


@pytest.mark.asyncio
async def test_readiness_is_true_only_when_all_categories_succeed() -> None:
    from mf_agents.runtime import AgentRuntime

    targets = {"generator": FakeTarget(True), "oracle": FakeTarget(True)}
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(FakeAgent(targets)),
        message_bus=FakeBus(),
    )

    health = await runtime.check_readiness()

    assert health.ready is True
    assert health.entry_point is True
    assert health.redis is True
    assert health.targets == {"generator": True, "oracle": True}


@pytest.mark.asyncio
async def test_shutdown_waits_for_inflight_readiness_before_closing_runtime() -> None:
    from mf_agents.runtime import AgentRuntime

    class BlockingConnectBus(FakeBus):
        def __init__(self) -> None:
            super().__init__()
            self.connect_started = asyncio.Event()
            self.release_connect = asyncio.Event()

        async def connect(self) -> None:
            self.connected += 1
            self.connect_started.set()
            await self.release_connect.wait()

    agent = FakeAgent({"generator": FakeTarget(True)})
    bus = BlockingConnectBus()
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(agent),
        message_bus=bus,
    )
    readiness_task = asyncio.create_task(runtime.check_readiness())
    await asyncio.wait_for(bus.connect_started.wait(), timeout=0.05)
    shutdown_task = asyncio.create_task(runtime.shutdown())
    await asyncio.sleep(0)
    shutdown_waited_for_readiness = not shutdown_task.done()

    bus.release_connect.set()
    readiness = await readiness_task
    await shutdown_task

    assert shutdown_waited_for_readiness is True
    assert readiness.ready is True
    assert runtime.ready is False
    assert runtime.health.ready is False
    assert runtime._bus_connected is False
    assert agent.stopped == 1
    assert bus.connected == 1
    assert bus.closed == 1


@pytest.mark.asyncio
async def test_readiness_is_false_when_agent_declares_no_domain_targets() -> None:
    from mf_agents.runtime import AgentRuntime

    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(FakeAgent({})),
        message_bus=FakeBus(),
    )

    health = await runtime.check_readiness()

    assert health.ready is False
    assert "domain targets" in health.reason


@pytest.mark.asyncio
async def test_mapping_health_requires_literal_true() -> None:
    from mf_agents.runtime import AgentRuntime

    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(FakeAgent({"generator": LiteralHealthTarget(1)})),
        message_bus=FakeBus(),
    )

    health = await runtime.check_readiness()

    assert health.ready is False
    assert health.targets == {"generator": False}


@pytest.mark.asyncio
async def test_hanging_domain_health_probe_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.runtime import AgentRuntime

    class HangingTarget:
        async def health_check(self):
            await asyncio.Event().wait()

    monkeypatch.setenv("AGENT_HEALTH_CHECK_TIMEOUT_SECONDS", "0.01")
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(FakeAgent({"generator": HangingTarget()})),
        message_bus=FakeBus(),
    )
    readiness_task = asyncio.create_task(runtime.check_readiness())

    await asyncio.sleep(0.03)

    assert readiness_task.done()
    health = await readiness_task
    assert health.ready is False
    assert health.targets == {"generator": False}
    await asyncio.wait_for(runtime.shutdown(), timeout=0.05)


def test_sync_health_timeout_does_not_delay_asyncio_run_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.runtime import _check_target_health

    class BlockingTarget:
        def health_check(self):
            time.sleep(0.2)
            return {"healthy": True}

    monkeypatch.setenv("AGENT_HEALTH_CHECK_TIMEOUT_SECONDS", "0.01")
    started = time.monotonic()

    healthy = asyncio.run(_check_target_health(BlockingTarget()))

    assert healthy is False
    assert time.monotonic() - started < 0.15


@pytest.mark.asyncio
async def test_production_runtime_requires_secure_agent_signing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.runtime import AgentRuntime

    monkeypatch.delenv("AGENT_MESSAGE_HMAC_SECRET")

    class RuntimeAgent(BaseAgent):
        def runtime_targets(self):
            return {"generator": FakeTarget(True)}

    agent = RuntimeAgent("generator_coord")
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(agent),
        message_bus=FakeBus(),
        production=True,
    )

    health = await runtime.check_readiness()

    assert health.ready is False
    assert "sign" in health.reason.lower()


@pytest.mark.asyncio
async def test_production_runtime_accepts_explicit_hmac_secret() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.runtime import AgentRuntime

    class RuntimeAgent(BaseAgent):
        def runtime_targets(self):
            return {"generator": FakeTarget(True)}

    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(RuntimeAgent("generator_coord")),
        message_bus=FakeBus(),
        production=True,
    )

    health = await runtime.check_readiness()

    assert health.ready is True


@pytest.mark.asyncio
async def test_production_runtime_rejects_unavailable_sigstore_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.runtime import AgentRuntime

    monkeypatch.delenv("AGENT_MESSAGE_HMAC_SECRET")
    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", "missing-sign-command --json")
    monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", "missing-verify-command --json")

    class RuntimeAgent(BaseAgent):
        def runtime_targets(self):
            return {"generator": FakeTarget(True)}

    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(RuntimeAgent("generator_coord")),
        message_bus=FakeBus(),
        production=True,
    )

    health = await runtime.check_readiness()

    assert health.ready is False
    assert "sign" in health.reason.lower()


def test_generator_runtime_targets_include_required_default_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generator_coord.agent import GeneratorCoordAgent

    for name in (
        "GENERATOR_DISCOVERY_URI",
        "GENERATOR_CLIENT_TARGETS",
        "HFM_3D_GENERATOR_TARGET",
        "FRAGFM_GENERATOR_TARGET",
    ):
        monkeypatch.delenv(name, raising=False)
    optional = FakeTarget(True)
    teacher = FakeTarget(True)
    agent = GeneratorCoordAgent(
        generator_clients={"uas": optional},
        teacher_adapter=teacher,
        crg_repository=object(),
    )

    assert agent.runtime_targets() == {
        "generator.fragfm": None,
        "generator.hfm_3d": None,
        "generator.uas": optional,
        "teacher": teacher,
    }


@pytest.mark.asyncio
async def test_uas_compatibility_health_never_executes_or_reports_ready(tmp_path) -> None:
    from generator_coord.agent import UASLocalGeneratorClient

    marker = tmp_path / "uas-health.json"
    runner = tmp_path / "uas.py"
    runner.write_text(
        "import json,sys\n"
        "request=json.load(sys.stdin)\n"
        f"open({str(marker)!r},'w').write(json.dumps(request,sort_keys=True))\n"
        "print(json.dumps({'candidates': []}))\n",
        encoding="utf-8",
    )

    health = await UASLocalGeneratorClient(f"{sys.executable} {runner}").health_check()

    assert not marker.exists()
    assert health == {
        "healthy": False,
        "generator_name": "uas",
        "version": "0.1.0",
        "reason": "local UAS compatibility client is never production READY",
    }


def test_validation_runtime_targets_accept_supported_string_level_keys() -> None:
    from validation_agent.agent import ValidationAgent

    targets = {f"L{level}": FakeTarget(True) for level in range(5)}
    agent = ValidationAgent(oracles=targets, crg_repository=object())

    assert agent.runtime_targets() == {
        f"oracle.L{level}": targets[f"L{level}"] for level in range(5)
    }


@pytest.mark.asyncio
async def test_local_oracle_health_check_invokes_oracle_protocol() -> None:
    from validation_agent.agent import _BatchEvaluateOnlyOracle

    class Oracle:
        def __init__(self) -> None:
            self.calls = []

        async def evaluate(self, molecules, properties):
            self.calls.append((molecules, properties))
            return {"C": {"admet_score": 0.0}}

    oracle = Oracle()

    health = await _BatchEvaluateOnlyOracle(oracle, level=0).health_check()

    assert health == {"healthy": True}
    assert oracle.calls == [(["C"], ["admet_score"])]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scores",
    (
        {},
        {"admet_score": "0.5"},
        {"admet_score": float("nan")},
        {"admet_score": float("inf")},
    ),
)
async def test_local_oracle_health_requires_finite_level_score(scores) -> None:
    from validation_agent.agent import _BatchEvaluateOnlyOracle

    class Oracle:
        async def evaluate(self, molecules, properties):
            return {"C": scores}

    health = await _BatchEvaluateOnlyOracle(Oracle(), level=0).health_check()

    assert health == {"healthy": False}


@pytest.mark.asyncio
async def test_grpc_health_checks_send_deadlines_and_nonempty_protocols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generator_coord.agent import GeneratorGrpcClient
    from mf_core.proto_gen.moleculeforge.v1.core import audit_pb2
    from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2
    from retrosyn_agent.agent import HUMURouteEncoderGrpcClient
    from supply_agent.agent import SupplyOracleGrpcClient
    from validation_agent.agent import OracleGrpcClient

    monkeypatch.setenv("AGENT_HEALTH_CHECK_TIMEOUT_SECONDS", "0.25")
    calls = {}

    class GeneratorStub:
        async def Info(self, request, timeout=None):
            calls["generator"] = (request, timeout)
            return generator_pb2.GeneratorInfo(
                name="hfm_3d",
                version="1",
                max_batch_size=1,
                requires_gpu=False,
                runtime_status=audit_pb2.GENERATOR_RUNTIME_STATUS_READY,
                artifacts=[
                    audit_pb2.ArtifactRef(
                        name="checkpoint",
                        version="1",
                        checksum="sha256:test",
                        required=True,
                    )
                ],
            )

    generator = GeneratorGrpcClient.__new__(GeneratorGrpcClient)
    generator.stub = GeneratorStub()

    class OracleStub:
        async def Evaluate(self, request, timeout=None):
            calls["oracle"] = (request, timeout)
            raise AssertionError("oracle readiness must not execute Evaluate")

    class OracleChannel:
        async def channel_ready(self):
            calls["oracle_channel"] = True

    oracle = OracleGrpcClient.__new__(OracleGrpcClient)
    oracle.target = "unused"
    oracle.level = 0
    oracle.oracle_name = "rdkit"
    oracle.channel = OracleChannel()
    oracle.stub = OracleStub()

    class EncoderStub:
        async def Encode(self, request, timeout=None):
            calls["encoder"] = (request, timeout)
            return SimpleNamespace(
                humu_embedding=struct.pack("<129f", 1.0, *([0.0] * 128)),
                curvature=1.0,
            )

    encoder = HUMURouteEncoderGrpcClient.__new__(HUMURouteEncoderGrpcClient)
    encoder.stub = EncoderStub()

    supply_pb2 = ModuleType("mf_core.proto_gen.moleculeforge.v1.oracle.supply_pb2")
    supply_pb2.AvailabilityRequest = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(
        sys.modules,
        "mf_core.proto_gen.moleculeforge.v1.oracle.supply_pb2",
        supply_pb2,
    )

    class SupplyStub:
        async def CheckAvailability(self, request, timeout=None):
            calls["supply"] = (request, timeout)
            return SimpleNamespace(smiles="C")

    supply = SupplyOracleGrpcClient.__new__(SupplyOracleGrpcClient)
    supply.stub = SupplyStub()

    assert (await generator.health_check())["healthy"] is True
    assert await oracle.health_check() == {"healthy": True}
    assert await encoder.health_check() == {"healthy": True}
    assert await supply.health_check() == {"healthy": True}
    assert calls["generator"][1] == 0.25
    assert calls["oracle_channel"] is True
    assert "oracle" not in calls
    assert calls["encoder"][1] == 0.25
    assert calls["encoder"][0].entity_type == "route"
    assert json.loads(calls["encoder"][0].input_data) == {
        "reactions": ["C>>C"],
        "target_smiles": "C",
    }
    assert calls["supply"][1] == 0.25
    assert calls["supply"][0].smiles == "C"
    assert calls["supply"][0].request_id == "supply-health"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_name",
    ("cig", "generator", "oracle", "route_encoder", "supply"),
)
async def test_grpc_client_close_is_idempotent(client_name: str) -> None:
    class Channel:
        def __init__(self) -> None:
            self.closed = 0

        async def close(self) -> None:
            self.closed += 1

    channel = Channel()
    client_type = _grpc_client_types()[client_name]
    client = client_type.__new__(client_type)
    client.channel = channel

    await client.close()
    await client.close()

    assert channel.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_name",
    ("cig", "generator", "oracle", "route_encoder", "supply"),
)
async def test_grpc_client_close_retries_failed_channel_close(client_name: str) -> None:
    class Channel:
        def __init__(self) -> None:
            self.closed = 0

        async def close(self) -> None:
            self.closed += 1
            if self.closed == 1:
                raise RuntimeError("channel close failed")

    channel = Channel()
    client_type = _grpc_client_types()[client_name]
    client = client_type.__new__(client_type)
    client.channel = channel

    with pytest.raises(RuntimeError, match="channel close failed"):
        await client.close()
    await client.close()
    await client.close()

    assert channel.closed == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_name",
    ("cig", "generator", "oracle", "route_encoder", "supply"),
)
async def test_grpc_client_concurrent_close_closes_channel_once(client_name: str) -> None:
    class Channel:
        def __init__(self) -> None:
            self.closed = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def close(self) -> None:
            self.closed += 1
            self.started.set()
            await self.release.wait()

    channel = Channel()
    client_type = _grpc_client_types()[client_name]
    client = client_type.__new__(client_type)
    client.channel = channel
    first = asyncio.create_task(client.close())
    await asyncio.wait_for(channel.started.wait(), timeout=0.1)
    second = asyncio.create_task(client.close())
    await asyncio.sleep(0)

    assert channel.closed == 1

    channel.release.set()
    await asyncio.gather(first, second)
    await client.close()

    assert channel.closed == 1


@pytest.mark.asyncio
async def test_nl2obj_stop_closes_cig_channel_without_closing_bus() -> None:
    from nl2obj.agent import CIGCompilerGrpcClient, NL2ObjAgent

    class Channel:
        def __init__(self) -> None:
            self.closed = 0

        async def close(self) -> None:
            self.closed += 1

    channel = Channel()
    client = CIGCompilerGrpcClient.__new__(CIGCompilerGrpcClient)
    client.channel = channel
    bus = FakeBus()
    agent = NL2ObjAgent(
        message_bus=bus,
        cig_compiler_client=client,
        crg_repository=object(),
    )

    await agent.stop()
    await agent.stop()

    assert channel.closed == 1
    assert bus.closed == 0


@pytest.mark.asyncio
async def test_stopping_affected_agents_closes_owned_channels_without_buses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generator_coord.agent import GeneratorCoordAgent, GeneratorGrpcClient
    from mf_core.routing.task_router import GENERATOR_NAMES
    from retrosyn_agent.agent import HUMURouteEncoderGrpcClient, RetroSynAgent
    from supply_agent.agent import SupplyAgent, SupplyOracleGrpcClient
    from validation_agent.agent import OracleGrpcClient, ValidationAgent

    for name in ("GENERATOR_DISCOVERY_URI", "GENERATOR_CLIENT_TARGETS"):
        monkeypatch.delenv(name, raising=False)
    for generator_name in GENERATOR_NAMES:
        monkeypatch.delenv(f"{generator_name.upper()}_GENERATOR_TARGET", raising=False)

    class Channel:
        def __init__(self) -> None:
            self.closed = 0

        async def close(self) -> None:
            self.closed += 1

    def client_with_channel(client_type):
        channel = Channel()
        client = client_type.__new__(client_type)
        client.channel = channel
        return client, channel

    generator_client, generator_channel = client_with_channel(GeneratorGrpcClient)
    oracle_client, oracle_channel = client_with_channel(OracleGrpcClient)
    route_encoder_client, route_encoder_channel = client_with_channel(HUMURouteEncoderGrpcClient)
    supply_client, supply_channel = client_with_channel(SupplyOracleGrpcClient)
    buses = [FakeBus() for _ in range(4)]
    agents = [
        GeneratorCoordAgent(
            message_bus=buses[0],
            generator_clients={
                "hfm_3d": generator_client,
                "fragfm": generator_client,
            },
            generator_targets={},
            crg_repository=object(),
        ),
        ValidationAgent(
            message_bus=buses[1],
            oracles={
                0: oracle_client,
                1: oracle_client,
                2: None,
                3: None,
                4: None,
            },
            crg_repository=object(),
        ),
        RetroSynAgent(
            message_bus=buses[2],
            route_planners={"planner": object()},
            route_encoder_client=route_encoder_client,
            crg_repository=object(),
        ),
        SupplyAgent(
            message_bus=buses[3],
            supply_client=supply_client,
            crg_repository=object(),
        ),
    ]

    for agent in agents:
        await agent.stop()
        await agent.stop()

    assert [
        generator_channel.closed,
        oracle_channel.closed,
        route_encoder_channel.closed,
        supply_channel.closed,
    ] == [1, 1, 1, 1]
    assert [bus.closed for bus in buses] == [0, 0, 0, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_name", ("oracle", "planner", "supply"))
async def test_runtime_target_wrappers_forward_close(wrapper_name: str) -> None:
    from retrosyn_agent.agent import _PlannerHealthTarget
    from supply_agent.agent import _SupplyClientTarget
    from validation_agent.agent import _BatchEvaluateOnlyOracle

    class Client:
        def __init__(self) -> None:
            self.closed = 0

        async def close(self) -> None:
            self.closed += 1

    client = Client()
    wrappers = {
        "oracle": _BatchEvaluateOnlyOracle(client, oracle_name="rdkit"),
        "planner": _PlannerHealthTarget(client),
        "supply": _SupplyClientTarget(client),
    }

    await wrappers[wrapper_name].close()

    assert client.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("embedding", "curvature"),
    (
        (b"x", 1.0),
        (struct.pack("<128f", *([0.0] * 128)), 1.0),
        (struct.pack("<129f", float("nan"), *([0.0] * 128)), 1.0),
        (struct.pack("<129f", *([0.0] * 129)), 1.0),
        (struct.pack("<129f", 1.0, *([0.0] * 128)), 0.0),
    ),
)
async def test_humu_health_requires_legal_129_float32_embedding(
    embedding: bytes,
    curvature: float,
) -> None:
    from retrosyn_agent.agent import HUMURouteEncoderGrpcClient

    class EncoderStub:
        async def Encode(self, request, timeout=None):
            return SimpleNamespace(humu_embedding=embedding, curvature=curvature)

    encoder = HUMURouteEncoderGrpcClient.__new__(HUMURouteEncoderGrpcClient)
    encoder.stub = EncoderStub()

    assert await encoder.health_check() == {"healthy": False}


@pytest.mark.parametrize("async_wrapper", (False, True))
def test_local_oracle_timeout_does_not_delay_asyncio_run_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    async_wrapper: bool,
) -> None:
    from mf_agents.runtime import _check_target_health
    from validation_agent.agent import _BatchEvaluateOnlyOracle

    if async_wrapper:

        class BlockingOracle:
            async def evaluate(self, molecules, properties):
                time.sleep(0.2)
                return {"C": {"admet_score": 0.0}}

    else:

        class BlockingOracle:
            def evaluate(self, molecules, properties):
                time.sleep(0.2)
                return {"C": {"admet_score": 0.0}}

    monkeypatch.setenv("AGENT_HEALTH_CHECK_TIMEOUT_SECONDS", "0.01")
    started = time.monotonic()

    healthy = asyncio.run(_check_target_health(_BatchEvaluateOnlyOracle(BlockingOracle(), level=0)))

    assert healthy is False
    assert time.monotonic() - started < 0.15


@pytest.mark.parametrize("async_wrapper", (False, True))
def test_supply_client_timeout_does_not_delay_asyncio_run_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    async_wrapper: bool,
) -> None:
    from mf_agents.runtime import _check_target_health
    from supply_agent.agent import _SupplyClientTarget

    if async_wrapper:

        class BlockingSupplyClient:
            async def check_availability(self, smiles):
                time.sleep(0.2)
                return {"smiles": smiles}

    else:

        class BlockingSupplyClient:
            def check_availability(self, smiles):
                time.sleep(0.2)
                return {"smiles": smiles}

    monkeypatch.setenv("AGENT_HEALTH_CHECK_TIMEOUT_SECONDS", "0.01")
    started = time.monotonic()

    healthy = asyncio.run(_check_target_health(_SupplyClientTarget(BlockingSupplyClient())))

    assert healthy is False
    assert time.monotonic() - started < 0.15


def test_validation_default_oracles_do_not_restore_legacy_quantum_l4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from validation_agent.agent import _build_default_oracles

    for variable in (
        "L1_ADMET_ORACLE_TARGET",
        "L1_BOLTZ2_ORACLE_TARGET",
        "L2_DOCK_ORACLE_TARGET",
        "L3_FEP_ORACLE_TARGET",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("L4_QUANTUM_ORACLE_COMMAND", "quantum-runner")

    assert set(_build_default_oracles()) == {"rdkit"}


@pytest.mark.asyncio
async def test_retrosyn_command_health_check_invokes_planning_protocol(tmp_path) -> None:
    from retrosyn_agent.agent import ExternalCommandRetrosynPlanner

    marker = tmp_path / "planner-health.json"
    runner = tmp_path / "planner.py"
    runner.write_text(
        "import json,sys\n"
        "request=json.load(sys.stdin)\n"
        f"open({str(marker)!r},'w').write(json.dumps(request,sort_keys=True))\n"
        "print(json.dumps({'routes': []}))\n",
        encoding="utf-8",
    )

    health = await ExternalCommandRetrosynPlanner(f"{sys.executable} {runner}").health_check()

    assert health == {"healthy": True}
    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "engine": "external_command",
        "max_routes": 1,
        "smiles": "C",
    }


@pytest.mark.asyncio
async def test_retrosyn_agent_uses_configured_grpc_service_and_preserves_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import grpc
    from mf_core.proto_gen.moleculeforge.v1.retrosyn import retrosyn_pb2, retrosyn_pb2_grpc
    from retrosyn_agent.agent import RetroSynAgent

    received = []

    class Service(retrosyn_pb2_grpc.RetrosynServiceServicer):
        async def FindRoutes(self, request, context):
            received.append(request)
            return retrosyn_pb2.RetrosynthesisResponse(
                request_id=request.request_id,
                project_id=request.project_id,
                run_id=request.run_id,
                candidate_id=request.candidate_id,
                candidate_index=request.candidate_index,
                canonical_smiles=request.canonical_smiles,
                total_routes_found=1,
                routes=[
                    retrosyn_pb2.SyntheticRoute(
                        route_id="route-remote",
                        predicted_yield=0.8,
                        reaction_smiles=["CO.C>>CCO"],
                        steps=[
                            retrosyn_pb2.SyntheticRouteStep(
                                step_id="step-1",
                                reaction="CO.C>>CCO",
                                reaction_type="coupling",
                                reactants=[
                                    {"smiles": "CO", "amount_mmol": 1.0},
                                    {"smiles": "C", "amount_mmol": 1.2},
                                ],
                                conditions={
                                    "temperature_C": 25.0,
                                    "time_h": 2.0,
                                    "source": "planner",
                                },
                                reagents=["base", "catalyst"],
                                purification="column_chromatography",
                                operation="add",
                                building_blocks=[
                                    {"smiles": "CO"},
                                    {"smiles": "C"},
                                ],
                                yield_fraction=0.73,
                            )
                        ],
                    )
                ],
            )

    class CRGRepository:
        def __init__(self) -> None:
            self.beliefs = []

        async def write_workflow_belief(self, **kwargs) -> None:
            self.beliefs.append(kwargs)

    server = grpc.aio.server()
    retrosyn_pb2_grpc.add_RetrosynServiceServicer_to_server(Service(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    monkeypatch.setenv("RETROSYN_SERVICE_TARGET", f"127.0.0.1:{port}")
    monkeypatch.delenv("HUMU_ENCODER_TARGET", raising=False)
    for name in (
        "RETROSYN_PLANNER_COMMAND",
        "RETROSYN_PLANNER_COMMANDS_JSON",
        "RASCORE_PLANNER_COMMAND",
        "RSGPT_PLANNER_COMMAND",
        "UALIGN_PLANNER_COMMAND",
        "AIZYNTH_PLANNER_COMMAND",
        "AIZYNTH_CONFIG_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    repository = CRGRepository()
    agent = RetroSynAgent(crg_repository=repository)
    try:
        result = await agent.process(
            {
                "project_id": "project-1",
                "run_id": "run-1",
                "request_id": "request-1",
                "candidate_id": "candidate-1",
                "candidate_index": 2,
                "canonical_smiles": "CCO",
                "smiles": "CCO",
                "max_routes": 1,
                "engine": "rsgpt",
            }
        )
    finally:
        await agent.stop()
        await server.stop(None)

    assert len(received) == 1
    assert received[0].candidate_id == "candidate-1"
    assert received[0].candidate_index == 2
    assert received[0].engine == "rsgpt"
    assert result["project_id"] == "project-1"
    assert result["candidate_id"] == "candidate-1"
    assert result["candidate_index"] == 2
    assert result["canonical_smiles"] == "CCO"
    assert result["routes"][0]["steps"] == [
        {
            "step_id": "step-1",
            "reaction": "CO.C>>CCO",
            "reaction_type": "coupling",
            "reactants": [
                {"smiles": "CO", "amount_mmol": 1.0},
                {"smiles": "C", "amount_mmol": 1.2},
            ],
            "conditions": {
                "source": "planner",
                "temperature_C": 25.0,
                "time_h": 2.0,
            },
            "reagents": ["base", "catalyst"],
            "purification": "column_chromatography",
            "operation": "add",
            "building_blocks": [{"smiles": "CO"}, {"smiles": "C"}],
            "yield": 0.73,
        }
    ]
    assert result["layers"]["strategy"]["engine"] == "retrosyn_service"


@pytest.mark.asyncio
async def test_retrosyn_grpc_client_rejects_changed_candidate_identity() -> None:
    import grpc
    from mf_core.proto_gen.moleculeforge.v1.retrosyn import retrosyn_pb2, retrosyn_pb2_grpc
    from retrosyn_agent.agent import RetrosynGrpcClient

    class Service(retrosyn_pb2_grpc.RetrosynServiceServicer):
        async def FindRoutes(self, request, context):
            return retrosyn_pb2.RetrosynthesisResponse(
                request_id=request.request_id,
                project_id=request.project_id,
                run_id=request.run_id,
                candidate_id="different-candidate",
                candidate_index=request.candidate_index,
                canonical_smiles=request.canonical_smiles,
            )

    server = grpc.aio.server()
    retrosyn_pb2_grpc.add_RetrosynServiceServicer_to_server(Service(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    client = RetrosynGrpcClient(f"127.0.0.1:{port}")
    try:
        with pytest.raises(RuntimeError, match="candidate_id"):
            await client.plan_routes(
                "CCO",
                max_routes=1,
                request_context={
                    "engine": "rsgpt",
                    "project_id": "project-1",
                    "run_id": "run-1",
                    "request_id": "request-1",
                    "candidate_id": "candidate-1",
                    "candidate_index": 0,
                    "canonical_smiles": "CCO",
                },
            )
    finally:
        await client.close()
        await server.stop(None)


@pytest.mark.asyncio
async def test_retrosyn_agent_explicit_planner_overrides_configured_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from retrosyn_agent.agent import RetroSynAgent

    class Planner:
        async def find_routes(self, smiles: str, max_routes: int) -> list[dict]:
            return [
                {
                    "route_id": "route-injected",
                    "steps": [
                        {
                            "step_id": "step-1",
                            "reaction": "CO.C>>CCO",
                            "reaction_type": "coupling",
                            "reactants": [
                                {"smiles": "CO", "amount_mmol": 1.0},
                                {"smiles": "C", "amount_mmol": 1.2},
                            ],
                            "conditions": {
                                "temperature_C": 25.0,
                                "time_h": 2.0,
                                "source": "injected",
                            },
                            "yield": 0.8,
                            "building_blocks": [{"smiles": "CO"}, {"smiles": "C"}],
                        }
                    ],
                }
            ]

    monkeypatch.setenv("RETROSYN_SERVICE_TARGET", "127.0.0.1:1")
    monkeypatch.delenv("HUMU_ENCODER_TARGET", raising=False)
    agent = RetroSynAgent(planner=Planner(), crg_repository=None)

    result = await agent.process({"smiles": "CCO", "max_routes": 1})

    assert result["routes"][0]["route_id"] == "route-injected"
    assert "retrosyn_service" not in agent.runtime_targets()


@pytest.mark.asyncio
async def test_retrosyn_runtime_targets_probe_configured_aizynth_business_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from retrosyn_agent.agent import RetroSynAgent

    calls = []

    class Planner:
        async def find_routes(self, smiles, max_routes=10):
            calls.append((smiles, max_routes))
            return []

    class AiZynthRetrosyn:
        @classmethod
        def from_env(cls):
            return Planner()

    module = ModuleType("mf_retrosyn.aizynth.retrosyn")
    module.AiZynthRetrosyn = AiZynthRetrosyn
    monkeypatch.setitem(sys.modules, "mf_retrosyn.aizynth.retrosyn", module)
    monkeypatch.setenv("AIZYNTH_CONFIG_PATH", "/configured/aizynth.yml")
    for name in (
        "RETROSYN_PLANNER_COMMAND",
        "RETROSYN_PLANNER_COMMANDS_JSON",
        "RASCORE_PLANNER_COMMAND",
        "RSGPT_PLANNER_COMMAND",
        "UALIGN_PLANNER_COMMAND",
        "AIZYNTH_PLANNER_COMMAND",
    ):
        monkeypatch.delenv(name, raising=False)
    agent = RetroSynAgent(route_encoder_client=FakeTarget(True), crg_repository=object())

    target = agent.runtime_targets()["planner"]
    health = await target.health_check()

    assert health == {"healthy": True}
    assert calls == [("C", 1)]


@pytest.mark.asyncio
@pytest.mark.parametrize("async_wrapper", (False, True))
async def test_retrosyn_planner_health_probe_respects_runtime_timeout(
    monkeypatch: pytest.MonkeyPatch,
    async_wrapper: bool,
) -> None:
    from mf_agents.runtime import _check_target_health
    from retrosyn_agent.agent import _PlannerHealthTarget

    if async_wrapper:

        class BlockingPlanner:
            async def find_routes(self, smiles, max_routes=10):
                time.sleep(0.2)
                return []

    else:

        class BlockingPlanner:
            def find_routes(self, smiles, max_routes=10):
                time.sleep(0.2)
                return []

    monkeypatch.setenv("AGENT_HEALTH_CHECK_TIMEOUT_SECONDS", "0.01")
    started = time.monotonic()

    healthy = await _check_target_health(_PlannerHealthTarget(BlockingPlanner()))

    assert healthy is False
    assert time.monotonic() - started < 0.15


def test_retrosyn_planner_timeout_does_not_delay_asyncio_run_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.runtime import _check_target_health
    from retrosyn_agent.agent import _PlannerHealthTarget

    class BlockingPlanner:
        async def find_routes(self, smiles, max_routes=10):
            time.sleep(0.2)
            return []

    monkeypatch.setenv("AGENT_HEALTH_CHECK_TIMEOUT_SECONDS", "0.01")
    started = time.monotonic()

    healthy = asyncio.run(_check_target_health(_PlannerHealthTarget(BlockingPlanner())))

    assert healthy is False
    assert time.monotonic() - started < 0.15


@pytest.mark.asyncio
async def test_critic_rule_registry_health_requires_complete_valid_contract() -> None:
    from critic_agent.agent import _RuleRegistryTarget

    class Rule:
        def __init__(self, rule_id: str) -> None:
            self.rule_id = rule_id

        def evaluate(self, smiles, properties):
            return {}

    assert await _RuleRegistryTarget([object()]).health_check() == {"healthy": False}
    assert await _RuleRegistryTarget([Rule("duplicate"), Rule("duplicate")]).health_check() == {
        "healthy": False
    }
    assert await _RuleRegistryTarget(
        [Rule("valid")],
        load_failures=["rule module failed"],
    ).health_check() == {"healthy": False}
    assert await _RuleRegistryTarget([Rule("valid")]).health_check() == {"healthy": True}


@pytest.mark.asyncio
async def test_sila2_health_check_invokes_no_step_dry_run_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from srb_agent.agent import SRBAgent

    marker = tmp_path / "sila2-health.json"
    runner = tmp_path / "sila2.py"
    runner.write_text(
        "import json,sys\n"
        "request=json.load(sys.stdin)\n"
        "assert request['dry_run'] is True\n"
        "assert request['sila2_plan']['steps'] == []\n"
        f"open({str(marker)!r},'w').write(json.dumps(request,sort_keys=True))\n"
        "print(json.dumps({'healthy': True}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SILA2_PLAN_COMMAND", f"{sys.executable} {runner}")

    target = SRBAgent(crg_repository=object()).runtime_targets()["sila2"]
    health = await target.health_check()

    assert health == {"healthy": True}
    request = json.loads(marker.read_text(encoding="utf-8"))
    assert request["health_check"] is True
    assert request["target_smiles"] == "C"


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ("{}", "not-json"))
async def test_sila2_health_check_rejects_nonaffirmative_responses(
    output: str,
    tmp_path,
) -> None:
    from srb_agent.agent import _Sila2CommandTarget

    runner = tmp_path / "sila2-unhealthy.py"
    runner.write_text(
        f"print({output!r})\n",
        encoding="utf-8",
    )

    health = await _Sila2CommandTarget(f"{sys.executable} {runner}").health_check()

    assert health == {"healthy": False}


@pytest.mark.asyncio
async def test_production_runtime_rejects_non_redis_bus() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.runtime import AgentRuntime

    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(FakeAgent()),
        message_bus=InMemoryBus(),
        production=True,
    )

    health = await runtime.check_readiness()

    assert health.ready is False
    assert health.redis is False
    assert "Redis" in health.reason


@pytest.mark.asyncio
async def test_runtime_emits_heartbeat_and_shutdown_cleans_agent_and_bus() -> None:
    from mf_agents.runtime import AgentRuntime

    agent = FakeAgent({"generator": FakeTarget(True)})
    bus = FakeBus()
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(agent),
        message_bus=bus,
        heartbeat_interval=0.01,
    )

    await runtime.start()
    await asyncio.sleep(0.03)
    assert runtime.ready is True
    await runtime.shutdown()

    assert runtime.ready is False
    assert agent.started == 1
    assert agent.stopped == 1
    assert bus.closed == 1
    assert any(subject == "agent.generator_coord.heartbeat" for subject, _ in bus.published)


@pytest.mark.asyncio
async def test_heartbeat_failure_marks_runtime_unready_and_wakes_run() -> None:
    from mf_agents.runtime import AgentRuntime

    class FailingHeartbeatBus(FakeBus):
        def __init__(self) -> None:
            super().__init__()
            self.publish_started = asyncio.Event()

        async def publish(self, subject: str, payload: bytes) -> None:
            self.publish_started.set()
            raise RuntimeError("heartbeat publish failed")

    agent = FakeAgent({"generator": FakeTarget(True)})
    bus = FailingHeartbeatBus()
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(agent),
        message_bus=bus,
        heartbeat_interval=60.0,
    )
    runtime.install_signal_handlers = lambda: None
    run_task = asyncio.create_task(runtime.run())
    await asyncio.wait_for(bus.publish_started.wait(), timeout=0.05)
    try:
        await asyncio.wait_for(runtime._shutdown_event.wait(), timeout=0.05)
        woke_run = True
    except TimeoutError:
        woke_run = False
        run_task.cancel()
    run_result = (await asyncio.gather(run_task, return_exceptions=True))[0]

    assert woke_run is True
    assert runtime.ready is False
    assert runtime.health.ready is False
    assert "heartbeat" in runtime.health.reason
    assert isinstance(run_result, RuntimeError)
    assert str(run_result) == "heartbeat publish failed"
    assert agent.stopped == 1
    assert bus.closed == 1


@pytest.mark.asyncio
async def test_message_bus_background_failure_marks_runtime_unready_and_wakes_run() -> None:
    from mf_agents.runtime import AgentRuntime

    class StartAwareAgent(FakeAgent):
        def __init__(self) -> None:
            super().__init__({"generator": FakeTarget(True)})
            self.start_completed = asyncio.Event()

        async def start(self) -> None:
            await super().start()
            self.start_completed.set()

    class FailingBackgroundBus(FakeBus):
        def __init__(self) -> None:
            super().__init__()
            self.fail_listener = asyncio.Event()

        async def wait_for_background_failure(self) -> None:
            await self.fail_listener.wait()
            raise RuntimeError("Redis listener failed")

    agent = StartAwareAgent()
    bus = FailingBackgroundBus()
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(agent),
        message_bus=bus,
        heartbeat_interval=60.0,
    )
    runtime.install_signal_handlers = lambda: None
    run_task = asyncio.create_task(runtime.run())
    await asyncio.wait_for(agent.start_completed.wait(), timeout=0.05)
    bus.fail_listener.set()
    try:
        await asyncio.wait_for(runtime._shutdown_event.wait(), timeout=0.05)
        woke_run = True
    except TimeoutError:
        woke_run = False
        run_task.cancel()
    run_result = (await asyncio.gather(run_task, return_exceptions=True))[0]

    assert woke_run is True
    assert runtime.ready is False
    assert runtime.health.ready is False
    assert "message bus" in runtime.health.reason
    assert isinstance(run_result, RuntimeError)
    assert str(run_result) == "Redis listener failed"
    assert agent.stopped == 1
    assert bus.closed == 1


@pytest.mark.asyncio
async def test_runtime_is_the_only_owner_that_closes_agent_message_bus() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.runtime import AgentRuntime

    class CountingBus(InMemoryBus):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            await super().close()

    class RuntimeAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = ["agent.generator_coord.request"]
            self.target = FakeTarget(True)

        def runtime_targets(self):
            return {"generator": self.target}

    bus = CountingBus()
    agent = RuntimeAgent(bus)
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(agent),
        message_bus=bus,
        heartbeat_interval=60.0,
    )

    await runtime.start()
    callback_count_while_running = bus.callback_count
    await runtime.shutdown()

    assert callback_count_while_running == 1
    assert bus.callback_count == 0
    assert bus.close_calls == 1
    assert agent._started is False
    assert runtime.ready is False
    assert runtime._bus_connected is False


@pytest.mark.asyncio
async def test_runtime_rejects_restart_after_successful_shutdown() -> None:
    from mf_agents.runtime import AgentRuntime

    agent = FakeAgent({"generator": FakeTarget(True)})
    bus = FakeBus()
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(agent),
        message_bus=bus,
        heartbeat_interval=60.0,
    )

    await runtime.start()
    await runtime.shutdown()
    readiness_after_shutdown = await runtime.check_readiness()
    restart_error: RuntimeError | None = None
    try:
        await runtime.start()
    except RuntimeError as error:
        restart_error = error
    if runtime._heartbeat_task is not None:
        await runtime.shutdown()

    assert restart_error is not None
    assert str(restart_error) == "Agent runtime has been shut down"
    assert readiness_after_shutdown.ready is False
    assert readiness_after_shutdown.reason == "Agent runtime has been shut down"
    assert runtime.ready is False
    assert runtime.health.ready is False
    assert agent.started == 1
    assert agent.stopped == 1
    assert bus.connected == 1
    assert bus.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("error", "cancel"))
async def test_runtime_retries_message_bus_close_after_failure_or_cancellation(
    failure: str,
) -> None:
    from mf_agents.runtime import AgentRuntime

    class FailOnceCloseBus(FakeBus):
        async def close(self) -> None:
            self.closed += 1
            if self.closed == 1:
                if failure == "cancel":
                    raise asyncio.CancelledError
                raise RuntimeError("bus close failed")

    agent = FakeAgent({"generator": FakeTarget(True)})
    bus = FailOnceCloseBus()
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(agent),
        message_bus=bus,
        heartbeat_interval=60.0,
    )
    await runtime.start()

    expected_error = asyncio.CancelledError if failure == "cancel" else RuntimeError
    with pytest.raises(expected_error):
        await runtime.shutdown()
    first_shutdown_state = (
        runtime.ready,
        runtime._bus_connected,
        agent.stopped,
        bus.closed,
    )

    await runtime.shutdown()

    assert first_shutdown_state == (False, True, 1, 1)
    assert runtime._bus_connected is False
    assert agent.stopped == 1
    assert bus.closed == 2


@pytest.mark.asyncio
async def test_runtime_shutdown_retries_cleanup_of_partial_redis_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.messaging.redis_bus import RedisBus
    from mf_agents.runtime import AgentRuntime
    from redis import asyncio as redis_async

    class Client:
        def __init__(self) -> None:
            self.close_calls = 0

        async def ping(self) -> None:
            raise ConnectionError("Redis ping failed")

        async def aclose(self) -> None:
            self.close_calls += 1
            if self.close_calls < 3:
                raise RuntimeError("Redis client close failed")

    client = Client()
    monkeypatch.setattr(redis_async, "from_url", lambda url: client)
    bus = RedisBus(allow_fallback=False)
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(FakeAgent({"generator": FakeTarget(True)})),
        message_bus=bus,
    )

    health = await runtime.check_readiness()
    with pytest.raises(RuntimeError, match="Redis client close failed"):
        await runtime.shutdown()
    retained_after_first_shutdown = bus._client
    await runtime.shutdown()

    assert health.ready is False
    assert runtime._bus_connected is False
    assert retained_after_first_shutdown is client
    assert client.close_calls == 3
    assert bus._client is None


@pytest.mark.asyncio
async def test_concurrent_runtime_start_creates_one_agent_and_heartbeat() -> None:
    from mf_agents.runtime import AgentRuntime

    class BlockingAgent(FakeAgent):
        def __init__(self) -> None:
            super().__init__({"generator": FakeTarget(True)})
            self.start_entered = asyncio.Event()
            self.release_start = asyncio.Event()

        async def start(self) -> None:
            self.started += 1
            self.start_entered.set()
            await self.release_start.wait()

    agent = BlockingAgent()
    bus = FakeBus()
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(agent),
        message_bus=bus,
        heartbeat_interval=60.0,
    )
    first = asyncio.create_task(runtime.start())
    await asyncio.wait_for(agent.start_entered.wait(), timeout=0.05)
    second = asyncio.create_task(runtime.start())
    await asyncio.sleep(0)
    agent.release_start.set()
    await asyncio.gather(first, second)
    await asyncio.sleep(0)
    heartbeat_tasks = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_coro().__qualname__ == "AgentRuntime._heartbeat_loop"
    ]

    await runtime.shutdown()
    live_after_shutdown = [task for task in heartbeat_tasks if not task.done()]
    for task in live_after_shutdown:
        task.cancel()
    if live_after_shutdown:
        await asyncio.gather(*live_after_shutdown, return_exceptions=True)

    assert agent.started == 1
    assert len(heartbeat_tasks) == 1
    assert live_after_shutdown == []
    assert agent.stopped == 1
    assert bus.closed == 1


@pytest.mark.asyncio
async def test_runtime_start_failure_rolls_back_subscriptions_before_retry() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.runtime import AgentRuntime

    class FailSecondSubscriptionOnceBus(InMemoryBus):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        async def subscribe(self, subject, cb):
            if subject == "mf.second" and not self.failed:
                self.failed = True
                raise RuntimeError("second subscription failed")
            return await super().subscribe(subject, cb)

    class SubscriptionAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = ["mf.first", "mf.second"]
            self.target = FakeTarget(True)

        def runtime_targets(self):
            return {"generator": self.target}

    bus = FailSecondSubscriptionOnceBus()
    agent = SubscriptionAgent(bus)
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(agent),
        message_bus=bus,
        heartbeat_interval=60.0,
    )

    with pytest.raises(RuntimeError, match="second subscription failed"):
        await runtime.start()
    failed_state = (
        runtime.ready,
        runtime.health.ready,
        bus.callback_count,
        runtime._heartbeat_task,
    )

    await runtime.start()
    retry_callback_count = bus.callback_count
    await runtime.shutdown()

    assert failed_state == (False, False, 0, None)
    assert retry_callback_count == 2
    assert bus.callback_count == 0


@pytest.mark.asyncio
async def test_cancelled_runtime_start_rolls_back_subscriptions_before_retry() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.runtime import AgentRuntime

    class BlockSecondSubscriptionOnceBus(InMemoryBus):
        def __init__(self) -> None:
            super().__init__()
            self.blocked = False
            self.second_started = asyncio.Event()

        async def subscribe(self, subject, cb):
            if subject == "mf.second" and not self.blocked:
                self.blocked = True
                self.second_started.set()
                await asyncio.Event().wait()
            return await super().subscribe(subject, cb)

    class SubscriptionAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = ["mf.first", "mf.second"]
            self.target = FakeTarget(True)

        def runtime_targets(self):
            return {"generator": self.target}

    bus = BlockSecondSubscriptionOnceBus()
    agent = SubscriptionAgent(bus)
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(agent),
        message_bus=bus,
        heartbeat_interval=60.0,
    )
    start_task = asyncio.create_task(runtime.start())
    await asyncio.wait_for(bus.second_started.wait(), timeout=0.05)

    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start_task
    cancelled_state = (
        runtime.ready,
        runtime.health.ready,
        bus.callback_count,
        runtime._heartbeat_task,
    )

    await runtime.start()
    retry_callback_count = bus.callback_count
    await runtime.shutdown()

    assert cancelled_state == (False, False, 0, None)
    assert retry_callback_count == 2
    assert bus.callback_count == 0


@pytest.mark.asyncio
async def test_runtime_start_and_shutdown_are_serialized() -> None:
    from mf_agents.runtime import AgentRuntime

    class BlockingAgent(FakeAgent):
        def __init__(self) -> None:
            super().__init__({"generator": FakeTarget(True)})
            self.start_entered = asyncio.Event()
            self.release_start = asyncio.Event()

        async def start(self) -> None:
            self.started += 1
            self.start_entered.set()
            await self.release_start.wait()

    agent = BlockingAgent()
    bus = FakeBus()
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(agent),
        message_bus=bus,
        heartbeat_interval=60.0,
    )
    start_task = asyncio.create_task(runtime.start())
    await asyncio.wait_for(agent.start_entered.wait(), timeout=0.05)
    shutdown_task = asyncio.create_task(runtime.shutdown())
    await asyncio.sleep(0)
    state_while_starting = (
        shutdown_task.done(),
        agent.stopped,
        bus.closed,
    )

    agent.release_start.set()
    await asyncio.gather(start_task, shutdown_task)
    final_state = (runtime.ready, agent.stopped, bus.closed)
    await runtime.shutdown()

    assert state_while_starting == (False, 0, 0)
    assert final_state == (False, 1, 1)
    assert agent.stopped == 1
    assert bus.closed == 1


@pytest.mark.asyncio
async def test_cancelled_shutdown_waiting_for_start_still_cleans_runtime() -> None:
    from mf_agents.runtime import AgentRuntime

    class BlockingAgent(FakeAgent):
        def __init__(self) -> None:
            super().__init__({"generator": FakeTarget(True)})
            self.start_entered = asyncio.Event()
            self.release_start = asyncio.Event()

        async def start(self) -> None:
            self.started += 1
            self.start_entered.set()
            await self.release_start.wait()

    agent = BlockingAgent()
    bus = FakeBus()
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(agent),
        message_bus=bus,
        heartbeat_interval=60.0,
    )
    start_task = asyncio.create_task(runtime.start())
    await asyncio.wait_for(agent.start_entered.wait(), timeout=0.05)
    shutdown_task = asyncio.create_task(runtime.shutdown())
    await asyncio.sleep(0)

    shutdown_task.cancel()
    await asyncio.sleep(0)
    completed_while_starting = shutdown_task.done()
    agent.release_start.set()
    await start_task
    with pytest.raises(asyncio.CancelledError):
        await shutdown_task

    assert completed_while_starting is False
    assert runtime.ready is False
    assert runtime._heartbeat_task is None
    assert agent.started == 1
    assert agent.stopped == 1
    assert bus.closed == 1


@pytest.mark.asyncio
async def test_shutdown_continues_all_cleanup_after_heartbeat_and_stop_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from mf_agents.runtime import AgentRuntime

    class FailingAgent(FakeAgent):
        async def stop(self) -> None:
            self.stopped += 1
            raise RuntimeError("agent stop failed")

    class FailingBus(FakeBus):
        async def close(self) -> None:
            self.closed += 1
            raise RuntimeError("bus close failed")

    async def failed_heartbeat() -> None:
        raise RuntimeError("heartbeat failed")

    agent = FailingAgent({"generator": FakeTarget(True)})
    bus = FailingBus()
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(agent),
        message_bus=bus,
    )
    runtime.agent = agent
    runtime._bus_connected = True
    runtime._heartbeat_task = asyncio.create_task(failed_heartbeat())
    await asyncio.sleep(0)

    with caplog.at_level(logging.ERROR, logger="mf_agents.runtime"):
        with pytest.raises(ExceptionGroup) as exc_info:
            await runtime.shutdown()

    assert [str(error) for error in exc_info.value.exceptions] == [
        "heartbeat failed",
        "agent stop failed",
        "bus close failed",
    ]
    assert "heartbeat failed" in caplog.text
    assert agent.stopped == 1
    assert bus.closed == 1
    assert runtime._heartbeat_task is None
    assert runtime._bus_connected is True


@pytest.mark.asyncio
async def test_shutdown_propagates_caller_cancellation_after_cleanup() -> None:
    from mf_agents.runtime import AgentRuntime

    heartbeat_cancelled = asyncio.Event()
    heartbeat_release = asyncio.Event()

    async def cancellation_resistant_heartbeat() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            heartbeat_cancelled.set()
            await heartbeat_release.wait()

    agent = FakeAgent({"generator": FakeTarget(True)})
    bus = FakeBus()
    runtime = AgentRuntime(
        "generator_coord",
        entry_point_loader=_loader(agent),
        message_bus=bus,
    )
    runtime.agent = agent
    runtime._bus_connected = True
    runtime._heartbeat_task = asyncio.create_task(cancellation_resistant_heartbeat())
    shutdown_task = asyncio.create_task(runtime.shutdown())
    await asyncio.wait_for(heartbeat_cancelled.wait(), timeout=0.05)

    shutdown_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await shutdown_task

    assert agent.stopped == 1
    assert bus.closed == 1
    assert runtime._heartbeat_task is None
    assert runtime._bus_connected is False
    heartbeat_release.set()


@pytest.mark.asyncio
async def test_actual_downstream_agents_complete_correlated_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from critic_agent.agent import ScientificCriticAgent
    from generator_coord.agent import GeneratorCoordAgent
    from mf_agents.base.agent import AGENT_PROTOCOLS_BY_SUBJECT
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage
    from mf_core.proto_gen.moleculeforge.v1.core import audit_pb2
    from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2, router_pb2
    from nl2obj.agent import NL2ObjAgent
    from orchestrator.workflow.graph_builder import (
        WorkflowGraph,
        create_initial_state,
        full_workflow_critic_properties,
        validate_full_workflow_policies,
    )
    from orchestrator_svc.main import FullWorkflowClients
    from retrosyn_agent.agent import RetroSynAgent
    from srb_agent.agent import SRBAgent
    from supply_agent.agent import SupplyAgent
    from validation_agent.agent import ValidationAgent

    for env_name in (
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "HUMU_ENCODER_TARGET",
        "SILA2_PLAN_COMMAND",
    ):
        monkeypatch.delenv(env_name, raising=False)

    sila2_runner = tmp_path / "sila2_execute.py"
    sila2_runner.write_text(
        "import json,sys\n"
        "request=json.load(sys.stdin)\n"
        "identity_keys=('project_id','candidate_id','candidate_index',"
        "'canonical_smiles','run_id','request_id','route_id','ssp_id','target_smiles')\n"
        "response={key:request[key] for key in identity_keys}\n"
        "response.update({'status':'completed','job_id':'job-memory-pipeline'})\n"
        "print(json.dumps(response,sort_keys=True))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "SILA2_PLAN_COMMAND",
        f"{sys.executable} {sila2_runner}",
    )

    target_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    project_id = "project-memory-pipeline"
    run_id = "run-memory-pipeline"
    trace_id = "trace-memory-pipeline"
    candidate_id = "candidate-aspirin"
    candidate_index = 0
    route = {
        "route_id": "route-aspirin-1",
        "predicted_score": 0.9,
        "predicted_yield": 0.8,
        "steps": [
            {
                "step_id": "step-1",
                "reaction": ("O=C(O)c1ccccc1O.CC(=O)OC(C)=O>>CC(=O)Oc1ccccc1C(=O)O"),
                "reaction_type": "acetylation",
                "reactants": [
                    {
                        "smiles": "O=C(O)c1ccccc1O",
                        "amount_mmol": 1.0,
                    },
                    {
                        "smiles": "CC(=O)OC(C)=O",
                        "amount_mmol": 1.2,
                    },
                ],
                    "conditions": {
                        "temperature_C": 80.0,
                        "time_h": 2.0,
                    },
                    "yield": 0.8,
                    "building_blocks": [
                        {"smiles": "O=C(O)c1ccccc1O"},
                        {"smiles": "CC(=O)OC(C)=O"},
                ],
                "reagents": ["pyridine"],
                "purification": "recrystallization",
                "operation": "react",
            }
        ],
    }

    class MemoryRepository:
        async def get_run_crg(self, run_id: str) -> dict:
            return {"beliefs": [], "edges": []}

        async def write_workflow_belief(self, **kwargs) -> None:
            return None

    class MemoryCompiler:
        async def compile_intent(self, request: dict) -> dict:
            project_id = str(request["project_id"])
            lorentz = [1.0, *([0.0] * 128)]
            return {
                "cig": {
                    "project_id": project_id,
                    "objectives": [
                        {
                            "id": "qed",
                            "name": "QED",
                            "type": "MAXIMIZE",
                            "target_value": 0.8,
                            "property": "qed",
                            "weight": 1.0,
                            "pareto_tier": 1,
                        }
                    ],
                    "edges": [],
                    "hyperedges": [],
                    "constraints": {"mw": "<500"},
                    "created_by": "integration-test",
                },
                "hciv": {
                    "coordinates": lorentz,
                    "curvature": 1.0,
                    "molecule_smiles": "",
                },
                "intent_cone": {
                    "axis": lorentz,
                    "half_angle": 0.25,
                    "curvature": 1.0,
                    "property_weights": {"qed": 1.0},
                },
            }

    class MemoryICLM:
        def __init__(self) -> None:
            self.update_requests: list[generator_pb2.ModelUpdateRequest] = []
            self.artifact = audit_pb2.ArtifactRef(
                name="iclm_checkpoint",
                version="integration-v1",
                checksum=f"sha256:{'b' * 64}",
                required=True,
            )

        async def info(self) -> generator_pb2.GeneratorInfo:
            return generator_pb2.GeneratorInfo(
                name="iclm",
                version="integration-v1",
                max_batch_size=8,
                runtime_status=audit_pb2.GENERATOR_RUNTIME_STATUS_READY,
                status_message="ready",
                default_params={"student_embedding_dim": "4"},
                artifacts=[self.artifact],
            )

        async def generate(
            self,
            request: generator_pb2.GenerateRequest,
        ) -> generator_pb2.GenerateResponse:
            molecule = {
                "candidate_id": candidate_id,
                "smiles": target_smiles,
                "canonical_smiles": target_smiles,
                "properties": {
                    "mw": 180.159,
                    "molecular_weight": 180.159,
                    "logp": 1.3101,
                    "tpsa": 63.6,
                    "hbd": 1,
                    "hba": 3,
                    "rotatable_bonds": 2,
                    "ring_count": 1,
                    "aromatic_rings": 1,
                    "heavy_atoms": 13,
                    "qed": 0.9,
                    "sa_score": 2.401,
                    "pains_alerts": 0,
                    "pains_alert_count": 0,
                    "formal_charge": 0,
                    "num_h_bond_donors": 1,
                    "num_h_bond_acceptors": 3,
                },
            }
            return generator_pb2.GenerateResponse(
                generator_name="iclm",
                generation_id=request.request_id,
                request_id=request.request_id,
                molecules=[
                    json.dumps(molecule, sort_keys=True).encode("utf-8")
                    for _ in range(request.batch_size)
                ],
                artifacts=[self.artifact],
                molecule_payload_schema="molecule.v1",
                embedding_payload_schema="humu.float32.v1",
            )

        async def update_model(
            self,
            request: generator_pb2.ModelUpdateRequest,
        ) -> generator_pb2.ModelUpdateResponse:
            self.update_requests.append(
                generator_pb2.ModelUpdateRequest.FromString(
                    request.SerializeToString(deterministic=True)
                )
            )
            return generator_pb2.ModelUpdateResponse(
                acknowledged=True,
                active_version=request.target_checkpoint_version,
                updated_samples=request.rows,
            )

    class MemoryRouter:
        def __init__(self) -> None:
            self.feedback_requests: list[router_pb2.RouterFeedbackRequest] = []

        async def route(self, request: router_pb2.RouterRequest) -> router_pb2.RouterResponse:
            return router_pb2.RouterResponse(
                selected_generators=["iclm"],
                selection_weights=[1.0],
                expected_rewards=[0.5],
                allocations=[
                    router_pb2.GeneratorAllocation(
                        generator_name="iclm",
                        n_samples=request.n_samples,
                        normalized_weight=1.0,
                        expected_reward=0.5,
                    )
                ],
                strategy="task_aware_router",
                state_version=1,
            )

        async def submit_feedback(
            self,
            request: router_pb2.RouterFeedbackRequest,
        ) -> router_pb2.RouterFeedbackResponse:
            self.feedback_requests.append(request)
            return router_pb2.RouterFeedbackResponse(
                acknowledged=True,
                state_version=len(self.feedback_requests),
            )

    class MemoryTeacher:
        def __init__(self) -> None:
            self.groups: list[dict] = []

        async def adapt(self, group: dict) -> dict:
            self.groups.append(dict(group))
            row_count = len(group["records"])
            return {
                "teacher_score": 0.9,
                "teacher_source": "hypseek",
                "teacher_version": "integration-v1",
                "synthetic": False,
                "teacher_embeddings": [
                    [0.1, 0.2, 0.3, 0.4] for _ in range(row_count)
                ],
            }

    class MemoryOracle:
        async def evaluate(
            self,
            molecules: list[str],
            properties: list[str],
            *,
            request_context=None,
        ) -> dict:
            return {
                molecule: {property_name: 0.9 for property_name in properties}
                for molecule in molecules
            }

    class MemoryPlanner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        async def find_routes(self, smiles: str, max_routes: int) -> list[dict]:
            self.calls.append((smiles, max_routes))
            return [route]

    class MemoryCatalog:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def check_availability(self, smiles: str, **identity) -> dict:
            self.calls.append((smiles, dict(identity)))
            return {
                "smiles": smiles,
                "available": True,
                "catalog_id": f"memory-{len(self.calls)}",
                "source": "memory-catalog",
                "source_timestamp": "2026-07-29T00:00:00Z",
                "price": 5.0,
                "currency": "USD",
                "lead_time_days": 2,
                "evidence_id": f"catalog-evidence-{len(self.calls)}",
                "catalog_version": "memory-catalog.v1",
                "catalog_checksum": f"sha256:{'a' * 64}",
                **identity,
            }

    planner = MemoryPlanner()
    catalog = MemoryCatalog()
    repository = MemoryRepository()
    iclm = MemoryICLM()
    router = MemoryRouter()
    teacher = MemoryTeacher()
    monkeypatch.setenv(
        "GENERATOR_COORD_STATE_PATH",
        str(tmp_path / "generator_coord_feedback.json"),
    )
    bus = InMemoryBus()
    await bus.connect()
    agents = [
        NL2ObjAgent(
            message_bus=bus,
            cig_compiler_client=MemoryCompiler(),
            crg_repository=repository,
        ),
        GeneratorCoordAgent(
            message_bus=bus,
            generator_clients={"iclm": iclm},
            router_client=router,
            teacher_adapter=teacher,
            crg_repository=repository,
        ),
        ValidationAgent(
            message_bus=bus,
            oracles={"rdkit": MemoryOracle()},
            crg_repository=repository,
        ),
        RetroSynAgent(
            message_bus=bus,
            planner=planner,
            route_encoder_client=None,
            crg_repository=repository,
        ),
        SupplyAgent(
            message_bus=bus,
            supply_client=catalog,
            crg_repository=repository,
        ),
        SRBAgent(message_bus=bus, crg_repository=repository),
        ScientificCriticAgent(message_bus=bus, crg_repository=repository),
    ]
    for agent in agents:
        await agent.start()
    client = AgentRequestClient(bus, sender="workflow")

    async def request_agent(
        subject: str,
        request_id: str,
        payload: dict,
    ) -> dict:
        protocol = AGENT_PROTOCOLS_BY_SUBJECT[subject]
        response = await client.request(
            subject,
            {
                **payload,
                "trace_id": trace_id,
                "parent_id": "request-memory-pipeline",
                "run_id": run_id,
                "request_id": request_id,
                "schema_version": protocol.schema_version,
            },
            payload_type_url=protocol.payload_type_url,
            timeout=2.0,
        )
        return dict(response)

    identity = {
        "project_id": project_id,
        "candidate_id": candidate_id,
        "candidate_index": candidate_index,
        "canonical_smiles": target_smiles,
    }
    try:
        retrosyn = await request_agent(
            "agent.retrosyn.request",
            "request-memory-retrosyn",
            {
                **identity,
                "smiles": target_smiles,
                "max_routes": 2,
                "engine": "memory",
            },
        )
        planned_route = retrosyn["routes"][0]
        building_blocks = planned_route["steps"][0]["building_blocks"]
        supply = await request_agent(
            "agent.supply.request",
            "request-memory-supply",
            {
                **identity,
                "workflow_scope": "full",
                "route_id": planned_route["route_id"],
                "smiles": target_smiles,
                "building_blocks": building_blocks,
            },
        )
        srb = await request_agent(
            "agent.srb.request",
            "request-memory-srb",
            {
                **identity,
                "workflow_scope": "full",
                "route_id": planned_route["route_id"],
                "molecule": {"smiles": target_smiles},
                "pathways": [planned_route],
            },
        )
        candidate = {
            **identity,
            "smiles": target_smiles,
            "generator_name": "hfm_3d",
            "properties": {
                "mw": 180.159,
                "molecular_weight": 180.159,
                "logp": 1.3101,
                "tpsa": 63.6,
                "hbd": 1,
                "hba": 3,
                "rotatable_bonds": 2,
                "ring_count": 1,
                "aromatic_rings": 1,
                "heavy_atoms": 13,
                "qed": 0.55,
                "sa_score": 2.401,
                "pains_alerts": 0,
                "pains_alert_count": 0,
                "formal_charge": 0,
                "num_h_bond_donors": 1,
                "num_h_bond_acceptors": 3,
            },
        }
        validation = {
            "candidate_id": candidate_id,
            "candidate_index": candidate_index,
            "canonical_smiles": target_smiles,
            "outcome": "PASS",
        }
        critic_properties = full_workflow_critic_properties(
            {
                "request": {
                    "isoform_data_count": 2,
                    "kinase_selectivity_ratio": 100.0,
                },
                "retrosyn": retrosyn,
                "supply": supply,
                "srb": srb,
            },
            candidate,
            validation,
        )
        critic_pass = await request_agent(
            "agent.critic.request",
            "request-memory-critic-pass",
            {
                **identity,
                "workflow_scope": "full",
                "smiles": target_smiles,
                "properties": critic_properties,
            },
        )
        blocking_properties = {
            **critic_properties,
            "retrosyn_route_count": 0,
        }
        critic_block = await request_agent(
            "agent.critic.request",
            "request-memory-critic-block",
            {
                **identity,
                "workflow_scope": "full",
                "smiles": target_smiles,
                "properties": blocking_properties,
            },
        )
        executed_srb = await request_agent(
            "agent.srb.request",
            "request-memory-srb-execute",
            {
                **identity,
                "action": "execute",
                "workflow_scope": "full",
                "route_id": planned_route["route_id"],
                "protocols": srb["protocols"],
            },
        )

        assert planner.calls == [(target_smiles, 2)]
        assert retrosyn["routes"]
        assert planned_route["steps"]
        assert planned_route["steps"][0]["reactants"][0]["amount_mmol"] == 1.0
        assert planned_route["steps"][0]["conditions"] == {
            "temperature_C": 80.0,
            "time_h": 2.0,
        }
        assert len(catalog.calls) == 2
        assert supply["supply_assessment"]["overall_feasibility"] == "available"
        assert all(record["evidence_id"] for record in supply["block_assessments"])
        assert all(record["catalog_version"] for record in supply["block_assessments"])
        assert all(record["catalog_checksum"] for record in supply["block_assessments"])
        assert srb["protocols"]
        assert srb["protocols"][0]["steps"]
        assert srb["protocols"][0]["sila2_plan"]["steps"]
        assert srb["protocols"][0]["xdl_xml"]
        assert executed_srb["status"] == "executed"
        assert executed_srb["route_id"] == planned_route["route_id"]
        assert executed_srb["protocols"][0]["sila2_execution"]["status"] == "completed"
        assert executed_srb["protocols"][0]["sila2_execution"]["job_id"] == (
            "job-memory-pipeline"
        )
        assert critic_pass["verdict"] == "pass"
        assert critic_pass["blocking_failed"] == 0
        assert critic_block["verdict"] == "fail"
        assert any(
            result["rule_id"] == "workflow_retrosyn_routes"
            and result["verdict"] == "fail"
            and result["blocking"] is True
            for result in critic_block["rule_results"]
        )

        responses = (
            (retrosyn, "request-memory-retrosyn"),
            (supply, "request-memory-supply"),
            (srb, "request-memory-srb"),
            (executed_srb, "request-memory-srb-execute"),
            (critic_pass, "request-memory-critic-pass"),
            (critic_block, "request-memory-critic-block"),
        )
        for response, request_id in responses:
            assert response["project_id"] == project_id
            assert response["candidate_id"] == candidate_id
            assert response["candidate_index"] == candidate_index
            assert response["canonical_smiles"] == target_smiles
            assert response["run_id"] == run_id
            assert response["request_id"] == request_id

        for subject in (
            "agent.retrosyn.request",
            "agent.supply.request",
            "agent.srb.request",
            "agent.critic.request",
        ):
            envelope = AgentMessage()
            envelope.ParseFromString(bus.last_published[subject])
            payload = json.loads(envelope.payload.decode("utf-8"))
            assert envelope.trace_id == trace_id
            assert envelope.run_id == run_id
            assert payload["trace_id"] == trace_id
            assert payload["run_id"] == run_id
            assert payload["project_id"] == project_id
            assert payload["candidate_id"] == candidate_id
            assert payload["candidate_index"] == candidate_index
            assert payload["canonical_smiles"] == target_smiles

        workflow_request = {
            "project_id": project_id,
            "request_id": "request-memory-full-graph",
            "workflow_scope": "full",
            "n_samples": 1,
            "generation_strategy": "iclm",
            "retrosyn_engine": "memory",
            "retrosyn_max_routes": 2,
            "isoform_data_count": 2,
            "kinase_selectivity_ratio": 100.0,
            "validation_policy": {
                "oracle_level": 0,
                "batch_size": 1,
                "max_concurrency": 1,
                "thresholds": [
                    {
                        "level": 0,
                        "oracle": "rdkit",
                        "metric": "qed",
                        "direction": "maximize",
                        "value": 0.5,
                    }
                ],
                "oracle_inputs": {},
            },
            "teacher_policy": {
                "teacher_source": "hypseek",
                "teacher_version": "integration-v1",
                "allow_synthetic": False,
                "kd_weight": 0.25,
            },
            "selection_policy": {
                "criteria": [{"metric": "qed", "direction": "maximize"}],
            },
        }
        workflow_request.update(validate_full_workflow_policies(workflow_request))
        graph_state = create_initial_state(
            "design an aspirin-like molecule",
            run_id=run_id,
            trace_id=trace_id,
        )
        graph_state["request"] = workflow_request
        graph_state["max_refinements"] = 0

        final_state = await WorkflowGraph(
            clients=FullWorkflowClients(client),
        ).build().ainvoke(graph_state)

        assert final_state["status"] == "EXECUTING"
        assert final_state["history"] == [
            "PLANNING",
            "GENERATING",
            "VALIDATING",
            "RETROSYN",
            "CRITIC",
            "EXECUTING",
        ]
        assert final_state["validation"]["outcome"] == "PASS"
        assert final_state["supply"]["supply_assessment"]["overall_feasibility"] == (
            "available"
        )
        assert final_state["critic"]["verdict"] == "pass"
        assert final_state["srb_execution"]["status"] == "executed"
        execution = final_state["srb_execution"]["protocols"][0]["sila2_execution"]
        assert execution["status"] == "completed"
        assert execution["request_id"].endswith(":execute")
        assert len(teacher.groups) == 1
        assert len(router.feedback_requests) == 1
        assert len(iclm.update_requests) == 1
        assert iclm.update_requests[0].teacher_source == "hypseek"
    finally:
        for agent in reversed(agents):
            await agent.stop()
        await bus.close()


@pytest.mark.asyncio
async def test_real_redis_nested_request_reply_when_configured() -> None:
    redis_url = os.environ.get("REDIS_URL", "").strip()
    if not redis_url:
        pytest.skip("REDIS_URL is not configured; real Redis request/reply not executed")

    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import RedisBus
    from mf_agents.messaging.request_client import AgentRequestClient

    class RedisValidationAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("validation_agent", message_bus=message_bus)
            self._subscription_subjects = ["agent.validation.request"]

        async def process(self, payload: Mapping) -> Mapping:
            return {"validated": payload["value"]}

    class RedisGeneratorAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = ["agent.generator_coord.request"]
            self.validation_client = AgentRequestClient(message_bus)

        async def process(self, payload: Mapping) -> Mapping:
            nested = await self.validation_client.request(
                "agent.validation.request",
                {
                    "parent_id": payload["request_id"],
                    "request_id": f"{payload['request_id']}-validation",
                    "run_id": payload["run_id"],
                    "schema_version": "validation.request.v1",
                    "trace_id": payload["trace_id"],
                    "value": payload["value"],
                },
                payload_type_url=("type.moleculeforge.ai/agent/validation/request.v1"),
                timeout=1.0,
            )
            return {"value": nested["validated"]}

    client_bus = RedisBus(url=redis_url, allow_fallback=False, connect_timeout=1.0)
    generator_bus = RedisBus(url=redis_url, allow_fallback=False, connect_timeout=1.0)
    validation_bus = RedisBus(url=redis_url, allow_fallback=False, connect_timeout=1.0)
    for bus in (client_bus, generator_bus, validation_bus):
        await bus.connect()
    generator = RedisGeneratorAgent(generator_bus)
    validation = RedisValidationAgent(validation_bus)
    await generator.start()
    await validation.start()
    client = AgentRequestClient(client_bus)
    try:
        for bus in (client_bus, generator_bus, validation_bus):
            assert await bus.roundtrip(timeout=1.0) is True
        result = await client.request(
            "agent.generator_coord.request",
            {
                "parent_id": "parent-real-redis",
                "request_id": "request-real-redis",
                "run_id": "run-real-redis",
                "schema_version": "generator_coord.request.v1",
                "trace_id": "trace-real-redis",
                "value": "redis",
            },
            payload_type_url=("type.moleculeforge.ai/agent/generator_coord/request.v1"),
            timeout=1.0,
        )

        assert result == {
            "request_id": "request-real-redis",
            "run_id": "run-real-redis",
            "schema_version": "generator_coord.request.v1",
            "value": "redis",
        }
        assert client_bus.listener_count == 1
        assert generator_bus.listener_count == 1
        assert validation_bus.listener_count == 1
        assert client_bus.callback_count == 0
        assert generator_bus.callback_count == 1
        assert validation_bus.callback_count == 1

        with pytest.raises(TimeoutError):
            await client.request(
                "agent.critic.request",
                {
                    "parent_id": "parent-real-redis-timeout",
                    "request_id": "request-real-redis-timeout",
                    "run_id": "run-real-redis-timeout",
                    "schema_version": "critic.request.v1",
                    "trace_id": "trace-real-redis-timeout",
                },
                payload_type_url=("type.moleculeforge.ai/agent/critic/request.v1"),
                timeout=0.1,
            )

        assert client_bus.listener_count == 1
        assert client_bus.callback_count == 0
    finally:
        await generator.stop()
        await validation.stop()
        for bus in (client_bus, generator_bus, validation_bus):
            await bus.close()
