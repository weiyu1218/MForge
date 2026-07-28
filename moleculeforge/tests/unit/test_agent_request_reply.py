"""Agent request/reply protocol behavior."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

PAYLOAD_TYPE_URL = "type.moleculeforge.ai/agent/generator_coord/request.v1"
SCHEMA_VERSION = "generator_coord.request.v1"
TEST_AGENT_HMAC_SECRET = "task-3-agent-test-secret"
COMPATIBILITY_SUBJECTS = (
    ("generator_coord", "orchestrator.generate.request"),
    ("validation_agent", "orchestrator.validate.check"),
    ("retrosyn_agent", "orchestrator.retrosyn.plan"),
    ("supply_agent", "orchestrator.supply.check"),
    ("srb_agent", "orchestrator.srb.compile"),
    ("critic_agent", "orchestrator.critic.evaluate"),
)
PROTOCOLLESS_REQUEST_SUBJECTS = (
    ("nl2obj", "agent.nl2obj.request"),
    ("nl2obj", "orchestrator.nl2obj.resolve"),
    ("orchestrator", "orchestrator.design.request"),
)


@pytest.fixture(autouse=True)
def _configure_agent_hmac_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MESSAGE_HMAC_SECRET", TEST_AGENT_HMAC_SECRET)


def _request_payload(
    request_id: str,
    *,
    trace_id: str = "trace-1",
    parent_id: str = "parent-1",
    run_id: str = "run-1",
    schema_version: str = SCHEMA_VERSION,
    value: str = "value",
) -> dict[str, str]:
    return {
        "trace_id": trace_id,
        "parent_id": parent_id,
        "run_id": run_id,
        "request_id": request_id,
        "schema_version": schema_version,
        "value": value,
    }


def _agent_type():
    from mf_agents.base.agent import BaseAgent

    class EchoAgent(BaseAgent):
        async def process(self, payload: Mapping) -> Mapping:
            await asyncio.sleep(float(payload.get("delay", 0)))
            return {"value": payload["value"]}

    return EchoAgent


async def _start_echo_agent(bus):
    agent = _agent_type()("generator_coord", message_bus=bus)
    agent._subscription_subjects = ["agent.generator_coord.request"]
    await agent.start()
    return agent


@pytest.mark.asyncio
async def test_request_subscribes_to_reply_before_publish() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient

    class RecordingBus(InMemoryBus):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[tuple[str, str]] = []

        async def subscribe(self, subject, cb):
            self.events.append(("subscribe", subject))
            return await super().subscribe(subject, cb)

        async def publish(self, subject, payload):
            self.events.append(("publish", subject))
            return await super().publish(subject, payload)

    bus = RecordingBus()
    await bus.connect()
    agent = await _start_echo_agent(bus)
    bus.events.clear()

    result = await AgentRequestClient(bus).request(
        "agent.generator_coord.request",
        _request_payload("request-1"),
        payload_type_url=PAYLOAD_TYPE_URL,
        timeout=0.5,
    )

    assert result["value"] == "value"
    assert bus.events[0][0] == "subscribe"
    assert bus.events[1] == ("publish", "agent.generator_coord.request")
    await agent.stop()


@pytest.mark.asyncio
async def test_success_response_correlates_trace_parent_and_business_ids() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

    class RecordingBus(InMemoryBus):
        def __init__(self) -> None:
            super().__init__()
            self.envelopes: list[AgentMessage] = []

        async def publish(self, subject, payload):
            envelope = AgentMessage()
            envelope.ParseFromString(payload)
            if envelope.signature:
                self.envelopes.append(envelope)
            return await super().publish(subject, payload)

    bus = RecordingBus()
    await bus.connect()
    agent = await _start_echo_agent(bus)

    result = await AgentRequestClient(bus).request(
        "agent.generator_coord.request",
        _request_payload("request-2"),
        payload_type_url=PAYLOAD_TYPE_URL,
        timeout=0.5,
    )

    request, response = bus.envelopes
    assert response.trace_id == request.trace_id == "trace-1"
    assert response.parent_id == request.message_id
    assert response.lineage["parent_id"] == request.message_id
    assert response.request_id == request.request_id == "request-2"
    assert response.run_id == request.run_id == "run-1"
    assert response.schema_version == request.schema_version == SCHEMA_VERSION
    assert response.ttl == request.ttl - 1
    assert result == {
        "run_id": "run-1",
        "request_id": "request-2",
        "schema_version": SCHEMA_VERSION,
        "value": "value",
    }
    await agent.stop()


@pytest.mark.asyncio
async def test_timeout_raises_and_removes_reply_callback() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient

    bus = InMemoryBus()
    await bus.connect()

    with pytest.raises(TimeoutError):
        await AgentRequestClient(bus).request(
            "agent.generator_coord.request",
            _request_payload("request-timeout"),
            payload_type_url=PAYLOAD_TYPE_URL,
            timeout=0.01,
        )

    assert bus.callback_count == 0
    await bus.close()


@pytest.mark.asyncio
async def test_process_error_is_typed_upstream_error_and_listener_survives() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient, UpstreamAgentError

    class FailingAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = ["agent.generator_coord.request"]
            self.calls = 0

        async def process(self, payload: Mapping) -> Mapping:
            self.calls += 1
            raise ValueError(f"invalid value: {payload['value']}")

    bus = InMemoryBus()
    await bus.connect()
    agent = FailingAgent(bus)
    await agent.start()
    client = AgentRequestClient(bus)

    for request_id in ("request-error-1", "request-error-2"):
        with pytest.raises(UpstreamAgentError) as exc_info:
            await client.request(
                "agent.generator_coord.request",
                _request_payload(request_id, value=request_id),
                payload_type_url=PAYLOAD_TYPE_URL,
                timeout=0.5,
            )
        assert exc_info.value.upstream_type == "ValueError"
        assert request_id in str(exc_info.value)

    assert agent.calls == 2
    await agent.stop()


@pytest.mark.asyncio
async def test_concurrent_requests_cannot_consume_each_others_response() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient

    class ConcurrentAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = ["agent.generator_coord.request"]
            self.started: list[str] = []
            self.completed: list[str] = []
            self.both_started = asyncio.Event()

        async def process(self, payload: Mapping) -> Mapping:
            value = str(payload["value"])
            self.started.append(value)
            if len(self.started) == 2:
                self.both_started.set()
            await asyncio.wait_for(self.both_started.wait(), timeout=0.2)
            if value == "A":
                await asyncio.sleep(0.02)
            self.completed.append(value)
            return {"value": value}

    bus = InMemoryBus()
    await bus.connect()
    agent = ConcurrentAgent(bus)
    await agent.start()
    client = AgentRequestClient(bus)
    first = _request_payload("request-a", value="A")
    second = _request_payload("request-b", value="B")

    result_a, result_b = await asyncio.gather(
        client.request(
            "agent.generator_coord.request",
            first,
            payload_type_url=PAYLOAD_TYPE_URL,
            timeout=0.5,
        ),
        client.request(
            "agent.generator_coord.request",
            second,
            payload_type_url=PAYLOAD_TYPE_URL,
            timeout=0.5,
        ),
    )

    assert (result_a["request_id"], result_a["value"]) == ("request-a", "A")
    assert (result_b["request_id"], result_b["value"]) == ("request-b", "B")
    assert set(agent.started) == {"A", "B"}
    assert agent.completed == ["B", "A"]
    assert bus.callback_count == 1
    await agent.stop()


@pytest.mark.asyncio
async def test_agent_callback_can_await_nested_request_on_same_bus() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient

    class ValidationAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("validation_agent", message_bus=message_bus)
            self._subscription_subjects = ["agent.validation.request"]

        async def process(self, payload: Mapping) -> Mapping:
            return {"validated": payload["value"]}

    class GeneratorAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = ["agent.generator_coord.request"]
            self.client = AgentRequestClient(message_bus)

        async def process(self, payload: Mapping) -> Mapping:
            nested = await self.client.request(
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
                timeout=0.2,
            )
            return {"value": nested["validated"]}

    bus = InMemoryBus()
    await bus.connect()
    validation = ValidationAgent(bus)
    generator = GeneratorAgent(bus)
    await validation.start()
    await generator.start()

    result = await AgentRequestClient(bus).request(
        "agent.generator_coord.request",
        _request_payload("request-nested", value="nested"),
        payload_type_url=PAYLOAD_TYPE_URL,
        timeout=0.5,
    )

    assert result["value"] == "nested"
    assert bus.callback_count == 2
    await generator.stop()


async def _signed_request(
    *,
    recipient: str = "generator_coord",
    payload_type_url: str = PAYLOAD_TYPE_URL,
    payload: Mapping | None = None,
    ttl: int = 4,
) -> AgentMessage:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

    class Sender(BaseAgent):
        async def process(self, payload: Mapping) -> Mapping:
            return {}

    bus = InMemoryBus()
    await bus.connect()
    sender = Sender("orchestrator", message_bus=bus)
    request_payload = dict(payload or _request_payload("request-validation"))
    await sender.publish_agent_message(
        "capture",
        recipient=recipient,
        message_type="request",
        payload=request_payload,
        payload_type_url=payload_type_url,
        trace_id=str(request_payload["trace_id"]),
        reply_to="_reply.validation",
        request_id=str(request_payload["request_id"]),
        parent_id=str(request_payload["parent_id"]),
        run_id=str(request_payload["run_id"]),
        schema_version=str(request_payload["schema_version"]),
        lineage={"parent_id": str(request_payload["parent_id"])},
        ttl=ttl,
    )
    encoded = bus.last_published["capture"]
    envelope = AgentMessage()
    envelope.ParseFromString(encoded)
    await bus.close()
    return envelope


def _clear_envelope_schema_version(envelope: AgentMessage) -> None:
    payload = json.loads(envelope.payload.decode("utf-8"))
    payload["schema_version"] = ""
    envelope.schema_version = ""
    envelope.payload = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", ("event", "response", "error"))
async def test_canonical_subject_rejects_signed_nonrequest_without_dispatch(
    message_type: str,
) -> None:
    from mf_agents.base.agent import AgentProtocolError, BaseAgent

    class RecordingAgent(BaseAgent):
        def __init__(self) -> None:
            super().__init__("generator_coord")
            self.calls = 0

        async def process(self, payload: Mapping) -> Mapping:
            self.calls += 1
            return payload

    agent = RecordingAgent()
    envelope = await _signed_request()
    envelope.message_type = message_type
    envelope.signature = _agent_type()("orchestrator")._sign_agent_message(envelope)

    with pytest.raises(AgentProtocolError, match="signed request"):
        await agent.handle_bus_message(
            "agent.generator_coord.request",
            envelope.SerializeToString(),
        )

    assert agent.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda envelope: setattr(envelope, "recipient", "critic_agent"), "recipient"),
        (
            lambda envelope: setattr(
                envelope,
                "payload_type_url",
                "type.moleculeforge.ai/agent/critic/request.v1",
            ),
            "payload_type_url",
        ),
        (
            lambda envelope: setattr(envelope, "parent_id", "wrong-parent"),
            "parent lineage",
        ),
        (lambda envelope: setattr(envelope, "trace_id", "wrong-trace"), "trace"),
        (lambda envelope: setattr(envelope, "request_id", "wrong-request"), "request"),
    ],
)
async def test_request_rejects_invalid_signed_protocol_before_dispatch(
    mutation,
    message: str,
) -> None:
    from mf_agents.base.agent import AgentProtocolError

    agent = _agent_type()("generator_coord")
    envelope = await _signed_request()
    mutation(envelope)
    sender = _agent_type()("orchestrator")
    envelope.signature = sender._sign_agent_message(envelope)

    with pytest.raises(AgentProtocolError, match=message):
        await agent.handle_bus_message(
            "agent.generator_coord.request",
            envelope.SerializeToString(),
        )


@pytest.mark.asyncio
async def test_request_rejects_tampered_signature_before_dispatch() -> None:
    from mf_agents.base.agent import AgentProtocolError

    agent = _agent_type()("generator_coord")
    envelope = await _signed_request()
    envelope.payload = json.dumps(
        _request_payload("request-validation", value="tampered"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    with pytest.raises(AgentProtocolError, match="signature"):
        await agent.handle_bus_message(
            "agent.generator_coord.request",
            envelope.SerializeToString(),
        )


@pytest.mark.asyncio
async def test_canonical_request_rejects_unsigned_json_before_dispatch() -> None:
    from mf_agents.base.agent import AgentProtocolError, BaseAgent

    class RecordingAgent(BaseAgent):
        def __init__(self) -> None:
            super().__init__("generator_coord")
            self.calls = 0

        async def process(self, payload: Mapping) -> Mapping:
            self.calls += 1
            return payload

    agent = RecordingAgent()

    with pytest.raises(AgentProtocolError, match="signed AgentMessage"):
        await agent.handle_bus_message(
            "agent.generator_coord.request",
            json.dumps(_request_payload("request-unsigned")).encode(),
        )

    assert agent.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(("agent_name", "subject"), PROTOCOLLESS_REQUEST_SUBJECTS)
async def test_redis_protocol_less_request_subject_rejects_unsigned_json(
    agent_name: str,
    subject: str,
) -> None:
    from mf_agents.base.agent import AgentProtocolError, BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class RedisLikeBus(InMemoryBus):
        is_redis = True

    class RecordingAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__(agent_name, message_bus=message_bus)
            self._subscription_subjects = [subject]
            self.calls = 0

        async def process(self, payload: Mapping) -> Mapping:
            self.calls += 1
            return payload

    bus = RedisLikeBus()
    await bus.connect()
    agent = RecordingAgent(bus)

    with pytest.raises(AgentProtocolError, match="signed AgentMessage"):
        await agent.handle_bus_message(
            subject,
            json.dumps(_request_payload("request-unsigned-protocol-less")).encode(),
        )

    assert agent.calls == 0
    await bus.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda envelope: setattr(envelope, "reply_to", ""), "reply_to"),
        (lambda envelope: setattr(envelope, "ttl", 1), "response hop"),
        (
            _clear_envelope_schema_version,
            "schema_version",
        ),
        (
            lambda envelope: setattr(
                envelope,
                "payload",
                json.dumps(
                    _request_payload("request-validation", run_id="wrong-run"),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
            ),
            "run_id correlation",
        ),
    ),
)
async def test_redis_protocol_less_request_enforces_common_request_contract(
    mutation,
    message: str,
) -> None:
    from mf_agents.base.agent import AgentProtocolError, BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class RedisLikeBus(InMemoryBus):
        is_redis = True

    class RecordingAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("nl2obj", message_bus=message_bus)
            self._subscription_subjects = ["agent.nl2obj.request"]
            self.calls = 0

        async def process(self, payload: Mapping) -> Mapping:
            self.calls += 1
            return payload

    bus = RedisLikeBus()
    await bus.connect()
    agent = RecordingAgent(bus)
    envelope = await _signed_request(recipient="nl2obj")
    mutation(envelope)
    envelope.signature = _agent_type()("orchestrator")._sign_agent_message(envelope)

    with pytest.raises(AgentProtocolError, match=message):
        await agent.handle_bus_message(
            "agent.nl2obj.request",
            envelope.SerializeToString(),
        )

    assert agent.calls == 0
    await bus.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload_type_url", "schema_version", "message"),
    (
        ("", "nl2obj.generic-request", "payload_type_url"),
        (
            "type.moleculeforge.ai/agent/nl2obj/generic-request",
            "",
            "schema_version",
        ),
    ),
)
async def test_generic_client_requires_type_and_schema(
    payload_type_url: str,
    schema_version: str,
    message: str,
) -> None:
    from mf_agents.base.agent import AgentProtocolError
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient

    bus = InMemoryBus()
    await bus.connect()

    try:
        with pytest.raises(AgentProtocolError, match=message):
            await AgentRequestClient(bus).request(
                "agent.nl2obj.request",
                {
                    "trace_id": "trace-generic-validation",
                    "parent_id": "parent-generic-validation",
                    "run_id": "run-generic-validation",
                    "request_id": "request-generic-validation",
                    "schema_version": schema_version,
                },
                payload_type_url=payload_type_url,
                timeout=0.01,
            )
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_generic_nl2obj_request_returns_signed_correlated_response() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage
    from nl2obj.agent import NL2ObjAgent

    class Compiler:
        async def compile_intent(self, request: dict) -> dict:
            return {
                "cig": {"project_id": request["project_id"]},
                "hciv": {"embedding": [1.0]},
                "intent_cone": {"axes": []},
            }

    class Repository:
        async def get_run_crg(self, run_id: str) -> dict:
            return {"run_id": run_id, "beliefs": []}

        async def write_workflow_belief(self, **belief: object) -> None:
            return None

    class RecordingBus(InMemoryBus):
        def __init__(self) -> None:
            super().__init__()
            self.envelopes: list[tuple[str, AgentMessage]] = []

        async def publish(self, subject: str, payload: bytes) -> None:
            envelope = AgentMessage()
            envelope.ParseFromString(payload)
            if envelope.signature:
                self.envelopes.append((subject, envelope))
            await super().publish(subject, payload)

    bus = RecordingBus()
    await bus.connect()
    agent = NL2ObjAgent(
        message_bus=bus,
        cig_compiler_client=Compiler(),
        crg_repository=Repository(),
    )
    await agent.start()
    client = AgentRequestClient(bus, sender="workflow")
    payload_type_url = "type.moleculeforge.ai/agent/nl2obj/custom-request"
    schema_version = "nl2obj.custom-request"

    try:
        result = await client.request(
            "agent.nl2obj.request",
            {
                "trace_id": "trace-generic-nl2obj",
                "parent_id": "parent-generic-nl2obj",
                "run_id": "run-generic-nl2obj",
                "request_id": "request-generic-nl2obj",
                "schema_version": schema_version,
                "project_id": "project-generic-nl2obj",
                "intent": "design a kinase inhibitor",
            },
            payload_type_url=payload_type_url,
            timeout=0.5,
        )
    finally:
        await agent.stop()

    request = next(
        envelope for subject, envelope in bus.envelopes if subject == "agent.nl2obj.request"
    )
    response = next(envelope for subject, envelope in bus.envelopes if subject == request.reply_to)
    assert result["status"] == "resolved"
    assert result["run_id"] == "run-generic-nl2obj"
    assert result["request_id"] == "request-generic-nl2obj"
    assert result["schema_version"] == schema_version
    assert response.sender == "nl2obj"
    assert response.recipient == "workflow"
    assert response.message_type == "response"
    assert response.payload_type_url == payload_type_url
    assert response.schema_version == schema_version
    assert response.trace_id == request.trace_id
    assert response.parent_id == request.message_id
    assert response.lineage["parent_id"] == request.message_id
    assert response.run_id == request.run_id
    assert response.request_id == request.request_id
    assert response.ttl == request.ttl - 1
    assert BaseAgent("workflow").verify_agent_message(response) is True


@pytest.mark.asyncio
async def test_generic_nl2obj_error_returns_signed_correlated_error() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient, UpstreamAgentError
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage
    from nl2obj.agent import NL2ObjAgent

    class RecordingBus(InMemoryBus):
        def __init__(self) -> None:
            super().__init__()
            self.envelopes: list[tuple[str, AgentMessage]] = []

        async def publish(self, subject: str, payload: bytes) -> None:
            envelope = AgentMessage()
            envelope.ParseFromString(payload)
            if envelope.signature:
                self.envelopes.append((subject, envelope))
            await super().publish(subject, payload)

    bus = RecordingBus()
    await bus.connect()
    agent = NL2ObjAgent(
        message_bus=bus,
        cig_compiler_client=object(),
        crg_repository=object(),
    )
    await agent.start()
    client = AgentRequestClient(bus, sender="workflow")
    payload_type_url = "type.moleculeforge.ai/agent/nl2obj/error-request"
    schema_version = "nl2obj.error-request"

    try:
        with pytest.raises(UpstreamAgentError) as exc_info:
            await client.request(
                "agent.nl2obj.request",
                {
                    "trace_id": "trace-generic-error",
                    "parent_id": "parent-generic-error",
                    "run_id": "run-generic-error",
                    "request_id": "request-generic-error",
                    "schema_version": schema_version,
                },
                payload_type_url=payload_type_url,
                timeout=0.5,
            )
    finally:
        await agent.stop()

    request = next(
        envelope for subject, envelope in bus.envelopes if subject == "agent.nl2obj.request"
    )
    response = next(envelope for subject, envelope in bus.envelopes if subject == request.reply_to)
    assert exc_info.value.upstream_type == "ValueError"
    assert str(exc_info.value) == "intent text is required"
    assert exc_info.value.run_id == request.run_id
    assert exc_info.value.request_id == request.request_id
    assert response.sender == "nl2obj"
    assert response.recipient == "workflow"
    assert response.message_type == "error"
    assert response.payload_type_url == payload_type_url
    assert response.schema_version == schema_version
    assert response.trace_id == request.trace_id
    assert response.parent_id == request.message_id
    assert response.lineage["parent_id"] == request.message_id
    assert response.run_id == request.run_id
    assert response.request_id == request.request_id
    assert response.ttl == request.ttl - 1
    assert BaseAgent("workflow").verify_agent_message(response) is True


async def _start_workflow_domain_agents(
    bus,
    *,
    failing_nl2obj_run_ids: set[str] | None = None,
    candidates_by_run: Mapping[str, list[dict]] | None = None,
    validation_by_candidate_id: Mapping[str, bool] | None = None,
    validation_by_smiles: Mapping[str, bool] | None = None,
    validation_refinement_run_ids: set[str] | None = None,
    validation_failure_attempts_by_run: Mapping[str, set[int]] | None = None,
    empty_route_run_ids: set[str] | None = None,
    critic_refinement_run_ids: set[str] | None = None,
    sequence: list[str] | None = None,
) -> dict[str, list[dict]]:
    from mf_agents.base.agent import BaseAgent

    calls: dict[str, list[dict]] = {}
    failing_runs = set(failing_nl2obj_run_ids or ())
    candidate_rows = dict(candidates_by_run or {})
    validation_candidate_outcomes = dict(validation_by_candidate_id or {})
    validation_outcomes = dict(validation_by_smiles or {})
    validation_refinement_runs = set(validation_refinement_run_ids or ())
    validation_failure_attempts = {
        str(run_id): set(attempts)
        for run_id, attempts in (validation_failure_attempts_by_run or {}).items()
    }
    empty_route_runs = set(empty_route_run_ids or ())
    critic_refinement_runs = set(critic_refinement_run_ids or ())
    validation_attempts: dict[tuple[str, str, str], int] = {}
    critic_attempts: dict[str, int] = {}

    class WorkflowDomainAgent(BaseAgent):
        def __init__(self, name: str, subject: str) -> None:
            super().__init__(name, message_bus=bus)
            self._subscription_subjects = [subject]

        async def process(self, payload: Mapping) -> Mapping:
            request = dict(payload)
            calls.setdefault(self.name, []).append(request)
            if sequence is not None:
                sequence.append(self.name)
            run_id = str(request["run_id"])
            if self.name == "nl2obj":
                if run_id in failing_runs:
                    raise RuntimeError(f"intent compiler failed for {run_id}")
                return {
                    "status": "resolved",
                    "cig": {"project_id": request["project_id"]},
                    "hciv": {"coordinates": [1.0]},
                    "intent_cone": {"axis": [1.0]},
                }
            if self.name == "generator_coord":
                return {
                    "status": "dispatched",
                    "candidates": candidate_rows.get(
                        run_id,
                        [
                            {
                                "candidate_id": f"candidate-{run_id}",
                                "smiles": "CCO",
                            }
                        ],
                    ),
                }
            if self.name == "validation_agent":
                smiles = str(request["smiles"])
                candidate_id = str(request.get("candidate_id") or "")
                attempt_key = (run_id, candidate_id, smiles)
                attempt = validation_attempts.get(attempt_key, 0)
                validation_attempts[attempt_key] = attempt + 1
                forced_failure = (
                    run_id in validation_refinement_runs and attempt == 0
                ) or attempt in validation_failure_attempts.get(run_id, set())
                passed = (
                    False
                    if forced_failure
                    else validation_candidate_outcomes.get(
                        candidate_id,
                        validation_outcomes.get(
                            smiles,
                            not run_id.endswith("rejected"),
                        ),
                    )
                )
                result = {
                    "status": "validated" if passed else "failed",
                    "overall_passed": passed,
                    "max_oracle_level": 0,
                    "cascade": {
                        "L0_filter": {
                            "completed": True,
                            "passed": passed,
                        }
                    },
                    "upgrade_path": ["L0"],
                }
                if forced_failure:
                    result["reason"] = "oracle threshold failed"
                return result
            if self.name == "retrosyn_agent":
                return {
                    "status": "planned",
                    "routes": (
                        []
                        if run_id in empty_route_runs
                        else [
                            {
                                "route_id": f"route-{run_id}",
                                "building_blocks": ["C", "CO"],
                            }
                        ]
                    ),
                }
            if self.name == "supply_agent":
                return {
                    "status": "assessed",
                    "supply_assessment": {
                        "overall_feasibility": "available",
                    },
                }
            if self.name == "srb_agent":
                return {
                    "status": "compiled",
                    "protocols": [{"protocol_id": f"protocol-{run_id}"}],
                }
            if self.name == "critic_agent":
                attempt = critic_attempts.get(run_id, 0)
                critic_attempts[run_id] = attempt + 1
                if run_id in critic_refinement_runs and attempt == 0:
                    return {
                        "verdict": "fail",
                        "reason": "blocking rule failed",
                        "rule_results": [
                            {
                                "rule_id": "rule-test",
                                "verdict": "fail",
                            }
                        ],
                    }
                return {"verdict": "pass", "rule_results": []}
            raise AssertionError(f"unexpected Agent: {self.name}")

    agents = [
        WorkflowDomainAgent("nl2obj", "agent.nl2obj.request"),
        WorkflowDomainAgent("generator_coord", "agent.generator_coord.request"),
        WorkflowDomainAgent("validation_agent", "agent.validation.request"),
        WorkflowDomainAgent("retrosyn_agent", "agent.retrosyn.request"),
        WorkflowDomainAgent("supply_agent", "agent.supply.request"),
        WorkflowDomainAgent("srb_agent", "agent.srb.request"),
        WorkflowDomainAgent("critic_agent", "agent.critic.request"),
    ]
    for agent in agents:
        await agent.start()
    return calls


@pytest.mark.asyncio
async def test_generic_orchestrator_request_returns_signed_correlated_response() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage
    from orchestrator.agent import OrchestratorAgent

    class Repository:
        async def get_run_crg(self, run_id: str) -> dict:
            return {"run_id": run_id, "beliefs": []}

        async def write_workflow_belief(self, **belief: object) -> None:
            return None

    class RecordingBus(InMemoryBus):
        def __init__(self) -> None:
            super().__init__()
            self.envelopes: list[tuple[str, AgentMessage]] = []

        async def publish(self, subject: str, payload: bytes) -> None:
            envelope = AgentMessage()
            envelope.ParseFromString(payload)
            if envelope.signature:
                self.envelopes.append((subject, envelope))
            await super().publish(subject, payload)

    bus = RecordingBus()
    await bus.connect()
    calls = await _start_workflow_domain_agents(bus)
    agent = OrchestratorAgent(message_bus=bus, crg_repository=Repository())
    await agent.start()
    client = AgentRequestClient(bus, sender="workflow")
    payload_type_url = "type.moleculeforge.ai/agent/orchestrator/custom-request"
    schema_version = "orchestrator.custom-request"

    try:
        result = await client.request(
            "orchestrator.design.request",
            {
                "trace_id": "trace-generic-orchestrator",
                "parent_id": "parent-generic-orchestrator",
                "run_id": "run-generic-orchestrator",
                "request_id": "request-generic-orchestrator",
                "schema_version": schema_version,
                "project_id": "project-generic-orchestrator",
                "intent": "design a kinase inhibitor",
                "workflow_scope": "full",
                "max_refinements": 0,
            },
            payload_type_url=payload_type_url,
            timeout=2.0,
        )
    finally:
        await agent.stop()

    request = next(
        envelope for subject, envelope in bus.envelopes if subject == "orchestrator.design.request"
    )
    response = next(envelope for subject, envelope in bus.envelopes if subject == request.reply_to)
    assert result["project_id"] == "project-generic-orchestrator"
    assert result["status"] == "completed"
    assert result["current_stage"] == "CRITIC"
    assert result["history"] == [
        "PLANNING",
        "GENERATING",
        "VALIDATING",
        "RETROSYN",
        "CRITIC",
    ]
    assert result["candidates"] == [
        {
            "candidate_id": "candidate-run-generic-orchestrator",
            "smiles": "CCO",
        }
    ]
    assert set(calls) == {
        "nl2obj",
        "generator_coord",
        "validation_agent",
        "retrosyn_agent",
        "supply_agent",
        "srb_agent",
        "critic_agent",
    }
    assert result["run_id"] == "run-generic-orchestrator"
    assert result["request_id"] == "request-generic-orchestrator"
    assert result["schema_version"] == schema_version
    assert response.sender == "orchestrator"
    assert response.recipient == "workflow"
    assert response.message_type == "response"
    assert response.payload_type_url == payload_type_url
    assert response.schema_version == schema_version
    assert response.trace_id == request.trace_id
    assert response.parent_id == request.message_id
    assert response.lineage["parent_id"] == request.message_id
    assert response.run_id == request.run_id
    assert response.request_id == request.request_id
    assert response.ttl == request.ttl - 1
    assert BaseAgent("workflow").verify_agent_message(response) is True


@pytest.mark.asyncio
async def test_orchestrator_downstream_uses_the_passing_candidate_occurrence() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient
    from orchestrator.agent import OrchestratorAgent

    run_id = "run-multi-candidate"
    bus = InMemoryBus()
    await bus.connect()
    calls = await _start_workflow_domain_agents(
        bus,
        candidates_by_run={
            run_id: [
                {"candidate_id": "BAD", "smiles": "CCO"},
                {"candidate_id": "GOOD", "smiles": "CCO"},
            ]
        },
        validation_by_candidate_id={"BAD": False, "GOOD": True},
    )
    orchestrator = OrchestratorAgent(message_bus=bus, crg_repository=None)
    await orchestrator.start()

    try:
        result = await AgentRequestClient(bus, sender="workflow").request(
            "orchestrator.design.request",
            {
                "trace_id": "trace-multi-candidate",
                "parent_id": "parent-multi-candidate",
                "run_id": run_id,
                "request_id": "request-multi-candidate",
                "schema_version": "orchestrator.multi-candidate-request",
                "project_id": "project-multi-candidate",
                "intent": "design two candidates",
                "workflow_scope": "full",
                "max_refinements": 0,
            },
            payload_type_url=("type.moleculeforge.ai/agent/orchestrator/multi-candidate-request"),
            timeout=2.0,
        )
    finally:
        await orchestrator.stop()

    assert result["status"] == "completed"
    assert [call.get("candidate_id") for call in calls["validation_agent"]] == ["BAD", "GOOD"]
    assert calls["retrosyn_agent"][0]["candidate_id"] == "GOOD"
    assert calls["supply_agent"][0]["candidate_id"] == "GOOD"
    assert calls["srb_agent"][0]["candidate_id"] == "GOOD"
    assert calls["critic_agent"][0]["candidate_id"] == "GOOD"


def test_candidate_validation_pairs_reserve_reversed_id_and_smiles_occurrences() -> None:
    import orchestrator.agent as orchestrator_module

    first = {"candidate_id": "DUP", "smiles": "CCO"}
    second = {"candidate_id": "DUP", "smiles": "CCC"}
    state = {
        "candidates": [first, second],
        "validation": {
            "results": [
                {
                    "candidate_id": "DUP",
                    "smiles": "CCC",
                    "overall_passed": True,
                },
                {
                    "candidate_id": "DUP",
                    "smiles": "CCO",
                    "overall_passed": False,
                },
            ]
        },
    }

    pairs = orchestrator_module._candidate_validation_pairs(state)

    assert pairs[0][0] is second
    assert pairs[1][0] is first
    assert orchestrator_module._selected_candidate(state) is second


def test_candidate_validation_pairs_reserve_explicit_match_before_fuzzy_row() -> None:
    import orchestrator.agent as orchestrator_module

    explicit = {"candidate_id": "EXPLICIT", "smiles": "CCO"}
    fuzzy = {"candidate_id": "FUZZY", "smiles": "CCO"}
    state = {
        "candidates": [explicit, fuzzy],
        "validation": {
            "results": [
                {"smiles": "CCO", "overall_passed": True},
                {
                    "candidate_id": "EXPLICIT",
                    "smiles": "CCO",
                    "overall_passed": False,
                },
            ]
        },
    }

    pairs = orchestrator_module._candidate_validation_pairs(state)

    assert pairs[0][0] is fuzzy
    assert pairs[1][0] is explicit
    assert orchestrator_module._selected_candidate(state) is fuzzy


def test_candidate_validation_pairs_leave_unknown_explicit_id_unmatched() -> None:
    import orchestrator.agent as orchestrator_module

    candidate = {"candidate_id": "KNOWN", "smiles": "CCO"}
    state = {
        "candidates": [candidate],
        "validation": {
            "results": [
                {
                    "candidate_id": "UNKNOWN",
                    "smiles": "CCO",
                    "overall_passed": True,
                },
                {
                    "candidate_id": "KNOWN",
                    "smiles": "CCO",
                    "overall_passed": False,
                },
            ]
        },
    }

    pairs = orchestrator_module._candidate_validation_pairs(state)

    assert pairs == [(candidate, state["validation"]["results"][1])]
    with pytest.raises(RuntimeError, match="passing validated candidate"):
        orchestrator_module._selected_candidate(state)


@pytest.mark.asyncio
async def test_full_agent_critic_uses_shared_policy_and_engineering_keeps_default() -> None:
    import orchestrator.agent as orchestrator_module
    from critic_agent.agent import ScientificCriticAgent

    class CriticRequestClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.critic = ScientificCriticAgent(crg_repository=None)

        async def request(
            self,
            subject: str,
            payload: dict,
            *,
            payload_type_url: str,
            timeout: float,
        ) -> dict:
            self.calls.append(
                {
                    "subject": subject,
                    "payload": dict(payload),
                    "payload_type_url": payload_type_url,
                    "timeout": timeout,
                }
            )
            return await self.critic.process(payload)

    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    candidate = {
        "candidate_id": "candidate-aspirin",
        "smiles": smiles,
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
    state = {
        "run_id": "run-real-critic",
        "trace_id": "trace-real-critic",
        "request": {
            "request_id": "request-real-critic",
            "project_id": "project-real-critic",
            "isoform_data_count": 2,
            "kinase_selectivity_ratio": 100.0,
        },
        "candidates": [candidate],
        "validation": {
            "passed": True,
            "results": [
                {
                    "candidate_id": "candidate-aspirin",
                    "smiles": smiles,
                    "overall_passed": True,
                    "cascade": {},
                }
            ],
        },
        "supply": {
            "supply_assessment": {
                "total_blocks": 2,
                "commercially_available": 2,
                "supplier_diversity": 3,
                "avg_price_per_gram": 15.0,
            }
        },
        "srb": {
            "protocols": [
                {
                    "steps": [{"step_id": "1"}, {"step_id": "2"}],
                    "total_estimated_cost_usd": 30.0,
                }
            ]
        },
    }

    full_request_client = CriticRequestClient()
    full_result = await orchestrator_module._FullAgentWorkflowClients(
        full_request_client
    ).review_candidates(state)
    engineering_request_client = CriticRequestClient()
    engineering_result = await orchestrator_module._EngineeringAgentWorkflowClients(
        engineering_request_client
    ).review_candidates(state)

    assert full_result["verdict"] == "pass"
    assert full_result["blocking_failed"] == 0
    full_properties = full_request_client.calls[0]["payload"]["properties"]
    assert full_properties["building_block_availability"] == 1.0
    assert full_properties["critical_material_suppliers"] == 3
    assert full_properties["estimated_cost_per_gram"] == 15.0
    assert full_properties["synthesis_steps"] == 2
    assert full_properties["isoform_data_count"] == 2
    assert full_properties["kinase_selectivity_ratio"] == 100.0
    assert full_properties["_critic_blocking_rule_ids"]
    assert engineering_result["verdict"] == "fail"
    assert engineering_result["blocking_failed"] > 0
    assert (
        "_critic_blocking_rule_ids"
        not in engineering_request_client.calls[0]["payload"]["properties"]
    )


@pytest.mark.asyncio
async def test_engineering_workflow_skips_full_synthesis_agents() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient
    from orchestrator.agent import OrchestratorAgent

    sequence: list[str] = []
    bus = InMemoryBus()
    await bus.connect()
    await _start_workflow_domain_agents(bus, sequence=sequence)
    orchestrator = OrchestratorAgent(message_bus=bus, crg_repository=None)
    await orchestrator.start()

    try:
        result = await AgentRequestClient(bus, sender="workflow").request(
            "orchestrator.design.request",
            {
                "trace_id": "trace-engineering",
                "parent_id": "parent-engineering",
                "run_id": "run-engineering",
                "request_id": "request-engineering",
                "schema_version": "orchestrator.engineering-request",
                "project_id": "project-engineering",
                "intent": "design an engineering candidate",
                "workflow_scope": "engineering",
                "max_refinements": 0,
            },
            payload_type_url=("type.moleculeforge.ai/agent/orchestrator/engineering-request"),
            timeout=2.0,
        )
    finally:
        await orchestrator.stop()

    assert result["status"] == "completed"
    assert sequence == [
        "nl2obj",
        "generator_coord",
        "validation_agent",
        "critic_agent",
    ]
    assert "retrosyn" not in result
    assert "supply" not in result
    assert "srb" not in result


@pytest.mark.asyncio
async def test_orchestrator_refinement_sends_serialized_critic_feedback() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient
    from orchestrator.agent import OrchestratorAgent

    run_id = "run-critic-refinement"
    bus = InMemoryBus()
    await bus.connect()
    calls = await _start_workflow_domain_agents(
        bus,
        critic_refinement_run_ids={run_id},
    )
    orchestrator = OrchestratorAgent(message_bus=bus, crg_repository=None)
    await orchestrator.start()

    try:
        result = await AgentRequestClient(bus, sender="workflow").request(
            "orchestrator.design.request",
            {
                "trace_id": "trace-critic-refinement",
                "parent_id": "parent-critic-refinement",
                "run_id": run_id,
                "request_id": "request-critic-refinement",
                "schema_version": "orchestrator.critic-refinement-request",
                "project_id": "project-critic-refinement",
                "intent": "refine a candidate",
                "workflow_scope": "full",
                "max_refinements": 1,
            },
            payload_type_url=("type.moleculeforge.ai/agent/orchestrator/critic-refinement-request"),
            timeout=2.0,
        )
    finally:
        await orchestrator.stop()

    assert result["status"] == "completed"
    generator_calls = calls["generator_coord"]
    assert len(generator_calls) == 2
    assert "generation_feedback" not in generator_calls[0]["generator_params"]
    assert json.loads(generator_calls[1]["generator_params"]["generation_feedback"]) == [
        {
            "source": "critic",
            "refinement_count": 1,
            "verdict": "fail",
            "reason": "blocking rule failed",
            "rule_results": [
                {
                    "rule_id": "rule-test",
                    "verdict": "fail",
                }
            ],
        }
    ]


@pytest.mark.asyncio
async def test_orchestrator_refinement_sends_serialized_validation_feedback() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient
    from orchestrator.agent import OrchestratorAgent

    run_id = "run-validation-refinement"
    bus = InMemoryBus()
    await bus.connect()
    calls = await _start_workflow_domain_agents(
        bus,
        validation_refinement_run_ids={run_id},
    )
    orchestrator = OrchestratorAgent(message_bus=bus, crg_repository=None)
    await orchestrator.start()

    try:
        result = await AgentRequestClient(bus, sender="workflow").request(
            "orchestrator.design.request",
            {
                "trace_id": "trace-validation-refinement",
                "parent_id": "parent-validation-refinement",
                "run_id": run_id,
                "request_id": "request-validation-refinement",
                "schema_version": "orchestrator.validation-refinement-request",
                "project_id": "project-validation-refinement",
                "intent": "refine a candidate after validation",
                "workflow_scope": "engineering",
                "max_refinements": 1,
            },
            payload_type_url=(
                "type.moleculeforge.ai/agent/orchestrator/validation-refinement-request"
            ),
            timeout=2.0,
        )
    finally:
        await orchestrator.stop()

    assert result["status"] == "completed"
    generator_calls = calls["generator_coord"]
    assert len(generator_calls) == 2
    assert "generation_feedback" not in generator_calls[0]["generator_params"]
    feedback = json.loads(generator_calls[1]["generator_params"]["generation_feedback"])
    assert feedback[0]["source"] == "validation"
    assert feedback[0]["refinement_count"] == 1
    assert feedback[0]["passed"] is False
    assert feedback[0]["reason"] == "oracle threshold failed"
    assert feedback[0]["results"][0]["candidate_id"] == f"candidate-{run_id}"
    assert feedback[0]["results"][0]["reason"] == "oracle threshold failed"


@pytest.mark.asyncio
async def test_orchestrator_refinement_replaces_stale_downstream_state() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient
    from orchestrator.agent import OrchestratorAgent

    run_id = "run-mixed-refinement"
    bus = InMemoryBus()
    await bus.connect()
    calls = await _start_workflow_domain_agents(
        bus,
        critic_refinement_run_ids={run_id},
        validation_failure_attempts_by_run={run_id: {1}},
    )
    orchestrator = OrchestratorAgent(message_bus=bus, crg_repository=None)
    await orchestrator.start()

    try:
        result = await AgentRequestClient(bus, sender="workflow").request(
            "orchestrator.design.request",
            {
                "trace_id": "trace-mixed-refinement",
                "parent_id": "parent-mixed-refinement",
                "run_id": run_id,
                "request_id": "request-mixed-refinement",
                "schema_version": "orchestrator.mixed-refinement-request",
                "project_id": "project-mixed-refinement",
                "intent": "refine through critic and validation",
                "workflow_scope": "engineering",
                "max_refinements": 2,
            },
            payload_type_url=("type.moleculeforge.ai/agent/orchestrator/mixed-refinement-request"),
            timeout=2.0,
        )
    finally:
        await orchestrator.stop()

    assert result["status"] == "completed"
    generator_calls = calls["generator_coord"]
    assert len(generator_calls) == 3
    assert "generation_feedback" not in generator_calls[0]["generator_params"]
    assert [
        entry["source"]
        for entry in json.loads(generator_calls[1]["generator_params"]["generation_feedback"])
    ] == ["critic"]
    third_feedback = json.loads(generator_calls[2]["generator_params"]["generation_feedback"])
    assert [entry["source"] for entry in third_feedback] == [
        "critic",
        "validation",
    ]
    assert third_feedback[1]["reason"] == "oracle threshold failed"


@pytest.mark.asyncio
async def test_full_workflow_handles_empty_retrosyn_routes_without_domain_error() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient
    from orchestrator.agent import OrchestratorAgent

    run_id = "run-empty-routes"
    bus = InMemoryBus()
    await bus.connect()
    calls = await _start_workflow_domain_agents(
        bus,
        empty_route_run_ids={run_id},
    )
    orchestrator = OrchestratorAgent(message_bus=bus, crg_repository=None)
    await orchestrator.start()

    try:
        result = await AgentRequestClient(bus, sender="workflow").request(
            "orchestrator.design.request",
            {
                "trace_id": "trace-empty-routes",
                "parent_id": "parent-empty-routes",
                "run_id": run_id,
                "request_id": "request-empty-routes",
                "schema_version": "orchestrator.empty-routes-request",
                "project_id": "project-empty-routes",
                "intent": "design a candidate",
                "workflow_scope": "full",
                "max_refinements": 0,
            },
            payload_type_url=("type.moleculeforge.ai/agent/orchestrator/empty-routes-request"),
            timeout=2.0,
        )
    finally:
        await orchestrator.stop()

    assert result["status"] == "completed"
    assert result["supply"]["supply_assessment"]["overall_feasibility"] == "unavailable"
    assert result["supply"]["skip_reason"] == "retrosyn.routes is empty"
    assert result["srb"] == {
        "status": "skipped",
        "protocols": [],
        "skip_reason": "supply feasibility is unavailable",
    }
    assert "supply_agent" not in calls
    assert "srb_agent" not in calls
    assert len(calls["critic_agent"]) == 1


@pytest.mark.asyncio
async def test_orchestrator_empty_candidate_batches_refine_then_escalate() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient
    from orchestrator.agent import OrchestratorAgent

    run_id = "run-empty-candidates"
    bus = InMemoryBus()
    await bus.connect()
    calls = await _start_workflow_domain_agents(
        bus,
        candidates_by_run={run_id: []},
    )
    orchestrator = OrchestratorAgent(message_bus=bus, crg_repository=None)
    await orchestrator.start()

    try:
        result = await AgentRequestClient(bus, sender="workflow").request(
            "orchestrator.design.request",
            {
                "trace_id": "trace-empty-candidates",
                "parent_id": "parent-empty-candidates",
                "run_id": run_id,
                "request_id": "request-empty-candidates",
                "schema_version": "orchestrator.empty-candidates-request",
                "project_id": "project-empty-candidates",
                "intent": "design a candidate",
                "workflow_scope": "full",
                "max_refinements": 1,
            },
            payload_type_url=("type.moleculeforge.ai/agent/orchestrator/empty-candidates-request"),
            timeout=2.0,
        )
    finally:
        await orchestrator.stop()

    assert result["status"] == "rejected"
    assert result["history"] == [
        "PLANNING",
        "GENERATING",
        "VALIDATING",
        "REFINING",
        "GENERATING",
        "VALIDATING",
        "ESCALATING",
    ]
    assert result["validation"] == {
        "passed": False,
        "reason": "no valid candidates",
        "results": [],
    }
    assert len(calls["generator_coord"]) == 2
    assert "validation_agent" not in calls
    assert "retrosyn_agent" not in calls
    assert "critic_agent" not in calls


@pytest.mark.asyncio
async def test_orchestrator_requests_have_isolated_workflow_state() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient
    from orchestrator.agent import OrchestratorAgent

    class Repository:
        async def get_run_crg(self, run_id: str) -> dict:
            return {"run_id": run_id, "beliefs": []}

        async def write_workflow_belief(self, **belief: object) -> None:
            return None

    bus = InMemoryBus()
    await bus.connect()
    await _start_workflow_domain_agents(bus)
    orchestrator = OrchestratorAgent(message_bus=bus, crg_repository=Repository())
    await orchestrator.start()
    client = AgentRequestClient(bus, sender="workflow")
    payload_type_url = "type.moleculeforge.ai/agent/orchestrator/isolation-request"
    schema_version = "orchestrator.isolation-request"

    async def request(run_id: str, *, max_refinements: int = 0) -> Mapping:
        return await client.request(
            "orchestrator.design.request",
            {
                "trace_id": f"trace-{run_id}",
                "parent_id": f"parent-{run_id}",
                "run_id": run_id,
                "request_id": f"request-{run_id}",
                "schema_version": schema_version,
                "project_id": f"project-{run_id}",
                "intent": f"design molecule for {run_id}",
                "workflow_scope": "full",
                "max_refinements": max_refinements,
            },
            payload_type_url=payload_type_url,
            timeout=2.0,
        )

    try:
        sequential = [
            await request("run-sequential-1"),
            await request("run-sequential-2"),
            await request("run-sequential-3"),
        ]
        accepted, rejected = await asyncio.gather(
            request("run-concurrent-accepted"),
            request("run-concurrent-rejected", max_refinements=1),
        )
    finally:
        await orchestrator.stop()

    expected_history = [
        "PLANNING",
        "GENERATING",
        "VALIDATING",
        "RETROSYN",
        "CRITIC",
    ]
    assert [result["history"] for result in sequential] == [expected_history] * 3
    assert all(result["status"] == "completed" for result in sequential)
    assert accepted["status"] == "completed"
    assert accepted["history"] == expected_history
    assert rejected["status"] == "rejected"
    assert rejected["current_stage"] == "ESCALATING"
    assert rejected["history"] == [
        "PLANNING",
        "GENERATING",
        "VALIDATING",
        "REFINING",
        "GENERATING",
        "VALIDATING",
        "ESCALATING",
    ]
    for result in [*sequential, accepted, rejected]:
        assert {str(event["run_id"]) for event in result["events"]} == {str(result["run_id"])}


@pytest.mark.asyncio
async def test_orchestrator_rejects_a_nonterminal_graph_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import orchestrator.agent as orchestrator_module

    class NonterminalGraph:
        def __init__(self, clients, workflow_scope: str) -> None:
            assert clients is None
            assert workflow_scope == "state_only"

        def build(self):
            return self

        async def ainvoke(self, state: dict, config: dict) -> dict:
            assert config["recursion_limit"] == 25
            return {
                **state,
                "status": "VALIDATING",
                "history": ["PLANNING", "GENERATING", "VALIDATING"],
                "events": [],
            }

    monkeypatch.setattr(orchestrator_module, "WorkflowGraph", NonterminalGraph)
    agent = orchestrator_module.OrchestratorAgent(crg_repository=None)

    with pytest.raises(RuntimeError, match="non-terminal stage"):
        await agent.run_design_workflow(
            {
                "project_id": "project-nonterminal",
                "run_id": "run-nonterminal",
                "trace_id": "trace-nonterminal",
                "intent": "design a molecule",
                "workflow_scope": "state_only",
                "validation_passed": True,
                "max_refinements": 0,
            }
        )


@pytest.mark.asyncio
async def test_orchestrator_domain_failure_returns_signed_error() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient, UpstreamAgentError
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage
    from orchestrator.agent import OrchestratorAgent

    class RecordingBus(InMemoryBus):
        def __init__(self) -> None:
            super().__init__()
            self.envelopes: list[tuple[str, AgentMessage]] = []

        async def publish(self, subject: str, payload: bytes) -> None:
            envelope = AgentMessage()
            envelope.ParseFromString(payload)
            if envelope.signature:
                self.envelopes.append((subject, envelope))
            await super().publish(subject, payload)

    bus = RecordingBus()
    await bus.connect()
    await _start_workflow_domain_agents(
        bus,
        failing_nl2obj_run_ids={"run-orchestrator-error"},
    )
    orchestrator = OrchestratorAgent(message_bus=bus, crg_repository=None)
    await orchestrator.start()
    client = AgentRequestClient(bus, sender="workflow")

    try:
        with pytest.raises(UpstreamAgentError, match="intent compiler failed") as exc_info:
            await client.request(
                "orchestrator.design.request",
                {
                    "trace_id": "trace-orchestrator-error",
                    "parent_id": "parent-orchestrator-error",
                    "run_id": "run-orchestrator-error",
                    "request_id": "request-orchestrator-error",
                    "schema_version": "orchestrator.error-request",
                    "project_id": "project-orchestrator-error",
                    "intent": "invalid intent",
                    "workflow_scope": "full",
                    "max_refinements": 0,
                },
                payload_type_url=("type.moleculeforge.ai/agent/orchestrator/error-request"),
                timeout=2.0,
            )
    finally:
        await orchestrator.stop()

    outer_request = next(
        envelope for subject, envelope in bus.envelopes if subject == "orchestrator.design.request"
    )
    outer_error = next(
        envelope for subject, envelope in bus.envelopes if subject == outer_request.reply_to
    )
    assert exc_info.value.upstream_type == "UpstreamAgentError"
    assert outer_error.message_type == "error"
    assert outer_error.sender == "orchestrator"
    assert outer_error.recipient == "workflow"
    assert outer_error.parent_id == outer_request.message_id
    assert BaseAgent("workflow").verify_agent_message(outer_error) is True


@pytest.mark.asyncio
async def test_redis_custom_event_subscription_accepts_signed_event() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class RedisLikeBus(InMemoryBus):
        is_redis = True

    class RecordingAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = [
                "agent.generator_coord.request",
                "generator.progress",
            ]
            self.received: list[tuple[str, bytes, str]] = []

        async def handle_message(
            self,
            subject: str,
            payload: bytes,
            reply_to: str = "",
        ) -> None:
            self.received.append((subject, payload, reply_to))

    sender_bus = InMemoryBus()
    await sender_bus.connect()
    sender = BaseAgent("orchestrator", message_bus=sender_bus)
    envelope = await sender.publish_agent_message(
        "capture",
        recipient="generator_coord",
        message_type="event",
        payload={"progress": 0.5},
        payload_type_url="type.moleculeforge.ai/agent/generator/progress.v1",
        reply_to="events.reply",
        ttl=4,
    )
    receiver_bus = RedisLikeBus()
    await receiver_bus.connect()
    receiver = RecordingAgent(receiver_bus)

    await receiver.handle_bus_message(
        "generator.progress",
        envelope.SerializeToString(),
    )

    assert receiver.received == [("generator.progress", b'{"progress":0.5}', "events.reply")]
    await sender_bus.close()
    await receiver_bus.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", ("request", "response", "error"))
async def test_redis_custom_event_subscription_rejects_signed_non_event(
    message_type: str,
) -> None:
    from mf_agents.base.agent import AgentProtocolError, BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class RedisLikeBus(InMemoryBus):
        is_redis = True

    class RecordingAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = [
                "agent.generator_coord.request",
                "generator.progress",
            ]
            self.calls = 0

        async def handle_message(
            self,
            subject: str,
            payload: bytes,
            reply_to: str = "",
        ) -> None:
            self.calls += 1

    bus = RedisLikeBus()
    await bus.connect()
    agent = RecordingAgent(bus)
    envelope = await _signed_request()
    envelope.message_type = message_type
    envelope.signature = _agent_type()("orchestrator")._sign_agent_message(envelope)

    with pytest.raises(AgentProtocolError):
        await agent.handle_bus_message(
            "generator.progress",
            envelope.SerializeToString(),
        )

    assert agent.calls == 0
    await bus.close()


@pytest.mark.asyncio
async def test_redis_custom_event_subscription_rejects_unsigned_payload() -> None:
    from mf_agents.base.agent import AgentProtocolError, BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class RedisLikeBus(InMemoryBus):
        is_redis = True

    class RecordingAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("legacy_agent", message_bus=message_bus)
            self._subscription_subjects = ["legacy.event"]
            self.calls = 0

        async def handle_message(
            self,
            subject: str,
            payload: bytes,
            reply_to: str = "",
        ) -> None:
            self.calls += 1

    bus = RedisLikeBus()
    await bus.connect()
    agent = RecordingAgent(bus)

    with pytest.raises(AgentProtocolError, match="signed AgentMessage"):
        await agent.handle_bus_message("legacy.event", b'{"value":"unsigned"}')

    assert agent.calls == 0
    await bus.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("agent_name", "subject"), COMPATIBILITY_SUBJECTS)
async def test_redis_compatibility_alias_rejects_unsigned_json_before_dispatch(
    agent_name: str,
    subject: str,
) -> None:
    from mf_agents.base.agent import AgentProtocolError, BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class RedisLikeBus(InMemoryBus):
        is_redis = True

    class RecordingAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__(agent_name, message_bus=message_bus)
            self._subscription_subjects = [self.protocol.subject, subject]
            self.calls = 0

        async def process(self, payload: Mapping) -> Mapping:
            self.calls += 1
            return payload

    bus = RedisLikeBus()
    await bus.connect()
    agent = RecordingAgent(bus)

    with pytest.raises(AgentProtocolError, match="signed AgentMessage"):
        await agent.handle_bus_message(
            subject,
            json.dumps(_request_payload("request-unsigned-alias")).encode(),
        )

    assert agent.calls == 0
    await bus.close()


@pytest.mark.asyncio
async def test_redis_compatibility_alias_rejects_malformed_payload_before_dispatch() -> None:
    from mf_agents.base.agent import AgentProtocolError, BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class RedisLikeBus(InMemoryBus):
        is_redis = True

    class RecordingAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = [
                "agent.generator_coord.request",
                "orchestrator.generate.request",
            ]
            self.calls = 0

        async def process(self, payload: Mapping) -> Mapping:
            self.calls += 1
            return payload

    bus = RedisLikeBus()
    await bus.connect()
    agent = RecordingAgent(bus)

    with pytest.raises(AgentProtocolError, match="signed AgentMessage"):
        await agent.handle_bus_message(
            "orchestrator.generate.request",
            b"\x80",
        )

    assert agent.calls == 0
    await bus.close()


@pytest.mark.asyncio
async def test_redis_compatibility_alias_accepts_signed_canonical_request() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class RedisLikeBus(InMemoryBus):
        is_redis = True

    class RecordingAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = [
                "agent.generator_coord.request",
                "orchestrator.generate.request",
            ]
            self.calls = 0

        async def process(self, payload: Mapping) -> Mapping:
            self.calls += 1
            return {"value": payload["value"]}

    bus = RedisLikeBus()
    await bus.connect()
    agent = RecordingAgent(bus)
    envelope = await _signed_request()

    await agent.handle_bus_message(
        "orchestrator.generate.request",
        envelope.SerializeToString(),
    )

    assert agent.calls == 1
    await bus.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda envelope: setattr(envelope, "schema_version", "generator_coord.request.v2"),
            "schema_version",
        ),
        (
            lambda envelope: setattr(
                envelope,
                "payload",
                json.dumps(
                    _request_payload("request-validation", run_id="wrong-run"),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
            ),
            "run_id correlation",
        ),
    ],
)
async def test_redis_compatibility_alias_enforces_schema_and_correlation(
    mutation,
    message: str,
) -> None:
    from mf_agents.base.agent import AgentProtocolError, BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class RedisLikeBus(InMemoryBus):
        is_redis = True

    class RecordingAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = [
                "agent.generator_coord.request",
                "orchestrator.generate.request",
            ]
            self.calls = 0

        async def process(self, payload: Mapping) -> Mapping:
            self.calls += 1
            return payload

    bus = RedisLikeBus()
    await bus.connect()
    agent = RecordingAgent(bus)
    envelope = await _signed_request()
    mutation(envelope)
    envelope.signature = _agent_type()("orchestrator")._sign_agent_message(envelope)

    with pytest.raises(AgentProtocolError, match=message):
        await agent.handle_bus_message(
            "orchestrator.generate.request",
            envelope.SerializeToString(),
        )

    assert agent.calls == 0
    await bus.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", ("event", "response", "error"))
async def test_compatibility_request_alias_rejects_signed_nonrequest_without_dispatch(
    message_type: str,
) -> None:
    from mf_agents.base.agent import AgentProtocolError, BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class RedisLikeBus(InMemoryBus):
        is_redis = True

    class RecordingAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = [
                "agent.generator_coord.request",
                "orchestrator.generate.request",
            ]
            self.calls = 0

        async def process(self, payload: Mapping) -> Mapping:
            self.calls += 1
            return payload

    bus = RedisLikeBus()
    await bus.connect()
    agent = RecordingAgent(bus)
    envelope = await _signed_request()
    envelope.message_type = message_type
    envelope.signature = _agent_type()("orchestrator")._sign_agent_message(envelope)

    with pytest.raises(AgentProtocolError, match="signed request"):
        await agent.handle_bus_message(
            "orchestrator.generate.request",
            envelope.SerializeToString(),
        )

    assert agent.calls == 0
    await bus.close()


@pytest.mark.asyncio
async def test_in_memory_compatibility_alias_preserves_unsigned_json_dispatch() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class RecordingAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = [
                "agent.generator_coord.request",
                "orchestrator.generate.request",
            ]
            self.calls = 0

        async def process(self, payload: Mapping) -> Mapping:
            self.calls += 1
            return {"value": payload["value"]}

    bus = InMemoryBus()
    await bus.connect()
    agent = RecordingAgent(bus)

    await agent.handle_bus_message(
        "orchestrator.generate.request",
        json.dumps(_request_payload("request-local-alias")).encode(),
    )

    assert agent.calls == 1
    await bus.close()


@pytest.mark.asyncio
async def test_request_timeout_starts_before_publish_and_cleans_subscription() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient

    class BlockingPublishBus(InMemoryBus):
        def __init__(self) -> None:
            super().__init__()
            self.release = asyncio.Event()

        async def publish(self, subject, payload):
            await self.release.wait()
            await super().publish(subject, payload)

    bus = BlockingPublishBus()
    await bus.connect()
    request_task = asyncio.create_task(
        AgentRequestClient(bus).request(
            "agent.generator_coord.request",
            _request_payload("request-publish-timeout"),
            payload_type_url=PAYLOAD_TYPE_URL,
            timeout=0.01,
        )
    )

    await asyncio.sleep(0.03)

    assert request_task.done()
    with pytest.raises(TimeoutError):
        await request_task
    assert bus.callback_count == 0
    bus.release.set()
    await bus.close()


@pytest.mark.asyncio
async def test_request_timeout_covers_subscribe() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient

    class BlockingSubscribeBus(InMemoryBus):
        async def subscribe(self, subject, cb):
            await asyncio.Event().wait()

    bus = BlockingSubscribeBus()
    await bus.connect()
    request_task = asyncio.create_task(
        AgentRequestClient(bus).request(
            "agent.generator_coord.request",
            _request_payload("request-subscribe-timeout"),
            payload_type_url=PAYLOAD_TYPE_URL,
            timeout=0.01,
        )
    )

    await asyncio.sleep(0.03)

    assert request_task.done()
    with pytest.raises(TimeoutError):
        await request_task
    assert bus.callback_count == 0
    await bus.close()


@pytest.mark.asyncio
async def test_request_timeout_does_not_wait_past_deadline_for_unsubscribe() -> None:
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient

    class BlockingUnsubscribeBus(InMemoryBus):
        async def unsubscribe(self, subscription):
            await super().unsubscribe(subscription)
            await asyncio.Event().wait()

    bus = BlockingUnsubscribeBus()
    await bus.connect()
    request_task = asyncio.create_task(
        AgentRequestClient(bus).request(
            "agent.generator_coord.request",
            _request_payload("request-unsubscribe-timeout"),
            payload_type_url=PAYLOAD_TYPE_URL,
            timeout=0.01,
        )
    )

    await asyncio.sleep(0.03)

    assert request_task.done()
    with pytest.raises(TimeoutError):
        await request_task
    assert bus.callback_count == 0
    await bus.close()


@pytest.mark.asyncio
async def test_service_rejects_request_without_response_hop_before_dispatch() -> None:
    from mf_agents.base.agent import AgentProtocolError, BaseAgent

    class RecordingAgent(BaseAgent):
        def __init__(self) -> None:
            super().__init__("generator_coord")
            self.calls = 0

        async def process(self, payload: Mapping) -> Mapping:
            self.calls += 1
            return payload

    agent = RecordingAgent()
    envelope = await _signed_request(ttl=1)

    with pytest.raises(AgentProtocolError, match="response hop"):
        await agent.handle_bus_message(
            "agent.generator_coord.request",
            envelope.SerializeToString(),
        )

    assert agent.calls == 0


def test_sender_identity_cannot_forge_default_agent_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.lineage.sigstore_signer import SigstoreSigner
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

    monkeypatch.delenv("AGENT_MESSAGE_HMAC_SECRET")
    receiver = BaseAgent("generator_coord")
    envelope = AgentMessage(
        trace_id="trace-forged",
        message_id="message-forged",
        sender="orchestrator",
        recipient="generator_coord",
        message_type="request",
        payload=b"{}",
        payload_type_url=PAYLOAD_TYPE_URL,
        ttl=2,
    )
    payload = receiver._agent_message_signing_payload(envelope)
    envelope.signature = SigstoreSigner(identity_token=envelope.sender).sign(payload)

    assert receiver.verify_agent_message(envelope) is False


def test_explicit_agent_hmac_secret_is_shared_but_wrong_secret_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

    monkeypatch.setenv("AGENT_MESSAGE_HMAC_SECRET", "sender-secret")
    sender = BaseAgent("orchestrator")
    envelope = AgentMessage(
        trace_id="trace-secret",
        message_id="message-secret",
        sender="orchestrator",
        recipient="generator_coord",
        message_type="request",
        payload=b"{}",
        payload_type_url=PAYLOAD_TYPE_URL,
        ttl=2,
    )
    envelope.signature = sender._sign_agent_message(envelope)

    monkeypatch.setenv("AGENT_MESSAGE_HMAC_SECRET", "receiver-secret")
    receiver = BaseAgent("generator_coord")

    assert receiver.verify_agent_message(envelope) is False


def test_client_rejects_expired_zero_ttl_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mf_agents.base.agent import AgentProtocolError, BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

    monkeypatch.setenv("AGENT_MESSAGE_HMAC_SECRET", "ttl-secret")
    request = AgentMessage(
        trace_id="trace-ttl-response",
        message_id="message-ttl-request",
        sender="orchestrator",
        recipient="generator_coord",
        message_type="request",
        payload=b"{}",
        payload_type_url=PAYLOAD_TYPE_URL,
        ttl=1,
        run_id="run-ttl-response",
        request_id="request-ttl-response",
        parent_id="parent-ttl-response",
        schema_version=SCHEMA_VERSION,
    )
    response = AgentMessage(
        trace_id=request.trace_id,
        message_id="message-ttl-response",
        sender="generator_coord",
        recipient="orchestrator",
        message_type="response",
        payload=json.dumps(
            {
                "run_id": request.run_id,
                "request_id": request.request_id,
                "schema_version": request.schema_version,
            },
            sort_keys=True,
        ).encode(),
        payload_type_url=request.payload_type_url,
        ttl=0,
        lineage={"parent_id": request.message_id},
        run_id=request.run_id,
        request_id=request.request_id,
        parent_id=request.message_id,
        schema_version=request.schema_version,
    )
    response.signature = BaseAgent("generator_coord")._sign_agent_message(response)

    with pytest.raises(AgentProtocolError, match="ttl expired"):
        AgentRequestClient(InMemoryBus())._decode_response(
            response.SerializeToString(),
            request=request,
            expected_sender="generator_coord",
        )


def test_external_sigstore_verification_rejects_negative_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from mf_agents.base.agent import BaseAgent
    from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

    sign_command = (
        f'{sys.executable} -c "import json;'
        "print(json.dumps({'signature':'external-signature'}))\""
    )
    verify_command = f"{sys.executable} -c \"import json;print(json.dumps({{'valid':'false'}}))\""
    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", sign_command)
    monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", verify_command)
    sender = BaseAgent("orchestrator")
    receiver = BaseAgent("generator_coord")
    envelope = AgentMessage(
        trace_id="trace-external-negative",
        message_id="message-external-negative",
        sender="orchestrator",
        recipient="generator_coord",
        message_type="request",
        payload=b"{}",
        payload_type_url=PAYLOAD_TYPE_URL,
        ttl=2,
    )
    envelope.signature = sender._sign_agent_message(envelope)

    assert receiver.verify_agent_message(envelope) is False


@pytest.mark.parametrize(
    ("sign_command", "verify_command"),
    [
        ("/usr/bin/true", ""),
        ("", "/usr/bin/true"),
    ],
)
def test_production_signing_rejects_incomplete_sigstore_with_hmac_secret(
    monkeypatch: pytest.MonkeyPatch,
    sign_command: str,
    verify_command: str,
) -> None:
    from mf_agents.base.agent import BaseAgent

    monkeypatch.setenv("SIGSTORE_SIGN_COMMAND", sign_command)
    monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", verify_command)

    assert BaseAgent("generator_coord").production_signing_configured is False


@pytest.mark.parametrize("result_field", ("valid", "signature_valid"))
def test_sigstore_signer_external_verification_rejects_string_false(
    monkeypatch: pytest.MonkeyPatch,
    result_field: str,
) -> None:
    import sys

    from mf_agents.lineage.sigstore_signer import SigstoreSigner

    verify_command = (
        f"{sys.executable} -c \"import json;print(json.dumps({{{result_field!r}:'false'}}))\""
    )
    monkeypatch.delenv("SIGSTORE_SIGN_COMMAND", raising=False)
    monkeypatch.setenv("SIGSTORE_VERIFY_COMMAND", verify_command)

    assert SigstoreSigner("orchestrator").verify(b"payload", b"signature") is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "field"),
    [
        ({"value": "ok", "run_id": "changed"}, "run_id"),
        ({"value": "ok", "request_id": "changed"}, "request_id"),
        ({"value": "ok", "schema_version": "changed"}, "schema_version"),
    ],
)
async def test_changed_process_correlation_becomes_upstream_protocol_error(
    result: Mapping,
    field: str,
) -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus
    from mf_agents.messaging.request_client import AgentRequestClient, UpstreamAgentError

    class InvalidAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self._subscription_subjects = ["agent.generator_coord.request"]

        async def process(self, payload: Mapping) -> Mapping:
            return result

    bus = InMemoryBus()
    await bus.connect()
    agent = InvalidAgent(bus)
    await agent.start()

    with pytest.raises(UpstreamAgentError) as exc_info:
        await AgentRequestClient(bus).request(
            "agent.generator_coord.request",
            _request_payload(f"request-invalid-{field}"),
            payload_type_url=PAYLOAD_TYPE_URL,
            timeout=0.5,
        )

    assert exc_info.value.upstream_type == "AgentProtocolError"
    assert field in str(exc_info.value)
    await agent.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_field", ("run_id", "request_id", "schema_version"))
async def test_compatibility_json_requires_correlation_fields(
    missing_field: str,
) -> None:
    from mf_agents.base.agent import AgentProtocolError, BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class RecordingAgent(BaseAgent):
        def __init__(self, message_bus) -> None:
            super().__init__("generator_coord", message_bus=message_bus)
            self.calls = 0

        async def process(self, payload: Mapping) -> Mapping:
            self.calls += 1
            return {"value": payload["value"]}

    bus = InMemoryBus()
    await bus.connect()
    agent = RecordingAgent(bus)
    payload = _request_payload("compatibility-request")
    payload.pop(missing_field)

    with pytest.raises(AgentProtocolError, match=f"{missing_field} is required"):
        await agent.handle_message(
            "orchestrator.generate.request",
            json.dumps(payload).encode("utf-8"),
        )

    assert agent.calls == 0
    await bus.close()


@pytest.mark.asyncio
async def test_compatibility_json_echoes_correlation_fields() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class EchoAgent(BaseAgent):
        async def process(self, payload: Mapping) -> Mapping:
            return {"value": payload["value"]}

    bus = InMemoryBus()
    await bus.connect()
    agent = EchoAgent("generator_coord", message_bus=bus)
    replies: list[dict] = []

    async def record_reply(message: dict) -> None:
        replies.append(json.loads(message["data"].decode("utf-8")))

    await bus.subscribe("compatibility.reply", record_reply)
    await agent.handle_message(
        "orchestrator.generate.request",
        json.dumps(_request_payload("compatibility-request")).encode("utf-8"),
        "compatibility.reply",
    )
    await asyncio.sleep(0)

    assert replies == [
        {
            "request_id": "compatibility-request",
            "run_id": "run-1",
            "schema_version": SCHEMA_VERSION,
            "value": "value",
        }
    ]
    await bus.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_field", ("run_id", "request_id", "schema_version"))
async def test_compatibility_json_rejects_changed_process_correlation(
    changed_field: str,
) -> None:
    from mf_agents.base.agent import AgentProtocolError, BaseAgent

    class InvalidAgent(BaseAgent):
        async def process(self, payload: Mapping) -> Mapping:
            return {"value": payload["value"], changed_field: "changed"}

    agent = InvalidAgent("generator_coord")

    with pytest.raises(AgentProtocolError, match=f"agent process changed {changed_field}"):
        await agent.handle_message(
            "orchestrator.generate.request",
            json.dumps(_request_payload("compatibility-request")).encode("utf-8"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_field", ("run_id", "request_id", "schema_version"))
async def test_compatibility_json_rejects_in_place_correlation_mutation(
    changed_field: str,
) -> None:
    from mf_agents.base.agent import AgentProtocolError, BaseAgent

    class InvalidAgent(BaseAgent):
        async def process(self, payload: Mapping) -> Mapping:
            payload[changed_field] = "changed"
            return payload

    agent = InvalidAgent("generator_coord")

    with pytest.raises(AgentProtocolError, match=f"agent process changed {changed_field}"):
        await agent.handle_message(
            "orchestrator.generate.request",
            json.dumps(_request_payload("compatibility-request")).encode("utf-8"),
        )


@pytest.mark.asyncio
async def test_compatibility_json_echoes_non_empty_legacy_schema_version() -> None:
    from mf_agents.base.agent import BaseAgent
    from mf_agents.messaging.redis_bus import InMemoryBus

    class LegacyAgent(BaseAgent):
        async def process(self, payload: Mapping) -> Mapping:
            return {"value": payload["value"]}

    bus = InMemoryBus()
    await bus.connect()
    agent = LegacyAgent("generator_coord", message_bus=bus)
    payload = _request_payload("compatibility-request")
    payload["schema_version"] = "legacy.generator.v0"
    replies: list[dict] = []

    async def record_reply(message: dict) -> None:
        replies.append(json.loads(message["data"].decode("utf-8")))

    await bus.subscribe("compatibility.reply", record_reply)
    await agent.handle_message(
        "orchestrator.generate.request",
        json.dumps(payload).encode("utf-8"),
        "compatibility.reply",
    )
    await asyncio.sleep(0)

    assert replies == [
        {
            "request_id": "compatibility-request",
            "run_id": "run-1",
            "schema_version": "legacy.generator.v0",
            "value": "value",
        }
    ]
    await bus.close()


def test_six_agents_use_base_agent_handle_message_path() -> None:
    from critic_agent.agent import ScientificCriticAgent
    from generator_coord.agent import GeneratorCoordAgent
    from retrosyn_agent.agent import RetroSynAgent
    from srb_agent.agent import SRBAgent
    from supply_agent.agent import SupplyAgent
    from validation_agent.agent import ValidationAgent

    for agent_type in (
        GeneratorCoordAgent,
        ValidationAgent,
        RetroSynAgent,
        SupplyAgent,
        SRBAgent,
        ScientificCriticAgent,
    ):
        assert "handle_message" not in agent_type.__dict__
        assert callable(agent_type.__dict__["process"])
