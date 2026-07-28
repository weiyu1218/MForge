"""Agent request/reply protocol behavior."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

import pytest

PAYLOAD_TYPE_URL = "type.moleculeforge.ai/agent/generator_coord/request.v1"
SCHEMA_VERSION = "generator_coord.request.v1"
TEST_AGENT_HMAC_SECRET = "task-3-agent-test-secret"


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
):
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
