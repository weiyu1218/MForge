"""Correlated Agent request/reply client."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Mapping
from typing import Any

from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

from mf_agents.base.agent import (
    AGENT_PROTOCOLS_BY_SUBJECT,
    AgentProtocolError,
    BaseAgent,
    _uuid7,
)
from mf_agents.messaging.redis_bus import _bounded_unsubscribe, _deadline_after


class UpstreamAgentError(RuntimeError):
    """Typed error returned by an upstream Agent."""

    def __init__(
        self,
        upstream_type: str,
        message: str,
        *,
        run_id: str,
        request_id: str,
    ) -> None:
        super().__init__(message)
        self.upstream_type = upstream_type
        self.run_id = run_id
        self.request_id = request_id


class _RequestPeer(BaseAgent):
    async def process(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return payload


class AgentRequestClient:
    def __init__(
        self,
        message_bus,
        *,
        sender: str = "orchestrator",
        ttl: int = 16,
    ) -> None:
        if ttl <= 1:
            raise ValueError("request ttl must allow a response hop")
        self.message_bus = message_bus
        self.sender = sender
        self.ttl = ttl
        self._peer = _RequestPeer(sender, message_bus=message_bus)

    async def request(
        self,
        subject: str,
        payload: Mapping,
        *,
        payload_type_url: str,
        timeout: float,
    ) -> Mapping:
        protocol = AGENT_PROTOCOLS_BY_SUBJECT.get(subject)
        if protocol is None:
            raise AgentProtocolError(f"unknown canonical Agent subject: {subject}")
        request_payload = dict(payload)
        self._validate_request_payload(
            request_payload,
            payload_type_url=payload_type_url,
            expected_payload_type_url=protocol.payload_type_url,
            expected_schema_version=protocol.schema_version,
        )
        reply_to = f"_reply.{self.sender}.{uuid.uuid4().hex}"
        envelope = AgentMessage(
            trace_id=str(request_payload["trace_id"]),
            message_id=_uuid7(),
            sender=self.sender,
            recipient=protocol.agent_name,
            message_type="request",
            reply_to=reply_to,
            payload=json.dumps(
                request_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
            payload_type_url=payload_type_url,
            timestamp_ns=time.time_ns(),
            lineage={"parent_id": str(request_payload["parent_id"])},
            ttl=self.ttl,
            run_id=str(request_payload["run_id"]),
            request_id=str(request_payload["request_id"]),
            parent_id=str(request_payload["parent_id"]),
            schema_version=str(request_payload["schema_version"]),
        )
        envelope.signature = self._peer._sign_agent_message(envelope)
        future: asyncio.Future[Mapping] = asyncio.get_running_loop().create_future()
        deadline = _deadline_after(timeout)

        async def on_response(message: dict[str, Any]) -> None:
            if future.done():
                return
            try:
                result = self._decode_response(
                    message["data"],
                    request=envelope,
                    expected_sender=protocol.agent_name,
                )
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

        subscription = None
        try:
            async with asyncio.timeout_at(deadline):
                subscription = await self.message_bus.subscribe(reply_to, on_response)
                await self.message_bus.publish(subject, envelope.SerializeToString())
                return await future
        finally:
            if subscription is not None:
                await _bounded_unsubscribe(self.message_bus, subscription, deadline)

    @staticmethod
    def _validate_request_payload(
        payload: Mapping[str, Any],
        *,
        payload_type_url: str,
        expected_payload_type_url: str,
        expected_schema_version: str,
    ) -> None:
        if payload_type_url != expected_payload_type_url:
            raise AgentProtocolError(f"unexpected payload_type_url: {payload_type_url}")
        for field in ("trace_id", "parent_id", "run_id", "request_id", "schema_version"):
            if not str(payload.get(field) or ""):
                raise AgentProtocolError(f"agent request {field} is required")
        if str(payload["schema_version"]) != expected_schema_version:
            raise AgentProtocolError(f"unexpected schema_version: {payload['schema_version']}")

    def _decode_response(
        self,
        encoded: bytes,
        *,
        request: AgentMessage,
        expected_sender: str,
    ) -> Mapping:
        response = AgentMessage()
        try:
            response.ParseFromString(encoded)
        except Exception as exc:
            raise AgentProtocolError("upstream response is not an AgentMessage") from exc
        if not response.signature or not self._peer.verify_agent_message(response):
            raise AgentProtocolError("upstream response signature verification failed")
        if response.ttl <= 0:
            raise AgentProtocolError("upstream response ttl expired")
        expected = {
            "sender": expected_sender,
            "recipient": self.sender,
            "trace_id": request.trace_id,
            "parent_id": request.message_id,
            "run_id": request.run_id,
            "request_id": request.request_id,
            "schema_version": request.schema_version,
            "payload_type_url": request.payload_type_url,
        }
        for field, value in expected.items():
            if str(getattr(response, field)) != value:
                raise AgentProtocolError(f"upstream response {field} correlation mismatch")
        if response.lineage.get("parent_id") != request.message_id:
            raise AgentProtocolError("upstream response parent lineage mismatch")
        if response.ttl != request.ttl - 1:
            raise AgentProtocolError("upstream response ttl mismatch")
        if response.message_type not in {"response", "error"}:
            raise AgentProtocolError(f"unexpected upstream message_type: {response.message_type}")
        try:
            payload = json.loads(response.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AgentProtocolError("upstream response payload must be valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise AgentProtocolError("upstream response payload must be a JSON object")
        for field in ("run_id", "request_id", "schema_version"):
            if str(payload.get(field) or "") != str(getattr(request, field)):
                raise AgentProtocolError(f"upstream response payload {field} correlation mismatch")
        if response.message_type == "error":
            raise UpstreamAgentError(
                str(payload.get("error_type") or "UpstreamAgentError"),
                str(payload.get("error_message") or "upstream Agent failed"),
                run_id=request.run_id,
                request_id=request.request_id,
            )
        return dict(payload)
