"""Base agent class for MoleculeForge agent system."""

import asyncio
import hashlib
import hmac
import inspect
import json
import os
import shlex
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from mf_core.artifacts import CommandRequirement, check_command, require_available
from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

from mf_agents.lineage.sigstore_signer import SigstoreSigner

AGENT_MESSAGE_TYPES = frozenset({"request", "response", "event", "error"})


class AgentProtocolError(RuntimeError):
    """Raised when an Agent envelope violates the request/reply contract."""


@dataclass(frozen=True)
class AgentProtocol:
    entry_point: str
    agent_name: str
    subject: str
    payload_type_url: str
    schema_version: str


class _HMACMessageSigner:
    def __init__(self, secret: str | bytes) -> None:
        encoded = secret.encode("utf-8") if isinstance(secret, str) else secret
        if not encoded:
            raise ValueError("Agent message HMAC secret must not be empty")
        self._secret = encoded

    def sign(self, payload: bytes) -> bytes:
        return hmac.new(self._secret, payload, hashlib.sha256).digest()

    def verify(self, payload: bytes, signature: bytes) -> bool:
        return hmac.compare_digest(self.sign(payload), signature)


AGENT_PROTOCOLS = (
    AgentProtocol(
        "generator_coord",
        "generator_coord",
        "agent.generator_coord.request",
        "type.moleculeforge.ai/agent/generator_coord/request.v1",
        "generator_coord.request.v1",
    ),
    AgentProtocol(
        "validation",
        "validation_agent",
        "agent.validation.request",
        "type.moleculeforge.ai/agent/validation/request.v1",
        "validation.request.v1",
    ),
    AgentProtocol(
        "retrosyn",
        "retrosyn_agent",
        "agent.retrosyn.request",
        "type.moleculeforge.ai/agent/retrosyn/request.v1",
        "retrosyn.request.v1",
    ),
    AgentProtocol(
        "supply",
        "supply_agent",
        "agent.supply.request",
        "type.moleculeforge.ai/agent/supply/request.v1",
        "supply.request.v1",
    ),
    AgentProtocol(
        "srb",
        "srb_agent",
        "agent.srb.request",
        "type.moleculeforge.ai/agent/srb/request.v1",
        "srb.request.v1",
    ),
    AgentProtocol(
        "critic",
        "critic_agent",
        "agent.critic.request",
        "type.moleculeforge.ai/agent/critic/request.v1",
        "critic.request.v1",
    ),
)
AGENT_PROTOCOLS_BY_NAME = {protocol.agent_name: protocol for protocol in AGENT_PROTOCOLS}
AGENT_PROTOCOLS_BY_SUBJECT = {protocol.subject: protocol for protocol in AGENT_PROTOCOLS}
_SIGSTORE_SIGN_COMMAND = CommandRequirement(
    "sigstore_sign_command",
    "SIGSTORE_SIGN_COMMAND",
    required=False,
)
_SIGSTORE_VERIFY_COMMAND = CommandRequirement(
    "sigstore_verify_command",
    "SIGSTORE_VERIFY_COMMAND",
    required=False,
)


def agent_health_check_timeout_seconds() -> float:
    timeout = float(os.getenv("AGENT_HEALTH_CHECK_TIMEOUT_SECONDS", "5"))
    if timeout <= 0:
        raise ValueError("AGENT_HEALTH_CHECK_TIMEOUT_SECONDS must be positive")
    return timeout


async def run_health_probe_in_daemon(probe: Callable[[], Any]) -> Any:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()

    def deliver(value: Any, error: BaseException | None) -> None:
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(value)

    def schedule(value: Any, error: BaseException | None) -> None:
        try:
            loop.call_soon_threadsafe(deliver, value, error)
        except RuntimeError:
            return

    def run() -> None:
        try:
            value = probe()
        except BaseException as error:
            schedule(None, error)
        else:
            schedule(value, None)

    threading.Thread(
        target=run,
        name="agent-health-probe",
        daemon=True,
    ).start()
    return await future


class BaseAgent:
    """Base class for all MoleculeForge agents.

    Agents communicate via the Redis-backed message bus and can subscribe to
    topics to receive and process messages.
    """

    def __init__(
        self,
        name: str,
        message_bus=None,
        signer: SigstoreSigner | None = None,
        hmac_secret: str | bytes | None = None,
    ):
        self.name = name
        self.message_bus = message_bus
        self.protocol = AGENT_PROTOCOLS_BY_NAME.get(name)
        configured_secret = hmac_secret
        if configured_secret is None:
            configured_secret = os.getenv("AGENT_MESSAGE_HMAC_SECRET", "")
        if signer is not None and configured_secret:
            raise ValueError("Configure either an Agent signer or HMAC secret, not both")
        self.signer = signer or (
            _HMACMessageSigner(configured_secret) if configured_secret else None
        )
        self._hmac_secret_configured = bool(configured_secret)
        self.sigstore_sign_command = os.getenv("SIGSTORE_SIGN_COMMAND", "").strip()
        self.sigstore_verify_command = os.getenv("SIGSTORE_VERIFY_COMMAND", "").strip()
        self.sigstore_rekor_url = os.getenv("SIGSTORE_REKOR_URL", "https://rekor.sigstore.dev")
        self.sigstore_identity_token = os.getenv("SIGSTORE_IDENTITY_TOKEN", "").strip()
        self._agent_signature_cache: dict[str, dict] = {}
        self._subscription_subjects: list[str] = []

    @property
    def production_signing_configured(self) -> bool:
        commands_configured = bool(self.sigstore_sign_command or self.sigstore_verify_command)
        if not commands_configured:
            return self._hmac_secret_configured
        if not self.sigstore_sign_command or not self.sigstore_verify_command:
            return False
        env = {
            **os.environ,
            "SIGSTORE_SIGN_COMMAND": self.sigstore_sign_command,
            "SIGSTORE_VERIFY_COMMAND": self.sigstore_verify_command,
        }
        return (
            check_command(_SIGSTORE_SIGN_COMMAND, env=env).available
            and check_command(_SIGSTORE_VERIFY_COMMAND, env=env).available
        )

    async def handle_message(self, subject: str, payload: bytes, reply_to: str = "") -> None:
        """Handle compatibility payloads through the common process path."""
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AgentProtocolError("agent payload must be a JSON object") from exc
        if not isinstance(data, Mapping):
            raise AgentProtocolError("agent payload must be a JSON object")
        correlation = {}
        for field in ("run_id", "request_id", "schema_version"):
            if not str(data.get(field) or ""):
                raise AgentProtocolError(f"agent payload {field} is required")
            correlation[field] = str(data[field])
        result = await self.process(dict(data))
        if not isinstance(result, Mapping):
            raise AgentProtocolError("agent process response must be a mapping")
        response = dict(result)
        for field in ("run_id", "request_id", "schema_version"):
            expected = correlation[field]
            if field in response and str(response[field]) != expected:
                raise AgentProtocolError(f"agent process changed {field}")
            response[field] = expected
        if reply_to:
            await self.publish(
                reply_to,
                json.dumps(
                    response,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )

    async def process(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError(f"{type(self).__name__}.process() is required")

    def runtime_targets(self) -> Mapping[str, Any]:
        return {}

    async def start(self) -> None:
        """Start the agent, subscribing to all registered subjects."""
        if self.message_bus:
            for subject in self._subscription_subjects:
                await self.message_bus.subscribe(subject, cb=self.handle_bus_message)

    async def stop(self) -> None:
        """Stop the agent and close the message bus connection."""
        if self.message_bus:
            await self.message_bus.close()

    async def publish(self, subject: str, payload: bytes) -> None:
        """Publish a message to a subject.

        Args:
            subject: Message subject to publish to.
            payload: Raw message payload bytes.
        """
        if self.message_bus:
            await self.message_bus.publish(subject, payload)

    async def handle_bus_message(
        self,
        subject: str,
        payload: bytes,
        reply_to: str = "",
    ) -> None:
        envelope = self._parse_agent_message(payload)
        if envelope is None or not envelope.signature:
            if self.protocol is not None and subject == self.protocol.subject:
                raise AgentProtocolError("canonical agent request requires a signed AgentMessage")
            await self.handle_message(subject, payload, reply_to)
            return
        if (
            self.protocol is not None
            and subject == self.protocol.subject
            and envelope.message_type != "request"
        ):
            raise AgentProtocolError("canonical agent subject accepts only a signed request")
        self._validate_agent_recipient(envelope.recipient)
        if envelope.recipient and envelope.recipient not in {self.name, "*"}:
            raise AgentProtocolError(f"agent message recipient mismatch: {envelope.recipient}")
        self._validate_agent_message_type(envelope.message_type)
        self._validate_agent_payload_type_url(envelope.payload_type_url)
        if envelope.ttl <= 0:
            raise AgentProtocolError("agent message ttl expired")
        if not self.verify_agent_message(envelope):
            raise AgentProtocolError("agent message signature verification failed")
        if envelope.message_type == "request" and self.protocol is not None:
            await self._handle_request(subject, envelope)
            return
        await self.handle_message(
            subject,
            envelope.payload,
            envelope.reply_to or reply_to,
        )

    @staticmethod
    def _parse_agent_message(payload: bytes) -> AgentMessage | None:
        envelope = AgentMessage()
        try:
            envelope.ParseFromString(payload)
        except Exception:
            return None
        if not (
            envelope.message_id
            or envelope.trace_id
            or envelope.sender
            or envelope.recipient
            or envelope.payload_type_url
            or envelope.signature
        ):
            return None
        return envelope

    async def publish_agent_message(
        self,
        subject: str,
        *,
        recipient: str,
        message_type: str,
        payload: bytes | str | Mapping[str, Any],
        payload_type_url: str = "",
        trace_id: str = "",
        message_id: str = "",
        reply_to: str = "",
        lineage: Mapping[str, str] | None = None,
        ttl: int = 16,
        run_id: str = "",
        request_id: str = "",
        parent_id: str = "",
        schema_version: str = "",
        jsonld_context: Mapping[str, Any] | None = None,
        jsonld_type: str = "",
        jsonld_id: str = "",
    ) -> AgentMessage:
        self._validate_agent_recipient(recipient)
        self._validate_agent_message_type(message_type)
        self._validate_agent_payload_type_url(payload_type_url)
        payload = self._encode_agent_payload(
            payload,
            jsonld_context=jsonld_context,
            jsonld_type=jsonld_type,
            jsonld_id=jsonld_id,
        )
        envelope = AgentMessage(
            trace_id=trace_id or uuid.uuid4().hex,
            message_id=message_id or _uuid7(),
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            payload_type_url=payload_type_url,
            timestamp_ns=time.time_ns(),
            lineage=dict(lineage or {}),
            ttl=ttl,
            run_id=run_id,
            request_id=request_id,
            parent_id=parent_id,
            schema_version=schema_version,
        )
        if reply_to:
            envelope.reply_to = reply_to
        envelope.signature = self._sign_agent_message(envelope)
        await self.publish(subject, envelope.SerializeToString())
        return envelope

    async def _handle_request(self, subject: str, envelope: AgentMessage) -> None:
        protocol = self.protocol
        if protocol is None:
            raise AgentProtocolError(f"agent request protocol is not configured: {self.name}")
        if subject != protocol.subject and subject not in self._subscription_subjects:
            raise AgentProtocolError(f"unexpected agent request subject: {subject}")
        if envelope.recipient != self.name:
            raise AgentProtocolError(f"agent request recipient mismatch: {envelope.recipient}")
        if envelope.payload_type_url != protocol.payload_type_url:
            raise AgentProtocolError(f"unexpected payload_type_url: {envelope.payload_type_url}")
        if envelope.schema_version != protocol.schema_version:
            raise AgentProtocolError(f"unexpected schema_version: {envelope.schema_version}")
        if envelope.ttl <= 1:
            raise AgentProtocolError("agent request ttl must allow a response hop")
        if not envelope.reply_to:
            raise AgentProtocolError("agent request reply_to is required")
        if not envelope.sender:
            raise AgentProtocolError("agent request sender is required")
        if not envelope.message_id:
            raise AgentProtocolError("agent request message_id is required")
        if not envelope.trace_id:
            raise AgentProtocolError("agent request trace_id is required")
        if not envelope.parent_id:
            raise AgentProtocolError("agent request parent_id is required")
        if not envelope.run_id:
            raise AgentProtocolError("agent request run_id is required")
        if not envelope.request_id:
            raise AgentProtocolError("agent request request_id is required")
        if envelope.lineage.get("parent_id") != envelope.parent_id:
            raise AgentProtocolError("agent request parent lineage mismatch")
        try:
            payload = json.loads(envelope.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AgentProtocolError("agent request payload must be valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise AgentProtocolError("agent request payload must be a JSON object")
        self._validate_request_payload_correlation(envelope, payload)
        try:
            result = await self.process(payload)
            response = self._correlated_process_result(envelope, result)
        except Exception as exc:
            await self._publish_request_error(envelope, exc)
            return
        await self.publish_agent_message(
            envelope.reply_to,
            recipient=envelope.sender,
            message_type="response",
            payload=response,
            payload_type_url=envelope.payload_type_url,
            trace_id=envelope.trace_id,
            reply_to="",
            lineage={"parent_id": envelope.message_id},
            ttl=envelope.ttl - 1,
            run_id=envelope.run_id,
            request_id=envelope.request_id,
            parent_id=envelope.message_id,
            schema_version=envelope.schema_version,
        )

    @staticmethod
    def _validate_request_payload_correlation(
        envelope: AgentMessage,
        payload: Mapping[str, Any],
    ) -> None:
        expected = {
            "trace_id": envelope.trace_id,
            "parent_id": envelope.parent_id,
            "run_id": envelope.run_id,
            "request_id": envelope.request_id,
            "schema_version": envelope.schema_version,
        }
        for field, value in expected.items():
            if str(payload.get(field) or "") != value:
                raise AgentProtocolError(f"agent request {field} correlation mismatch")

    @staticmethod
    def _correlated_process_result(
        envelope: AgentMessage,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            raise AgentProtocolError("agent process response must be a mapping")
        response = dict(result)
        expected = {
            "run_id": envelope.run_id,
            "request_id": envelope.request_id,
            "schema_version": envelope.schema_version,
        }
        for field, value in expected.items():
            if field in response and str(response[field]) != value:
                raise AgentProtocolError(f"agent process changed {field}")
            response[field] = value
        return response

    async def _publish_request_error(
        self,
        envelope: AgentMessage,
        error: Exception,
    ) -> None:
        await self.publish_agent_message(
            envelope.reply_to,
            recipient=envelope.sender,
            message_type="error",
            payload={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "run_id": envelope.run_id,
                "request_id": envelope.request_id,
                "schema_version": envelope.schema_version,
            },
            payload_type_url=envelope.payload_type_url,
            trace_id=envelope.trace_id,
            lineage={"parent_id": envelope.message_id},
            ttl=envelope.ttl - 1,
            run_id=envelope.run_id,
            request_id=envelope.request_id,
            parent_id=envelope.message_id,
            schema_version=envelope.schema_version,
        )

    @staticmethod
    def _validate_agent_recipient(recipient: str) -> None:
        if not recipient:
            raise ValueError("agent recipient is required")

    @staticmethod
    def _validate_agent_message_type(message_type: str) -> None:
        if message_type not in AGENT_MESSAGE_TYPES:
            allowed = ", ".join(sorted(AGENT_MESSAGE_TYPES))
            raise ValueError(f"agent message_type must be one of: {allowed}")

    @staticmethod
    def _validate_agent_payload_type_url(payload_type_url: str) -> None:
        if not payload_type_url:
            raise ValueError("agent payload_type_url is required")

    @staticmethod
    def _encode_agent_payload(
        payload: bytes | str | Mapping[str, Any],
        *,
        jsonld_context: Mapping[str, Any] | None,
        jsonld_type: str,
        jsonld_id: str,
    ) -> bytes:
        if isinstance(payload, bytes):
            if jsonld_context or jsonld_type or jsonld_id:
                raise TypeError("JSON-LD metadata requires a mapping payload")
            return payload
        if isinstance(payload, str):
            if jsonld_context or jsonld_type or jsonld_id:
                raise TypeError("JSON-LD metadata requires a mapping payload")
            return payload.encode("utf-8")
        document = dict(payload)
        if jsonld_context is not None:
            document["@context"] = dict(jsonld_context)
        if jsonld_type:
            document["@type"] = jsonld_type
        if jsonld_id:
            document["@id"] = jsonld_id
        return json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def verify_agent_message(self, message: AgentMessage | bytes) -> bool:
        if isinstance(message, bytes):
            envelope = AgentMessage()
            envelope.ParseFromString(message)
            message = envelope
        if not message.signature:
            return False
        payload = self._agent_message_signing_payload(message)
        if self.sigstore_sign_command or self.sigstore_verify_command:
            return self._verify_agent_message_with_command(message, payload, message.signature)
        if self.signer is None:
            return False
        return self.signer.verify(payload, message.signature)

    def _sign_agent_message(self, message: AgentMessage) -> bytes:
        payload = self._agent_message_signing_payload(message)
        if self.sigstore_sign_command:
            return self._sign_agent_message_with_command(message, payload)
        if self.signer is None:
            raise RuntimeError(
                "Agent message signing requires AGENT_MESSAGE_HMAC_SECRET or SIGSTORE_SIGN_COMMAND"
            )
        return self.signer.sign(payload)

    @staticmethod
    def _agent_message_signing_payload(message: AgentMessage) -> bytes:
        envelope = AgentMessage()
        envelope.CopyFrom(message)
        envelope.signature = b""
        return envelope.SerializeToString(deterministic=True)

    def _sign_agent_message_with_command(
        self,
        message: AgentMessage,
        payload: bytes,
    ) -> bytes:
        payload_hash = hashlib.sha256(payload).hexdigest()
        command_payload = {
            "artifact_id": message.message_id,
            "artifact_type": "agent_message",
            "payload_hash": payload_hash,
            "trace_id": message.trace_id,
            "sender": message.sender,
            "recipient": message.recipient,
            "message_type": message.message_type,
            "rekor_url": self.sigstore_rekor_url,
            "identity_token": self.sigstore_identity_token,
        }
        timeout = float(os.getenv("SIGSTORE_COMMAND_TIMEOUT_SECONDS", "30"))
        _require_command_available(_SIGSTORE_SIGN_COMMAND, self.sigstore_sign_command)
        result = subprocess.run(
            shlex.split(self.sigstore_sign_command),
            input=json.dumps(command_payload, sort_keys=True).encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Agent message signing command failed: {stderr}")
        try:
            response = json.loads(result.stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Agent message signing command returned invalid JSON") from exc
        if not isinstance(response, dict) or not response.get("signature"):
            raise RuntimeError("Agent message signing command must return a signature")
        signature = str(response["signature"])
        self._agent_signature_cache[signature] = {
            "payload_hash": str(response.get("payload_hash") or payload_hash),
            "signature_type": str(response.get("signature_type") or "sigstore_rekor"),
            "certificate": response.get("certificate"),
            "rekor_entry": response.get("rekor_entry"),
            "bundle": response.get("bundle"),
        }
        return signature.encode("utf-8")

    def _verify_agent_message_with_command(
        self,
        message: AgentMessage,
        payload: bytes,
        signature: bytes,
    ) -> bool:
        payload_hash = hashlib.sha256(payload).hexdigest()
        signature_text = signature.decode("utf-8", errors="replace")
        cached = self._agent_signature_cache.get(signature_text)
        if self.sigstore_verify_command:
            command_payload = {
                "artifact_id": message.message_id,
                "artifact_type": "agent_message",
                "payload_hash": payload_hash,
                "signature": signature_text,
                "trace_id": message.trace_id,
                "sender": message.sender,
                "recipient": message.recipient,
                "message_type": message.message_type,
                "expected_identity": message.sender or self.name,
                "bundle": cached or {},
                "rekor_url": self.sigstore_rekor_url,
            }
            timeout = float(os.getenv("SIGSTORE_COMMAND_TIMEOUT_SECONDS", "30"))
            _require_command_available(
                _SIGSTORE_VERIFY_COMMAND,
                self.sigstore_verify_command,
            )
            result = subprocess.run(
                shlex.split(self.sigstore_verify_command),
                input=json.dumps(command_payload, sort_keys=True).encode("utf-8"),
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"Agent message verification command failed: {stderr}")
            try:
                response = json.loads(result.stdout.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Agent message verification command returned invalid JSON"
                ) from exc
            if not isinstance(response, dict):
                raise RuntimeError("Agent message verification command must return a JSON object")
            if "valid" in response:
                return response["valid"] is True
            if "signature_valid" in response:
                return response["signature_valid"] is True
            raise RuntimeError("Agent message verification command must return valid")
        return bool(cached and cached.get("payload_hash") == payload_hash)

    async def read_shared_crg(self, run_id: str) -> dict:
        repository = getattr(self, "crg_repository", None)
        if repository is None:
            raise RuntimeError("crg_repository is not configured")
        read_crg = getattr(repository, "get_run_crg", None)
        if not callable(read_crg):
            raise TypeError("crg_repository must expose get_run_crg(run_id)")
        result = read_crg(run_id)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, dict):
            raise TypeError("get_run_crg(run_id) must return a dictionary")
        return result


def _uuid7() -> str:
    timestamp_ms = time.time_ns() // 1_000_000
    random_bits = uuid.uuid4().int & ((1 << 74) - 1)
    value = (timestamp_ms & ((1 << 48) - 1)) << 80
    value |= 0x7 << 76
    value |= ((random_bits >> 62) & 0x0FFF) << 64
    value |= 0b10 << 62
    value |= random_bits & ((1 << 62) - 1)
    return str(uuid.UUID(int=value))


def _require_command_available(
    requirement: CommandRequirement,
    command: str,
) -> None:
    required_requirement = CommandRequirement(
        requirement.name,
        requirement.env_var,
        required=True,
    )
    env = {**os.environ, requirement.env_var: command}
    require_available([check_command(required_requirement, env=env)])


def ensure_default_event_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
