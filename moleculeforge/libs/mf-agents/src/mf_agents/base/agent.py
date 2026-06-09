"""Base agent class for MoleculeForge agent system."""
import asyncio
import hashlib
import inspect
import json
import os
import shlex
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from mf_core.artifacts import CommandRequirement, check_command, require_available
from mf_core.proto_gen.moleculeforge.v1.agent.message_pb2 import AgentMessage

from mf_agents.lineage.sigstore_signer import SigstoreSigner

AGENT_MESSAGE_TYPES = frozenset({"request", "response", "event", "error"})
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


class BaseAgent(ABC):
    """Abstract base class for all MoleculeForge agents.

    Agents communicate via the Redis-backed message bus and can subscribe to
    topics to receive and process messages.
    """

    def __init__(self, name: str, message_bus=None, signer: SigstoreSigner | None = None):
        self.name = name
        self.message_bus = message_bus
        self.signer = signer or SigstoreSigner(identity_token=name)
        self.sigstore_sign_command = os.getenv("SIGSTORE_SIGN_COMMAND", "").strip()
        self.sigstore_verify_command = os.getenv("SIGSTORE_VERIFY_COMMAND", "").strip()
        self.sigstore_rekor_url = os.getenv("SIGSTORE_REKOR_URL", "https://rekor.sigstore.dev")
        self.sigstore_identity_token = os.getenv("SIGSTORE_IDENTITY_TOKEN", "").strip()
        self._agent_signature_cache: dict[str, dict] = {}
        self._subscription_subjects: list[str] = []

    @abstractmethod
    async def handle_message(
        self, subject: str, payload: bytes, reply_to: str = ""
    ) -> None:
        """Handle an incoming message on a subscribed subject.

        Args:
            subject: Message subject the payload was published on.
            payload: Raw message payload bytes.
            reply_to: Optional reply subject for request-reply pattern.
        """
        ...

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
            await self.handle_message(subject, payload, reply_to)
            return
        self._validate_agent_recipient(envelope.recipient)
        if envelope.recipient and envelope.recipient not in {self.name, "*"}:
            raise RuntimeError(
                f"agent message recipient mismatch: {envelope.recipient}"
            )
        self._validate_agent_message_type(envelope.message_type)
        self._validate_agent_payload_type_url(envelope.payload_type_url)
        if envelope.ttl <= 0:
            raise RuntimeError("agent message ttl expired")
        if not self.verify_agent_message(envelope):
            raise RuntimeError("agent message signature verification failed")
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
        )
        if reply_to:
            envelope.reply_to = reply_to
        envelope.signature = self._sign_agent_message(envelope)
        await self.publish(subject, envelope.SerializeToString())
        return envelope

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
        signer = SigstoreSigner(identity_token=message.sender or self.name)
        return signer.verify(payload, message.signature)

    def _sign_agent_message(self, message: AgentMessage) -> bytes:
        payload = self._agent_message_signing_payload(message)
        if self.sigstore_sign_command:
            return self._sign_agent_message_with_command(message, payload)
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
                raise RuntimeError(
                    "Agent message verification command must return a JSON object"
                )
            if "valid" in response:
                return bool(response["valid"])
            if "signature_valid" in response:
                return bool(response["signature_valid"])
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
