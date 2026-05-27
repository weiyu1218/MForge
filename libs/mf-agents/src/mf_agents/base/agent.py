"""Base agent class for MoleculeForge agent system."""
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for all MoleculeForge agents.

    Agents communicate via the Redis-backed message bus and can subscribe to
    topics to receive and process messages.
    """

    def __init__(self, name: str, message_bus=None):
        self.name = name
        self.message_bus = message_bus
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
                await self.message_bus.subscribe(subject, cb=self.handle_message)

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
