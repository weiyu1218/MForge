from abc import ABC, abstractmethod
from typing import AsyncIterator
from mf_core.types.molecule import Molecule
from mf_core.types.humu import IntentCone


class GeneratorPlugin(ABC):
    @abstractmethod
    async def generate(
        self,
        batch_size: int,
        intent_cone: IntentCone | None = None,
        **kwargs,
    ) -> list[Molecule]:
        ...

    @abstractmethod
    async def info(self) -> dict:
        ...

    async def generate_stream(
        self, batch_size: int, total: int, **kwargs
    ) -> AsyncIterator[list[Molecule]]:
        for _ in range(0, total, batch_size):
            yield await self.generate(batch_size, **kwargs)
