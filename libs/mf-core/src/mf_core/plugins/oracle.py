from abc import ABC, abstractmethod


class OraclePlugin(ABC):
    @abstractmethod
    async def evaluate(
        self, molecules: list[str], properties: list[str]
    ) -> dict[str, dict[str, float]]:
        ...

    @abstractmethod
    async def predict_with_uncertainty(
        self, molecules: list[str], properties: list[str]
    ) -> dict[str, tuple[dict, dict]]:
        ...

    @abstractmethod
    def oracle_level(self) -> int:
        ...
