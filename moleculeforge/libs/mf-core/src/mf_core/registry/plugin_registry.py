from importlib.metadata import entry_points
from typing import TypeVar, Generic
from mf_core.plugins.generator import GeneratorPlugin
from mf_core.plugins.oracle import OraclePlugin

T = TypeVar("T")


class PluginRegistry(Generic[T]):
    def __init__(self, group: str):
        self._group = group
        self._plugins: dict[str, type] = {}

    def discover(self) -> dict[str, type]:
        discovered = {}
        try:
            for ep in entry_points(group=self._group):
                cls = ep.load()
                discovered[ep.name] = cls
        except Exception:
            pass  # entry_points lookup may fail if package not installed
        return discovered

    def register(self, name: str, plugin_cls: type) -> None:
        self._plugins[name] = plugin_cls

    def get(self, name: str) -> type | None:
        if name in self._plugins:
            return self._plugins[name]
        discovered = self.discover()
        return discovered.get(name)

    def list_all(self) -> list[str]:
        discovered = self.discover()
        all_names = set(self._plugins.keys()) | set(discovered.keys())
        return sorted(all_names)


generator_registry = PluginRegistry[GeneratorPlugin]("moleculeforge.generators")
oracle_registry = PluginRegistry[OraclePlugin]("moleculeforge.oracles")
