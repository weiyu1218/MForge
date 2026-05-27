# mf-core

Core data structures, plugin abstract base classes, and protobuf-generated stubs.
**Stable, side-effect-free, and depended on by every other layer.**

## Layout

```
src/mf_core/
├── proto_gen/    Generated from protos/ (do not edit by hand)
├── types/        Pydantic models: MoleculeModel, CIG, CRG, HCIV, SSP, Pareto, AuditMessage
├── plugins/      ABCs: BaseGenerator, BaseOracle, BaseRetrosynModel, BaseEncoder
├── registry/     Plugin discovery via `importlib.metadata` entry-points
├── exceptions/   Unified exception tree
└── utils/        UUID, hashing, time helpers
```

## Plugin registration

Each plugin's `pyproject.toml` declares an entry-point in the right group:

```toml
[project.entry-points."moleculeforge.generators"]
hfm_3d = "mf_generators.hfm_3d:HFM3DGenerator"
```

At runtime:

```python
from mf_core.registry import PluginRegistry
gens = PluginRegistry.load_generators()    # {"hfm_3d": HFM3DGenerator, ...}
```
