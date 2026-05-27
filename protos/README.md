# protos · Protocol-First Contract Layer

All inter-service communication contracts live here. **Edit `.proto` files first, then regenerate stubs.**

## Layout

```
protos/moleculeforge/v1/
├── core/          # Shared message types (Molecule, HCIV, CIG, CRG, SSP, audit)
├── agent/         # Agent messaging + orchestrator/critic services
├── generator/     # Generator and TAR router services
├── oracle/        # Oracle (Boltz-2, FEP, etc.)
├── retrosyn/      # Retrosynthesis services
└── humu/          # HUMU encoder service
```

## Workflow

```bash
# 1. Edit a .proto file
vim protos/moleculeforge/v1/generator/generator.proto

# 2. Lint
cd protos && buf lint

# 3. Detect breaking changes vs main
buf breaking --against '.git#branch=main'

# 4. Regenerate Python + Go + TS stubs
buf generate

# 5. Generated code lands in libs/mf-core/src/mf_core/proto_gen/
```

## Versioning

- Path includes major version: `moleculeforge/v1/...`.
- A `v2/` directory is added side-by-side when a breaking change is needed.
- v1 messages are **never deleted** — only deprecated and superseded.
