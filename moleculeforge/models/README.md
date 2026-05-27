# models/

All ML implementations live here. Each subdirectory is an independent
Python package registered as an entry-point plugin (see `mf-core/registry`).

| Sub-tree | Plugin group | Count |
|----------|--------------|-------|
| `mf-generators/`   | `moleculeforge.generators` | 8 |
| `mf-oracles/`      | `moleculeforge.oracles`    | 5 |
| `mf-retrosyn/`     | `moleculeforge.retrosyn`   | 3 |
| `mf-encoders/`     | `moleculeforge.encoders`   | 4 |

All packages share the layout in `MoleculeForge_CodeArchitecture.md` §8 —
`generator.py`/`oracle.py`/`retrosyn.py`/`encoder.py` is the public entry
point that implements the corresponding ABC from `mf-core.plugins`.

Checkpoints live under each package's `checkpoints/` (DVC-tracked, not in
git). Default config files live under `configs/`.
