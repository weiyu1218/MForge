# mf-chem

Single chemistry abstraction layer used by every service. Direct RDKit
imports outside of `adapters/` are forbidden by the import-linter contract.

## Layout

```
src/mf_chem/
├── molecule/        SMILES/InChI parsing, canonicalisation, descriptors,
│                    conformers, fingerprints
├── pharmacophore/   3D pharmacophore extraction / matching / alignment
├── reaction/        SMARTS parsing, reaction tree, templates
├── filters/         PAINS / leadlike / Lipinski / reactive groups
└── adapters/        rdkit_adapter, openbabel_adapter, openff_adapter
```
