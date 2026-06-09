# W9 HFM Decoder Production Training Run Plan

Date: 2026-06-06
Scope: Owner A, W9 HFM-3D neural geometry decoder production-training preparation

## Goal

Prepare a real HFM neural geometry decoder candidate without overwriting the
protected `checkpoints/hfm3d_4h200/` smoke artifact. This document is a run plan
only. It does not authorize a long training job, deployment default changes, or
production promotion.

## Current Baseline

Engineering path:

- Source loader:
  `mf_generators.hfm_3d.decoder.neural_geometry_decoder.load_geometry_training_examples()`
- Training/export helper:
  `mf_generators.hfm_3d.decoder.neural_geometry_decoder.train_geometry_decoder_artifact()`
- CLI wrapper:
  `models/mf-generators/hfm_3d/train_geometry_decoder.py`
- Runner:
  `python -m mf_generators.hfm_3d.decoder.neural_geometry_decoder --artifact <artifact.pt>`
- Runtime contract:
  `HFM_MOLECULAR_DECODER_COMMAND` JSON stdin/stdout contract consumed by
  `HFM3DGenerator`.

Current smoke artifact:

- `checkpoints/hfm3d_4h200/decoder.json`
- Entries: 1
- Molecule: ethanol / `CCO`
- Latent dimension: 129
- SDF: present
- HUMU checkpoint provenance: pytest temp path
- Status: smoke/full-flow only, not production geometry evidence

## Source Artifact Requirements

Use a new source artifact path under:

```text
data/processing/generator_artifacts/hfm_decoder_source_YYYYMMDD_<run_id>.json
```

Required JSON shape:

```json
{
  "schema": "moleculeforge.hfm_3d.decoder_source.v1",
  "humu_checkpoint": "checkpoints/humu/best_model.pt",
  "entries": [
    {
      "id": "stable-molecule-id",
      "smiles": "CCO",
      "latent": [1.0, 0.0],
      "sdf": "SDF mol block text"
    }
  ]
}
```

The `latent` example above is abbreviated for readability. Each real entry must
contain exactly 129 numeric Lorentz full-coordinate values.

Minimum source gates for the first production-candidate run:

- at least 1000 valid entries after loader validation;
- no pytest temp paths, `/tmp/pytest-*` paths, or transient local checkpoint
  provenance;
- every entry has non-empty `id`, canonicalizable `smiles`, 129-dimensional
  Lorentz-valid `latent`, and RDKit-parseable `sdf`;
- SDF canonical SMILES matches the entry SMILES after hydrogen removal;
- duplicate canonical SMILES are either removed or carry explicit conformer IDs
  in `id`;
- max observed heavy-atom count is less than or equal to the planned
  `--max-atoms`;
- source notes record molecule count, unique canonical SMILES count, element
  coverage, atom-count range, source data provenance, and HUMU checkpoint
  provenance.

Run this source preflight before training:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/hfm_3d/src" \
  .venv/bin/python - <<'PY'
from collections import Counter
from pathlib import Path

from mf_generators.hfm_3d.decoder.neural_geometry_decoder import (
    load_geometry_training_examples,
)

source = Path(
    "data/processing/generator_artifacts/"
    "hfm_decoder_source_YYYYMMDD_<run_id>.json"
)
examples = load_geometry_training_examples(source)
unique_smiles = {example.smiles for example in examples}
atom_counts = [len(example.atom_types) for example in examples]
elements = Counter(atom for example in examples for atom in example.atom_types)

assert len(examples) >= 1000, len(examples)
assert len(unique_smiles) >= 1000, len(unique_smiles)
assert max(atom_counts) <= 64, max(atom_counts)
assert min(atom_counts) >= 1, min(atom_counts)
assert all(example.latent.numel() == 129 for example in examples)

print("entries", len(examples))
print("unique_smiles", len(unique_smiles))
print("atom_count_min", min(atom_counts))
print("atom_count_max", max(atom_counts))
print("elements", dict(sorted(elements.items())))
PY
```

If a legitimate source has repeated SMILES for multiple conformers, replace the
`len(unique_smiles) >= 1000` assertion with a recorded conformer-aware criterion
before launching training.

## Candidate Output Directory

Use a new non-protected directory:

```text
checkpoints/hfm_geometry_decoder_candidate_YYYYMMDD_<run_id>/
```

Expected files after a complete run:

- `neural_geometry_decoder.pt`
- `source.sha256`
- `runner_smoke.json`
- `hfm_generator_smoke.json`
- `training_run_record.md`

Do not write to:

- `checkpoints/hfm3d_4h200`
- `checkpoints/humu`
- `checkpoints/fragfm`
- any existing production or historical artifact directory

## Training Command Template

Record the source hash before training:

```bash
mkdir -p checkpoints/hfm_geometry_decoder_candidate_YYYYMMDD_<run_id>
sha256sum \
  data/processing/generator_artifacts/hfm_decoder_source_YYYYMMDD_<run_id>.json \
  > checkpoints/hfm_geometry_decoder_candidate_YYYYMMDD_<run_id>/source.sha256
```

Train a first production candidate:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/hfm_3d/src" \
  .venv/bin/python models/mf-generators/hfm_3d/train_geometry_decoder.py \
    --decoder-artifact data/processing/generator_artifacts/hfm_decoder_source_YYYYMMDD_<run_id>.json \
    --output-artifact checkpoints/hfm_geometry_decoder_candidate_YYYYMMDD_<run_id>/neural_geometry_decoder.pt \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --max-atoms 64 \
    --device cpu
```

Notes:

- `--device cpu` is conservative for the current workstation. Use GPU only after
  confirming resource availability and that it will not interfere with external
  `/workspace/SemMol` work.
- `--max-atoms` must be greater than or equal to the source preflight maximum.
- Do not resume from or write into `checkpoints/hfm3d_4h200/`.
- Do not change HUMU pretraining, HFM Lorentz flow architecture, or benchmark
  thresholds for this run.

## Required Post-Training Checks

Artifact load check:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/hfm_3d/src" \
  .venv/bin/python - <<'PY'
from pathlib import Path

from mf_generators.hfm_3d.decoder.neural_geometry_decoder import (
    NeuralGeometryDecoderArtifact,
)

artifact_path = Path(
    "checkpoints/hfm_geometry_decoder_candidate_YYYYMMDD_<run_id>/"
    "neural_geometry_decoder.pt"
)
artifact = NeuralGeometryDecoderArtifact.load(artifact_path, map_location="cpu")

assert artifact.latent_dim == 129, artifact.latent_dim
assert artifact.max_atoms == 64, artifact.max_atoms
assert len(artifact.entries) >= 1000, len(artifact.entries)
assert artifact.source_decoder_artifact.endswith(
    "hfm_decoder_source_YYYYMMDD_<run_id>.json"
)

print("latent_dim", artifact.latent_dim)
print("max_atoms", artifact.max_atoms)
print("entries", len(artifact.entries))
print("source_decoder_artifact", artifact.source_decoder_artifact)
PY
```

Runner smoke:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/hfm_3d/src" \
  .venv/bin/python - <<'PY' > /tmp/hfm_decoder_request_YYYYMMDD_<run_id>.json
import json
from pathlib import Path

source = Path(
    "data/processing/generator_artifacts/"
    "hfm_decoder_source_YYYYMMDD_<run_id>.json"
)
payload = json.loads(source.read_text(encoding="utf-8"))
latent = payload["entries"][0]["latent"]
print(json.dumps({"latent": latent}, separators=(",", ":"), sort_keys=True))
PY

PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/hfm_3d/src" \
  .venv/bin/python -m mf_generators.hfm_3d.decoder.neural_geometry_decoder \
    --artifact checkpoints/hfm_geometry_decoder_candidate_YYYYMMDD_<run_id>/neural_geometry_decoder.pt \
    < /tmp/hfm_decoder_request_YYYYMMDD_<run_id>.json \
    > checkpoints/hfm_geometry_decoder_candidate_YYYYMMDD_<run_id>/runner_smoke.json

.venv/bin/python - <<'PY'
import json
from pathlib import Path

payload = json.loads(
    Path(
        "checkpoints/hfm_geometry_decoder_candidate_YYYYMMDD_<run_id>/"
        "runner_smoke.json"
    ).read_text(encoding="utf-8")
)
assert payload["smiles"]
assert payload["atom_types"]
assert payload["coordinates"]
assert payload["metadata"]["decoder_mode"] == "neural_geometry_decoder"
assert len(payload["atom_types"]) == len(payload["coordinates"])
print("runner_smoke_pass", payload["smiles"], len(payload["atom_types"]))
PY
```

HFM generator smoke through the production command contract:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/hfm_3d/src" \
HFM_MOLECULAR_DECODER_COMMAND=".venv/bin/python -m mf_generators.hfm_3d.decoder.neural_geometry_decoder --artifact checkpoints/hfm_geometry_decoder_candidate_YYYYMMDD_<run_id>/neural_geometry_decoder.pt" \
  .venv/bin/python - <<'PY'
import asyncio
import json
from pathlib import Path

from mf_generators.hfm_3d import HFM3DGenerator

async def main() -> None:
    generator = HFM3DGenerator(
        checkpoint_path="checkpoints/hfm3d_4h200/best_model.pt",
        mode="production_real",
    )
    molecules = await generator.generate(
        batch_size=1,
        sampling_seed=42,
        flow_steps=0,
    )
    molecule = molecules[0]
    assert molecule.smiles
    assert molecule.sdf_bytes
    assert molecule.metadata["decoder_mode"] == "neural_geometry_decoder"
    output = {
        "smiles": molecule.smiles,
        "decoder_mode": molecule.metadata["decoder_mode"],
        "decoder_entry_id": molecule.metadata.get("decoder_entry_id", ""),
        "sdf_bytes": len(molecule.sdf_bytes or b""),
    }
    Path(
        "checkpoints/hfm_geometry_decoder_candidate_YYYYMMDD_<run_id>/"
        "hfm_generator_smoke.json"
    ).write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, sort_keys=True))

asyncio.run(main())
PY
```

## Benchmark And Promotion Evidence

Record these before any W9 promotion decision:

- exact source artifact path and hash;
- exact training command and environment, excluding secrets;
- wall-clock training time;
- source preflight summary;
- artifact load check output;
- runner smoke result;
- HFM generator smoke result;
- held-out or external geometry benchmark method and result;
- selected deployment mode: `HFM_MOLECULAR_DECODER_COMMAND`,
  `HFM_DECODER_PATH`, or injected decoder;
- cluster readiness evidence with real service config and request/response logs.

Current benchmark caveat:

- Existing HFM benchmark helpers require `HFM_DECODER_PATH`; they do not exercise
  a command-only neural decoder path.
- Do not edit W5 benchmark thresholds to make W9 pass.
- If command-decoder benchmark wiring is needed, treat it as a separate
  coordination task because W5 benchmark ownership belongs to Owner B.

## Stop Conditions

Stop and ask before:

- launching the training command above;
- lowering source-data requirements;
- changing epochs, model size, output directory, or device for a long run;
- choosing the production deployment mode;
- changing Docker/Kubernetes/Helm HFM defaults;
- editing W5 benchmark thresholds or Owner B implementation files;
- overwriting `checkpoints/hfm3d_4h200/`;
- modifying HUMU pretraining or HFM Lorentz flow architecture;
- killing or modifying external `/workspace/SemMol` processes.

## Back-Check

- This plan does not start training.
- This plan does not modify code.
- This plan writes future candidates only to a new non-protected directory.
- This plan keeps `checkpoints/hfm3d_4h200/decoder.json` documented as smoke
  evidence only.
