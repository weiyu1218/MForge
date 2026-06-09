# W10 HCIV Production Training Run Plan

Date: 2026-06-06
Scope: Owner A, W10 Enc_intent / HCIV production-training preparation

## Goal

Prepare a production-candidate `HCIVEncoder` checkpoint for `HCIV_CHECKPOINT_PATH`
without changing CIG compiler semantics, HUMU pretraining, HFM steering rules, or
deployment defaults. This document is a run plan only. It does not authorize a
long training job or production promotion.

## Current Baseline

Engineering path:

- Source loader:
  `cig_compiler_svc.domain.hciv_training.load_hciv_training_examples()`
- Training/export helper:
  `cig_compiler_svc.domain.hciv_training.train_hciv_encoder_checkpoint()`
- CLI wrapper:
  `services/cig-compiler-svc/train_hciv_encoder.py`
- Production loader:
  `cig_compiler_svc.domain.hciv_encoder.load_hciv_encoder_checkpoint()`
- Runtime env:
  `HCIV_CHECKPOINT_PATH`

Current local tests use tiny `dim=8` data. Production CIG compiler and training
CLI use `dim=128`, so production targets must have `129` Lorentz
full-coordinate values.

## Source Data Requirements

Use a new source data path under:

```text
data/processing/generator_artifacts/hciv_supervised_train_YYYYMMDD_<run_id>.jsonl
```

Required JSONL shape:

```json
{"id":"stable-example-id","cig":{"intent_id":"intent-1","objective_nodes":[{"id":"obj_qed","type":"continuous_maximize","oracle":"rdkit","weight":1.0}],"source_user_input":"maximize QED"},"target_hciv":{"coordinates":[1.0,0.0]},"weight":1.0}
```

The coordinates example above is abbreviated for readability. Each production
record must contain exactly 129 numeric Lorentz full-coordinate values.

Minimum source gates for the first production-candidate run:

- at least 1000 valid records after loader validation;
- every record has a stable non-empty `id`;
- every record has a `cig` object parseable as `ChemicalIntentGraph`;
- every `target_hciv` has 129 finite Lorentz-valid coordinates;
- no target is self-distilled from `hash_encode_hciv()` or the random local-demo
  encoder;
- records cover multiple objective types, oracle names, objective weights, and
  directed edge or hyperedge patterns;
- source notes record example count, objective-type coverage, oracle coverage,
  target provenance, and any train/validation split.

Run this source preflight before training:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:services/cig-compiler-svc/src" \
  .venv/bin/python - <<'PY'
from collections import Counter
from pathlib import Path

from cig_compiler_svc.domain.hciv_training import load_hciv_training_examples

source = Path(
    "data/processing/generator_artifacts/"
    "hciv_supervised_train_YYYYMMDD_<run_id>.jsonl"
)
examples = load_hciv_training_examples(source, dim=128, curvature=1.0)
intent_ids = {example.cig.intent_id for example in examples}
objective_types = Counter(
    str(node.type) for example in examples for node in example.cig.objective_nodes
)
oracles = Counter(
    str(node.oracle) for example in examples for node in example.cig.objective_nodes
)
edge_count = sum(len(getattr(example.cig, "edges", [])) for example in examples)
hyperedge_count = sum(
    len(getattr(example.cig, "hyperedges", [])) for example in examples
)

assert len(examples) >= 1000, len(examples)
assert len(intent_ids) >= 1000, len(intent_ids)
assert objective_types, "missing objective coverage"
assert edge_count + hyperedge_count > 0, "missing directed topology examples"
assert all(example.target_coordinates.numel() == 129 for example in examples)

print("examples", len(examples))
print("unique_intents", len(intent_ids))
print("objective_types", dict(sorted(objective_types.items())))
print("oracles", dict(sorted(oracles.items())))
print("edge_count", edge_count)
print("hyperedge_count", hyperedge_count)
PY
```

If the approved source intentionally contains repeated `intent_id` values, record
the split and replace the `len(intent_ids) >= 1000` assertion with the approved
deduplication criterion before training.

## Candidate Output Directory

Use a new non-protected directory:

```text
checkpoints/hciv_encoder_candidate_YYYYMMDD_<run_id>/
```

Expected files after a complete run:

- `hciv_encoder.pt`
- `hciv_encoder.manifest.json`
- `source.sha256`
- `compiler_smoke.json`
- `training_run_record.md`

Do not write to:

- `checkpoints/humu`
- `checkpoints/hfm3d_4h200`
- `checkpoints/fragfm`
- any existing production or historical artifact directory

## Training Command Template

Record the source hash before training:

```bash
mkdir -p checkpoints/hciv_encoder_candidate_YYYYMMDD_<run_id>
sha256sum \
  data/processing/generator_artifacts/hciv_supervised_train_YYYYMMDD_<run_id>.jsonl \
  > checkpoints/hciv_encoder_candidate_YYYYMMDD_<run_id>/source.sha256
```

Train a first production candidate:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:services/cig-compiler-svc/src" \
  .venv/bin/python services/cig-compiler-svc/train_hciv_encoder.py \
    --data data/processing/generator_artifacts/hciv_supervised_train_YYYYMMDD_<run_id>.jsonl \
    --output-checkpoint checkpoints/hciv_encoder_candidate_YYYYMMDD_<run_id>/hciv_encoder.pt \
    --manifest checkpoints/hciv_encoder_candidate_YYYYMMDD_<run_id>/hciv_encoder.manifest.json \
    --dim 128 \
    --curvature 1.0 \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --device cpu
```

Notes:

- `--device cpu` is conservative for the current workstation. Use GPU only after
  confirming resource availability and that it will not interfere with external
  `/workspace/SemMol` work.
- Do not train from demo `hash` or `random` targets.
- Do not change the `HCIVEncoder` architecture as part of this run.
- Do not inject HCIV coordinates into HFM `humu_embedding`; W2 steering rules
  still apply.

## Required Post-Training Checks

Checkpoint load check:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:services/cig-compiler-svc/src" \
  .venv/bin/python - <<'PY'
from pathlib import Path

import torch

from cig_compiler_svc.domain.hciv_encoder import load_hciv_encoder_checkpoint

checkpoint = Path(
    "checkpoints/hciv_encoder_candidate_YYYYMMDD_<run_id>/hciv_encoder.pt"
)
payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
encoder = load_hciv_encoder_checkpoint(str(checkpoint), dim=128, curvature=1.0)

assert payload["schema"] == "moleculeforge.cig_compiler.hciv_encoder.v1"
assert payload["dim"] == 128
assert payload["curvature"] == 1.0
assert encoder.dim == 128
assert encoder.curvature == 1.0
print("checkpoint_load_pass", checkpoint)
PY
```

CIG compiler production learned smoke:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:services/cig-compiler-svc/src" \
HCIV_CHECKPOINT_PATH="checkpoints/hciv_encoder_candidate_YYYYMMDD_<run_id>/hciv_encoder.pt" \
  .venv/bin/python - <<'PY'
import asyncio
import json
from pathlib import Path

from cig_compiler_svc.domain.compiler import CIGCompiler

async def main() -> None:
    compiler = CIGCompiler(
        hciv_dim=128,
        semantic_parser=lambda _: {
            "properties": [
                {"name": "qed", "direction": "maximize"},
                {"name": "logp", "direction": "minimize"},
            ]
        },
        enable_grounding=False,
    )
    cig, hciv, cone = await compiler.compile(
        "maximize QED while minimizing logP",
        seed=42,
    )
    assert len(hciv.coordinates) == 129
    assert len(cone.axis) == 129
    output = {
        "intent_id": cig.intent_id,
        "hciv_dim": len(hciv.coordinates),
        "cone_axis_dim": len(cone.axis),
        "curvature": hciv.curvature,
    }
    Path(
        "checkpoints/hciv_encoder_candidate_YYYYMMDD_<run_id>/"
        "compiler_smoke.json"
    ).write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, sort_keys=True))

asyncio.run(main())
PY
```

Deployment smoke after explicit approval:

- set `HCIV_CHECKPOINT_PATH` to the candidate path in a controlled service
  environment;
- compile a representative NL intent through the CIG compiler service;
- verify response HCIV and intent cone are 129-dimensional;
- verify downstream generation receives intent metadata without turning a plain
  HCIV payload into HFM `humu_embedding`.

## Promotion Decision Data

Record these before any W10 promotion decision:

- exact source path and hash;
- exact training command and environment, excluding secrets;
- wall-clock training time;
- manifest values, especially `example_count`, `epochs`, and `final_loss`;
- source preflight summary;
- checkpoint load check output;
- CIG compiler production learned smoke output;
- downstream intent-conditioned generation evidence;
- cluster readiness evidence with real service config and request/response logs.

## Stop Conditions

Stop and ask before:

- launching the training command above;
- lowering source-data requirements;
- changing epochs, batch size, learning rate, output directory, or device for a
  long run;
- changing Docker/Kubernetes/Helm CIG defaults;
- choosing a permanent production checkpoint path;
- editing Owner B implementation files;
- changing W2 HFM steering rules;
- modifying HUMU pretraining or HFM Lorentz flow architecture;
- overwriting protected checkpoints.

## Back-Check

- This plan does not start training.
- This plan does not modify code.
- This plan keeps hash/random HCIV encoders local-demo only.
- This plan writes future candidates only to a new non-protected directory.
