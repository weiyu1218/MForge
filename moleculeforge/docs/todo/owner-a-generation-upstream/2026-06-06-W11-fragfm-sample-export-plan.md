# W11 FragFM Sample Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a focused FragFM sample export/report tool that writes generated SMILES for downstream MOSES/benchmark preparation without changing benchmark thresholds.

**Architecture:** Implement a small `mf_generators.fragfm.sample_export` module that instantiates `FragFMGenerator`, generates a requested number of molecules, writes one SMILES per line, and writes a JSON report with validity/uniqueness and artifact paths. Keep this separate from benchmark assertions so W5 thresholds stay unchanged.

**Tech Stack:** Python, pytest, RDKit, FragFMGenerator, JSON/text artifacts.

---

### Task 1: Sample Export Module

**Files:**
- Create: `models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py`
- Modify: `tests/unit/test_generators.py`

- [x] **Step 1: Write failing unit test**

Add this test to `TestFragFMGenerator` near the existing FragFM quality tests:

```python
def test_fragfm_sample_export_writes_smiles_and_report(self, tmp_path) -> None:
    from mf_generators.fragfm.sample_export import export_fragfm_samples

    vocab_path = tmp_path / "fragfm_vocab.json"
    vocab_path.write_text(
        json.dumps(
            {
                "fragments": ["CC", "O", "N"],
                "assembly_rules": [
                    {
                        "id": "ethanol",
                        "fragments": ["CC", "O"],
                        "product": "CCO",
                        "sa_score_bin": 2,
                    },
                    {
                        "id": "ethylamine",
                        "fragments": ["CC", "N"],
                        "product": "CCN",
                        "sa_score_bin": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "fragfm_generated.smi"
    report_path = tmp_path / "fragfm_generated.report.json"

    report = export_fragfm_samples(
        vocab_path=vocab_path,
        output_path=output_path,
        report_path=report_path,
        sample_count=3,
    )

    smiles = output_path.read_text(encoding="utf-8").splitlines()
    assert len(smiles) == 3
    assert all(Chem.MolFromSmiles(smile) is not None for smile in smiles)
    assert report["schema_version"] == "fragfm_sample_export_report.v1"
    assert report["requested_samples"] == 3
    assert report["generated_samples"] == 3
    assert report["valid_smiles"] == 3
    assert report["validity"] == pytest.approx(1.0)
    assert report["unique_smiles"] == 2
    assert report["output_path"] == str(output_path)
    written_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert written_report == report
```

- [x] **Step 2: Run RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_writes_smiles_and_report -q
```

Expected: FAIL because `mf_generators.fragfm.sample_export` does not exist.

Observed: FAIL with `ModuleNotFoundError: No module named
'mf_generators.fragfm.sample_export'`.

- [x] **Step 3: Implement minimal module**

Create `sample_export.py` with:

```python
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from rdkit import Chem

from mf_generators.fragfm.generator import FragFMGenerator


def export_fragfm_samples(
    *,
    vocab_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    sample_count: int = 100,
    checkpoint_path: str | Path | None = None,
    rate_matrix_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    generator = FragFMGenerator(
        vocab_path=str(vocab_path),
        checkpoint_path=str(checkpoint_path or ""),
        rate_matrix_path=str(rate_matrix_path or ""),
        device=device,
    )
    molecules = asyncio.run(generator.generate(batch_size=sample_count))
    smiles = [str(molecule.smiles) for molecule in molecules if molecule.smiles]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(smiles) + ("\n" if smiles else ""), encoding="utf-8")
    report = build_sample_report(
        smiles=smiles,
        requested_samples=sample_count,
        output_path=output,
        vocab_path=vocab_path,
        checkpoint_path=checkpoint_path,
        rate_matrix_path=rate_matrix_path,
    )
    if report_path is not None:
        report_output = Path(report_path)
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
```

Also include `build_sample_report()` and `main()` CLI.

- [x] **Step 4: Run GREEN**

Run the RED command again. Expected: PASS.

Observed: PASS with 1 item and the existing disabled-plugin `asyncio_mode`
warning.

### Task 2: CLI Smoke And Docs

**Files:**
- Modify: `docs/todo/owner-a-generation-upstream/README.md`
- Modify: `docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`
- Modify: `docs/todo/owner-a-generation-upstream/progress.md`

- [x] **Step 1: Run small CLI smoke on local test artifact**

Run a temporary small-vocab CLI smoke inside pytest or `/tmp`, not protected checkpoint paths.

Observed: exit code 0 with a `/tmp` vocab, 3 generated samples, validity 1.0,
and uniqueness 0.6666666666666666.

- [x] **Step 2: Run 5k candidate sample export smoke**

Run:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python -m mf_generators.fragfm.sample_export \
    --vocab checkpoints/fragfm_humu_5k/vocab.json \
    --checkpoint checkpoints/fragfm_humu_5k/best_model.pt \
    --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt \
    --output data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.smi \
    --report data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.report.json \
    --samples 8 \
    --device cpu
```

Expected: exit 0, report has `generated_samples=8` and `validity=1.0`.

Observed: exit code 0. Report
`data/processing/generator_artifacts/fragfm_humu_5k_sample_smoke.report.json`
has 8 generated samples, 8 valid SMILES, validity 1.0, 8 unique SMILES, and
uniqueness 1.0.

- [x] **Step 3: Update docs**

Record that this is benchmark preparation only. It does not satisfy official
MOSES/GuacaMol/PMO acceptance and does not relax thresholds.

### Task 3: Verification

- [x] **Step 1: Focused pytest**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/unit/test_generators.py::TestFragFMGenerator::test_fragfm_sample_export_writes_smiles_and_report -q
```

Observed: exit code 0. The test passed with 1 item. Warning is the existing
disabled-plugin `asyncio_mode` warning.

- [x] **Step 2: Static checks**

Run:

```bash
python3 -m py_compile \
  models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py \
  tests/unit/test_generators.py

git diff --check -- \
  moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py \
  moleculeforge/tests/unit/test_generators.py \
  moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-sample-export-plan.md \
  moleculeforge/docs/todo/owner-a-generation-upstream/README.md \
  moleculeforge/docs/todo/owner-a-generation-upstream/progress.md \
  moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md
```

Observed:

- `python3 -m py_compile models/mf-generators/fragfm/src/mf_generators/fragfm/sample_export.py tests/unit/test_generators.py`
  exited 0.
- Final tracked-file `git diff --check`, manual trailing whitespace scan,
  reserved-marker scan, and process scan are recorded in `progress.md`.

- [x] **Step 3: Back-check**

Confirm no protected checkpoints were changed, no benchmark thresholds were
relaxed, no HUMU pretraining files were modified, and no Owner B implementation
files were touched.

Observed: the final back-check is recorded in `progress.md`.
