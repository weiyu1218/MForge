# W9 HFM Neural Geometry Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a concrete local HFM neural geometry decoder training/export/runner path that plugs into the existing `HFM_MOLECULAR_DECODER_COMMAND` contract.

**Architecture:** Keep HFM flow and HUMU frozen. Train a small geometry decoder from existing HFM decoder entries containing `latent` + SDF geometry; at inference, select the nearest decoder entry for SMILES/atom count and use the neural model to predict coordinates for that molecule. The runner returns the existing HFM molecular decoder JSON schema.

**Tech Stack:** Python, PyTorch, RDKit, existing `HFM3DGenerator` molecular decoder contract.

---

## File Structure

- Create `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/__init__.py`
  - Exports the neural geometry decoder API.
- Create `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/neural_geometry_decoder.py`
  - Model, artifact load/save, SDF parsing, nearest-entry selection, JSON command runner.
- Create `models/mf-generators/hfm_3d/train_geometry_decoder.py`
  - CLI for training a geometry decoder from an SDF-backed HFM decoder artifact.
- Modify `moleculeforge/tests/unit/test_generators.py`
  - Focused tests for artifact training helpers, runner JSON, and HFM generator consumption.
- Modify progress/architecture docs after implementation.

## Task 1: Add Decoder Module And Artifact Schema

**Files:**

- Create: `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/__init__.py`
- Create: `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/neural_geometry_decoder.py`
- Test: `tests/unit/test_generators.py`

- [x] **Step 1: Write failing test for loading SDF-backed decoder entries**

Add a test that creates two decoder entries with `latent`, `smiles`, and `sdf`, then calls:

```python
from mf_generators.hfm_3d.decoder.neural_geometry_decoder import load_geometry_training_examples

examples = load_geometry_training_examples(decoder_artifact)
assert examples[0].smiles == "CCO"
assert examples[0].latent.shape == (129,)
assert examples[0].atom_types == ["C", "C", "O"]
assert examples[0].coordinates.shape == (3, 3)
```

- [x] **Step 2: Run RED**

Run:

```bash
uv run pytest tests/unit/test_generators.py::TestHFM3DGenerator::test_neural_geometry_decoder_loads_sdf_training_examples -q
```

Expected: FAIL because the module does not exist.

- [x] **Step 3: Implement minimal loader**

Implement:

```python
@dataclass(frozen=True)
class GeometryTrainingExample:
    entry_id: str
    smiles: str
    latent: torch.Tensor
    atom_types: list[str]
    coordinates: torch.Tensor

def load_geometry_training_examples(decoder_artifact: str | Path) -> list[GeometryTrainingExample]:
    path = Path(decoder_artifact)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload["entries"]
    return [_geometry_example_from_decoder_entry(entry) for entry in entries]
```

Use RDKit to parse SDF, remove hydrogens for canonical atom alignment, and read conformer positions.

- [x] **Step 4: Run GREEN**

Run the same test and expect PASS.

## Task 2: Add Neural Model And Tiny Training Loop

**Files:**

- Modify: `decoder/neural_geometry_decoder.py`
- Create: `train_geometry_decoder.py`
- Test: `tests/unit/test_generators.py`

- [x] **Step 1: Write failing test for one-epoch CPU training**

The test should:

- create a tiny decoder artifact with SDF entries,
- call `train_geometry_decoder_artifact(decoder_artifact, output_artifact, epochs=1, batch_size=2, device="cpu")`,
- assert an output artifact exists,
- load it with `NeuralGeometryDecoderArtifact.load(output_artifact, map_location="cpu")`,
- assert it can predict coordinates for a 129-dimensional latent.

- [x] **Step 2: Run RED**

Run:

```bash
uv run pytest tests/unit/test_generators.py::TestHFM3DGenerator::test_neural_geometry_decoder_trains_tiny_artifact -q
```

Expected: FAIL because training/export helpers are missing.

- [x] **Step 3: Implement minimal model**

Use a bounded fixed-size model:

```python
class NeuralGeometryDecoder(torch.nn.Module):
    def __init__(self, latent_dim: int = 129, max_atoms: int = 64):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, max_atoms * 3),
        )
        self.max_atoms = max_atoms

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        coordinates = self.network(latent)
        return coordinates.reshape(latent.shape[0], self.max_atoms, 3)
```

Training target:

- pad coordinates to `max_atoms`,
- mask loss to true atom count,
- train with MSE.

Artifact should include:

- `model_state`,
- `latent_dim`,
- `max_atoms`,
- `entries` with `id`, `smiles`, `latent`, `atom_types`,
- `source_decoder_artifact`.

- [x] **Step 4: Run GREEN**

Run the tiny training test and expect PASS.

## Task 3: Add JSON Runner Compatible With HFM Command Contract

**Files:**

- Modify: `decoder/neural_geometry_decoder.py`
- Test: `tests/unit/test_generators.py`

- [x] **Step 1: Write failing test for stdin/stdout runner**

The test should run:

```bash
python -m mf_generators.hfm_3d.decoder.neural_geometry_decoder --artifact <artifact.pt>
```

stdin:

```json
{"latent": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
```

Expected stdout:

```json
{
  "smiles": "CCO",
  "atom_types": ["C", "C", "O"],
  "coordinates": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.1, 0.8, 0.0]],
  "metadata": {"decoder_mode": "neural_geometry_decoder"}
}
```

- [x] **Step 2: Run RED**

Run the single runner test and expect FAIL.

- [x] **Step 3: Implement runner**

Implement `main(argv=None) -> int`:

- parse `--artifact`,
- read stdin JSON,
- require `latent`,
- load artifact,
- select nearest entry by latent distance,
- predict coordinates,
- emit JSON.

- [x] **Step 4: Run GREEN**

Run the runner test and expect PASS.

## Task 4: Verify HFM Generator Consumes Runner-Compatible Output

**Files:**

- Modify: `tests/unit/test_generators.py`

- [x] **Step 1: Add integration-style unit test**

Use `HFM3DGenerator` with `molecular_decoder=<loaded neural decoder object or command adapter>` and assert:

- molecule SMILES is valid,
- `sdf_bytes` is present,
- metadata includes `decoder_mode=neural_geometry_decoder`.

- [x] **Step 2: Run focused tests**

Run:

```bash
uv run pytest tests/unit/test_generators.py::TestHFM3DGenerator::test_neural_geometry_decoder_loads_sdf_training_examples tests/unit/test_generators.py::TestHFM3DGenerator::test_neural_geometry_decoder_trains_tiny_artifact tests/unit/test_generators.py::TestHFM3DGenerator::test_neural_geometry_decoder_runner_outputs_hfm_contract tests/unit/test_generators.py::TestHFM3DGenerator::test_hfm_generator_consumes_neural_geometry_decoder_output -q
```

Expected: PASS.

## Task 5: Static Verification And Documentation

**Files:**

- Modify: `docs/todo/owner-a-generation-upstream/progress.md`
- Modify: `docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- Modify: `docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- Modify: `docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`

- [x] **Step 1: Run static checks**

```bash
python -m py_compile \
  moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/__init__.py \
  moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/decoder/neural_geometry_decoder.py \
  moleculeforge/models/mf-generators/hfm_3d/train_geometry_decoder.py \
  moleculeforge/tests/unit/test_generators.py
```

Expected: exit code 0.

- [x] **Step 2: Run file-level focused test**

```bash
uv run pytest tests/unit/test_generators.py -q
```

Expected: exit code 0.

- [x] **Step 3: Update docs**

Record:

- local neural geometry decoder command target exists,
- tiny CPU artifact training smoke passed,
- this is engineering readiness, not production quality,
- remaining gates are real data/artifact, production env, cluster validation, benchmark quality.

## Self-Review

- Spec coverage: Covers train/export/runner, HFM command contract, and local testability.
- Placeholder scan: No placeholder patterns are used as implementation instructions.
- Type consistency: The runner emits the existing molecular decoder JSON schema consumed by `HFM3DGenerator`.
