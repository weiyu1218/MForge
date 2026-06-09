# W10 Enc_intent Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a concrete supervised training/export path that writes `HCIVEncoder` checkpoints suitable for `HCIV_CHECKPOINT_PATH`.

**Architecture:** Reuse the existing `HCIVEncoder` and production checkpoint loader. Add a small training helper under `cig_compiler_svc.domain` and a thin service-local CLI wrapper. Training data must include explicit target HCIV coordinates; hash/random demo encoders remain out of production scope.

**Tech Stack:** Python, PyTorch, Pydantic CIG models, existing `CIGCompiler`.

---

## File Structure

- Create `services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_training.py`
  - Owns supervised dataset loading, target validation and checkpoint export.
- Create `services/cig-compiler-svc/train_hciv_encoder.py`
  - Thin CLI wrapper around `train_hciv_encoder_checkpoint()`.
- Modify `tests/unit/test_cic_compiler.py`
  - Adds focused RED/GREEN coverage for dataset loading, tiny training/export and production compiler loading.
- Modify Owner A docs after verification.

## Task 1: Add Supervised Training Dataset Loader

**Files:**

- Create: `services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_training.py`
- Test: `tests/unit/test_cic_compiler.py`

- [x] **Step 1: Write failing test for JSONL records**

Add a test under `TestCICCompiler`:

```python
def test_hciv_training_examples_load_cig_and_target(self, tmp_path) -> None:
    from cig_compiler_svc.domain.hciv_training import load_hciv_training_examples
    from mf_core.types.cig import ChemicalIntentGraph, ObjectiveNode, ObjectiveType

    data_path = tmp_path / "hciv_train.jsonl"
    cig = ChemicalIntentGraph(
        intent_id="cig-train-1",
        objective_nodes=[
            ObjectiveNode(
                id="obj_qed",
                type=ObjectiveType.CONTINUOUS_MAXIMIZE,
                oracle="rdkit",
                weight=1.0,
            )
        ],
        source_user_input="maximize QED",
    )
    data_path.write_text(
        json.dumps(
            {
                "id": "example-1",
                "cig": cig.model_dump(mode="json", by_alias=True),
                "target_hciv": [1.0] + [0.0] * 8,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_hciv_training_examples(data_path, dim=8)

    assert len(examples) == 1
    assert examples[0].example_id == "example-1"
    assert examples[0].cig.intent_id == "cig-train-1"
    assert examples[0].target_coordinates.shape == (9,)
```

- [x] **Step 2: Run RED**

Run:

```bash
uv run pytest tests/unit/test_cic_compiler.py::TestCICCompiler::test_hciv_training_examples_load_cig_and_target -q
```

Expected: FAIL because `cig_compiler_svc.domain.hciv_training` does not exist.

- [x] **Step 3: Implement minimal loader**

Create `hciv_training.py` with:

```python
@dataclass(frozen=True)
class HCIVTrainingExample:
    example_id: str
    cig: ChemicalIntentGraph
    target_coordinates: torch.Tensor
    weight: float = 1.0

def load_hciv_training_examples(path: str | Path, dim: int = 128) -> list[HCIVTrainingExample]:
    records = _load_json_or_jsonl(path)
    examples = [_example_from_record(index, record, dim=dim) for index, record in enumerate(records)]
    if not examples:
        raise ValueError("HCIV training data requires at least one example")
    return examples
```

Validation:

- `cig` must be a JSON object parseable as `ChemicalIntentGraph`.
- `target_hciv` may be either a list of `dim + 1` floats or a mapping with `coordinates`.
- coordinates must be finite and length `dim + 1`.
- `weight` defaults to `1.0` and must be positive.

- [x] **Step 4: Run GREEN**

Run the same test and expect PASS.

## Task 2: Add Training / Export Helper

**Files:**

- Modify: `services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_training.py`
- Test: `tests/unit/test_cic_compiler.py`

- [x] **Step 1: Write failing test for tiny checkpoint export**

Add:

```python
def test_train_hciv_encoder_checkpoint_writes_loadable_artifact(
    self,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cig_compiler_svc.domain.compiler import CIGCompiler
    from cig_compiler_svc.domain.hciv_training import train_hciv_encoder_checkpoint
    from mf_core.types.cig import ChemicalIntentGraph, ObjectiveNode, ObjectiveType

    data_path = tmp_path / "hciv_train.jsonl"
    checkpoint_path = tmp_path / "hciv.pt"
    manifest_path = tmp_path / "hciv.manifest.json"
    cig = ChemicalIntentGraph(
        intent_id="cig-train-1",
        objective_nodes=[
            ObjectiveNode(
                id="obj_qed",
                type=ObjectiveType.CONTINUOUS_MAXIMIZE,
                oracle="rdkit",
                weight=1.0,
            )
        ],
        source_user_input="maximize QED",
    )
    data_path.write_text(
        json.dumps(
            {
                "id": "example-1",
                "cig": cig.model_dump(mode="json", by_alias=True),
                "target_hciv": [1.0] + [0.0] * 8,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    train_hciv_encoder_checkpoint(
        data_path,
        checkpoint_path,
        manifest_path=manifest_path,
        dim=8,
        epochs=1,
        batch_size=1,
        device="cpu",
    )
    monkeypatch.setenv("HCIV_CHECKPOINT_PATH", str(checkpoint_path))
    compiler = CIGCompiler(
        hciv_dim=8,
        semantic_parser=lambda _: {"properties": [{"name": "qed", "direction": "maximize"}]},
        enable_grounding=False,
    )
    _, hciv, cone = _run(compiler.compile("maximize QED", seed=7))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert checkpoint_path.exists()
    assert manifest["schema"] == "moleculeforge.cig_compiler.hciv_encoder.v1"
    assert manifest["example_count"] == 1
    assert len(hciv.coordinates) == 9
    assert cone.apex == hciv
```

- [x] **Step 2: Run RED**

Run:

```bash
uv run pytest tests/unit/test_cic_compiler.py::TestCICCompiler::test_train_hciv_encoder_checkpoint_writes_loadable_artifact -q
```

Expected: FAIL because `train_hciv_encoder_checkpoint()` does not exist.

- [x] **Step 3: Implement minimal training helper**

Implement:

```python
def train_hciv_encoder_checkpoint(
    data_path: str | Path,
    output_checkpoint: str | Path,
    *,
    manifest_path: str | Path | None = None,
    dim: int = 128,
    curvature: float = 1.0,
    epochs: int = 5,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    learning_rate: float = 1e-3,
) -> dict[str, object]:
```

Use the existing `HCIVEncoder`; each forward pass calls `encoder.encode(example.cig)` and compares predicted full coordinates to the target coordinates with weighted MSE. Save:

```python
torch.save(
    {
        "schema": "moleculeforge.cig_compiler.hciv_encoder.v1",
        "state_dict": encoder.state_dict(),
        "dim": dim,
        "curvature": curvature,
    },
    output_checkpoint,
)
```

Write a JSON manifest when requested.

- [x] **Step 4: Run GREEN**

Run the same test and expect PASS.

## Task 3: Add CLI Wrapper

**Files:**

- Create: `services/cig-compiler-svc/train_hciv_encoder.py`
- Test: `tests/unit/test_cic_compiler.py`

- [x] **Step 1: Write failing CLI test**

Add a helper near `_run()`:

```python
def _load_hciv_train_module():
    script = ROOT / "services/cig-compiler-svc/train_hciv_encoder.py"
    spec = importlib.util.spec_from_file_location("cig_hciv_train", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
```

Add a test that writes a tiny JSONL dataset, calls:

```python
exit_code = module.main([
    "--data", str(data_path),
    "--output-checkpoint", str(checkpoint_path),
    "--manifest", str(manifest_path),
    "--dim", "8",
    "--epochs", "1",
    "--batch-size", "1",
    "--device", "cpu",
])
```

Assert exit code 0 and both files exist.

- [x] **Step 2: Run RED**

Run:

```bash
uv run pytest tests/unit/test_cic_compiler.py::TestCICCompiler::test_train_hciv_encoder_cli_writes_checkpoint_and_manifest -q
```

Expected: FAIL because `train_hciv_encoder.py` does not exist.

- [x] **Step 3: Implement CLI wrapper**

Create a thin argparse wrapper around `train_hciv_encoder_checkpoint()` with args:

- `--data`
- `--output-checkpoint`
- `--manifest`
- `--dim`
- `--curvature`
- `--epochs`
- `--batch-size`
- `--device`
- `--learning-rate`

- [x] **Step 4: Run GREEN**

Run the same test and expect PASS.

## Task 4: Verification And Docs

**Files:**

- Modify: `docs/todo/owner-a-generation-upstream/progress.md`
- Modify: `docs/architecture/current-implementation-vs-corearchitecture-v2.md`
- Modify: `docs/architecture/corearchitecture-v2-completion-tasksplit.md`
- Modify: `docs/architecture/corearchitecture-v2-completion-interface-acceptance.md`

- [x] **Step 1: Run focused W10 tests**

Run:

```bash
uv run pytest \
  tests/unit/test_cic_compiler.py::TestCICCompiler::test_hciv_training_examples_load_cig_and_target \
  tests/unit/test_cic_compiler.py::TestCICCompiler::test_train_hciv_encoder_checkpoint_writes_loadable_artifact \
  tests/unit/test_cic_compiler.py::TestCICCompiler::test_train_hciv_encoder_cli_writes_checkpoint_and_manifest \
  tests/unit/test_cic_compiler.py::TestCICCompiler::test_production_learned_loads_checkpoint \
  -q
```

Expected: PASS.

- [x] **Step 2: Run static checks**

Run:

```bash
python -m py_compile \
  moleculeforge/services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_training.py \
  moleculeforge/services/cig-compiler-svc/train_hciv_encoder.py \
  moleculeforge/tests/unit/test_cic_compiler.py
```

Expected: exit code 0.

- [x] **Step 3: Run CIC file-level regression**

Run:

```bash
uv run pytest tests/unit/test_cic_compiler.py -q
```

Expected: exit code 0.

- [x] **Step 4: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: exit code 0.

- [x] **Step 5: Update docs**

Record:

- W10 local train/export path exists.
- Tiny CPU training smoke passed.
- Checkpoint is loadable via existing production `HCIV_CHECKPOINT_PATH`.
- This is engineering readiness, not production checkpoint quality.
