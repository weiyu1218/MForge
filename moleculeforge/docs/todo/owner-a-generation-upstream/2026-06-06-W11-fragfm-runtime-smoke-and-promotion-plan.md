# W11 FragFM Runtime Smoke And Promotion Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Prove the current W11 FragFM HUMU 5k deployment artifact can be loaded through the service runtime path, and record the policy for promoting future FragFM artifacts.

**Architecture:** Keep this as a narrow Owner A/W11 increment. Add one runtime smoke regression around `fragfm_generator_svc._build_generator()` and `FragFMGenerator.generate()`, then minimally harden checkpoint loading if the smoke exposes a hidden-dimension mismatch. Record that `checkpoints/fragfm_humu_5k/` is a strict-local engineering candidate and define the separate promotion path for production artifacts.

**Tech Stack:** Python, pytest, PyTorch, RDKit, gRPC service module loading, Owner A markdown docs.

---

### Task 1: Runtime Smoke Regression

**Files:**
- Modify: `tests/unit/test_service_artifact_status.py`
- Modify if RED exposes a runtime gap: `models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py`

- [x] **Step 1: Write the failing test**

Add a focused test next to `test_fragfm_deployment_wires_artifact_and_sampler_env`:

```python
@pytest.mark.asyncio
async def test_fragfm_deployment_default_artifact_loads_and_generates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "fragfm_service_humu_5k_runtime_smoke_test",
        ROOT / "services/fragfm-generator-svc/src/fragfm_generator_svc/main.py",
    )
    monkeypatch.setenv(
        "FRAGFM_VOCAB_PATH",
        str(ROOT / "checkpoints/fragfm_humu_5k/vocab.json"),
    )
    monkeypatch.setenv(
        "FRAGFM_CHECKPOINT_PATH",
        str(ROOT / "checkpoints/fragfm_humu_5k/best_model.pt"),
    )
    monkeypatch.setenv(
        "FRAGFM_RATE_MATRIX_PATH",
        str(ROOT / "checkpoints/fragfm_humu_5k/rate_matrix.pt"),
    )
    monkeypatch.setenv("FRAGFM_HUMU_CURVATURE", "1.0")

    generator = module._build_generator()
    molecules = await generator.generate(batch_size=1)

    assert len(molecules) == 1
    assert molecules[0].smiles
    assert molecules[0].metadata["generator_name"] == "fragfm"
    assert molecules[0].metadata["fragment_vocabulary"].endswith(
        "checkpoints/fragfm_humu_5k/vocab.json"
    )
    assert generator._model is not None
```

- [x] **Step 2: Run RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/unit/test_service_artifact_status.py::test_fragfm_deployment_default_artifact_loads_and_generates -q
```

Observed before the fix: the service runtime smoke did not complete in a
reasonable time under current local load. Follow-up probes showed slow PyTorch
import / artifact loading plus per-rule transition scoring needed runtime
hardening before this could serve as a useful regression.

- [x] **Step 3: Minimal implementation**

If RED fails on a checkpoint hidden-dimension mismatch, infer the `TwoLevelDFM` hidden dimension from `fragment_encoder.weight.shape[1]` before constructing the model:

```python
    def _load_model_from_checkpoint(self, checkpoint_path: str, device: str):
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        hidden_dim = self._checkpoint_hidden_dim(state)
        model = TwoLevelDFM(vocab_size=len(self.vocab), hidden_dim=hidden_dim)
        model.load_state_dict(state, strict=False)
        model.to(device)
        return model

    def _checkpoint_hidden_dim(self, state: Mapping[str, object]) -> int:
        weight = state.get("fragment_encoder.weight")
        if weight is None or not hasattr(weight, "shape"):
            return 256
        shape = tuple(int(dim) for dim in weight.shape)
        if len(shape) != 2 or shape[1] <= 0:
            return 256
        return shape[1]
```

Then call `_load_model_from_checkpoint()` from `__init__`. The implemented fix
also caches scored rules and computes sparse transition scores in a batch.

- [x] **Step 4: Run GREEN**

Run the same focused pytest command. Expected: PASS.

### Task 2: Promotion Policy Docs

**Files:**
- Create: `docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-artifact-promotion-policy.md`
- Modify: `docs/todo/owner-a-generation-upstream/README.md`
- Modify: `docs/todo/owner-a-generation-upstream/progress.md`
- Modify if needed: `docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md`

- [x] **Step 1: Write the policy**

Create a policy document that states:

```markdown
# W11 FragFM Artifact Promotion Policy

Date: 2026-06-06
Scope: Owner A, W11 FragFM shared HUMU conditional-space artifacts

## Current Candidate

`checkpoints/fragfm_humu_5k/` is a strict-local engineering candidate. It can be
used for local service smoke and deployment-default hardening, but it is not
final production acceptance.

## Protected Paths

Do not overwrite:

- `checkpoints/fragfm`
- `checkpoints/humu`
- `checkpoints/hfm3d_4h200`

## Promotion Rule

Future production FragFM artifacts must be written to a new explicit directory,
for example `checkpoints/fragfm_humu_production/` or a dated immutable artifact
directory. Deployment defaults must not be moved to that path until all
promotion gates below have recorded evidence.

## Promotion Gates

- HUMU coverage gate: strict quality report status `pass`, 129-dimensional HUMU
  coverage at the declared threshold, and zero invalid HUMU embeddings.
- Runtime gate: service `_build_generator()` loads vocab, checkpoint, and rate
  matrix, then generates at least one valid molecule with the configured paths.
- Training gate: manifest records data source, record count, fragment count,
  epochs, hidden dimension, optimizer choices, and HUMU coverage.
- Benchmark gate: GuacaMol, PMO, and MOSES thresholds are not relaxed.
- Deployment gate: Docker Compose, raw Kubernetes, Helm, and cluster smoke
  evidence point to the promoted artifact.
```

- [x] **Step 2: Update Owner A entry docs**

Add the policy link and the observed runtime-smoke result to the README and handoff current W11 sections.

- [x] **Step 3: Update progress**

Append a dated progress entry with files changed, verification commands, and back-check notes.

### Task 3: Verification

**Files:**
- Test: `tests/unit/test_service_artifact_status.py`
- Test: `models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py`
- Test: touched docs

- [x] **Step 1: Focused pytest**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/unit/test_service_artifact_status.py::test_fragfm_deployment_default_artifact_loads_and_generates \
  tests/unit/test_service_artifact_status.py::test_fragfm_deployment_wires_artifact_and_sampler_env -q
```

- [x] **Step 2: Existing FragFM generator focused pytest**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/unit/test_generators.py::TestFragFMGenerator::test_loads_checkpoint_and_rate_matrix_artifacts -q
```

- [x] **Step 3: Strict W11 quality CLI**

Run:

```bash
PYTHONPATH="libs/mf-core/src:libs/mf-humu/src:libs/mf-chem/src:models/mf-generators/fragfm/src" \
  .venv/bin/python -m mf_generators.fragfm.quality \
    --vocab checkpoints/fragfm_humu_5k/vocab.json \
    --checkpoint checkpoints/fragfm_humu_5k/best_model.pt \
    --rate-matrix checkpoints/fragfm_humu_5k/rate_matrix.pt \
    --min-humu-coverage 1.0 \
    --strict \
    --output checkpoints/fragfm_humu_5k/quality_report.json
```

- [x] **Step 4: Static checks**

Run:

```bash
python3 -m py_compile \
  tests/unit/test_service_artifact_status.py \
  models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py

git diff --check -- \
  moleculeforge/models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py \
  moleculeforge/tests/unit/test_service_artifact_status.py \
  moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-runtime-smoke-and-promotion-plan.md \
  moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-W11-fragfm-artifact-promotion-policy.md \
  moleculeforge/docs/todo/owner-a-generation-upstream/README.md \
  moleculeforge/docs/todo/owner-a-generation-upstream/progress.md \
  moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-06-new-session-handoff.md
```

- [x] **Step 5: Back-check**

Confirm:

- No protected checkpoint directory was overwritten.
- No HUMU pretraining code/config/checkpoint continuation was modified.
- No Owner B implementation file was modified.
- `checkpoints/fragfm_humu_5k/` remains documented as local engineering evidence, not production acceptance.
