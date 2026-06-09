# 2026-06-04 Owner A Stage-Gate Re-Acceptance

## Scope

This is a read-only acceptance pass for Owner A generation-upstream work after the user paused implementation.

Reviewed Owner A gates:

- W2 pocket / intent HUMU embedding producer.
- W6 TAR ProxylessNAS runner.
- W8-E JMCG engineering skeleton.

Owner B progress was read only through the task split and interface acceptance documents. No Owner B code was modified.

## Verification Commands

Passed:

- `python -m py_compile moleculeforge/services/orchestrator-svc/src/orchestrator_svc/main.py moleculeforge/services/generator-router-svc/src/generator_router_svc/tar_proxyless_runner.py moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/jmcg_sampler.py moleculeforge/models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/inference/__init__.py moleculeforge/tests/unit/test_service_artifact_status.py moleculeforge/tests/unit/test_task_router.py moleculeforge/tests/unit/test_generators.py`
- `git diff --check`
- `uv run python -m generator_router_svc.tar_proxyless_runner` with a two-round KRAS reward payload.
- `uv run python - <<'PY' ... JMCGEngineeringSampler normal input smoke ... PY`

Not run:

- No pytest was run during this re-acceptance pass.

## Findings

### Important: W2 accepts mathematically invalid 129-dimensional embeddings

`orchestrator_svc.main._valid_hfm_feedback_embedding()` currently checks only:

- value is a list,
- length is 129,
- each item can be converted with `float()`.

It does not reject:

- all-zero 129-dimensional vectors,
- `NaN`,
- `Inf`,
- vectors that are not valid Lorentz hyperboloid full coordinates.

Reproduction result:

```text
all_zero_129 True
nan_129 True
inf_129 True
dim_128 False
```

Impact:

- W2 correctly rejects 128-dimensional HCIV as steering feedback.
- But a bad 129-dimensional `intent_cone.axis` can still become `humu_embedding`.
- This weakens the C1/C3 claim that steering-capable records are valid HUMU / Lorentz embeddings.

Acceptance status:

- W2 is directionally correct and passed earlier focused pytest.
- W2 should not be treated as fully hardened until finite + Lorentz compatibility validation is added.

### Important: W8-E accepts invalid 129-dimensional embeddings for alignment scoring

`JMCGEngineeringSampler._embedding_from_mapping()` only checks embedding length against `embedding_dim`.

Reproduction result:

```text
all_zero_129 {'pair_count': 1, 'ignored': 0, 'score': 1.0}
nan_129 {'pair_count': 1, 'ignored': 0, 'score': nan}
inf_129 {'pair_count': 1, 'ignored': 0, 'score': 1.0}
dim_128 {'pair_count': 0, 'ignored': 2, 'score': 0.0}
```

Impact:

- Existing 128-dimensional invalid-dimension behavior is covered.
- But invalid 129-dimensional inputs are treated as steering/alignment-capable.
- `nan` can propagate into `joint_score`.

Acceptance status:

- W8-E is a valid engineering skeleton and does not change default HFM generation behavior.
- W8-E needs the same stricter embedding validation before it can be considered robust.

### Moderate: W8-E silently ignores packed float32 molecule HUMU bytes

`Molecule.humu_embedding` is bytes, but `jmcg_sampler._embedding_from_bytes()` currently expects UTF-8 JSON bytes. Common HUMU service output uses packed float32 bytes.

Reproduction result:

```text
{'pair_count': 0, 'ignored': 0, 'score': 0.0}
```

Impact:

- Packed 129-float molecule embeddings are silently dropped.
- The drop is not counted in `ignored_embedding_count`.
- This can hide why alignment scoring did not happen.

Acceptance status:

- Not a blocker for W8-E's contract skeleton.
- Should be fixed or explicitly documented before relying on `Molecule.humu_embedding` bytes in downstream alignment.

### Passed: W6 TAR runner is locally acceptable

The W6 runner:

- reuses `ProxylessSearchScheduler`,
- consumes the existing stdin JSON payload shape,
- returns service-compatible architecture probabilities/logits/rounds,
- has command-level smoke evidence.

Acceptance status:

- W6 local code gate is acceptable.
- Remaining gates are production reward data, production env/deployment, focused pytest under W4.

## Back-Check

- The stage review did not modify business code.
- The review did not modify Owner B code.
- HUMU pretraining, HUMU encoder architecture, HFM architecture, and checkpoints remain untouched.
- `/workspace/SemMol` and `/workspace/Projects` were not modified or executed.
- The next scientifically reasonable step is a small hardening gate for shared 129-dimensional embedding validation, then focused tests with user authorization.

## Follow-Up Resolution

The hardening gate was completed on 2026-06-04 and recorded in `moleculeforge/docs/todo/owner-a-generation-upstream/2026-06-04-embedding-validation-hardening.md`.

Resolved:

- W2 now rejects invalid 129-dimensional intent/pocket embeddings via shared Lorentz validation.
- HFM feedback consumer now drops invalid 129-dimensional records.
- W8-E now rejects invalid 129-dimensional alignment embeddings.
- W8-E now decodes packed little-endian float32 `Molecule.humu_embedding` bytes.

Verification:

- New focused tests failed RED first, then passed after implementation.
- `uv run pytest tests/unit/test_generators.py tests/unit/test_service_artifact_status.py -q` passed 273 items with exit code 0 and one existing LangGraph deprecation warning.
