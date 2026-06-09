# CoreArchitecture v2 Owner A W4 Focused Validation Record

Date: 2026-06-05
Role: Owner A, generation-upstream

## Scope

This record captures the current W4 focused validation pass after the W9/W10/W11
hardening gate.

No business code was changed during this validation pass. Owner B code was read
for diagnosis only and was not modified. `/workspace/SemMol` and
`/workspace/Projects` were not modified or executed.

All commands were run from `moleculeforge/` with `.env` loaded and proxy env
unset:

```bash
set -a; source .env; set +a
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"
```

## Results

| Area | Command | Result | Classification |
|---|---|---:|---|
| W1 CRG unit gate | `uv run pytest tests/unit/test_graph_repo.py -q` | exit 1; 14 items, 11 passed, 3 failed | Owner B test/patch seam regression |
| W2 orchestrator feedback producer | 8 focused tests in `tests/unit/test_service_artifact_status.py` | exit 0; 8 passed | Pass |
| W2/W8 HFM JMCG consumer | 12 focused tests in `tests/unit/test_generators.py::TestHFM3DGenerator` | exit 0; 12 passed | Pass |
| C2 predicate/downstream agent regression | `uv run pytest tests/unit/test_validation_agent.py tests/unit/test_srb_agent.py -q` | exit 0; 32 passed | Pass |
| C1 generator coordinator regression | `uv run pytest tests/unit/test_generator_coord_agent.py -q` | exit 0; 20 passed | Pass |
| W3 mf-eval local provider gate | `uv run pytest tests/unit/test_mf_eval.py -q` | exit 0; 24 passed | Pass |
| W5 benchmark harness | `uv run pytest tests/benchmark -q` | exit 1; 18 items, 8 failed, 10 skipped | Expected production data/quality gate |
| W11 FragFM quality pytest | 6 focused W11 quality tests | exit 0; 6 passed | Pass |
| W11 FragFM quality CLI | `uv run python -m mf_generators.fragfm.quality --vocab checkpoints/fragfm/vocab.json --checkpoint checkpoints/fragfm/best_model.pt --rate-matrix checkpoints/fragfm/rate_matrix.pt --min-humu-coverage 0.0 --strict` | exit 0; `status=pass`, `rules=50`, `humu_embedding_coverage=0.0`, checkpoint/rate matrix loadable | Runtime smoke only |
| W13 KD pytest | `uv run pytest tests/unit/test_cross_paradigm_kd.py -q` | exit 0; 18 passed | Pass |
| W13 KD CLI | `uv run python -m mf_core.routing.kd_artifacts ... --expected-dim 2 --min-embeddings 2 --strict` | exit 0; report `status=pass`, canonical artifact `cross_paradigm_teacher_embeddings.v1 2 2` | Pass |
| W9/W10/W11 hardening regression | 4 focused hardening tests | exit 0; 4 passed | Pass |

## W1 Diagnosis

The W1 unit failure is reproducible and concentrated in three tests:

- `test_merge_agent_beliefs_merges_shared_crg_into_final_state`
- `test_merge_agent_beliefs_deduplicates_existing_beliefs`
- `test_merge_agent_beliefs_falls_through_when_no_repository`

All three fail before exercising merge behavior because the tests patch:

```text
orchestrator_svc.main.build_shared_crg_repository_from_env
```

but `orchestrator_svc.main` does not expose that symbol at module scope. The
implementation currently imports it inside `_merge_agent_beliefs_into_crg()`:

```python
from mf_core.db.repositories import build_shared_crg_repository_from_env
```

This should be handed to Owner B as a W1 unit-test compatibility issue. Two
reasonable Owner B fixes are possible:

- Patch `mf_core.db.repositories.build_shared_crg_repository_from_env` in the
  tests.
- Or intentionally expose a module-level import seam in `orchestrator_svc.main`.

Owner A did not change either path.

## W5 Diagnosis

The W5 benchmark failure is not a local implementation regression from this
stage. It reflects production-quality/data gates:

- GuacaMol rediscovery/median/MPO failures are using repeated local `CCO`
  generated samples and do not meet production thresholds.
- PMO LogP/QED/multi-objective failures are also based on the local `CCO`
  baseline (`LogP` around `0.249825`, `QED` around `0.4068`), below the configured
  thresholds.
- Skips are caused by missing official resource env:
  `CROSSDOCKED_BENCHMARK_JSONL`, `MOSES_REFERENCE_SMILES_PATH`,
  `FRAGFM_MOSES_GENERATED_SMILES_PATH`, and `PMO_SCORE_TABLE_PATH`.

W5 remains blocked on official benchmark data and production-quality generated
samples. Thresholds were not relaxed.

## Large Group Note

The broad W2 command:

```bash
uv run pytest tests/unit/test_generators.py tests/unit/test_service_artifact_status.py -q
```

collected 285 items but was terminated after roughly 14 minutes with exit code 143
because it advanced only a few test points. This was not recorded as pass or
failure. The validation switched to work-item focused gates listed above.

## Back-Check

- [x] No business code was changed during this validation pass.
- [x] Owner B implementation files were not modified.
- [x] `.env` was read/loaded only; no env file changes were made.
- [x] W1 failure was isolated to a single Owner B patch seam.
- [x] W5 failure was classified as production data/quality gate, not threshold
      relaxation work.
- [x] W11 local artifact remains explicitly runtime-smoke only because HUMU
      coverage is 0.0.
- [x] W13 local KD artifact utility remains an artifact handoff gate, not proof
      of production distillation quality.
