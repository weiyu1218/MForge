# MoleculeForge

End-to-end molecular property prediction & inverse-design platform with a
natural-language workbench, multi-GPU inference, and a real persistence
layer (SQLite). Tested on **4× NVIDIA H200** with CUDA-enabled PyTorch.

## Is it end-to-end runnable?

**Yes.** A single `uvicorn` process boots the API gateway, the reasoning
pipeline, the SQLite database, and the static frontend. From the browser
you can paste a free-form intent (中英文) and watch the reasoning chain
run live, with new candidate molecules drawn on the right and known drugs
correctly flagged.

The 148 unit + e2e tests pass without Docker / external services.

```
NL intent ─▶ nl_parse ─▶ objectives ─▶ generation ─▶ scoring (4× H200)
        ─▶ constraint_filter ─▶ novelty ─▶ ranking ─▶ summary ─▶ DB + UI
```

---

## Quick start

> Prerequisite: CUDA driver visible, `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
> The repository ships with `/workspace/MForge/moleculeforge/.venv` already
> populated. If you cloned into a different host, run **Step 1** to
> reproduce it.

### Step 1 — install dependencies (only the first time)

```bash
cd /workspace/MForge/moleculeforge

# Install every workspace package + dev extras into ./.venv
uv sync --all-extras --all-packages

# The system-installed CUDA torch is reused via system-site-packages,
# so we pin numpy<2 to keep it ABI-compatible.
sudo .venv/bin/pip install --force-reinstall --no-deps "numpy<2"
```

### Step 2 — sanity-check GPUs

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
# Expected: CUDA: True devices: 4
```

### Step 3 — start the full stack (API + reasoning pipeline + SQLite + UI)

```bash
cd /workspace/MForge/moleculeforge
.venv/bin/python -m uvicorn api_gateway.main:app --host 0.0.0.0 --port 8000
```

That single process gives you:

| URL                                | What it serves                        |
|-----------------------------------|---------------------------------------|
| `http://<host>:8000/`             | Reasoning workbench (frontend)        |
| `http://<host>:8000/docs`         | OpenAPI / Swagger                      |
| `http://<host>:8000/health`       | `{status, gpu, devices}`              |
| `http://<host>:8000/v1/predict`   | Single-molecule prediction             |
| `http://<host>:8000/v1/reason/*`  | NL → reasoning → results               |

To run in background:

```bash
nohup .venv/bin/python -m uvicorn api_gateway.main:app \
    --host 0.0.0.0 --port 8000 > /tmp/mf-api.log 2>&1 &
disown
until curl -s -m 2 http://localhost:8000/health >/dev/null; do sleep 1; done
echo "API up"
```

To stop:

```bash
pkill -f 'uvicorn api_gateway'
```

### Step 4 — open the workbench

Browse to **`http://<server-ip>:8000/`**.

Try one of the example chips, or paste your own:

- `Design 24 KRAS G12C covalent inhibitors with MW < 500, LogP 1-4, with a Michael acceptor warhead, prioritise drug-likeness.`
- `Lead optimise aspirin and ibuprofen for COX-2, generate 24 candidates, MW < 400.` *(returns both novel and known molecules)*
- `帮我设计 12 个针对 PARP 的抗肿瘤分子，分子量 250-450，含碳酰胺基。`

Click **▷ Run reasoning** to watch the live trace; click any molecule
card to open the detail drawer (full descriptors + ADMET).

---

## Command-line E2E test (no browser required)

```bash
# Submit a run
RID=$(curl -s -X POST http://localhost:8000/v1/reason/runs \
   -H 'Content-Type: application/json' \
   -d '{"intent":"Lead optimise aspirin and ibuprofen for COX-2, generate 24 candidates."}' \
   | python3 -c 'import json,sys;print(json.load(sys.stdin)["run_id"])')
echo "run_id=$RID"

# Wait for completion
until [ "$(curl -s http://localhost:8000/v1/reason/runs/$RID \
   | python3 -c 'import sys,json;print(json.load(sys.stdin)["status"])')" = "completed" ]; do sleep 2; done

# Inspect
curl -s http://localhost:8000/v1/reason/runs/$RID | python3 -m json.tool | head -40
```

Expected output: 30–50 candidates, ~2 known (Aspirin / Ibuprofen),
the rest novel. Devices used: `cuda:0..3`.

---

## Run the test suite

```bash
cd /workspace/MForge/moleculeforge

# Full unit + e2e (~7 min on 4× H200)
.venv/bin/python -m pytest tests/unit tests/e2e -p no:warnings

# Just the new reasoning workbench tests (~5 min)
.venv/bin/python -m pytest tests/e2e/test_reason_workbench.py -p no:warnings -v
```

Expected: **148 passed, 11 skipped** (the 11 are placeholder docker-only
tests in `tests/e2e/test_kras_g12c_pilot.py` that require external
services — they are intentionally skipped, not failures).

---

## Architecture

```
/workspace/MForge/moleculeforge/
├── ui/public/             SPA (HTML + CSS + ES module, no build step)
├── services/api-gateway/  FastAPI app + reason / molecules / design routers
├── agents/
│   ├── nl2obj/            NL → objectives parser (regex + heuristics, zh+en)
│   └── orchestrator/      Reasoning pipeline (8 stages, SSE, persistence)
├── libs/
│   ├── mf-chem/           MolPredictEngine (RDKit + HUMU + ADMET head)
│   ├── mf-humu/           Lorentz manifold, intent cone, encoders
│   └── mf-core/           Plugin ABCs, types, db.store (SQLite)
├── models/
│   ├── mf-generators/     9 generator plugins (HFM-3D, RDKit-Random, …)
│   └── mf-oracles/        ADMET, Boltz2, GNINA, OpenFE, RDKit oracle
├── data/
│   └── moleculeforge.db   SQLite — auto-created on first boot, seeded
│                           with 81 known drug molecules (DrugBank tagged)
└── tests/                 unit + e2e (real, no mocks)
```

### Multi-GPU strategy

`MolPredictEngine` (in `libs/mf-chem/src/mf_chem/predict/engine.py`) is a
process-wide singleton that initialises one HUMU encoder + one ADMET head
**per visible CUDA device** at startup. Inference batches are
round-robined across them, so a `predict_batch()` of N molecules saturates
all 4 H200s without any extra orchestration.

### Reasoning pipeline stages

`agents/orchestrator/src/orchestrator/pipeline.py` runs every NL request
through 8 stages, each persisted as a row in `reasoning_steps` and pushed
live over SSE:

1. `nl_parse` — tokens, targets, indications, task class
2. `objectives` — compiled CIG (constraints + priorities + n_samples)
3. `generation` — RDKit-Random over scaffold + warhead-aware templates
4. `scoring` — RDKit physicochemistry + HUMU embedding + ADMET head (4 GPUs)
5. `constraint_filter` — numeric ranges + must-include / must-exclude SMARTS
6. `novelty` — InChIKey lookup against the 81-drug catalog
7. `ranking` — weighted utility + Pareto front on (QED, –SA, –|logP – 2.5|)
8. `summary` — DB write + final SSE event

---

## Troubleshooting

| Symptom | Fix |
|--|--|
| `cannot open shared object file: libtorch_global_deps.so` | Re-run **Step 1**; the wrong torch wheel may have leaked in. The venv must reuse the system CUDA torch via `system-site-packages`. |
| `numpy 2.x ABI` warning at import | `sudo .venv/bin/pip install --force-reinstall --no-deps "numpy<2"` |
| Port 8000 already in use | `pkill -f 'uvicorn api_gateway'` then re-launch |
| Frontend blank / no devices | Hit `/health` directly. If `device_count: 0`, check `nvidia-smi` and CUDA driver. |
| Empty results panel | Your intent's constraints are too strict — relax MW / LogP, or remove the SMARTS rule. The reasoning chain still records *why* every candidate was rejected (see step 5 detail). |

---

## License & data

Code is proprietary (see `LICENSE`). The seeded known-molecule catalog
references DrugBank IDs as labels only; no protected DrugBank data is
redistributed. Datasets in `/workspace/MForge/zzzzz/` (CrossDocked, PDBBind,
USPTO-MIT, RetroPath, ChEMBL) are stored locally and not included in
shipping artifacts.
