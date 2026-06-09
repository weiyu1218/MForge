# MoleculeForge UI

Lightweight single-page dashboard for MoleculeForge. Served by the
`api-gateway` FastAPI app from `ui/public/` — no separate Node/React build step.

## Run

```bash
cd /workspace/MForge/moleculeforge
.venv/bin/python -m uvicorn api_gateway.main:app --host 0.0.0.0 --port 8000
```

Then browse http://localhost:8000/ . The dashboard auto-detects available
CUDA devices and surfaces them in the status chip.

## Tabs

| Tab             | Endpoint                               | Purpose                                                      |
|-----------------|----------------------------------------|--------------------------------------------------------------|
| Predict         | `POST /v1/predict`                     | Single-molecule RDKit + HUMU + ADMET scoring                  |
| Batch           | `POST /v1/molecules/batch`             | Multi-GPU parallel scoring of a SMILES list                   |
| Design Loop     | `POST /v1/design/`, `GET /v1/design/*` | Submit a design job, poll status, browse Pareto results       |
| Retrosynthesis  | `POST /v1/routes/plan`                 | SMARTS-based disconnection suggestions                        |
| Similarity      | `POST /v1/molecules/search`            | Morgan + Tanimoto similarity over local catalogue             |

## Static assets

- `public/index.html` — the SPA shell
- `public/styles.css` — theme
- `public/app.js` — client logic (vanilla ES)

To point the gateway at a different UI folder set `MF_UI_DIR` before launch:

```bash
MF_UI_DIR=/path/to/custom/ui .venv/bin/python -m uvicorn api_gateway.main:app
```

## Planned roadmap (Phase 2)

See `ARCHITECTURE.md` for the long-term Next.js + 3Dmol rework plan.
