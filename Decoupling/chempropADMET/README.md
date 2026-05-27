# Chemprop ADMET Microservice

Stateless inference service wrapping chemprop MPNN models for ADMET property prediction.

## Quick Start

### 1. Create conda environment

```bash
cd /workspace/MForge/Decoupling/chempropADMET
conda env create -f environment.yml
conda activate chempropADMET
```

Or manually:

```bash
conda create -n chempropADMET python=3.11 -y
conda activate chempropADMET
conda install -c pytorch pytorch -y
conda install -c conda-forge rdkit -y
pip install chemprop fastapi "uvicorn[standard]" pydantic httpx numpy
```

### 2. Place model checkpoints

Put trained chemprop checkpoints under `models/<endpoint>/model.ckpt`:

```
models/
├── solubility/
│   └── model.ckpt
├── lipophilicity/
│   └── model.ckpt
├── permeability/
│   └── model.ckpt
└── ...
```

### 3. Start the service

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8901
```

### 4. Test

```bash
bash test_service.sh
```

## API

### `GET /health`

Returns service status, available endpoints, and device.

### `POST /predict`

Request body:
```json
{
  "smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
  "endpoints": ["solubility", "lipophilicity"],
  "batch_size": 64
}
```

Response:
```json
{
  "results": [
    {"smiles": "CCO", "predictions": {"solubility": 0.52, "lipophilicity": -0.31}},
    {"smiles": "c1ccccc1", "predictions": {"solubility": -1.2, "lipophilicity": 2.1}},
    {"smiles": "CC(=O)O", "predictions": {"solubility": 0.88, "lipophilicity": -0.05}}
  ],
  "n_molecules": 3,
  "endpoints_used": ["solubility", "lipophilicity"]
}
```

## Client Usage

```python
from client import admet_predict

results = admet_predict(["CCO", "c1ccccc1"], endpoints=["solubility"])
```

## Integration with Multi-Agent System

Wrap as a LangChain tool (uncomment the block in `client.py`):

```python
from langchain_core.tools import tool

@tool
def admet_tool(smiles_csv: str) -> str:
    """Predict ADMET properties for molecules given comma-separated SMILES."""
    import json
    smiles = [s.strip() for s in smiles_csv.split(",") if s.strip()]
    results = admet_predict(smiles)
    return json.dumps(results, indent=2)
```

## Docker

```bash
docker build -t chemprop-admet .
docker run -p 8901:8901 -v $(pwd)/models:/app/models chemprop-admet
```

## Config

Edit `config.py` to change host, port, batch size, device, or model paths.
Set `DEVICE=cuda` if GPU is available.
