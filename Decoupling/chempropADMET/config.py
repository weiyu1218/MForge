"""Configuration for the ADMET inference service."""

from pathlib import Path

# Server
HOST = "0.0.0.0"
PORT = 8901

# Inference
DEFAULT_BATCH_SIZE = 64
MAX_BATCH_SIZE = 512
DEVICE = "auto"  # "auto" | "cpu" | "cuda"

# Model registry — local paths or HuggingFace-style IDs
# Users should populate MODEL_DIR with their checkpoint files.
MODEL_DIR = Path(__file__).parent / "models"

# Default ADMET endpoints to load. Each key maps to a checkpoint sub-directory.
ADMET_ENDPOINTS = {
    "solubility": MODEL_DIR / "solubility",
    "lipophilicity": MODEL_DIR / "lipophilicity",
    "permeability": MODEL_DIR / "permeability",
    "bbb": MODEL_DIR / "bbb",            # blood-brain barrier
    "hia": MODEL_DIR / "hia",            # human intestinal absorption
    "bioavailability": MODEL_DIR / "bioavailability",
    "cyp_inhibition": MODEL_DIR / "cyp_inhibition",
    "herg": MODEL_DIR / "herg",
    "ld50": MODEL_DIR / "ld50",
    "clearance": MODEL_DIR / "clearance",
}
