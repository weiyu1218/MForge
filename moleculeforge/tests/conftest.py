"""Pytest global config: auto-add all workspace package src dirs to sys.path."""
import atexit
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_TEST_DB_DIR: tempfile.TemporaryDirectory[str] | None = None
if not os.environ.get("MF_DB_PATH"):
    _TEST_DB_DIR = tempfile.TemporaryDirectory(prefix="moleculeforge-pytest-")
    atexit.register(_TEST_DB_DIR.cleanup)
    os.environ["MF_DB_PATH"] = str(Path(_TEST_DB_DIR.name) / "moleculeforge.db")

PACKAGE_SRC_DIRS = [
    ROOT / "libs" / "mf-core" / "src",
    ROOT / "libs" / "mf-humu" / "src",
    ROOT / "libs" / "mf-chem" / "src",
    ROOT / "libs" / "mf-agents" / "src",
    ROOT / "libs" / "mf-eval" / "src",
    ROOT / "libs" / "mf-telemetry" / "src",
    ROOT / "models" / "mf-encoders" / "humu_mol_encoder" / "src",
    ROOT / "models" / "mf-encoders" / "humu_pocket_encoder" / "src",
    ROOT / "models" / "mf-encoders" / "humu_route_encoder" / "src",
    ROOT / "models" / "mf-generators" / "hfm_3d" / "src",
    ROOT / "models" / "mf-generators" / "fragfm" / "src",
    ROOT / "models" / "mf-generators" / "crem_3d" / "src",
    ROOT / "models" / "mf-generators" / "mmpt_rag" / "src",
    ROOT / "models" / "mf-generators" / "incremental_clm" / "src",
    ROOT / "models" / "mf-generators" / "uas" / "src",
    ROOT / "models" / "mf-generators" / "rdkit_random" / "src",
    ROOT / "models" / "mf-oracles" / "rdkit-oracle" / "src",
    ROOT / "services" / "cig-compiler-svc" / "src",
    ROOT / "services" / "provenance-svc" / "src",
    ROOT / "agents" / "srb_agent" / "src",
    ROOT / "wetlab" / "xdl-compiler" / "src",
    ROOT / "pipelines" / "mvp_pipeline" / "src",
    ROOT / "pipelines" / "pareto_bo" / "src",
    ROOT / "agents" / "orchestrator" / "src",
]

for p in PACKAGE_SRC_DIRS:
    if p.exists():
        path_str = str(p)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
