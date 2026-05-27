"""Pytest global config: auto-add all workspace package src dirs to sys.path."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

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
    ROOT / "models" / "mf-encoders" / "humu_intent_encoder" / "src",
    ROOT / "models" / "mf-generators" / "hfm_3d" / "src",
    ROOT / "models" / "mf-generators" / "fragfm" / "src",
    ROOT / "models" / "mf-generators" / "lamgen_3d" / "src",
    ROOT / "models" / "mf-generators" / "crem_3d" / "src",
    ROOT / "models" / "mf-generators" / "mmpt_rag" / "src",
    ROOT / "models" / "mf-generators" / "evomol_rl" / "src",
    ROOT / "models" / "mf-generators" / "incremental_clm" / "src",
    ROOT / "models" / "mf-generators" / "uas" / "src",
    ROOT / "models" / "mf-generators" / "rdkit_random" / "src",
    ROOT / "models" / "mf-oracles" / "rdkit-oracle" / "src",
    ROOT / "services" / "cig-compiler-svc" / "src",
    ROOT / "services" / "provenance-svc" / "src",
    ROOT / "agents" / "srb_agent" / "src",
    ROOT / "wetlab" / "xdl-compiler" / "src",
    ROOT / "pipelines" / "mvp_pipeline" / "src",
    ROOT / "agents" / "orchestrator" / "src",
]

for p in PACKAGE_SRC_DIRS:
    if p.exists():
        path_str = str(p)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
