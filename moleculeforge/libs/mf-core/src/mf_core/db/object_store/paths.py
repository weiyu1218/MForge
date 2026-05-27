"""Path utilities for object store artifacts."""
from __future__ import annotations

BUCKET = "mf-data"


def checkpoint_path(run_id: str, model_name: str) -> str:
    return f"checkpoints/{run_id}/{model_name}.pt"


def artifact_path(run_id: str, artifact_name: str) -> str:
    return f"artifacts/{run_id}/{artifact_name}"


def molecule_data_path(inchikey: str) -> str:
    return f"molecules/{inchikey[:2]}/{inchikey}.json"


def conformer_path(mol_id: str, conformer_num: int) -> str:
    return f"conformers/{mol_id}/{conformer_num:04d}.sdf"


def report_path(run_id: str) -> str:
    return f"reports/{run_id}/report.html"


def md_trajectory_path(run_id: str, sim_id: str) -> str:
    return f"md_trajectories/{run_id}/{sim_id}.xtc"


def fep_result_path(run_id: str, pair_id: str) -> str:
    return f"fep_results/{run_id}/{pair_id}.json"


def model_weights_path(model_name: str, version: str) -> str:
    return f"models/{model_name}/{version}/weights/"


def audit_log_path(date: str, run_id: str) -> str:
    return f"audit_logs/{date}/{run_id}/signed.jsonl"


def xdl_protocol_path(mol_id: str, route_id: str) -> str:
    return f"xdl_protocols/{mol_id}/{route_id}.xdl"


def extract_inchikey_from_path(path: str) -> str | None:
    parts = path.rstrip(".json").rstrip(".sdf").split("/")
    for p in parts:
        if len(p) == 27 and "-" in p:
            return p
    return None
