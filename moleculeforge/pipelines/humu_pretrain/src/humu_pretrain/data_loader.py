"""Data loaders for HUMU pretraining — ChEMBL molecules, CrossDocked pockets, USPTO-MIT routes."""
from __future__ import annotations

import gzip
import io
import json
import math
import os
import random
import re
import shlex
import tarfile
import zipfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset


_ALL_HUMU_SOURCE_KEYS = {
    "mol": "mol_source",
    "pocket": "pocket_source",
    "route": "route_source",
    "route_eval": "route_eval_source",
    "joint": "joint_source",
    "activity": "activity_source",
    "protacpedia": "protac_sources",
    "protacdb": "protacdb_source",
    "protac8k": "protac8k_source",
    "rcsb_mmcif": "rcsb_mmcif_source",
    "interface_skempi2": "interface_skempi2_source",
    "pdcdb": "pdcdb_source",
    "retropath_templates": "retropath_template_source",
}

_POCKET_POINT_FIELDS = (
    "coords",
    "elements",
    "residue_types",
    "atom_chain_ids",
    "atom_names",
    "residue_ids",
)

_MOL_REQUIRED_PAIR_TYPES = {
    "mol_self",
    "mol_pocket",
    "mol_route",
    "mol_pocket_route",
    "activity_pair",
    "protac_component",
}

_PROTEIN_PAIR_TYPES = {
    "protein_interface",
    "interface_mutation",
}


class MoleculeDataset(Dataset):
    """Loads ChEMBL molecules from ingested JSONL shards."""

    def __init__(self, data_dir: str, max_mols: int | None = None):
        self.samples: list[dict] = []
        manifest_path = os.path.join(data_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            for shard_path in manifest.get("shards", []):
                full_path = os.path.join(data_dir, shard_path)
                if os.path.exists(full_path):
                    with open(full_path) as f:
                        for line in f:
                            self.samples.append(json.loads(line))
                            if max_mols and len(self.samples) >= max_mols:
                                break
                if max_mols and len(self.samples) >= max_mols:
                    break
        # Fallback: scan for .jsonl files directly
        if not self.samples:
            p = Path(data_dir)
            for fpath in p.glob("*.jsonl"):
                with open(fpath) as f:
                    for line in f:
                        self.samples.append(json.loads(line))
                        if max_mols and len(self.samples) >= max_mols:
                            break
                if max_mols and len(self.samples) >= max_mols:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return {
            "smiles": sample.get("smiles", ""),
            "inchikey": sample.get("inchikey", ""),
            "mw": sample.get("mw", 0.0),
            "logp": sample.get("logp", 0.0),
        }


class PocketDataset(Dataset):
    """Loads CrossDocked binding pocket data from ingested JSONL."""

    def __init__(
        self,
        data_dir: str,
        max_samples: int | None = None,
        max_points: int | None = None,
    ):
        self.samples: list[dict] = []
        self.data_dir = Path(data_dir)
        self.max_points = _positive_int_or_none(max_points)
        p = Path(data_dir)
        for fpath in p.glob("*.jsonl"):
            with open(fpath) as f:
                for line in f:
                    self.samples.append(json.loads(line))
                    if max_samples and len(self.samples) >= max_samples:
                        break
            if max_samples and len(self.samples) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self._load_pocket_record(self.samples[idx])
        payload = {
            "index": sample.get("index"),
            "pdb_id": sample.get("pdb_id", ""),
            "coords": sample["coords"],
            "elements": sample["elements"],
            "residue_types": sample["residue_types"],
            "ligand_smiles": sample.get("ligand_smiles", ""),
            "source_receptor_pdb": sample.get("source_receptor_pdb"),
            "source_dataset": sample.get("source_dataset"),
            "split": sample.get("split"),
        }
        _copy_pocket_optional_fields(payload, sample)
        return payload

    def _load_pocket_record(self, record: dict) -> dict:
        if {"coords", "elements", "residue_types"}.issubset(record):
            return _limit_pocket_points(dict(record), self.max_points)

        pocket_path = None
        if "pocket_path" in record:
            pocket_path = self.data_dir / str(record["pocket_path"])
        elif "index" in record:
            pocket_path = self.data_dir / f"pocket_{int(record['index']):06d}.json"
        elif "pdb_id" in record:
            pocket_path = self.data_dir / f"pocket_{record['pdb_id']}.json"

        if pocket_path is None or not pocket_path.exists():
            raise FileNotFoundError(f"Pocket coordinate file not found for record: {record}")
        with open(pocket_path) as handle:
            pocket = json.load(handle)
        atoms = pocket.get("pocket_atoms", [])
        if not atoms:
            raise ValueError(f"Pocket coordinate file has no pocket_atoms: {pocket_path}")

        loaded = dict(record)
        loaded["coords"] = [[atom["x"], atom["y"], atom["z"]] for atom in atoms]
        loaded["elements"] = [atom["element"] for atom in atoms]
        loaded["residue_types"] = [atom["residue"] for atom in atoms]
        loaded["pdb_id"] = record.get("pdb_id", pocket.get("pdb_id", ""))
        _copy_pocket_optional_fields(loaded, pocket)
        return _limit_pocket_points(loaded, self.max_points)


class RouteDataset(Dataset):
    """Loads USPTO-MIT reaction route data from ingested JSONL."""

    def __init__(self, data_dir: str, max_samples: int | None = None):
        self.samples: list[dict] = []
        p = Path(data_dir)
        for fpath in p.glob("*.jsonl"):
            with open(fpath) as f:
                for line in f:
                    self.samples.append(json.loads(line))
                    if max_samples and len(self.samples) >= max_samples:
                        break
            if max_samples and len(self.samples) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        reactions = sample.get("reactions")
        if reactions is None and "reaction_smiles" in sample:
            reactions = [sample["reaction_smiles"]]
        if reactions is None:
            reactions = _extract_reactions_from_tree(sample)
        return {
            "id": sample.get("id", ""),
            "source_split": sample.get("source_split", sample.get("split", "")),
            "root_smiles": sample.get("root_smiles", ""),
            "n_steps": sample.get("n_steps", sample.get("steps", len(reactions))),
            "steps": sample.get("steps", sample.get("n_steps", len(reactions))),
            "tree_depth": sample.get("tree_depth", 1),
            "reaction_types": sample.get("reaction_types", []),
            "reactions": reactions,
            "intermediates": sample.get("intermediates", []),
            "score": sample.get("score", 0.0),
        }


class _IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[self.indices[idx]]


class PairedHUMUDataset(Dataset):
    """Build true HUMU positive pairs from pocket ligands and route products."""

    def __init__(
        self,
        pocket_dir: str,
        route_dir: str,
        max_samples: int | None = None,
        mol_dir: str | None = None,
        joint_dir: str | None = None,
        activity_dirs: list[str] | None = None,
        protac_dirs: list[str] | None = None,
        protac8k_dir: str | None = None,
        rcsb_mmcif_dir: str | None = None,
        interface_skempi2_dir: str | None = None,
        pdcdb_dir: str | None = None,
        route_eval_dir: str | None = None,
        retropath_template_dirs: list[str] | None = None,
        require_pocket_route: bool = False,
        require_protac_component: bool = False,
        pocket_esm2_max_sequence_length: int | None = None,
        max_pocket_points: int | None = None,
    ):
        self.max_pocket_points = _positive_int_or_none(max_pocket_points)
        self.pockets = PocketDataset(
            pocket_dir,
            max_samples=max_samples,
            max_points=self.max_pocket_points,
        )
        self.routes = RouteDataset(route_dir, max_samples=max_samples)
        self.molecules = MoleculeDataset(mol_dir, max_mols=max_samples) if mol_dir else None
        self.joint_dir = Path(joint_dir) if joint_dir else None
        self.protac_component_records: list[dict] = []
        self.protac_ternary_records: list[dict] = []
        self.protein_interface_records: list[dict] = []
        self.interface_mutation_records: list[dict] = []
        self.pdc_component_records: list[dict] = []
        self.activity_pair_records: list[dict] = []
        self.route_template_records: list[dict] = []
        self.pocket_esm2_max_sequence_length = _positive_int_or_none(
            pocket_esm2_max_sequence_length
        )
        self.filtered_counts = {
            "pocket_esm2_sequence": 0,
            "joint_esm2_sequence": 0,
        }
        self.joint_records = (
            list(_iter_jsonl_records(joint_dir, max_samples=max_samples))
            if joint_dir
            else []
        )
        self.samples: list[dict] = []
        if self.molecules is not None:
            for mol_index, molecule in enumerate(self.molecules.samples):
                smiles = molecule.get("smiles")
                if not smiles:
                    continue
                self.samples.append(
                    {
                        "pair_type": "mol_self",
                        "mol_id": molecule.get("inchikey") or f"mol:{mol_index}",
                        "pocket_id": None,
                        "route_id": None,
                        "ligand_smiles": smiles,
                        "positive_smiles": smiles,
                        "target_id": None,
                        "source_dataset": molecule.get("source_dataset") or "mol",
                        "source_name": "mol",
                        "split": molecule.get("split") or "train",
                        "_source_kind": "mol_self",
                        "_source_index": mol_index,
                    }
                )
        for pocket_index, pocket in enumerate(self.pockets.samples):
            ligand_smiles = pocket.get("ligand_smiles")
            pocket_id = pocket.get("pdb_id") or str(pocket.get("index", ""))
            if not ligand_smiles:
                raise ValueError(f"Pocket record is missing ligand_smiles: {pocket_id}")
            if not pocket_id:
                raise ValueError("Pocket record is missing pdb_id")
            if _pocket_exceeds_esm2_max_sequence_length(
                pocket,
                self.pockets._load_pocket_record,
                self.pocket_esm2_max_sequence_length,
            ):
                self.filtered_counts["pocket_esm2_sequence"] += 1
                continue
            self.samples.append(
                {
                    "pair_type": "mol_pocket",
                    "mol_id": f"pocket_ligand:{pocket_id}",
                    "pocket_id": pocket_id,
                    "route_id": None,
                    "ligand_smiles": ligand_smiles,
                    "target_id": pocket.get("source_receptor_pdb"),
                    "source_dataset": pocket.get("source_dataset") or "crossdocked",
                    "source_name": "pocket",
                    "split": pocket.get("split") or "train",
                    "_source_kind": "pocket",
                    "_source_index": pocket_index,
                }
            )
        for route_index, route in enumerate(self.routes.samples):
            ligand_smiles = route.get("root_smiles")
            route_id = route.get("id")
            if not ligand_smiles:
                raise ValueError(f"Route record is missing root_smiles: {route_id}")
            if not route_id:
                raise ValueError("Route record is missing id")
            self.samples.append(
                {
                    "pair_type": "mol_route",
                    "mol_id": f"route_product:{route_id}",
                    "pocket_id": None,
                    "route_id": route_id,
                    "ligand_smiles": ligand_smiles,
                    "target_id": None,
                    "source_dataset": route.get("source_dataset") or "uspto_mit",
                    "source_name": "route",
                    "split": route.get("source_split") or "train",
                    "_source_kind": "route",
                    "_source_index": route_index,
                }
            )
        for joint_index, record in enumerate(self.joint_records):
            ligand_smiles = record.get("ligand_smiles") or record.get("root_smiles")
            joint_id = record.get("id")
            pocket_id = record.get("pdb_id") or record.get("pocket_id")
            route_id = record.get("route_id") or joint_id
            if not ligand_smiles:
                raise ValueError(f"Joint HUMU record is missing ligand_smiles: {joint_id}")
            if not joint_id:
                raise ValueError("Joint HUMU record is missing id")
            if not pocket_id:
                raise ValueError(f"Joint HUMU record is missing pocket id: {joint_id}")
            if not route_id:
                raise ValueError(f"Joint HUMU record is missing route id: {joint_id}")
            if _joint_exceeds_esm2_max_sequence_length(
                record,
                self.joint_dir or Path("."),
                str(pocket_id),
                self.pocket_esm2_max_sequence_length,
            ):
                self.filtered_counts["joint_esm2_sequence"] += 1
                continue
            self.samples.append(
                {
                    "pair_type": "mol_pocket_route",
                    "mol_id": f"joint:{joint_id}",
                    "pocket_id": pocket_id,
                    "route_id": route_id,
                    "ligand_smiles": ligand_smiles,
                    "target_id": record.get("target_id") or record.get("source_receptor_pdb"),
                    "source_dataset": record.get("source_dataset") or "joint",
                    "source_name": "joint",
                    "split": record.get("split") or "train",
                    "_source_kind": "joint",
                    "_source_index": joint_index,
                }
            )
        for activity_dir in activity_dirs or []:
            for record in _iter_activity_pair_records(activity_dir, max_samples=max_samples):
                record_index = len(self.activity_pair_records)
                self.activity_pair_records.append(record)
                self.samples.append(
                    {
                        **record,
                        "pair_type": "activity_pair",
                        "mol_id": f"activity_pair:{record['record_id']}",
                        "pocket_id": None,
                        "route_id": None,
                        "split": record.get("split") or "train",
                        "_source_kind": "activity_pair",
                        "_source_index": record_index,
                    }
                )
        for protac_dir in protac_dirs or []:
            for record in _iter_protac_component_records(protac_dir, max_samples=max_samples):
                record_index = len(self.protac_component_records)
                self.protac_component_records.append(record)
                self.samples.append(
                    {
                        **record,
                        "pair_type": "protac_component",
                        "mol_id": (
                            "protac_component:"
                            f"{record['record_id']}:{record['component_type']}"
                        ),
                        "pocket_id": None,
                        "route_id": None,
                        "target_id": record.get("target_id"),
                        "split": record.get("split") or "train",
                        "source_name": record.get("source_name") or _source_name_from_path(protac_dir),
                        "_source_kind": "protac_component",
                        "_source_index": record_index,
                    }
                )
                if record.get("pair_type") == "protac_component_library":
                    self.samples[-1]["pair_type"] = "protac_component_library"
        if protac8k_dir:
            for record in _iter_protac_ternary_records(
                protac8k_dir,
                max_samples=max_samples,
            ):
                record_index = len(self.protac_ternary_records)
                self.protac_ternary_records.append(record)
                self.samples.append(
                    {
                        **record,
                        "pair_type": "protac_ternary",
                        "mol_id": f"protac_ternary:{record['record_id']}",
                        "pocket_id": record.get("target_pocket_id"),
                        "route_id": None,
                        "ligand_smiles": record.get("ligand_smiles"),
                        "target_id": record.get("target_id"),
                        "split": record.get("split") or "train",
                        "source_name": "protac8k",
                        "_source_kind": "protac_ternary",
                        "_source_index": record_index,
                    }
                )
        if rcsb_mmcif_dir:
            for record in _iter_protein_interface_records(
                rcsb_mmcif_dir,
                source_name="rcsb_mmcif",
                max_samples=max_samples,
            ):
                record_index = len(self.protein_interface_records)
                self.protein_interface_records.append(record)
                self.samples.append(
                    {
                        **record,
                        "pair_type": "protein_interface",
                        "mol_id": None,
                        "pocket_id": record.get("pdb_id") or record["record_id"],
                        "route_id": None,
                        "ligand_smiles": None,
                        "target_id": record.get("pdb_id"),
                        "split": record.get("split") or "train",
                        "source_name": "rcsb_mmcif",
                        "_source_kind": "protein_interface",
                        "_source_index": record_index,
                    }
                )
        if interface_skempi2_dir:
            for record in _iter_interface_mutation_records(
                interface_skempi2_dir,
                max_samples=max_samples,
            ):
                record_index = len(self.interface_mutation_records)
                self.interface_mutation_records.append(record)
                self.samples.append(
                    {
                        **record,
                        "pair_type": "interface_mutation",
                        "mol_id": None,
                        "pocket_id": record.get("pdb_complex") or record["record_id"],
                        "route_id": None,
                        "ligand_smiles": None,
                        "target_id": record.get("pdb_complex"),
                        "split": record.get("split") or "train",
                        "source_name": "interface_skempi2",
                        "_source_kind": "interface_mutation",
                        "_source_index": record_index,
                    }
                )
        if pdcdb_dir:
            for record in _iter_pdc_component_records(
                pdcdb_dir,
                max_samples=max_samples,
            ):
                record_index = len(self.pdc_component_records)
                self.pdc_component_records.append(record)
                self.samples.append(
                    {
                        **record,
                        "pair_type": "pdc_component",
                        "mol_id": f"pdc_component:{record['record_id']}",
                        "pocket_id": record.get("peptide_id") or record["record_id"],
                        "route_id": None,
                        "ligand_smiles": record.get("ligand_smiles"),
                        "target_id": record.get("target_id"),
                        "split": record.get("split") or "train",
                        "source_name": "pdcdb",
                        "_source_kind": "pdc_component",
                        "_source_index": record_index,
                    }
                )
        for route_template_dir in retropath_template_dirs or []:
            for record in _iter_route_template_records(
                route_template_dir,
                max_samples=max_samples,
            ):
                record_index = len(self.route_template_records)
                self.route_template_records.append(record)
                self.samples.append(
                    {
                        **record,
                        "pair_type": "route_template",
                        "mol_id": f"route_template:{record['record_id']}",
                        "pocket_id": None,
                        "route_id": record["route_id"],
                        "target_id": None,
                        "split": record.get("split") or "train",
                        "_source_kind": "route_template",
                        "_source_index": record_index,
                    }
                )
        if route_eval_dir:
            for record in _iter_route_eval_template_records(
                route_eval_dir,
                max_samples=max_samples,
            ):
                record_index = len(self.route_template_records)
                self.route_template_records.append(record)
                self.samples.append(
                    {
                        **record,
                        "pair_type": "route_template",
                        "mol_id": f"route_template:{record['record_id']}",
                        "pocket_id": None,
                        "route_id": record["route_id"],
                        "target_id": None,
                        "split": record.get("split") or "valid",
                        "_source_kind": "route_template",
                        "_source_index": record_index,
                    }
                )
        if require_pocket_route and not any(
            sample["pair_type"] == "mol_pocket_route" for sample in self.samples
        ):
            raise ValueError("data.joint_source must contain usable mol-pocket-route records")
        if require_protac_component and not any(
            sample["pair_type"] == "protac_component" for sample in self.samples
        ):
            raise ValueError("data.protac_sources must contain usable protac component records")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = dict(self.samples[idx])
        source_index = sample.pop("_source_index")
        source_kind = sample.pop("_source_kind")
        if source_kind == "pocket":
            pocket = self.pockets[source_index]
            sample["pocket"] = {
                "pdb_id": sample["pocket_id"],
                "coords": pocket["coords"],
                "elements": pocket["elements"],
                "residue_types": pocket["residue_types"],
                "source_name": "pocket",
            }
            _copy_pocket_optional_fields(sample["pocket"], pocket)
            sample["route"] = None
            return sample

        if source_kind == "route":
            sample["pocket"] = None
            sample["route"] = self.routes[source_index]
            return sample

        if source_kind in {"mol_self", "activity_pair"}:
            sample["pocket"] = None
            sample["route"] = None
            return sample

        if source_kind == "protac_component":
            sample["pocket"] = None
            sample["route"] = None
            return sample

        if source_kind == "route_template":
            sample["pocket"] = None
            sample["route"] = self.route_template_records[source_index]["route"]
            return sample

        if source_kind == "protac_ternary":
            record = self.protac_ternary_records[source_index]
            sample["pocket"] = None
            sample["route"] = None
            sample["target_pocket"] = _protein_payload_from_record(
                record,
                "target_pocket",
                record.get("record_id", sample["mol_id"]),
            )
            sample["e3_pocket"] = _protein_payload_from_record(
                record,
                "e3_pocket",
                record.get("record_id", sample["mol_id"]),
            )
            return sample

        if source_kind == "protein_interface":
            record = self.protein_interface_records[source_index]
            anchor, positive = _protein_interface_views(record)
            sample["pocket"] = None
            sample["route"] = None
            sample["interface_anchor"] = anchor
            sample["interface_positive"] = positive
            return sample

        if source_kind == "interface_mutation":
            record = self.interface_mutation_records[source_index]
            anchor, positive = _interface_mutation_views(record)
            sample["pocket"] = None
            sample["route"] = None
            sample["interface_anchor"] = anchor
            sample["interface_positive"] = positive
            sample["interface_affinity_delta"] = _affinity_delta(record)
            return sample

        if source_kind == "pdc_component":
            record = self.pdc_component_records[source_index]
            sample["pocket"] = None
            sample["route"] = None
            if record.get("peptide_pocket") or record.get("peptide_sequence"):
                sample["peptide_pocket"] = _pdc_peptide_payload(record)
            return sample

        record = self.joint_records[source_index]
        sample["pocket"] = _pocket_payload_from_record(
            record,
            self.joint_dir or Path("."),
            sample["pocket_id"],
            max_points=self.max_pocket_points,
        )
        sample["pocket"]["source_name"] = "joint"
        sample["route"] = _route_payload_from_record(record, sample["route_id"])
        return sample


def preflight_humu_data_contract(config: dict) -> dict:
    """Validate HUMU data contracts before starting a training process."""
    _reject_intent_pretrain_config(config)
    data_cfg = config.get("data", {})
    loss_weights = config.get("loss_weights", {})
    objective_sampling_cfg = data_cfg.get("objective_sampling", {}) or {}
    pocket_encoder_cfg = config.get("encoders", {}).get("pocket", {})
    require_pocket_route = float(loss_weights.get("pocket_route", 0.0) or 0.0) > 0.0
    require_pocket_esm2 = bool(pocket_encoder_cfg.get("use_esm2", False))
    esm2_dim = int(pocket_encoder_cfg.get("esm2_dim", 1280))

    pocket_dir = data_cfg.get("pocket_source", "")
    route_dir = data_cfg.get("route_source", "")
    joint_dir = data_cfg.get("joint_source")
    mol_dir = data_cfg.get("mol_source")
    route_eval_dir = data_cfg.get("route_eval_source")
    protacdb_dir = data_cfg.get("protacdb_source")
    protac8k_dir = data_cfg.get("protac8k_source")
    rcsb_mmcif_dir = data_cfg.get("rcsb_mmcif_source")
    interface_skempi2_dir = data_cfg.get("interface_skempi2_source")
    pdcdb_dir = data_cfg.get("pdcdb_source")
    retropath_template_dirs = _configured_source_dirs(
        data_cfg,
        "retropath_template_source",
        "retropath_template_sources",
    )
    activity_dirs = _configured_source_dirs(data_cfg, "activity_source", "activity_sources")
    protac_dirs = _configured_source_dirs(data_cfg, "protac_source", "protac_sources")
    if protacdb_dir:
        protac_dirs.append(str(protacdb_dir))
    require_protac_component = (
        float(loss_weights.get("protac_component", 0.0) or 0.0) > 0.0
    )

    _require_data_dir("pocket_source", pocket_dir)
    _require_data_dir("route_source", route_dir)

    report = {
        "required": {
            "joint_source": require_pocket_route,
        },
        "sources": {
            "pocket_source": _source_report(pocket_dir),
            "route_source": _source_report(route_dir),
        },
    }
    report["source_registry"] = _build_source_registry(data_cfg)
    if data_cfg.get("require_all_humu_sources"):
        _validate_required_source_registry(report["source_registry"])

    if mol_dir:
        _require_data_dir("mol_source", mol_dir)
        report["sources"]["mol_source"] = {
            **_source_report(mol_dir),
            "records": _dataset_manifest_record_count(mol_dir) or _count_jsonl_records(mol_dir),
        }

    if require_pocket_route:
        _require_data_dir("joint_source", joint_dir or "")
        joint_records = _validate_joint_source(joint_dir)
        if joint_records == 0:
            raise ValueError(f"data.joint_source contains no joint records: {joint_dir}")
        report["sources"]["joint_source"] = {
            **_source_report(joint_dir),
            "records": joint_records,
        }
    elif joint_dir:
        _require_data_dir("joint_source", joint_dir)
        report["sources"]["joint_source"] = {
            **_source_report(joint_dir),
            "records": _validate_joint_source(joint_dir),
        }

    if require_pocket_esm2:
        report["sources"]["pocket_source"]["esm2_records"] = _validate_pocket_esm2_source(
            pocket_dir,
            esm2_dim,
        )

    if activity_dirs:
        activity_source_reports = []
        activity_total = 0
        for activity_dir in activity_dirs:
            _require_data_dir("activity_source", activity_dir)
            records = _validate_activity_source(activity_dir)
            activity_total += records
            activity_source_reports.append(
                {
                    **_source_report(activity_dir),
                    "records": records,
                }
            )
        report["sources"]["activity_sources"] = {
            "records": activity_total,
            "sources": activity_source_reports,
        }
        first_activity_dir = activity_dirs[0]
        report["sources"]["activity_source"] = activity_source_reports[0] | {
            "path": str(Path(first_activity_dir))
        }

    if protac_dirs:
        protac_source_reports = []
        protac_total = 0
        for protac_dir in protac_dirs:
            _require_data_dir("protac_source", protac_dir)
            records = _validate_protac_source(protac_dir)
            protac_total += records
            protac_source_reports.append(
                {
                    **_source_report(protac_dir),
                    "records": records,
                }
            )
        if require_protac_component and protac_total == 0:
            raise ValueError("data.protac_sources contains no protac component records")
        report["sources"]["protac_sources"] = {
            "records": protac_total,
            "sources": protac_source_reports,
        }
    elif require_protac_component:
        raise FileNotFoundError("data.protac_sources is required for HUMU pretraining")

    if route_eval_dir:
        _require_data_dir("route_eval_source", route_eval_dir)
        report["sources"]["route_eval_source"] = {
            **_source_report(route_eval_dir),
            "records": _dataset_manifest_record_count(route_eval_dir) or _count_jsonl_records(route_eval_dir),
        }
    if retropath_template_dirs:
        total = 0
        reports = []
        for template_dir in retropath_template_dirs:
            _require_data_dir("retropath_template_source", template_dir)
            records = _dataset_manifest_record_count(template_dir) or _count_jsonl_records(template_dir)
            total += records
            reports.append({**_source_report(template_dir), "records": records})
        report["sources"]["retropath_template_sources"] = {
            "records": total,
            "sources": reports,
        }
    for key, value in (
        ("protac8k_source", protac8k_dir),
        ("rcsb_mmcif_source", rcsb_mmcif_dir),
        ("interface_skempi2_source", interface_skempi2_dir),
        ("pdcdb_source", pdcdb_dir),
    ):
        if value:
            _require_data_dir(key, value)
            source_report = {
                **_source_report(value),
                "records": _dataset_manifest_record_count(value) or _count_jsonl_records(value),
            }
            if key == "protac8k_source" and _objective_enabled(
                objective_sampling_cfg,
                "protac_ternary",
            ):
                trainable_records = _count_iterator_records(
                    _iter_protac_ternary_records(value)
                )
                if trainable_records == 0:
                    raise ValueError("data.protac8k_source contains no protac_ternary records")
                source_report["trainable_records"] = trainable_records
            if key == "pdcdb_source" and _objective_enabled(
                objective_sampling_cfg,
                "pdc_component",
            ):
                trainable_records = _count_iterator_records(
                    _iter_pdc_component_records(value)
                )
                if trainable_records == 0:
                    raise ValueError("data.pdcdb_source contains no pdc_component records")
                source_report["trainable_records"] = trainable_records
            report["sources"][key] = source_report

    return report


def _stratified_train_val_split(
    dataset: Dataset,
    eval_split_ratio: float,
    seed: int,
) -> tuple[Subset, Subset]:
    """Split a paired dataset into train/val while preserving every objective.

    Splits each pair_type independently so the training subset always contains
    at least one record per objective (required by the objective batch sampler)
    and the validation subset is representative across objectives instead of
    being dominated by the largest sources.
    """
    by_pair_type: dict[str, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        pair_type = _dataset_pair_type(dataset, index) or "unknown"
        by_pair_type[pair_type].append(index)
    rng = random.Random(seed)
    shuffled: dict[str, list[int]] = {}
    for pair_type in sorted(by_pair_type):
        indices = list(by_pair_type[pair_type])
        rng.shuffle(indices)
        shuffled[pair_type] = indices
    train_indices: list[int] = []
    val_indices: list[int] = []
    val_counts: dict[str, int] = {}
    for pair_type in sorted(shuffled):
        indices = shuffled[pair_type]
        count = len(indices)
        # keep at least one record per objective in the training subset so the
        # objective batch sampler always has data for every objective
        val_count = min(int(count * eval_split_ratio), max(count - 1, 0))
        val_counts[pair_type] = val_count
    # if flooring left the validation set empty, promote one record from the
    # largest pair_type that can still spare one (count >= 2)
    if sum(val_counts.values()) == 0:
        candidates = [pt for pt, idx in shuffled.items() if len(idx) >= 2]
        if candidates:
            largest = max(candidates, key=lambda pt: len(shuffled[pt]))
            val_counts[largest] = 1
    for pair_type in sorted(shuffled):
        indices = shuffled[pair_type]
        val_count = val_counts[pair_type]
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def create_dataloaders(config: dict) -> dict[str, DataLoader]:
    """Create DataLoader instances from config."""
    _reject_intent_pretrain_config(config)
    data_cfg = config.get("data", {})
    batch_size = config.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 4)
    if data_cfg.get("require_all_humu_sources"):
        _validate_required_source_registry(_build_source_registry(data_cfg))

    pocket_dir = data_cfg.get("pocket_source", "")
    route_dir = data_cfg.get("route_source", "")
    mol_dir = data_cfg.get("mol_source")
    joint_dir = data_cfg.get("joint_source")
    loss_weights = config.get("loss_weights", {})
    require_pocket_route = float(loss_weights.get("pocket_route", 0.0) or 0.0) > 0.0
    require_protac_component = (
        float(loss_weights.get("protac_component", 0.0) or 0.0) > 0.0
    )
    objective_sampling_cfg = data_cfg.get("objective_sampling", {}) or {}
    objective_sampling_enabled = bool(objective_sampling_cfg.get("enabled", False))
    source_cap = _positive_int_or_none(
        config.get("max_samples")
        or objective_sampling_cfg.get("source_cap")
        or data_cfg.get("source_sample_cap")
    )
    dataset_max_samples = config.get("max_samples") or source_cap
    protac_dirs = _configured_source_dirs(data_cfg, "protac_source", "protac_sources")
    protacdb_dir = data_cfg.get("protacdb_source")
    if protacdb_dir and str(protacdb_dir) not in protac_dirs:
        protac_dirs.append(str(protacdb_dir))
    protac8k_dir = data_cfg.get("protac8k_source")
    rcsb_mmcif_dir = data_cfg.get("rcsb_mmcif_source")
    interface_skempi2_dir = data_cfg.get("interface_skempi2_source")
    pdcdb_dir = data_cfg.get("pdcdb_source")
    activity_dirs = _configured_source_dirs(data_cfg, "activity_source", "activity_sources")
    route_eval_dir = data_cfg.get("route_eval_source")
    retropath_template_dirs = _configured_source_dirs(
        data_cfg,
        "retropath_template_source",
        "retropath_template_sources",
    )
    pocket_esm2_max_sequence_length = _pocket_esm2_max_sequence_length(config)
    max_pocket_points = _max_pocket_points(config)

    _require_data_dir("pocket_source", pocket_dir)
    _require_data_dir("route_source", route_dir)
    if mol_dir and _objective_enabled(objective_sampling_cfg, "mol_self"):
        _require_data_dir("mol_source", mol_dir)
    if require_pocket_route:
        _require_data_dir("joint_source", joint_dir or "")
    elif joint_dir:
        _require_data_dir("joint_source", joint_dir)
    if require_protac_component and not protac_dirs:
        raise FileNotFoundError("data.protac_sources is required for HUMU pretraining")
    for protac_dir in protac_dirs:
        _require_data_dir("protac_source", protac_dir)
    for activity_dir in activity_dirs:
        _require_data_dir("activity_source", activity_dir)
    if route_eval_dir and _objective_enabled(objective_sampling_cfg, "route_template"):
        _require_data_dir("route_eval_source", route_eval_dir)
    for template_dir in retropath_template_dirs:
        _require_data_dir("retropath_template_source", template_dir)
    if protac8k_dir and _objective_enabled(objective_sampling_cfg, "protac_ternary"):
        _require_data_dir("protac8k_source", protac8k_dir)
    if rcsb_mmcif_dir and _objective_enabled(objective_sampling_cfg, "protein_interface"):
        _require_data_dir("rcsb_mmcif_source", rcsb_mmcif_dir)
    if interface_skempi2_dir and _objective_enabled(objective_sampling_cfg, "interface_mutation"):
        _require_data_dir("interface_skempi2_source", interface_skempi2_dir)
    if pdcdb_dir and _objective_enabled(objective_sampling_cfg, "pdc_component"):
        _require_data_dir("pdcdb_source", pdcdb_dir)

    paired_ds = PairedHUMUDataset(
        pocket_dir,
        route_dir,
        max_samples=dataset_max_samples,
        mol_dir=mol_dir if _objective_enabled(objective_sampling_cfg, "mol_self") else None,
        joint_dir=joint_dir,
        activity_dirs=(
            activity_dirs if _objective_enabled(objective_sampling_cfg, "activity_pair") else None
        ),
        protac_dirs=protac_dirs,
        protac8k_dir=(
            protac8k_dir if _objective_enabled(objective_sampling_cfg, "protac_ternary") else None
        ),
        rcsb_mmcif_dir=(
            rcsb_mmcif_dir
            if _objective_enabled(objective_sampling_cfg, "protein_interface")
            else None
        ),
        interface_skempi2_dir=(
            interface_skempi2_dir
            if _objective_enabled(objective_sampling_cfg, "interface_mutation")
            else None
        ),
        pdcdb_dir=(
            pdcdb_dir if _objective_enabled(objective_sampling_cfg, "pdc_component") else None
        ),
        route_eval_dir=(
            route_eval_dir if _objective_enabled(objective_sampling_cfg, "route_template") else None
        ),
        retropath_template_dirs=(
            retropath_template_dirs
            if _objective_enabled(objective_sampling_cfg, "route_template")
            else None
        ),
        require_pocket_route=require_pocket_route,
        require_protac_component=require_protac_component,
        pocket_esm2_max_sequence_length=pocket_esm2_max_sequence_length,
        max_pocket_points=max_pocket_points,
    )
    _require_non_empty("paired", paired_ds)

    train_ds: Dataset = paired_ds
    validation_ds: Dataset | None = None
    eval_cfg = config.get("eval", {}) or {}
    eval_every = int(eval_cfg.get("every_n_epochs", 0) or 0)
    eval_split_ratio = float(eval_cfg.get("eval_split_ratio", 0.0) or 0.0)
    if eval_every > 0 and eval_split_ratio > 0.0:
        if not 0.0 < eval_split_ratio < 1.0:
            raise ValueError("eval.eval_split_ratio must be between 0 and 1")
        if len(paired_ds) < 2:
            raise ValueError("HUMU validation split requires at least two paired records")
        train_ds, validation_ds = _stratified_train_val_split(
            paired_ds,
            eval_split_ratio,
            int(config.get("seed", 42)),
        )
        if len(train_ds) == 0:
            raise ValueError("eval.eval_split_ratio leaves no HUMU training records")
        if len(validation_ds) == 0:
            raise ValueError("eval.eval_split_ratio leaves no HUMU validation records")
    if objective_sampling_enabled:
        batch_sampler = TargetRatioMultiSourceBatchSampler(
            train_ds,
            batch_size=batch_size,
            objective_ratios=objective_sampling_cfg.get("objectives", {}),
            steps_per_epoch=int(objective_sampling_cfg.get("steps_per_epoch", 0) or 0),
            alpha=float(objective_sampling_cfg.get("alpha", 0.5)),
            seed=int(config.get("seed", 42)),
        )
        loaders = {
            "paired": DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=data_cfg.get("pin_memory", False),
                collate_fn=_record_collate,
            ),
        }
    else:
        loaders = {
            "paired": DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=data_cfg.get("shuffle", True),
                num_workers=num_workers,
                pin_memory=data_cfg.get("pin_memory", False),
                collate_fn=_record_collate,
            ),
        }
    if validation_ds is not None:
        loaders["validation"] = DataLoader(
            validation_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=data_cfg.get("pin_memory", False),
            collate_fn=_record_collate,
        )
    return loaders


class TargetRatioMultiSourceBatchSampler(Sampler[list[int]]):
    """Build fixed-size HUMU batches from configured objective ratios."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        objective_ratios: dict,
        steps_per_epoch: int = 0,
        alpha: float = 0.5,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0 for objective sampling")
        if not objective_ratios:
            raise ValueError("data.objective_sampling.objectives is required")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.alpha = float(alpha)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.seed = int(seed)
        self.epoch = 0
        self.steps_per_epoch = (
            int(steps_per_epoch)
            if int(steps_per_epoch or 0) > 0
            else max(1, math.ceil(len(dataset) / float(batch_size)))
        )
        self.objective_ratios = {
            str(name): float(value)
            for name, value in objective_ratios.items()
            if float(value) > 0.0
        }
        if not self.objective_ratios:
            raise ValueError("objective sampling requires at least one positive ratio")
        self.quotas = _objective_batch_quotas(self.objective_ratios, self.batch_size)
        self.indices_by_objective = _indices_by_pair_type(dataset)
        self.indices_by_objective_source = _indices_by_pair_type_and_source(dataset)
        self.source_schedule_by_objective = {
            objective: _source_schedule(source_indices, self.alpha)
            for objective, source_indices in self.indices_by_objective_source.items()
        }
        missing = [
            objective
            for objective in self.quotas
            if not self.indices_by_objective.get(objective)
        ]
        if missing:
            raise ValueError(
                "objective sampling has no HUMU records for objectives: "
                f"{', '.join(sorted(missing))}"
            )

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed * 1_000_003 + self.epoch)
        shuffled_indices: dict[tuple[str, str], list[int]] = {}
        for objective in sorted(self.indices_by_objective_source):
            for source_name in sorted(self.indices_by_objective_source[objective]):
                permutation = list(self.indices_by_objective_source[objective][source_name])
                rng.shuffle(permutation)
                shuffled_indices[(objective, source_name)] = permutation
        objective_cursors = defaultdict(int)
        source_cursors = defaultdict(int)
        for _step in range(self.steps_per_epoch):
            batch: list[int] = []
            for objective, count in self.quotas.items():
                source_schedule = self.source_schedule_by_objective[objective]
                for _ in range(count):
                    objective_cursor = objective_cursors[objective]
                    source_name = source_schedule[objective_cursor % len(source_schedule)]
                    indices = shuffled_indices[(objective, source_name)]
                    source_cursor_key = (objective, source_name)
                    source_cursor = source_cursors[source_cursor_key]
                    offset = source_cursor * self.world_size + self.rank
                    batch.append(indices[offset % len(indices)])
                    objective_cursors[objective] = objective_cursor + 1
                    source_cursors[source_cursor_key] = source_cursor + 1
            yield batch


def _objective_batch_quotas(objective_ratios: dict[str, float], batch_size: int) -> dict[str, int]:
    total = sum(objective_ratios.values())
    if total <= 0:
        raise ValueError("objective sampling ratios must sum to > 0")
    raw = {
        objective: (weight / total) * batch_size
        for objective, weight in objective_ratios.items()
    }
    quotas = {objective: int(math.floor(value)) for objective, value in raw.items()}
    remainder = batch_size - sum(quotas.values())
    ranked = sorted(
        raw,
        key=lambda objective: (raw[objective] - quotas[objective], raw[objective]),
        reverse=True,
    )
    for objective in ranked[:remainder]:
        quotas[objective] += 1
    return {objective: count for objective, count in quotas.items() if count > 0}


def _indices_by_pair_type(dataset: Dataset) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        pair_type = _dataset_pair_type(dataset, index)
        if pair_type:
            indices[pair_type].append(index)
    return dict(indices)


def _indices_by_pair_type_and_source(dataset: Dataset) -> dict[str, dict[str, list[int]]]:
    indices: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index in range(len(dataset)):
        pair_type = _dataset_pair_type(dataset, index)
        if not pair_type:
            continue
        source_name = _dataset_source_name(dataset, index)
        indices[pair_type][source_name].append(index)
    return {
        objective: dict(source_indices)
        for objective, source_indices in indices.items()
    }


def _source_schedule(source_indices: dict[str, list[int]], alpha: float) -> list[str]:
    schedule = []
    for source_name, indices in source_indices.items():
        weight = max(1, int(round(len(indices) ** max(0.0, alpha))))
        schedule.extend([source_name] * weight)
    if not schedule:
        raise ValueError("source schedule requires at least one source")
    return schedule


def _objective_enabled(objective_sampling_cfg: dict, objective: str) -> bool:
    objectives = objective_sampling_cfg.get("objectives") or {}
    return bool(objective_sampling_cfg.get("enabled", False)) and float(
        objectives.get(objective, 0.0) or 0.0
    ) > 0.0


def _require_data_dir(key: str, value: str) -> None:
    if not value:
        raise FileNotFoundError(f"data.{key} is required for HUMU pretraining")
    if not os.path.isdir(value):
        raise FileNotFoundError(f"data.{key} does not exist: {value}")


def _reject_intent_pretrain_config(config: dict) -> None:
    if "intent" in (config.get("loss_weights") or {}):
        raise ValueError("loss_weights.intent is not supported by HUMU pretraining")
    data_cfg = config.get("data") or {}
    if "intent_source" in data_cfg:
        raise ValueError("data.intent_source is not supported by HUMU pretraining")
    if "joint_oversample_factor" in data_cfg:
        raise ValueError(
            "data.joint_oversample_factor is not supported by HUMU pretraining; "
            "use data.objective_sampling.objectives instead"
        )


def _require_non_empty(key: str, dataset: Dataset) -> None:
    if len(dataset) == 0:
        raise ValueError(f"data.{key} contains no HUMU training records")


def _configured_source_dirs(data_cfg: dict, single_key: str, multi_key: str) -> list[str]:
    dirs: list[str] = []

    def add(value) -> None:
        if not value:
            return
        if isinstance(value, dict):
            value = value.get("path")
        if not value:
            return
        text = str(value)
        if text not in dirs:
            dirs.append(text)

    add(data_cfg.get(single_key))
    values = data_cfg.get(multi_key) or []
    if isinstance(values, str | os.PathLike) or isinstance(values, dict):
        add(values)
    else:
        for value in values:
            add(value)
    return dirs


def _pocket_esm2_max_sequence_length(config: dict) -> int | None:
    pocket_cfg = config.get("encoders", {}).get("pocket", {}) or {}
    if not bool(pocket_cfg.get("use_esm2", False)):
        return None
    return _positive_int_or_none(pocket_cfg.get("esm2_max_sequence_length"))


def _max_pocket_points(config: dict) -> int | None:
    data_cfg = config.get("data", {}) or {}
    return _positive_int_or_none(data_cfg.get("max_pocket_points"))


def _positive_int_or_none(value) -> int | None:
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def _pocket_exceeds_esm2_max_sequence_length(
    record: dict,
    load_record,
    max_length: int | None,
) -> bool:
    if max_length is None:
        return False
    if _has_esm2_payload(record):
        return _record_exceeds_esm2_max_sequence_length(record, max_length)
    return _record_exceeds_esm2_max_sequence_length(load_record(record), max_length)


def _joint_exceeds_esm2_max_sequence_length(
    record: dict,
    base_dir: Path,
    pocket_id: str,
    max_length: int | None,
) -> bool:
    if max_length is None:
        return False
    if _has_esm2_payload(record):
        return _record_exceeds_esm2_max_sequence_length(record, max_length)
    pocket = _pocket_payload_from_record(record, base_dir, pocket_id)
    return _record_exceeds_esm2_max_sequence_length(pocket, max_length)


def _has_esm2_payload(record: dict) -> bool:
    if record.get("esm2_embedding") is not None:
        return True
    sequence = record.get("protein_sequence") or record.get("sequence")
    return isinstance(sequence, str) and sequence != ""


def _record_exceeds_esm2_max_sequence_length(record: dict, max_length: int) -> bool:
    if record.get("esm2_embedding") is not None:
        return False
    sequence = record.get("protein_sequence") or record.get("sequence")
    return isinstance(sequence, str) and len(sequence) > max_length


def _dataset_pair_type(dataset: Dataset, index: int) -> str | None:
    if isinstance(dataset, PairedHUMUDataset):
        return dataset.samples[index].get("pair_type")
    if isinstance(dataset, _IndexedDataset):
        return _dataset_pair_type(dataset.dataset, dataset.indices[index])
    if isinstance(dataset, Subset):
        return _dataset_pair_type(dataset.dataset, int(dataset.indices[index]))
    sample = dataset[index]
    return sample.get("pair_type") if isinstance(sample, dict) else None


def _dataset_source_name(dataset: Dataset, index: int) -> str:
    if isinstance(dataset, PairedHUMUDataset):
        sample = dataset.samples[index]
        return _normalise_source_name(
            str(sample.get("source_name") or sample.get("source_dataset") or "unknown")
        )
    if isinstance(dataset, _IndexedDataset):
        return _dataset_source_name(dataset.dataset, dataset.indices[index])
    if isinstance(dataset, Subset):
        return _dataset_source_name(dataset.dataset, int(dataset.indices[index]))
    sample = dataset[index]
    if isinstance(sample, dict):
        return _record_source_name(sample)
    return "unknown"


def _iter_jsonl_records(data_dir: str, max_samples: int | None = None):
    count = 0
    for fpath in Path(data_dir).glob("*.jsonl"):
        with open(fpath) as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)
                count += 1
                if max_samples and count >= max_samples:
                    return


def _source_report(data_dir: str) -> dict:
    path = Path(data_dir)
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "jsonl_files": len(list(path.glob("*.jsonl"))) if path.is_dir() else 0,
    }


def _build_source_registry(data_cfg: dict) -> dict[str, dict]:
    protac_dirs = _configured_source_dirs(data_cfg, "protac_source", "protac_sources")
    registry = {}
    for source_name, config_key in _ALL_HUMU_SOURCE_KEYS.items():
        if source_name == "protacpedia":
            paths = [
                path
                for path in protac_dirs
                if _source_name_from_path(path) == "protacpedia"
            ]
        elif source_name == "activity":
            paths = _configured_source_dirs(data_cfg, "activity_source", "activity_sources")
        elif source_name == "retropath_templates":
            paths = _configured_source_dirs(
                data_cfg,
                "retropath_template_source",
                "retropath_template_sources",
            )
        else:
            value = data_cfg.get(config_key)
            paths = [str(value)] if value else []
        registry[source_name] = {
            "config_key": config_key,
            "paths": paths,
            "configured": bool(paths),
            "trainable": source_name
            in {
                "mol",
                "pocket",
                "route",
                "route_eval",
                "joint",
                "activity",
                "protacpedia",
                "protacdb",
                "protac8k",
                "rcsb_mmcif",
                "interface_skempi2",
                "pdcdb",
                "retropath_templates",
            },
            "objectives": _source_objectives(source_name),
        }
    return registry


def _validate_required_source_registry(registry: dict[str, dict]) -> None:
    for source_name in _ALL_HUMU_SOURCE_KEYS:
        entry = registry[source_name]
        if not entry["configured"]:
            raise FileNotFoundError(
                f"data.{entry['config_key']} is required for HUMU pretraining"
            )
        for path in entry["paths"]:
            _require_data_dir(entry["config_key"], path)


def _source_objectives(source_name: str) -> list[str]:
    return {
        "mol": ["mol_self"],
        "pocket": ["mol_pocket"],
        "route": ["mol_route"],
        "route_eval": ["route_template"],
        "joint": ["mol_pocket_route"],
        "activity": ["activity_pair"],
        "protacpedia": ["protac_component", "protac_ternary"],
        "protacdb": ["protac_component_library"],
        "protac8k": ["protac_ternary"],
        "rcsb_mmcif": ["protein_interface", "protac_ternary"],
        "interface_skempi2": ["interface_mutation", "protein_interface"],
        "pdcdb": ["pdc_component"],
        "retropath_templates": ["route_template"],
    }.get(source_name, [])


def _source_name_from_path(path: str | os.PathLike) -> str:
    return _normalise_source_name(Path(path).name)


def _dataset_manifest_record_count(data_dir: str | os.PathLike) -> int | None:
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    for key in ("n_records", "record_count", "downloaded_valid_mmcif_files", "n_files"):
        if manifest.get(key) is not None:
            return int(manifest[key])
    outputs = manifest.get("outputs")
    if isinstance(outputs, list):
        total = 0
        found = False
        for item in outputs:
            if isinstance(item, dict) and item.get("n_records") is not None:
                total += int(item["n_records"])
                found = True
        if found:
            return total
    components = manifest.get("components")
    if isinstance(components, dict):
        total = 0
        found = False
        for item in components.values():
            if isinstance(item, dict):
                value = item.get("xlsx_valid_smiles_rows", item.get("n_records"))
                if value is not None:
                    total += int(value)
                    found = True
        if found:
            return total
    return None


def _count_jsonl_records(data_dir: str | os.PathLike) -> int:
    count = 0
    for path in Path(data_dir).glob("*.jsonl"):
        if path.name.startswith("reject") or "sdf_index" in path.name:
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
    return count


def _count_iterator_records(records) -> int:
    return sum(1 for _record in records)


def _validate_joint_source(data_dir: str) -> int:
    count = 0
    base_dir = Path(data_dir)
    for record in _iter_jsonl_records(data_dir):
        joint_id = record.get("id")
        ligand_smiles = record.get("ligand_smiles") or record.get("root_smiles")
        pocket_id = record.get("pdb_id") or record.get("pocket_id")
        route_id = record.get("route_id") or joint_id
        if not joint_id:
            raise ValueError("HUMU joint record requires id")
        if not ligand_smiles or not _is_valid_smiles(ligand_smiles):
            raise ValueError(f"HUMU joint record requires valid ligand_smiles: {joint_id}")
        if not pocket_id:
            raise ValueError(f"HUMU joint record requires pocket id: {joint_id}")
        if not route_id:
            raise ValueError(f"HUMU joint record requires route id: {joint_id}")
        _pocket_payload_from_record(record, base_dir, str(pocket_id))
        _route_payload_from_record(record, str(route_id))
        count += 1
    return count


def _validate_activity_source(data_dir: str) -> int:
    manifest_count = _activity_manifest_record_count(data_dir)
    if manifest_count is not None:
        _validate_activity_source_sample(data_dir, manifest_count)
        return manifest_count
    count = 0
    for record in _iter_jsonl_records(data_dir):
        _validate_activity_record(record)
        count += 1
    if count == 0:
        raise ValueError(f"data.activity_source contains no activity records: {data_dir}")
    return count


def _activity_manifest_record_count(data_dir: str) -> int | None:
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    value = manifest.get("n_records", manifest.get("record_count"))
    if value is None:
        return None
    count = int(value)
    if count <= 0:
        raise ValueError(f"data.activity_source contains no activity records: {data_dir}")
    output = manifest.get("output")
    if output and not Path(output).exists():
        raise FileNotFoundError(f"data.activity_source manifest output does not exist: {output}")
    return count


def _validate_activity_source_sample(data_dir: str, manifest_count: int) -> None:
    for record in _iter_jsonl_records(data_dir, max_samples=1):
        _validate_activity_record(record)
        return
    if manifest_count > 0:
        raise ValueError(f"data.activity_source contains no activity records: {data_dir}")


def _validate_activity_record(record: dict, *, validate_smiles: bool = True) -> None:
    smiles = (
        record.get("ligand_smiles")
        or record.get("canonical_smiles")
        or record.get("smiles")
    )
    target_id = record.get("target_id") or record.get("target_chembl_id")
    activity_value = record.get("activity_value", record.get("pchembl_value"))
    if not smiles:
        raise ValueError("HUMU activity record requires valid ligand_smiles")
    if validate_smiles and not _is_valid_smiles(smiles):
        raise ValueError("HUMU activity record requires valid ligand_smiles")
    if not target_id:
        raise ValueError(f"HUMU activity record requires target_id: {smiles}")
    if activity_value is None:
        raise ValueError(f"HUMU activity record requires activity_value: {smiles}")
    try:
        float(activity_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"HUMU activity record requires numeric activity_value: {smiles}") from exc


def _validate_protac_source(data_dir: str) -> int:
    count = 0
    for _record in _iter_protac_component_records(data_dir):
        count += 1
    return count


def _validate_pocket_esm2_source(data_dir: str, esm2_dim: int) -> int:
    manifest_count = _pocket_esm2_manifest_record_count(data_dir)
    if manifest_count is not None:
        _validate_pocket_esm2_source_sample(data_dir, esm2_dim, manifest_count)
        return manifest_count
    dataset = PocketDataset(data_dir)
    count = 0
    for record in dataset.samples:
        pocket = dataset._load_pocket_record(record)
        pocket_id = pocket.get("pdb_id", pocket.get("index", count))
        _validate_pocket_esm2_fields(pocket, str(pocket_id), esm2_dim)
        count += 1
    if count == 0:
        raise ValueError(f"data.pocket_source contains no pocket records: {data_dir}")
    return count


def _pocket_esm2_manifest_record_count(data_dir: str) -> int | None:
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    if not manifest.get("esm2_input"):
        return None
    value = manifest.get("n_records", manifest.get("record_count"))
    if value is None:
        return None
    count = int(value)
    if count <= 0:
        raise ValueError(f"data.pocket_source contains no pocket records: {data_dir}")
    return count


def _validate_pocket_esm2_source_sample(
    data_dir: str,
    esm2_dim: int,
    manifest_count: int,
) -> None:
    dataset = PocketDataset(data_dir, max_samples=1)
    if not dataset.samples and manifest_count > 0:
        raise ValueError(f"data.pocket_source contains no pocket records: {data_dir}")
    for record in dataset.samples[:1]:
        pocket = dataset._load_pocket_record(record)
        pocket_id = pocket.get("pdb_id", pocket.get("index", 0))
        _validate_pocket_esm2_fields(pocket, str(pocket_id), esm2_dim)


def _validate_pocket_esm2_fields(record: dict, record_id: str, esm2_dim: int) -> None:
    embedding = record.get("esm2_embedding")
    if embedding is not None:
        if not isinstance(embedding, list) or len(embedding) != esm2_dim:
            raise ValueError(
                f"Pocket record {record_id} has invalid esm2_embedding length"
            )
        return
    sequence = record.get("protein_sequence") or record.get("sequence")
    if not isinstance(sequence, str) or not sequence:
        raise ValueError(
            "ESM-2 input requires protein_sequence, sequence, or esm2_embedding "
            f"for pocket record {record_id}"
        )


def _pocket_payload_from_record(
    record: dict,
    base_dir: Path,
    pocket_id: str,
    max_points: int | None = None,
) -> dict:
    if {"coords", "elements", "residue_types"}.issubset(record):
        payload = {
            "pdb_id": pocket_id,
            "coords": record["coords"],
            "elements": record["elements"],
            "residue_types": record["residue_types"],
        }
        _copy_pocket_optional_fields(payload, record)
        return _limit_pocket_points(payload, max_points)
    pocket_path = record.get("pocket_path")
    if not pocket_path:
        raise ValueError(f"Joint HUMU record requires pocket_path or inline coords: {record}")
    path = base_dir / str(pocket_path)
    if not path.exists():
        raise FileNotFoundError(f"Pocket coordinate file not found for joint record: {path}")
    with open(path) as handle:
        pocket = json.load(handle)
    atoms = pocket.get("pocket_atoms", [])
    if not atoms:
        raise ValueError(f"Pocket coordinate file has no pocket_atoms: {path}")
    payload = {
        "pdb_id": pocket_id,
        "coords": [[atom["x"], atom["y"], atom["z"]] for atom in atoms],
        "elements": [atom["element"] for atom in atoms],
        "residue_types": [atom["residue"] for atom in atoms],
    }
    _copy_pocket_optional_fields(payload, pocket)
    _copy_pocket_optional_fields(payload, record)
    return _limit_pocket_points(payload, max_points)


def _limit_pocket_points(payload: dict, max_points: int | None) -> dict:
    limit = _positive_int_or_none(max_points)
    coords = payload.get("coords")
    if limit is None or not isinstance(coords, list) or len(coords) <= limit:
        return payload

    indices = _evenly_spaced_indices(len(coords), limit)
    limited = dict(payload)
    for key in _POCKET_POINT_FIELDS:
        values = limited.get(key)
        if isinstance(values, list) and len(values) == len(coords):
            limited[key] = [values[index] for index in indices]
    return limited


def _evenly_spaced_indices(size: int, limit: int) -> list[int]:
    if limit >= size:
        return list(range(size))
    if limit == 1:
        return [0]
    last = size - 1
    return [round(index * last / (limit - 1)) for index in range(limit)]


def _copy_pocket_optional_fields(target: dict, source: dict) -> None:
    for key in (
        "protein_sequence",
        "sequence",
        "esm2_embedding",
        "protein_chains",
        "source_name",
        "source_dataset",
    ):
        value = source.get(key)
        if value is None or value == "":
            continue
        target[key] = value


def _route_payload_from_record(record: dict, route_id: str) -> dict:
    route = dict(record.get("route") or {})
    if not route:
        route = dict(record)
    reactions = route.get("reactions")
    if reactions is None and "reaction_smiles" in route:
        reactions = [route["reaction_smiles"]]
    if not isinstance(reactions, list) or not reactions:
        raise ValueError(f"Joint HUMU record requires route reactions: {route_id}")
    return {
        "id": route_id,
        "source_split": route.get("source_split", route.get("split", "")),
        "root_smiles": route.get("root_smiles", record.get("ligand_smiles", "")),
        "n_steps": route.get("n_steps", route.get("steps", len(reactions))),
        "steps": route.get("steps", route.get("n_steps", len(reactions))),
        "tree_depth": route.get("tree_depth", 1),
        "reaction_types": route.get("reaction_types", []),
        "reactions": reactions,
        "intermediates": route.get("intermediates", []),
        "score": route.get("score", 0.0),
    }


def _iter_protac_component_records(data_dir: str, max_samples: int | None = None):
    base_dir = Path(data_dir)
    count = 0
    protacpedia_path = base_dir / "protacpedia.jsonl"
    if protacpedia_path.exists():
        with open(protacpedia_path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record_id = str(
                    record.get("protacdb_id")
                    or record.get("record_id")
                    or f"protacpedia-{line_number}"
                )
                protac_smiles = _required_valid_smiles(
                    record,
                    ("protac_canonical_smiles", "protac_smiles"),
                    record_id,
                    "protac",
                )
                for component_type, keys in (
                    ("e3_binder", ("e3_binder_canonical_smiles", "e3_binder_smiles")),
                    ("target_ligand", ("ligand_canonical_smiles", "ligand_smiles")),
                    ("linker", ("linker_canonical_smiles", "linker_smiles")),
                ):
                    component_smiles = _optional_valid_smiles(
                        record,
                        keys,
                        record_id,
                        component_type,
                    )
                    if not component_smiles:
                        continue
                    yield {
                        "record_id": record_id,
                        "ligand_smiles": protac_smiles,
                        "component_smiles": component_smiles,
                        "component_type": component_type,
                        "source_dataset": record.get("source") or "PROTACpedia",
                        "source_name": "protacpedia",
                        "target_id": record.get("target"),
                        "split": record.get("split") or "train",
                    }
                    count += 1
                    if max_samples and count >= max_samples:
                        return
    for component_path in _protacdb_component_paths(base_dir):
        component_type = component_path.stem
        with component_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("smiles_valid") is False:
                    continue
                record_id = str(record.get("record_id") or f"{component_type}-{line_number}")
                component_smiles = _optional_valid_smiles(
                    record,
                    ("canonical_smiles", "smiles"),
                    record_id,
                    component_type,
                )
                if not component_smiles:
                    continue
                yield {
                    "record_id": record_id,
                    "pair_type": "protac_component_library",
                    "ligand_smiles": None,
                    "component_smiles": component_smiles,
                    "component_type": str(record.get("component") or component_type),
                    "component_self": True,
                    "source_dataset": record.get("source") or "PROTAC-DB",
                    "source_name": "protacdb",
                    "target_id": record.get("target"),
                    "split": record.get("split") or "train",
                }
                count += 1
                if max_samples and count >= max_samples:
                    return


def _protacdb_component_paths(base_dir: Path) -> list[Path]:
    names = ("e3_ligand", "linker", "warhead", "mg", "xtac")
    return [base_dir / f"{name}.jsonl" for name in names if (base_dir / f"{name}.jsonl").exists()]


def _iter_protac_ternary_records(data_dir: str, max_samples: int | None = None):
    count = 0
    base_dir = Path(data_dir)
    for path in _existing_named_jsonl(base_dir, ("protac8k", "protac_ternary")):
        for record in _iter_jsonl_file(path):
            record_id = str(record.get("record_id") or record.get("id") or f"protac8k:{count}")
            protac_smiles = _required_valid_smiles(
                record,
                ("protac_smiles", "ligand_smiles", "canonical_smiles", "smiles"),
                record_id,
                "protac",
            )
            target_ligand_smiles = _optional_valid_smiles(
                record,
                ("target_ligand_smiles", "target_smiles"),
                record_id,
                "target_ligand",
            )
            e3_ligand_smiles = _optional_valid_smiles(
                record,
                ("e3_ligand_smiles", "e3_smiles", "ligase_ligand_smiles"),
                record_id,
                "e3_ligand",
            )
            target_pocket = record.get("target_pocket")
            e3_pocket = record.get("e3_pocket") or record.get("ligase_pocket")
            if not target_ligand_smiles and not e3_ligand_smiles and not target_pocket and not e3_pocket:
                continue
            yield {
                "record_id": record_id,
                "ligand_smiles": protac_smiles,
                "target_ligand_smiles": target_ligand_smiles,
                "e3_ligand_smiles": e3_ligand_smiles,
                "target_pocket": target_pocket,
                "e3_pocket": e3_pocket,
                "target_id": record.get("target_id") or record.get("target"),
                "source_dataset": record.get("source") or "PROTAC-8K",
                "source_name": "protac8k",
                "split": record.get("split") or "train",
            }
            count += 1
            if max_samples and count >= max_samples:
                return
    for record in _iter_protac8k_feature_records(base_dir, count, max_samples):
        yield record
        count += 1
        if max_samples and count >= max_samples:
            return


def _iter_protac8k_feature_records(
    base_dir: Path,
    start_count: int,
    max_samples: int | None,
):
    matrices = _load_protac8k_feature_matrices(base_dir)
    if matrices is None:
        return
    protac_features, target_features, e3_features = matrices
    n_rows = int(protac_features.shape[0])
    if target_features.shape[0] != n_rows or e3_features.shape[0] != n_rows:
        raise ValueError("PROTAC-8K feature matrices must have identical row counts")
    remaining = None if max_samples is None else max(max_samples - start_count, 0)
    n_records = n_rows if remaining is None else min(n_rows, remaining)
    for row_index in range(n_records):
        yield {
            "record_id": f"protac8k_feature:{row_index}",
            "protac_feature": protac_features[row_index].astype(float).tolist(),
            "target_feature": target_features[row_index].astype(float).tolist(),
            "e3_feature": e3_features[row_index].astype(float).tolist(),
            "source_dataset": "PROTAC-8K",
            "source_name": "protac8k",
            "split": "train",
        }


def _load_protac8k_feature_matrices(base_dir: Path):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for PROTAC-8K feature contracts") from exc

    local_feature_dir = base_dir / "features"
    local_paths = {
        "protac": local_feature_dir / "protac_feature.npy",
        "target": local_feature_dir / "target_feature.npy",
        "e3": local_feature_dir / "e3_feature.npy",
    }
    if all(path.exists() for path in local_paths.values()):
        return (
            np.load(local_paths["protac"], allow_pickle=False),
            np.load(local_paths["target"], allow_pickle=False),
            np.load(local_paths["e3"], allow_pickle=False),
        )

    archive_path = _protac8k_archive_path(base_dir)
    if archive_path is None:
        return None
    with zipfile.ZipFile(archive_path) as archive:
        return (
            _load_npy_from_zip(np, archive, "PROTAC-8K/features/protac_feature.npy"),
            _load_npy_from_zip(np, archive, "PROTAC-8K/features/target_feature.npy"),
            _load_npy_from_zip(np, archive, "PROTAC-8K/features/e3_feature.npy"),
        )


def _protac8k_archive_path(base_dir: Path) -> Path | None:
    direct = base_dir / "PROTAC-8K.zip"
    if direct.exists():
        return direct
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    source = manifest.get("source")
    if not source:
        return None
    archive_path = Path(str(source))
    return archive_path if archive_path.exists() else None


def _load_npy_from_zip(np, archive: zipfile.ZipFile, member: str):
    with archive.open(member) as handle:
        return np.load(io.BytesIO(handle.read()), allow_pickle=False)


def _iter_protein_interface_records(
    data_dir: str,
    *,
    source_name: str,
    max_samples: int | None = None,
):
    count = 0
    base_dir = Path(data_dir)
    paths = _existing_named_jsonl(base_dir, ("interfaces", "protein_interfaces", "structures"))
    for path in paths:
        for record in _iter_jsonl_file(path):
            if not _has_protein_payload_contract(record):
                continue
            record_id = str(
                record.get("record_id")
                or record.get("id")
                or record.get("pdb_id")
                or f"{source_name}:{count}"
            )
            yield {
                **record,
                "record_id": record_id,
                "source_dataset": record.get("source_dataset") or source_name,
                "source_name": source_name,
                "split": record.get("split") or "train",
            }
            count += 1
            if max_samples and count >= max_samples:
                return


def _iter_interface_mutation_records(data_dir: str, max_samples: int | None = None):
    count = 0
    base_dir = Path(data_dir)
    archive_path = _skempi_structure_archive(base_dir)
    for path in _existing_named_jsonl(base_dir, ("interface_mutations", "skempi2")):
        for record in _iter_jsonl_file(path):
            if _affinity_delta(record) is None:
                continue
            if not (
                record.get("wt_interface")
                or record.get("interface_anchor")
                or record.get("pdb_path")
                or archive_path
            ):
                continue
            record_id = str(record.get("record_id") or record.get("id") or f"skempi2:{count}")
            item = {
                **record,
                "record_id": record_id,
                "source_dataset": record.get("source_dataset") or "SKEMPI2",
                "source_name": "interface_skempi2",
                "split": record.get("split") or "train",
            }
            if archive_path and not item.get("pdb_archive_path"):
                item["pdb_archive_path"] = str(archive_path)
            if not _has_explicit_interface_mutation_views(item):
                mutation = item.get("mutations_cleaned") or item.get("mutations_pdb")
                if _parse_skempi_mutations(mutation) is None:
                    continue
            yield item
            count += 1
            if max_samples and count >= max_samples:
                return


def _iter_pdc_component_records(data_dir: str, max_samples: int | None = None):
    count = 0
    base_dir = Path(data_dir)
    for path in _existing_named_jsonl(base_dir, ("pdc_components", "pdc_component")):
        for record in _iter_jsonl_file(path):
            record_id = str(record.get("record_id") or record.get("PDC_ID") or f"pdcdb:{count}")
            normalised = _normalise_pdc_component_record(record, record_id)
            if normalised is None:
                continue
            yield normalised
            count += 1
            if max_samples and count >= max_samples:
                return
    if count:
        return
    for record in _pdc_manifest_component_records(base_dir):
        if max_samples and count >= max_samples:
            return
        yield record
        count += 1
    if count:
        return
    linker_smiles_by_id = _pdc_linker_smiles_by_id(base_dir)
    if not linker_smiles_by_id:
        return
    pdc_smiles_by_cid = _pdc_manifest_smiles_by_pubchem_cid(base_dir)
    for path in _existing_named_jsonl(base_dir, ("pdc",)):
        for record in _iter_jsonl_file(path):
            record_id = str(record.get("PDC_ID") or record.get("record_id") or f"pdcdb:{count}")
            linker_id = str(record.get("Linker_ID") or record.get("linker_id") or "")
            component_smiles = linker_smiles_by_id.get(linker_id)
            if not component_smiles:
                continue
            peptide_sequence = _pdc_peptide_sequence(record)
            if peptide_sequence:
                yield {
                    "record_id": record_id,
                    "component_smiles": component_smiles,
                    "component_type": "linker",
                    "peptide_sequence": peptide_sequence,
                    "peptide_id": record.get("Peptide_ID"),
                    "target_id": record.get("receptor_id"),
                    "source_dataset": record.get("source") or "PDCdb",
                    "source_name": "pdcdb",
                    "split": record.get("split") or "train",
                }
                count += 1
                if max_samples and count >= max_samples:
                    return
                continue
            pdc_cid = str(record.get("Puchem_CID") or record.get("PubChem_CID") or "")
            pdc_smiles = pdc_smiles_by_cid.get(pdc_cid)
            if not pdc_smiles:
                continue
            yield {
                "record_id": record_id,
                "ligand_smiles": pdc_smiles,
                "component_smiles": component_smiles,
                "component_type": "linker",
                "peptide_id": record.get("Peptide_ID"),
                "target_id": record.get("receptor_id"),
                "source_dataset": record.get("source") or "PDCdb",
                "source_name": "pdcdb",
                "split": record.get("split") or "train",
            }
            count += 1
            if max_samples and count >= max_samples:
                return


def _normalise_pdc_component_record(record: dict, record_id: str) -> dict | None:
    component_smiles = _optional_valid_smiles(
        record,
        ("component_smiles", "linker_smiles", "payload_smiles", "canonical_smiles", "smiles"),
        record_id,
        "pdc_component",
    )
    if not component_smiles:
        return None
    ligand_smiles = _optional_valid_smiles(
        record,
        ("ligand_smiles", "pdc_smiles", "conjugate_smiles"),
        record_id,
        "pdc_ligand",
    )
    peptide_pocket = record.get("peptide_pocket")
    peptide_sequence = record.get("peptide_sequence") or record.get("Peptide_Sequence")
    has_explicit_sequence = (
        _looks_like_peptide_sequence(peptide_sequence)
        and (
            peptide_pocket
            or record.get("sequence_source") in {"explicit", "peptide_sequence", "Peptide_Sequence"}
            or record.get("peptide_sequence_source") in {"explicit", "peptide_sequence", "Peptide_Sequence"}
            or record.get("Peptide_Sequence") is not None
        )
    )
    if not ligand_smiles and not peptide_pocket and not has_explicit_sequence:
        return None
    output = {
        "record_id": record_id,
        "component_smiles": component_smiles,
        "component_type": record.get("component_type") or "pdc_component",
        "peptide_id": record.get("peptide_id") or record.get("Peptide_ID"),
        "target_id": record.get("target_id") or record.get("receptor_id"),
        "linker_id": record.get("linker_id") or record.get("Linker_ID"),
        "source_dataset": record.get("source_dataset") or record.get("source") or "PDCdb",
        "source_name": "pdcdb",
        "split": record.get("split") or "train",
    }
    if ligand_smiles:
        output["ligand_smiles"] = ligand_smiles
    if peptide_pocket or has_explicit_sequence:
        output["peptide_sequence"] = str(peptide_sequence).strip().upper() if peptide_sequence else None
        output["peptide_pocket"] = peptide_pocket
    return output


def _pdc_manifest_component_records(base_dir: Path) -> list[dict]:
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = manifest.get("pdc_component_records") or []
    if not isinstance(records, list):
        return []
    output = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        record_id = str(record.get("record_id") or record.get("PDC_ID") or f"pdcdb_manifest:{index}")
        normalised = _normalise_pdc_component_record(record, record_id)
        if normalised is not None:
            output.append(normalised)
    return output


def _pdc_linker_smiles_by_id(base_dir: Path) -> dict[str, str]:
    smiles_by_id: dict[str, str] = _pdc_manifest_linker_smiles_by_id(base_dir)
    for path in _existing_named_jsonl(base_dir, ("linker", "linkers")):
        for record in _iter_jsonl_file(path):
            linker_id = str(record.get("Linker_ID") or record.get("linker_id") or "")
            if not linker_id:
                continue
            smiles = _optional_valid_smiles(
                record,
                ("linker_smiles", "component_smiles", "canonical_smiles", "smiles"),
                linker_id,
                "pdc_linker",
            )
            if smiles:
                smiles_by_id[linker_id] = smiles
    return smiles_by_id


def _pdc_manifest_linker_smiles_by_id(base_dir: Path) -> dict[str, str]:
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    cached = manifest.get("linker_smiles_by_id") or {}
    if not isinstance(cached, dict):
        return {}
    smiles_by_id: dict[str, str] = {}
    for linker_id, smiles in cached.items():
        record = {"smiles": smiles}
        value = _optional_valid_smiles(
            record,
            ("smiles",),
            str(linker_id),
            "pdc_linker",
        )
        if value:
            smiles_by_id[str(linker_id)] = value
    return smiles_by_id


def _pdc_manifest_smiles_by_pubchem_cid(base_dir: Path) -> dict[str, str]:
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    cached = manifest.get("pdc_smiles_by_pubchem_cid") or {}
    if not isinstance(cached, dict):
        return {}
    smiles_by_cid: dict[str, str] = {}
    for cid, smiles in cached.items():
        record = {"smiles": smiles}
        value = _optional_valid_smiles(
            record,
            ("smiles",),
            str(cid),
            "pdc_ligand",
        )
        if value:
            smiles_by_cid[str(cid)] = value
    return smiles_by_cid


def _pdc_peptide_sequence(record: dict) -> str | None:
    for key in ("peptide_sequence", "Peptide_Sequence"):
        value = record.get(key)
        if _looks_like_peptide_sequence(value):
            return str(value).strip().upper()
    return None


def _existing_named_jsonl(base_dir: Path, stems: tuple[str, ...]) -> list[Path]:
    return [
        base_dir / f"{stem}.jsonl"
        for stem in stems
        if (base_dir / f"{stem}.jsonl").exists()
    ]


def _iter_jsonl_file(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _has_protein_payload_contract(record: dict) -> bool:
    if any(record.get(key) for key in ("interface", "interface_anchor", "coords")):
        return True
    return bool(record.get("mmcif_path") or record.get("pdb_path"))


def _protein_payload_from_record(record: dict, key: str, record_id: str) -> dict | None:
    payload = record.get(key)
    if payload is None:
        return None
    if isinstance(payload, dict):
        return _validate_protein_payload(payload, record_id)
    if isinstance(payload, str):
        return _protein_payload_from_path(Path(payload), record_id)
    raise ValueError(f"HUMU protein payload {key} must be a dict or path: {record_id}")


def _protein_interface_views(record: dict) -> tuple[dict, dict]:
    record_id = str(record.get("record_id") or record.get("pdb_id") or "protein_interface")
    source_name = record.get("source_name") or record.get("source_dataset")
    anchor = (
        _protein_payload_from_record(record, "interface_anchor", record_id)
        or _protein_payload_from_record(record, "interface", record_id)
    )
    positive = (
        _protein_payload_from_record(record, "interface_positive", record_id)
        or _protein_payload_from_record(record, "interface", record_id)
    )
    if anchor is not None and positive is not None:
        _set_payload_source_name(anchor, source_name)
        _set_payload_source_name(positive, source_name)
        return anchor, positive
    payload = _structure_payload_from_record(record, record_id)
    _set_payload_source_name(payload, source_name)
    return _protein_payload_view(payload, 0), _protein_payload_view(payload, 1)


def _interface_mutation_views(record: dict) -> tuple[dict, dict]:
    record_id = str(record.get("record_id") or record.get("id") or "interface_mutation")
    source_name = record.get("source_name") or record.get("source_dataset")
    anchor = (
        _protein_payload_from_record(record, "wt_interface", record_id)
        or _protein_payload_from_record(record, "interface_anchor", record_id)
    )
    positive = (
        _protein_payload_from_record(record, "mut_interface", record_id)
        or _protein_payload_from_record(record, "interface_positive", record_id)
    )
    if anchor is not None and positive is not None:
        _set_payload_source_name(anchor, source_name)
        _set_payload_source_name(positive, source_name)
        return anchor, positive
    payload = _structure_payload_from_record(record, record_id)
    _set_payload_source_name(payload, source_name)
    mutations = _parse_skempi_mutations(
        record.get("mutations_cleaned") or record.get("mutations_pdb")
    )
    if mutations is None:
        raise ValueError(f"SKEMPI2 record requires parseable mutation: {record_id}")
    return payload, _mutated_residue_payload(payload, mutations)


def _set_payload_source_name(payload: dict | None, source_name) -> None:
    if payload is not None and source_name and not payload.get("source_name"):
        payload["source_name"] = str(source_name)


def _pdc_peptide_payload(record: dict) -> dict:
    record_id = str(record.get("record_id") or "pdc_component")
    payload = record.get("peptide_pocket")
    if isinstance(payload, dict):
        return _validate_protein_payload(payload, record_id)
    sequence = record.get("peptide_sequence")
    if not _looks_like_peptide_sequence(sequence):
        raise ValueError(f"PDC component record requires peptide_pocket or peptide_sequence: {record_id}")
    return _sequence_pocket(str(sequence), record_id)


def _has_explicit_interface_mutation_views(record: dict) -> bool:
    return bool(
        (record.get("wt_interface") or record.get("interface_anchor"))
        and (record.get("mut_interface") or record.get("interface_positive"))
    )


def _structure_payload_from_record(record: dict, record_id: str) -> dict:
    if {"coords", "elements", "residue_types"}.issubset(record):
        return _validate_protein_payload(record, record_id)
    if record.get("mmcif_path"):
        return _protein_payload_from_path(Path(str(record["mmcif_path"])), record_id)
    if record.get("pdb_path"):
        return _protein_payload_from_path(Path(str(record["pdb_path"])), record_id)
    archive_path = record.get("pdb_archive_path")
    if archive_path:
        pdb_id = str(record.get("pdb_id") or str(record.get("pdb_complex", "")).split("_", 1)[0])
        return _protein_payload_from_skempi_archive(Path(str(archive_path)), pdb_id, record_id)
    raise ValueError(f"Protein interface record has no structure payload: {record_id}")


def _protein_payload_from_path(path: Path, record_id: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Protein structure path does not exist: {path}")
    if path.suffix == ".gz" or path.name.endswith(".cif.gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    if path.name.endswith(".cif") or path.name.endswith(".cif.gz"):
        return _parse_mmcif_payload(text, record_id)
    return _parse_pdb_payload(text, record_id)


@lru_cache(maxsize=4096)
def _protein_payload_from_skempi_archive(archive_path: Path, pdb_id: str, record_id: str) -> dict:
    if not archive_path.exists():
        raise FileNotFoundError(f"SKEMPI2 structure archive does not exist: {archive_path}")
    member_name = f"PDBs/{pdb_id.upper()}.pdb"
    with tarfile.open(archive_path, "r:gz") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError as exc:
            raise FileNotFoundError(f"SKEMPI2 archive has no member {member_name}") from exc
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"SKEMPI2 archive member is not readable: {member_name}")
        text = extracted.read().decode("utf-8", errors="ignore")
    return _parse_pdb_payload(text, record_id)


def _parse_mmcif_payload(text: str, record_id: str, max_atoms: int = 512) -> dict:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        field_index = index + 1
        fields = []
        while field_index < len(lines) and lines[field_index].strip().startswith("_atom_site."):
            fields.append(lines[field_index].strip())
            field_index += 1
        if not fields:
            continue
        names = [field.split(".", 1)[1] for field in fields]
        required = {"Cartn_x", "Cartn_y", "Cartn_z"}
        if not required.issubset(names):
            continue
        return _atom_site_payload(lines[field_index:], names, record_id, max_atoms)
    raise ValueError(f"mmCIF structure has no atom_site coordinates: {record_id}")


def _atom_site_payload(rows: list[str], names: list[str], record_id: str, max_atoms: int) -> dict:
    idx = {name: names.index(name) for name in names}
    coords = []
    elements = []
    residues = []
    atom_chain_ids = []
    residue_ids = []
    for raw in rows:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("_") or stripped == "loop_":
            break
        try:
            values = shlex.split(stripped)
        except ValueError:
            continue
        if len(values) < len(names):
            continue
        try:
            x = float(values[idx["Cartn_x"]])
            y = float(values[idx["Cartn_y"]])
            z = float(values[idx["Cartn_z"]])
        except (KeyError, ValueError):
            continue
        coords.append([x, y, z])
        elements.append(values[idx.get("type_symbol", 0)].upper())
        residue_key = "label_comp_id" if "label_comp_id" in idx else "auth_comp_id"
        residues.append(values[idx.get(residue_key, 0)].upper())
        chain_key = "auth_asym_id" if "auth_asym_id" in idx else "label_asym_id"
        residue_id_key = "auth_seq_id" if "auth_seq_id" in idx else "label_seq_id"
        atom_chain_ids.append(values[idx.get(chain_key, 0)])
        residue_ids.append(values[idx.get(residue_id_key, 0)])
        if len(coords) >= max_atoms:
            break
    return _validate_protein_payload(
        {
            "coords": coords,
            "elements": elements,
            "residue_types": residues,
            "atom_chain_ids": atom_chain_ids,
            "residue_ids": residue_ids,
        },
        record_id,
    )


def _parse_pdb_payload(text: str, record_id: str, max_atoms: int = 512) -> dict:
    coords = []
    elements = []
    residues = []
    atom_chain_ids = []
    residue_ids = []
    for line in io.StringIO(text):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        residue = line[17:20].strip().upper() or "UNK"
        element = line[76:78].strip().upper() or line[12:14].strip()[0].upper()
        residues.append(residue)
        elements.append(element)
        atom_chain_ids.append(line[21].strip() or "")
        residue_ids.append(line[22:26].strip())
        if len(coords) >= max_atoms:
            break
    return _validate_protein_payload(
        {
            "coords": coords,
            "elements": elements,
            "residue_types": residues,
            "atom_chain_ids": atom_chain_ids,
            "residue_ids": residue_ids,
        },
        record_id,
    )


def _validate_protein_payload(payload: dict, record_id: str) -> dict:
    coords = payload.get("coords")
    elements = payload.get("elements")
    residues = payload.get("residue_types")
    if not isinstance(coords, list) or not coords:
        raise ValueError(f"Protein payload requires coords: {record_id}")
    if not isinstance(elements, list) or len(elements) != len(coords):
        raise ValueError(f"Protein payload requires one element per coordinate: {record_id}")
    if not isinstance(residues, list) or len(residues) != len(coords):
        raise ValueError(f"Protein payload requires one residue per coordinate: {record_id}")
    output = dict(payload)
    output["coords"] = coords
    output["elements"] = [str(element).upper() for element in elements]
    output["residue_types"] = [str(residue).upper() for residue in residues]
    return output


def _protein_payload_view(payload: dict, offset: int) -> dict:
    indices = list(range(offset, len(payload["coords"]), 2))
    if not indices:
        indices = list(range(len(payload["coords"])))
    view = dict(payload)
    for key in ("coords", "elements", "residue_types", "atom_chain_ids", "residue_ids"):
        values = payload.get(key)
        if isinstance(values, list) and len(values) == len(payload["coords"]):
            view[key] = [values[index] for index in indices]
    return view


def _mutated_residue_payload(payload: dict, mutations) -> dict:
    output = dict(payload)
    output["residue_types"] = list(payload["residue_types"])
    chain_ids = payload.get("atom_chain_ids") or [""] * len(output["residue_types"])
    residue_ids = payload.get("residue_ids") or [""] * len(output["residue_types"])
    for mutation in _normalise_skempi_mutation_list(mutations):
        chain_id = str(mutation["chain_id"])
        residue_number = int(mutation["residue_number"])
        residue_name = _AA1_TO_AA3.get(str(mutation["mutant"]).upper(), "UNK")
        for index, (chain, residue_id) in enumerate(zip(chain_ids, residue_ids, strict=False)):
            if str(chain) == chain_id and _residue_number(residue_id) == residue_number:
                output["residue_types"][index] = residue_name
    return output


def _normalise_skempi_mutation_list(mutations) -> list[dict]:
    if isinstance(mutations, tuple) and len(mutations) == 3:
        chain_id, residue_number, mutant = mutations
        return [
            {
                "wildtype": None,
                "chain_id": chain_id,
                "residue_number": residue_number,
                "mutant": mutant,
            }
        ]
    if isinstance(mutations, dict):
        return [mutations]
    return list(mutations)


def _parse_skempi_mutations(value) -> list[dict] | None:
    if not isinstance(value, str):
        return None
    mutations = []
    for token in re.split(r"[,;\\s]+", value.strip()):
        if not token:
            continue
        match = re.match(r"^([A-Za-z])([A-Za-z])(-?\d+)([A-Za-z])$", token)
        if not match:
            return None
        mutations.append(
            {
                "wildtype": match.group(1).upper(),
                "chain_id": match.group(2),
                "residue_number": int(match.group(3)),
                "mutant": match.group(4).upper(),
            }
        )
    if not mutations:
        return None
    return mutations


def _parse_skempi_mutation(value) -> tuple[str, int, str] | None:
    mutations = _parse_skempi_mutations(value)
    if not mutations or len(mutations) != 1:
        return None
    mutation = mutations[0]
    return mutation["chain_id"], mutation["residue_number"], mutation["mutant"]


def _residue_number(value) -> int | None:
    match = re.search(r"-?\d+", str(value))
    return int(match.group(0)) if match else None


def _affinity_delta(record: dict) -> float | None:
    wt = record.get("affinity_wt_m")
    mut = record.get("affinity_mut_m")
    try:
        wt_value = float(wt)
        mut_value = float(mut)
    except (TypeError, ValueError):
        return None
    if wt_value <= 0 or mut_value <= 0:
        return None
    return abs(math.log10(mut_value) - math.log10(wt_value))


def _skempi_structure_archive(base_dir: Path) -> Path | None:
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    value = manifest.get("structure_archive")
    return Path(value) if value else None


def _looks_like_peptide_sequence(value) -> bool:
    if not isinstance(value, str):
        return False
    sequence = value.strip().upper()
    if "PEPTIDE" in sequence:
        return False
    return len(sequence) >= 2 and all(char in _AA1_TO_AA3 for char in sequence)


def _sequence_pocket(sequence: str, record_id: str) -> dict:
    sequence = sequence.strip().upper()
    if not _looks_like_peptide_sequence(sequence):
        raise ValueError(f"Invalid peptide sequence: {record_id}")
    coords = [[float(index) * 3.8, 0.0, 0.0] for index, _aa in enumerate(sequence)]
    return {
        "coords": coords,
        "elements": ["C"] * len(sequence),
        "residue_types": [_AA1_TO_AA3.get(aa, "UNK") for aa in sequence],
        "protein_sequence": sequence,
    }


_AA1_TO_AA3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


def _iter_activity_pair_records(data_dir: str, max_samples: int | None = None):
    pending_by_target: dict[str, dict] = {}
    source_name = _source_name_from_path(data_dir)
    count = 0
    for record in _iter_jsonl_records(data_dir):
        _validate_activity_record(record, validate_smiles=False)
        item = _activity_pair_item(record)
        target_id = item["target_id"]
        previous = pending_by_target.pop(target_id, None)
        if previous is None:
            pending_by_target[target_id] = item
            continue
        yield {
            "record_id": f"{source_name}:{target_id}:{count}",
            "ligand_smiles": previous["ligand_smiles"],
            "positive_smiles": item["ligand_smiles"],
            "activity_value": previous["activity_value"],
            "positive_activity_value": item["activity_value"],
            "activity_delta": abs(previous["activity_value"] - item["activity_value"]),
            "target_id": target_id,
            "source_dataset": source_name,
            "source_name": source_name,
        }
        count += 1
        if max_samples and count >= max_samples:
            return


def _activity_pair_item(record: dict) -> dict:
    smiles = (
        record.get("ligand_smiles")
        or record.get("canonical_smiles")
        or record.get("smiles")
    )
    target_id = record.get("target_id") or record.get("target_chembl_id")
    activity_value = record.get("activity_value", record.get("pchembl_value"))
    return {
        "ligand_smiles": str(smiles),
        "target_id": str(target_id),
        "activity_value": float(activity_value),
    }


def _iter_route_template_records(data_dir: str, max_samples: int | None = None):
    count = 0
    for record in _iter_jsonl_records(data_dir):
        if record.get("valid") is False:
            continue
        template = record.get("template") or record.get("reaction_smiles")
        if not isinstance(template, str) or ">>" not in template:
            continue
        record_id = str(record.get("template_id") or f"route_template:{count}")
        yield {
            "record_id": record_id,
            "ligand_smiles": None,
            "source_dataset": record.get("source_dataset") or "retropath_templates",
            "source_name": "retropath_templates",
            "route_id": record_id,
                "route": {
                    "id": record_id,
                    "root_smiles": "",
                    "template": template,
                    "reactions": [template],
                "steps": 1,
                "tree_depth": 1,
                "reaction_types": ["template"],
                "intermediates": [],
                "score": float(record.get("reactions_count", 1) or 1),
            },
        }
        count += 1
        if max_samples and count >= max_samples:
            return


def _iter_route_eval_template_records(data_dir: str, max_samples: int | None = None):
    count = 0
    for record in _iter_jsonl_records(data_dir):
        route = _route_payload_from_record(record, str(record.get("id") or f"route_eval:{count}"))
        product_smiles = route.get("root_smiles")
        if not product_smiles or not _is_valid_smiles(product_smiles):
            continue
        record_id = str(route["id"])
        yield {
            "record_id": record_id,
            "ligand_smiles": product_smiles,
            "source_dataset": "route_eval",
            "source_name": "route_eval",
            "route_id": record_id,
            "route": route,
        }
        count += 1
        if max_samples and count >= max_samples:
            return


def _product_smiles_from_reaction_like(reaction: str) -> str | None:
    product = reaction.split(">>", 1)[1].split(".", 1)[0].strip()
    if not product or not _is_valid_smiles(product):
        return None
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for HUMU route template encoding") from exc
    mol = Chem.MolFromSmiles(product, sanitize=False)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _required_valid_smiles(
    record: dict,
    keys: tuple[str, ...],
    record_id: str,
    label: str,
) -> str:
    smiles = _optional_valid_smiles(record, keys, record_id, label)
    if not smiles:
        raise ValueError(f"HUMU PROTAC record {record_id} requires {label} SMILES")
    return smiles


def _optional_valid_smiles(
    record: dict,
    keys: tuple[str, ...],
    record_id: str,
    label: str,
) -> str | None:
    for key in keys:
        smiles = record.get(key)
        if smiles is None or smiles == "":
            continue
        smiles = str(smiles)
        if smiles.strip().lower() in {"none", "null", "nan", "na", "n/a", "."}:
            continue
        valid_flag = record.get(f"{key}_valid")
        if valid_flag is False:
            continue
        if not _is_valid_smiles(smiles):
            continue
        return smiles
    return None


@lru_cache(maxsize=262_144)
def _is_valid_smiles(smiles: str) -> bool:
    try:
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise RuntimeError("RDKit is required for HUMU paired SMILES validation") from exc

    if not isinstance(smiles, str) or not smiles.strip():
        return False
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False
        sanitize_ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        try:
            Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops)
        except Exception:  # noqa: BLE001
            return False
    return mol is not None and mol.GetNumAtoms() > 0


def _record_collate(records: list[dict]) -> dict:
    if not records:
        raise ValueError("HUMU paired batch contains no records")
    invalid_records = [
        record
        for record in records
        if record.get("pair_type") in _MOL_REQUIRED_PAIR_TYPES
        and not _is_valid_smiles(record.get("ligand_smiles", ""))
    ]
    invalid_ternary_records = [
        record
        for record in records
        if record.get("pair_type") == "protac_ternary"
        and not (
            _is_valid_smiles(record.get("ligand_smiles", ""))
            or _has_protac8k_feature_payload(record)
        )
    ]
    if invalid_records:
        sample = invalid_records[0]
        raise ValueError(
            "HUMU paired batch contains invalid ligand_smiles "
            f"mol_id={sample.get('mol_id', '')} "
            f"pair_type={sample.get('pair_type', '')}"
        )
    if invalid_ternary_records:
        sample = invalid_ternary_records[0]
        raise ValueError(
            "HUMU protac_ternary batch requires valid ligand_smiles or "
            "protac_feature/target_feature/e3_feature "
            f"mol_id={sample.get('mol_id', '')}"
        )
    invalid_pdc_records = [
        record
        for record in records
        if record.get("pair_type") == "pdc_component"
        and not (
            _is_valid_smiles(record.get("ligand_smiles", ""))
            or record.get("peptide_pocket") is not None
            or _looks_like_peptide_sequence(record.get("peptide_sequence"))
        )
    ]
    if invalid_pdc_records:
        sample = invalid_pdc_records[0]
        raise ValueError(
            "HUMU pdc_component batch requires valid ligand_smiles or "
            "peptide_pocket/peptide_sequence "
            f"mol_id={sample.get('mol_id', '')}"
        )
    invalid_components = [
        record
        for record in records
        if record.get("component_smiles") is not None
        and not _is_valid_smiles(record.get("component_smiles", ""))
    ]
    if invalid_components:
        sample = invalid_components[0]
        raise ValueError(
            "HUMU paired batch contains invalid component_smiles "
            f"mol_id={sample.get('mol_id', '')} "
            f"pair_type={sample.get('pair_type', '')}"
        )
    invalid_positive_smiles = [
        record
        for record in records
        if record.get("positive_smiles") is not None
        and not _is_valid_smiles(record.get("positive_smiles", ""))
    ]
    if invalid_positive_smiles:
        sample = invalid_positive_smiles[0]
        raise ValueError(
            "HUMU paired batch contains invalid positive_smiles "
            f"mol_id={sample.get('mol_id', '')} "
            f"pair_type={sample.get('pair_type', '')}"
        )
    keys = set().union(*(record.keys() for record in records))
    batch = {key: [record.get(key) for record in records] for key in keys}
    pair_type_counts = Counter(str(record.get("pair_type", "")) for record in records)
    source_names = [_record_source_name(record) for record in records]
    source_counts = Counter(source_names)
    for source_name in _ALL_HUMU_SOURCE_KEYS:
        source_counts.setdefault(source_name, 0)
    unique_source_coverage = len(set(source_names)) / float(len(source_names))
    batch["pair_type_counts"] = dict(pair_type_counts)
    batch["source_counts"] = dict(source_counts)
    batch["unique_source_coverage"] = unique_source_coverage
    batch["source_repeat_rate"] = 1.0 - unique_source_coverage
    return batch


def _has_protac8k_feature_payload(record: dict) -> bool:
    return all(
        isinstance(record.get(key), list | tuple) and len(record.get(key)) > 0
        for key in ("protac_feature", "target_feature", "e3_feature")
    )


def _record_source_name(record: dict) -> str:
    value = record.get("source_name") or record.get("source_dataset") or "unknown"
    return _normalise_source_name(str(value))


def _normalise_source_name(value: str) -> str:
    lowered = value.strip().lower().replace("-", "_")
    if "protacpedia" in lowered:
        return "protacpedia"
    if "protac_db" in lowered or "protacdb" in lowered:
        return "protacdb"
    if "bindingdb" in lowered:
        return "bindingdb_activity"
    if "retropath" in lowered:
        return "retropath_templates"
    if "route_eval" in lowered:
        return "route_eval"
    if "crossdocked" in lowered or "pdbbind" in lowered:
        return "pocket"
    if "uspto" in lowered:
        return "route"
    return lowered or "unknown"


def _extract_reactions_from_tree(sample: dict) -> list[str]:
    reactions: list[str] = []

    def visit(node: dict) -> None:
        reaction = node.get("reaction_smiles")
        if isinstance(reaction, str) and ">>" in reaction:
            reactions.append(reaction)
        for child in node.get("children", []):
            if isinstance(child, dict):
                visit(child)

    if "trees" in sample:
        for tree in sample["trees"]:
            if isinstance(tree, dict):
                visit(tree)
    else:
        visit(sample)
    return reactions
