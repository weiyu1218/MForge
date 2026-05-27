"""Data loaders for HUMU pretraining — ChEMBL molecules, CrossDocked pockets, USPTO-MIT routes."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


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

    def __init__(self, data_dir: str, max_samples: int | None = None):
        self.samples: list[dict] = []
        self.data_dir = Path(data_dir)
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
            return record

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
        return loaded


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


class IntentDataset:
    """Loads CIG intent features and indexes them by explicit sample keys."""

    def __init__(self, data_dir: str):
        self.samples = list(_iter_jsonl_records(data_dir))
        if not self.samples:
            raise ValueError(f"data.intent_source contains no intent records: {data_dir}")
        self._index: dict[tuple[str, str], dict] = {}
        for record in self.samples:
            intent = self._normalize_intent(record)
            for key in ("mol_id", "target_id", "ligand_smiles", "source_dataset", "id"):
                value = record.get(key)
                if value is None or value == "":
                    continue
                index_key = (key, str(value))
                if index_key in self._index:
                    raise ValueError(f"Duplicate HUMU intent key: {key}={value}")
                self._index[index_key] = intent

    def match(self, sample: dict) -> dict | None:
        for key in ("mol_id", "target_id", "ligand_smiles", "source_dataset"):
            value = sample.get(key)
            if value is not None and value != "":
                found = self._index.get((key, str(value)))
                if found is not None:
                    return found
        return None

    def _normalize_intent(self, record: dict) -> dict:
        if "targets" not in record and "objective_nodes" not in record:
            raise ValueError("HUMU intent record requires targets or objective_nodes")
        return {
            "targets": record.get("targets", {}),
            "weights": record.get("weights", {}),
            "constraints": record.get("constraints", {}),
            "objective_nodes": record.get("objective_nodes", []),
            "edges": record.get("edges", []),
            "intent_id": record.get("intent_id", record.get("id", "")),
            "target_id": record.get("target_id", ""),
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
        joint_dir: str | None = None,
        intent_dir: str | None = None,
        require_intent: bool = False,
        require_pocket_route: bool = False,
        pocket_esm2_max_sequence_length: int | None = None,
    ):
        self.pockets = PocketDataset(pocket_dir, max_samples=max_samples)
        self.routes = RouteDataset(route_dir, max_samples=max_samples)
        self.joint_dir = Path(joint_dir) if joint_dir else None
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
        self.intent_dataset = IntentDataset(intent_dir) if intent_dir else None
        self.samples: list[dict] = []
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
                    "split": pocket.get("split") or "train",
                    "_source_kind": "pocket",
                    "_source_index": pocket_index,
                }
            )
            self._attach_intent(self.samples[-1], require_intent)
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
                    "split": route.get("source_split") or "train",
                    "_source_kind": "route",
                    "_source_index": route_index,
                }
            )
            self._attach_intent(self.samples[-1], require_intent)
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
                    "split": record.get("split") or "train",
                    "_source_kind": "joint",
                    "_source_index": joint_index,
                }
            )
            self._attach_intent(self.samples[-1], require_intent)
        if require_pocket_route and not any(
            sample["pair_type"] == "mol_pocket_route" for sample in self.samples
        ):
            raise ValueError("data.joint_source must contain usable mol-pocket-route records")

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
            }
            _copy_pocket_optional_fields(sample["pocket"], pocket)
            sample["route"] = None
            return sample

        if source_kind == "route":
            sample["pocket"] = None
            sample["route"] = self.routes[source_index]
            return sample

        record = self.joint_records[source_index]
        sample["pocket"] = _pocket_payload_from_record(
            record,
            self.joint_dir or Path("."),
            sample["pocket_id"],
        )
        sample["route"] = _route_payload_from_record(record, sample["route_id"])
        return sample

    def _attach_intent(self, sample: dict, require_intent: bool) -> None:
        if self.intent_dataset is None:
            if require_intent:
                raise FileNotFoundError(
                    "data.intent_source is required when intent loss is enabled"
                )
            sample["intent"] = None
            return
        intent = self.intent_dataset.match(sample)
        if intent is None and require_intent:
            raise ValueError(
                "No HUMU intent record matched sample "
                f"mol_id={sample.get('mol_id')} target_id={sample.get('target_id')}"
            )
        sample["intent"] = intent


def preflight_humu_data_contract(config: dict) -> dict:
    """Validate HUMU data contracts before starting a training process."""
    data_cfg = config.get("data", {})
    loss_weights = config.get("loss_weights", {})
    pocket_encoder_cfg = config.get("encoders", {}).get("pocket", {})
    require_pocket_route = float(loss_weights.get("pocket_route", 0.0) or 0.0) > 0.0
    require_intent = float(loss_weights.get("intent", 0.0) or 0.0) > 0.0
    require_pocket_esm2 = bool(pocket_encoder_cfg.get("use_esm2", False))
    esm2_dim = int(pocket_encoder_cfg.get("esm2_dim", 1280))

    pocket_dir = data_cfg.get("pocket_source", "")
    route_dir = data_cfg.get("route_source", "")
    joint_dir = data_cfg.get("joint_source")
    intent_dir = data_cfg.get("intent_source")
    activity_dir = data_cfg.get("activity_source")

    _require_data_dir("pocket_source", pocket_dir)
    _require_data_dir("route_source", route_dir)

    report = {
        "required": {
            "joint_source": require_pocket_route,
            "intent_source": require_intent,
        },
        "sources": {
            "pocket_source": _source_report(pocket_dir),
            "route_source": _source_report(route_dir),
        },
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

    if require_intent:
        _require_data_dir("intent_source", intent_dir or "")
        intent_records = _validate_intent_source(intent_dir)
        if intent_records == 0:
            raise ValueError(f"data.intent_source contains no intent records: {intent_dir}")
        report["sources"]["intent_source"] = {
            **_source_report(intent_dir),
            "records": intent_records,
        }
    elif intent_dir:
        _require_data_dir("intent_source", intent_dir)
        report["sources"]["intent_source"] = {
            **_source_report(intent_dir),
            "records": _validate_intent_source(intent_dir),
        }

    if activity_dir:
        _require_data_dir("activity_source", activity_dir)
        report["sources"]["activity_source"] = {
            **_source_report(activity_dir),
            "records": _validate_activity_source(activity_dir),
        }

    return report


def create_dataloaders(config: dict) -> dict[str, DataLoader]:
    """Create DataLoader instances from config."""
    data_cfg = config.get("data", {})
    batch_size = config.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 4)

    pocket_dir = data_cfg.get("pocket_source", "")
    route_dir = data_cfg.get("route_source", "")
    joint_dir = data_cfg.get("joint_source")
    intent_dir = data_cfg.get("intent_source")
    loss_weights = config.get("loss_weights", {})
    require_pocket_route = float(loss_weights.get("pocket_route", 0.0) or 0.0) > 0.0
    require_intent = float(loss_weights.get("intent", 0.0) or 0.0) > 0.0
    joint_oversample_factor = int(data_cfg.get("joint_oversample_factor", 1) or 1)
    pocket_esm2_max_sequence_length = _pocket_esm2_max_sequence_length(config)

    _require_data_dir("pocket_source", pocket_dir)
    _require_data_dir("route_source", route_dir)
    if require_pocket_route:
        _require_data_dir("joint_source", joint_dir or "")
    elif joint_dir:
        _require_data_dir("joint_source", joint_dir)
    if require_intent:
        _require_data_dir("intent_source", intent_dir or "")
    elif intent_dir:
        _require_data_dir("intent_source", intent_dir)

    paired_ds = PairedHUMUDataset(
        pocket_dir,
        route_dir,
        max_samples=config.get("max_samples"),
        joint_dir=joint_dir,
        intent_dir=intent_dir,
        require_intent=require_intent,
        require_pocket_route=require_pocket_route,
        pocket_esm2_max_sequence_length=pocket_esm2_max_sequence_length,
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
        validation_size = max(1, int(len(paired_ds) * eval_split_ratio))
        train_size = len(paired_ds) - validation_size
        if train_size <= 0:
            raise ValueError("eval.eval_split_ratio leaves no HUMU training records")
        generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
        train_ds, validation_ds = random_split(
            paired_ds,
            [train_size, validation_size],
            generator=generator,
        )
    train_ds = _oversample_joint_training_records(train_ds, joint_oversample_factor)

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


def _require_data_dir(key: str, value: str) -> None:
    if not value:
        raise FileNotFoundError(f"data.{key} is required for HUMU pretraining")
    if not os.path.isdir(value):
        raise FileNotFoundError(f"data.{key} does not exist: {value}")


def _require_non_empty(key: str, dataset: Dataset) -> None:
    if len(dataset) == 0:
        raise ValueError(f"data.{key} contains no HUMU training records")


def _pocket_esm2_max_sequence_length(config: dict) -> int | None:
    pocket_cfg = config.get("encoders", {}).get("pocket", {}) or {}
    if not bool(pocket_cfg.get("use_esm2", False)):
        return None
    return _positive_int_or_none(pocket_cfg.get("esm2_max_sequence_length"))


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


def _oversample_joint_training_records(dataset: Dataset, factor: int) -> Dataset:
    if factor <= 1 or len(dataset) == 0:
        return dataset
    base_indices = list(range(len(dataset)))
    joint_indices = [
        index
        for index in base_indices
        if _dataset_pair_type(dataset, index) == "mol_pocket_route"
    ]
    if not joint_indices:
        return dataset
    return _IndexedDataset(dataset, base_indices + joint_indices * (factor - 1))


def _dataset_pair_type(dataset: Dataset, index: int) -> str | None:
    if isinstance(dataset, PairedHUMUDataset):
        return dataset.samples[index].get("pair_type")
    if isinstance(dataset, _IndexedDataset):
        return _dataset_pair_type(dataset.dataset, dataset.indices[index])
    if isinstance(dataset, Subset):
        return _dataset_pair_type(dataset.dataset, int(dataset.indices[index]))
    sample = dataset[index]
    return sample.get("pair_type") if isinstance(sample, dict) else None


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


def _validate_intent_source(data_dir: str) -> int:
    count = 0
    for record in _iter_jsonl_records(data_dir):
        _validate_intent_record(record)
        count += 1
    return count


def _validate_intent_record(record: dict) -> None:
    if "targets" not in record and "objective_nodes" not in record:
        raise ValueError("HUMU intent record requires targets or objective_nodes")


def _validate_activity_source(data_dir: str) -> int:
    count = 0
    for record in _iter_jsonl_records(data_dir):
        _validate_activity_record(record)
        count += 1
    if count == 0:
        raise ValueError(f"data.activity_source contains no activity records: {data_dir}")
    return count


def _validate_activity_record(record: dict) -> None:
    smiles = (
        record.get("ligand_smiles")
        or record.get("canonical_smiles")
        or record.get("smiles")
    )
    target_id = record.get("target_id") or record.get("target_chembl_id")
    activity_value = record.get("activity_value", record.get("pchembl_value"))
    if not smiles or not _is_valid_smiles(smiles):
        raise ValueError("HUMU activity record requires valid ligand_smiles")
    if not target_id:
        raise ValueError(f"HUMU activity record requires target_id: {smiles}")
    if activity_value is None:
        raise ValueError(f"HUMU activity record requires activity_value: {smiles}")
    try:
        float(activity_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"HUMU activity record requires numeric activity_value: {smiles}") from exc


def _validate_pocket_esm2_source(data_dir: str, esm2_dim: int) -> int:
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


def _pocket_payload_from_record(record: dict, base_dir: Path, pocket_id: str) -> dict:
    if {"coords", "elements", "residue_types"}.issubset(record):
        payload = {
            "pdb_id": pocket_id,
            "coords": record["coords"],
            "elements": record["elements"],
            "residue_types": record["residue_types"],
        }
        _copy_pocket_optional_fields(payload, record)
        return payload
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
    return payload


def _copy_pocket_optional_fields(target: dict, source: dict) -> None:
    for key in ("protein_sequence", "sequence", "esm2_embedding", "protein_chains"):
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
        if not _is_valid_smiles(record.get("ligand_smiles", ""))
    ]
    if invalid_records:
        sample = invalid_records[0]
        raise ValueError(
            "HUMU paired batch contains invalid ligand_smiles "
            f"mol_id={sample.get('mol_id', '')} "
            f"pair_type={sample.get('pair_type', '')}"
        )
    keys = set().union(*(record.keys() for record in records))
    return {key: [record.get(key) for record in records] for key in keys}


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
