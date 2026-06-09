#!/usr/bin/env python3
"""HFM-3D generator training CLI.

Trains the Lorentz Flow Matching velocity field and molecular decoder.

Usage:
    python train.py --data /path/to/chembl/processed --epochs 50 --batch-size 128
    python train.py --data /path/to/chembl/processed --resume checkpoints/hfm3d/best.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int


def main():
    parser = argparse.ArgumentParser(description="HFM-3D flow matching training")
    parser.add_argument("--data", type=str, required=True, help="Path to processed ChEMBL data dir")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--dim", type=int, default=128, help="Manifold dimension")
    parser.add_argument("--n-steps", type=int, default=20, help="ODE integration steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/hfm3d/",
        help="Output directory",
    )
    parser.add_argument(
        "--humu-checkpoint",
        type=str,
        default=os.environ.get("HUMU_CHECKPOINT_PATH") or os.environ.get("HUMU_CHECKPOINT"),
        help="Path to pretrained HUMU checkpoint containing encoder_mol",
    )
    parser.add_argument(
        "--decoder-artifact",
        type=str,
        default=os.environ.get("HFM_DECODER_PATH") or None,
        help="Path to write HFM nearest-neighbor decoder artifact",
    )
    parser.add_argument(
        "--decoder-max-entries",
        type=int,
        default=int(os.environ.get("HFM_DECODER_MAX_ENTRIES", "4096")),
        help="Maximum molecules to encode into decoder artifact",
    )
    parser.add_argument(
        "--kd-teacher-embeddings",
        type=str,
        default=os.environ.get("HFM_KD_TEACHER_EMBEDDINGS") or "",
        help="JSON artifact containing teacher embedding targets for KD loss",
    )
    parser.add_argument(
        "--kd-weight",
        type=float,
        default=float(os.environ.get("HFM_KD_WEIGHT", "0.0")),
        help="Weight for teacher embedding distillation loss",
    )
    parser.add_argument(
        "--kd-generator-idx",
        type=int,
        default=int(os.environ.get("HFM_KD_GENERATOR_IDX", "0")),
        help="Generator index used for KD teacher target lookup",
    )
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    if not args.humu_checkpoint:
        parser.error("HUMU checkpoint is required via --humu-checkpoint or HUMU_CHECKPOINT_PATH")
    if not Path(args.humu_checkpoint).is_file():
        parser.error(f"HUMU checkpoint not found: {args.humu_checkpoint}")
    if args.decoder_max_entries < 1:
        parser.error("--decoder-max-entries must be >= 1")
    if args.kd_weight < 0.0:
        parser.error("--kd-weight must be >= 0")
    if args.kd_teacher_embeddings and not Path(args.kd_teacher_embeddings).is_file():
        parser.error(f"KD teacher embedding artifact not found: {args.kd_teacher_embeddings}")

    # Add project paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, os.path.join(project_root, "libs", "mf-core", "src"))
    sys.path.insert(0, os.path.join(project_root, "libs", "mf-humu", "src"))
    sys.path.insert(
        0,
        os.path.join(project_root, "models", "mf-encoders", "humu_mol_encoder", "src"),
    )
    sys.path.insert(0, os.path.join(project_root, "models", "mf-generators", "hfm_3d", "src"))

    distributed = _distributed_context_from_env()
    device = _get_device(args.device, distributed)
    _setup_distributed(distributed, device)
    torch.manual_seed(args.seed + distributed.rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + distributed.rank)

    try:
        if _is_main_process(distributed):
            print(f"Using device: {device}")
            if distributed.enabled:
                print(
                    "Distributed training: "
                    f"world_size={distributed.world_size} backend={dist.get_backend()}"
                )

        from mf_core.routing.cross_paradigm_kd import CrossParadigmKDLayer
        from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder
        from mf_generators.hfm_3d.model.lorentz_flow_matching import LorentzFlowMatching

        encoder = HUMUMoleculeEncoder(dim=args.dim, curvature=1.0).to(device)
        _load_humu_molecule_encoder_checkpoint(encoder, args.humu_checkpoint, device)
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        flow_model = LorentzFlowMatching(dim=args.dim, curvature=1.0, n_steps=args.n_steps)
        flow_model.to(device)
        decoder = nn.Sequential(
            nn.Linear(args.dim + 1, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024),
        )
        decoder.to(device)
        kd_layer = None
        if args.kd_teacher_embeddings:
            kd_layer = CrossParadigmKDLayer(
                n_generators=max(args.kd_generator_idx + 1, 1),
            ).to(device)
            kd_layer.update_teacher_embedding_targets(
                args.kd_generator_idx,
                _load_kd_teacher_embeddings(args.kd_teacher_embeddings, device),
            )

        start_epoch, best_loss, optimizer_state = 0, float("inf"), None
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        decoder_artifact = (
            Path(args.decoder_artifact)
            if args.decoder_artifact
            else output_dir / "decoder.json"
        )
        if args.resume:
            if not os.path.exists(args.resume):
                raise FileNotFoundError(f"HFM-3D resume checkpoint not found: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=True)
            flow_state = ckpt.get("flow_model") or ckpt.get("model") or {}
            flow_model.load_state_dict(flow_state, strict=False)
            decoder.load_state_dict(ckpt.get("decoder", {}), strict=False)
            optimizer_state = ckpt.get("optimizer")
            start_epoch = int(ckpt.get("epoch", 0))
            best_loss = float(ckpt.get("loss", float("inf")))
            if _is_main_process(distributed):
                print(f"Resumed from epoch {start_epoch}")

        flow_model = _wrap_distributed(flow_model, distributed, device)
        decoder = _wrap_distributed(decoder, distributed, device)

        all_params = list(flow_model.parameters()) + list(decoder.parameters())
        optimizer = AdamW(all_params, lr=args.lr, weight_decay=1e-5)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        samples = _load_molecules(args.data)
        if not samples:
            raise ValueError(
                f"HFM-3D data directory contains no HFM-3D training records: {args.data}"
            )
        if _is_main_process(distributed):
            _write_decoder_artifact(
                samples,
                encoder,
                device,
                decoder_artifact,
                args.humu_checkpoint,
                args.decoder_max_entries,
            )
        if distributed.enabled:
            dist.barrier()
        loader, sampler = _build_dataloader(samples, args.batch_size, distributed)

        for epoch in range(start_epoch, args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            flow_model.train()
            decoder.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                batch_size = len(batch)
                embeddings_list = []
                for smiles in batch:
                    with torch.no_grad():
                        emb = encoder.encode(smiles)
                    embeddings_list.append(emb)
                embeddings = torch.cat(embeddings_list, dim=0).to(device)

                manifold = _module(flow_model).manifold
                x0 = _sample_lorentz_prior(manifold, batch_size, args.dim, device)
                x1 = embeddings

                optimizer.zero_grad()
                flow_loss = flow_model(x0, x1)
                decoded = decoder(x1)
                fingerprint_target = _smiles_fingerprint_targets(batch, device)
                recon_loss = functional.binary_cross_entropy_with_logits(
                    decoded,
                    fingerprint_target,
                )
                kd_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                if kd_layer is not None and args.kd_weight > 0.0:
                    kd_loss = kd_layer.compute_distillation_loss(
                        [x1],
                        [args.kd_generator_idx],
                    )

                loss = flow_loss + 0.1 * recon_loss + args.kd_weight * kd_loss
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()

                epoch_loss += float(loss.detach().cpu().item())
                n_batches += 1

                if n_batches % 50 == 0 and _is_main_process(distributed):
                    print(
                        f"  Epoch {epoch + 1} batch {n_batches}: "
                        f"flow={flow_loss.item():.4f} recon={recon_loss.item():.4f} "
                        f"kd={kd_loss.item():.4f}"
                    )

            avg_loss = _reduce_epoch_loss(epoch_loss, n_batches, device, distributed)
            if not torch.isfinite(torch.tensor(avg_loss)):
                raise RuntimeError(f"HFM-3D training produced non-finite loss at epoch {epoch + 1}")
            if _is_main_process(distributed):
                print(f"Epoch {epoch + 1}/{args.epochs} avg_loss={avg_loss:.4f}")

                if best_loss == float("inf") or avg_loss < best_loss:
                    best_loss = avg_loss
                    _save_ckpt(
                        flow_model,
                        decoder,
                        optimizer,
                        epoch,
                        avg_loss,
                        output_dir / "best_model.pt",
                        args.humu_checkpoint,
                        decoder_artifact,
                        args.kd_teacher_embeddings,
                        args.kd_weight,
                        args.kd_generator_idx,
                    )

                if (epoch + 1) % args.save_every == 0:
                    _save_ckpt(
                        flow_model,
                        decoder,
                        optimizer,
                        epoch,
                        avg_loss,
                        output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
                        args.humu_checkpoint,
                        decoder_artifact,
                        args.kd_teacher_embeddings,
                        args.kd_weight,
                        args.kd_generator_idx,
                    )

        if _is_main_process(distributed):
            _save_ckpt(
                flow_model,
                decoder,
                optimizer,
                args.epochs,
                best_loss,
                output_dir / "final_model.pt",
                args.humu_checkpoint,
                decoder_artifact,
                args.kd_teacher_embeddings,
                args.kd_weight,
                args.kd_generator_idx,
            )
            print(f"Training complete. Best loss: {best_loss:.4f}")
    finally:
        _cleanup_distributed(distributed)


class MoleculeDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.smiles = [_record_smiles(sample) for sample in samples]

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> str:
        return self.smiles[index]


def _load_molecules(data_dir: str) -> list[dict]:
    """Load processed molecules from JSONL."""
    samples = []
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"HFM-3D training data path does not exist: {data_path}")
    manifest = data_path / "manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            m = json.load(f)
        for shard in m.get("shards", []):
            spath = data_path / shard
            if spath.exists():
                with open(spath) as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
        return samples
    for fpath in data_path.glob("*.jsonl"):
        if fpath.name == "rejects.jsonl":
            continue
        with open(fpath) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    return samples


def _record_smiles(sample: dict) -> str:
    smiles = sample.get("smiles")
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("HFM-3D training record requires non-empty smiles")
    return smiles.strip()


def _distributed_context_from_env() -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistributedContext(
        enabled=world_size > 1,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )


def _get_device(device_arg: str, distributed: DistributedContext) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        if distributed.enabled:
            torch.cuda.set_device(distributed.local_rank)
            return torch.device("cuda", distributed.local_rank)
        return torch.device("cuda")
    return torch.device("cpu")


def _setup_distributed(distributed: DistributedContext, device: torch.device) -> None:
    if not distributed.enabled or dist.is_initialized():
        return
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=distributed.rank,
        world_size=distributed.world_size,
    )


def _cleanup_distributed(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process(distributed: DistributedContext) -> bool:
    return not distributed.enabled or distributed.rank == 0


def _wrap_distributed(
    model: nn.Module,
    distributed: DistributedContext,
    device: torch.device,
) -> nn.Module:
    if not distributed.enabled:
        return model
    if device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
        )
    return DistributedDataParallel(model)


def _module(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _build_dataloader(
    samples: list[dict],
    batch_size: int,
    distributed: DistributedContext,
) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = MoleculeDataset(samples)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=True,
            drop_last=False,
        )
        if distributed.enabled
        else None
    )
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=_collate_smiles,
        ),
        sampler,
    )


def _collate_smiles(batch: list[str]) -> list[str]:
    return list(batch)


def _sample_lorentz_prior(
    manifold,
    batch_size: int,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    origin = manifold.origin(dim).to(device=device).expand(batch_size, -1)
    tangent = torch.zeros(batch_size, dim + 1, device=device)
    tangent[..., 1:] = torch.randn(batch_size, dim, device=device) * (0.1 / math.sqrt(dim))
    return manifold.expmap(origin, tangent)


def _load_humu_molecule_encoder_checkpoint(
    encoder: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("HUMU checkpoint must be a state dictionary")
    state_dict = _extract_humu_molecule_state_dict(checkpoint, encoder)
    expected_keys = set(encoder.state_dict())
    missing_keys = sorted(expected_keys.difference(state_dict))
    if missing_keys:
        raise RuntimeError(
            "HUMU checkpoint is missing HUMUMoleculeEncoder keys: "
            + ", ".join(missing_keys[:10])
        )
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if key in expected_keys
    }
    try:
        encoder.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"HUMU checkpoint is incompatible with HUMUMoleculeEncoder: {checkpoint_path}"
        ) from exc


def _extract_humu_molecule_state_dict(
    checkpoint: dict,
    encoder: nn.Module,
) -> dict:
    for key in ("encoder_mol", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return _strip_module_prefix(value)
    expected_keys = set(encoder.state_dict())
    if expected_keys.intersection(checkpoint):
        return _strip_module_prefix(checkpoint)
    raise ValueError("HUMU checkpoint does not contain encoder_mol state")


def _strip_module_prefix(state_dict: dict) -> dict:
    prefixes = ("module.", "_orig_mod.", "inner.")
    normalized = {}
    for key, value in state_dict.items():
        normalized_key = key
        if isinstance(normalized_key, str):
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if normalized_key.startswith(prefix):
                        normalized_key = normalized_key.removeprefix(prefix)
                        changed = True
        normalized[normalized_key] = value
    return normalized


def _smiles_fingerprint_targets(smiles_batch: list[str], device: torch.device) -> torch.Tensor:
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for HFM-3D decoder fingerprint targets") from exc
    targets = []
    for smiles in smiles_batch:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"HFM-3D training record has invalid SMILES: {smiles}")
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        targets.append(torch.tensor([float(bit) for bit in fingerprint.ToBitString()]))
    return torch.stack(targets, dim=0).to(device=device, dtype=torch.float32)


def _load_kd_teacher_embeddings(path_value: str, device: torch.device) -> torch.Tensor:
    from mf_core.routing.cross_paradigm_kd import load_teacher_embeddings_artifact

    return load_teacher_embeddings_artifact(path_value, device=device)


def _write_decoder_artifact(
    samples: list[dict],
    encoder: nn.Module,
    device: torch.device,
    path: Path,
    humu_checkpoint: str,
    max_entries: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = samples[:max_entries]
    entries = []
    with torch.no_grad():
        for index, sample in enumerate(selected):
            smiles = _record_smiles(sample)
            latent = encoder.encode(smiles).squeeze(0).detach().cpu().float().tolist()
            entries.append(
                {
                    "id": _record_id(sample, index),
                    "smiles": smiles,
                    "latent": [float(value) for value in latent],
                    "sdf": _build_decoder_sdf(smiles, seed=index + 1),
                }
            )
    payload = {
        "humu_checkpoint": str(humu_checkpoint),
        "entries": entries,
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _build_decoder_sdf(smiles: str, seed: int) -> str:
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for HFM-3D decoder artifact SDF") from exc
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"HFM-3D training record has invalid SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol, randomSeed=int(seed))
        if status != 0:
            raise RuntimeError(f"HFM-3D decoder artifact SDF failed for {smiles}")
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    return Chem.MolToMolBlock(mol)


def _record_id(sample: dict, index: int) -> str:
    for key in ("id", "chembl_id", "molecule_id"):
        value = sample.get(key)
        if value is not None and str(value):
            return str(value)
    return str(index)


def _reduce_epoch_loss(
    epoch_loss: float,
    n_batches: int,
    device: torch.device,
    distributed: DistributedContext,
) -> float:
    stats = torch.tensor([epoch_loss, float(n_batches)], dtype=torch.float64, device=device)
    if distributed.enabled:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_loss, total_batches = stats.detach().cpu().tolist()
    return float(total_loss) / max(float(total_batches), 1.0)


def _save_ckpt(
    flow_model,
    decoder,
    optimizer,
    epoch,
    loss,
    path,
    humu_checkpoint,
    decoder_artifact,
    kd_teacher_embeddings="",
    kd_weight=0.0,
    kd_generator_idx=0,
):
    flow_state = _module(flow_model).state_dict()
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "model": flow_state,
        "flow_model": flow_state,
        "decoder": _module(decoder).state_dict(),
        "optimizer": optimizer.state_dict(),
        "humu_checkpoint": str(humu_checkpoint),
        "decoder_artifact": str(decoder_artifact),
        "kd_teacher_embeddings": str(kd_teacher_embeddings or ""),
        "kd_weight": float(kd_weight),
        "kd_generator_idx": int(kd_generator_idx),
    }, str(path))


if __name__ == "__main__":
    main()
