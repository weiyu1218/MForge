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
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
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
    parser.add_argument("--output-dir", type=str, default="checkpoints/hfm3d/", help="Output directory")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Add project paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, os.path.join(project_root, "libs", "mf-core", "src"))
    sys.path.insert(0, os.path.join(project_root, "libs", "mf-humu", "src"))
    sys.path.insert(0, os.path.join(project_root, "models", "mf-encoders", "humu_mol_encoder", "src"))
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

        from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder
        from mf_generators.hfm_3d.model.lorentz_flow_matching import LorentzFlowMatching

        encoder = HUMUMoleculeEncoder(dim=args.dim, curvature=1.0).to(device)
        encoder.eval()
        flow_model = LorentzFlowMatching(dim=args.dim, curvature=1.0, n_steps=args.n_steps)
        flow_model.to(device)
        decoder = nn.Sequential(
            nn.Linear(args.dim + 1, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024),
        )
        decoder.to(device)

        start_epoch, best_loss, optimizer_state = 0, float("inf"), None
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.resume:
            if not os.path.exists(args.resume):
                raise FileNotFoundError(f"HFM-3D resume checkpoint not found: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            flow_model.load_state_dict(ckpt.get("flow_model", {}), strict=False)
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
            raise ValueError(f"HFM-3D data directory contains no HFM-3D training records: {args.data}")
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
                x0 = manifold.expmap(
                    torch.zeros(batch_size, args.dim + 1, device=device),
                    torch.randn(batch_size, args.dim + 1, device=device),
                )
                x0[..., 0] = torch.sqrt(1 + x0[..., 1:].pow(2).sum(dim=-1))
                x1 = embeddings

                optimizer.zero_grad()
                flow_loss = flow_model(x0, x1)
                decoded = decoder(x1)
                recon_loss = ((decoded - decoded.mean(dim=0, keepdim=True)) ** 2).mean()

                loss = flow_loss + 0.1 * recon_loss
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()

                epoch_loss += float(loss.detach().cpu().item())
                n_batches += 1

                if n_batches % 50 == 0 and _is_main_process(distributed):
                    print(
                        f"  Epoch {epoch + 1} batch {n_batches}: "
                        f"flow={flow_loss.item():.4f} recon={recon_loss.item():.4f}"
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
                    )

                if (epoch + 1) % args.save_every == 0:
                    _save_ckpt(
                        flow_model,
                        decoder,
                        optimizer,
                        epoch,
                        avg_loss,
                        output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
                    )

        if _is_main_process(distributed):
            _save_ckpt(
                flow_model,
                decoder,
                optimizer,
                args.epochs,
                best_loss,
                output_dir / "final_model.pt",
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
    for fpath in data_path.glob("*.jsonl"):
        with open(fpath) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    if not samples:
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


def _save_ckpt(flow_model, decoder, optimizer, epoch, loss, path):
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "flow_model": _module(flow_model).state_dict(),
        "decoder": _module(decoder).state_dict(),
        "optimizer": optimizer.state_dict(),
    }, str(path))


if __name__ == "__main__":
    main()
