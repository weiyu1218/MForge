#!/usr/bin/env python3
"""Train the UAS molecule autoencoder on HUMU embedding vectors."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from collections.abc import Iterable
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="UAS autoencoder training")
    parser.add_argument(
        "--data",
        required=True,
        help="JSON/JSONL file or directory with embeddings",
    )
    parser.add_argument("--output-dir", required=True, help="Output artifact directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", default="cuda", help="Training device")
    parser.add_argument(
        "--kd-teacher-embeddings",
        default="",
        help="JSON artifact containing teacher embedding targets for KD loss",
    )
    parser.add_argument(
        "--kd-weight",
        type=float,
        default=0.0,
        help="Weight for teacher embedding distillation loss",
    )
    parser.add_argument(
        "--kd-generator-idx",
        type=int,
        default=0,
        help="Generator index used for KD teacher target lookup",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.kd_weight < 0.0:
        raise ValueError("--kd-weight must be >= 0")
    if args.kd_generator_idx < 0:
        raise ValueError("--kd-generator-idx must be non-negative")
    if args.kd_teacher_embeddings and not Path(args.kd_teacher_embeddings).is_file():
        raise FileNotFoundError(
            f"KD teacher embedding artifact not found: {args.kd_teacher_embeddings}"
        )

    _add_project_paths()
    from mf_core.routing.cross_paradigm_kd import (
        CrossParadigmKDLayer,
        load_teacher_embeddings_artifact,
    )
    from mf_generators.uas.autoencoder.molecule_ae import MoleculeAutoencoder

    embedding_rows = list(_load_embeddings(args.data))
    _validate_embedding_rows(embedding_rows)
    embeddings = torch.tensor(embedding_rows, dtype=torch.float32)
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("UAS training embeddings must contain exactly 129 finite values")
    dim = int(embeddings.shape[1])
    device_name = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(device_name)
    latent_dim = max(1, dim // 2)
    model = MoleculeAutoencoder(input_dim=dim, latent_dim=latent_dim).to(device)
    kd_layer = None
    if args.kd_teacher_embeddings:
        kd_layer = CrossParadigmKDLayer(
            n_generators=max(args.kd_generator_idx + 1, 1),
        ).to(device)
        teacher_target = kd_layer.update_teacher_embedding_targets(
            args.kd_generator_idx,
            load_teacher_embeddings_artifact(args.kd_teacher_embeddings, device=device),
        )
        if teacher_target.numel() != latent_dim:
            raise ValueError("UAS KD teacher embedding dimension must match latent dimension")
    loader = DataLoader(
        TensorDataset(embeddings),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        total_loss = 0.0
        batches = 0
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, latent = model(batch)
            recon_loss = ((recon - batch) ** 2).mean(dim=tuple(range(1, batch.ndim))).mean()
            kd_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            if kd_layer is not None and args.kd_weight > 0.0:
                kd_loss = kd_layer.compute_distillation_loss(
                    [latent],
                    [args.kd_generator_idx],
                )
            loss = recon_loss + args.kd_weight * kd_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            batches += 1
        avg_loss = total_loss / max(batches, 1)
        LOGGER.info("Epoch %s/%s: loss=%.6f", epoch + 1, args.epochs, avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / "autoencoder.pt")

    torch.save(model.state_dict(), output_dir / "final_autoencoder.pt")
    torch.save(embeddings, output_dir / "reference_embeddings.pt")
    checkpoint_path = output_dir / "autoencoder.pt"
    checkpoint_digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            checkpoint_digest.update(chunk)
    (output_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "uas_training.v1",
                "records": int(embeddings.shape[0]),
                "dim": dim,
                "latent_dim": latent_dim,
                "epochs": args.epochs,
                "best_loss": best_loss,
                "autoencoder_path": "autoencoder.pt",
                "autoencoder_sha256": f"sha256:{checkpoint_digest.hexdigest()}",
                "reference_embeddings_path": "reference_embeddings.pt",
                "kd_teacher_embeddings": str(args.kd_teacher_embeddings or ""),
                "kd_weight": float(args.kd_weight),
                "kd_generator_idx": int(args.kd_generator_idx),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _add_project_paths() -> None:
    project_root = Path(__file__).resolve().parents[3]
    for rel_path in (
        ("libs", "mf-core", "src"),
        ("models", "mf-generators", "uas", "src"),
    ):
        sys.path.insert(0, str(project_root.joinpath(*rel_path)))


def _load_embeddings(path_value: str) -> Iterable[list[object]]:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"UAS training data not found: {path}")
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    for file_path in files:
        if not file_path.is_file() or file_path.suffix not in {".json", ".jsonl"}:
            continue
        yield from _load_embedding_file(file_path)


def _validate_embedding_rows(rows: list[list[object]]) -> None:
    if not rows:
        raise ValueError("UAS training data must contain at least one embedding")
    for row in rows:
        if len(row) != 129:
            raise ValueError(
                "UAS training embeddings must contain exactly 129 finite values"
            )
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(
                    "UAS training embeddings must contain exactly 129 finite values"
                )
            try:
                finite = math.isfinite(float(value))
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(
                    "UAS training embeddings must contain exactly 129 finite values"
                )


def _load_embedding_file(path: Path) -> Iterable[list[object]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield _embedding_from_record(json.loads(line))
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("embeddings"), list):
        for item in payload["embeddings"]:
            yield _embedding_from_record(item)
        return
    if isinstance(payload, list):
        for item in payload:
            yield _embedding_from_record(item)
        return
    yield _embedding_from_record(payload)


def _embedding_from_record(record: object) -> list[object]:
    value = record.get("embedding") if isinstance(record, dict) else record
    if not isinstance(value, list) or not value:
        raise ValueError("UAS embedding record requires a non-empty embedding list")
    return list(value)


if __name__ == "__main__":
    main()
