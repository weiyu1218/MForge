#!/usr/bin/env python3
"""FragFM training CLI for fragment vocabulary and two-level DFM artifacts."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="FragFM SA-aware DFM training")
    parser.add_argument(
        "--data",
        required=True,
        help="JSON/JSONL file or directory of FragFM records",
    )
    parser.add_argument("--output-dir", required=True, help="Artifact output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Transformer hidden dimension")
    parser.add_argument("--rate-loss-weight", type=float, default=0.1, help="SA rate loss weight")
    parser.add_argument(
        "--rate-optimizer",
        choices=("adamw", "sgd"),
        default="adamw",
        help="Optimizer for SA-aware rate matrix parameters",
    )
    parser.add_argument(
        "--disable-rate-grad-clip",
        action="store_true",
        help="Skip gradient clipping for SA-aware rate matrix parameters",
    )
    parser.add_argument("--device", default="cuda", help="Training device")
    parser.add_argument(
        "--resume",
        default="",
        help="Optional checkpoint path to resume model weights",
    )
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
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
    parser.add_argument(
        "--humu-embedding-dim",
        type=int,
        default=129,
        help="Expected Lorentz full-coordinate dimension for rule HUMU embeddings",
    )
    parser.add_argument(
        "--humu-curvature",
        type=float,
        default=1.0,
        help="Curvature used when validating rule HUMU embeddings",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Log batch progress every N batches; set 0 to disable batch logs",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.hidden_dim % 8 != 0:
        raise ValueError("--hidden-dim must be divisible by 8")
    if args.kd_weight < 0.0:
        raise ValueError("--kd-weight must be >= 0")
    if args.kd_generator_idx < 0:
        raise ValueError("--kd-generator-idx must be non-negative")
    if args.humu_embedding_dim <= 1:
        raise ValueError("--humu-embedding-dim must be greater than 1")
    if args.humu_curvature <= 0.0:
        raise ValueError("--humu-curvature must be positive")
    if args.log_every < 0:
        raise ValueError("--log-every must be >= 0")
    if args.kd_teacher_embeddings and not Path(args.kd_teacher_embeddings).is_file():
        raise FileNotFoundError(
            f"KD teacher embedding artifact not found: {args.kd_teacher_embeddings}"
        )

    _add_project_paths()
    from mf_core.routing.cross_paradigm_kd import (
        CrossParadigmKDLayer,
        load_teacher_embeddings_artifact,
    )
    from mf_generators.fragfm.model.sa_aware_rate_matrix import SAAwareRateMatrix
    from mf_generators.fragfm.model.two_level_dfm import TwoLevelDFM

    records = _load_records(
        Path(args.data),
        expected_humu_dim=args.humu_embedding_dim,
        humu_curvature=args.humu_curvature,
    )
    fragments = _build_fragment_list(records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    _write_vocab_artifact(vocab_path, fragments, records)
    LOGGER.info(
        "Loaded %s FragFM records with %s fragments; output_dir=%s",
        len(records),
        len(fragments),
        output_dir,
    )

    dataset = FragFMDataset(records, fragments)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
    )

    device = _resolve_training_device(args.device)
    if str(args.device) != str(device):
        LOGGER.warning(
            "Requested training device %s is unavailable; using %s",
            args.device,
            device,
        )
    LOGGER.info("Using training device: %s", device)
    model = TwoLevelDFM(vocab_size=len(fragments), hidden_dim=args.hidden_dim).to(device)
    rate_matrix = SAAwareRateMatrix(vocab_size=len(fragments)).to(device)
    kd_layer = None
    if args.kd_teacher_embeddings:
        kd_layer = CrossParadigmKDLayer(
            n_generators=max(args.kd_generator_idx + 1, 1),
        ).to(device)
        teacher_target = kd_layer.update_teacher_embedding_targets(
            args.kd_generator_idx,
            load_teacher_embeddings_artifact(args.kd_teacher_embeddings, device=device),
        )
        if teacher_target.numel() != args.hidden_dim:
            raise ValueError("FragFM KD teacher embedding dimension must match --hidden-dim")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)

    model_optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    rate_optimizer = _build_rate_optimizer(
        rate_matrix.parameters(),
        optimizer_name=args.rate_optimizer,
        lr=args.lr,
        weight_decay=1e-5,
    )

    best_loss = float("inf")
    total_batches = len(loader)
    for epoch in range(args.epochs):
        model.train()
        rate_matrix.train()
        total_loss = 0.0
        batches = 0
        epoch_started_at = time.monotonic()
        for batch in loader:
            batch_started_at = time.monotonic()
            fragment_ids = batch["fragment_ids"].to(device)
            targets = batch["targets"].to(device)
            sa_bins = batch["sa_bins"].to(device)
            lengths = batch["lengths"].to(device)
            molecule_ids = torch.zeros(
                fragment_ids.shape[0],
                fragment_ids.shape[1],
                args.hidden_dim,
                dtype=torch.float32,
                device=device,
            )

            model_optimizer.zero_grad()
            rate_optimizer.zero_grad()
            logits = model(fragment_ids, molecule_ids)
            dfm_loss = functional.cross_entropy(
                logits.reshape(-1, len(fragments)),
                targets.reshape(-1),
                ignore_index=-100,
            )
            rate_loss = _rate_transition_loss(rate_matrix, fragment_ids, lengths, sa_bins)
            kd_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            if kd_layer is not None and args.kd_weight > 0.0:
                fragment_embeddings = model.fragment_encoder(fragment_ids)
                kd_embeddings = _mean_valid_fragment_embeddings(
                    fragment_embeddings,
                    lengths,
                )
                kd_loss = kd_layer.compute_distillation_loss(
                    [kd_embeddings],
                    [args.kd_generator_idx],
                )
            loss = dfm_loss + args.rate_loss_weight * rate_loss + args.kd_weight * kd_loss
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            if not args.disable_rate_grad_clip:
                nn.utils.clip_grad_norm_(list(rate_matrix.parameters()), 1.0)
            model_optimizer.step()
            rate_optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            batches += 1
            if _should_log_batch(
                batch_number=batches,
                total_batches=total_batches,
                log_every=args.log_every,
            ):
                elapsed_seconds = time.monotonic() - batch_started_at
                LOGGER.info(
                    "Epoch %s/%s batch %s/%s: loss=%.4f dfm=%.4f rate=%.4f kd=%.4f "
                    "batch_seconds=%.2f",
                    epoch + 1,
                    args.epochs,
                    batches,
                    total_batches,
                    float(loss.detach().cpu().item()),
                    float(dfm_loss.detach().cpu().item()),
                    float(rate_loss.detach().cpu().item()),
                    float(kd_loss.detach().cpu().item()),
                    elapsed_seconds,
                )

        avg_loss = total_loss / max(batches, 1)
        LOGGER.info(
            "Epoch %s/%s: loss=%.4f epoch_seconds=%.2f",
            epoch + 1,
            args.epochs,
            avg_loss,
            time.monotonic() - epoch_started_at,
        )
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            torch.save(rate_matrix.state_dict(), output_dir / "rate_matrix.pt")
        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt")

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    torch.save(rate_matrix.state_dict(), output_dir / "final_rate_matrix.pt")
    humu_stats = _humu_embedding_stats(records)
    manifest = _training_manifest_payload(
        records=records,
        fragments=fragments,
        epochs=args.epochs,
        best_loss=best_loss,
        vocab_path=vocab_path,
        output_dir=output_dir,
        kd_teacher_embeddings=str(args.kd_teacher_embeddings or ""),
        kd_weight=args.kd_weight,
        kd_generator_idx=args.kd_generator_idx,
        humu_embedding_dim=args.humu_embedding_dim,
        humu_curvature=args.humu_curvature,
        rate_optimizer=args.rate_optimizer,
        rate_grad_clip=not bool(args.disable_rate_grad_clip),
        requested_device=str(args.device),
        actual_device=str(device),
        log_every=args.log_every,
        humu_stats=humu_stats,
    )
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


class FragFMDataset(Dataset):
    def __init__(self, records: list[dict], fragments: list[str]):
        self.records = records
        self.fragment_to_idx = {fragment: index for index, fragment in enumerate(fragments)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        ids = [self.fragment_to_idx[fragment] for fragment in record["fragments"]]
        return {
            "fragment_ids": torch.tensor(ids, dtype=torch.long),
            "sa_bin": torch.tensor(int(record["sa_score_bin"]), dtype=torch.long),
        }


def _add_project_paths() -> None:
    project_root = Path(__file__).resolve().parents[3]
    for rel_path in (
        ("libs", "mf-core", "src"),
        ("libs", "mf-humu", "src"),
        ("libs", "mf-chem", "src"),
        ("models", "mf-generators", "fragfm", "src"),
    ):
        sys.path.insert(0, str(project_root.joinpath(*rel_path)))


def _load_records(
    path: Path,
    *,
    expected_humu_dim: int = 129,
    humu_curvature: float = 1.0,
) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"FragFM training data not found: {path}")
    records: list[dict] = []
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    for file_path in files:
        if not file_path.is_file() or file_path.suffix not in {".json", ".jsonl"}:
            continue
        records.extend(_load_record_file(file_path))
    if not records:
        raise ValueError(f"FragFM training data contains no records: {path}")
    return [
        _normalize_record(
            index,
            record,
            expected_humu_dim=expected_humu_dim,
            humu_curvature=humu_curvature,
        )
        for index, record in enumerate(records)
    ]


def _load_record_file(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        loaded = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    loaded.append(json.loads(line))
        return loaded
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("assembly_rules"), list):
        return list(payload["assembly_rules"])
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported FragFM JSON payload: {path}")


def _normalize_record(
    index: int,
    record: object,
    *,
    expected_humu_dim: int = 129,
    humu_curvature: float = 1.0,
) -> dict:
    if not isinstance(record, dict):
        raise TypeError("FragFM training record must be a JSON object")
    fragments = record.get("fragments")
    product = record.get("product")
    if not isinstance(fragments, list) or not fragments:
        raise ValueError("FragFM training record requires non-empty fragments")
    if not isinstance(product, str) or not product:
        raise ValueError("FragFM training record requires product")
    canonical_product = _canonical_smiles(product)
    sa_score_bin = int(record.get("sa_score_bin", 5))
    if not 0 <= sa_score_bin <= 9:
        raise ValueError("FragFM training record sa_score_bin must be in [0, 9]")
    normalized = {
        "id": str(record.get("id", index)),
        "fragments": [str(fragment) for fragment in fragments],
        "product": canonical_product,
        "sa_score_bin": sa_score_bin,
    }
    humu_embedding = record.get("humu_embedding")
    if humu_embedding is not None:
        from mf_core.geometry import normalize_lorentz_embedding

        normalized_embedding = normalize_lorentz_embedding(
            humu_embedding,
            expected_dim=expected_humu_dim,
            curvature=humu_curvature,
        )
        if normalized_embedding is None:
            raise ValueError(
                "FragFM training record humu_embedding must be a finite "
                f"{expected_humu_dim}-dimensional Lorentz full-coordinate vector"
            )
        normalized["humu_embedding"] = normalized_embedding
    return normalized


def _canonical_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("RDKit is required for FragFM training data validation") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid FragFM product SMILES: {smiles}")
    return Chem.MolToSmiles(mol)


def _build_fragment_list(records: list[dict]) -> list[str]:
    fragments = {fragment for record in records for fragment in record["fragments"]}
    if not fragments:
        raise ValueError("FragFM training records produced empty fragment vocabulary")
    return sorted(fragments)


def _build_rate_optimizer(
    parameters,
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
):
    if optimizer_name == "adamw":
        return AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        from torch.optim import SGD

        return SGD(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported rate optimizer: {optimizer_name}")


def _resolve_training_device(requested_device: str) -> torch.device:
    if requested_device == "cpu" or torch.cuda.is_available():
        return torch.device(requested_device)
    return torch.device("cpu")


def _should_log_batch(
    *,
    batch_number: int,
    total_batches: int,
    log_every: int,
) -> bool:
    return bool(
        log_every
        and (
            batch_number == 1
            or batch_number % log_every == 0
            or batch_number == total_batches
        )
    )


def _training_manifest_payload(
    *,
    records: list[dict],
    fragments: list[str],
    epochs: int,
    best_loss: float,
    vocab_path: Path,
    output_dir: Path,
    kd_teacher_embeddings: str,
    kd_weight: float,
    kd_generator_idx: int,
    humu_embedding_dim: int,
    humu_curvature: float,
    rate_optimizer: str,
    rate_grad_clip: bool,
    requested_device: str,
    actual_device: str,
    log_every: int,
    humu_stats: dict[str, float | int] | None = None,
) -> dict:
    return {
        "schema_version": "fragfm_training.v1",
        "records": len(records),
        "fragments": len(fragments),
        "epochs": int(epochs),
        "best_loss": float(best_loss),
        "vocab_path": str(vocab_path),
        "checkpoint_path": str(output_dir / "best_model.pt"),
        "rate_matrix_path": str(output_dir / "rate_matrix.pt"),
        "kd_teacher_embeddings": str(kd_teacher_embeddings or ""),
        "kd_weight": float(kd_weight),
        "kd_generator_idx": int(kd_generator_idx),
        "humu_embedding_dim": int(humu_embedding_dim),
        "humu_curvature": float(humu_curvature),
        "rate_optimizer": str(rate_optimizer),
        "rate_grad_clip": bool(rate_grad_clip),
        "requested_device": str(requested_device),
        "actual_device": str(actual_device),
        "log_every": int(log_every),
        **(humu_stats or _humu_embedding_stats(records)),
    }


def _write_vocab_artifact(path: Path, fragments: list[str], records: list[dict]) -> None:
    payload = {
        "fragments": fragments,
        "assembly_rules": [_vocab_rule_payload(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _vocab_rule_payload(record: dict) -> dict:
    payload = {
        "id": record["id"],
        "fragments": record["fragments"],
        "product": record["product"],
        "sa_score_bin": record["sa_score_bin"],
    }
    if "humu_embedding" in record:
        payload["humu_embedding"] = record["humu_embedding"]
    return payload


def _humu_embedding_stats(records: list[dict]) -> dict[str, float | int]:
    humu_embedding_count = sum(1 for record in records if "humu_embedding" in record)
    return {
        "humu_embedding_count": humu_embedding_count,
        "humu_embedding_coverage": humu_embedding_count / max(len(records), 1),
    }


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    max_len = max(int(item["fragment_ids"].numel()) for item in batch)
    fragment_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    targets = torch.full((len(batch), max_len), -100, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    sa_bins = torch.zeros(len(batch), dtype=torch.long)
    for row, item in enumerate(batch):
        ids = item["fragment_ids"]
        length = int(ids.numel())
        fragment_ids[row, :length] = ids
        targets[row, :length] = ids
        lengths[row] = length
        sa_bins[row] = item["sa_bin"]
    return {
        "fragment_ids": fragment_ids,
        "targets": targets,
        "lengths": lengths,
        "sa_bins": sa_bins,
    }


def _rate_transition_loss(
    rate_matrix,
    fragment_ids: torch.Tensor,
    lengths: torch.Tensor,
    sa_bins: torch.Tensor,
) -> torch.Tensor:
    if hasattr(rate_matrix, "base_rate") and hasattr(rate_matrix, "sa_score_embedding"):
        return _sparse_rate_transition_loss(rate_matrix, fragment_ids, lengths, sa_bins)
    losses = []
    rates = rate_matrix(sa_bins)
    for row in range(fragment_ids.shape[0]):
        length = int(lengths[row].item())
        if length < 2:
            continue
        for pos in range(length - 1):
            left = fragment_ids[row, pos]
            right = fragment_ids[row, pos + 1]
            losses.append(
                functional.cross_entropy(rates[row, left].unsqueeze(0), right.unsqueeze(0))
            )
    if not losses:
        return rates.sum() * 0.0
    return torch.stack(losses).mean()


def _sparse_rate_transition_loss(
    rate_matrix,
    fragment_ids: torch.Tensor,
    lengths: torch.Tensor,
    sa_bins: torch.Tensor,
) -> torch.Tensor:
    vocab_size = int(rate_matrix.base_rate.shape[0])
    left_chunks = []
    right_chunks = []
    sa_bin_chunks = []
    for row in range(fragment_ids.shape[0]):
        length = int(lengths[row].item())
        if length < 2:
            continue
        transitions = length - 1
        left_chunks.append(fragment_ids[row, :transitions])
        right_chunks.append(fragment_ids[row, 1:length])
        sa_bin_chunks.append(sa_bins[row].expand(transitions))
    if not left_chunks:
        return rate_matrix.base_rate.sum() * 0.0
    left_indices = torch.cat(left_chunks)
    right_indices = torch.cat(right_chunks)
    sa_bin_indices = torch.cat(sa_bin_chunks)
    vocab_offsets = torch.arange(
        vocab_size,
        dtype=torch.long,
        device=fragment_ids.device,
    )
    row_offsets = left_indices * vocab_size
    sa_embedding_indices = row_offsets.unsqueeze(1) + vocab_offsets.unsqueeze(0)
    sa_embedding_rows = rate_matrix.sa_score_embedding.weight[
        sa_bin_indices.unsqueeze(1),
        sa_embedding_indices,
    ]
    logits = rate_matrix.base_rate[left_indices] * (1 + torch.tanh(sa_embedding_rows))
    return functional.cross_entropy(logits, right_indices)


def _mean_valid_fragment_embeddings(
    fragment_embeddings: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    positions = torch.arange(fragment_embeddings.shape[1], device=fragment_embeddings.device)
    mask = positions.unsqueeze(0) < lengths.to(device=fragment_embeddings.device).unsqueeze(1)
    masked = fragment_embeddings * mask.unsqueeze(-1).to(dtype=fragment_embeddings.dtype)
    denominator = lengths.clamp(min=1).to(
        device=fragment_embeddings.device,
        dtype=fragment_embeddings.dtype,
    )
    return masked.sum(dim=1) / denominator.unsqueeze(-1)


if __name__ == "__main__":
    main()
