#!/usr/bin/env python3
"""FragFM training CLI for fragment vocabulary and two-level DFM artifacts."""
from __future__ import annotations

import argparse
import json
import logging
import sys
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
    parser.add_argument("--device", default="cuda", help="Training device")
    parser.add_argument(
        "--resume",
        default="",
        help="Optional checkpoint path to resume model weights",
    )
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.hidden_dim % 8 != 0:
        raise ValueError("--hidden-dim must be divisible by 8")

    _add_project_paths()
    from mf_generators.fragfm.model.sa_aware_rate_matrix import SAAwareRateMatrix
    from mf_generators.fragfm.model.two_level_dfm import TwoLevelDFM

    records = _load_records(Path(args.data))
    fragments = _build_fragment_list(records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    _write_vocab_artifact(vocab_path, fragments, records)

    dataset = FragFMDataset(records, fragments)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
    )

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model = TwoLevelDFM(vocab_size=len(fragments), hidden_dim=args.hidden_dim).to(device)
    rate_matrix = SAAwareRateMatrix(vocab_size=len(fragments)).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state, strict=False)

    optimizer = AdamW(
        list(model.parameters()) + list(rate_matrix.parameters()),
        lr=args.lr,
        weight_decay=1e-5,
    )

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        rate_matrix.train()
        total_loss = 0.0
        batches = 0
        for batch in loader:
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

            optimizer.zero_grad()
            logits = model(fragment_ids, molecule_ids)
            dfm_loss = functional.cross_entropy(
                logits.reshape(-1, len(fragments)),
                targets.reshape(-1),
                ignore_index=-100,
            )
            rate_loss = _rate_transition_loss(rate_matrix, fragment_ids, lengths, sa_bins)
            loss = dfm_loss + args.rate_loss_weight * rate_loss
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(rate_matrix.parameters()),
                1.0,
            )
            optimizer.step()

            total_loss += float(loss.detach().cpu().item())
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        LOGGER.info("Epoch %s/%s: loss=%.4f", epoch + 1, args.epochs, avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            torch.save(rate_matrix.state_dict(), output_dir / "rate_matrix.pt")
        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt")

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    torch.save(rate_matrix.state_dict(), output_dir / "final_rate_matrix.pt")
    manifest = {
        "schema_version": "fragfm_training.v1",
        "records": len(records),
        "fragments": len(fragments),
        "epochs": args.epochs,
        "best_loss": best_loss,
        "vocab_path": str(vocab_path),
        "checkpoint_path": str(output_dir / "best_model.pt"),
        "rate_matrix_path": str(output_dir / "rate_matrix.pt"),
    }
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


def _load_records(path: Path) -> list[dict]:
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
    return [_normalize_record(index, record) for index, record in enumerate(records)]


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


def _normalize_record(index: int, record: object) -> dict:
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
    return {
        "id": str(record.get("id", index)),
        "fragments": [str(fragment) for fragment in fragments],
        "product": canonical_product,
        "sa_score_bin": sa_score_bin,
    }


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


def _write_vocab_artifact(path: Path, fragments: list[str], records: list[dict]) -> None:
    payload = {
        "fragments": fragments,
        "assembly_rules": [
            {
                "id": record["id"],
                "fragments": record["fragments"],
                "product": record["product"],
                "sa_score_bin": record["sa_score_bin"],
            }
            for record in records
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


if __name__ == "__main__":
    main()
