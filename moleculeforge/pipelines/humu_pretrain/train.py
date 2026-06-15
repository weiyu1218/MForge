#!/usr/bin/env python3
"""HUMU pre-training CLI entry point.

Usage:
    python train.py --config configs/models/humu_pretrain.yaml
    python train.py --config configs/models/humu_pretrain.yaml \
        --resume checkpoints/humu/best_model.pt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description="HUMU pre-training pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate HUMU data contracts and exit without training",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.resume:
        config["resume_from"] = args.resume
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr
    if args.device:
        config["device"] = args.device

    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(project_root, "pipelines", "humu_pretrain", "src"))
    sys.path.insert(0, os.path.join(project_root, "libs", "mf-core", "src"))
    sys.path.insert(0, os.path.join(project_root, "libs", "mf-humu", "src"))
    for rel_path in (
        ("models", "mf-encoders", "humu_mol_encoder", "src"),
        ("models", "mf-encoders", "humu_pocket_encoder", "src"),
        ("models", "mf-encoders", "humu_route_encoder", "src"),
    ):
        sys.path.insert(0, os.path.join(project_root, *rel_path))

    if args.preflight_only:
        from humu_pretrain.data_loader import preflight_humu_data_contract

        print(json.dumps(preflight_humu_data_contract(config), sort_keys=True))
        return

    from humu_pretrain.pipeline import run

    result = asyncio.run(run(config))
    print(f"Training complete: {result}")


if __name__ == "__main__":
    main()
