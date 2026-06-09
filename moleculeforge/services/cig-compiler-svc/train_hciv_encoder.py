"""Train and export a CIG Enc_intent / HCIV encoder checkpoint."""
from __future__ import annotations

import argparse

from cig_compiler_svc.domain.hciv_training import train_hciv_encoder_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="JSON/JSONL supervised CIG-HCIV data")
    parser.add_argument(
        "--output-checkpoint",
        required=True,
        help="Path to write HCIV encoder checkpoint",
    )
    parser.add_argument("--manifest", default=None, help="Optional JSON manifest path")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args(argv)

    train_hciv_encoder_checkpoint(
        args.data,
        args.output_checkpoint,
        manifest_path=args.manifest,
        dim=args.dim,
        curvature=args.curvature,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        learning_rate=args.learning_rate,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
