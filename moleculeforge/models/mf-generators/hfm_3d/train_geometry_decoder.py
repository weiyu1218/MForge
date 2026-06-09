"""Train an HFM-3D neural geometry decoder artifact."""
from __future__ import annotations

import argparse

from mf_generators.hfm_3d.decoder.neural_geometry_decoder import (
    train_geometry_decoder_artifact,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decoder-artifact",
        required=True,
        help="Source HFM decoder JSON artifact with latent + SDF entries",
    )
    parser.add_argument(
        "--output-artifact",
        required=True,
        help="Path to write the neural geometry decoder .pt artifact",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-atoms", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args(argv)

    train_geometry_decoder_artifact(
        args.decoder_artifact,
        args.output_artifact,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        max_atoms=args.max_atoms,
        learning_rate=args.learning_rate,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
