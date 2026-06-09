"""Export FragFM generated SMILES for benchmark preparation."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from rdkit import Chem

from mf_generators.fragfm.generator import FragFMGenerator


def export_fragfm_samples(
    *,
    vocab_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    sample_count: int = 100,
    checkpoint_path: str | Path | None = None,
    rate_matrix_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    output = Path(output_path)
    if report_path is not None:
        report_output = Path(report_path)
        if output.expanduser().resolve(strict=False) == report_output.expanduser().resolve(
            strict=False
        ):
            raise ValueError("FragFM sample export report_path must differ from output_path")
        _ensure_parent_directory(report_output)
    _ensure_parent_directory(output)
    generator = FragFMGenerator(
        vocab_path=str(vocab_path),
        checkpoint_path=str(checkpoint_path or ""),
        rate_matrix_path=str(rate_matrix_path or ""),
        device=device,
    )
    molecules = asyncio.run(generator.generate(batch_size=sample_count))
    smiles = [str(molecule.smiles) for molecule in molecules if molecule.smiles]
    report = build_sample_report(
        smiles=smiles,
        requested_samples=sample_count,
        output_path=output,
        vocab_path=vocab_path,
        checkpoint_path=checkpoint_path,
        rate_matrix_path=rate_matrix_path,
    )
    temp_output = _temporary_sibling(output)
    temp_report = None
    if report_path is not None:
        temp_report = _temporary_sibling(report_output)
    try:
        temp_output.write_text(
            "\n".join(smiles) + ("\n" if smiles else ""),
            encoding="utf-8",
        )
        if report_path is not None:
            assert temp_report is not None
            temp_report.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temp_report.replace(report_output)
        temp_output.replace(output)
    except Exception:
        temp_output.unlink(missing_ok=True)
        if temp_report is not None:
            temp_report.unlink(missing_ok=True)
        raise
    return report


def _ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def build_sample_report(
    *,
    smiles: list[str],
    requested_samples: int,
    output_path: str | Path,
    vocab_path: str | Path,
    checkpoint_path: str | Path | None = None,
    rate_matrix_path: str | Path | None = None,
) -> dict[str, Any]:
    valid_smiles = [smile for smile in smiles if Chem.MolFromSmiles(smile) is not None]
    generated_count = len(smiles)
    valid_count = len(valid_smiles)
    return {
        "schema_version": "fragfm_sample_export_report.v1",
        "requested_samples": int(requested_samples),
        "generated_samples": generated_count,
        "valid_smiles": valid_count,
        "validity": valid_count / max(generated_count, 1),
        "unique_smiles": len(set(valid_smiles)),
        "uniqueness": len(set(valid_smiles)) / max(valid_count, 1),
        "output_path": str(output_path),
        "vocab_path": str(vocab_path),
        "checkpoint_path": str(checkpoint_path or ""),
        "rate_matrix_path": str(rate_matrix_path or ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export FragFM generated SMILES for benchmark preparation"
    )
    parser.add_argument("--vocab", required=True, help="FragFM vocabulary JSON artifact")
    parser.add_argument("--output", required=True, help="Output SMILES path")
    parser.add_argument("--report", default="", help="Optional JSON report output path")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--checkpoint", default="", help="Optional FragFM checkpoint artifact")
    parser.add_argument("--rate-matrix", default="", help="Optional FragFM rate matrix artifact")
    parser.add_argument("--device", default="cpu", help="Torch device")
    args = parser.parse_args(argv)

    report = export_fragfm_samples(
        vocab_path=args.vocab,
        output_path=args.output,
        report_path=args.report or None,
        sample_count=args.samples,
        checkpoint_path=args.checkpoint or None,
        rate_matrix_path=args.rate_matrix or None,
        device=args.device,
    )
    if not args.report:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
