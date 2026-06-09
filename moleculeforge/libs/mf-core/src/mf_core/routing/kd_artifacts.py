"""Teacher embedding artifact export and preflight helpers for KD training."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "cross_paradigm_teacher_embeddings.v1"


def export_teacher_embeddings_artifact(
    input_path: str | Path,
    output_path: str | Path,
    *,
    embedding_field: str = "teacher_embedding",
    expected_dim: int | None = None,
    min_embeddings: int = 1,
) -> dict[str, Any]:
    """Export canonical teacher_embeddings JSON from JSON/JSONL records."""

    embeddings, report = _collect_teacher_embeddings(
        input_path,
        embedding_field=embedding_field,
        expected_dim=expected_dim,
        min_embeddings=min_embeddings,
    )
    if report["status"] != "pass":
        raise ValueError("; ".join(report["messages"]))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "embedding_count": len(embeddings),
        "embedding_dim": len(embeddings[0]) if embeddings else 0,
        "teacher_embeddings": embeddings,
    }
    Path(output_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return build_teacher_embeddings_report(
        output_path,
        expected_dim=expected_dim,
        min_embeddings=min_embeddings,
    )


def build_teacher_embeddings_report(
    artifact_path: str | Path,
    *,
    embedding_field: str = "teacher_embedding",
    expected_dim: int | None = None,
    min_embeddings: int = 1,
) -> dict[str, Any]:
    """Return a JSON-serializable preflight report for teacher embeddings."""

    _embeddings, report = _collect_teacher_embeddings(
        artifact_path,
        embedding_field=embedding_field,
        expected_dim=expected_dim,
        min_embeddings=min_embeddings,
    )
    return report


def _collect_teacher_embeddings(
    path_value: str | Path,
    *,
    embedding_field: str,
    expected_dim: int | None,
    min_embeddings: int,
) -> tuple[list[list[float]], dict[str, Any]]:
    if not isinstance(embedding_field, str) or not embedding_field:
        raise ValueError("embedding_field must be a non-empty string")
    if expected_dim is not None and expected_dim <= 0:
        raise ValueError("expected_dim must be positive")
    if min_embeddings <= 0:
        raise ValueError("min_embeddings must be positive")

    path = Path(path_value)
    messages: list[str] = []
    embeddings: list[list[float]] = []
    embedding_dim = 0

    try:
        payload = _load_json_or_jsonl(path)
        raw_embeddings = _extract_embeddings(payload, embedding_field=embedding_field)
    except Exception as exc:
        raw_embeddings = []
        messages.append(str(exc))

    for index, raw_embedding in enumerate(raw_embeddings):
        normalized = _normalize_embedding(raw_embedding)
        if normalized is None:
            messages.append(f"teacher embedding {index} must be a finite numeric sequence")
            continue
        if embedding_dim == 0:
            embedding_dim = len(normalized)
        elif len(normalized) != embedding_dim:
            messages.append(
                f"teacher embedding {index} dimension {len(normalized)} "
                f"does not match first embedding dimension {embedding_dim}"
            )
            continue
        embeddings.append(normalized)

    if not embeddings:
        messages.append("teacher embedding artifact contains no valid embeddings")
    if embeddings and len(embeddings) < min_embeddings:
        messages.append(
            f"teacher embedding count {len(embeddings)} is below required {min_embeddings}"
        )
    if embeddings and expected_dim is not None and embedding_dim != expected_dim:
        messages.append(
            f"teacher embedding dimension {embedding_dim} does not match expected {expected_dim}"
        )

    status = "pass" if not messages else "fail"
    report = {
        "schema_version": "cross_paradigm_teacher_embeddings_report.v1",
        "status": status,
        "artifact_path": str(path),
        "embedding_field": embedding_field,
        "embedding_count": len(embeddings),
        "embedding_dim": embedding_dim,
        "expected_dim": expected_dim,
        "min_embeddings": min_embeddings,
        "messages": messages,
    }
    return embeddings, report


def _load_json_or_jsonl(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"teacher embedding input not found: {path}")
    if path.suffix == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_embeddings(payload: object, *, embedding_field: str) -> list[object]:
    if isinstance(payload, Mapping):
        value = payload.get("teacher_embeddings") or payload.get("embeddings")
        if value is not None:
            if not isinstance(value, list):
                raise ValueError("teacher_embeddings must be a list")
            return list(value)
        if embedding_field in payload:
            return [payload[embedding_field]]
        raise ValueError("teacher embedding payload requires teacher_embeddings or embedding field")
    if isinstance(payload, list):
        if not payload:
            return []
        if all(isinstance(item, Mapping) for item in payload):
            embeddings = []
            for index, item in enumerate(payload):
                if embedding_field not in item:
                    raise ValueError(
                        f"teacher embedding record {index} lacks field {embedding_field!r}"
                    )
                embeddings.append(item[embedding_field])
            return embeddings
        return list(payload)
    raise ValueError("teacher embedding payload must be JSON object or list")


def _normalize_embedding(value: object) -> list[float] | None:
    if isinstance(value, str | bytes | bytearray):
        return None
    if not isinstance(value, Sequence):
        return None
    try:
        embedding = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not embedding or not all(math.isfinite(item) for item in embedding):
        return None
    return embedding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-paradigm KD teacher artifact utility")
    parser.add_argument("--input", required=True, help="Teacher embedding JSON or JSONL input")
    parser.add_argument("--output", default="", help="Optional canonical artifact output path")
    parser.add_argument(
        "--embedding-field",
        default="teacher_embedding",
        help="Record field to read when input is JSONL or a list of records",
    )
    parser.add_argument("--expected-dim", type=int, default=None)
    parser.add_argument("--min-embeddings", type=int, default=1)
    parser.add_argument("--report", default="", help="Optional report JSON path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when the report status is fail",
    )
    args = parser.parse_args(argv)

    if args.output:
        embeddings, report = _collect_teacher_embeddings(
            args.input,
            embedding_field=args.embedding_field,
            expected_dim=args.expected_dim,
            min_embeddings=args.min_embeddings,
        )
        if report["status"] == "pass":
            payload = {
                "schema_version": SCHEMA_VERSION,
                "embedding_count": len(embeddings),
                "embedding_dim": len(embeddings[0]) if embeddings else 0,
                "teacher_embeddings": embeddings,
            }
            Path(args.output).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    else:
        report = build_teacher_embeddings_report(
            args.input,
            embedding_field=args.embedding_field,
            expected_dim=args.expected_dim,
            min_embeddings=args.min_embeddings,
        )

    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        Path(args.report).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    if args.strict and report["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
