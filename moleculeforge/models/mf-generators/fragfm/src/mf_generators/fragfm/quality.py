"""FragFM artifact quality reporting for shared HUMU conditioning."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mf_core.geometry import normalize_lorentz_embedding


def build_quality_report(
    *,
    vocab_path: str | Path,
    checkpoint_path: str | Path | None = None,
    rate_matrix_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    min_humu_coverage: float = 1.0,
    expected_humu_dim: int = 129,
    curvature: float = 1.0,
) -> dict[str, Any]:
    """Return a JSON-serializable local quality report for FragFM artifacts."""

    if not 0.0 <= min_humu_coverage <= 1.0:
        raise ValueError("min_humu_coverage must be in [0, 1]")
    if expected_humu_dim <= 1:
        raise ValueError("expected_humu_dim must be greater than 1")
    if curvature <= 0.0:
        raise ValueError("curvature must be positive")

    messages: list[str] = []
    vocab = _load_vocab(Path(vocab_path))
    fragments = vocab["fragments"]
    rules = vocab["assembly_rules"]
    rules_count = len(rules)

    humu_embedding_count = 0
    invalid_humu_embeddings = 0
    for rule in rules:
        if "humu_embedding" not in rule:
            continue
        normalized = normalize_lorentz_embedding(
            rule.get("humu_embedding"),
            expected_dim=expected_humu_dim,
            curvature=curvature,
        )
        if normalized is None:
            invalid_humu_embeddings += 1
            messages.append(f"invalid HUMU embedding for rule {rule.get('id', '<unknown>')}")
        else:
            humu_embedding_count += 1

    humu_embedding_coverage = humu_embedding_count / max(rules_count, 1)
    if humu_embedding_coverage < min_humu_coverage:
        messages.append(
            "HUMU embedding coverage "
            f"{humu_embedding_coverage:.4f} is below required {min_humu_coverage:.4f}"
        )

    checkpoint_loadable = _loadable_checkpoint(
        checkpoint_path,
        vocab_size=len(fragments),
        messages=messages,
    )
    rate_matrix_loadable = _loadable_rate_matrix(
        rate_matrix_path,
        vocab_size=len(fragments),
        messages=messages,
    )
    manifest_consistent = _check_manifest_consistency(
        manifest_path,
        vocab_path=vocab_path,
        checkpoint_path=checkpoint_path,
        rate_matrix_path=rate_matrix_path,
        rules_count=rules_count,
        fragment_count=len(fragments),
        humu_embedding_count=humu_embedding_count,
        humu_embedding_coverage=humu_embedding_coverage,
        expected_humu_dim=expected_humu_dim,
        curvature=curvature,
        messages=messages,
    )

    status = "pass"
    if (
        invalid_humu_embeddings
        or humu_embedding_coverage < min_humu_coverage
        or not checkpoint_loadable
        or not rate_matrix_loadable
        or not manifest_consistent
    ):
        status = "fail"

    return {
        "schema_version": "fragfm_quality_report.v1",
        "status": status,
        "vocab_path": str(vocab_path),
        "checkpoint_path": str(checkpoint_path or ""),
        "rate_matrix_path": str(rate_matrix_path or ""),
        "manifest_path": str(manifest_path or ""),
        "rules": rules_count,
        "fragments": len(fragments),
        "humu_embedding_count": humu_embedding_count,
        "humu_embedding_coverage": humu_embedding_coverage,
        "invalid_humu_embeddings": invalid_humu_embeddings,
        "min_humu_coverage": float(min_humu_coverage),
        "expected_humu_dim": int(expected_humu_dim),
        "curvature": float(curvature),
        "checkpoint_loadable": checkpoint_loadable,
        "rate_matrix_loadable": rate_matrix_loadable,
        "manifest_consistent": manifest_consistent,
        "messages": messages,
    }


def _load_vocab(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"FragFM vocabulary artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("FragFM vocabulary artifact must be a JSON object")
    fragments = payload.get("fragments")
    rules = payload.get("assembly_rules")
    if not isinstance(fragments, list) or not fragments:
        raise ValueError("FragFM vocabulary artifact requires non-empty fragments")
    if not isinstance(rules, list) or not rules:
        raise ValueError("FragFM vocabulary artifact requires non-empty assembly_rules")
    if not all(isinstance(fragment, str) and fragment for fragment in fragments):
        raise ValueError("FragFM vocabulary fragments must be non-empty strings")
    for rule in rules:
        if not isinstance(rule, Mapping):
            raise ValueError("FragFM assembly rules must be JSON objects")
    return {"fragments": fragments, "assembly_rules": rules}


def _loadable_checkpoint(
    path: str | Path | None,
    *,
    vocab_size: int,
    messages: list[str],
) -> bool:
    if path in (None, ""):
        messages.append("FragFM checkpoint path is missing")
        return False
    artifact_path = Path(path)
    if not artifact_path.is_file():
        messages.append(f"FragFM checkpoint artifact not found: {artifact_path}")
        return False
    try:
        import torch

        state = torch.load(artifact_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - exact torch errors vary
        messages.append(f"FragFM checkpoint failed to load: {exc}")
        return False
    if not isinstance(state, Mapping):
        messages.append("FragFM checkpoint must load to a state-dict mapping")
        return False
    weight = state.get("fragment_encoder.weight")
    if weight is None or not hasattr(weight, "shape"):
        messages.append("FragFM checkpoint requires fragment_encoder.weight")
        return False
    weight_shape = tuple(int(dim) for dim in weight.shape)
    if not weight_shape or weight_shape[0] != vocab_size:
        messages.append(
            "FragFM checkpoint fragment vocabulary size "
            f"{weight_shape[0] if weight_shape else '<scalar>'} "
            f"does not match vocab artifact {vocab_size}"
        )
        return False
    return True


def _loadable_rate_matrix(
    path: str | Path | None,
    *,
    vocab_size: int,
    messages: list[str],
) -> bool:
    if path in (None, ""):
        messages.append("FragFM rate matrix path is missing")
        return False
    artifact_path = Path(path)
    if not artifact_path.is_file():
        messages.append(f"FragFM rate matrix artifact not found: {artifact_path}")
        return False
    try:
        import torch

        state = torch.load(artifact_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - exact torch errors vary
        messages.append(f"FragFM rate matrix failed to load: {exc}")
        return False
    if not isinstance(state, Mapping):
        messages.append("FragFM rate matrix must load to a state-dict mapping")
        return False
    base_rate = state.get("base_rate")
    if base_rate is None or not hasattr(base_rate, "shape"):
        messages.append("FragFM rate matrix requires base_rate")
        return False
    shape = tuple(int(dim) for dim in base_rate.shape)
    if shape != (vocab_size, vocab_size):
        messages.append(
            "FragFM rate matrix shape "
            f"{shape} does not match vocab artifact {(vocab_size, vocab_size)}"
        )
        return False
    sa_embedding = state.get("sa_score_embedding.weight")
    if sa_embedding is None or not hasattr(sa_embedding, "shape"):
        messages.append("FragFM rate matrix requires sa_score_embedding.weight")
        return False
    sa_shape = tuple(int(dim) for dim in sa_embedding.shape)
    expected_sa_shape = (10, vocab_size * vocab_size)
    if sa_shape != expected_sa_shape:
        messages.append(
            "FragFM rate matrix sa_score_embedding.weight shape "
            f"{sa_shape} does not match expected {expected_sa_shape}"
        )
        return False
    return True


def _check_manifest_consistency(
    path: str | Path | None,
    *,
    vocab_path: str | Path,
    checkpoint_path: str | Path | None,
    rate_matrix_path: str | Path | None,
    rules_count: int,
    fragment_count: int,
    humu_embedding_count: int,
    humu_embedding_coverage: float,
    expected_humu_dim: int,
    curvature: float,
    messages: list[str],
) -> bool:
    if path in (None, ""):
        return True
    manifest_path = Path(path)
    if not manifest_path.is_file():
        messages.append(f"FragFM training manifest not found: {manifest_path}")
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        messages.append(f"FragFM training manifest failed to load: {exc}")
        return False
    if not isinstance(payload, Mapping):
        messages.append("FragFM training manifest must be a JSON object")
        return False

    consistent = True
    if payload.get("schema_version") != "fragfm_training.v1":
        messages.append(
            "FragFM training manifest schema_version must be fragfm_training.v1"
        )
        consistent = False
    consistent = _manifest_int_matches(
        payload,
        key="records",
        expected=rules_count,
        messages=messages,
    ) and consistent
    consistent = _manifest_int_matches(
        payload,
        key="fragments",
        expected=fragment_count,
        messages=messages,
    ) and consistent
    consistent = _manifest_int_matches(
        payload,
        key="humu_embedding_count",
        expected=humu_embedding_count,
        messages=messages,
    ) and consistent
    consistent = _manifest_float_matches(
        payload,
        key="humu_embedding_coverage",
        expected=humu_embedding_coverage,
        messages=messages,
    ) and consistent
    consistent = _manifest_int_matches(
        payload,
        key="humu_embedding_dim",
        expected=expected_humu_dim,
        messages=messages,
    ) and consistent
    consistent = _manifest_float_matches(
        payload,
        key="humu_curvature",
        expected=curvature,
        messages=messages,
    ) and consistent
    consistent = _manifest_path_matches(
        payload,
        key="vocab_path",
        expected=vocab_path,
        messages=messages,
    ) and consistent
    consistent = _manifest_path_matches(
        payload,
        key="checkpoint_path",
        expected=checkpoint_path,
        messages=messages,
    ) and consistent
    consistent = _manifest_path_matches(
        payload,
        key="rate_matrix_path",
        expected=rate_matrix_path,
        messages=messages,
    ) and consistent
    return consistent


def _manifest_int_matches(
    payload: Mapping[str, Any],
    *,
    key: str,
    expected: int,
    messages: list[str],
) -> bool:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        messages.append(f"FragFM training manifest requires integer {key}")
        return False
    if value != expected:
        messages.append(
            f"FragFM training manifest {key}={value} does not match artifact {expected}"
        )
        return False
    return True


def _manifest_float_matches(
    payload: Mapping[str, Any],
    *,
    key: str,
    expected: float,
    messages: list[str],
) -> bool:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        messages.append(f"FragFM training manifest requires numeric {key}")
        return False
    numeric_value = float(value)
    if abs(numeric_value - float(expected)) > 1e-9:
        messages.append(
            f"FragFM training manifest {key}={numeric_value} "
            f"does not match artifact {float(expected)}"
        )
        return False
    return True


def _manifest_path_matches(
    payload: Mapping[str, Any],
    *,
    key: str,
    expected: str | Path | None,
    messages: list[str],
) -> bool:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        messages.append(f"FragFM training manifest requires path {key}")
        return False
    if expected in (None, ""):
        messages.append(
            f"FragFM training manifest {key}={value} has no matching quality input"
        )
        return False
    manifest_artifact_path = Path(value).expanduser().resolve(strict=False)
    expected_artifact_path = Path(expected).expanduser().resolve(strict=False)
    if manifest_artifact_path != expected_artifact_path:
        messages.append(
            f"FragFM training manifest {key}={value} "
            f"does not match quality input {expected}"
        )
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FragFM shared HUMU artifact quality report")
    parser.add_argument("--vocab", required=True, help="FragFM vocabulary JSON artifact")
    parser.add_argument("--checkpoint", default="", help="FragFM model checkpoint artifact")
    parser.add_argument("--rate-matrix", default="", help="FragFM SA-aware rate matrix artifact")
    parser.add_argument("--manifest", default="", help="FragFM training manifest artifact")
    parser.add_argument(
        "--min-humu-coverage",
        type=float,
        default=1.0,
        help="Minimum required fraction of rules with valid HUMU embeddings",
    )
    parser.add_argument("--humu-dim", type=int, default=129, help="Expected HUMU dimension")
    parser.add_argument("--curvature", type=float, default=1.0, help="Lorentz curvature")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when the report status is fail",
    )
    args = parser.parse_args(argv)

    report = build_quality_report(
        vocab_path=args.vocab,
        checkpoint_path=args.checkpoint,
        rate_matrix_path=args.rate_matrix,
        manifest_path=args.manifest,
        min_humu_coverage=args.min_humu_coverage,
        expected_humu_dim=args.humu_dim,
        curvature=args.curvature,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        _write_report_atomic(Path(args.output), encoded + "\n")
    else:
        print(encoded)
    if args.strict and report["status"] != "pass":
        return 1
    return 0


def _write_report_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
