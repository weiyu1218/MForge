"""Command-line interface for Sigstore signing and verification.

Usage:
    # Sign a file (requires SIGSTORE_ID_TOKEN or browser OAuth)
    python cli.py sign --file /data/dataset.csv

    # Sign with explicit bundle output
    python cli.py sign --file model.ckpt --bundle model.ckpt.sigstore

    # Sign JSON data
    python cli.py sign --json '{"smiles":["CCO"],"score":0.9}' --bundle pred.sigstore

    # Verify a file
    python cli.py verify \
        --file /data/dataset.csv \
        --bundle /data/dataset.csv.sigstore \
        --identity "https://github.com/org/repo/.github/workflows/ci.yml@refs/heads/main" \
        --issuer "https://token.actions.githubusercontent.com"

    # Using GitHub Actions OIDC
    OIDC_STRATEGY=github python cli.py sign --file artifact.tar.gz
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from sigstore_manager import SigstoreConfig, SigstoreManager


def cmd_sign(args: argparse.Namespace) -> int:
    cfg = SigstoreConfig(
        env=args.env,
        oidc_strategy=args.oidc_strategy,
        bundle_dir=Path(args.bundle_dir) if args.bundle_dir else None,
    )
    mgr = SigstoreManager(config=cfg)

    try:
        if args.file:
            result = mgr.sign_file(args.file, args.bundle)
            print(f"Signed: {result.artifact_path}")
            print(f"Bundle: {result.bundle_path}")
            print(f"SHA-256: {result.digest_hex}")
            if result.rekor_log_index is not None:
                print(f"Rekor log index: {result.rekor_log_index}")
        elif args.json_data:
            data = json.loads(args.json_data)
            if not args.bundle:
                print("ERROR: --bundle is required for JSON signing", file=sys.stderr)
                return 1
            result = mgr.sign_json(data, args.bundle)
            print(f"Signed JSON → {result.bundle_path}")
            print(f"SHA-256: {result.digest_hex}")
        elif args.data_base64:
            raw = base64.b64decode(args.data_base64)
            if not args.bundle:
                print("ERROR: --bundle is required for bytes signing", file=sys.stderr)
                return 1
            result = mgr.sign_bytes(raw, args.bundle)
            print(f"Signed {len(raw)} bytes → {result.bundle_path}")
            print(f"SHA-256: {result.digest_hex}")
        else:
            print("ERROR: specify --file, --json, or --data-base64", file=sys.stderr)
            return 1

        return 0

    except Exception as exc:
        print(f"SIGNING FAILED: {exc}", file=sys.stderr)
        return 1


def cmd_verify(args: argparse.Namespace) -> int:
    cfg = SigstoreConfig(
        env=args.env,
        oidc_strategy=args.oidc_strategy,
    )
    mgr = SigstoreManager(config=cfg)

    try:
        if args.file:
            result = mgr.verify_file(args.file, args.bundle, args.identity, args.issuer)
        elif args.data_base64:
            raw = base64.b64decode(args.data_base64)
            result = mgr.verify_bytes(raw, args.bundle, args.identity, args.issuer)
        else:
            print("ERROR: specify --file or --data-base64", file=sys.stderr)
            return 1

        if result.valid:
            print("VERIFIED: artifact is authentic and untampered.")
            if result.rekor_log_index is not None:
                print(f"Rekor log index: {result.rekor_log_index}")
            return 0
        else:
            print(f"FAILED: {result.error}", file=sys.stderr)
            return 1

    except Exception as exc:
        print(f"VERIFICATION ERROR: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sigstore sign & verify CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env", default="production", choices=["production", "staging"])
    parser.add_argument("--oidc-strategy", default="env", choices=["env", "github", "interactive"])

    sub = parser.add_subparsers(dest="command", required=True)

    # -- sign --
    sp = sub.add_parser("sign", help="Sign an artifact")
    sp.add_argument("--file", help="File path to sign")
    sp.add_argument("--json", dest="json_data", help="JSON string to sign")
    sp.add_argument("--data-base64", help="Base64-encoded data to sign")
    sp.add_argument("--bundle", help="Output bundle path (required for --json/--data-base64)")
    sp.add_argument("--bundle-dir", help="Directory to write bundles into")

    # -- verify --
    vp = sub.add_parser("verify", help="Verify an artifact")
    vp.add_argument("--file", help="File path to verify")
    vp.add_argument("--data-base64", help="Base64-encoded data to verify")
    vp.add_argument("--bundle", required=True, help="Path to .sigstore bundle")
    vp.add_argument("--identity", required=True, help="Expected certificate identity")
    vp.add_argument("--issuer", required=True, help="Expected OIDC issuer")

    args = parser.parse_args()

    if args.command == "sign":
        sys.exit(cmd_sign(args))
    elif args.command == "verify":
        sys.exit(cmd_verify(args))


if __name__ == "__main__":
    main()
