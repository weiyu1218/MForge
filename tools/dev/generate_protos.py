"""Generate gRPC/protobuf Python stubs using grpcio-tools.

Replaces ``buf generate`` for environments where buf CLI is unavailable.
Output mirrors the path set in buf.gen.yaml:
  libs/mf-core/src/mf_core/proto_gen/

Usage::

    # From the moleculeforge/ root:
    python tools/dev/generate_protos.py

Or via Makefile::

    make proto-gen

Requires: grpcio-tools  (pip install grpcio-tools)
"""

from __future__ import annotations

import subprocess
import sys
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to this script's parent-parent = moleculeforge/)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]          # moleculeforge/
PROTO_ROOT = ROOT / "protos"
OUT_DIR = ROOT / "libs" / "mf-core" / "src" / "mf_core" / "proto_gen"

# All .proto files to compile
PROTO_FILES = sorted(PROTO_ROOT.rglob("*.proto"))


def _grpc_tools_include() -> Path:
    """Return the path to grpcio-tools' bundled google/protobuf headers."""
    try:
        from grpc_tools import protoc as _  # noqa: F401
        import grpc_tools
        return Path(grpc_tools.__file__).parent / "_proto"
    except ImportError:
        print("ERROR: grpcio-tools not found.  Install with:")
        print("    pip install grpcio-tools")
        sys.exit(1)


def generate() -> None:
    google_include = _grpc_tools_include()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure the output directory is importable
    init = OUT_DIR / "__init__.py"
    if not init.exists():
        init.write_text("# auto-generated\n")

    rel_files = [str(p.relative_to(PROTO_ROOT)) for p in PROTO_FILES]
    print(f"Found {len(PROTO_FILES)} .proto files.")
    print(f"Output dir: {OUT_DIR}")

    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={PROTO_ROOT}",
        f"--proto_path={google_include}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        f"--pyi_out={OUT_DIR}",
        *rel_files,
    ]

    print("Running protoc …")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        sys.exit(result.returncode)

    # grpcio-tools uses absolute imports; patch them to relative-safe style
    _fix_imports(OUT_DIR)
    print("Done.  Stubs written to:", OUT_DIR)


def _fix_imports(out_dir: Path) -> None:
    """Rewrite generated imports to use the mf_core.proto_gen package."""
    for d in out_dir.rglob("*/"):
        init = d / "__init__.py"
        if not init.exists():
            init.write_text("# auto-generated\n")
    pattern = re.compile(r"^from moleculeforge(\.[\w.]+) import (.+)$", re.MULTILINE)
    for path in out_dir.rglob("*_pb2_grpc.py"):
        text = path.read_text()
        rewritten = pattern.sub(r"from mf_core.proto_gen.moleculeforge\1 import \2", text)
        rewritten = rewritten.replace(
            "raise NotImplementedError('Method not implemented!')",
            "raise RuntimeError('Method not implemented!')",
        )
        if rewritten != text:
            path.write_text(rewritten)


if __name__ == "__main__":
    generate()
