#!/usr/bin/env python3
"""Download pre-trained ADMET checkpoints from a model registry.

Currently supports downloading from a user-supplied URL or local path.
Extend `MODEL_URLS` to point at your hosted checkpoints.

Usage:
    python download_models.py                  # download all
    python download_models.py solubility       # download one
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from config import ADMET_ENDPOINTS, MODEL_DIR

# ------------------------------------------------------------------
# Map endpoint name → download URL or local zip path.
# Fill these in with your actual model hosting.
# ------------------------------------------------------------------
MODEL_URLS: dict[str, str] = {
    # "solubility": "https://your-host/models/solubility.zip",
    # "lipophilicity": "https://your-host/models/lipophilicity.zip",
    # ...
}


def download_endpoint(name: str, url: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "model.ckpt"
    if target.exists():
        print(f"  [skip] {name} — already exists at {target}")
        return

    print(f"  Downloading {name} → {dest} …")
    if url.startswith("http"):
        archive = dest / "_download.zip"
        urlretrieve(url, archive)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
        archive.unlink()
    else:
        # Local path copy
        src = Path(url)
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ADMET model checkpoints")
    parser.add_argument("endpoints", nargs="*", help="Specific endpoints to download")
    args = parser.parse_args()

    targets = args.endpoints or list(MODEL_URLS.keys())

    print(f"Model directory: {MODEL_DIR}")
    for name in targets:
        url = MODEL_URLS.get(name)
        if url is None:
            print(f"  [warn] No URL configured for '{name}' — skipping")
            continue
        dest = ADMET_ENDPOINTS[name]
        download_endpoint(name, url, dest)

    print("Done.")


if __name__ == "__main__":
    main()
