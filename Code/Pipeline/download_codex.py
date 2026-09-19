"""Download the FAFB connectome CSVs from codex.flywire.ai.

Usage:
    1. Log in at https://codex.flywire.ai, open your account page, and copy
       your "Codex API token" (not the separate CAVE/FlyWire token).
    2. Set it as an environment variable:
         set CODEX_API_TOKEN=...       (PowerShell: $env:CODEX_API_TOKEN=...)
         export CODEX_API_TOKEN=...    (macOS/Linux)
    3. python download_codex.py
"""
from __future__ import annotations

import gzip
import os
import shutil
import sys
from pathlib import Path

import requests

from common import load_config, resolve_path

API_BASE = "https://codex.flywire.ai/api"
DATASET = "fafb"

# Codex's "data_product" keys for the three files we need to build the graph.
PRODUCTS = ["classification", "coordinates", "connections_princeton"]


def download_product(data_product: str, token: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"{API_BASE}/download_resource"
    params = {"data_product": data_product, "dataset": DATASET, "api_token": token}
    dest_gz = out_dir / f"{data_product}.csv.gz"
    dest_csv = out_dir / f"{data_product}.csv"

    print(f"Downloading {data_product} ...")
    with requests.get(url, params=params, stream=True, timeout=300) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("content-type", ""):
            sys.exit(
                f"GET {r.url}\nreturned a web page instead of a file -- "
                f"your CODEX_API_TOKEN is probably missing or wrong."
            )
        with open(dest_gz, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    try:
        with gzip.open(dest_gz, "rb") as f_in, open(dest_csv, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        dest_gz.unlink()
    except gzip.BadGzipFile:
        dest_gz.rename(dest_csv)  # some products may already be served uncompressed
    print(f"  -> {dest_csv}")
    return dest_csv


def main():
    token = os.environ.get("CODEX_API_TOKEN", "")
    if not token:
        sys.exit(
            "No API token set. Get one from your account page at "
            "https://codex.flywire.ai (\"Codex API token\" field), then set "
            "it as the CODEX_API_TOKEN environment variable."
        )

    cfg = load_config()
    raw_dir = resolve_path(cfg, "raw_dir")
    for data_product in PRODUCTS:
        download_product(data_product, token, raw_dir)


if __name__ == "__main__":
    main()
