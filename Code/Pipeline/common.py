"""Shared helpers for the FlyBrain data pipeline.

Codex's exact CSV column names have drifted between dataset releases, and
this repo can't authenticate to Codex to pin them down ahead of time.
Instead of hardcoding guessed column names, every loader here does
best-effort auto-detection against a list of known aliases and fails loudly
with the actual header printed out, so a mismatch is a two-second fix
instead of a silent bug.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import yaml

HERE = Path(__file__).resolve().parent


def load_config(path: str | Path = HERE / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(cfg: dict, key: str, **fmt) -> Path:
    """Resolve a paths.<key> entry from config.yaml. If it contains a
    placeholder like {max_in_degree}, pass the value as a keyword arg to
    fill it in (see paths.processed_dir)."""
    base = HERE
    raw = cfg["paths"][key]
    if fmt:
        raw = raw.format(**fmt)
    return (base / raw).resolve()


def find_column(columns: Iterable[str], *aliases: str) -> str | None:
    """Case/underscore-insensitive match of the first alias found in columns."""
    norm = {c.lower().replace("-", "_"): c for c in columns}
    for alias in aliases:
        a = alias.lower().replace("-", "_")
        if a in norm:
            return norm[a]
    # substring fallback, e.g. "pt_root_id" matching alias "root_id"
    for alias in aliases:
        a = alias.lower().replace("-", "_")
        for norm_col, orig in norm.items():
            if a in norm_col:
                return orig
    return None


def require_column(columns: Iterable[str], name: str, *aliases: str) -> str:
    col = find_column(columns, name, *aliases)
    if col is None:
        cols = list(columns)
        sys.exit(
            f"\nCould not find a '{name}' column.\n"
            f"Looked for aliases: {[name, *aliases]}\n"
            f"Actual columns in the file: {cols}\n\n"
            f"Fix: open the CSV header, find the right column name, and add "
            f"it to the alias list in common.py's caller, or rename it in "
            f"config.yaml if this is a configurable field.\n"
        )
    return col
