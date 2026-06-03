"""Membaca artefak CSV hasil training di static/assets."""

from __future__ import annotations

import csv
from pathlib import Path

from config import ASSETS_DIR


def asset_path(name: str) -> Path:
    return ASSETS_DIR / name


def asset_exists(name: str) -> bool:
    return asset_path(name).exists()


def load_csv_rows(filename: str) -> list[dict]:
    path = asset_path(filename)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
