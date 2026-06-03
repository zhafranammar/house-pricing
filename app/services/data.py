"""Akses dataset CSV."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from config import DATASET_PATH


@lru_cache(maxsize=1)
def load_raw_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATASET_PATH}. "
            "Jalankan: python scripts/download_dataset.py"
        )
    return pd.read_csv(DATASET_PATH)
