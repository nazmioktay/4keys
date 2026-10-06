"""Kaynak panel önbelleği: parquet, anahtar = kaynak + versiyon + parametre hash'i + veri anlık görüntü hash'i + evren/tarih."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import pandas as pd

from .. import config
from ..guard import current_zone
from .base import Source


def cache_key(source: Source, data_hash: str, universe: list[str], dates: pd.DatetimeIndex) -> str:
    parts = [source.param_hash(), data_hash, source.data_signature(), ",".join(sorted(universe)), str(dates.min()), str(dates.max()), str(len(dates))]
    zone = current_zone()
    if zone != "main":  # bölge kesimi panel içeriğini değiştirir; "main" eklenmez -> mevcut önbellek anahtarları geçerli kalır
        parts.append(f"zone={zone}")
    blob = "|".join(parts)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cache_path(source: Source, key: str) -> Path:
    return Path(config.CACHE_DIR) / "sources" / source.name / f"v{source.version}" / f"{key}.parquet"


def load_or_build(
    source: Source, universe: list[str], dates: pd.DatetimeIndex, data_hash: str, builder: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    """Önbellekte varsa okur, yoksa `builder()` (varsayılan `source.to_panel`) ile üretip yazar."""
    path = cache_path(source, cache_key(source, data_hash, universe, dates))
    if path.exists():
        return pd.read_parquet(path)
    panel = (builder or (lambda: source.to_panel(universe, dates)))()
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path)
    return panel
