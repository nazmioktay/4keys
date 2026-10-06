"""Parquet önbelleği + nihai pencere kilidi + anlık görüntü hash'i.

Düzen: `CACHE_DIR/<kind>/<SEMBOL>.parquet`; kind'lar `um_1d`, `um_1h`, `spot_1d`, `um_funding`,
`delivery_um_1d`, `delivery_cm_1d`. Önbellek nihai pencere sonrasını DA saklar (ham arşiv);
YÜKLEYİCİLER varsayılan olarak onu keser (`research.guard`)."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pandas as pd

from .. import config
from ..guard import cut_final_test

KLINE_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
_MANIFEST_LOCK = threading.Lock()
KINDS = ("um_1d", "um_1h", "spot_1d", "um_funding", "delivery_um_1d", "delivery_cm_1d")


def cache_dir() -> Path:
    return Path(config.CACHE_DIR)


def _path(kind: str, symbol: str) -> Path:
    if kind not in KINDS:
        raise ValueError(f"bilinmeyen kind: {kind}")
    return cache_dir() / kind / f"{symbol}.parquet"


def _manifest_path() -> Path:
    return cache_dir() / "manifest.json"


def _load_manifest() -> dict:
    try:
        return json.loads(_manifest_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_frame(kind: str, symbol: str, df: pd.DataFrame) -> Path:
    """DataFrame'i parquet'e yazar (indeks korunur) ve manifest'e sha256 kaydeder."""
    path = _path(kind, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    digest = _sha256(path)
    with _MANIFEST_LOCK:  # eşzamanlı indirmelerde manifest yarışını önler
        manifest = _load_manifest()
        manifest[f"{kind}/{symbol}.parquet"] = digest
        _manifest_path().write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


def read_frame(kind: str, symbol: str) -> pd.DataFrame | None:
    path = _path(kind, symbol)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def list_symbols(kind: str) -> list[str]:
    d = cache_dir() / kind
    return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []


def snapshot_hash() -> str:
    """Önbelleğin içerik özeti: manifest'teki (yol, sha256) çiftlerinin sıralı sha256'sı."""
    manifest = _load_manifest()
    h = hashlib.sha256()
    for key in sorted(manifest):
        h.update(f"{key}:{manifest[key]}\n".encode())
    return h.hexdigest()[:16]


def load_klines(
    symbol: str,
    kind: str = "um_1d",
    allow_final_test: bool = False,
    final_test_experiment_id: str | None = None,
) -> pd.DataFrame | None:
    """Sembol klines'ı (indeks = open_time). Varsayılan: nihai pencere (>= 2025-10-01) KESİLİR."""
    df = read_frame(kind, symbol)
    if df is None:
        return None
    return cut_final_test(df, allow_final_test, final_test_experiment_id)


def load_funding(
    symbol: str, allow_final_test: bool = False, final_test_experiment_id: str | None = None
) -> pd.DataFrame | None:
    """Funding olayları (indeks = ödeme zamanı; kolonlar: rate, interval_hours)."""
    df = read_frame("um_funding", symbol)
    if df is None:
        return None
    return cut_final_test(df, allow_final_test, final_test_experiment_id)


def load_panel(
    field: str,
    symbols: list[str] | None = None,
    kind: str = "um_1d",
    allow_final_test: bool = False,
    final_test_experiment_id: str | None = None,
) -> pd.DataFrame:
    """tarih × sembol panel (ör. field='close'|'open'|'quote_volume'). Eksik günler NaN kalır — doldurulmaz."""
    symbols = symbols if symbols is not None else list_symbols(kind)
    columns = {}
    for s in symbols:
        df = load_klines(s, kind, allow_final_test, final_test_experiment_id)
        if df is not None and field in df.columns and len(df):
            columns[s] = df[field]
    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(columns).sort_index()
