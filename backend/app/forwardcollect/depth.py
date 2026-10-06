"""Emir defteri derinlik bantları: orta fiyattan ±N baz puan içindeki bid/ask NOTIONAL (USDT)."""

from __future__ import annotations

import logging
from typing import Callable

import requests

from app.db import repository as db
from app.db.models import DEPTH_BANDS_BP

logger = logging.getLogger(__name__)

FAPI = "https://fapi.binance.com"


def compute_depth_bands(bids: list, asks: list, bands_bp=DEPTH_BANDS_BP) -> dict | None:
    """`bids`: [[fiyat, miktar], ...] (en iyi önce, azalan), `asks`: artan. Borsa sınırlı seviye verdiği için kapsanan mesafe
    (`depth_coverage_bp`) hesaplanır; kapsanmayan bantlar None (EKSİK veri — sıfır DEĞİL)."""
    if not bids or not asks:
        return None
    b = [(float(p), float(q)) for p, q in bids]
    a = [(float(p), float(q)) for p, q in asks]
    best_bid, best_ask = b[0][0], a[0][0]
    if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
        return None
    mid = (best_bid + best_ask) / 2.0
    coverage = min((mid - b[-1][0]) / mid, (a[-1][0] - mid) / mid) * 1e4
    out: dict = {"mid_price": mid, "spread_bps": (best_ask - best_bid) / mid * 1e4, "depth_coverage_bp": coverage}
    for bp in bands_bp:
        if bp <= coverage:
            out[f"bid_{bp}bp"] = sum(p * q for p, q in b if p >= mid * (1 - bp / 1e4))
            out[f"ask_{bp}bp"] = sum(p * q for p, q in a if p <= mid * (1 + bp / 1e4))
        else:
            out[f"bid_{bp}bp"] = None
            out[f"ask_{bp}bp"] = None
    return out


def _default_get_json(url: str, params: dict | None = None):
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_depth_bands(symbol: str, get_json: Callable = _default_get_json) -> dict | None:
    try:
        book = get_json(f"{FAPI}/fapi/v1/depth", {"symbol": symbol, "limit": 1000})
        return compute_depth_bands(book["bids"], book["asks"])
    except Exception:  # noqa: BLE001 - toplama opsiyoneldir, ana akışı bozmamalı
        logger.warning("depth alınamadı: %s", symbol, exc_info=True)
        return None


def collect_depth(symbols: list[str], get_json: Callable = _default_get_json) -> dict[str, dict | None]:
    results = {}
    for symbol in symbols:
        values = fetch_depth_bands(symbol, get_json)
        if values is not None:
            db.record_depth_band_snapshot(symbol, values)
        results[symbol] = values
    return results
