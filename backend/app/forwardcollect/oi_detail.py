"""Ayrıntılı açık pozisyon: OI + top-trader / global long-short oranları + taker alış/satış (son 5 dk)."""

from __future__ import annotations

import logging
from typing import Callable

import requests

from app.db import repository as db

logger = logging.getLogger(__name__)

FAPI = "https://fapi.binance.com"


def _default_get_json(url: str, params: dict | None = None):
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _last(rows):
    return rows[-1] if rows else None


def fetch_oi_detail(symbol: str, get_json: Callable = _default_get_json) -> dict | None:
    """Her uç nokta bağımsızdır: biri başarısız olursa o alanlar None kalır (kısmi satır). Hiçbiri alınamazsa None."""
    params = {"symbol": symbol, "period": "5m", "limit": 1}
    out: dict = {}

    def attempt(label: str, fn) -> None:
        try:
            out.update(fn())
        except Exception:
            logger.warning("%s alınamadı: %s", label, symbol, exc_info=True)

    attempt("openInterest", lambda: {"open_interest": _num(get_json(f"{FAPI}/fapi/v1/openInterest", {"symbol": symbol})["openInterest"])})
    attempt("openInterestHist", lambda: {"open_interest_value": _num(_last(get_json(f"{FAPI}/futures/data/openInterestHist", params))["sumOpenInterestValue"])})
    attempt("topLongShortAccountRatio", lambda: {"top_ls_account": _num(_last(get_json(f"{FAPI}/futures/data/topLongShortAccountRatio", params))["longShortRatio"])})
    attempt("topLongShortPositionRatio", lambda: {"top_ls_position": _num(_last(get_json(f"{FAPI}/futures/data/topLongShortPositionRatio", params))["longShortRatio"])})
    attempt("globalLongShortAccountRatio", lambda: {"global_ls_account": _num(_last(get_json(f"{FAPI}/futures/data/globalLongShortAccountRatio", params))["longShortRatio"])})

    def taker():
        row = _last(get_json(f"{FAPI}/futures/data/takerlongshortRatio", params))
        return {"taker_buy_sell_ratio": _num(row["buySellRatio"]), "taker_buy_vol": _num(row["buyVol"]), "taker_sell_vol": _num(row["sellVol"])}

    attempt("takerlongshortRatio", taker)
    return out if any(v is not None for v in out.values()) else None


def collect_oi_detail(symbols: list[str], get_json: Callable = _default_get_json) -> dict[str, dict | None]:
    results = {}
    for symbol in symbols:
        values = fetch_oi_detail(symbol, get_json)
        if values is not None:
            db.record_oi_detail_snapshot(symbol, values)
        results[symbol] = values
    return results
