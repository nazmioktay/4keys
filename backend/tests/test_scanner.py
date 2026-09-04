import numpy as np
import pandas as pd
import pytest

from app.exchanges.base import Exchange
from app.screener.scanner import _select_candidate_symbols, scan_market


class _FakeTickerExchange(Exchange):
    """Sabit bir ticker/OHLCV kümesi olan test borsası — hacim/fiyat
    ön-filtresini (bkz. `_select_candidate_symbols`) ağ çağrısı olmadan
    test etmek için."""

    def __init__(self, tickers: dict[str, dict]) -> None:
        self._tickers = tickers

    def list_symbols(self, quote_currency, market_type):
        return list(self._tickers.keys())

    def fetch_tickers(self, quote_currency, market_type):
        return dict(self._tickers)

    def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
        n = 100
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC").tz_convert(None)
        return pd.DataFrame(
            {
                "timestamp": idx,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": rng.uniform(100, 200, n),
            }
        )


def test_select_candidate_symbols_excludes_low_price(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "screener_min_price", 0.1)
    monkeypatch.setattr(settings, "screener_volume_top_pct", 100.0)

    exchange = _FakeTickerExchange(
        {
            "A/USDT:USDT": {"last": 0.05, "quote_volume": 1_000_000},  # 0.1'in altında -> elenir
            "B/USDT:USDT": {"last": 10.0, "quote_volume": 500_000},
        }
    )
    result = _select_candidate_symbols(exchange)
    assert result == ["B/USDT:USDT"]


def test_select_candidate_symbols_keeps_top_volume_percentage(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "screener_min_price", 0.0)
    monkeypatch.setattr(settings, "screener_volume_top_pct", 20.0)

    tickers = {f"S{i}/USDT:USDT": {"last": 10.0, "quote_volume": float(i)} for i in range(10)}
    exchange = _FakeTickerExchange(tickers)
    result = _select_candidate_symbols(exchange)

    # 10 sembolun %20'si = 2, hacme göre en yüksek 2 (S9, S8) tutulmalı
    assert len(result) == 2
    assert set(result) == {"S9/USDT:USDT", "S8/USDT:USDT"}


def test_select_candidate_symbols_empty_when_nothing_passes_price_floor(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "screener_min_price", 100.0)
    monkeypatch.setattr(settings, "screener_volume_top_pct", 100.0)

    exchange = _FakeTickerExchange({"A/USDT:USDT": {"last": 1.0, "quote_volume": 1000}})
    assert _select_candidate_symbols(exchange) == []


def test_scan_market_only_scores_prefiltered_symbols(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "screener_min_price", 1.0)
    monkeypatch.setattr(settings, "screener_volume_top_pct", 50.0)
    monkeypatch.setattr(settings, "feature_snapshot_symbols", "")

    tickers = {
        "LOW/USDT:USDT": {"last": 0.01, "quote_volume": 999_999_999},  # fiyat elenir
        "HIGH_VOL/USDT:USDT": {"last": 5.0, "quote_volume": 1000},
        "TOP_VOL/USDT:USDT": {"last": 5.0, "quote_volume": 5000},
    }
    exchange = _FakeTickerExchange(tickers)
    results = scan_market(exchange)

    scanned_symbols = {r.symbol for r in results}
    assert "LOW/USDT:USDT" not in scanned_symbols
    # %50 -> 2 aday arasında en yüksek hacimli olan (TOP_VOL) taranmalı
    assert "TOP_VOL/USDT:USDT" in scanned_symbols
