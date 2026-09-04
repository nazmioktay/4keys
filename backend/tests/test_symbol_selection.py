import numpy as np
import pandas as pd
import pytest

from app.core.config import settings
from app.exchanges.base import Exchange
from app.ml.symbol_selection import select_training_symbols


class _FakeExchange(Exchange):
    """`select_training_symbols` testleri için: her sembol için önceden
    tanımlı bir kapanış/hacim serisi döner."""

    def __init__(self, series: dict[str, pd.DataFrame]) -> None:
        self._series = series

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        return list(self._series.keys())

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        return self._series[symbol]


def _series_from_returns(returns: np.ndarray, volume: float) -> pd.DataFrame:
    close = 100 * np.cumprod(1 + returns)
    close = np.insert(close, 0, 100.0)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(close), freq="1h"),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(len(close), volume),
        }
    )


@pytest.fixture(autouse=True)
def _default_thresholds(monkeypatch):
    monkeypatch.setattr(settings, "ml_primary_symbol", "BTC/USDT:USDT")
    monkeypatch.setattr(settings, "ml_train_max_symbols", 5)
    monkeypatch.setattr(settings, "ml_min_correlation_with_primary", 0.4)
    monkeypatch.setattr(settings, "ml_min_quote_volume_24h", 1_000_000.0)


def test_primary_symbol_always_included_first():
    rng = np.random.default_rng(0)
    btc_returns = rng.normal(0, 0.01, 99)
    series = {"BTC/USDT:USDT": _series_from_returns(btc_returns, volume=1_000_000)}
    exchange = _FakeExchange(series)

    selected = select_training_symbols(exchange, candidates=[])
    assert selected == ["BTC/USDT:USDT"]


def test_low_liquidity_symbol_excluded():
    rng = np.random.default_rng(1)
    btc_returns = rng.normal(0, 0.01, 99)
    # ETH, BTC ile birebir aynı getiriyi izliyor (korelasyon ~1) ama hacmi çok düşük.
    series = {
        "BTC/USDT:USDT": _series_from_returns(btc_returns, volume=1_000_000),
        "ETH/USDT:USDT": _series_from_returns(btc_returns, volume=1.0),
    }
    exchange = _FakeExchange(series)

    selected = select_training_symbols(exchange, candidates=["ETH/USDT:USDT"])
    assert selected == ["BTC/USDT:USDT"]


def test_uncorrelated_symbol_excluded():
    rng = np.random.default_rng(2)
    btc_returns = rng.normal(0, 0.01, 99)
    unrelated_returns = rng.normal(0, 0.01, 99)  # bağımsız rastgele seri -> düşük korelasyon
    series = {
        "BTC/USDT:USDT": _series_from_returns(btc_returns, volume=1_000_000),
        "XYZ/USDT:USDT": _series_from_returns(unrelated_returns, volume=1_000_000),
    }
    exchange = _FakeExchange(series)

    selected = select_training_symbols(exchange, candidates=["XYZ/USDT:USDT"])
    assert selected == ["BTC/USDT:USDT"]


def test_liquid_and_correlated_symbol_included():
    rng = np.random.default_rng(3)
    btc_returns = rng.normal(0, 0.01, 99)
    # ETH getirisi BTC'ye yüksek korelasyonlu (aynı seri + küçük gürültü).
    eth_returns = btc_returns + rng.normal(0, 0.001, 99)
    series = {
        "BTC/USDT:USDT": _series_from_returns(btc_returns, volume=1_000_000),
        "ETH/USDT:USDT": _series_from_returns(eth_returns, volume=1_000_000),
    }
    exchange = _FakeExchange(series)

    selected = select_training_symbols(exchange, candidates=["ETH/USDT:USDT"])
    assert selected == ["BTC/USDT:USDT", "ETH/USDT:USDT"]


def test_max_symbols_cap_keeps_highest_correlation_first():
    rng = np.random.default_rng(4)
    btc_returns = rng.normal(0, 0.01, 99)
    series = {"BTC/USDT:USDT": _series_from_returns(btc_returns, volume=1_000_000)}
    candidates = []
    for i in range(4):
        noise_scale = 0.001 * (i + 1)  # daha yüksek i -> daha düşük korelasyon
        symbol = f"ALT{i}/USDT:USDT"
        series[symbol] = _series_from_returns(btc_returns + rng.normal(0, noise_scale, 99), volume=1_000_000)
        candidates.append(symbol)
    exchange = _FakeExchange(series)

    selected = select_training_symbols(exchange, candidates=candidates, max_symbols=3)
    assert selected[0] == "BTC/USDT:USDT"
    assert len(selected) == 3
    # En yüksek korelasyonlu iki aday (ALT0, ALT1) seçilmeli.
    assert set(selected[1:]) == {"ALT0/USDT:USDT", "ALT1/USDT:USDT"}
