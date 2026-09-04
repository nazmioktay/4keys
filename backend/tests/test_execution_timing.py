import numpy as np
import pandas as pd

from app.exchanges.base import Exchange
from app.rl.execution_timing import analyze_hurst_execution_timing


class _TrendingExchange(Exchange):
    """Kesintisiz, monoton bir uptrend — Hurst yüksek (trend-devamlılığı)
    çıkmalı ve geciktirmek HER ZAMAN daha pahalıya mal olmalı (close
    monoton arttığı için delayed > immediate matematiksel olarak kesin)."""

    def list_symbols(self, quote_currency, market_type):
        return ["BTC/USDT:USDT"]

    def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
        n = max(limit, 1000)
        close = np.linspace(100, 300, n)  # kesin monoton, gürültüsüz
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": np.full(n, 1000.0),
            }
        )


class _FlatNoiseExchange(Exchange):
    """Yatay (trendsiz), küçük rastgele gürültülü seri — yeterli örnek
    üretip üç grubu da (low/mid/high H) dolduracak kadar çeşitlilik sağlar."""

    def list_symbols(self, quote_currency, market_type):
        return ["BTC/USDT:USDT"]

    def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
        n = max(limit, 1000)
        rng = np.random.default_rng(7)
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": rng.uniform(800, 1200, n),
            }
        )


def test_trending_series_high_hurst_bucket_shows_positive_slippage_from_delay():
    report = analyze_hurst_execution_timing(_TrendingExchange(), "BTC/USDT:USDT", "1h", 1000, delay_bars=3)
    assert report.total_samples > 0
    high_bucket = next(b for b in report.buckets if b.label == "high")
    # Monoton artan seride geciktirmek HER ZAMAN daha pahalıya mal olur.
    assert high_bucket.samples > 0
    assert high_bucket.mean_delayed_slippage_pct > 0


def test_analyze_hurst_execution_timing_runs_and_covers_buckets():
    report = analyze_hurst_execution_timing(_FlatNoiseExchange(), "BTC/USDT:USDT", "1h", 1000, delay_bars=3)
    assert report.total_samples > 0
    assert len(report.buckets) == 3
    assert {b.label for b in report.buckets} == {"low", "mid", "high"}


def test_analyze_hurst_execution_timing_insufficient_data_returns_empty():
    class _TinyExchange(Exchange):
        def list_symbols(self, quote_currency, market_type):
            return ["X"]

        def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
            close = np.linspace(100, 110, 5)
            return pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h"),
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": np.full(5, 1000.0),
                }
            )

    report = analyze_hurst_execution_timing(_TinyExchange(), "X", "1h", 5, delay_bars=3)
    assert report.total_samples == 0
    assert report.buckets == []
