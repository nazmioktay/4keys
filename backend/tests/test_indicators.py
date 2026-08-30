import numpy as np
import pandas as pd

from app.screener.indicators import compute_indicators, composite_score


def _make_trending_ohlcv(direction: str, n: int = 100) -> pd.DataFrame:
    base = np.linspace(100, 200, n) if direction == "up" else np.linspace(200, 100, n)
    noise = np.random.default_rng(42).normal(0, 0.5, n)
    close = base + noise
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    )
    return df


def test_uptrend_scores_positive_for_long():
    df = _make_trending_ohlcv("up")
    indicators = compute_indicators(df)
    score = composite_score(indicators)
    assert score > 0


def test_downtrend_scores_negative_for_short():
    df = _make_trending_ohlcv("down")
    indicators = compute_indicators(df)
    score = composite_score(indicators)
    assert score < 0
