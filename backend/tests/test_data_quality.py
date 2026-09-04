import pandas as pd

from app.ml.data_quality import count_missing_candles


def _make_ohlcv(timestamps: list) -> pd.DataFrame:
    n = len(timestamps)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.0] * n,
            "high": [1.0] * n,
            "low": [1.0] * n,
            "close": [1.0] * n,
            "volume": [1.0] * n,
        }
    )


def test_count_missing_candles_zero_for_contiguous_series():
    idx = pd.date_range("2024-01-01", periods=50, freq="1h")
    ohlcv = _make_ohlcv(list(idx))
    assert count_missing_candles(ohlcv, timeframe_minutes=60) == 0


def test_count_missing_candles_detects_gap():
    idx = list(pd.date_range("2024-01-01", periods=10, freq="1h"))
    # 5 saatlik bir boşluk bırak (4 mum eksik).
    idx = idx[:5] + [idx[5] + pd.Timedelta(hours=4)] + idx[6:]
    ohlcv = _make_ohlcv(idx)
    assert count_missing_candles(ohlcv, timeframe_minutes=60) == 4


def test_count_missing_candles_handles_short_series():
    ohlcv = _make_ohlcv([pd.Timestamp("2024-01-01")])
    assert count_missing_candles(ohlcv, timeframe_minutes=60) == 0
