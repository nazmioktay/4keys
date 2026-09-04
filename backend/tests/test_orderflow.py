import pandas as pd
import pytest

from app.ml.orderflow_features import latest_taker_buy_ratio_norm, merge_taker_flow_features


class _FakeExchangeWithTakerFlow:
    def fetch_taker_flow(self, symbol: str, timeframe: str, limit: int, since=None) -> pd.DataFrame:
        n = limit
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h").astype("datetime64[ns]"),
                "volume": [100.0] * n,
                "taker_buy_base_volume": [80.0] * n,  # %80 agresif alım -> ratio_norm = 0.6
            }
        )


class _FakeExchangeWithoutTakerFlow:
    pass


class _FakeExchangeRaises:
    def fetch_taker_flow(self, symbol: str, timeframe: str, limit: int, since=None):
        raise RuntimeError("ağ hatası")


def _sample_features(n: int) -> pd.DataFrame:
    # `app.ml.features.build_features` timestamp'i HER ZAMAN nanosaniyeye
    # zorlar (pandas 3.x'te date_range varsayılanı mikrosaniye olduğu için)
    # — burada da aynı gerçek davranış taklit ediliyor.
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h").astype("datetime64[ns]").astype("int64"),
            "rsi_norm": [0.1] * n,
        }
    )


def _sample_ohlcv(n: int) -> pd.DataFrame:
    return pd.DataFrame({"close": [100.0] * n, "volume": [100.0] * n})


def test_merge_taker_flow_features_computes_ratio():
    features = _sample_features(5)
    ohlcv = _sample_ohlcv(5)
    result = merge_taker_flow_features(features, ohlcv, _FakeExchangeWithTakerFlow(), "BTC/USDT:USDT", "1h")
    assert result["taker_buy_ratio_norm"].tolist() == pytest.approx([0.6] * 5)


def test_merge_taker_flow_features_unsupported_exchange_returns_nan_column():
    features = _sample_features(5)
    ohlcv = _sample_ohlcv(5)
    result = merge_taker_flow_features(features, ohlcv, _FakeExchangeWithoutTakerFlow(), "BTC/USDT:USDT", "1h")
    assert result["taker_buy_ratio_norm"].isna().all()


def test_merge_taker_flow_features_network_error_returns_nan_column():
    features = _sample_features(5)
    ohlcv = _sample_ohlcv(5)
    result = merge_taker_flow_features(features, ohlcv, _FakeExchangeRaises(), "BTC/USDT:USDT", "1h")
    assert result["taker_buy_ratio_norm"].isna().all()


def test_latest_taker_buy_ratio_norm_returns_scaled_value():
    value = latest_taker_buy_ratio_norm(_FakeExchangeWithTakerFlow(), "BTC/USDT:USDT", "1h")
    assert value == pytest.approx(0.6)


def test_latest_taker_buy_ratio_norm_unsupported_exchange_returns_neutral():
    value = latest_taker_buy_ratio_norm(_FakeExchangeWithoutTakerFlow(), "BTC/USDT:USDT", "1h")
    assert value == 0.0


def test_latest_taker_buy_ratio_norm_network_error_returns_neutral():
    value = latest_taker_buy_ratio_norm(_FakeExchangeRaises(), "BTC/USDT:USDT", "1h")
    assert value == 0.0
