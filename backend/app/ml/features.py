import pandas as pd

from app.screener.indicators import compute_indicators

FEATURE_COLUMNS = [
    "rsi_norm",
    "macd_hist_norm",
    "ema_gap",
    "momentum",
    "volume_ratio",
    "price_position",
    "return_1",
    "return_3",
    "return_5",
]


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """OHLCV verisinden ML modeli için özellik matrisi üretir.

    `compute_indicators`'ı çağırıp göstergeleri ölçekli/normalize edilmiş
    özelliklere dönüştürür. Dönen DataFrame, kaynak OHLCV ile aynı index'e
    sahiptir; ilk satırlar (rolling pencereler dolmadığı için) NaN içerir.
    """
    ind = compute_indicators(ohlcv)

    features = pd.DataFrame(index=ind.index)
    features["rsi_norm"] = (ind["rsi"] - 50) / 50  # -1..1
    features["macd_hist_norm"] = (ind["macd_hist"] / ind["close"]).clip(-0.05, 0.05) * 20
    features["ema_gap"] = ((ind["ema_fast"] - ind["ema_slow"]) / ind["close"]).clip(-0.1, 0.1) * 10
    features["momentum"] = (ind["momentum"] / 20).clip(-5, 5)
    features["volume_ratio"] = (ind["volume"] / ind["volume_sma"]).clip(0, 5)

    rolling_low = ind["low"].rolling(20).min()
    rolling_high = ind["high"].rolling(20).max()
    span = (rolling_high - rolling_low).replace(0, float("nan"))
    features["price_position"] = ((ind["close"] - rolling_low) / span).clip(0, 1)

    features["return_1"] = ind["close"].pct_change(1).clip(-0.2, 0.2) * 10
    features["return_3"] = ind["close"].pct_change(3).clip(-0.2, 0.2) * 10
    features["return_5"] = ind["close"].pct_change(5).clip(-0.2, 0.2) * 10

    features["close"] = ind["close"]
    return features


def latest_feature_vector(ohlcv: pd.DataFrame) -> pd.Series | None:
    """Canlı tahmin için en son (tam dolu) özellik satırını döner."""
    features = build_features(ohlcv).dropna()
    if features.empty:
        return None
    return features.iloc[-1]
