import numpy as np
import pandas as pd

from app.screener.indicators import compute_indicators

from .advanced_indicators import (
    dynamic_support_resistance,
    heikin_ashi,
    linear_regression_channel,
    mavilim_w,
    nadaraya_watson_envelope,
    pmax,
    stoch_rsi_log,
    wavetrend,
)

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
    # --- Kullanıcının manuel işlemde kullandığı ek göstergeler ---
    "ha_trend",
    "ha_body_pct",
    "stoch_rsi_k",
    "stoch_rsi_d",
    "mavilim_gap",
    "pmax_trend",
    "pmax_dist_pct",
    "linreg_zscore",
    "linreg_slope_norm",
    "wt_diff_norm",
    "wt_cross",
    "nwe_position",
    "sr_dist_support_pct",
    "sr_dist_resistance_pct",
    "sr_level_count_norm",
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

    # --- Heikin Ashi ---
    ha = heikin_ashi(ohlcv)
    features["ha_trend"] = np.where(ha["ha_close"] > ha["ha_open"], 1.0, -1.0)
    features["ha_body_pct"] = ((ha["ha_close"] - ha["ha_open"]) / ohlcv["close"]).clip(-0.05, 0.05) * 20

    # --- Stochastic RSI (log-getiri) ---
    sr = stoch_rsi_log(ohlcv["close"])
    features["stoch_rsi_k"] = sr["stoch_rsi_k"] * 2 - 1  # 0..1 -> -1..1
    features["stoch_rsi_d"] = sr["stoch_rsi_d"] * 2 - 1

    # --- MavilimW ---
    mav = mavilim_w(ohlcv["close"])
    features["mavilim_gap"] = ((ohlcv["close"] - mav) / ohlcv["close"]).clip(-0.1, 0.1) * 10

    # --- PMax ---
    pm = pmax(ohlcv)
    features["pmax_trend"] = pm["pmax_trend"]
    features["pmax_dist_pct"] = ((ohlcv["close"] - pm["pmax"]) / ohlcv["close"]).clip(-0.1, 0.1) * 10

    # --- Doğrusal Regresyon Kanalı ---
    lr = linear_regression_channel(ohlcv["close"], length=min(100, max(len(ohlcv) // 2, 10)))
    lr_std_safe = lr["linreg_std"].replace(0, float("nan"))
    features["linreg_zscore"] = ((ohlcv["close"] - lr["linreg_mid"]) / lr_std_safe).clip(-5, 5)
    features["linreg_slope_norm"] = (lr["linreg_slope"] / ohlcv["close"] * 100).clip(-5, 5)

    # --- WaveTrend (LazyBear) ---
    wt = wavetrend(ohlcv)
    features["wt_diff_norm"] = (wt["wt1"] - wt["wt2"]).clip(-50, 50) / 10
    features["wt_cross"] = wt["wt_cross"]

    # --- Nadaraya-Watson Envelope (causal) ---
    nwe = nadaraya_watson_envelope(ohlcv["close"])
    nwe_span = (nwe["nwe_upper"] - nwe["nwe_lower"]).replace(0, float("nan"))
    features["nwe_position"] = (((ohlcv["close"] - nwe["nwe_mid"]) / nwe_span) * 2).clip(-3, 3)

    # --- Dynamic Support/Resistance ---
    sr_levels = dynamic_support_resistance(ohlcv)
    features["sr_dist_support_pct"] = sr_levels["sr_dist_support_pct"].clip(0, 20).fillna(20)
    features["sr_dist_resistance_pct"] = sr_levels["sr_dist_resistance_pct"].clip(0, 20).fillna(20)
    features["sr_level_count_norm"] = (sr_levels["sr_level_count"] / 5).clip(0, 3)

    features["close"] = ind["close"]
    return features


def latest_feature_vector(ohlcv: pd.DataFrame) -> pd.Series | None:
    """Canlı tahmin için en son (tam dolu) özellik satırını döner."""
    features = build_features(ohlcv).dropna()
    if features.empty:
        return None
    return features.iloc[-1]
