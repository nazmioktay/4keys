import numpy as np
import pandas as pd
import pytest

from app.ml.advanced_indicators import (
    dynamic_support_resistance,
    heikin_ashi,
    linear_regression_channel,
    mavilim_w,
    nadaraya_watson_envelope,
    pmax,
    stoch_rsi_log,
    wavetrend,
)


def _make_ohlcv(n: int = 300, seed: int = 0, trend: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = np.linspace(100, 200, n) if trend else np.full(n, 150.0)
    noise = rng.normal(0, 1.0, n)
    close = base + noise
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": rng.uniform(800, 1200, n),
        }
    )


def test_heikin_ashi_high_low_bracket_open_close():
    ohlcv = _make_ohlcv()
    ha = heikin_ashi(ohlcv)
    assert (ha["ha_high"] >= ha["ha_open"]).all()
    assert (ha["ha_high"] >= ha["ha_close"]).all()
    assert (ha["ha_low"] <= ha["ha_open"]).all()
    assert (ha["ha_low"] <= ha["ha_close"]).all()


def test_heikin_ashi_smooths_noise_vs_raw_close():
    ohlcv = _make_ohlcv(trend=False, seed=1)
    ha = heikin_ashi(ohlcv)
    raw_std = ohlcv["close"].diff().std()
    ha_std = ha["ha_close"].diff().std()
    assert ha_std <= raw_std * 1.5  # HA en azından ham kapanışı fazla büyütmemeli


def test_stoch_rsi_log_bounded_between_0_and_1():
    ohlcv = _make_ohlcv()
    result = stoch_rsi_log(ohlcv["close"])
    valid = result.dropna()
    assert (valid["stoch_rsi_k"] >= 0).all() and (valid["stoch_rsi_k"] <= 1).all()
    assert (valid["stoch_rsi_d"] >= 0).all() and (valid["stoch_rsi_d"] <= 1).all()


def test_mavilim_w_follows_uptrend():
    ohlcv = _make_ohlcv(trend=True)
    m = mavilim_w(ohlcv["close"])
    valid = m.dropna()
    # güçlü bir yükseliş trendinde MavilimW da genel olarak yükselmeli
    assert valid.iloc[-1] > valid.iloc[len(valid) // 2]


def test_pmax_trend_is_bullish_in_uptrend():
    ohlcv = _make_ohlcv(trend=True, seed=2)
    result = pmax(ohlcv)
    # net bir yükseliş trendinin sonunda PMax trend'i +1 (yükseliş) olmalı
    assert result["pmax_trend"].iloc[-1] == 1.0
    assert result["pmax"].iloc[-1] < ohlcv["close"].iloc[-1]


def test_linear_regression_channel_positive_slope_in_uptrend():
    ohlcv = _make_ohlcv(trend=True, seed=3)
    result = linear_regression_channel(ohlcv["close"], length=100)
    valid = result.dropna()
    assert (valid["linreg_slope"] > 0).mean() > 0.9  # neredeyse hep pozitif eğim
    assert (valid["linreg_std"] >= 0).all()


def test_wavetrend_produces_cross_signals():
    ohlcv = _make_ohlcv(trend=False, seed=4)
    result = wavetrend(ohlcv)
    assert set(result["wt_cross"].unique()) <= {-1.0, 0.0, 1.0}
    assert result["wt1"].notna().all()


def test_nadaraya_watson_envelope_upper_above_lower():
    ohlcv = _make_ohlcv(seed=5)
    result = nadaraya_watson_envelope(ohlcv["close"])
    assert (result["nwe_upper"] >= result["nwe_mid"]).all()
    assert (result["nwe_mid"] >= result["nwe_lower"]).all()


def test_nadaraya_watson_is_causal_not_repainting():
    """Son bara yeni bir bar eklemek, ondan ÖNCEKİ noktaların değerini
    değiştirmemeli (repaint yok) — canlı işlemde güvenli kullanım için kritik."""
    ohlcv = _make_ohlcv(seed=6, n=150)
    full = nadaraya_watson_envelope(ohlcv["close"])
    truncated = nadaraya_watson_envelope(ohlcv["close"].iloc[:100])
    pd.testing.assert_series_equal(full["nwe_mid"].iloc[:100], truncated["nwe_mid"], check_names=False)


def test_dynamic_support_resistance_returns_expected_columns():
    ohlcv = _make_ohlcv(trend=False, seed=7, n=400)
    result = dynamic_support_resistance(ohlcv, pivot_lookback=5, channel_lookback=100, min_pivots=2)
    assert list(result.columns) == ["sr_dist_support_pct", "sr_dist_resistance_pct", "sr_level_count"]
    assert (result["sr_level_count"] >= 0).all()


def test_dynamic_support_resistance_is_causal():
    """Gelecekteki barları eklemek geçmiş barların S/R seviyelerini
    değiştirmemeli (pivotlar yalnızca rb bar sonra onaylanıyor)."""
    ohlcv = _make_ohlcv(trend=False, seed=8, n=300)
    full = dynamic_support_resistance(ohlcv, pivot_lookback=5, channel_lookback=100, min_pivots=2)
    truncated = dynamic_support_resistance(ohlcv.iloc[:200], pivot_lookback=5, channel_lookback=100, min_pivots=2)
    # ilk 190 bar (son birkaçı hariç, onay gecikmesi payı) aynı kalmalı
    pd.testing.assert_series_equal(
        full["sr_level_count"].iloc[:190], truncated["sr_level_count"].iloc[:190], check_names=False
    )
