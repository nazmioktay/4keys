import numpy as np
import pandas as pd
import pytest

from app.ml.labeling import label_future_peak_trough
from tests.test_xgboost import TrendExchange


def test_label_future_peak_trough_matches_manual_calculation():
    high = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    low = pd.Series([9.0, 18.0, 28.0, 38.0, 48.0])
    close = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    ohlcv = pd.DataFrame({"high": high, "low": low, "close": close})

    result = label_future_peak_trough(ohlcv, horizon=2)

    # satır 0: gelecek 2 bar (index 1,2) -> high max=30, low min=18
    assert result["future_peak_pct"].iloc[0] == pytest.approx((30 - 10) / 10 * 100)
    assert result["future_trough_pct"].iloc[0] == pytest.approx((18 - 10) / 10 * 100)
    # son `horizon` satır NaN olmalı (gelecek bilinmiyor)
    assert result["future_peak_pct"].iloc[-1:].isna().all()
    assert result["future_peak_pct"].iloc[-2:].isna().all()


def test_label_future_peak_trough_never_leaks_current_or_past_bar():
    """Causal-safety: satır t'nin etiketi SADECE t+1..t+horizon barlarını
    kullanmalı — t barının kendisi (entry) veya geçmiş barlar dahil OLMAMALI."""
    n = 50
    high = pd.Series(np.arange(n, dtype=float) + 100)
    low = pd.Series(np.arange(n, dtype=float) + 99)
    close = pd.Series(np.arange(n, dtype=float) + 99.5)
    ohlcv = pd.DataFrame({"high": high, "low": low, "close": close})

    result = label_future_peak_trough(ohlcv, horizon=5)

    # Monoton artan seride, future_peak her zaman ileri barların en
    # yükseğini (t+5) yakalamalı — t barının kendi high'ını DEĞİL.
    for t in range(n - 5):
        expected_future_high = high.iloc[t + 1 : t + 6].max()
        expected_peak_pct = (expected_future_high - close.iloc[t]) / close.iloc[t] * 100
        assert abs(result["future_peak_pct"].iloc[t] - expected_peak_pct) < 1e-9


def test_train_dynamic_exit_model_returns_isolated_metrics_without_persisting(monkeypatch):
    from app.ml import dynamic_exit as dynamic_exit_module

    save_calls = []
    monkeypatch.setattr(dynamic_exit_module.DynamicExitModel, "save", lambda self, path=None: save_calls.append(path))

    exchange = TrendExchange(seed=5)
    result = dynamic_exit_module.train_dynamic_exit_model(
        exchange, ["UPUSDT", "DOWNUSDT"], timeframe="4h", lookback=400, horizon=10, persist=False
    )

    assert result.rows_used > 0
    assert isinstance(result.out_of_sample.peak_mae, float)
    assert isinstance(result.out_of_sample.trough_mae, float)
    assert save_calls == []  # persist=False -> save() ASLA çağrılmamalı


def test_dynamic_exit_model_predict_returns_peak_and_trough_arrays():
    from app.ml.dynamic_exit import train_dynamic_exit_model
    from app.ml.features import FEATURE_COLUMNS

    exchange = TrendExchange(seed=6)
    result = train_dynamic_exit_model(exchange, ["UPUSDT", "DOWNUSDT"], timeframe="4h", lookback=400, horizon=10, persist=False)

    sample = pd.DataFrame([[0.0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS)
    peak, trough = result.model.predict(sample)
    assert peak.shape == (1,)
    assert trough.shape == (1,)
