import numpy as np
import pandas as pd
import pytest

from app.exchanges.base import Exchange
from app.ml.regime import RegimeModel, build_regime_labeled_dataset, compute_regime_features, fit_regime_model
from app.ml.train import train_signal_models_by_regime


class _TwoRegimeExchange(Exchange):
    """Yarısı düşük volatiliteli/yatay, yarısı yüksek volatiliteli/trendli
    iki belirgin rejimden oluşan sentetik seri — GMM'nin bunları ayırt
    edip edemediğini test eder."""

    def list_symbols(self, quote_currency, market_type):
        return ["BTC/USDT:USDT"]

    def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
        n = max(limit, 1000)
        rng = np.random.default_rng(0)
        half = n // 2
        calm = 100 + np.cumsum(rng.normal(0, 0.05, half))
        volatile_trend = calm[-1] + np.cumsum(rng.normal(0.3, 2.0, n - half))
        close = np.concatenate([calm, volatile_trend])
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


def test_compute_regime_features_shape_and_warmup_nan():
    ohlcv = pd.DataFrame({"close": np.linspace(100, 200, 50)})
    feats = compute_regime_features(ohlcv, vol_window=20, trend_window=20)
    assert list(feats.columns) == ["regime_volatility", "regime_trend"]
    assert feats.iloc[:20].isna().all().all()  # warm-up (close.shift(1) + rolling(20))
    assert not feats.iloc[20:].isna().any().any()


def test_regime_model_fit_predict_labels_are_sorted_by_volatility():
    rng = np.random.default_rng(1)
    low_vol = pd.DataFrame({"regime_volatility": rng.normal(0.001, 0.0002, 200), "regime_trend": rng.normal(0, 0.0001, 200)})
    high_vol = pd.DataFrame({"regime_volatility": rng.normal(0.05, 0.005, 200), "regime_trend": rng.normal(0, 0.001, 200)})
    combined = pd.concat([low_vol, high_vol], ignore_index=True)

    model = RegimeModel(n_regimes=2)
    model.fit(combined)
    labels = model.predict(combined)

    # 0 = düşük volatilite rejimi olmalı -> ilk 200 satırın çoğu 0 etiketli olmalı
    assert (labels.iloc[:200] == 0).mean() > 0.8
    assert (labels.iloc[200:] == 1).mean() > 0.8


def test_regime_model_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    data = pd.DataFrame({"regime_volatility": rng.normal(0.01, 0.005, 300), "regime_trend": rng.normal(0, 0.001, 300)})
    model = RegimeModel(n_regimes=3)
    model.fit(data)

    path = tmp_path / "regime_test.joblib"
    model.save(path)
    loaded = RegimeModel.load_from(path)

    original_labels = model.predict(data)
    loaded_labels = loaded.predict(data)
    assert (original_labels == loaded_labels).all()


def test_regime_model_predict_requires_fit():
    model = RegimeModel()
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame({"regime_volatility": [0.01], "regime_trend": [0.0]}))


def test_fit_regime_model_returns_summaries_covering_all_regimes():
    exchange = _TwoRegimeExchange()
    model, summaries = fit_regime_model(exchange, ["BTC/USDT:USDT"], "1h", 1000, n_regimes=2)
    assert len(summaries) == 2
    assert sum(s.samples for s in summaries) > 0
    # rejimler volatiliteye göre sıralı olmalı (0 = daha düşük ortalama volatilite)
    assert summaries[0].mean_volatility <= summaries[1].mean_volatility


def test_build_regime_labeled_dataset_excludes_warmup_rows():
    exchange = _TwoRegimeExchange()
    model, _summaries = fit_regime_model(exchange, ["BTC/USDT:USDT"], "1h", 1000, n_regimes=2)
    X, y, regime, time_frac = build_regime_labeled_dataset(exchange, ["BTC/USDT:USDT"], "1h", 1000, model)
    assert len(X) == len(y) == len(regime) == len(time_frac)
    assert len(X) > 0
    assert (regime >= 0).all()


def test_train_signal_models_by_regime_produces_result_per_regime():
    exchange = _TwoRegimeExchange()
    _regime_model, results = train_signal_models_by_regime(
        exchange, ["BTC/USDT:USDT"], n_regimes=2, timeframe="1h", lookback=1000, holdout_frac=0.2, walk_forward_splits=2, persist=False
    )
    assert len(results) == 2
    for r in results:
        assert r.regime in (0, 1)
        assert r.samples >= 0
