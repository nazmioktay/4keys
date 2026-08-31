import numpy as np
import pandas as pd

from app.exchanges.base import Exchange
from app.ml.dataset import build_training_dataset, build_training_dataset_with_time
from app.ml.model import SignalModel
from app.ml.train import train_signal_model_validated
from app.ml.validation import split_out_of_sample, walk_forward_splits


class TrendExchange(Exchange):
    """Testler için gerçek ağ çağrısı yapmayan, belirgin bir yön trendi
    olan sentetik borsa (XGBoost'un öğrenebileceği bir örüntü olması için)."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        return ["UPUSDT", "DOWNUSDT"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        n = max(limit, 400)
        if symbol == "UPUSDT":
            base = np.linspace(100, 400, n)
        else:
            base = np.linspace(400, 100, n)
        noise = self._rng.normal(0, 1.0, n)
        close = base + noise
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": self._rng.uniform(800, 1200, n),
            }
        )


def test_signal_model_defaults_to_xgboost():
    model = SignalModel()
    assert model.algorithm == "xgboost"


def test_xgboost_model_fits_and_predicts():
    exchange = TrendExchange(seed=1)
    X, y = build_training_dataset(exchange, ["UPUSDT", "DOWNUSDT"], "4h", 400, horizon=5, threshold_pct=0.5)
    model = SignalModel(algorithm="xgboost")
    model.fit(X, y)

    predictions, confidences = model.predict_batch(X)
    assert len(predictions) == len(X)
    assert ((confidences >= 0) & (confidences <= 1)).all()


def test_mlp_algorithm_still_selectable_for_comparison():
    exchange = TrendExchange(seed=2)
    X, y = build_training_dataset(exchange, ["UPUSDT", "DOWNUSDT"], "4h", 400, horizon=5, threshold_pct=0.5)
    model = SignalModel(algorithm="mlp")
    model.fit(X, y)
    assert model.algorithm == "mlp"
    predictions, _ = model.predict_batch(X)
    assert len(predictions) == len(X)


def test_shap_values_only_supported_for_xgboost():
    exchange = TrendExchange(seed=3)
    X, y = build_training_dataset(exchange, ["UPUSDT", "DOWNUSDT"], "4h", 400, horizon=5, threshold_pct=0.5)

    xgb_model = SignalModel(algorithm="xgboost")
    xgb_model.fit(X, y)
    importance = xgb_model.shap_values(X)
    assert set(importance["feature"]) == set(X.columns) | set()
    assert (importance["mean_abs_shap"] >= 0).all()
    # en yüksek katkılı özellik en üstte olmalı (azalan sırada)
    assert (importance["mean_abs_shap"].diff().dropna() <= 1e-9).all()

    mlp_model = SignalModel(algorithm="mlp")
    mlp_model.fit(X, y)
    try:
        mlp_model.shap_values(X)
        assert False, "MLP için SHAP hata vermeliydi"
    except ValueError:
        pass


def test_split_out_of_sample_never_leaks_future_rows_into_train():
    exchange = TrendExchange(seed=4)
    X, y, time_frac = build_training_dataset_with_time(exchange, ["UPUSDT"], "4h", 400, horizon=5, threshold_pct=0.5)
    X_train, y_train, X_holdout, y_holdout = split_out_of_sample(X, y, time_frac, holdout_frac=0.2)

    assert len(X_train) + len(X_holdout) == len(X)
    assert time_frac[X_train.index].max() <= time_frac[X_holdout.index].min()
    assert len(X_holdout) > 0


def test_walk_forward_splits_are_purged_and_chronological():
    time_frac = pd.Series(np.linspace(0, 1, 500))
    splits = walk_forward_splits(time_frac, n_splits=5, embargo_frac=0.02)
    assert len(splits) > 0

    for train_idx, test_idx in splits:
        max_train_time = time_frac.iloc[train_idx].max()
        min_test_time = time_frac.iloc[test_idx].min()
        # embargo boşluğu: test penceresinin başlangıcı, eğitim setinin
        # en son gördüğü zamandan en az embargo_frac kadar sonra olmalı
        assert min_test_time - max_train_time >= 0.02 - 1e-9


def test_train_signal_model_validated_produces_walk_forward_and_oos_reports():
    exchange = TrendExchange(seed=5)
    result = train_signal_model_validated(
        exchange,
        ["UPUSDT", "DOWNUSDT"],
        horizon=5,
        threshold_pct=0.5,
        timeframe="4h",
        lookback=400,
        holdout_frac=0.2,
        walk_forward_splits=4,
    )

    assert result.rows_used > 0
    assert len(result.walk_forward.folds) > 0
    assert 0.0 <= result.walk_forward.mean_accuracy <= 1.0
    assert result.out_of_sample.holdout_rows > 0
    assert 0.0 <= result.out_of_sample.accuracy <= 1.0
