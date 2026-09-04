import numpy as np
import pandas as pd
import pytest

from app.exchanges.base import Exchange
from app.ml.patchtst_model import PatchTSTSignalModel
from app.ml.train import train_patchtst_signal_model


class _TrendExchange(Exchange):
    """Belirgin bir yön trendi olan, gerçek ağ çağrısı yapmayan sentetik borsa."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        return ["UPUSDT", "DOWNUSDT"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since=None) -> pd.DataFrame:
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


def test_patchtst_model_fit_predict_roundtrip():
    rng = np.random.default_rng(1)
    n_samples, seq_len, n_features = 200, 20, 24
    X = rng.normal(0, 1, (n_samples, seq_len, n_features)).astype("float32")
    y = rng.choice([-1, 0, 1], size=n_samples)

    model = PatchTSTSignalModel(seq_len=seq_len, patch_len=5, stride=5, d_model=16, nhead=2, num_layers=1)
    report = model.fit(X, y, epochs=2, batch_size=32)
    assert report.epochs_run == 2

    predictions, confidences = model.predict_batch(X[:5])
    assert len(predictions) == 5
    assert set(predictions.tolist()) <= {-1, 0, 1}
    assert ((confidences >= 0) & (confidences <= 1)).all()

    single_prediction = model.predict(X[0])
    assert single_prediction.direction in {"long", "short", "neutral"}
    assert 0 <= single_prediction.confidence <= 1


def test_patchtst_model_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    n_samples, seq_len, n_features = 100, 20, 24
    X = rng.normal(0, 1, (n_samples, seq_len, n_features)).astype("float32")
    y = rng.choice([-1, 0, 1], size=n_samples)

    model = PatchTSTSignalModel(seq_len=seq_len, patch_len=5, stride=5, d_model=16, nhead=2, num_layers=1)
    model.fit(X, y, epochs=2)

    path = tmp_path / "patchtst_test.pt"
    model.save(path)

    loaded = PatchTSTSignalModel.load_from(path)
    pred_original, conf_original = model.predict_batch(X[:3])
    pred_loaded, conf_loaded = loaded.predict_batch(X[:3])

    assert (pred_original == pred_loaded).all()
    np.testing.assert_allclose(conf_original, conf_loaded, rtol=1e-5)


def test_patchtst_model_requires_fit_before_predict():
    model = PatchTSTSignalModel()
    with pytest.raises(RuntimeError):
        model.predict_batch(np.zeros((1, 20, 24), dtype="float32"))


def test_patchtst_model_fit_with_validation_enables_early_stopping():
    rng = np.random.default_rng(3)
    n_samples, seq_len, n_features = 200, 20, 24
    X = rng.normal(0, 1, (n_samples, seq_len, n_features)).astype("float32")
    y = rng.choice([-1, 0, 1], size=n_samples)
    X_val = rng.normal(0, 1, (40, seq_len, n_features)).astype("float32")
    y_val = rng.choice([-1, 0, 1], size=40)

    model = PatchTSTSignalModel(seq_len=seq_len, patch_len=5, stride=5, d_model=16, nhead=2, num_layers=1)
    report = model.fit(X, y, epochs=50, X_val=X_val, y_val=y_val, patience=0)

    assert report.best_val_loss is not None
    assert report.epochs_run <= 50
    if report.epochs_run < 50:
        assert report.stopped_early is True


def test_train_patchtst_signal_model_produces_oos_report():
    exchange = _TrendExchange(seed=4)
    result = train_patchtst_signal_model(
        exchange,
        ["UPUSDT", "DOWNUSDT"],
        seq_len=10,
        patch_len=5,
        stride=5,
        d_model=16,
        nhead=2,
        num_layers=1,
        horizon=5,
        threshold_pct=0.5,
        timeframe="4h",
        lookback=400,
        holdout_frac=0.2,
        epochs=3,
    )

    assert result.rows_used > 0
    assert result.training.epochs_run == 3
    assert result.out_of_sample.holdout_rows > 0
    assert 0.0 <= result.out_of_sample.accuracy <= 1.0
    assert result.model.feature_columns is not None


def test_train_patchtst_signal_model_raises_on_insufficient_data():
    exchange = _TrendExchange(seed=5)
    with pytest.raises(ValueError):
        train_patchtst_signal_model(exchange, ["UPUSDT"], seq_len=330, lookback=400, epochs=1)
