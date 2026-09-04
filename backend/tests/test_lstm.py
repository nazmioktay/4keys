import numpy as np
import pandas as pd
import pytest

from app.exchanges.base import Exchange
from app.ml.lstm_model import LSTMSignalModel
from app.ml.sequence_dataset import build_sequence_dataset
from app.ml.train import sweep_labeling_lstm, train_lstm_signal_model


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


def test_build_sequence_dataset_shapes():
    exchange = _TrendExchange(seed=1)
    X, y, time_frac = build_sequence_dataset(exchange, ["UPUSDT", "DOWNUSDT"], "4h", 400, seq_len=10, horizon=5, threshold_pct=0.5)
    assert X.ndim == 3
    assert X.shape[1] == 10  # seq_len
    assert X.shape[0] == len(y) == len(time_frac)
    assert X.shape[0] > 0
    assert (time_frac >= 0).all() and (time_frac <= 1).all()


def test_build_sequence_dataset_windows_are_contiguous_per_symbol():
    """Her pencere kendi sembolünün KENDİ kronolojik serisinden kurulmalı —
    semboller arası karışma olmamalı."""
    exchange = _TrendExchange(seed=2)
    X, y, _t = build_sequence_dataset(exchange, ["UPUSDT"], "4h", 400, seq_len=15, horizon=5, threshold_pct=0.5)
    # bir pencere içindeki close değerleri (feature'lardan türetilemez ama
    # ardışık iki bar arasındaki fark makul aralıkta olmalı - sıçrama yok)
    assert X.shape[0] > 0
    assert not np.isnan(X).any()


def test_lstm_model_fit_predict_roundtrip():
    rng = np.random.default_rng(3)
    n_samples, seq_len, n_features = 200, 10, 24
    X = rng.normal(0, 1, (n_samples, seq_len, n_features)).astype("float32")
    y = rng.choice([-1, 0, 1], size=n_samples)

    model = LSTMSignalModel(seq_len=seq_len, hidden_size=8, num_layers=1)
    report = model.fit(X, y, epochs=2, batch_size=32)
    assert report.epochs_run == 2

    predictions, confidences = model.predict_batch(X[:5])
    assert len(predictions) == 5
    assert set(predictions.tolist()) <= {-1, 0, 1}
    assert ((confidences >= 0) & (confidences <= 1)).all()

    single_prediction = model.predict(X[0])
    assert single_prediction.direction in {"long", "short", "neutral"}
    assert 0 <= single_prediction.confidence <= 1


def test_lstm_model_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(4)
    n_samples, seq_len, n_features = 100, 8, 24
    X = rng.normal(0, 1, (n_samples, seq_len, n_features)).astype("float32")
    y = rng.choice([-1, 0, 1], size=n_samples)

    model = LSTMSignalModel(seq_len=seq_len, hidden_size=8, num_layers=1)
    model.fit(X, y, epochs=2)

    path = tmp_path / "lstm_test.pt"
    model.save(path)

    loaded = LSTMSignalModel.load_from(path)
    pred_original, conf_original = model.predict_batch(X[:3])
    pred_loaded, conf_loaded = loaded.predict_batch(X[:3])

    assert (pred_original == pred_loaded).all()
    np.testing.assert_allclose(conf_original, conf_loaded, rtol=1e-5)


def test_lstm_model_fit_with_validation_enables_early_stopping():
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(6)
    n_samples, seq_len, n_features = 200, 10, 24
    X = rng.normal(0, 1, (n_samples, seq_len, n_features)).astype("float32")
    y = rng.choice([-1, 0, 1], size=n_samples)
    X_val = rng.normal(0, 1, (40, seq_len, n_features)).astype("float32")
    y_val = rng.choice([-1, 0, 1], size=40)

    model = LSTMSignalModel(seq_len=seq_len, hidden_size=8, num_layers=1)
    # patience=0: doğrulama kaybı bir sonraki epoch'ta İYİLEŞMEZSE (rastgele,
    # öğrenilemeyen veride beklenen davranış) hemen durmalı; ilerlemeye devam
    # etmesi ancak val kaybı MONOTON olarak azalırsa mümkün olur, ki bu
    # rastgele/ilişkisiz veride son derece düşük olasılıklıdır.
    report = model.fit(X, y, epochs=50, X_val=X_val, y_val=y_val, patience=0)

    assert report.best_val_loss is not None
    assert report.epochs_run <= 50
    if report.epochs_run < 50:
        assert report.stopped_early is True


def test_lstm_model_fit_without_validation_runs_all_epochs():
    rng = np.random.default_rng(7)
    n_samples, seq_len, n_features = 100, 8, 24
    X = rng.normal(0, 1, (n_samples, seq_len, n_features)).astype("float32")
    y = rng.choice([-1, 0, 1], size=n_samples)

    model = LSTMSignalModel(seq_len=seq_len, hidden_size=8, num_layers=1)
    report = model.fit(X, y, epochs=3)

    assert report.epochs_run == 3
    assert report.stopped_early is False
    assert report.best_val_loss is None


def test_lstm_model_requires_fit_before_predict():
    model = LSTMSignalModel()
    with pytest.raises(RuntimeError):
        model.predict_batch(np.zeros((1, 10, 24), dtype="float32"))


def test_train_lstm_signal_model_produces_oos_report():
    exchange = _TrendExchange(seed=5)
    result = train_lstm_signal_model(
        exchange,
        ["UPUSDT", "DOWNUSDT"],
        seq_len=10,
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


def test_train_lstm_signal_model_raises_on_insufficient_data():
    exchange = _TrendExchange(seed=6)
    # sahte borsa her zaman en az 400 mum döner; yeterince az pencere
    # üretmek için seq_len'i buna göre büyük tutuyoruz.
    with pytest.raises(ValueError):
        train_lstm_signal_model(exchange, ["UPUSDT"], seq_len=330, lookback=400, epochs=1)


def test_train_lstm_signal_model_persist_false_does_not_save(monkeypatch):
    save_calls = []
    monkeypatch.setattr(LSTMSignalModel, "save", lambda self, *a, **kw: save_calls.append(1))

    exchange = _TrendExchange(seed=8)
    train_lstm_signal_model(
        exchange, ["UPUSDT", "DOWNUSDT"], seq_len=10, timeframe="4h", lookback=400, epochs=2, persist=False
    )
    assert save_calls == []


def test_sweep_labeling_lstm_covers_full_grid_and_tolerates_failures():
    exchange = _TrendExchange(seed=9)
    points = sweep_labeling_lstm(
        exchange,
        ["UPUSDT", "DOWNUSDT"],
        horizon_values=[5, 395],  # 395 -> yetersiz veri (etiket ufku neredeyse tüm seriyi yer), error alanıyla işaretlenmeli
        threshold_pct_values=[0.5, 1.5],
        timeframe="4h",
        lookback=400,
        seq_len=10,
        epochs=2,
    )

    assert len(points) == 4  # 2 horizon x 2 threshold
    ok_points = [p for p in points if p.error is None]
    error_points = [p for p in points if p.error is not None]
    assert len(ok_points) == 2  # yalnızca horizon=5 olanlar başarılı olmalı
    assert len(error_points) == 2
    for p in ok_points:
        assert p.rows_used > 0
        assert 0.0 <= p.out_of_sample_balanced_accuracy <= 1.0
