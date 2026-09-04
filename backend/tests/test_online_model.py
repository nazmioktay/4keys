import numpy as np
import pandas as pd
import pytest

from app.exchanges.base import Exchange
from app.ml.online_model import OnlineSignalModel, run_prequential_evaluation
from app.ml.train import train_online_signal_model


class _TrendExchange(Exchange):
    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def list_symbols(self, quote_currency, market_type):
        return ["UPUSDT"]

    def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
        n = max(limit, 400)
        base = np.linspace(100, 400, n)
        close = base + self._rng.normal(0, 1.0, n)
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


def test_online_signal_model_learn_and_predict():
    model = OnlineSignalModel(n_models=3, seed=0)
    rng = np.random.default_rng(1)
    for _ in range(50):
        feats = {"a": float(rng.random()), "b": float(rng.random())}
        label = 1 if feats["a"] > 0.5 else -1
        model.learn_one(feats, label)

    prediction = model.predict_one({"a": 0.9, "b": 0.1})
    assert prediction.direction in {"long", "short", "neutral"}
    assert 0.0 <= prediction.confidence <= 1.0


def test_online_signal_model_requires_learning_before_predict_raises_on_save(tmp_path):
    model = OnlineSignalModel()
    with pytest.raises(RuntimeError):
        model.save(tmp_path / "x.joblib")


def test_online_signal_model_save_load_roundtrip(tmp_path):
    model = OnlineSignalModel(n_models=3, seed=0)
    rng = np.random.default_rng(2)
    for _ in range(50):
        feats = {"a": float(rng.random())}
        model.learn_one(feats, 1 if feats["a"] > 0.5 else -1)

    path = tmp_path / "online_test.joblib"
    model.save(path)
    loaded = OnlineSignalModel.load_from(path)

    original = model.predict_one({"a": 0.9})
    reloaded = loaded.predict_one({"a": 0.9})
    assert original.direction == reloaded.direction
    assert original.confidence == pytest.approx(reloaded.confidence)


def test_run_prequential_evaluation_produces_windows_and_reasonable_metrics():
    rng = np.random.default_rng(3)
    n = 1200
    a = rng.random(n)
    y = pd.Series(np.where(a > 0.5, 1, -1))
    X = pd.DataFrame({"a": a, "b": rng.random(n)})

    model, report = run_prequential_evaluation(X, y, n_models=5, window_size=300)

    assert report.rows_used == n
    assert len(report.windows) == 4  # 1200 / 300
    assert 0.0 <= report.overall_accuracy <= 1.0
    assert 0.0 <= report.overall_balanced_accuracy <= 1.0
    # Basit, öğrenilebilir bir örüntüde (a>0.5 -> long) son pencerede
    # modelin ilk pencereden daha iyi (ya da en azından kötü olmayan)
    # performans göstermesi beklenir — adaptasyonun kaba bir kontrolü.
    assert report.windows[-1].accuracy >= report.windows[0].accuracy - 0.1


def test_train_online_signal_model_end_to_end():
    exchange = _TrendExchange(seed=5)
    model, report = train_online_signal_model(exchange, ["UPUSDT"], timeframe="4h", lookback=400, n_models=3, window_size=100, persist=False)
    assert report.rows_used > 0
    assert 0.0 <= report.overall_accuracy <= 1.0


def test_train_online_signal_model_writes_disabled_status_when_rejected(tmp_path, monkeypatch):
    from app.core.config import settings
    from app.ml import train as train_module
    from app.ml.model_status import is_model_enabled

    status_target = tmp_path / "online_model.joblib"
    monkeypatch.setattr(train_module, "DEFAULT_ONLINE_MODEL_PATH", status_target)
    monkeypatch.setattr(settings, "ml_min_balanced_accuracy", 1.01)  # her zaman reddedilsin

    exchange = _TrendExchange(seed=6)
    _model, report = train_online_signal_model(
        exchange, ["UPUSDT"], timeframe="4h", lookback=400, n_models=3, window_size=100
    )

    assert report.accepted is False
    assert is_model_enabled(status_target) is False


def test_train_online_signal_model_writes_enabled_status_when_accepted(tmp_path, monkeypatch):
    from app.core.config import settings
    from app.ml import train as train_module
    from app.ml.model_status import is_model_enabled

    status_target = tmp_path / "online_model.joblib"
    status_target.write_text("placeholder")  # is_model_enabled dosya varlığını da kontrol eder
    monkeypatch.setattr(train_module, "DEFAULT_ONLINE_MODEL_PATH", status_target)
    monkeypatch.setattr(settings, "ml_min_balanced_accuracy", 0.0)  # her zaman kabul edilsin
    monkeypatch.setattr(OnlineSignalModel, "save", lambda self, *a, **kw: None)

    exchange = _TrendExchange(seed=7)
    _model, report = train_online_signal_model(
        exchange, ["UPUSDT"], timeframe="4h", lookback=400, n_models=3, window_size=100
    )

    assert report.accepted is True
    assert is_model_enabled(status_target) is True


def test_train_online_signal_model_raises_on_insufficient_data():
    class _TinyExchange(Exchange):
        def list_symbols(self, quote_currency, market_type):
            return ["UPUSDT"]

        def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
            close = np.linspace(100, 110, 10)
            return pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=10, freq="4h"),
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": np.full(10, 1000.0),
                }
            )

    with pytest.raises(ValueError):
        train_online_signal_model(_TinyExchange(), ["UPUSDT"], timeframe="4h", lookback=10, persist=False)
