import numpy as np
import pandas as pd
import pytest

from app.engine.decision import DecisionEngine
from app.engine.positions import PaperPositionStore
from app.exchanges.base import Exchange
from app.ml.model import Prediction, SignalModel
from app.ml.lstm_model import LSTMSignalModel
from app.ml.online_model import OnlineSignalModel


def test_combine_predictions_agreement_boosts_confidence():
    xgb = Prediction(direction="long", confidence=0.6)
    lstm = Prediction(direction="long", confidence=0.7)
    combined = DecisionEngine._combine_predictions(xgb, lstm)
    assert combined.direction == "long"
    # (0.6+0.7)/2*1.1 = 0.715
    assert combined.confidence == pytest.approx(0.715)


def test_combine_predictions_agreement_capped_at_one():
    xgb = Prediction(direction="short", confidence=0.95)
    lstm = Prediction(direction="short", confidence=0.98)
    combined = DecisionEngine._combine_predictions(xgb, lstm)
    assert combined.confidence <= 1.0


def test_combine_predictions_opposite_directions_goes_neutral():
    xgb = Prediction(direction="long", confidence=0.8)
    lstm = Prediction(direction="short", confidence=0.9)
    combined = DecisionEngine._combine_predictions(xgb, lstm)
    assert combined.direction == "neutral"
    assert combined.confidence == pytest.approx(0.8)  # min(0.8, 0.9)


def test_combine_predictions_one_neutral_discounts_directional():
    xgb = Prediction(direction="neutral", confidence=0.5)
    lstm = Prediction(direction="long", confidence=0.8)
    combined = DecisionEngine._combine_predictions(xgb, lstm)
    assert combined.direction == "long"
    assert combined.confidence == pytest.approx(0.8 * 0.7)


def test_combine_predictions_no_lstm_returns_xgb_unchanged():
    xgb = Prediction(direction="long", confidence=0.6)
    combined = DecisionEngine._combine_predictions(xgb, None)
    assert combined is xgb


class _FakeExchange(Exchange):
    """Ensemble entegrasyon testi için sabit bir OHLCV serisi döner."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng(42)

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        return ["BTCUSDT"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        n = max(limit, 220)
        close = 100 + np.cumsum(self._rng.normal(0, 1, n))
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": self._rng.uniform(800, 1200, n),
            }
        )


def test_decision_engine_ensemble_uses_lstm_when_provided():
    from app.ml.dataset import build_training_dataset
    from app.ml.sequence_dataset import build_sequence_dataset

    exchange = _FakeExchange()

    X, y = build_training_dataset(exchange, ["BTCUSDT"], "1h", 220, horizon=5, threshold_pct=1.0)
    xgb_model = SignalModel()
    xgb_model.fit(X, y)

    X_seq, y_seq, _t = build_sequence_dataset(exchange, ["BTCUSDT"], "1h", 220, seq_len=10, horizon=5, threshold_pct=1.0)
    lstm_model = LSTMSignalModel(seq_len=10, hidden_size=8, num_layers=1)
    lstm_model.fit(X_seq, y_seq, epochs=2)
    lstm_model.feature_columns = list(X.columns)  # gerçek eğitim akışında train.py bunu ayarlar

    engine = DecisionEngine(
        exchange=exchange,
        model=xgb_model,
        positions=PaperPositionStore(),
        timeframe="1h",
        lookback=220,
        open_confidence=0.0,
        close_confidence=0.0,
        lstm_model=lstm_model,
    )

    result = engine._predict("BTCUSDT")
    assert result is not None
    prediction, price, _feature_row = result
    assert prediction.direction in {"long", "short", "neutral"}
    assert 0.0 <= prediction.confidence <= 1.0
    assert price > 0


def test_decision_engine_ensemble_uses_online_model_when_provided():
    from app.ml.dataset import build_training_dataset

    exchange = _FakeExchange()

    X, y = build_training_dataset(exchange, ["BTCUSDT"], "1h", 220, horizon=5, threshold_pct=1.0)
    xgb_model = SignalModel()
    xgb_model.fit(X, y)

    online_model = OnlineSignalModel(n_models=3, seed=0)
    for (_idx, row), label in zip(X.iterrows(), y):
        online_model.learn_one(row.to_dict(), int(label))

    engine = DecisionEngine(
        exchange=exchange,
        model=xgb_model,
        positions=PaperPositionStore(),
        timeframe="1h",
        lookback=220,
        open_confidence=0.0,
        close_confidence=0.0,
        online_model=online_model,
    )

    result = engine._predict("BTCUSDT")
    assert result is not None
    prediction, price, _feature_row = result
    assert prediction.direction in {"long", "short", "neutral"}
    assert 0.0 <= prediction.confidence <= 1.0
    assert price > 0
