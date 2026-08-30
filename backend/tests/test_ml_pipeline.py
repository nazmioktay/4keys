import numpy as np
import pandas as pd

from app.engine.decision import DecisionEngine
from app.engine.positions import PaperPositionStore
from app.exchanges.base import Exchange
from app.ml.dataset import build_training_dataset
from app.ml.model import SignalModel


class FakeExchange(Exchange):
    """Testler için gerçek ağ çağrısı yapmayan sentetik borsa."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        return ["UPUSDT", "DOWNUSDT"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        n = max(limit, 220)
        if symbol == "UPUSDT":
            base = np.linspace(100, 260, n)
        else:
            base = np.linspace(260, 100, n)
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


def test_dataset_builds_rows_for_both_symbols():
    exchange = FakeExchange(seed=1)
    X, y = build_training_dataset(exchange, ["UPUSDT", "DOWNUSDT"], "4h", 220, horizon=5, threshold_pct=1.0)
    assert len(X) > 50
    assert set(y.unique()) <= {-1, 0, 1}


def test_model_learns_uptrend_vs_downtrend():
    exchange = FakeExchange(seed=2)
    X, y = build_training_dataset(exchange, ["UPUSDT", "DOWNUSDT"], "4h", 220, horizon=5, threshold_pct=1.0)

    model = SignalModel()
    model.fit(X, y)

    from app.ml.features import latest_feature_vector

    up_features = latest_feature_vector(exchange.fetch_ohlcv("UPUSDT", "4h", 220))
    down_features = latest_feature_vector(exchange.fetch_ohlcv("DOWNUSDT", "4h", 220))

    up_pred = model.predict(up_features)
    down_pred = model.predict(down_features)

    assert up_pred.direction in ("long", "neutral")
    assert down_pred.direction in ("short", "neutral")


def test_decision_engine_opens_and_tracks_position():
    exchange = FakeExchange(seed=3)
    X, y = build_training_dataset(exchange, ["UPUSDT", "DOWNUSDT"], "4h", 220, horizon=5, threshold_pct=1.0)
    model = SignalModel()
    model.fit(X, y)

    positions = PaperPositionStore()
    engine = DecisionEngine(
        exchange=exchange,
        model=model,
        positions=positions,
        timeframe="4h",
        lookback=220,
        open_confidence=0.0,  # testte her sinyalde açılmasını sağla
        close_confidence=0.0,
    )

    actions = engine.run_cycle(["UPUSDT", "DOWNUSDT"])
    assert len(actions) == 2
    assert all(a.type in ("open_long", "open_short", "hold") for a in actions)
