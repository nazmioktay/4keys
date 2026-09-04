"""`/ml/*` endpoint'lerinin, alttaki eğitim/tahmin fonksiyonlarını mocklayarak,
gerçekten pydantic response_model'e uyan bir JSON döndürdüğünü doğrular.

Bu dosyanın var olma nedeni: `/ml/train-meta` uzun süre `response_model=TrainResponse`
(walk-forward/out-of-sample gibi birçok zorunlu alan içeren XGBoost'a özgü
model) kullanıyordu ama yalnızca `rows_used`/`symbols_used` dolduruyordu —
sonuç, canlıda her çağrıldığında 500 hatasıyla patlıyordu (pydantic
validation error), hiçbir test bunu yakalamamıştı."""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.exchanges.base import Exchange
from app.main import app


class _TrendExchange(Exchange):
    def list_symbols(self, quote_currency, market_type):
        return ["UPUSDT", "DOWNUSDT"]

    def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
        n = max(limit, 400)
        base = np.linspace(100, 400, n) if symbol == "UPUSDT" else np.linspace(400, 100, n)
        close = base + np.random.default_rng(0).normal(0, 1.0, n)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.random.default_rng(0).uniform(800, 1200, n),
            }
        )


def test_train_meta_endpoint_returns_response_matching_its_schema(monkeypatch):
    from app.api.routes import ml as ml_routes
    from app.ml.dataset import build_training_dataset
    from app.ml.model import SignalModel

    # Birincil modeli gerçekten (küçük veriyle) eğitip belleğe alır —
    # dosya sistemine/DEFAULT_MODEL_PATH'e dokunmadan `_model_exists` ve
    # `SignalModel.load_from`'u bu modeli döndürecek şekilde mockluyoruz.
    X, y = build_training_dataset(_TrendExchange(), ["UPUSDT", "DOWNUSDT"], "1h", 400, horizon=5, threshold_pct=0.5)
    primary = SignalModel(algorithm="xgboost")
    primary.fit(X, y)

    monkeypatch.setattr(ml_routes, "get_exchange", lambda *_a, **_k: _TrendExchange())
    monkeypatch.setattr(ml_routes.settings, "ml_train_timeframe", "1h")
    monkeypatch.setattr(ml_routes.settings, "ml_train_lookback", 400)
    monkeypatch.setattr(ml_routes, "_model_exists", lambda: True)
    monkeypatch.setattr(SignalModel, "load_from", classmethod(lambda cls, path=None: primary))

    client = TestClient(app)
    response = client.post("/ml/train-meta", json={"symbols": ["UPUSDT", "DOWNUSDT"], "horizon": 5, "threshold_pct": 0.5})

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"rows_used", "symbols_used"}
    assert body["symbols_used"] == 2
    assert body["rows_used"] > 0
