"""Özellik sadeleştirme (plan 3.5) erişilebilir mi? (denetim bulgusu 5)

`drop_features` yalnızca `prepare_walk_forward`'da vardı; walk-forward sistem
backtest'i, rotası ve önerilen listeyi üreten bir uç nokta yoktu."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.api.routes import backtest as backtest_routes
from app.backtest import walk_forward
from app.backtest.schemas import SystemBacktestRequest
from app.auth.tokens import create_token
from app.backtest.walk_forward import WalkForwardSystemReport
from app.main import app
from app.ml.features import ALL_FEATURE_COLUMNS


def _client() -> TestClient:
    # Tüm rotalar auth middleware'inin arkasında (bkz. app.auth.middleware).
    return TestClient(app, headers={"Authorization": f"Bearer {create_token('test')}"})


def _report() -> WalkForwardSystemReport:
    return WalkForwardSystemReport(
        symbol="BTC/USDT:USDT", timeframe="1h", folds=[], trades_closed=0, win_rate_pct=0.0, total_pnl_pct=0.0,
        max_drawdown_pct=0.0, score=0.0, profitable_folds=0, final_test_reserved_from=None, includes_final_test=False,
    )


def test_run_walk_forward_system_backtest_forwards_drop_features(monkeypatch):
    seen = {}

    def fake_prepare(exchange, request, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(walk_forward, "prepare_walk_forward", fake_prepare)
    monkeypatch.setattr(walk_forward, "evaluate_walk_forward", lambda *a, **k: SimpleNamespace(warnings=[]))

    report = walk_forward.run_walk_forward_system_backtest(
        None, SystemBacktestRequest(), drop_features=[ALL_FEATURE_COLUMNS[1]]
    )

    assert seen["drop_features"] == [ALL_FEATURE_COLUMNS[1]]
    assert any("sadeleştirme" in w for w in report.warnings)  # sonuç sadeleştirilmiş modele ait diye işaretlenir


def test_walk_forward_route_accepts_and_forwards_drop_features(monkeypatch):
    seen = {}

    def fake_run(exchange, request, **kwargs):
        seen.update(kwargs)
        return _report()

    monkeypatch.setattr(backtest_routes, "get_exchange", lambda *_a, **_k: object())
    monkeypatch.setattr(backtest_routes, "run_walk_forward_system_backtest", fake_run)

    response = _client().post(
        "/backtest/system/walk-forward", json={"drop_features": [ALL_FEATURE_COLUMNS[1], ALL_FEATURE_COLUMNS[2]]}
    )

    assert response.status_code == 200
    assert seen["drop_features"] == [ALL_FEATURE_COLUMNS[1], ALL_FEATURE_COLUMNS[2]]


def test_walk_forward_route_default_does_not_drop_anything(monkeypatch):
    seen = {}
    monkeypatch.setattr(backtest_routes, "get_exchange", lambda *_a, **_k: object())
    monkeypatch.setattr(backtest_routes, "run_walk_forward_system_backtest", lambda *a, **k: seen.update(k) or _report())

    assert _client().post("/backtest/system/walk-forward", json={}).status_code == 200
    assert not seen["drop_features"]


def test_walk_forward_route_rejects_unknown_feature_names(monkeypatch):
    called = []
    monkeypatch.setattr(backtest_routes, "get_exchange", lambda *_a, **_k: object())
    monkeypatch.setattr(backtest_routes, "run_walk_forward_system_backtest", lambda *a, **k: called.append(1) or _report())

    response = _client().post("/backtest/system/walk-forward", json={"drop_features": ["yok_boyle_bir_ozellik"]})

    assert response.status_code == 422
    assert "yok_boyle_bir_ozellik" in response.text
    assert called == []


def test_suggest_drop_features_endpoint_returns_redundant_features_read_only(monkeypatch):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, len(ALL_FEATURE_COLUMNS))), columns=ALL_FEATURE_COLUMNS)
    first, second = ALL_FEATURE_COLUMNS[0], ALL_FEATURE_COLUMNS[1]
    X[second] = X[first] * 3 + rng.normal(scale=0.001, size=500)  # `second`, `first`'ün kopyası
    monkeypatch.setattr(backtest_routes, "get_exchange", lambda *_a, **_k: object())
    monkeypatch.setattr(backtest_routes, "build_training_dataset", lambda *a, **k: (X, pd.Series(np.zeros(500))))

    response = _client().get("/backtest/system/suggest-drop-features", params={"threshold": 0.9})

    assert response.status_code == 200
    body = response.json()
    assert body["drop_features"] == [second]
    assert [first, second] in body["clusters"]
    assert body["rows"] == 500 and body["threshold"] == 0.9
