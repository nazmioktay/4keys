from fastapi.testclient import TestClient

from app.main import app
from app.monitoring.metrics import record_ml_prediction, record_scheduler_job, record_trade_closed


def test_metrics_endpoint_exposes_starlette_and_business_metrics():
    client = TestClient(app)
    client.get("/health")
    record_trade_closed("long", 5.0)
    record_ml_prediction("BTC/USDT:USDT", "long", 0.75)
    record_scheduler_job("screener_refresh", True)

    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text

    assert 'starlette_requests_total{app_name="4keys-backend"' in body
    assert "fourkeys_trades_closed_total" in body
    assert 'fourkeys_ml_prediction_confidence{direction="long",symbol="BTC/USDT:USDT"} 0.75' in body
    assert 'fourkeys_scheduler_job_last_run_ok{job_id="screener_refresh"} 1.0' in body


def test_record_trade_closed_labels_win_loss_flat_correctly():
    from prometheus_client import REGISTRY

    record_trade_closed("long", 3.0)
    record_trade_closed("short", -2.0)
    record_trade_closed("long", 0.0)

    win = REGISTRY.get_sample_value("fourkeys_trades_closed_total", {"direction": "long", "result": "win"})
    loss = REGISTRY.get_sample_value("fourkeys_trades_closed_total", {"direction": "short", "result": "loss"})
    flat = REGISTRY.get_sample_value("fourkeys_trades_closed_total", {"direction": "long", "result": "flat"})
    assert win and win >= 1
    assert loss and loss >= 1
    assert flat and flat >= 1
