from fastapi.testclient import TestClient

from app.main import app


def test_unhandled_exception_returns_json_with_cors_headers(monkeypatch):
    """Bir route içinde beklenmeyen bir exception fırlarsa, kullanıcı düz bir
    bağlantı hatası ("Failed to fetch") yerine anlamlı bir JSON hata mesajı
    almalı VE yanıt CORS başlıklarını taşımalı (frontend'in bunu okuyabilmesi
    için) — bkz. app/main.py::unhandled_exception_handler."""
    from app.api.routes import dca as dca_routes

    def _boom(*args, **kwargs):
        raise RuntimeError("borsa erişilemedi (test)")

    monkeypatch.setattr(dca_routes, "get_exchange", _boom)

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/dca/optimize",
        json={"symbol": "BTC/USDT:USDT", "balance": 500},
        headers={"Origin": "http://localhost:5173"},
    )

    assert response.status_code == 502
    assert "borsa erişilemedi" in response.json()["detail"]
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
