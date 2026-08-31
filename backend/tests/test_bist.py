from unittest.mock import MagicMock, patch

import pytest

from app.bist import service as bist_service
from app.bist.schemas import BistOrderRequest
from app.bist.service import BistTradingDisabled, login, login_verify, place_bist_order
from app.core.config import settings
from app.exchanges.algolab import AlgoLabExchange
from app.security import kill_switch


@pytest.fixture(autouse=True)
def _reset_state():
    bist_service.reset_for_tests()
    kill_switch.reset_for_tests()
    yield
    bist_service.reset_for_tests()
    kill_switch.reset_for_tests()


def _exchange() -> AlgoLabExchange:
    return AlgoLabExchange(api_key="test-key", base_url="https://api.example.test")


@patch("app.exchanges.algolab.requests.post")
def test_login_returns_token(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"content": {"token": "tok-123"}}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    exchange = _exchange()
    token = exchange.login("user", "pass")

    assert token == "tok-123"
    called_url = mock_post.call_args.args[0]
    assert called_url == "https://api.example.test/LoginUser"
    assert mock_post.call_args.kwargs["headers"]["APIKEY"] == "test-key"


@patch("app.exchanges.algolab.requests.post")
def test_login_control_sets_session_hash(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"content": {"hash": "session-hash-abc"}}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    exchange = _exchange()
    assert exchange.is_authenticated() is False

    session_hash = exchange.login_control("tok-123", "000000")

    assert session_hash == "session-hash-abc"
    assert exchange.is_authenticated() is True


def test_operations_require_auth_before_login():
    exchange = _exchange()
    with pytest.raises(PermissionError):
        exchange.fetch_positions()
    with pytest.raises(PermissionError):
        exchange.list_symbols("TRY", "equity")


@patch("app.exchanges.algolab.requests.get")
def test_fetch_ohlcv_parses_bars(mock_get):
    exchange = _exchange()
    exchange._session_hash = "already-authed"

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "content": [
            {"date": "2024-01-01T00:00:00", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000},
            {"date": "2024-01-02T00:00:00", "open": 10.5, "high": 12, "low": 10, "close": 11.5, "volume": 1200},
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    df = exchange.fetch_ohlcv("GARAN", "1d", 200)

    assert len(df) == 2
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["close"].iloc[-1] == 11.5
    headers = mock_get.call_args.kwargs["headers"]
    assert headers["Authorization"] == "already-authed"


@patch("app.bist.service.AlgoLabExchange")
def test_service_login_uses_settings_credentials_when_not_provided(mock_exchange_cls, monkeypatch):
    monkeypatch.setattr(settings, "algolab_api_key", type(settings.algolab_api_key)("key-x"))
    monkeypatch.setattr(settings, "algolab_username", "settings-user")
    monkeypatch.setattr(settings, "algolab_password", type(settings.algolab_password)("settings-pass"))

    mock_instance = MagicMock()
    mock_instance.login.return_value = "tok-from-settings"
    mock_exchange_cls.return_value = mock_instance

    token = login(None, None)

    assert token == "tok-from-settings"
    mock_instance.login.assert_called_once_with("settings-user", "settings-pass")


def test_service_login_fails_without_api_key(monkeypatch):
    monkeypatch.setattr(settings, "algolab_api_key", type(settings.algolab_api_key)(""))
    with pytest.raises(BistTradingDisabled, match="API anahtarı"):
        login("u", "p")


@patch("app.bist.service.AlgoLabExchange")
def test_place_order_blocked_by_kill_switch(mock_exchange_cls, monkeypatch):
    monkeypatch.setattr(settings, "algolab_api_key", type(settings.algolab_api_key)("key-x"))
    monkeypatch.setattr(settings, "enable_bist_trading", True)
    kill_switch.activate("acil", triggered_by="manual")

    request = BistOrderRequest(symbol="GARAN", direction="buy", quantity=10, confirm=True)
    with pytest.raises(BistTradingDisabled, match="Kill switch aktif"):
        place_bist_order(request)


@patch("app.bist.service.AlgoLabExchange")
def test_place_order_blocked_when_trading_disabled(mock_exchange_cls, monkeypatch):
    monkeypatch.setattr(settings, "algolab_api_key", type(settings.algolab_api_key)("key-x"))
    monkeypatch.setattr(settings, "enable_bist_trading", False)

    request = BistOrderRequest(symbol="GARAN", direction="buy", quantity=10, confirm=True)
    with pytest.raises(BistTradingDisabled, match="devre dışı"):
        place_bist_order(request)


@patch("app.bist.service.AlgoLabExchange")
def test_place_order_blocked_without_confirm(mock_exchange_cls, monkeypatch):
    monkeypatch.setattr(settings, "algolab_api_key", type(settings.algolab_api_key)("key-x"))
    monkeypatch.setattr(settings, "enable_bist_trading", True)

    request = BistOrderRequest(symbol="GARAN", direction="buy", quantity=10, confirm=False)
    with pytest.raises(BistTradingDisabled, match="confirm=true"):
        place_bist_order(request)


@patch("app.bist.service.AlgoLabExchange")
def test_place_order_blocked_without_session(mock_exchange_cls, monkeypatch):
    monkeypatch.setattr(settings, "algolab_api_key", type(settings.algolab_api_key)("key-x"))
    monkeypatch.setattr(settings, "enable_bist_trading", True)

    mock_instance = MagicMock()
    mock_instance.is_authenticated.return_value = False
    mock_exchange_cls.return_value = mock_instance

    request = BistOrderRequest(symbol="GARAN", direction="buy", quantity=10, confirm=True)
    with pytest.raises(BistTradingDisabled, match="oturum"):
        place_bist_order(request)


@patch("app.bist.service.AlgoLabExchange")
def test_place_order_succeeds_when_all_gates_pass(mock_exchange_cls, monkeypatch):
    monkeypatch.setattr(settings, "algolab_api_key", type(settings.algolab_api_key)("key-x"))
    monkeypatch.setattr(settings, "enable_bist_trading", True)

    mock_instance = MagicMock()
    mock_instance.is_authenticated.return_value = True
    mock_instance.send_order.return_value = {"status": "ok", "orderId": "123"}
    mock_exchange_cls.return_value = mock_instance

    request = BistOrderRequest(symbol="GARAN", direction="buy", quantity=10, confirm=True)
    result = place_bist_order(request)

    assert result == {"status": "ok", "orderId": "123"}
    mock_instance.send_order.assert_called_once()
