import pytest

from app.core.config import settings
from app.trading.executor import LiveTradingDisabled, place_live_order
from app.trading.schemas import OrderRequest


@pytest.fixture(autouse=True)
def _reset_settings():
    original = (
        settings.enable_live_trading,
        settings.binance_api_key,
        settings.binance_api_secret,
    )
    yield
    settings.enable_live_trading, settings.binance_api_key, settings.binance_api_secret = original


def test_order_blocked_when_live_trading_disabled():
    settings.enable_live_trading = False
    request = OrderRequest(symbol="BTC/USDT:USDT", side="buy", amount=0.001, confirm=True)
    with pytest.raises(LiveTradingDisabled, match="Canlı işlem devre dışı"):
        place_live_order(request)


def test_order_blocked_without_confirm_even_if_enabled():
    settings.enable_live_trading = True
    request = OrderRequest(symbol="BTC/USDT:USDT", side="buy", amount=0.001, confirm=False)
    with pytest.raises(LiveTradingDisabled, match="confirm=true"):
        place_live_order(request)


def test_order_blocked_without_credentials():
    from pydantic import SecretStr

    settings.enable_live_trading = True
    settings.binance_api_key = SecretStr("")
    settings.binance_api_secret = SecretStr("")
    request = OrderRequest(symbol="BTC/USDT:USDT", side="buy", amount=0.001, confirm=True)
    with pytest.raises(LiveTradingDisabled, match="API anahtarı"):
        place_live_order(request)


def test_limit_order_requires_price():
    settings.enable_live_trading = True
    request = OrderRequest(symbol="BTC/USDT:USDT", side="buy", order_type="limit", amount=0.001, confirm=True)
    with pytest.raises(ValueError, match="price zorunludur"):
        place_live_order(request)
