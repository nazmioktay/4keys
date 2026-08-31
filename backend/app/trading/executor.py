import logging

from app.core.config import settings
from app.exchanges.binance import BinanceExchange

from .schemas import OrderRequest

logger = logging.getLogger(__name__)


class LiveTradingDisabled(Exception):
    """Canlı işlem güvenlik kapılarından biri kapalı olduğunda fırlatılır."""


def _credentials_present() -> bool:
    return bool(settings.binance_api_key.get_secret_value()) and bool(settings.binance_api_secret.get_secret_value())


def get_trading_exchange() -> BinanceExchange:
    """Ayarlardaki kimlik bilgileriyle kimlik doğrulamalı Binance istemcisi oluşturur.

    NOT: Bu fonksiyonun kendisi hiçbir güvenlik kontrolü yapmaz (bakiye/pozisyon
    okumak için kullanılabilir olmalı). Gerçek emir gönderme kontrolleri
    `place_live_order` içindedir.
    """
    if not _credentials_present():
        raise LiveTradingDisabled(
            "Binance API anahtarı tanımlı değil. FOURKEYS_BINANCE_API_KEY ve "
            "FOURKEYS_BINANCE_API_SECRET ortam değişkenlerini (.env) ayarlayın."
        )
    return BinanceExchange(
        api_key=settings.binance_api_key.get_secret_value(),
        api_secret=settings.binance_api_secret.get_secret_value(),
        testnet=settings.binance_testnet,
    )


def place_live_order(request: OrderRequest) -> dict:
    """Gerçek borsaya emir gönderir — üç güvenlik kapısından geçer:

    1. `FOURKEYS_ENABLE_LIVE_TRADING=true` (ortam değişkeni, operatör düzeyinde anahtar)
    2. İstekte `confirm: true` (her çağrıda ayrı, yanlışlıkla tetiklenmeyi önler)
    3. Binance API anahtarlarının tanımlı olması

    Üçünden biri eksikse emir gönderilmez. `FOURKEYS_BINANCE_TESTNET=true`
    (varsayılan) iken bile bu kapılar aktiftir — testnet'te denemek isteyen
    kullanıcı yine de `enable_live_trading` ve `confirm`'i açıkça ayarlamalıdır.
    """
    if not settings.enable_live_trading:
        raise LiveTradingDisabled(
            "Canlı işlem devre dışı. Ortam değişkeninde FOURKEYS_ENABLE_LIVE_TRADING=true ayarlayın."
        )
    if not request.confirm:
        raise LiveTradingDisabled("İstekte confirm=true olmadan gerçek emir gönderilmez.")
    if request.order_type == "limit" and request.price is None:
        raise ValueError("Limit emir için price zorunludur.")

    exchange = get_trading_exchange()
    logger.warning(
        "LIVE ORDER (testnet=%s): %s %s %s amount=%s price=%s",
        settings.binance_testnet,
        request.side,
        request.symbol,
        request.order_type,
        request.amount,
        request.price,
    )
    return exchange.place_order(
        symbol=request.symbol,
        side=request.side,
        order_type=request.order_type,
        amount=request.amount,
        price=request.price,
        market_type=request.market_type,
    )
