import logging

from app.core.config import settings
from app.exchanges.binance import BinanceExchange
from app.security import kill_switch
from app.security.safety import check_withdrawals_disabled, enforce_leverage_cap

from .schemas import LeverageRequest, OrderRequest

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


def _run_cheap_safety_gates(confirm: bool, confirm_message: str) -> None:
    """Kimlik bilgisi/borsa çağrısı gerektirmeyen, hızlı ve ucuz kontroller —
    bunlar en anlaşılır hata mesajını vermek için önce çalışır."""
    if kill_switch.is_active():
        raise LiveTradingDisabled(f"Kill switch aktif: {kill_switch.status().reason}")
    if not settings.enable_live_trading:
        raise LiveTradingDisabled(
            "Canlı işlem devre dışı. Ortam değişkeninde FOURKEYS_ENABLE_LIVE_TRADING=true ayarlayın."
        )
    if not confirm:
        raise LiveTradingDisabled(confirm_message)


def _verify_withdrawals_disabled(exchange: BinanceExchange) -> None:
    if settings.require_api_key_permission_check:
        verified_safe, message = check_withdrawals_disabled(exchange)
        if not verified_safe:
            raise LiveTradingDisabled(message)


def place_live_order(request: OrderRequest) -> dict:
    """Gerçek borsaya emir gönderir — sırayla şu güvenlik kapılarından geçer:

    1. Kill switch aktif değil (bkz. `app.security.kill_switch`)
    2. `FOURKEYS_ENABLE_LIVE_TRADING=true` (ortam değişkeni, operatör düzeyinde anahtar)
    3. İstekte `confirm: true` (her çağrıda ayrı, yanlışlıkla tetiklenmeyi önler)
    4. Binance API anahtarlarının tanımlı olması
    5. Binance API anahtarının ÇEKİM izninin kapalı olduğu doğrulanmış
       (`FOURKEYS_REQUIRE_API_KEY_PERMISSION_CHECK=true`, varsayılan)

    Herhangi biri eksikse emir gönderilmez. `FOURKEYS_BINANCE_TESTNET=true`
    (varsayılan) iken bile bu kapılar aktiftir.
    """
    if request.order_type == "limit" and request.price is None:
        raise ValueError("Limit emir için price zorunludur.")

    _run_cheap_safety_gates(request.confirm, "İstekte confirm=true olmadan gerçek emir gönderilmez.")
    exchange = get_trading_exchange()
    _verify_withdrawals_disabled(exchange)

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


def set_live_leverage(request: LeverageRequest) -> dict:
    """Gerçek kaldıracı değiştirir — `place_live_order` ile aynı güvenlik
    kapılarından geçer, ARTI kod içi sabit kaldıraç tavanı (bkz.
    `app.security.safety.MAX_LEVERAGE`) — bu tavan `.env` ile aşılamaz."""
    enforce_leverage_cap(request.leverage)

    _run_cheap_safety_gates(request.confirm, "İstekte confirm=true olmadan kaldıraç değiştirilmez.")
    exchange = get_trading_exchange()
    _verify_withdrawals_disabled(exchange)

    logger.warning("LIVE LEVERAGE CHANGE (testnet=%s): %s -> %sx", settings.binance_testnet, request.symbol, request.leverage)
    return exchange.set_leverage(request.symbol, request.leverage)
