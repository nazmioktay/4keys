import logging

from app.core.config import settings
from app.exchanges.algolab import AlgoLabExchange
from app.security import kill_switch

from .schemas import BistOrderRequest

logger = logging.getLogger(__name__)

# Süreç ömrü boyunca paylaşılan tek AlgoLab oturumu — oturum hash'i burada
# tutulur, her istek yeniden login olmak zorunda kalmaz.
_exchange: AlgoLabExchange | None = None


class BistTradingDisabled(Exception):
    """BIST/VIOP güvenlik kapılarından biri kapalı olduğunda fırlatılır."""


def _api_key_present() -> bool:
    return bool(settings.algolab_api_key.get_secret_value())


def get_exchange() -> AlgoLabExchange:
    global _exchange
    if not _api_key_present():
        raise BistTradingDisabled(
            "AlgoLab API anahtarı tanımlı değil. FOURKEYS_ALGOLAB_API_KEY ortam değişkenini (.env) ayarlayın."
        )
    if _exchange is None:
        _exchange = AlgoLabExchange(api_key=settings.algolab_api_key.get_secret_value(), base_url=settings.algolab_base_url)
    return _exchange


def login(username: str | None, password: str | None) -> str:
    exchange = get_exchange()
    user = username or settings.algolab_username
    pw = password or settings.algolab_password.get_secret_value()
    if not user or not pw:
        raise BistTradingDisabled(
            "Kullanıcı adı/şifre verilmedi ve .env'de FOURKEYS_ALGOLAB_USERNAME/FOURKEYS_ALGOLAB_PASSWORD tanımlı değil."
        )
    return exchange.login(user, pw)


def login_verify(token: str, sms_code: str) -> bool:
    exchange = get_exchange()
    exchange.login_control(token, sms_code)
    return exchange.is_authenticated()


def place_bist_order(request: BistOrderRequest) -> dict:
    """BIST/VIOP'a gerçek emir gönderir — Binance canlı işlem katmanıyla
    (bkz. `app.trading.executor`) aynı güvenlik prensiplerini uygular:

    1. Kill switch aktif değil (bkz. `app.security.kill_switch`)
    2. `FOURKEYS_ENABLE_BIST_TRADING=true`
    3. İstekte `confirm: true`
    4. AlgoLab oturumu açık (login + login_verify tamamlanmış)
    """
    if kill_switch.is_active():
        raise BistTradingDisabled(f"Kill switch aktif: {kill_switch.status().reason}")
    if not settings.enable_bist_trading:
        raise BistTradingDisabled("BIST/VIOP canlı işlem devre dışı. FOURKEYS_ENABLE_BIST_TRADING=true ayarlayın.")
    if not request.confirm:
        raise BistTradingDisabled("İstekte confirm=true olmadan gerçek emir gönderilmez.")
    if request.order_type == "limit" and request.price is None:
        raise ValueError("Limit emir için price zorunludur.")

    exchange = get_exchange()
    if not exchange.is_authenticated():
        raise BistTradingDisabled("AlgoLab oturumu açık değil. Önce /bist/login ve /bist/login/verify çağırın.")

    logger.warning(
        "LIVE BIST ORDER: %s %s %s quantity=%s price=%s market=%s",
        request.direction,
        request.symbol,
        request.order_type,
        request.quantity,
        request.price,
        request.market_type,
    )
    return exchange.send_order(
        symbol=request.symbol,
        direction=request.direction,
        order_type=request.order_type,
        quantity=request.quantity,
        price=request.price,
        market_type=request.market_type,
    )


def reset_for_tests() -> None:
    global _exchange
    _exchange = None
