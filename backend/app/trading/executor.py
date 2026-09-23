import logging

from app.core.config import settings
from app.exchanges.binance import BinanceExchange
from app.security import kill_switch
from app.security.safety import check_withdrawals_disabled, enforce_leverage_cap

from .schemas import CancelOrderRequest, LeverageRequest, MarginModeRequest, OrderRequest

logger = logging.getLogger(__name__)


class LiveTradingDisabled(Exception):
    """Canlı işlem güvenlik kapılarından biri kapalı olduğunda fırlatılır."""


def _credentials_present() -> bool:
    return bool(settings.binance_api_key.get_secret_value()) and bool(settings.binance_api_secret.get_secret_value())


_cached_exchange: BinanceExchange | None = None


def get_trading_exchange() -> BinanceExchange:
    """Ayarlardaki kimlik bilgileriyle kimlik doğrulamalı Binance istemcisini
    döner — süreç ömrü boyunca TEK bir örnek olarak önbelleğe alınır.

    ÖNEMLİ: Her çağrıda yeni bir `BinanceExchange` (ve dolayısıyla yeni bir
    ccxt istemcisi) oluşturmak, ccxt'nin `create_order`/`set_leverage`
    içinde otomatik çağırdığı `load_markets()`in HER SEFERİNDE ~1.1MB'lık
    `/fapi/v1/exchangeInfo`i sıfırdan çekip ayrıştırmasına (bu container'da
    6-16sn arası sürüyor) yol açıyordu — art arda emir+kaldıraç gönderiminde
    bu, ccxt'nin timeout'unu (bkz. `BinanceExchange.__init__`) aşıp
    `RequestTimeout`a çarpabiliyordu. Tek örneği önbelleğe almak, ccxt'nin
    kendi iç piyasa önbelleğinin (`self.markets`) süreç boyunca kalıcı
    olmasını sağlar — ilk çağrıdan sonraki emirler bu pahalı çekimi
    tekrarlamaz.
    """
    global _cached_exchange
    if not _credentials_present():
        raise LiveTradingDisabled(
            "Binance API anahtarı tanımlı değil. FOURKEYS_BINANCE_API_KEY ve "
            "FOURKEYS_BINANCE_API_SECRET ortam değişkenlerini (.env) ayarlayın."
        )
    if _cached_exchange is None:
        _cached_exchange = BinanceExchange(
            api_key=settings.binance_api_key.get_secret_value(),
            api_secret=settings.binance_api_secret.get_secret_value(),
            testnet=settings.binance_testnet,
        )
    return _cached_exchange


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
    entry = exchange.place_order(
        symbol=request.symbol,
        side=request.side,
        order_type=request.order_type,
        amount=request.amount,
        price=request.price,
        market_type=request.market_type,
        reduce_only=request.reduce_only,
    )

    if not (request.stop_loss_price or request.take_profit_price):
        return entry

    # TP/SL yalnızca POZİSYON AÇAN/ARTIRAN ana emirlere anlamlıdır — kapatma
    # (reduce_only) emrinde zaten ters yönde bir emirdir, tekrar TP/SL
    # eklemek anlamsız/tehlikeli olurdu (bkz. aşağıdaki close_side mantığı).
    result: dict = {"entry": entry}
    close_side = "sell" if request.side == "buy" else "buy"
    if request.stop_loss_price and not request.reduce_only:
        logger.warning("LIVE STOP LOSS: %s %s stopPrice=%s", request.symbol, close_side, request.stop_loss_price)
        result["stop_loss"] = exchange.place_conditional_order(
            symbol=request.symbol,
            side=close_side,
            amount=request.amount,
            stop_price=request.stop_loss_price,
            kind="stop_loss",
            market_type=request.market_type,
        )
    if request.take_profit_price and not request.reduce_only:
        logger.warning("LIVE TAKE PROFIT: %s %s stopPrice=%s", request.symbol, close_side, request.take_profit_price)
        result["take_profit"] = exchange.place_conditional_order(
            symbol=request.symbol,
            side=close_side,
            amount=request.amount,
            stop_price=request.take_profit_price,
            kind="take_profit",
            market_type=request.market_type,
        )
    return result


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


def set_live_margin_mode(request: MarginModeRequest) -> dict:
    """Gerçek marjin modunu (cross/isolated) değiştirir — aynı güvenlik
    kapılarından geçer. Binance zaten o moddaysa -4046 hatası döner; bu
    hata (istenen durum zaten sağlanmış demek olduğu için) yutulup
    başarı gibi ele alınır."""
    _run_cheap_safety_gates(request.confirm, "İstekte confirm=true olmadan marjin modu değiştirilmez.")
    exchange = get_trading_exchange()
    _verify_withdrawals_disabled(exchange)

    logger.warning("LIVE MARGIN MODE CHANGE (testnet=%s): %s -> %s", settings.binance_testnet, request.symbol, request.mode)
    try:
        return exchange.set_margin_mode(request.symbol, request.mode)
    except Exception as exc:  # noqa: BLE001 - yalnızca "zaten bu modda" durumunu yut, başkasını yeniden fırlat
        if "-4046" in str(exc) or "No need to change margin type" in str(exc):
            return {"info": "already in requested margin mode"}
        raise


def cancel_live_order(request: CancelOrderRequest) -> dict:
    """Bekleyen (henüz dolmamış) gerçek bir emri iptal eder — aynı güvenlik
    kapılarından geçer (iptal etmek risk AZALTSA da, gerçek hesaba giden
    her yazma işlemi aynı tutarlı kapılardan geçirilir)."""
    _run_cheap_safety_gates(request.confirm, "İstekte confirm=true olmadan emir iptal edilmez.")
    exchange = get_trading_exchange()
    _verify_withdrawals_disabled(exchange)

    logger.warning("LIVE CANCEL ORDER (testnet=%s): %s order_id=%s", settings.binance_testnet, request.symbol, request.order_id)
    return exchange.cancel_order(request.order_id, request.symbol, request.market_type)
