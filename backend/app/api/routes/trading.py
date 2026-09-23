import time

from fastapi import APIRouter, HTTPException

from app.exchanges import get_exchange
from app.security.safety import MAX_LEVERAGE
from app.trading.executor import LiveTradingDisabled, get_trading_exchange, place_live_order, set_live_leverage
from app.trading.schemas import LeverageRequest, LeverageResult, OrderRequest, OrderResult

router = APIRouter(prefix="/trading", tags=["trading"])


@router.get("/price")
def price(symbol: str, market_type: str = "future") -> dict:
    """Herkese açık, kimlik doğrulamasız anlık fiyat — `.env`'deki hesap
    anahtarlarına dokunmaz, canlı işlem kapıları kapalıyken de çalışır."""
    try:
        last = get_exchange("binance").fetch_ticker_price(symbol, market_type)
    except Exception as exc:  # noqa: BLE001 - borsa/ağ hatası doğrudan mesaj olarak dönsün
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"symbol": symbol, "last": last, "max_leverage": MAX_LEVERAGE}


@router.get("/balance")
def balance(market_type: str = "future") -> dict:
    try:
        exchange = get_trading_exchange()
        return exchange.fetch_balance(market_type)
    except LiveTradingDisabled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/positions")
def positions() -> list[dict]:
    try:
        exchange = get_trading_exchange()
        return exchange.fetch_positions()
    except LiveTradingDisabled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/pnl-summary")
def pnl_summary() -> dict:
    """Gerçek hesabın GERÇEKLEŞMİŞ (kapanmış) PNL'ini gün/hafta/ay/toplam
    pencerelerinde özetler — Paper Trading/Otopilot'taki `/portfolio/pnl`
    ile aynı şekli döner, böylece frontend'de aynı `PnlCard` deseni
    kullanılabilir. Bkz. `BinanceExchange.fetch_income_history` — en son
    1000 kayıt çekilir (yeni bu hesapta pratikte tüm geçmiş)."""
    try:
        exchange = get_trading_exchange()
        records = exchange.fetch_income_history(limit=1000)
    except LiveTradingDisabled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    now_ms = int(time.time() * 1000)

    def window(days: float | None) -> dict:
        cutoff = now_ms - int(days * 24 * 3600 * 1000) if days else 0
        rows = [r for r in records if r["time"] >= cutoff]
        pnl = sum(r["income"] for r in rows)
        wins = sum(1 for r in rows if r["income"] > 0)
        count = len(rows)
        return {
            "pnl_quote": round(pnl, 4),
            "trade_count": count,
            "win_rate_pct": round(wins / count * 100, 2) if count else 0.0,
        }

    return {"daily": window(1), "weekly": window(7), "monthly": window(30), "total": window(None)}


@router.post("/order", response_model=OrderResult)
def order(payload: OrderRequest) -> OrderResult:
    """Gerçek borsaya emir gönderir.

    Bkz. `app/trading/executor.py` — bu uç nokta yalnızca
    `FOURKEYS_ENABLE_LIVE_TRADING=true` VE gövdede `confirm: true` ikisi
    birden sağlandığında gerçekten emir gönderir. Aksi halde 409 döner.
    """
    try:
        raw = place_live_order(payload)
    except LiveTradingDisabled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return OrderResult(raw=raw)


@router.post("/leverage", response_model=LeverageResult)
def leverage(payload: LeverageRequest) -> LeverageResult:
    """Gerçek kaldıracı değiştirir. `order` ile aynı güvenlik kapılarına ek
    olarak, kod içi sabit bir tavana (bkz. `app.security.safety.MAX_LEVERAGE`,
    Güvenlik Protokolü Bölüm 9.3) tabidir — bu tavan `.env` ile aşılamaz."""
    try:
        raw = set_live_leverage(payload)
    except LiveTradingDisabled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return LeverageResult(raw=raw)
