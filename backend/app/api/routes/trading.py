from fastapi import APIRouter, HTTPException

from app.trading.executor import LiveTradingDisabled, get_trading_exchange, place_live_order
from app.trading.schemas import OrderRequest, OrderResult

router = APIRouter(prefix="/trading", tags=["trading"])


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
