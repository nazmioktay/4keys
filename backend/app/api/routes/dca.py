from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.dca.optimizer import optimize_dca
from app.dca.schemas import DCAOptimizeRequest, DCAOptimizeResponse
from app.exchanges import get_exchange

router = APIRouter(prefix="/dca", tags=["dca"])


@router.post("/optimize", response_model=DCAOptimizeResponse)
def optimize(payload: DCAOptimizeRequest) -> DCAOptimizeResponse:
    exchange = get_exchange(settings.exchange_id)
    ohlcv = exchange.fetch_ohlcv(
        payload.symbol,
        payload.timeframe or settings.candle_timeframe,
        payload.lookback or settings.candle_lookback,
    )
    if len(ohlcv) < 30:
        raise HTTPException(status_code=422, detail="Optimizasyon için yeterli geçmiş veri yok.")

    prices = ohlcv["close"].to_numpy()
    candidates = optimize_dca(
        prices=prices,
        balance=payload.balance,
        direction=payload.direction,
        objective=payload.objective,
        allow_stop_loss=payload.allow_stop_loss,
        top_n=payload.top_n,
    )

    if not candidates:
        raise HTTPException(
            status_code=422,
            detail="Verilen veri aralığında hiçbir parametre kombinasyonu bir işlemi tamamlayamadı.",
        )

    return DCAOptimizeResponse(symbol=payload.symbol, candidates=candidates)
