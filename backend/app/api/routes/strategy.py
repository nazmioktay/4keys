from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.exchanges import get_exchange
from app.strategy.engine import run_backtest
from app.strategy.examples import EXAMPLES
from app.strategy.schemas import StrategyBacktestRequest, StrategyBacktestResult, StrategyDefinition

router = APIRouter(prefix="/strategy", tags=["strategy"])


@router.get("/examples", response_model=dict[str, StrategyDefinition])
def list_examples() -> dict[str, StrategyDefinition]:
    """Hazır strateji örnekleri — TradingView'da Pine Script yazmadan doğrudan
    /strategy/backtest gövdesine kopyalanıp denenebilir veya değiştirilebilir."""
    return EXAMPLES


@router.post("/backtest", response_model=StrategyBacktestResult)
def backtest(payload: StrategyBacktestRequest) -> StrategyBacktestResult:
    exchange = get_exchange(settings.exchange_id)
    ohlcv = exchange.fetch_ohlcv(
        payload.symbol,
        payload.timeframe or settings.candle_timeframe,
        payload.lookback or settings.candle_lookback,
    )
    if len(ohlcv) < 30:
        raise HTTPException(status_code=422, detail="Backtest için yeterli geçmiş veri yok.")

    return run_backtest(ohlcv, payload.strategy, payload.symbol)
