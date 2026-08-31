from fastapi import APIRouter, HTTPException

from app.backtest.runner import run_backtest_report
from app.backtest.schemas import BacktestReport, BacktestRequest
from app.core.config import settings
from app.exchanges import get_exchange

router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.post("/run", response_model=BacktestReport)
def run(payload: BacktestRequest) -> BacktestReport:
    """Güçlü, tek uçlu backtest: gerekli geçmiş veri miktarını otomatik
    keşfeder, eğitim/test (in-sample/out-of-sample) ayrımı yapar ve
    Sharpe/Sortino/Calmar/profit factor dahil zengin metrikler döner.

    `dca_params` veya `strategy` alanlarından tam olarak biri verilmelidir.
    `timeframe` boş bırakılırsa varsayılan ayarlar kullanılır.
    """
    if payload.timeframe is None:
        payload = payload.model_copy(update={"timeframe": settings.candle_timeframe})

    exchange = get_exchange(settings.exchange_id)
    try:
        return run_backtest_report(exchange, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
