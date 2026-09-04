from fastapi import APIRouter, HTTPException

from app.backtest.runner import run_backtest_report
from app.backtest.schemas import (
    BacktestReport,
    BacktestRequest,
    SystemBacktestReport,
    SystemBacktestRequest,
)
from app.backtest.system_runner import run_system_backtest
from app.core.config import settings
from app.db import repository as db
from app.exchanges import get_exchange
from app.ml.meta_label import DEFAULT_META_MODEL_PATH, MetaLabelModel
from app.ml.model import DEFAULT_MODEL_PATH, SignalModel

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


@router.post("/system/run", response_model=SystemBacktestReport)
def run_system(payload: SystemBacktestRequest) -> SystemBacktestReport:
    """"Sistem backtest"i: canlı karar motorunun kullandığı AYNI eğitilmiş
    modeli (+ varsa meta-label filtresi), `payload.symbol` için gerçek
    geçmiş mumlar (varsayılan: BTC/USDT:USDT futures perpetual, 1h,
    10.000 mum) üzerinde bar-bar tekrar oynatır. Sonuç DB'ye kaydedilir
    (Grafana candlestick/PnL panelleri ve `GET /backtest/system/latest`
    buradan okur)."""
    if not DEFAULT_MODEL_PATH.exists():
        raise HTTPException(status_code=422, detail="Model henüz eğitilmedi. Önce /ml/train çağırın.")

    exchange = get_exchange(settings.exchange_id)
    model = SignalModel.load_from()
    meta_model = MetaLabelModel.load_from() if DEFAULT_META_MODEL_PATH.exists() else None
    try:
        return run_system_backtest(exchange, model, meta_model, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/system/latest", response_model=SystemBacktestReport | None)
def get_latest_system_run(symbol: str | None = None) -> SystemBacktestReport | None:
    """En son kaydedilmiş sistem backtest çalıştırmasını döner (yoksa null)
    — sayfa yeniden açıldığında son sonucu göstermek için."""
    run = db.get_latest_backtest_run(symbol=symbol)
    if run is None:
        return None
    return SystemBacktestReport(**run)
