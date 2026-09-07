from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.backtest.runner import run_backtest_report
from app.backtest.schemas import (
    BacktestReport,
    BacktestRequest,
    SystemBacktestReport,
    SystemBacktestRequest,
)
from app.backtest.system_runner import (
    ConfidenceSweepPoint,
    PositionSizingSweepPoint,
    run_system_backtest,
    sweep_confidence_thresholds,
    sweep_position_sizing,
)
from app.core.config import settings
from app.db import repository as db
from app.exchanges import get_exchange
from app.ml.meta_label import DEFAULT_META_MODEL_PATH, MetaLabelModel
from app.ml.model import DEFAULT_MODEL_PATH, SignalModel
from app.ml.model_paths import DEFAULT_LSTM_MODEL_PATH
from app.ml.model_status import is_model_enabled
from app.ml.online_model import DEFAULT_ONLINE_MODEL_PATH, OnlineSignalModel

if TYPE_CHECKING:
    from app.ml.lstm_model import LSTMSignalModel

router = APIRouter(prefix="/backtest", tags=["backtest"])


def _load_ensemble_models() -> tuple[SignalModel, MetaLabelModel | None, LSTMSignalModel | None, OnlineSignalModel | None]:
    """`/system/run` ve `/system/sweep-confidence` AYNI eğitilmiş modelleri
    (canlı karar motorunun kullandığı) yükler — bkz. `app.ml.model_status`:
    bir model yalnızca EN SON eğitiminde kalite eşiğini geçtiyse aktiftir.

    `LSTMSignalModel` (torch, ~460MB) BİLEREK burada, sadece GERÇEKTEN
    aktifse import edilir — bkz. `app.ml.model_paths` docstring'i."""
    if not DEFAULT_MODEL_PATH.exists():
        raise HTTPException(status_code=422, detail="Model henüz eğitilmedi. Önce /ml/train çağırın.")
    model = SignalModel.load_from()
    meta_model = MetaLabelModel.load_from() if DEFAULT_META_MODEL_PATH.exists() else None
    lstm_model = None
    if is_model_enabled(DEFAULT_LSTM_MODEL_PATH):
        from app.ml.lstm_model import LSTMSignalModel

        lstm_model = LSTMSignalModel.load_from()
    online_model = OnlineSignalModel.load_from() if is_model_enabled(DEFAULT_ONLINE_MODEL_PATH) else None
    return model, meta_model, lstm_model, online_model


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
    modelleri (XGBoost birincil + varsa meta-label filtresi + varsa
    LSTM/online ensemble, bkz. `app.ml.model_status`), `payload.symbol`
    için gerçek geçmiş mumlar (varsayılan: BTC/USDT:USDT futures perpetual,
    1h, 10.000 mum) üzerinde bar-bar tekrar oynatır. Sonuç DB'ye kaydedilir
    (Grafana candlestick/PnL panelleri ve `GET /backtest/system/latest`
    buradan okur)."""
    exchange = get_exchange(settings.exchange_id)
    model, meta_model, lstm_model, online_model = _load_ensemble_models()
    try:
        return run_system_backtest(exchange, model, meta_model, payload, lstm_model=lstm_model, online_model=online_model)
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


class SweepConfidenceRequest(BaseModel):
    base_request: SystemBacktestRequest = Field(default_factory=SystemBacktestRequest)
    open_confidence_values: list[float] = [0.5, 0.55, 0.6, 0.65, 0.7]
    close_confidence_gap: float = Field(
        default=0.05,
        description="Her denenen open_confidence için close_confidence = open_confidence - bu değer (şemanın kendi 0.5/0.45 varsayılan boşluğuyla AYNI).",
    )


class SweepConfidencePoint(BaseModel):
    open_confidence: float
    close_confidence: float
    trades_closed: int
    win_rate_pct: float
    total_pnl_pct: float
    daily_pnl_pct: float
    max_drawdown_pct: float
    error: str | None = None


class SweepConfidenceResponse(BaseModel):
    points: list[SweepConfidencePoint]


@router.post("/system/sweep-confidence", response_model=SweepConfidenceResponse)
def sweep_confidence(payload: SweepConfidenceRequest) -> SweepConfidenceResponse:
    """Farklı `open_confidence` eşikleriyle art arda sistem backtest'i
    çalıştırıp her biri için işlem sayısı/kazanma oranı/PnL/max drawdown
    döner — "eşiği sıkılaştırmak kârlılığı artırır mı" sorusuna CEVAP
    değil, CEVABI BULMAK İÇİN VERİ sağlar (otomatik "en iyi"yi seçmez, bkz.
    `app.backtest.system_runner.sweep_confidence_thresholds` docstring'i:
    az işlemle görülen yüksek bir kazanma oranı, çok işlemle görülen daha
    düşük bir orandan DAHA GÜVENİLİR değildir — karar operatöre kalır).

    `base_request`'teki `open_confidence`/`close_confidence` alanları HER
    nokta için üzerine yazılır (`close_confidence_gap`'e göre türetilir);
    diğer tüm alanlar (sembol, ATR ayarları, pozisyon boyutlandırma, vb.)
    sabit kalır. Ara denemeler `backtest_runs` tablosuna YAZILMAZ."""
    exchange = get_exchange(settings.exchange_id)
    model, meta_model, lstm_model, online_model = _load_ensemble_models()
    points: list[ConfidenceSweepPoint] = sweep_confidence_thresholds(
        exchange,
        model,
        meta_model,
        payload.base_request,
        payload.open_confidence_values,
        close_confidence_gap=payload.close_confidence_gap,
        lstm_model=lstm_model,
        online_model=online_model,
    )
    return SweepConfidenceResponse(points=[SweepConfidencePoint(**p.__dict__) for p in points])


class SweepPositionSizingRequest(BaseModel):
    base_request: SystemBacktestRequest = Field(default_factory=SystemBacktestRequest)
    kelly_min_trades_values: list[int] = [10, 20, 40, 60]
    kelly_multiplier_values: list[float] = [0.5, 0.75, 1.0]


class SweepPositionSizingPoint(BaseModel):
    kelly_min_trades: int
    kelly_multiplier: float
    trades_closed: int
    win_rate_pct: float
    total_pnl_pct: float
    daily_pnl_pct: float
    max_drawdown_pct: float
    error: str | None = None


class SweepPositionSizingResponse(BaseModel):
    points: list[SweepPositionSizingPoint]


@router.post("/system/sweep-position-sizing", response_model=SweepPositionSizingResponse)
def sweep_position_sizing_route(payload: SweepPositionSizingRequest) -> SweepPositionSizingResponse:
    """`kelly_min_trades` × `kelly_multiplier` ızgarasında art arda sistem
    backtest'i çalıştırıp her kombinasyon için işlem sayısı/kazanma oranı/
    PnL/max drawdown döner — bkz. `app.backtest.system_runner.sweep_position_sizing`
    docstring'i: gerçek üretim backtest'inde, ilk ~40 işlemlik gürültülü
    pencerede Kelly'nin "tam Kelly=%0" hesaplayıp ~25-30 işlemi TAMAMEN
    SIFIR boyutla açtığı gözlendi (sermaye o dönem hiç kullanılmadı) —
    `kelly_min_trades`'i artırmanın bunu önleyip önlemediğini, `kelly_multiplier`'ı
    artırmanın (düşük drawdown'lı dönemlerde) getiriyi büyütüp büyütmediğini
    ölçmek için. Otomatik "en iyi"yi seçmez, karar operatöre kalır. Ara
    denemeler `backtest_runs` tablosuna YAZILMAZ."""
    exchange = get_exchange(settings.exchange_id)
    model, meta_model, lstm_model, online_model = _load_ensemble_models()
    points: list[PositionSizingSweepPoint] = sweep_position_sizing(
        exchange,
        model,
        meta_model,
        payload.base_request,
        payload.kelly_min_trades_values,
        payload.kelly_multiplier_values,
        lstm_model=lstm_model,
        online_model=online_model,
    )
    return SweepPositionSizingResponse(points=[SweepPositionSizingPoint(**p.__dict__) for p in points])
