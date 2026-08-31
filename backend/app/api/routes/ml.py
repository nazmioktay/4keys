from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.config import settings
from app.exchanges import get_exchange
from app.ml.dataset import LabelingMethod
from app.ml.features import latest_feature_vector
from app.ml.meta_label import DEFAULT_META_MODEL_PATH, MetaLabelModel
from app.ml.model import DEFAULT_MODEL_PATH, SignalModel
from app.ml.train import train_meta_label_model, train_signal_model
from app.screener.scanner import scan_market, top_long, top_short

router = APIRouter(prefix="/ml", tags=["ml"])


class TrainRequest(BaseModel):
    symbols: list[str] | None = None  # None -> screener top long+short kullanılır
    horizon: int = 5
    threshold_pct: float = 1.0
    labeling_method: LabelingMethod = "threshold"
    take_profit_pct: float = 2.0
    stop_loss_pct: float = 2.0
    calibrate: bool = True
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"


class TrainResponse(BaseModel):
    rows_used: int
    symbols_used: int


class TrainMetaRequest(BaseModel):
    symbols: list[str] | None = None
    horizon: int = 5
    threshold_pct: float = 1.0
    labeling_method: LabelingMethod = "threshold"
    take_profit_pct: float = 2.0
    stop_loss_pct: float = 2.0


class PredictResponse(BaseModel):
    symbol: str
    direction: str
    confidence: float
    meta_act: bool | None = None
    meta_confidence: float | None = None


def _model_exists() -> bool:
    return Path(DEFAULT_MODEL_PATH).exists()


def _resolve_symbols(exchange, symbols: list[str] | None) -> list[str]:
    if symbols:
        return symbols
    results = scan_market(exchange)
    picks = top_long(results, settings.screener_top_n) + top_short(results, settings.screener_top_n)
    return [r.symbol for r in picks]


@router.post("/train", response_model=TrainResponse)
def train(payload: TrainRequest) -> TrainResponse:
    exchange = get_exchange(settings.exchange_id)
    symbols = _resolve_symbols(exchange, payload.symbols)
    if not symbols:
        raise HTTPException(status_code=400, detail="Eğitim için sembol bulunamadı.")

    try:
        _, rows_used = train_signal_model(
            exchange,
            symbols,
            horizon=payload.horizon,
            threshold_pct=payload.threshold_pct,
            labeling_method=payload.labeling_method,
            take_profit_pct=payload.take_profit_pct,
            stop_loss_pct=payload.stop_loss_pct,
            calibrate=payload.calibrate,
            calibration_method=payload.calibration_method,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return TrainResponse(rows_used=rows_used, symbols_used=len(symbols))


@router.post("/train-meta", response_model=TrainResponse)
def train_meta(payload: TrainMetaRequest) -> TrainResponse:
    """Meta-label modelini eğitir: birincil modelin sinyaline "gir/girme"
    kararı verecek ikinci bir model (bkz. app/ml/meta_label.py). Önce
    /ml/train ile birincil model eğitilmiş olmalıdır."""
    if not _model_exists():
        raise HTTPException(status_code=409, detail="Önce birincil modeli /ml/train ile eğitin.")

    exchange = get_exchange(settings.exchange_id)
    symbols = _resolve_symbols(exchange, payload.symbols)
    if not symbols:
        raise HTTPException(status_code=400, detail="Eğitim için sembol bulunamadı.")

    primary_model = SignalModel.load_from()

    try:
        _, rows_used = train_meta_label_model(
            exchange,
            symbols,
            primary_model,
            horizon=payload.horizon,
            threshold_pct=payload.threshold_pct,
            labeling_method=payload.labeling_method,
            take_profit_pct=payload.take_profit_pct,
            stop_loss_pct=payload.stop_loss_pct,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return TrainResponse(rows_used=rows_used, symbols_used=len(symbols))


@router.get("/predict", response_model=PredictResponse)
def predict(symbol: str = Query(..., description="Örn: BTC/USDT:USDT")) -> PredictResponse:
    if not _model_exists():
        raise HTTPException(status_code=409, detail="Model henüz eğitilmedi. Önce /ml/train çağırın.")

    exchange = get_exchange(settings.exchange_id)
    ohlcv = exchange.fetch_ohlcv(symbol, settings.candle_timeframe, settings.candle_lookback)
    feature_row = latest_feature_vector(ohlcv)
    if feature_row is None:
        raise HTTPException(status_code=422, detail="Yeterli veri yok.")

    model = SignalModel.load_from()
    prediction = model.predict(feature_row)

    meta_act = None
    meta_confidence = None
    if DEFAULT_META_MODEL_PATH.exists():
        meta_model = MetaLabelModel.load_from()
        meta_decision = meta_model.decide(feature_row, prediction.confidence)
        meta_act = meta_decision.act
        meta_confidence = meta_decision.confidence

    return PredictResponse(
        symbol=symbol,
        direction=prediction.direction,
        confidence=prediction.confidence,
        meta_act=meta_act,
        meta_confidence=meta_confidence,
    )
