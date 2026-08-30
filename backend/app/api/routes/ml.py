from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.config import settings
from app.exchanges import get_exchange
from app.ml.features import latest_feature_vector
from app.ml.model import DEFAULT_MODEL_PATH, SignalModel
from app.ml.train import train_signal_model
from app.screener.scanner import scan_market, top_long, top_short

router = APIRouter(prefix="/ml", tags=["ml"])


class TrainRequest(BaseModel):
    symbols: list[str] | None = None  # None -> screener top long+short kullanılır
    horizon: int = 5
    threshold_pct: float = 1.0


class TrainResponse(BaseModel):
    rows_used: int
    symbols_used: int


class PredictResponse(BaseModel):
    symbol: str
    direction: str
    confidence: float


def _model_exists() -> bool:
    return Path(DEFAULT_MODEL_PATH).exists()


@router.post("/train", response_model=TrainResponse)
def train(payload: TrainRequest) -> TrainResponse:
    exchange = get_exchange(settings.exchange_id)

    symbols = payload.symbols
    if not symbols:
        results = scan_market(exchange)
        picks = top_long(results, settings.screener_top_n) + top_short(results, settings.screener_top_n)
        symbols = [r.symbol for r in picks]

    if not symbols:
        raise HTTPException(status_code=400, detail="Eğitim için sembol bulunamadı.")

    try:
        _, rows_used = train_signal_model(
            exchange, symbols, horizon=payload.horizon, threshold_pct=payload.threshold_pct
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
    return PredictResponse(symbol=symbol, direction=prediction.direction, confidence=prediction.confidence)
