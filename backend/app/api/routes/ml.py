from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.config import settings
from app.exchanges import get_exchange
from app.ml.dataset import LabelingMethod
from app.ml.features import latest_feature_vector
from app.ml.meta_label import DEFAULT_META_MODEL_PATH, MetaLabelModel
from app.ml.model import DEFAULT_MODEL_PATH, Algorithm, SignalModel
from app.ml.train import train_meta_label_model, train_signal_model_validated
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
    algorithm: Algorithm = "xgboost"
    holdout_frac: float = 0.2
    walk_forward_splits: int = 5


class FoldMetric(BaseModel):
    fold: int
    train_rows: int
    test_rows: int
    accuracy: float
    balanced_accuracy: float


class TrainResponse(BaseModel):
    rows_used: int
    symbols_used: int
    algorithm: Algorithm
    walk_forward_folds: list[FoldMetric]
    walk_forward_mean_accuracy: float
    walk_forward_mean_balanced_accuracy: float
    overfit_gap: float
    out_of_sample_rows: int
    out_of_sample_accuracy: float
    out_of_sample_balanced_accuracy: float


class ShapImportance(BaseModel):
    feature: str
    mean_abs_shap: float


class ExplainResponse(BaseModel):
    symbols_used: int
    rows_explained: int
    importances: list[ShapImportance]


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
        result = train_signal_model_validated(
            exchange,
            symbols,
            horizon=payload.horizon,
            threshold_pct=payload.threshold_pct,
            labeling_method=payload.labeling_method,
            take_profit_pct=payload.take_profit_pct,
            stop_loss_pct=payload.stop_loss_pct,
            calibrate=payload.calibrate,
            calibration_method=payload.calibration_method,
            algorithm=payload.algorithm,
            holdout_frac=payload.holdout_frac,
            walk_forward_splits=payload.walk_forward_splits,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    wf = result.walk_forward
    oos = result.out_of_sample
    return TrainResponse(
        rows_used=result.rows_used,
        symbols_used=len(symbols),
        algorithm=payload.algorithm,
        walk_forward_folds=[FoldMetric(**f.__dict__) for f in wf.folds],
        walk_forward_mean_accuracy=wf.mean_accuracy,
        walk_forward_mean_balanced_accuracy=wf.mean_balanced_accuracy,
        overfit_gap=wf.overfit_gap,
        out_of_sample_rows=oos.holdout_rows,
        out_of_sample_accuracy=oos.accuracy,
        out_of_sample_balanced_accuracy=oos.balanced_accuracy,
    )


@router.get("/explain", response_model=ExplainResponse)
def explain(symbol: str | None = Query(None, description="Boş bırakılırsa screener top long+short kullanılır")) -> ExplainResponse:
    """Eğitilmiş modelin tahminlerinde her özelliğin ortalama katkısını
    (SHAP değerleri) döner — rehberin "anlamsız feature'lar elenir"
    prensibinin karşılığı. Yalnızca XGBoost modeli için çalışır."""
    if not _model_exists():
        raise HTTPException(status_code=409, detail="Model henüz eğitilmedi. Önce /ml/train çağırın.")

    exchange = get_exchange(settings.exchange_id)
    symbols = _resolve_symbols(exchange, [symbol] if symbol else None)
    if not symbols:
        raise HTTPException(status_code=400, detail="Açıklama için sembol bulunamadı.")

    from app.ml.dataset import build_training_dataset

    X, _y = build_training_dataset(exchange, symbols, settings.candle_timeframe, settings.candle_lookback)
    if len(X) == 0:
        raise HTTPException(status_code=422, detail="Açıklama için yeterli veri yok.")

    model = SignalModel.load_from()
    try:
        importance = model.shap_values(X)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ExplainResponse(
        symbols_used=len(symbols),
        rows_explained=min(len(X), 200),
        importances=[ShapImportance(feature=row.feature, mean_abs_shap=row.mean_abs_shap) for row in importance.itertuples()],
    )


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
