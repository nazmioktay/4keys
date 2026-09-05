from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.config import settings
from app.exchanges import get_exchange
from app.ml.dataset import LabelingMethod
from app.ml.features import latest_feature_vector
from app.ml.meta_label import DEFAULT_META_MODEL_PATH, MetaLabelModel
from app.ml.lstm_model import DEFAULT_LSTM_MODEL_PATH, LSTMSignalModel
from app.ml.macro_features import latest_macro_feature_row
from app.ml.orderbook_features import latest_orderbook_feature_row
from app.ml.orderflow_features import latest_taker_buy_ratio_norm
from app.ml.model import DEFAULT_MODEL_PATH, Algorithm, SignalModel
from app.ml.patchtst_model import DEFAULT_PATCHTST_MODEL_PATH, PatchTSTSignalModel
from app.ml.sequence_dataset import build_sequence_dataset
from app.ml.symbol_selection import select_training_symbols
from app.ml.train import (
    sweep_labeling_lstm,
    sweep_lookback_values,
    train_all_models,
    train_lstm_signal_model,
    train_meta_label_model,
    train_online_signal_model,
    train_patchtst_signal_model,
    train_signal_model_validated,
    train_signal_models_by_regime,
)
from app.screener.scanner import scan_market, top_long, top_short

router = APIRouter(prefix="/ml", tags=["ml"])


class TrainRequest(BaseModel):
    symbols: list[str] | None = None  # None -> screener top long+short kullanılır
    horizon: int = 5
    threshold_pct: float = 1.0
    # Varsayılan artık "atr_triple_barrier": model artık "N mum sonra %X
    # hareket etti mi" (gerçek işlemle İLGİSİZ, keyfi bir hedef) yerine
    # "ATR-ölçekli bir kâr hedefine mi yoksa stop'a mı önce ulaştı" diye
    # eğitiliyor — bu, app.backtest.system_runner/DecisionEngine'in
    # KULLANDIĞI ATR tabanlı risk yönetimiyle AYNI mantık, etiketleme ile
    # gerçek çıkış arasındaki uyumsuzluğu (objective mismatch) giderir.
    labeling_method: LabelingMethod = "atr_triple_barrier"
    take_profit_pct: float = 1.5  # atr_triple_barrier'da ATR ÇARPANI olarak yorumlanır
    stop_loss_pct: float = 1.5  # atr_triple_barrier'da ATR ÇARPANI olarak yorumlanır
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
    out_of_sample_true_class_counts: dict[str, int] = {}
    out_of_sample_predicted_class_counts: dict[str, int] = {}
    accepted: bool = True
    rejection_reason: str | None = None


class ShapImportance(BaseModel):
    feature: str
    mean_abs_shap: float


class ExplainResponse(BaseModel):
    symbols_used: int
    rows_explained: int
    importances: list[ShapImportance]


class TrainLSTMRequest(BaseModel):
    symbols: list[str] | None = None  # None -> screener top long+short kullanılır
    lookback: int | None = None  # None -> settings.ml_train_lookback; tek sembol testlerinde daha yüksek değer denenebilir
    seq_len: int = 20
    horizon: int = 5
    threshold_pct: float = 1.0
    labeling_method: LabelingMethod = "threshold"
    take_profit_pct: float = 2.0
    stop_loss_pct: float = 2.0
    holdout_frac: float = 0.2
    val_frac: float = 0.15
    epochs: int = 30
    patience: int = 5
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    feature_columns: list[str] | None = None  # None -> ALL_FEATURE_COLUMNS; /ml/explain'in SHAP sıralamasından bir alt küme verilebilir
    seed: int | None = 42  # tekrarlanabilirlik için; None -> eski rastgele davranış


class TrainLSTMResponse(BaseModel):
    rows_used: int
    symbols_used: int
    seq_len: int
    epochs_run: int
    final_train_loss: float
    final_train_accuracy: float
    best_val_loss: float | None = None
    stopped_early: bool = False
    out_of_sample_rows: int
    out_of_sample_accuracy: float
    out_of_sample_balanced_accuracy: float
    accepted: bool = True
    rejection_reason: str | None = None


class PredictLSTMResponse(BaseModel):
    symbol: str
    direction: str
    confidence: float


class TrainPatchTSTRequest(BaseModel):
    symbols: list[str] | None = None
    lookback: int | None = None
    seq_len: int = 20
    horizon: int = 5
    threshold_pct: float = 1.0
    labeling_method: LabelingMethod = "threshold"
    take_profit_pct: float = 2.0
    stop_loss_pct: float = 2.0
    holdout_frac: float = 0.2
    val_frac: float = 0.15
    epochs: int = 30
    patience: int = 5
    patch_len: int = 5
    stride: int = 5
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.3
    feature_columns: list[str] | None = None
    seed: int | None = 42


class TrainPatchTSTResponse(BaseModel):
    rows_used: int
    symbols_used: int
    seq_len: int
    epochs_run: int
    final_train_loss: float
    final_train_accuracy: float
    best_val_loss: float | None = None
    stopped_early: bool = False
    out_of_sample_rows: int
    out_of_sample_accuracy: float
    out_of_sample_balanced_accuracy: float
    accepted: bool = True
    rejection_reason: str | None = None


class PredictPatchTSTResponse(BaseModel):
    symbol: str
    direction: str
    confidence: float


class TrainMetaResponse(BaseModel):
    rows_used: int
    symbols_used: int


class SweepLookbackRequest(BaseModel):
    symbols: list[str] | None = None  # None -> screener top long+short kullanılır
    lookback_values: list[int] = [2000, 4000, 6000, 8000, 10000]
    horizon: int = 5
    threshold_pct: float = 1.0
    labeling_method: LabelingMethod = "threshold"
    take_profit_pct: float = 2.0
    stop_loss_pct: float = 2.0
    algorithm: Algorithm = "xgboost"
    holdout_frac: float = 0.2
    walk_forward_splits: int = 5


class SweepLookbackPoint(BaseModel):
    lookback: int
    rows_used: int
    walk_forward_mean_accuracy: float
    walk_forward_mean_balanced_accuracy: float
    overfit_gap: float
    out_of_sample_rows: int
    out_of_sample_accuracy: float
    out_of_sample_balanced_accuracy: float
    error: str | None = None


class SweepLookbackResponse(BaseModel):
    symbols_used: int
    points: list[SweepLookbackPoint]


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
    # Eskiden burada screener'ın Top-N Long + Top-N Short çıktısı doğrudan
    # eğitim evreni olarak kullanılıyordu — bu liste yalnızca kısa vadeli
    # teknik skora göre seçiliyor, likidite/uyumluluk kontrolü yoktu ("zayıf
    # seçilmiş bir grup"). Şimdi bu ham liste yalnızca ADAY havuzu olarak
    # kullanılıyor; asıl seçim BTC-öncelikli + likidite/korelasyon
    # filtresinden geçen `select_training_symbols`'a devrediliyor.
    results = scan_market(exchange)
    picks = top_long(results, settings.screener_top_n) + top_short(results, settings.screener_top_n)
    candidates = [r.symbol for r in picks]
    return select_training_symbols(exchange, candidates)


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
        out_of_sample_true_class_counts=oos.true_class_counts,
        out_of_sample_predicted_class_counts=oos.predicted_class_counts,
        accepted=result.accepted,
        rejection_reason=result.rejection_reason,
    )


class TrainAllRequest(BaseModel):
    symbols: list[str] | None = None


class TrainAllStepModel(BaseModel):
    step: str  # "xgboost" | "meta_label" | "lstm" | "online" | "regime"
    ok: bool
    detail: str


class TrainAllResponse(BaseModel):
    symbols_used: int
    steps: list[TrainAllStepModel]


@router.post("/train-all", response_model=TrainAllResponse)
def train_all(payload: TrainAllRequest) -> TrainAllResponse:
    """Tüm modelleri (XGBoost -> meta-label -> LSTM -> online -> regime)
    TEK ÇAĞRIDA, deploy script'lerinde doğrulanmış AYNI parametrelerle
    sırayla eğitir (bkz. `app.ml.train.train_all_models` docstring'i).

    İlk kurulumda (henüz hiçbir model yokken, ör. yeni bir
    `fourkeys_ml_artifacts` volume'ünden sonra) veya toplu bir yeniden
    eğitim istendiğinde, 5 ayrı `curl`/deploy script'i yerine bunu
    kullanabilirsiniz. Bir adımın başarısız olması diğerlerini
    ENGELLEMEZ — her adımın sonucu `steps` listesinde ayrı raporlanır."""
    exchange = get_exchange(settings.exchange_id)
    symbols = _resolve_symbols(exchange, payload.symbols)
    if not symbols:
        raise HTTPException(status_code=400, detail="Eğitim için sembol bulunamadı.")

    results = train_all_models(exchange, symbols)
    return TrainAllResponse(
        symbols_used=len(symbols),
        steps=[TrainAllStepModel(step=r.step, ok=r.ok, detail=r.detail) for r in results],
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

    X, _y = build_training_dataset(exchange, symbols, settings.ml_train_timeframe, settings.ml_train_lookback)
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


@router.post("/train-lstm", response_model=TrainLSTMResponse)
def train_lstm(payload: TrainLSTMRequest) -> TrainLSTMResponse:
    """LSTM modelini eğitir (rehber Faz B). XGBoost'un (Faz A) tekil bar
    yerine, son `seq_len` barın özellik sekansını sırayla görür.

    Rehber "Faz A stabil/kârlı çalışmadan Faz B'ye geçilmez" der — bu
    endpoint bilinçli olarak bu kontrolü ZORUNLU KILMAZ (kullanıcı kararı),
    ama out-of-sample metrikleri (hiç görülmemiş son %20 veri üzerinde)
    modelin gerçekten mi öğrendiğini görünür kılar."""
    exchange = get_exchange(settings.exchange_id)
    symbols = _resolve_symbols(exchange, payload.symbols)
    if not symbols:
        raise HTTPException(status_code=400, detail="Eğitim için sembol bulunamadı.")

    try:
        result = train_lstm_signal_model(
            exchange,
            symbols,
            lookback=payload.lookback,
            seq_len=payload.seq_len,
            horizon=payload.horizon,
            threshold_pct=payload.threshold_pct,
            labeling_method=payload.labeling_method,
            take_profit_pct=payload.take_profit_pct,
            stop_loss_pct=payload.stop_loss_pct,
            holdout_frac=payload.holdout_frac,
            val_frac=payload.val_frac,
            epochs=payload.epochs,
            patience=payload.patience,
            hidden_size=payload.hidden_size,
            num_layers=payload.num_layers,
            dropout=payload.dropout,
            feature_columns=payload.feature_columns,
            seed=payload.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    oos = result.out_of_sample
    return TrainLSTMResponse(
        rows_used=result.rows_used,
        symbols_used=len(symbols),
        seq_len=payload.seq_len,
        epochs_run=result.training.epochs_run,
        final_train_loss=result.training.final_train_loss,
        final_train_accuracy=result.training.final_train_accuracy,
        best_val_loss=result.training.best_val_loss,
        stopped_early=result.training.stopped_early,
        out_of_sample_rows=oos.holdout_rows,
        out_of_sample_accuracy=oos.accuracy,
        out_of_sample_balanced_accuracy=oos.balanced_accuracy,
        accepted=result.accepted,
        rejection_reason=result.rejection_reason,
    )


@router.get("/predict-lstm", response_model=PredictLSTMResponse)
def predict_lstm(symbol: str = Query(..., description="Örn: BTC/USDT:USDT")) -> PredictLSTMResponse:
    if not Path(DEFAULT_LSTM_MODEL_PATH).exists():
        raise HTTPException(status_code=409, detail="LSTM modeli henüz eğitilmedi. Önce /ml/train-lstm çağırın.")

    exchange = get_exchange(settings.exchange_id)
    model = LSTMSignalModel.load_from()

    X, _y, _t = build_sequence_dataset(
        exchange,
        [symbol],
        settings.ml_train_timeframe,
        settings.ml_train_lookback,
        seq_len=model.seq_len,
        feature_columns=model.feature_columns,
    )
    if len(X) == 0:
        raise HTTPException(status_code=422, detail="Tahmin için yeterli veri yok (seq_len'e ulaşmıyor).")

    prediction = model.predict(X[-1])
    return PredictLSTMResponse(symbol=symbol, direction=prediction.direction, confidence=prediction.confidence)


@router.post("/train-patchtst", response_model=TrainPatchTSTResponse)
def train_patchtst(payload: TrainPatchTSTRequest) -> TrainPatchTSTResponse:
    """PatchTST'ten esinlenilmiş, patch-tabanlı Transformer sınıflandırıcıyı
    eğitir (bkz. `app.ml.patchtst_model`) — LSTM'e (Faz B) alternatif bir
    mimari denemesi. LSTM ile AYNI arayüzü/disiplini (holdout + erken
    durdurma + sınıf ağırlıklandırma) paylaşır, adil karşılaştırma için."""
    exchange = get_exchange(settings.exchange_id)
    symbols = _resolve_symbols(exchange, payload.symbols)
    if not symbols:
        raise HTTPException(status_code=400, detail="Eğitim için sembol bulunamadı.")

    try:
        result = train_patchtst_signal_model(
            exchange,
            symbols,
            lookback=payload.lookback,
            seq_len=payload.seq_len,
            horizon=payload.horizon,
            threshold_pct=payload.threshold_pct,
            labeling_method=payload.labeling_method,
            take_profit_pct=payload.take_profit_pct,
            stop_loss_pct=payload.stop_loss_pct,
            holdout_frac=payload.holdout_frac,
            val_frac=payload.val_frac,
            epochs=payload.epochs,
            patience=payload.patience,
            patch_len=payload.patch_len,
            stride=payload.stride,
            d_model=payload.d_model,
            nhead=payload.nhead,
            num_layers=payload.num_layers,
            dropout=payload.dropout,
            feature_columns=payload.feature_columns,
            seed=payload.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    oos = result.out_of_sample
    return TrainPatchTSTResponse(
        rows_used=result.rows_used,
        symbols_used=len(symbols),
        seq_len=payload.seq_len,
        epochs_run=result.training.epochs_run,
        final_train_loss=result.training.final_train_loss,
        final_train_accuracy=result.training.final_train_accuracy,
        best_val_loss=result.training.best_val_loss,
        stopped_early=result.training.stopped_early,
        out_of_sample_rows=oos.holdout_rows,
        out_of_sample_accuracy=oos.accuracy,
        out_of_sample_balanced_accuracy=oos.balanced_accuracy,
        accepted=result.accepted,
        rejection_reason=result.rejection_reason,
    )


@router.get("/predict-patchtst", response_model=PredictPatchTSTResponse)
def predict_patchtst(symbol: str = Query(..., description="Örn: BTC/USDT:USDT")) -> PredictPatchTSTResponse:
    if not Path(DEFAULT_PATCHTST_MODEL_PATH).exists():
        raise HTTPException(status_code=409, detail="PatchTST modeli henüz eğitilmedi. Önce /ml/train-patchtst çağırın.")

    exchange = get_exchange(settings.exchange_id)
    model = PatchTSTSignalModel.load_from()

    X, _y, _t = build_sequence_dataset(
        exchange,
        [symbol],
        settings.ml_train_timeframe,
        settings.ml_train_lookback,
        seq_len=model.seq_len,
        feature_columns=model.feature_columns,
    )
    if len(X) == 0:
        raise HTTPException(status_code=422, detail="Tahmin için yeterli veri yok (seq_len'e ulaşmıyor).")

    prediction = model.predict(X[-1])
    return PredictPatchTSTResponse(symbol=symbol, direction=prediction.direction, confidence=prediction.confidence)


@router.post("/train-meta", response_model=TrainMetaResponse)
def train_meta(payload: TrainMetaRequest) -> TrainMetaResponse:
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

    return TrainMetaResponse(rows_used=rows_used, symbols_used=len(symbols))


@router.post("/sweep-lookback", response_model=SweepLookbackResponse)
def sweep_lookback(payload: SweepLookbackRequest) -> SweepLookbackResponse:
    """Farklı `lookback` değerleriyle art arda eğitim yapıp her biri için
    walk-forward + out-of-sample metriklerini döner — "en küçük yeterli
    lookback nedir" sorusuna karar vermek için VERİ sağlar (otomatik "en
    iyi"yi seçmez, bkz. `app.ml.train.sweep_lookback_values` docstring'i).

    UYARI: her lookback değeri için sıfırdan bir eğitim (ve o değere göre
    yeniden Binance'ten veri çekimi) yapılır — birden çok değerle birden
    çok sembolde bu ÇOK uzun sürebilir. Production modelini DEĞİŞTİRMEZ
    (`persist=False`) — yalnızca karşılaştırma amaçlıdır, bittikten sonra
    ayrıca `/ml/train` çağırmak gerekir."""
    exchange = get_exchange(settings.exchange_id)
    symbols = _resolve_symbols(exchange, payload.symbols)
    if not symbols:
        raise HTTPException(status_code=400, detail="Eğitim için sembol bulunamadı.")

    points = sweep_lookback_values(
        exchange,
        symbols,
        payload.lookback_values,
        horizon=payload.horizon,
        threshold_pct=payload.threshold_pct,
        labeling_method=payload.labeling_method,
        take_profit_pct=payload.take_profit_pct,
        stop_loss_pct=payload.stop_loss_pct,
        algorithm=payload.algorithm,
        holdout_frac=payload.holdout_frac,
        walk_forward_splits=payload.walk_forward_splits,
    )
    return SweepLookbackResponse(symbols_used=len(symbols), points=[SweepLookbackPoint(**p.__dict__) for p in points])


class SweepLabelingRequest(BaseModel):
    symbols: list[str] | None = None
    lookback: int | None = None
    horizon_values: list[int] = [3, 5, 10, 15]
    threshold_pct_values: list[float] = [0.5, 1.0, 1.5, 2.0]
    seq_len: int = 20
    holdout_frac: float = 0.2
    val_frac: float = 0.15
    epochs: int = 30
    patience: int = 5


class SweepLabelingPoint(BaseModel):
    horizon: int
    threshold_pct: float
    rows_used: int
    final_train_accuracy: float
    out_of_sample_rows: int
    out_of_sample_accuracy: float
    out_of_sample_balanced_accuracy: float
    error: str | None = None


class SweepLabelingResponse(BaseModel):
    symbols_used: int
    points: list[SweepLabelingPoint]


@router.post("/sweep-labeling-lstm", response_model=SweepLabelingResponse)
def sweep_labeling_lstm_route(payload: SweepLabelingRequest) -> SweepLabelingResponse:
    """Farklı (`horizon`, `threshold_pct`) kombinasyonlarıyla LSTM'i art
    arda eğitip out-of-sample sonuçlarını döner — LSTM/PatchTST'in
    lookback/kapasite/mimari değişikliklerine rağmen aynı balanced_accuracy
    tavanına takılı kalması üzerine, etiketleme kalibrasyonunun sınırlayıcı
    faktör olup olmadığını test etmek için (bkz.
    `app.ml.train.sweep_labeling_lstm` docstring'i).

    UYARI: `len(horizon_values) * len(threshold_pct_values)` kadar sıfırdan
    eğitim yapılır — varsayılan ızgarada (4x4=16) birkaç dakika sürebilir.
    Production LSTM modelini DEĞİŞTİRMEZ (`persist=False`)."""
    exchange = get_exchange(settings.exchange_id)
    symbols = _resolve_symbols(exchange, payload.symbols)
    if not symbols:
        raise HTTPException(status_code=400, detail="Eğitim için sembol bulunamadı.")

    points = sweep_labeling_lstm(
        exchange,
        symbols,
        payload.horizon_values,
        payload.threshold_pct_values,
        lookback=payload.lookback,
        seq_len=payload.seq_len,
        holdout_frac=payload.holdout_frac,
        val_frac=payload.val_frac,
        epochs=payload.epochs,
        patience=payload.patience,
    )
    return SweepLabelingResponse(symbols_used=len(symbols), points=[SweepLabelingPoint(**p.__dict__) for p in points])


class TrainRegimeRequest(BaseModel):
    symbols: list[str] | None = None
    n_regimes: int = 3
    lookback: int | None = None
    horizon: int = 5
    threshold_pct: float = 1.0
    labeling_method: LabelingMethod = "threshold"
    take_profit_pct: float = 2.0
    stop_loss_pct: float = 2.0
    holdout_frac: float = 0.2
    walk_forward_splits: int = 5


class RegimeTrainingPoint(BaseModel):
    regime: int
    samples: int
    rows_used: int
    mean_volatility: float
    mean_trend: float
    walk_forward_mean_accuracy: float
    walk_forward_mean_balanced_accuracy: float
    overfit_gap: float
    out_of_sample_rows: int
    out_of_sample_accuracy: float
    out_of_sample_balanced_accuracy: float
    error: str | None = None


class TrainRegimeResponse(BaseModel):
    symbols_used: int
    n_regimes: int
    regimes: list[RegimeTrainingPoint]


@router.post("/train-regime", response_model=TrainRegimeResponse)
def train_regime(payload: TrainRegimeRequest) -> TrainRegimeResponse:
    """Hibrit rejim+ML (kullanıcı önerisi): piyasayı volatilite/trend
    uzayında GMM ile `n_regimes` rejime ayırır (bkz. `app.ml.regime` —
    tam Markov-Switching yerine, yeni bir bağımlılık gerektirmeyen bir
    yaklaşım) ve HER REJİM İÇİN AYRI bir XGBoost modeli eğitir.

    Amaç: tek bir global modelin (`/ml/train`) mi, yoksa rejime özel
    uzmanlaşmış modellerin mi daha iyi genellediğini KARŞILAŞTIRMAK için
    veri sağlamak — otomatik "en iyi"yi seçmez, canlı karar motoruna
    HENÜZ bağlanmadı (bkz. README roadmap)."""
    exchange = get_exchange(settings.exchange_id)
    symbols = _resolve_symbols(exchange, payload.symbols)
    if not symbols:
        raise HTTPException(status_code=400, detail="Eğitim için sembol bulunamadı.")

    try:
        _regime_model, results = train_signal_models_by_regime(
            exchange,
            symbols,
            n_regimes=payload.n_regimes,
            lookback=payload.lookback,
            horizon=payload.horizon,
            threshold_pct=payload.threshold_pct,
            labeling_method=payload.labeling_method,
            take_profit_pct=payload.take_profit_pct,
            stop_loss_pct=payload.stop_loss_pct,
            holdout_frac=payload.holdout_frac,
            walk_forward_splits=payload.walk_forward_splits,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return TrainRegimeResponse(
        symbols_used=len(symbols),
        n_regimes=payload.n_regimes,
        regimes=[RegimeTrainingPoint(**r.__dict__) for r in results],
    )


class TrainOnlineRequest(BaseModel):
    symbols: list[str] | None = None
    lookback: int | None = None
    horizon: int = 5
    threshold_pct: float = 1.0
    labeling_method: LabelingMethod = "threshold"
    take_profit_pct: float = 2.0
    stop_loss_pct: float = 2.0
    n_models: int = 10
    window_size: int = 500


class PrequentialWindowPointModel(BaseModel):
    window_index: int
    rows: int
    accuracy: float
    balanced_accuracy: float


class TrainOnlineResponse(BaseModel):
    symbols_used: int
    rows_used: int
    overall_accuracy: float
    overall_balanced_accuracy: float
    windows: list[PrequentialWindowPointModel]
    accepted: bool = True
    rejection_reason: str | None = None


@router.post("/train-online", response_model=TrainOnlineResponse)
def train_online(payload: TrainOnlineRequest) -> TrainOnlineResponse:
    """Kullanıcı önerisi: kavram kayması (concept drift) ile başa çıkmak
    için gerçek çevrimiçi (online) öğrenme — XGBoost/LSTM'in periyodik
    toptan (batch) yeniden eğitimi yerine, `river.forest.ARFClassifier`
    (Hoeffding ağaçlarından oluşan, kendi ADWIN kavram kayması tespitine
    sahip bir topluluk) verinin akışından bar-bar öğrenir.

    Değerlendirme "test-then-train" (prequential) protokolüyle yapılır —
    XGBoost'un statik holdout'undan FARKLI, ama online öğrenme
    literatüründe standart bir yöntem (bkz. `app.ml.online_model`
    docstring'i). `windows`, modelin ZAMAN İÇİNDE nasıl adapte olduğunu
    gösterir — erken pencerelerde düşük, sonraki pencerelerde yüksek
    accuracy, modelin öğrendiğinin (adaptasyonun) göstergesidir."""
    exchange = get_exchange(settings.exchange_id)
    symbols = _resolve_symbols(exchange, payload.symbols)
    if not symbols:
        raise HTTPException(status_code=400, detail="Eğitim için sembol bulunamadı.")

    try:
        _model, report = train_online_signal_model(
            exchange,
            symbols,
            lookback=payload.lookback,
            horizon=payload.horizon,
            threshold_pct=payload.threshold_pct,
            labeling_method=payload.labeling_method,
            take_profit_pct=payload.take_profit_pct,
            stop_loss_pct=payload.stop_loss_pct,
            n_models=payload.n_models,
            window_size=payload.window_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return TrainOnlineResponse(
        symbols_used=len(symbols),
        rows_used=report.rows_used,
        overall_accuracy=report.overall_accuracy,
        overall_balanced_accuracy=report.overall_balanced_accuracy,
        windows=[PrequentialWindowPointModel(**w.__dict__) for w in report.windows],
        accepted=report.accepted,
        rejection_reason=report.rejection_reason,
    )


@router.get("/predict", response_model=PredictResponse)
def predict(symbol: str = Query(..., description="Örn: BTC/USDT:USDT")) -> PredictResponse:
    if not _model_exists():
        raise HTTPException(status_code=409, detail="Model henüz eğitilmedi. Önce /ml/train çağırın.")

    exchange = get_exchange(settings.exchange_id)
    ohlcv = exchange.fetch_ohlcv(symbol, settings.ml_train_timeframe, settings.ml_train_lookback)
    feature_row = latest_feature_vector(ohlcv)
    if feature_row is None:
        raise HTTPException(status_code=422, detail="Yeterli veri yok.")
    for col, value in latest_macro_feature_row().items():
        feature_row[col] = value
    for col, value in latest_orderbook_feature_row(symbol).items():
        feature_row[col] = value
    feature_row["taker_buy_ratio_norm"] = latest_taker_buy_ratio_norm(exchange, symbol, settings.ml_train_timeframe)

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
